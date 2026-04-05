"""SQLite storage for evaluation results, checkpointing, and resume.

Schema:
  eval_runs      — top-level run metadata (one per overnight run)
  task_results   — per-model per-task results (the checkpoint unit)
  score_details  — per-dimension scores with judge reasoning
  tool_calls     — recorded tool call sequences for provenance
"""

from __future__ import annotations

import json
import logging
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS eval_runs (
    run_id         TEXT PRIMARY KEY,
    started_at     REAL NOT NULL,
    finished_at    REAL,
    status         TEXT NOT NULL DEFAULT 'running',  -- running, completed, failed
    models         TEXT NOT NULL,  -- JSON array of model names
    config_snapshot TEXT  -- JSON dump of eval config for reproducibility
);

CREATE TABLE IF NOT EXISTS task_results (
    run_id          TEXT NOT NULL,
    model_name      TEXT NOT NULL,
    task_id         TEXT NOT NULL,
    dimension       TEXT NOT NULL,
    difficulty      TEXT NOT NULL,
    status          TEXT NOT NULL DEFAULT 'pending',  -- pending, running, completed, failed, skipped
    started_at      REAL,
    finished_at     REAL,
    final_response  TEXT,
    total_turns     INTEGER,
    total_tool_calls INTEGER,
    total_latency_ms REAL,
    reached_max_turns INTEGER,  -- 0/1 boolean
    weighted_score  REAL,       -- final weighted composite for this task
    error           TEXT,
    PRIMARY KEY (run_id, model_name, task_id),
    FOREIGN KEY (run_id) REFERENCES eval_runs(run_id)
);

CREATE TABLE IF NOT EXISTS score_details (
    run_id        TEXT NOT NULL,
    model_name    TEXT NOT NULL,
    task_id       TEXT NOT NULL,
    dimension     TEXT NOT NULL,  -- scoring dimension name
    method        TEXT NOT NULL,  -- judge_rubric, sequence_match, checklist, etc.
    raw_score     REAL,
    normalized    REAL,           -- 0.0-1.0
    weight        REAL,
    judge_model   TEXT,           -- which judge scored this (flow-judge, reward-anything, null for deterministic)
    reasoning     TEXT,           -- judge reasoning text
    confidence    REAL,           -- judge confidence (from averaging)
    details_json  TEXT,           -- full details as JSON
    PRIMARY KEY (run_id, model_name, task_id, dimension),
    FOREIGN KEY (run_id, model_name, task_id)
        REFERENCES task_results(run_id, model_name, task_id)
);

CREATE TABLE IF NOT EXISTS tool_calls (
    run_id      TEXT NOT NULL,
    model_name  TEXT NOT NULL,
    task_id     TEXT NOT NULL,
    turn        INTEGER NOT NULL,
    call_index  INTEGER NOT NULL,  -- position within the turn
    tool_name   TEXT NOT NULL,
    arguments   TEXT,              -- JSON
    result      TEXT,              -- JSON mock response
    PRIMARY KEY (run_id, model_name, task_id, turn, call_index),
    FOREIGN KEY (run_id, model_name, task_id)
        REFERENCES task_results(run_id, model_name, task_id)
);

CREATE TABLE IF NOT EXISTS schema_version (
    version INTEGER NOT NULL
);

-- Indexes for common queries
CREATE INDEX IF NOT EXISTS idx_task_results_status
    ON task_results(run_id, status);
CREATE INDEX IF NOT EXISTS idx_task_results_model
    ON task_results(run_id, model_name);
CREATE INDEX IF NOT EXISTS idx_task_results_dimension
    ON task_results(run_id, dimension);
"""


@dataclass
class PendingTask:
    """A task that still needs to be run for a model."""

    model_name: str
    task_id: str
    dimension: str
    difficulty: str


class ResultsDB:
    """SQLite database for evaluation results and checkpointing."""

    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path))
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._init_schema()

    def _init_schema(self) -> None:
        cursor = self._conn.cursor()
        cursor.executescript(SCHEMA_SQL)

        # Check/set schema version
        cursor.execute("SELECT COUNT(*) FROM schema_version")
        if cursor.fetchone()[0] == 0:
            cursor.execute("INSERT INTO schema_version VALUES (?)", (SCHEMA_VERSION,))
        self._conn.commit()

    # --- Run management ---

    def create_run(self, run_id: str, models: list[str], config: dict | None = None) -> None:
        """Create a new evaluation run."""
        self._conn.execute(
            "INSERT INTO eval_runs (run_id, started_at, status, models, config_snapshot) VALUES (?, ?, ?, ?, ?)",
            (run_id, time.time(), "running", json.dumps(models), json.dumps(config) if config else None),
        )
        self._conn.commit()

    def finish_run(self, run_id: str, status: str = "completed") -> None:
        """Mark a run as finished."""
        self._conn.execute(
            "UPDATE eval_runs SET finished_at = ?, status = ? WHERE run_id = ?",
            (time.time(), status, run_id),
        )
        self._conn.commit()

    def get_run_status(self, run_id: str) -> str | None:
        """Get the status of a run, or None if it doesn't exist."""
        cursor = self._conn.execute("SELECT status FROM eval_runs WHERE run_id = ?", (run_id,))
        row = cursor.fetchone()
        return row[0] if row else None

    # --- Task checkpointing ---

    def register_tasks(
        self,
        run_id: str,
        models: list[str],
        tasks: list[tuple[str, str, str]],
    ) -> None:
        """Register all (model, task) pairs for a run as pending.

        Args:
            tasks: list of (task_id, dimension, difficulty) tuples
        """
        rows = [
            (run_id, model, task_id, dimension, difficulty)
            for model in models
            for task_id, dimension, difficulty in tasks
        ]
        self._conn.executemany(
            "INSERT OR IGNORE INTO task_results (run_id, model_name, task_id, dimension, difficulty) VALUES (?, ?, ?, ?, ?)",
            rows,
        )
        self._conn.commit()

    def get_pending_tasks(self, run_id: str, model_name: str) -> list[PendingTask]:
        """Get tasks that haven't completed for a model in this run."""
        cursor = self._conn.execute(
            "SELECT model_name, task_id, dimension, difficulty FROM task_results "
            "WHERE run_id = ? AND model_name = ? AND status IN ('pending', 'running')",
            (run_id, model_name),
        )
        return [PendingTask(*row) for row in cursor.fetchall()]

    def mark_task_running(self, run_id: str, model_name: str, task_id: str) -> None:
        self._conn.execute(
            "UPDATE task_results SET status = 'running', started_at = ? "
            "WHERE run_id = ? AND model_name = ? AND task_id = ?",
            (time.time(), run_id, model_name, task_id),
        )
        self._conn.commit()

    def save_task_result(
        self,
        run_id: str,
        model_name: str,
        task_id: str,
        final_response: str,
        total_turns: int,
        total_tool_calls: int,
        total_latency_ms: float,
        reached_max_turns: bool,
        weighted_score: float,
        error: str | None = None,
    ) -> None:
        """Save a completed task result (the checkpoint)."""
        status = "failed" if error else "completed"
        self._conn.execute(
            "UPDATE task_results SET "
            "status = ?, finished_at = ?, final_response = ?, "
            "total_turns = ?, total_tool_calls = ?, total_latency_ms = ?, "
            "reached_max_turns = ?, weighted_score = ?, error = ? "
            "WHERE run_id = ? AND model_name = ? AND task_id = ?",
            (
                status,
                time.time(),
                final_response,
                total_turns,
                total_tool_calls,
                total_latency_ms,
                int(reached_max_turns),
                weighted_score,
                error,
                run_id,
                model_name,
                task_id,
            ),
        )
        self._conn.commit()

    # --- Score details ---

    def save_score(
        self,
        run_id: str,
        model_name: str,
        task_id: str,
        dimension: str,
        method: str,
        raw_score: float | None,
        normalized: float | None,
        weight: float,
        judge_model: str | None = None,
        reasoning: str | None = None,
        confidence: float | None = None,
        details: dict | None = None,
    ) -> None:
        """Save a per-dimension score."""
        self._conn.execute(
            "INSERT OR REPLACE INTO score_details "
            "(run_id, model_name, task_id, dimension, method, raw_score, normalized, "
            "weight, judge_model, reasoning, confidence, details_json) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                run_id,
                model_name,
                task_id,
                dimension,
                method,
                raw_score,
                normalized,
                weight,
                judge_model,
                reasoning,
                confidence,
                json.dumps(details) if details else None,
            ),
        )
        self._conn.commit()

    # --- Tool call recording ---

    def save_tool_calls(
        self,
        run_id: str,
        model_name: str,
        task_id: str,
        calls: list[dict],
    ) -> None:
        """Save tool call sequence from a conversation.

        Args:
            calls: list of {turn, call_index, tool_name, arguments, result}
        """
        rows = [
            (
                run_id,
                model_name,
                task_id,
                c["turn"],
                c["call_index"],
                c["tool_name"],
                json.dumps(c.get("arguments")),
                json.dumps(c.get("result")),
            )
            for c in calls
        ]
        self._conn.executemany(
            "INSERT OR REPLACE INTO tool_calls "
            "(run_id, model_name, task_id, turn, call_index, tool_name, arguments, result) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            rows,
        )
        self._conn.commit()

    def get_tool_calls(self, run_id: str, model_name: str, task_id: str) -> list[dict]:
        """Retrieve tool calls for scoring."""
        cursor = self._conn.execute(
            "SELECT turn, call_index, tool_name, arguments, result FROM tool_calls "
            "WHERE run_id = ? AND model_name = ? AND task_id = ? "
            "ORDER BY turn, call_index",
            (run_id, model_name, task_id),
        )
        return [
            {
                "turn": row[0],
                "call_index": row[1],
                "name": row[2],
                "arguments": json.loads(row[3]) if row[3] else {},
                "result": json.loads(row[4]) if row[4] else {},
            }
            for row in cursor.fetchall()
        ]

    # --- Query helpers ---

    def get_model_scores(self, run_id: str, model_name: str) -> list[dict]:
        """Get all task scores for a model in a run."""
        cursor = self._conn.execute(
            "SELECT task_id, dimension, difficulty, weighted_score, status, total_latency_ms "
            "FROM task_results WHERE run_id = ? AND model_name = ? ORDER BY task_id",
            (run_id, model_name),
        )
        return [
            {
                "task_id": row[0],
                "dimension": row[1],
                "difficulty": row[2],
                "weighted_score": row[3],
                "status": row[4],
                "latency_ms": row[5],
            }
            for row in cursor.fetchall()
        ]

    def get_dimension_averages(self, run_id: str, model_name: str) -> dict[str, float]:
        """Get average weighted score per dimension for a model."""
        cursor = self._conn.execute(
            "SELECT dimension, AVG(weighted_score) FROM task_results "
            "WHERE run_id = ? AND model_name = ? AND status = 'completed' "
            "GROUP BY dimension",
            (run_id, model_name),
        )
        return {row[0]: row[1] for row in cursor.fetchall()}

    def get_run_summary(self, run_id: str) -> dict:
        """Get a summary of a run across all models."""
        cursor = self._conn.execute(
            "SELECT model_name, COUNT(*) as total, "
            "SUM(CASE WHEN status = 'completed' THEN 1 ELSE 0 END) as completed, "
            "SUM(CASE WHEN status = 'failed' THEN 1 ELSE 0 END) as failed, "
            "AVG(CASE WHEN status = 'completed' THEN weighted_score END) as avg_score "
            "FROM task_results WHERE run_id = ? GROUP BY model_name",
            (run_id,),
        )
        return {
            row[0]: {
                "total": row[1],
                "completed": row[2],
                "failed": row[3],
                "avg_score": row[4],
            }
            for row in cursor.fetchall()
        }

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> ResultsDB:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()
