# ruff: noqa: E501
#!/usr/bin/env python3
"""Generate calibration examples by running tasks through a target model.

Produces a JSONL file with (task, response) pairs ready for human scoring.

Usage:
    uv run python scripts/generate_calibration_set.py --model qwen3.5-9b
"""

import argparse
import json
import sys
import time
from pathlib import Path

from nite_eval.conversation_runner import run_conversation
from nite_eval.mock_tools import MockToolEnv
from nite_eval.model_manager import check_health, warm_up_model

TARGET_URL = "http://127.0.0.1:8080"
OUTPUT_DIR = Path("judges/calibration")

# Calibration tasks — inline definitions covering all judge-scored dimensions.
# These are simpler/shorter than the full eval tasks to keep hand-scoring manageable.
CALIBRATION_TASKS = [
    # --- Research quality (judge dimensions: coverage, accuracy, synthesis) ---
    {
        "id": "cal_research_01",
        "dimension": "research",
        "judge_dimensions": ["coverage", "synthesis"],
        "system_prompt": "You are a technical research assistant.",
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "web_search",
                    "description": "Search the web",
                    "parameters": {
                        "type": "object",
                        "properties": {"query": {"type": "string"}},
                        "required": ["query"],
                    },
                },
            },
        ],
        "user_message": "Compare REST and GraphQL APIs for a mobile app backend. Cover performance, flexibility, and caching.",
        "mock_responses": {
            "web_search": [
                {
                    "match": {"query_contains": "REST"},
                    "response": {
                        "content": {
                            "results": [
                                {
                                    "title": "REST vs GraphQL",
                                    "snippet": "REST uses fixed endpoints, cacheable via HTTP. GraphQL uses single endpoint, flexible queries, reduces over-fetching. REST better for simple CRUD, GraphQL for complex nested data.",
                                }
                            ]
                        }
                    },
                },
                {
                    "match": {"query_contains": "GraphQL"},
                    "response": {
                        "content": {
                            "results": [
                                {
                                    "title": "GraphQL Performance",
                                    "snippet": "GraphQL can cause N+1 queries without DataLoader. No HTTP caching by default. Persisted queries help. Better for mobile: reduces bandwidth via precise field selection.",
                                }
                            ]
                        }
                    },
                },
                {"match": {"query_contains": "any"}, "response": {"content": {"results": []}}},
            ],
        },
        "max_turns": 6,
    },
    {
        "id": "cal_research_02",
        "dimension": "research",
        "judge_dimensions": ["accuracy", "recommendation_quality"],
        "system_prompt": "You are a database technology advisor.",
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "web_search",
                    "description": "Search the web",
                    "parameters": {
                        "type": "object",
                        "properties": {"query": {"type": "string"}},
                        "required": ["query"],
                    },
                },
            },
        ],
        "user_message": "I need a vector database for 500K embeddings. Compare pgvector, Qdrant, and Pinecone for a solo developer on a budget.",
        "mock_responses": {
            "web_search": [
                {
                    "match": {"query_contains": "pgvector"},
                    "response": {
                        "content": {
                            "results": [
                                {
                                    "title": "pgvector",
                                    "snippet": "PostgreSQL extension. HNSW + IVFFlat indexes. Good to ~1M vectors. Free, runs in existing Postgres. Slower than dedicated vector DBs at scale.",
                                }
                            ]
                        }
                    },
                },
                {
                    "match": {"query_contains": "qdrant"},
                    "response": {
                        "content": {
                            "results": [
                                {
                                    "title": "Qdrant",
                                    "snippet": "Rust-based, high performance. Self-hosted free, cloud from $25/mo. HNSW with quantization. Filtering + payload storage. Good API.",
                                }
                            ]
                        }
                    },
                },
                {
                    "match": {"query_contains": "pinecone"},
                    "response": {
                        "content": {
                            "results": [
                                {
                                    "title": "Pinecone",
                                    "snippet": "Fully managed. Serverless from $0 (free tier 100K vectors). Low ops burden. Proprietary, vendor lock-in. Metadata filtering.",
                                }
                            ]
                        }
                    },
                },
                {"match": {"query_contains": "any"}, "response": {"content": {"results": []}}},
            ],
        },
        "max_turns": 8,
    },
    {
        "id": "cal_research_03",
        "dimension": "research",
        "judge_dimensions": ["coverage", "synthesis", "accuracy"],
        "system_prompt": "You are a technical research assistant helping evaluate deployment options.",
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "web_search",
                    "description": "Search the web",
                    "parameters": {
                        "type": "object",
                        "properties": {"query": {"type": "string"}},
                        "required": ["query"],
                    },
                },
            },
        ],
        "user_message": "Compare Fly.io, Railway, and Render for deploying a Python FastAPI backend with a Postgres database. I'm a solo developer, cost matters.",
        "mock_responses": {
            "web_search": [
                {
                    "match": {"query_contains": "fly.io"},
                    "response": {
                        "content": {
                            "results": [
                                {
                                    "title": "Fly.io pricing",
                                    "snippet": "Free tier: 3 shared VMs, 3GB storage. Postgres via fly postgres (not managed). Edge deployment globally. CLI-driven. $1.94/mo per extra VM.",
                                }
                            ]
                        }
                    },
                },
                {
                    "match": {"query_contains": "railway"},
                    "response": {
                        "content": {
                            "results": [
                                {
                                    "title": "Railway review",
                                    "snippet": "Usage-based: $5 credit free/mo. Managed Postgres included. GitHub deploy. Simple but costs scale fast. Good DX, limited config.",
                                }
                            ]
                        }
                    },
                },
                {
                    "match": {"query_contains": "render"},
                    "response": {
                        "content": {
                            "results": [
                                {
                                    "title": "Render pricing",
                                    "snippet": "Free tier: web services sleep after 15min. Managed Postgres from $7/mo. Auto-deploy from GitHub. Good for side projects. US/EU regions only.",
                                }
                            ]
                        }
                    },
                },
                {"match": {"query_contains": "any"}, "response": {"content": {"results": []}}},
            ],
        },
        "max_turns": 8,
    },
    # --- Planning quality (judge dimensions: specificity, risk_awareness, feasibility) ---
    {
        "id": "cal_planning_01",
        "dimension": "planning",
        "judge_dimensions": ["specificity", "risk_awareness"],
        "system_prompt": "You are a technical project planner.",
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "estimate_effort",
                    "description": "Estimate dev effort for a task",
                    "parameters": {
                        "type": "object",
                        "properties": {"task": {"type": "string"}, "stack": {"type": "string"}},
                        "required": ["task"],
                    },
                },
            },
        ],
        "user_message": "Plan a weekend project: build a CLI tool in Python that monitors a directory for new files and uploads them to S3. What should I build first vs second?",
        "mock_responses": {
            "estimate_effort": [
                {
                    "match": {"task_contains": "watch"},
                    "response": {
                        "content": {
                            "estimate_hours": 2,
                            "notes": "Use watchdog library. Event-based, handles OS differences.",
                        }
                    },
                },
                {
                    "match": {"task_contains": "upload"},
                    "response": {
                        "content": {
                            "estimate_hours": 3,
                            "notes": "boto3 with multipart upload. Add retry logic. Handle large files.",
                        }
                    },
                },
                {
                    "match": {"task_contains": "CLI"},
                    "response": {
                        "content": {"estimate_hours": 1, "notes": "Click or Typer. Config via YAML or env vars."}
                    },
                },
                {
                    "match": {"task_contains": "any"},
                    "response": {"content": {"estimate_hours": 2, "notes": "Moderate complexity."}},
                },
            ],
        },
        "max_turns": 6,
    },
    {
        "id": "cal_planning_02",
        "dimension": "planning",
        "judge_dimensions": ["completeness", "dependency_correctness", "risk_awareness"],
        "system_prompt": "You are a systems architect helping plan a migration.",
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "get_current_infrastructure",
                    "description": "Get current system details",
                    "parameters": {
                        "type": "object",
                        "properties": {"component": {"type": "string"}},
                        "required": ["component"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "estimate_effort",
                    "description": "Estimate migration effort",
                    "parameters": {"type": "object", "properties": {"task": {"type": "string"}}, "required": ["task"]},
                },
            },
        ],
        "user_message": "I'm moving a Django app from SQLite to PostgreSQL on Supabase. It's in production with ~500 users. Plan the migration so I don't lose data or have extended downtime.",
        "mock_responses": {
            "get_current_infrastructure": [
                {
                    "match": {"component_contains": "database"},
                    "response": {"content": {"db": "SQLite", "size_mb": 45, "tables": 12, "has_migrations": True}},
                },
                {
                    "match": {"component_contains": "any"},
                    "response": {"content": {"stack": "Django 4.2, SQLite, deployed on Railway"}},
                },
            ],
            "estimate_effort": [
                {
                    "match": {"task_contains": "schema"},
                    "response": {
                        "content": {
                            "estimate_hours": 2,
                            "notes": "Django migrations handle most of it. Watch for SQLite-specific field types.",
                        }
                    },
                },
                {
                    "match": {"task_contains": "data"},
                    "response": {
                        "content": {
                            "estimate_hours": 3,
                            "notes": "Use django-dbbackup or pg_dump/restore. Test with full dataset first.",
                        }
                    },
                },
                {
                    "match": {"task_contains": "any"},
                    "response": {"content": {"estimate_hours": 2, "notes": "Moderate complexity."}},
                },
            ],
        },
        "max_turns": 8,
    },
    {
        "id": "cal_planning_03",
        "dimension": "planning",
        "judge_dimensions": ["specificity", "feasibility"],
        "system_prompt": "You are a technical project planner for API development.",
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "estimate_effort",
                    "description": "Estimate dev effort",
                    "parameters": {"type": "object", "properties": {"task": {"type": "string"}}, "required": ["task"]},
                },
            },
        ],
        "user_message": "Plan adding rate limiting to an existing FastAPI app. It serves 3 endpoints, ~1000 req/min peak. I want per-user limits with Redis as the backend. Break it into tasks.",
        "mock_responses": {
            "estimate_effort": [
                {
                    "match": {"task_contains": "redis"},
                    "response": {
                        "content": {"estimate_hours": 1, "notes": "redis-py or aioredis. Connection pooling important."}
                    },
                },
                {
                    "match": {"task_contains": "middleware"},
                    "response": {
                        "content": {
                            "estimate_hours": 3,
                            "notes": "Sliding window counter in Redis. FastAPI middleware or dependency injection.",
                        }
                    },
                },
                {
                    "match": {"task_contains": "test"},
                    "response": {
                        "content": {
                            "estimate_hours": 2,
                            "notes": "Mock Redis for unit tests. Load test with locust for integration.",
                        }
                    },
                },
                {
                    "match": {"task_contains": "any"},
                    "response": {"content": {"estimate_hours": 2, "notes": "Standard complexity."}},
                },
            ],
        },
        "max_turns": 6,
    },
    # --- Code quality (judge dimensions: idiomatic, error_handling, architecture) ---
    {
        "id": "cal_coding_01",
        "dimension": "coding",
        "judge_dimensions": ["code_quality", "error_handling"],
        "system_prompt": "You are a Python developer.",
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "write_file",
                    "description": "Write content to a file",
                    "parameters": {
                        "type": "object",
                        "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
                        "required": ["path", "content"],
                    },
                },
            },
        ],
        "user_message": "Write a Python function that retries an HTTP request up to 3 times with exponential backoff. Use httpx. Handle timeouts and connection errors separately.",
        "mock_responses": {
            "write_file": [
                {"match": {"path_contains": "any"}, "response": {"content": {"status": "written"}}},
            ],
        },
        "max_turns": 4,
    },
    {
        "id": "cal_coding_02",
        "dimension": "coding",
        "judge_dimensions": ["code_quality", "architecture"],
        "system_prompt": "You are a Python developer building data processing tools.",
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "write_file",
                    "description": "Write content to a file",
                    "parameters": {
                        "type": "object",
                        "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
                        "required": ["path", "content"],
                    },
                },
            },
        ],
        "user_message": "Write a Python class that validates CSV data against a JSON schema. The schema defines column names, types (str, int, float, date), and required/optional. Return a list of validation errors with row number and column.",
        "mock_responses": {
            "write_file": [
                {"match": {"path_contains": "any"}, "response": {"content": {"status": "written"}}},
            ],
        },
        "max_turns": 4,
    },
    {
        "id": "cal_coding_03",
        "dimension": "coding",
        "judge_dimensions": ["code_quality", "error_handling", "architecture"],
        "system_prompt": "You are a Python developer building CLI tools.",
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "write_file",
                    "description": "Write content to a file",
                    "parameters": {
                        "type": "object",
                        "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
                        "required": ["path", "content"],
                    },
                },
            },
        ],
        "user_message": "Write a Python module that parses structured log lines (JSON format) from stdin, extracts error-level entries, groups them by 'service' field, and prints a summary table showing service name, error count, and most recent error message. Handle malformed lines gracefully.",
        "mock_responses": {
            "write_file": [
                {"match": {"path_contains": "any"}, "response": {"content": {"status": "written"}}},
            ],
        },
        "max_turns": 4,
    },
    # --- Agentic synthesis (judge dimensions: reasoning, practical_output) ---
    {
        "id": "cal_agentic_01",
        "dimension": "agentic",
        "judge_dimensions": ["reasoning_quality", "practical_output"],
        "system_prompt": "You are a helpful assistant with access to weather and calendar tools.",
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get weather forecast",
                    "parameters": {
                        "type": "object",
                        "properties": {"location": {"type": "string"}, "days": {"type": "integer"}},
                        "required": ["location"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "get_calendar",
                    "description": "Get calendar events",
                    "parameters": {"type": "object", "properties": {"date": {"type": "string"}}, "required": ["date"]},
                },
            },
        ],
        "user_message": "I'm planning an outdoor team lunch tomorrow in San Francisco. Should I go ahead with it?",
        "mock_responses": {
            "get_weather": [
                {
                    "match": {"location_contains": "san francisco"},
                    "response": {"content": {"forecast": "Partly cloudy, 62°F, 20% chance of rain, wind 12mph"}},
                },
                {"match": {"location_contains": "any"}, "response": {"content": {"forecast": "Clear, 70°F"}}},
            ],
            "get_calendar": [
                {
                    "match": {"date_contains": "any"},
                    "response": {
                        "content": {
                            "events": [
                                {"title": "Team standup", "time": "10:00", "duration": "30min"},
                                {"title": "Design review", "time": "14:00", "duration": "1hr"},
                            ]
                        }
                    },
                },
            ],
        },
        "max_turns": 6,
    },
    {
        "id": "cal_agentic_02",
        "dimension": "agentic",
        "judge_dimensions": ["reasoning_quality", "practical_output"],
        "system_prompt": "You are a DevOps assistant with access to monitoring tools.",
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "get_metrics",
                    "description": "Get service metrics",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "service": {"type": "string"},
                            "metric": {"type": "string"},
                            "period": {"type": "string"},
                        },
                        "required": ["service", "metric"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "get_logs",
                    "description": "Get recent log entries",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "service": {"type": "string"},
                            "level": {"type": "string"},
                            "limit": {"type": "integer"},
                        },
                        "required": ["service"],
                    },
                },
            },
        ],
        "user_message": "The API service feels slow today. Can you check what's going on?",
        "mock_responses": {
            "get_metrics": [
                {
                    "match": {"metric_contains": "latency"},
                    "response": {"content": {"p50_ms": 120, "p95_ms": 890, "p99_ms": 2300, "normal_p95_ms": 200}},
                },
                {
                    "match": {"metric_contains": "error"},
                    "response": {"content": {"error_rate": 0.08, "normal_error_rate": 0.01}},
                },
                {
                    "match": {"metric_contains": "cpu"},
                    "response": {"content": {"cpu_percent": 78, "normal_cpu_percent": 35}},
                },
                {"match": {"metric_contains": "any"}, "response": {"content": {"value": 42}}},
            ],
            "get_logs": [
                {
                    "match": {"service_contains": "api"},
                    "response": {
                        "content": {
                            "entries": [
                                {
                                    "level": "ERROR",
                                    "message": "Connection pool exhausted for db-primary",
                                    "count": 45,
                                    "first_seen": "08:15",
                                },
                                {
                                    "level": "WARN",
                                    "message": "Slow query: SELECT * FROM events WHERE... (3.2s)",
                                    "count": 120,
                                    "first_seen": "08:00",
                                },
                            ]
                        }
                    },
                },
            ],
        },
        "max_turns": 8,
    },
    {
        "id": "cal_agentic_03",
        "dimension": "agentic",
        "judge_dimensions": ["reasoning_quality", "practical_output"],
        "system_prompt": "You are a personal finance assistant with access to account and market data.",
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "get_account_balance",
                    "description": "Get account balances",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "account_type": {"type": "string", "enum": ["checking", "savings", "investment"]}
                        },
                        "required": ["account_type"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "get_recurring_expenses",
                    "description": "Get monthly recurring expenses",
                    "parameters": {"type": "object", "properties": {}},
                },
            },
        ],
        "user_message": "I want to know if I can afford to increase my monthly investment contribution by $500. Check my financial situation.",
        "mock_responses": {
            "get_account_balance": [
                {
                    "match": {"account_type": "checking"},
                    "response": {"content": {"balance": 4200, "avg_monthly_inflow": 6500}},
                },
                {"match": {"account_type": "savings"}, "response": {"content": {"balance": 15000, "target": 18000}}},
                {
                    "match": {"account_type": "investment"},
                    "response": {"content": {"balance": 32000, "monthly_contribution": 1000}},
                },
            ],
            "get_recurring_expenses": [
                {
                    "match": {},
                    "response": {
                        "content": {
                            "total_monthly": 4800,
                            "categories": {
                                "rent": 2000,
                                "utilities": 200,
                                "subscriptions": 150,
                                "groceries": 600,
                                "insurance": 350,
                                "car": 400,
                                "investment": 1000,
                                "misc": 100,
                            },
                        }
                    },
                },
            ],
        },
        "max_turns": 8,
    },
]


def main():
    parser = argparse.ArgumentParser(description="Generate calibration examples")
    parser.add_argument("--model", default="qwen3.5-9b", help="Target model name")
    parser.add_argument("--output", default=str(OUTPUT_DIR / "calibration_set.jsonl"))
    parser.add_argument("--count", type=int, default=0, help="Limit number of tasks (0=all)")
    args = parser.parse_args()

    if not check_health("http://127.0.0.1:8080"):
        print("Target server not running on :8080. Start llama-swap first.")
        sys.exit(1)

    print(f"Warming up model: {args.model}")
    if not warm_up_model(TARGET_URL, args.model, timeout=120.0):
        print("Model warm-up failed")
        sys.exit(1)

    tasks = CALIBRATION_TASKS
    if args.count > 0:
        tasks = tasks[: args.count]

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    results = []
    for i, task in enumerate(tasks):
        print(f"\n[{i + 1}/{len(tasks)}] Running: {task['id']} ({task['dimension']})")
        mock_env = MockToolEnv.from_task_yaml(task["mock_responses"])

        start = time.monotonic()
        conv_result = run_conversation(
            base_url=TARGET_URL,
            model_name=args.model,
            system_prompt=task["system_prompt"],
            tools=task["tools"],
            user_message=task["user_message"],
            mock_env=mock_env,
            max_turns=task["max_turns"],
        )
        elapsed = time.monotonic() - start

        print(f"  Turns: {conv_result.total_tool_calls} tool calls, {elapsed:.1f}s")

        entry = {
            "id": task["id"],
            "dimension": task["dimension"],
            "judge_dimensions": task["judge_dimensions"],
            "user_message": task["user_message"],
            "model": args.model,
            "response": conv_result.final_response,
            "tool_calls_made": conv_result.total_tool_calls,
            "turns_used": len(conv_result.turns),
            "human_scores": {},  # To be filled by human scorer
        }
        results.append(entry)

    with open(output_path, "w") as f:
        for entry in results:
            f.write(json.dumps(entry) + "\n")

    print(f"\nWrote {len(results)} examples to {output_path}")
    print("Next: uv run python scripts/score_calibration.py")


if __name__ == "__main__":
    main()
