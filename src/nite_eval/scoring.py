"""Scoring methods for evaluation tasks.

Implements deterministic scoring (sequence match, checklist, exact match, contains)
and judge-based scoring orchestration. Aggregates per-task scores into per-dimension
and composite scores.
"""

import logging
from dataclasses import dataclass

from nite_eval.judge import JudgeClient, JudgeError, JudgeResult

logger = logging.getLogger(__name__)


@dataclass
class ScoreResult:
    dimension: str
    method: str
    score: float  # Normalized 0.0-1.0
    weight: float
    details: dict


@dataclass
class TaskScore:
    task_id: str
    dimension: str
    scores: list[ScoreResult]
    weighted_total: float  # Weighted sum of normalized scores


# --- Deterministic scoring methods ---


def score_sequence_match(
    actual_calls: list[dict],
    expected_sequence: list[dict],
) -> float:
    """Score tool call sequence against expected sequence.

    Returns fraction of expected calls that match in order.
    """
    if not expected_sequence:
        return 1.0 if not actual_calls else 0.0

    matched = 0
    actual_idx = 0

    for expected in expected_sequence:
        while actual_idx < len(actual_calls):
            actual = actual_calls[actual_idx]
            actual_idx += 1
            if _call_matches(actual, expected):
                matched += 1
                break

    return matched / len(expected_sequence)


def score_subset_match(
    actual_calls: list[dict],
    expected_tools: list[str],
) -> float:
    """Score whether all expected tools were called (order-independent).

    Returns fraction of expected tools that appear in actual calls.
    """
    if not expected_tools:
        return 1.0

    called_tools = {c["name"] for c in actual_calls}
    matched = sum(1 for t in expected_tools if t in called_tools)
    return matched / len(expected_tools)


def score_checklist(
    response_text: str,
    criteria: list[str],
) -> float:
    """Score whether response text addresses checklist items.

    Simple contains-based check. Returns fraction of criteria found.
    """
    if not criteria:
        return 1.0

    matched = 0
    lower_response = response_text.lower()
    for criterion in criteria:
        # Extract key phrases from the criterion for matching
        key_words = [w.strip().lower() for w in criterion.split() if len(w.strip()) > 3]
        if any(kw in lower_response for kw in key_words):
            matched += 1

    return matched / len(criteria)


def score_contains_check(
    response_text: str,
    required_strings: list[str],
) -> float:
    """Score whether response contains all required strings."""
    if not required_strings:
        return 1.0

    lower_response = response_text.lower()
    matched = sum(1 for s in required_strings if s.lower() in lower_response)
    return matched / len(required_strings)


def score_exact_match(actual: str, expected: str) -> float:
    """Binary score: 1.0 if exact match, 0.0 otherwise."""
    return 1.0 if actual.strip() == expected.strip() else 0.0


def score_distractor_avoidance(
    actual_calls: list[dict],
    distractor_tools: list[str],
) -> float:
    """Score 1.0 if no distractor tools were called, 0.0 if any were."""
    called_tools = {c["name"] for c in actual_calls}
    if any(d in called_tools for d in distractor_tools):
        return 0.0
    return 1.0


# --- Judge-based scoring ---


def score_with_judge(
    judge: JudgeClient,
    dimension: str,
    rubric: str,
    task_description: str,
    model_response: str,
    use_averaging: bool = True,
) -> ScoreResult:
    """Score using the judge model with optional 3x averaging."""
    if use_averaging:
        result = judge.evaluate_with_averaging(dimension, rubric, task_description, model_response)
    else:
        result = judge.evaluate(dimension, rubric, task_description, model_response)

    if isinstance(result, JudgeError):
        logger.error("Judge scoring failed for %s: %s", dimension, result.error)
        return ScoreResult(
            dimension=dimension,
            method="judge_rubric",
            score=0.0,
            weight=0.0,
            details={"error": result.error, "raw": result.raw_response},
        )

    assert isinstance(result, JudgeResult)
    # Normalize score to 0-1 (assuming 1-5 scale)
    normalized = max(0.0, min(1.0, (result.score - 1.0) / 4.0))
    return ScoreResult(
        dimension=dimension,
        method="judge_rubric",
        score=normalized,
        weight=0.0,  # Set by caller from task config
        details={
            "raw_score": result.score,
            "reasoning": result.reasoning,
            "confidence": result.confidence,
        },
    )


# --- Aggregation ---


def aggregate_task_scores(scores: list[ScoreResult]) -> float:
    """Compute weighted average of score results."""
    total_weight = sum(s.weight for s in scores)
    if total_weight == 0:
        return 0.0
    return sum(s.score * s.weight for s in scores) / total_weight


def compute_dimension_score(task_scores: list[TaskScore], dimension: str) -> float:
    """Average all task scores within a dimension."""
    dim_scores = [ts.weighted_total for ts in task_scores if ts.dimension == dimension]
    if not dim_scores:
        return 0.0
    return sum(dim_scores) / len(dim_scores)


def compute_composite(
    dimension_scores: dict[str, float],
    weights: dict[str, float] | None = None,
) -> float:
    """Compute composite score across dimensions.

    Default: equal weighting across all dimensions.
    """
    if not dimension_scores:
        return 0.0

    if weights is None:
        # Equal weighting
        return sum(dimension_scores.values()) / len(dimension_scores)

    total_weight = sum(weights.get(d, 0.0) for d in dimension_scores)
    if total_weight == 0:
        return 0.0

    return sum(dimension_scores[d] * weights.get(d, 0.0) for d in dimension_scores) / total_weight


def _call_matches(actual: dict, expected: dict) -> bool:
    """Check if an actual tool call matches an expected one."""
    if actual.get("name") != expected.get("name"):
        return False

    expected_args = expected.get("args", expected.get("arguments", {}))
    if not expected_args:
        return True

    # Check args_must_contain style
    if "args_must_contain" in expected:
        actual_args = actual.get("arguments", {})
        for key, required_values in expected["args_must_contain"].items():
            actual_val = str(actual_args.get(key, "")).lower()
            if isinstance(required_values, list):
                if not all(rv.lower() in actual_val for rv in required_values):
                    return False
            elif isinstance(required_values, str) and required_values.lower() not in actual_val:
                return False
        return True

    # Exact arg match
    actual_args = actual.get("arguments", {})
    return all(actual_args.get(k) == v for k, v in expected_args.items())
