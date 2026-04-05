"""AST-based comparator for Hermes-format tool calls.

Compares model-generated <tool_call> output against gold-standard function calls
using structural matching — no LLM judge needed. Replaces BFCL integration.

Scoring dimensions:
- Function name accuracy (exact match)
- Argument name accuracy (exact match on required args)
- Argument value accuracy (exact + fuzzy matching)
- Sequence correctness (ordered matching for multi-step tasks)
- Call count accuracy (right number of calls)
"""

import logging
import math
from dataclasses import dataclass, field

from nite_eval.hermes_parser import ParsedResponse, ToolCall

logger = logging.getLogger(__name__)


@dataclass
class CallComparison:
    """Result of comparing a single tool call against its gold standard."""

    expected_name: str
    actual_name: str | None
    name_match: bool
    arg_matches: dict[str, bool] = field(default_factory=dict)
    arg_value_matches: dict[str, bool] = field(default_factory=dict)
    extra_args: list[str] = field(default_factory=list)
    missing_args: list[str] = field(default_factory=list)

    @property
    def name_score(self) -> float:
        return 1.0 if self.name_match else 0.0

    @property
    def arg_name_score(self) -> float:
        if not self.arg_matches:
            return 1.0  # No args expected
        return sum(self.arg_matches.values()) / len(self.arg_matches)

    @property
    def arg_value_score(self) -> float:
        if not self.arg_value_matches:
            return 1.0
        return sum(self.arg_value_matches.values()) / len(self.arg_value_matches)


@dataclass
class GoldCall:
    """A gold-standard expected tool call."""

    name: str
    arguments: dict = field(default_factory=dict)
    required_args: list[str] | None = None  # If None, all args are required
    fuzzy_args: list[str] = field(default_factory=list)  # Args where contains-match is OK


@dataclass
class ASTComparisonResult:
    """Full comparison result for a model response against gold standard."""

    call_comparisons: list[CallComparison]
    sequence_score: float  # How well the call order matches
    count_score: float  # Penalty for wrong number of calls
    extra_calls: list[ToolCall] = field(default_factory=list)  # Calls not in gold standard
    missing_calls: list[GoldCall] = field(default_factory=list)  # Expected but not made

    @property
    def name_accuracy(self) -> float:
        if not self.call_comparisons:
            return 0.0
        return sum(c.name_score for c in self.call_comparisons) / len(self.call_comparisons)

    @property
    def arg_name_accuracy(self) -> float:
        scored = [c for c in self.call_comparisons if c.name_match]
        if not scored:
            return 0.0
        return sum(c.arg_name_score for c in scored) / len(scored)

    @property
    def arg_value_accuracy(self) -> float:
        scored = [c for c in self.call_comparisons if c.name_match]
        if not scored:
            return 0.0
        return sum(c.arg_value_score for c in scored) / len(scored)

    @property
    def overall_score(self) -> float:
        """Weighted composite: name 30%, arg names 20%, arg values 30%, sequence 10%, count 10%."""
        # No calls expected and none made = perfect
        if not self.call_comparisons and self.count_score == 1.0:
            return 1.0
        return (
            0.30 * self.name_accuracy
            + 0.20 * self.arg_name_accuracy
            + 0.30 * self.arg_value_accuracy
            + 0.10 * self.sequence_score
            + 0.10 * self.count_score
        )


def compare_calls(
    parsed: ParsedResponse,
    gold_standard: list[GoldCall],
    ordered: bool = True,
) -> ASTComparisonResult:
    """Compare model tool calls against gold-standard expected calls.

    Args:
        parsed: Output from extract_tool_calls()
        gold_standard: Expected sequence of tool calls
        ordered: If True, calls must appear in the expected order.
                 If False, any order is accepted (set matching).
    """
    actual_calls = parsed.tool_calls

    if ordered:
        return _compare_ordered(actual_calls, gold_standard)
    return _compare_unordered(actual_calls, gold_standard)


def _compare_ordered(
    actual: list[ToolCall],
    expected: list[GoldCall],
) -> ASTComparisonResult:
    """Compare calls preserving order — greedy forward matching."""
    comparisons: list[CallComparison] = []
    matched_actual: set[int] = set()
    actual_idx = 0

    for gold in expected:
        best_match: CallComparison | None = None

        while actual_idx < len(actual):
            tc = actual[actual_idx]
            actual_idx += 1

            if tc.name == gold.name:
                comp = _compare_single(tc, gold)
                best_match = comp
                matched_actual.add(actual_idx - 1)
                break

        if best_match is None:
            # Expected call not found
            comparisons.append(
                CallComparison(
                    expected_name=gold.name,
                    actual_name=None,
                    name_match=False,
                    missing_args=list(gold.arguments.keys()),
                )
            )
        else:
            comparisons.append(best_match)

    # Collect extra calls
    extra = [actual[i] for i in range(len(actual)) if i not in matched_actual]

    # Sequence score: fraction of expected calls matched in order
    matched_count = sum(1 for c in comparisons if c.name_match)
    sequence_score = matched_count / len(expected) if expected else 1.0

    # Count score
    count_score = _count_score(len(actual), len(expected))

    # Missing calls
    missing = [expected[i] for i, c in enumerate(comparisons) if not c.name_match]

    return ASTComparisonResult(
        call_comparisons=comparisons,
        sequence_score=sequence_score,
        count_score=count_score,
        extra_calls=extra,
        missing_calls=missing,
    )


def _compare_unordered(
    actual: list[ToolCall],
    expected: list[GoldCall],
) -> ASTComparisonResult:
    """Compare calls ignoring order — best-match assignment."""
    comparisons: list[CallComparison] = []
    used_actual: set[int] = set()

    for gold in expected:
        best_match: CallComparison | None = None
        best_idx = -1
        best_score = -1.0

        for i, tc in enumerate(actual):
            if i in used_actual:
                continue
            if tc.name != gold.name:
                continue

            comp = _compare_single(tc, gold)
            score = (comp.arg_name_score + comp.arg_value_score) / 2
            if score > best_score:
                best_score = score
                best_match = comp
                best_idx = i

        if best_match is not None:
            comparisons.append(best_match)
            used_actual.add(best_idx)
        else:
            comparisons.append(
                CallComparison(
                    expected_name=gold.name,
                    actual_name=None,
                    name_match=False,
                    missing_args=list(gold.arguments.keys()),
                )
            )

    extra = [actual[i] for i in range(len(actual)) if i not in used_actual]
    matched_count = sum(1 for c in comparisons if c.name_match)
    sequence_score = matched_count / len(expected) if expected else 1.0
    count_score = _count_score(len(actual), len(expected))
    missing = [expected[i] for i, c in enumerate(comparisons) if not c.name_match]

    return ASTComparisonResult(
        call_comparisons=comparisons,
        sequence_score=sequence_score,
        count_score=count_score,
        extra_calls=extra,
        missing_calls=missing,
    )


def _compare_single(actual: ToolCall, gold: GoldCall) -> CallComparison:
    """Compare a single actual call against a gold standard call."""
    comp = CallComparison(
        expected_name=gold.name,
        actual_name=actual.name,
        name_match=(actual.name == gold.name),
    )

    # Determine which args to check
    expected_args = gold.arguments
    check_args = gold.required_args if gold.required_args is not None else list(expected_args.keys())

    for arg_name in check_args:
        if arg_name in actual.arguments:
            comp.arg_matches[arg_name] = True

            # Value comparison
            expected_val = expected_args.get(arg_name)
            actual_val = actual.arguments[arg_name]

            if arg_name in gold.fuzzy_args:
                comp.arg_value_matches[arg_name] = _fuzzy_match(actual_val, expected_val)
            else:
                comp.arg_value_matches[arg_name] = _exact_match(actual_val, expected_val)
        else:
            comp.arg_matches[arg_name] = False
            comp.arg_value_matches[arg_name] = False
            comp.missing_args.append(arg_name)

    # Track extra args (not errors, but noted)
    for arg_name in actual.arguments:
        if arg_name not in expected_args:
            comp.extra_args.append(arg_name)

    return comp


def _exact_match(actual: object, expected: object) -> bool:
    """Compare two values for equality, with type coercion for numbers."""
    if actual == expected:
        return True

    # Numeric comparison with tolerance
    if isinstance(actual, int | float) and isinstance(expected, int | float):
        if expected == 0:
            return actual == 0
        return math.isclose(float(actual), float(expected), rel_tol=1e-6)

    # String comparison (case-insensitive for simple strings)
    if isinstance(actual, str) and isinstance(expected, str):
        return actual.strip().lower() == expected.strip().lower()

    # List comparison (order-sensitive)
    if isinstance(actual, list) and isinstance(expected, list):
        if len(actual) != len(expected):
            return False
        return all(_exact_match(a, e) for a, e in zip(actual, expected, strict=False))

    # Dict comparison (recursive)
    if isinstance(actual, dict) and isinstance(expected, dict):
        if set(actual.keys()) != set(expected.keys()):
            return False
        return all(_exact_match(actual[k], expected[k]) for k in expected)

    return False


def _fuzzy_match(actual: object, expected: object) -> bool:
    """Check if actual value contains the expected value (for text args like queries)."""
    if expected is None:
        return True

    actual_str = str(actual).lower()
    expected_str = str(expected).lower()

    # Check if all key terms from expected appear in actual
    terms = [t.strip() for t in expected_str.split() if len(t.strip()) > 2]
    if not terms:
        return True

    matched = sum(1 for t in terms if t in actual_str)
    return matched / len(terms) >= 0.6


def _count_score(actual_count: int, expected_count: int) -> float:
    """Score for call count accuracy. 1.0 if exact, decays with difference."""
    if expected_count == 0:
        return 1.0 if actual_count == 0 else 0.0
    diff = abs(actual_count - expected_count)
    return max(0.0, 1.0 - (diff / expected_count))
