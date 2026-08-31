"""Client for the judge model running on RTX 3060 (port 9091).

Sends rubric-based evaluation prompts to the judge and parses structured responses.
Supports retry with averaging for variance reduction.
"""

import json
import logging
import re
import time
from dataclasses import dataclass

import httpx

logger = logging.getLogger(__name__)

# Match JSON blocks, including nested braces (greedy from first { to last })
JSON_BLOCK_RE = re.compile(r"\{[^{}]*\}", re.DOTALL)
# Fallback: extract score from patterns like "score": 4 or **Score: 4**
SCORE_FALLBACK_RE = re.compile(r"(?:\"score\"|score)\s*[:=]\s*(\d(?:\.\d)?)", re.IGNORECASE)

DEFAULT_JUDGE_URL = "http://127.0.0.1:9091/v1"
# Matches the model name run_nightly.sh serves on DEFAULT_JUDGE_URL. Was
# "selene-1-mini", whose GGUF is gone and which no launcher has served since
# the pipeline moved to the RewardAnything/Flow-Judge pair — so any caller
# relying on this default (scripts/smoke_test.py) was asking port 9091 for a
# model that was not loaded there.
DEFAULT_JUDGE_MODEL = "reward-anything"


@dataclass
class JudgeResult:
    reasoning: str
    score: float
    raw_response: str
    confidence: float | None = None


@dataclass
class JudgeError:
    error: str
    raw_response: str


def _extract_json_object(raw: str, required_key: str) -> dict:
    """Find the first JSON object in `raw` containing `required_key`.

    Judges wrap JSON in prose or fences often enough that a bare json.loads on
    the whole response is unreliable.
    """
    for candidate in JSON_BLOCK_RE.findall(raw):
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict) and required_key in parsed:
            return parsed

    parsed = json.loads(raw.strip())
    if not isinstance(parsed, dict) or required_key not in parsed:
        raise ValueError(f"no JSON object with key '{required_key}'")
    return parsed


def _parse_judge_response(raw: str) -> JudgeResult | JudgeError:
    """Extract structured score from judge response.

    Tries multiple strategies:
    1. Find JSON with 'score' key (simple regex for non-nested)
    2. Try parsing the entire response as JSON
    3. Try json.loads on the largest {...} block found via bracket matching
    4. Fallback: extract score from "score": N or "Score: N" patterns
    """
    # Strategy 1: Simple regex for non-nested JSON blocks
    matches = JSON_BLOCK_RE.findall(raw)
    for match in matches:
        try:
            parsed = json.loads(match)
            if "score" in parsed:
                return JudgeResult(
                    reasoning=parsed.get("reasoning", ""),
                    score=float(parsed["score"]),
                    raw_response=raw,
                )
        except (json.JSONDecodeError, ValueError, TypeError):
            continue

    # Strategy 2: Try parsing the full response as JSON
    try:
        parsed = json.loads(raw.strip())
        if "score" in parsed:
            return JudgeResult(
                reasoning=parsed.get("reasoning", ""),
                score=float(parsed["score"]),
                raw_response=raw,
            )
    except (json.JSONDecodeError, ValueError, TypeError):
        pass

    # Strategy 3: Bracket-matched extraction for nested JSON
    for i, ch in enumerate(raw):
        if ch == "{":
            depth = 0
            for j in range(i, len(raw)):
                if raw[j] == "{":
                    depth += 1
                elif raw[j] == "}":
                    depth -= 1
                    if depth == 0:
                        block = raw[i : j + 1]
                        try:
                            parsed = json.loads(block)
                            if "score" in parsed:
                                return JudgeResult(
                                    reasoning=parsed.get("reasoning", ""),
                                    score=float(parsed["score"]),
                                    raw_response=raw,
                                )
                        except (json.JSONDecodeError, ValueError, TypeError):
                            pass
                        break

    # Strategy 4: Regex fallback for score patterns in free text
    score_match = SCORE_FALLBACK_RE.search(raw)
    if score_match:
        score = float(score_match.group(1))
        # Extract reasoning as everything before the score
        reasoning = raw[: score_match.start()].strip()
        logger.warning("Used fallback score extraction: %.1f from raw response", score)
        return JudgeResult(
            reasoning=reasoning,
            score=score,
            raw_response=raw,
        )

    logger.error("Failed to parse judge response: %s", raw[:200])
    return JudgeError(error="no_valid_score_json", raw_response=raw)


class JudgeClient:
    """Client for the persistent judge model."""

    def __init__(
        self,
        base_url: str = DEFAULT_JUDGE_URL,
        model: str = DEFAULT_JUDGE_MODEL,
        temperature: float = 0.1,
        max_tokens: int = 1024,
        timeout: float = 60.0,
    ):
        self.base_url = base_url
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._client = httpx.Client(timeout=timeout)

    def evaluate(
        self,
        dimension: str,
        rubric: str,
        task_description: str,
        model_response: str,
        evidence: str = "",
    ) -> JudgeResult | JudgeError:
        """Send a single evaluation request to the judge."""
        prompt = self._build_prompt(dimension, rubric, task_description, model_response, evidence)
        return self._call(prompt)

    def evaluate_with_averaging(
        self,
        dimension: str,
        rubric: str,
        task_description: str,
        model_response: str,
        n_runs: int = 3,
        evidence: str = "",
    ) -> JudgeResult | JudgeError:
        """Run evaluation n times and average scores for variance reduction."""
        results: list[JudgeResult] = []
        errors: list[JudgeError] = []

        for i in range(n_runs):
            result = self.evaluate(dimension, rubric, task_description, model_response, evidence)
            if isinstance(result, JudgeError):
                errors.append(result)
                logger.warning("Judge run %d/%d failed: %s", i + 1, n_runs, result.error)
            else:
                results.append(result)

        if not results:
            return JudgeError(
                error=f"all_{n_runs}_runs_failed",
                raw_response="; ".join(e.raw_response for e in errors),
            )

        avg_score = sum(r.score for r in results) / len(results)
        score_variance = sum((r.score - avg_score) ** 2 for r in results) / len(results) if len(results) > 1 else 0.0

        return JudgeResult(
            reasoning=results[0].reasoning,  # Use first run's reasoning
            score=avg_score,
            raw_response=results[0].raw_response,
            confidence=1.0 - min(score_variance / 2.0, 1.0),  # High variance = low confidence
        )

    # Max characters for model response in judge prompt (~1500 tokens ≈ 6000 chars).
    # Leaves room for system prompt (~200 tokens), rubric, task description, and
    # judge output (max_tokens) within the judge's 4096 context window.
    MAX_RESPONSE_CHARS = 6000

    def _build_prompt(
        self,
        dimension: str,
        rubric: str,
        task_description: str,
        model_response: str,
        evidence: str = "",
    ) -> str:
        # Truncate long responses to fit judge context window
        if len(model_response) > self.MAX_RESPONSE_CHARS:
            model_response = (
                model_response[: self.MAX_RESPONSE_CHARS] + "\n\n[... response truncated for evaluation ...]"
            )

        # Criteria like no_hallucination and data_accuracy ask whether the
        # response's facts match what the tools actually returned. Without the
        # tool results the judge can only guess, so those criteria were
        # unjudgeable and previously fell through to a free 1.0.
        evidence_block = ""
        if evidence:
            if len(evidence) > self.MAX_RESPONSE_CHARS:
                evidence = evidence[: self.MAX_RESPONSE_CHARS] + "\n\n[... evidence truncated ...]"
            evidence_block = (
                "\n## Tool Results (ground truth)\n"
                "These are the actual values the tools returned. Any figure in the\n"
                "response that contradicts these, or that appears nowhere in them,\n"
                "is fabricated.\n\n"
                f"{evidence}\n"
            )

        return f"""You are a strict evaluator scoring "{dimension}" on a 5-point scale.

Be critical. Reserve 5 for work with no meaningful gaps, and 1 for clear
failures. Use the full range: 2 and 4 exist for responses that sit between
the anchors, which most do.

## Scoring Scale
1 = Poor — clear failures, major gaps
2 = Weak — meets some requirements, notable gaps
3 = Acceptable — adequate, meets basic requirements
4 = Strong — exceeds requirements, minor gaps only
5 = Excellent — no meaningful gaps

Pick the single integer from 1 to 5 that best fits. Do not default to 3;
if a response is better than "adequate" but short of "strong", score 4.

## Rubric for {dimension}
{rubric}

## Task
{task_description}
{evidence_block}
## Response to Evaluate
{model_response}

First write 2-3 sentences of reasoning, then output your score.
Output ONLY valid JSON: {{"reasoning": "your 2-3 sentence analysis", "score": N}}"""

    def _call(self, prompt: str, max_retries: int = 3) -> JudgeResult | JudgeError:
        last_error = ""
        for attempt in range(1, max_retries + 1):
            try:
                resp = self._client.post(
                    f"{self.base_url}/chat/completions",
                    json={
                        "model": self.model,
                        "messages": [{"role": "user", "content": prompt}],
                        "temperature": self.temperature,
                        "max_tokens": self.max_tokens,
                    },
                )
                resp.raise_for_status()
            except httpx.HTTPError as e:
                last_error = f"http_error: {e}"
                if attempt < max_retries:
                    import time

                    wait = 5 * attempt
                    logger.warning(
                        "Judge call failed (attempt %d/%d), retrying in %ds: %s", attempt, max_retries, wait, e
                    )
                    time.sleep(wait)
                    continue
                return JudgeError(error=last_error, raw_response="")

            data = resp.json()
            raw = data["choices"][0]["message"]["content"] or ""
            if not raw.strip():
                last_error = "empty_response"
                if attempt < max_retries:
                    import time

                    wait = 2 * attempt
                    logger.warning(
                        "Judge returned empty content (attempt %d/%d), retrying in %ds",
                        attempt,
                        max_retries,
                        wait,
                    )
                    time.sleep(wait)
                    continue
                return JudgeError(error=last_error, raw_response="")
            return _parse_judge_response(raw)

        return JudgeError(error=last_error, raw_response="")

    def evaluate_checklist(
        self,
        criteria: list[str],
        task_description: str,
        model_response: str,
    ) -> list[bool] | JudgeError:
        """Ask the judge which checklist criteria the response actually meets.

        One call for the whole checklist rather than one per criterion. Replaces
        substring matching, which counted a criterion as met if any word longer
        than three characters from it appeared anywhere in the response — so
        "Addresses embedding strategy" was satisfied by the word "strategy".

        Returns a list of booleans aligned with `criteria`.
        """
        if not criteria:
            return []

        numbered = "\n".join(f"{i + 1}. {c}" for i, c in enumerate(criteria))
        prompt = f"""You are checking whether a response addresses specific requirements.

For each numbered requirement, decide whether the response genuinely addresses
it. Mentioning a keyword is not enough — the response must actually cover the
substance of the requirement.

## Task Given To The Model
{task_description}

## Requirements
{numbered}

## Response To Check
{model_response[: self.MAX_RESPONSE_CHARS]}

Output ONLY valid JSON, an object with a "met" array of {len(criteria)} booleans
in the same order as the requirements:
{{"met": [true, false, ...]}}"""

        raw_result = self._call_raw(prompt)
        if isinstance(raw_result, JudgeError):
            return raw_result

        try:
            parsed = _extract_json_object(raw_result, "met")
            met = parsed["met"]
            if not isinstance(met, list):
                raise ValueError("'met' is not a list")
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            return JudgeError(error=f"checklist_parse_error: {e}", raw_response=raw_result)

        # Pad or trim so a miscounted response cannot silently shift alignment.
        met = [bool(x) for x in met][: len(criteria)]
        met += [False] * (len(criteria) - len(met))
        return met

    def _call_raw(self, prompt: str, max_retries: int = 3) -> str | JudgeError:
        """Send a prompt and return the raw text, for non-score judge calls."""
        last_error = ""
        for attempt in range(1, max_retries + 1):
            try:
                resp = self._client.post(
                    f"{self.base_url}/chat/completions",
                    json={
                        "model": self.model,
                        "messages": [{"role": "user", "content": prompt}],
                        "temperature": self.temperature,
                        "max_tokens": self.max_tokens,
                    },
                )
                resp.raise_for_status()
            except httpx.HTTPError as e:
                last_error = f"http_error: {e}"
                if attempt < max_retries:
                    time.sleep(5 * attempt)
                    continue
                return JudgeError(error=last_error, raw_response="")

            raw = resp.json()["choices"][0]["message"]["content"] or ""
            if raw.strip():
                return raw
            last_error = "empty_response"
            if attempt < max_retries:
                time.sleep(2 * attempt)

        return JudgeError(error=last_error, raw_response="")

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "JudgeClient":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()


# Dimensions where Flow-Judge outperforms (agentic: 5-bias matches true excellence)
FLOW_JUDGE_DIMENSIONS = frozenset({"reasoning_quality", "practical_output"})

DEFAULT_FLOW_JUDGE_MODEL = "flow-judge"
DEFAULT_REWARD_ANYTHING_MODEL = "reward-anything"


class RoutedJudgeClient:
    """Routes evaluation dimensions to the best-performing judge model.

    Flow-Judge handles agentic dimensions (reasoning_quality, practical_output)
    where its 5-bias correctly identifies excellence. RewardAnything handles
    all other dimensions where its 3-bias aligns with typical scores.

    Supports both shared-port (llama-swap) and split-port (direct servers)
    configurations via separate base_url per judge.
    """

    def __init__(
        self,
        base_url: str = DEFAULT_JUDGE_URL,
        flow_judge_model: str = DEFAULT_FLOW_JUDGE_MODEL,
        reward_anything_model: str = DEFAULT_REWARD_ANYTHING_MODEL,
        flow_judge_url: str | None = None,
        reward_anything_url: str | None = None,
        flow_judge_dimensions: frozenset[str] = FLOW_JUDGE_DIMENSIONS,
        temperature: float = 0.1,
        max_tokens: int = 1024,
        timeout: float = 120.0,
    ):
        self._flow_dims = flow_judge_dimensions
        self._flow = JudgeClient(
            base_url=flow_judge_url or base_url,
            model=flow_judge_model,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
        )
        self._reward = JudgeClient(
            base_url=reward_anything_url or base_url,
            model=reward_anything_model,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
        )

    def _select(self, dimension: str) -> JudgeClient:
        return self._flow if dimension in self._flow_dims else self._reward

    def evaluate(
        self,
        dimension: str,
        rubric: str,
        task_description: str,
        model_response: str,
        evidence: str = "",
    ) -> JudgeResult | JudgeError:
        """Route to the best judge for this dimension."""
        judge = self._select(dimension)
        logger.debug("Routing %s → %s", dimension, judge.model)
        return judge.evaluate(dimension, rubric, task_description, model_response, evidence)

    def evaluate_checklist(
        self,
        criteria: list[str],
        task_description: str,
        model_response: str,
    ) -> list[bool] | JudgeError:
        """Route a checklist check. Coverage is a conservative judgement, so it
        goes to the conservative judge rather than the excellence-recognising one."""
        judge = self._select("coverage")
        return judge.evaluate_checklist(criteria, task_description, model_response)

    def evaluate_with_averaging(
        self,
        dimension: str,
        rubric: str,
        task_description: str,
        model_response: str,
        n_runs: int = 3,
        evidence: str = "",
    ) -> JudgeResult | JudgeError:
        """Route to the best judge for this dimension, with variance reduction."""
        judge = self._select(dimension)
        logger.debug("Routing %s → %s (n=%d)", dimension, judge.model, n_runs)
        return judge.evaluate_with_averaging(dimension, rubric, task_description, model_response, n_runs, evidence)

    def close(self) -> None:
        self._flow.close()
        self._reward.close()

    def __enter__(self) -> "RoutedJudgeClient":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()
