"""Client for the judge model running on RTX 3060 (port 8081).

Sends rubric-based evaluation prompts to the judge and parses structured responses.
Supports retry with averaging for variance reduction.
"""

import json
import logging
import re
from dataclasses import dataclass

import httpx

logger = logging.getLogger(__name__)

# Match JSON blocks, including nested braces (greedy from first { to last })
JSON_BLOCK_RE = re.compile(r"\{[^{}]*\}", re.DOTALL)
# Fallback: extract score from patterns like "score": 4 or **Score: 4**
SCORE_FALLBACK_RE = re.compile(r"(?:\"score\"|score)\s*[:=]\s*(\d(?:\.\d)?)", re.IGNORECASE)

DEFAULT_JUDGE_URL = "http://127.0.0.1:8081/v1"
DEFAULT_JUDGE_MODEL = "selene-1-mini"


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
    ) -> JudgeResult | JudgeError:
        """Send a single evaluation request to the judge."""
        prompt = self._build_prompt(dimension, rubric, task_description, model_response)
        return self._call(prompt)

    def evaluate_with_averaging(
        self,
        dimension: str,
        rubric: str,
        task_description: str,
        model_response: str,
        n_runs: int = 3,
    ) -> JudgeResult | JudgeError:
        """Run evaluation n times and average scores for variance reduction."""
        results: list[JudgeResult] = []
        errors: list[JudgeError] = []

        for i in range(n_runs):
            result = self.evaluate(dimension, rubric, task_description, model_response)
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

    def _build_prompt(
        self,
        dimension: str,
        rubric: str,
        task_description: str,
        model_response: str,
    ) -> str:
        return f"""You are a strict evaluator scoring "{dimension}" on a 3-point scale.

IMPORTANT: Be critical. Most responses deserve a 3. Only give 5 for truly
exceptional work. Only give 1 for clear failures. If in doubt, score 3.

## Scoring Scale
1 = Poor (clear failures, major gaps)
3 = Acceptable (adequate, meets basic requirements)
5 = Excellent (exceptional, goes above and beyond)

You MUST pick exactly 1, 3, or 5. No other scores.

## Rubric for {dimension}
{rubric}

## Task
{task_description}

## Response to Evaluate
{model_response}

First write 2-3 sentences of reasoning, then output your score.
Output ONLY valid JSON: {{"reasoning": "your 2-3 sentence analysis", "score": N}}"""

    def _call(self, prompt: str) -> JudgeResult | JudgeError:
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
            return JudgeError(error=f"http_error: {e}", raw_response="")

        data = resp.json()
        raw = data["choices"][0]["message"]["content"]
        return _parse_judge_response(raw)

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "JudgeClient":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()
