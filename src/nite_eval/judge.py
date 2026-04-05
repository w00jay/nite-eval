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

JSON_BLOCK_RE = re.compile(r"\{[^{}]*\}", re.DOTALL)

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

    Expects JSON with 'reasoning' and 'score' keys, possibly embedded in text.
    """
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
        return f"""You are evaluating {dimension}. Analyze step by step, then score.

## Rubric
{rubric}

## Input
Task: {task_description}
Response: {model_response}

Output ONLY: {{"reasoning": "...", "score": N}}"""

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
