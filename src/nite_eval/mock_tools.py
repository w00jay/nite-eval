"""Deterministic tool response simulator for evaluation tasks.

Matches tool calls against predefined response maps from task YAML definitions.
Supports exact match, partial/contains match, sequenced responses (for error-recovery
tasks), and fallback responses.
"""

import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class MockResponse:
    """A single mock response with its match criteria."""

    match: dict
    response: dict | None = None
    error: str | None = None
    sequence: list[dict] | None = None


@dataclass
class MockToolEnv:
    """Deterministic tool environment for a single evaluation task."""

    responses: dict[str, list[MockResponse]] = field(default_factory=dict)
    call_counts: dict[str, int] = field(default_factory=dict)
    call_log: list[dict] = field(default_factory=list)

    @classmethod
    def from_task_yaml(cls, mock_responses: dict) -> "MockToolEnv":
        """Build a MockToolEnv from the mock_responses section of a task YAML."""
        env = cls()
        for tool_name, response_defs in mock_responses.items():
            env.responses[tool_name] = []
            for rdef in response_defs:
                match_criteria = rdef.get("match", {})
                env.responses[tool_name].append(
                    MockResponse(
                        match=match_criteria,
                        response=rdef.get("response"),
                        error=rdef.get("error"),
                        sequence=rdef.get("sequence"),
                    )
                )
        return env

    def call(self, tool_name: str, arguments: dict) -> dict:
        """Execute a mock tool call and return the response.

        Returns a dict with either 'content' or 'error' key.
        """
        call_number = self.call_counts.get(tool_name, 0) + 1
        self.call_counts[tool_name] = call_number
        self.call_log.append({"name": tool_name, "arguments": arguments, "call_number": call_number})

        if tool_name not in self.responses:
            logger.warning("No mock responses defined for tool: %s", tool_name)
            return {"error": f"No mock defined for tool '{tool_name}'"}

        for mock in self.responses[tool_name]:
            if self._matches(mock.match, arguments, call_number):
                return self._resolve_response(mock, call_number)

        logger.warning("No matching mock for %s(%s)", tool_name, arguments)
        return {"error": f"No matching mock response for {tool_name} with given arguments"}

    def _matches(self, match: dict, arguments: dict, call_number: int) -> bool:
        """Check if arguments satisfy match criteria."""
        for key, expected in match.items():
            # Special: match on call sequence number
            if key == "_call_number":
                if call_number != expected:
                    return False
                continue

            # Contains-style matching: {query_contains: "search term"}
            if key.endswith("_contains"):
                field_name = key.removesuffix("_contains")
                actual = arguments.get(field_name, "")
                if expected == "any":
                    if field_name not in arguments:
                        return False
                elif isinstance(actual, str):
                    if expected.lower() not in actual.lower():
                        return False
                elif isinstance(actual, list):
                    if not any(expected.lower() in str(item).lower() for item in actual):
                        return False
                else:
                    if expected.lower() not in str(actual).lower():
                        return False
                continue

            # Exact match on nested objects
            actual = arguments.get(key)
            if isinstance(expected, dict) and isinstance(actual, dict):
                if not all(actual.get(k) == v for k, v in expected.items()):
                    return False
            elif actual != expected:
                return False

        return True

    def _resolve_response(self, mock: MockResponse, call_number: int) -> dict:
        """Resolve the actual response, handling sequences."""
        if mock.sequence is not None:
            # Sequence: return responses in order, repeat last on overflow
            idx = min(call_number - 1, len(mock.sequence) - 1)
            return mock.sequence[idx]

        if mock.error:
            return {"error": mock.error}

        if mock.response:
            return mock.response

        return {"error": "Mock response has no content"}

    def get_call_log(self) -> list[dict]:
        """Return the full call log for scoring."""
        return list(self.call_log)

    def reset(self) -> None:
        """Reset call counts and log for a fresh run."""
        self.call_counts.clear()
        self.call_log.clear()
