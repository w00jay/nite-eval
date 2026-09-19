"""Model backends: local llama-server (in the runner), Anthropic, OpenAI-compatible."""

from nite_eval.providers.registry import (
    LOCAL_PROVIDERS,
    build_backend,
    is_local,
    model_id_for,
    native_tools_for,
)

__all__ = [
    "LOCAL_PROVIDERS",
    "build_backend",
    "is_local",
    "model_id_for",
    "native_tools_for",
]
