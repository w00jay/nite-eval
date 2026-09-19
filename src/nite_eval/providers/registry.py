"""Build a model backend from a `models:` entry in eval_config.yaml.

Local models return None: the runner's built-in llama-server path is already
the right backend, and returning None keeps that path byte-for-byte unchanged.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from nite_eval.conversation_runner import ModelBackend

LOCAL_PROVIDERS = frozenset({"local", "llama.cpp", "vllm"})


def is_local(cfg: dict[str, Any]) -> bool:
    return cfg.get("provider", "local") in LOCAL_PROVIDERS


def model_id_for(cfg: dict[str, Any]) -> str:
    """The identifier sent to the provider (falls back to the display name)."""
    return cfg.get("api_model") or cfg["name"]


def native_tools_for(cfg: dict[str, Any]) -> bool:
    """Whether this model gets native tool calling.

    Explicit config wins. Otherwise API models default to native — it is how
    they are actually used — and local models default to Hermes, preserving the
    existing behavior and the comparability of past runs. Set
    `native_tools: false` on an API model to run it the local way and measure
    what the tool format alone is worth.
    """
    if "native_tools" in cfg:
        return bool(cfg["native_tools"])
    return not is_local(cfg)


def build_backend(cfg: dict[str, Any]) -> ModelBackend | None:
    """Instantiate the backend for one configured model, or None for local.

    Raises ValueError on an unknown provider rather than guessing, so a typo
    fails at startup instead of halfway through an overnight run.
    """
    provider = cfg.get("provider", "local")
    if provider in LOCAL_PROVIDERS:
        return None

    model_id = model_id_for(cfg)
    if provider == "anthropic":
        from nite_eval.providers.anthropic_backend import AnthropicBackend

        return AnthropicBackend(
            model_id=model_id,
            api_key_env=cfg.get("api_key_env") or "ANTHROPIC_API_KEY",
            params=cfg.get("params"),
        )

    if provider in ("openai", "openai_compatible"):
        from nite_eval.providers.openai_backend import OpenAIBackend

        return OpenAIBackend(
            model_id=model_id,
            api_key_env=cfg.get("api_key_env") or "OPENAI_API_KEY",
            base_url=cfg.get("base_url"),
            provider=cfg.get("provider_label") or provider,
            temperature=cfg.get("temperature"),
            max_tokens_param=cfg.get("max_tokens_param") or "max_tokens",
            params=cfg.get("params"),
        )

    raise ValueError(f"{cfg['name']}: unknown provider {provider!r}")
