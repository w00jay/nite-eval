"""Existence checks for paths named in the llama-swap config.

llama-swap resolves a model's command line only when it swaps that model in,
so a missing binary or GGUF surfaces mid-sweep as a load failure rather than
as a setup problem. This turns that into a pre-run check.

It matters most for entries that do not use the stock llama.cpp build:
`bonsai2-27b-ptq1` points at a separate PrismML fork checkout, because stock
llama.cpp rejects PQ2_0/PTQ1_0 as unknown quant types. Nothing else in the repo
references that checkout, so a rebuild or a `git clean` there breaks exactly one
model with no other warning.
"""

from __future__ import annotations

from pathlib import Path

import yaml


def _referenced_paths(cmd: str) -> list[str]:
    """Absolute paths a llama-server command line depends on.

    Covers the binary itself and every flag value that names a file — `-m` for
    the GGUF, but also `--chat-template-file` and friends, which are equally
    fatal when absent. Relative paths are skipped: a bare `llama-server` is
    resolved against PATH and is not ours to verify.
    """
    return [tok for tok in cmd.split() if tok.startswith("/")]


def check_referenced_files(config_path: Path, model_names: set[str] | None = None) -> list[str]:
    """Confirm the binaries and model files named in the config exist.

    Pass `model_names` to check only the entries a run will actually use — a
    sweep of one model must not fail because an unrelated entry's GGUF has
    moved. Returns human-readable problems; empty means everything resolves.
    """
    if not config_path.exists():
        return [f"{config_path} not found — cannot verify referenced files"]

    try:
        cfg = yaml.safe_load(config_path.read_text()) or {}
    except yaml.YAMLError as e:
        return [f"{config_path} is not valid YAML: {e}"]

    problems: list[str] = []
    for name, entry in (cfg.get("models") or {}).items():
        if model_names is not None and name not in model_names:
            continue
        cmd = str((entry or {}).get("cmd", ""))
        for path in _referenced_paths(cmd):
            if not Path(path).exists():
                problems.append(f"{name}: {path} does not exist (named in {config_path})")
    return problems
