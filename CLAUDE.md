# nite-eval

Autonomous overnight LLM evaluation pipeline for local models on dual GPUs (RTX 3090 + RTX 3060).

## Architecture

- **GPU 0 (RTX 3090):** Target models via llama-swap on :8080
- **GPU 2 (Tesla P40):** Judge models as direct llama-server instances — RewardAnything on :9091, Flow-Judge on :9092
- **Orchestrator:** Python pipeline that swaps models, runs evals, stores results in SQLite
- **4 evaluation layers:** static benchmarks (lm-eval-harness), agentic (Inspect AI), tool-calling (BFCL or custom AST), custom tasks (Hermes format)
- **Layer 5 (Arena-Hard-Auto):** deferred — requires frontier-class judge

## Key Paths

- llama-server: `/home/woojay/P/llama.cpp/build/bin/llama-server`
- llama-swap: `/home/woojay/T/llama-swap/llama-swap`
- Target models: `/home/woojay/P/llama.cpp/build/bin/` (GGUFs)
- Judge models: `~/.cache/huggingface/` via `huggingface-cli`
- llama-swap config (targets): `config/llama_swap_config.yaml`
- llama-swap config (judges): `config/judge_swap_config.yaml`
- Task definitions: `PLANS/nite-eval-tasks.yaml` (15 tasks, to be split into `tasks/`)

## Target Models

- `qwen3.5-27b` — Qwen 3.5 27B Q4_K_M
- `qwen3.5-9b` — Qwen 3.5 9B Q4_K_M
- `gemma4-26b-a4b` — Gemma 4 26B-A4B Q4_K_M

## Commands

```bash
uv run ruff check --fix . && uv run ruff format .   # lint
uv run pyright                                        # type check
uv run python -m pytest -v                            # test (use uv's venv, not conda)

# Start servers
# Target models (GPU 1 = RTX 3090)
CUDA_VISIBLE_DEVICES=1 /home/woojay/T/llama-swap/llama-swap \
  --config config/llama_swap_config.yaml --listen :8080

# Judge models (GPU 2 = Tesla P40, both fit simultaneously)
CUDA_VISIBLE_DEVICES=2 /home/woojay/P/llama.cpp/build/bin/llama-server \
  -m /home/woojay/models/RewardAnything-8B-v1.Q6_K.gguf \
  --port 9091 -ngl 999 --ctx-size 4096 --no-webui
CUDA_VISIBLE_DEVICES=2 /home/woojay/P/llama.cpp/build/bin/llama-server \
  -m /home/woojay/models/Flow-Judge-v0.1.Q6_K.gguf \
  --port 9092 -ngl 999 --ctx-size 4096 --no-webui

# Smoke test
uv run python scripts/smoke_test.py --model qwen3.5-9b
```

## Conventions

- All model communication via OpenAI-compatible HTTP APIs
- Hermes tool-calling format (`<tool_call>` tags) for all target models
- Deterministic scoring where possible; judge only for subjective dimensions
- SQLite for results, checkpointing, and resume
- `uv` for package management (never pip)
- Tests via `uv run python -m pytest` (not bare `pytest`, which hits conda env)

## Module Map

- `src/nite_eval/hermes_parser.py` — Parse/validate `<tool_call>` XML tags
- `src/nite_eval/model_manager.py` — Start/stop llama-server, llama-swap, health checks
- `src/nite_eval/judge.py` — JudgeClient + RoutedJudgeClient (dimension→model routing), 3x averaging, JSON parsing
- `src/nite_eval/mock_tools.py` — Deterministic tool responses from task YAML definitions
- `src/nite_eval/conversation_runner.py` — Multi-turn agent loop with Hermes tool execution
- `src/nite_eval/scoring.py` — Sequence match, subset match, checklist, composite scoring
- `src/nite_eval/task_loader.py` — Load task YAML definitions from `tasks/` with filtering
- `src/nite_eval/results_db.py` — SQLite storage for results, checkpointing, and resume
- `src/nite_eval/rubrics.py` — Centralized rubric definitions for judge-scored dimensions
- `src/nite_eval/orchestrator.py` — Main pipeline: load tasks → run models → score → persist → report
- `src/nite_eval/report.py` — Markdown report generator with per-task breakdowns and latency
