# nite-eval

Autonomous overnight LLM evaluation pipeline for local GGUF models. Runs multi-turn agentic tasks with the Hermes tool-calling format, scores them with dimension-routed judge models, and produces comparison reports.

Built for dual-GPU rigs running [llama.cpp](https://github.com/ggerganov/llama.cpp) + [llama-swap](https://github.com/mostlygeek/llama-swap), but the orchestrator is just an OpenAI-compatible HTTP client — any backend that speaks `/v1/chat/completions` will work.

## What it does

Evaluates local models across 4 dimensions (15 tasks total):

- **Research** (3 tasks) — multi-step information gathering and synthesis
- **Planning** (3 tasks) — task decomposition, dependency ordering, risk assessment
- **Coding** (4 tasks) — code generation with iterative tool use
- **Agentic** (5 tasks) — multi-turn tool calling, error recovery, context maintenance

Each task runs as a multi-turn conversation with mock tools (deterministic responses defined in YAML). Scoring is a mix of deterministic methods (checklist, sequence match, subset match) and LLM judges (rubric-based ternary scoring). Results persist to SQLite with checkpoint/resume.

## Sample results

Run `run-20260418-042455` on the reference hardware, 4 models × 15 tasks, `max_tokens=4096`, qwen3.6 with `/no_think`:

| Model | Research | Planning | Coding | Agentic | Composite |
|-------|---------:|---------:|-------:|--------:|----------:|
| **qwen3.6-35b-a3b** | 0.82 | 0.93 | 0.28 | 0.81 | **0.71** |
| qwen3.5-27b | 0.70 | 0.72 | 0.21 | 0.81 | 0.61 |
| qwen3.5-9b | 0.70 | 0.69 | 0.24 | 0.77 | 0.60 |
| gemma4-26b-a4b | 0.72 | 0.69 | 0.28 | 0.69 | 0.60 |

Notes:
- Qwen models use the standard Hermes tool-call format. Gemma 4 emits tool calls in a Harmony-style format (`<|tool_call>call:FUNC{…}<tool_call|>`); the parser handles both.
- Reasoning-mode models (Qwen 3.6 MoE) need `/no_think` appended to the system prompt — without it, the model consumes the entire token budget inside `<think>…</think>` before producing an answer. Configure per-model via the `system_suffix` field in `config/eval_config.yaml`.
- Coding scores cluster at 0.21–0.28 across all models — a rubric ceiling (tasks ask for complete implementations within a 4096-token budget), not a per-model weakness.

## Hardware (reference setup)

| GPU | Role | Port |
|-----|------|------|
| RTX 3090 (24GB) | Target models via llama-swap | `:9070` |
| Tesla P40 (24GB) | Judge models (both fit simultaneously) | `:9091`, `:9092` |

You can use any GPU layout — just adjust ports and GPU indices in `.env`. CPU-only also works if you have patience.

## Setup

```bash
# 1. Clone and install
git clone https://github.com/w00jay/nite-eval.git
cd nite-eval
uv sync

# 2. Configure paths
cp .env.example .env
$EDITOR .env   # set LLAMA_SERVER_BIN, LLAMA_SWAP_BIN, GGUF_DIR, JUDGE_MODEL_DIR, GPUs

# 3. Configure target models for llama-swap
cp config/llama_swap_config.example.yaml config/llama_swap_config.yaml
$EDITOR config/llama_swap_config.yaml   # absolute paths to your GGUFs
```

You'll need:

- [llama.cpp](https://github.com/ggerganov/llama.cpp) built with CUDA (`llama-server` binary)
- [llama-swap](https://github.com/mostlygeek/llama-swap) binary
- Target GGUFs of your choice (defaults reference Qwen 3.5 and Gemma 4)
- Judge GGUFs:
  - [RewardAnything-8B-v1](https://huggingface.co/) (Q6_K)
  - [Flow-Judge-v0.1](https://huggingface.co/flowaicom/Flow-Judge-v0.1) (Q6_K)

## Usage

### Quick run

```bash
# All models — starts servers, evaluates, generates report, cleans up
./scripts/run_nightly.sh

# Single model
NITE_MODELS="qwen3.5-9b" ./scripts/run_nightly.sh

# Filter to one dimension
NITE_DIMENSION="agentic" ./scripts/run_nightly.sh

# Combine
NITE_MODELS="qwen3.5-27b qwen3.5-9b" NITE_DIMENSION="agentic" ./scripts/run_nightly.sh
```

The launcher starts target llama-swap and both judge servers, runs the orchestrator, and tears everything down on exit. Already-running servers are reused. Ctrl-C saves progress (resumable).

### Overnight (unattended)

```bash
nohup ./scripts/run_nightly.sh > results/nightly.log 2>&1 &
```

### Resume after interruption

```bash
uv run python -m nite_eval.orchestrator --resume run-20260405-232559
```

### Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `NITE_MODELS` | all from config | Space-separated model list |
| `NITE_DIMENSION` | all | Filter to one dimension |
| `NITE_CONFIG` | `config/eval_config.yaml` | Config path |
| `NITE_TARGET_GPU` | from `.env` or `1` | GPU index for target llama-swap |
| `NITE_JUDGE_GPU` | from `.env` or `2` | GPU index for judge servers |

Path/binary configuration lives in `.env` (see `.env.example`).

## Default target models

| Name | Model | Quant | Notes |
|------|-------|-------|-------|
| `qwen3.5-27b` | Qwen 3.5 27B | Q4_K_M | |
| `qwen3.5-9b` | Qwen 3.5 9B | Q4_K_M | |
| `gemma4-26b-a4b` | Gemma 4 26B-A4B | Q4_K_M | Emits Harmony-style tool calls |
| `qwen3.6-35b-a3b` | Qwen 3.6 35B-A3B | UD-Q4_K_S | MoE reasoning; needs `system_suffix: "/no_think"` |

Add or replace models by editing `config/llama_swap_config.yaml` and the `models:` block in `config/eval_config.yaml`. The `models:` block accepts an optional `system_suffix` per model for chat-template triggers like `/no_think`.

## Judges

Two judges with complementary biases, routed by scoring dimension:

| Judge | Params | Dimensions | Bias |
|-------|--------|------------|------|
| Flow-Judge (`:9092`) | 3.8B | `reasoning_quality`, `practical_output` | 5-bias (recognizes excellence) |
| RewardAnything (`:9091`) | 8B | everything else | 3-bias (conservative, accurate on average responses) |

Calibration: neither judge alone reaches Cohen's kappa > 0.6, but dimension routing exploits their complementary error profiles.

## Output

Results live in `results/runs/`:

- `eval_results.db` — SQLite with all results, scores, tool calls
- `run-YYYYMMDD-HHMMSS.md` — Markdown comparison report

## Development

```bash
uv run ruff check --fix . && uv run ruff format .   # lint
uv run pyright                                        # type check
uv run python -m pytest -v                            # tests
```

## Project structure

```
config/
  eval_config.yaml                    # models, judge URLs, scoring weights
  llama_swap_config.example.yaml      # template — copy to *.yaml and edit
  judge_swap_config.example.yaml      # template (legacy; direct llama-server is the default)
tasks/
  research/  planning/  coding/  agentic/
src/nite_eval/
  orchestrator.py            # main pipeline
  conversation_runner.py     # multi-turn agent loop with Hermes tool execution
  judge.py                   # JudgeClient + RoutedJudgeClient
  scoring.py                 # deterministic + judge-based scoring
  task_loader.py  results_db.py  report.py
  hermes_parser.py  mock_tools.py  rubrics.py
  model_manager.py  ast_comparator.py
scripts/
  run_nightly.sh             # unattended runner
  smoke_test.py              # quick pipeline check
  validate_judge_pipeline.py # judge sanity check with synthetic responses
  run_calibration.py         # judge calibration against human scores
```

## License

[MIT](LICENSE)
