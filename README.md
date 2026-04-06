# nite-eval

Autonomous overnight LLM evaluation pipeline for local models. Runs multi-turn agentic tasks with Hermes tool-calling format, scores with dimension-routed judge models, and produces comparison reports.

## What it does

Evaluates local GGUF models across 4 dimensions (15 tasks total):
- **Research** (3 tasks) — multi-step information gathering and synthesis
- **Planning** (3 tasks) — task decomposition, dependency ordering, risk assessment
- **Coding** (4 tasks) — code generation with iterative tool use
- **Agentic** (5 tasks) — multi-turn tool calling, error recovery, context maintenance

Each task runs as a multi-turn conversation with mock tools, scored by a mix of deterministic methods (checklist, sequence match, subset match) and LLM judges (rubric-based ternary scoring). Results persist to SQLite with checkpoint/resume support.

## Hardware

| GPU | Device | Role | Port |
|-----|--------|------|------|
| 1 | RTX 3090 (24GB) | Target models via llama-swap | :9070 |
| 2 | Tesla P40 (24GB) | Judge models (both run simultaneously) | :9091, :9092 |

## Usage

### Quick run (on-demand)

No prerequisites beyond having models downloaded. The script starts all servers and cleans them up on exit.

```bash
# Run all models — starts target + judges, evaluates, generates report, cleans up
./scripts/run_nightly.sh

# Single model
NITE_MODELS="qwen3.5-9b" ./scripts/run_nightly.sh

# Filter to one dimension
NITE_DIMENSION="agentic" ./scripts/run_nightly.sh

# Combine filters
NITE_MODELS="qwen3.5-27b qwen3.5-9b" NITE_DIMENSION="agentic" ./scripts/run_nightly.sh
```

The script starts target llama-swap (GPU 1) and both judge servers (GPU 2), runs the orchestrator, and cleans up all servers on exit. Skips starting any server already running. Ctrl-C saves progress (resumable).

### Overnight (unattended)

```bash
nohup ./scripts/run_nightly.sh > results/nightly.log 2>&1 &
```

### Manual (servers already running)

If you manage servers yourself:

```bash
# Start target llama-swap (GPU 1)
CUDA_VISIBLE_DEVICES=1 /home/woojay/T/llama-swap/llama-swap \
  --config config/llama_swap_config.yaml --listen :9070

# Start judges (GPU 2)
CUDA_VISIBLE_DEVICES=2 /home/woojay/P/llama.cpp/build/bin/llama-server \
  -m /home/woojay/models/RewardAnything-8B-v1.Q6_K.gguf \
  --port 9091 -ngl 999 --ctx-size 4096 -np 1 --no-webui &
CUDA_VISIBLE_DEVICES=2 /home/woojay/P/llama.cpp/build/bin/llama-server \
  -m /home/woojay/models/Flow-Judge-v0.1.Q6_K.gguf \
  --port 9092 -ngl 999 --ctx-size 4096 -np 1 --no-webui &

# Run orchestrator directly
uv run python -m nite_eval.orchestrator --models qwen3.5-27b qwen3.5-9b gemma4-26b-a4b
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
| `NITE_TARGET_GPU` | `1` | GPU ID for target llama-swap |
| `NITE_JUDGE_GPU` | `2` | GPU ID for judge servers |

## Target Models

| Name | Model | Quant |
|------|-------|-------|
| `qwen3.5-27b` | Qwen 3.5 27B | Q4_K_M |
| `qwen3.5-9b` | Qwen 3.5 9B | Q4_K_M |
| `gemma4-26b-a4b` | Gemma 4 26B-A4B | Q4_K_M |

## Judge Models

Two judges with complementary biases, routed by scoring dimension:

| Judge | Params | Dimensions | Bias |
|-------|--------|------------|------|
| Flow-Judge (`:9092`) | 3.8B | `reasoning_quality`, `practical_output` | 5-bias (recognizes excellence) |
| RewardAnything (`:9091`) | 8B | everything else | 3-bias (conservative, accurate on average responses) |

Calibration results: neither judge alone reaches Cohen's kappa > 0.6, but dimension-routing exploits their complementary error profiles.

## Output

Results are stored in `results/runs/`:
- `eval_results.db` — SQLite with all results, scores, tool calls
- `run-YYYYMMDD-HHMMSS.md` — Markdown comparison report

## Development

```bash
uv run ruff check --fix . && uv run ruff format .   # lint
uv run pyright                                        # type check
uv run python -m pytest -v                            # test (79 tests)
```

## Project Structure

```
config/
  eval_config.yaml           # models, judge URLs, scoring weights
  llama_swap_config.yaml     # target model definitions for llama-swap
  judge_swap_config.yaml     # judge model definitions (legacy, now direct)
tasks/
  research/                  # 3 research tasks (web_search, fetch_url)
  planning/                  # 3 planning tasks (check_dependency, estimate_effort)
  coding/                    # 4 coding tasks (write_file, run_tests)
  agentic/                   # 5 agentic tasks (multi-tool chains, error recovery)
src/nite_eval/
  orchestrator.py            # main pipeline: tasks → conversations → scoring → DB
  conversation_runner.py     # multi-turn agent loop with Hermes tool execution
  judge.py                   # JudgeClient + RoutedJudgeClient (dimension routing)
  scoring.py                 # deterministic + judge-based scoring methods
  task_loader.py             # load YAML task definitions with filtering
  results_db.py              # SQLite storage with checkpoint/resume
  report.py                  # Markdown report generator
  hermes_parser.py           # parse <tool_call> XML tags from model output
  mock_tools.py              # deterministic tool responses from YAML definitions
  rubrics.py                 # judge rubric definitions for all dimensions
  model_manager.py           # server lifecycle and health checks
  ast_comparator.py          # AST-based tool-call comparison
scripts/
  run_nightly.sh             # unattended eval runner (starts judges, runs pipeline)
  validate_judge_pipeline.py # end-to-end judge validation with synthetic responses
  smoke_test.py              # quick pipeline smoke test
  run_calibration.py         # judge calibration against human scores
```
