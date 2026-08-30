# nite-eval

Autonomous overnight LLM evaluation pipeline for local GGUF models. Runs multi-turn agentic tasks with the Hermes tool-calling format, scores them with dimension-routed judge models, and produces comparison reports.

Built for dual-GPU rigs running [llama.cpp](https://github.com/ggerganov/llama.cpp) + [llama-swap](https://github.com/mostlygeek/llama-swap), but the orchestrator is just an OpenAI-compatible HTTP client — any backend that speaks `/v1/chat/completions` will work.

## What it does

Evaluates local models across 4 dimensions (15 tasks total):

- **Research** (3 tasks) — multi-step information gathering and synthesis
- **Planning** (3 tasks) — task decomposition, dependency ordering, risk assessment
- **Coding** (4 tasks) — code generation with iterative tool use
- **Agentic** (5 tasks) — multi-turn tool calling, error recovery, context maintenance

Each task runs as a multi-turn conversation. Research, planning and agentic tasks use mock tools with responses defined in YAML; coding tasks run in a sandboxed container and are scored by hidden test suites that actually execute the model's code. Scoring mixes deterministic methods (tool-call matching, ordering, absence), judge rubrics on a 1–5 scale, and automated checks that run tests, `go vet` and the race detector. Results persist to SQLite with checkpoint/resume.

## Sample results

Run `run-20260418-234519` on the reference hardware, 5 models × 15 tasks, `max_tokens=4096`, qwen3.6 with `/no_think`:

> **Superseded as of 2026-08-30 — do not cite these numbers.** The harness was
> scoring failures as answers: truncated generations were judged as complete,
> `automated` criteria were hardcoded to `0.0`, `deterministic` criteria returned
> a free `1.0`, checklists matched on single keywords, and the judge prompt
> capped scores at 1/3/5. A six-model re-baseline on the current harness is
> pending; until then this table records what the old harness produced, not what
> the models do.
>
> **Also stale as of 2026-08-21.** These numbers were produced on llama.cpp build
> 8642 (`7c7d6ce5c`, 2026-04-03). That build cannot load `qwen3.8-27b`, so
> llama.cpp was updated to `cd26896c1` (2026-08-20) and a full re-baseline of
> all 6 models is pending. Scores below are not directly comparable to anything
> produced on the newer binary — sampler defaults, chat-template handling, and
> CUDA kernels all changed across those 4.5 months. The `docs/comparisons/`
> analysis rests on this same baseline.

| Model | Research | Planning | Coding | Agentic | Composite |
|-------|---------:|---------:|-------:|--------:|----------:|
| **qwen3.6-35b-a3b** (UD-Q4_K_S) | 0.82 | 0.90 | 0.28 | 0.78 | **0.70** |
| qwen3.6-35b-a3b-strix (Q4_K_M) | 0.77 | 0.77 | 0.21 | 0.77 | 0.63 |
| qwen3.5-27b | 0.75 | 0.69 | 0.21 | 0.82 | 0.62 |
| qwen3.5-9b | 0.75 | 0.69 | 0.28 | 0.74 | 0.62 |
| gemma4-26b-a4b | 0.68 | 0.69 | 0.28 | 0.67 | 0.58 |

Notes:
- Qwen models use the standard Hermes tool-call format. Gemma 4 emits tool calls in a Harmony-style format (`<|tool_call>call:FUNC{…}<tool_call|>`); the parser handles both.
- Reasoning-mode models (Qwen 3.6 MoE) need `/no_think` appended to the system prompt — without it, the model consumes the entire token budget inside `<think>…</think>` before producing an answer. Configure per-model via the `system_suffix` field in `config/eval_config.yaml`.
- Coding scores cluster at 0.15–0.42 across all models. **This was a harness defect, not a rubric ceiling and not a per-model weakness:** `automated` criteria returned a hardcoded `0.0` at 40–70% of each coding task's weight, and the mock `run_tests` reported success before the model had written anything. Both are fixed; on the current harness the same tasks score 0.85–1.00 with every criterion measured.
- The two Qwen3.6 entries above use the same base model with different quants (unsloth UD-Q4_K_S vs Sero/Strix Q4_K_M). Wikitext-2 perplexity is statistically identical (5.91 vs 5.92, ±0.04) — see [`docs/comparisons/qwen3-family-2026-04-19.md`](docs/comparisons/qwen3-family-2026-04-19.md) for the full investigation (PPL, GGUF metadata diff, chat-template diff, and verdict on what actually drives the eval-score gap).

## Hardware (reference setup)

| GPU | Role | Port |
|-----|------|------|
| RTX 3090 (24GB) | Target models via llama-swap | `:9070` |
| RTX 3060 (12GB) | Judge models (both fit simultaneously) | `:9091`, `:9092` |
| Tesla P40 (24GB) | *unused during evals* | — |

Judges moved from the P40 to the 3060 on 2026-08-21 so the P40 sits out entirely
— its throughput is far below the 3090's and it has no usable fp16 path, so
excluding it keeps run timings comparable. Both judges share the 3060: 6.3GB
(reward) + 3.0GB (flow) = 9.3GB of weights plus ~1.5GB KV/compute at `ctx 4096`,
so roughly 10.8GB of 11.8GB usable. It fits, but if a judge OOMs, lower its
`--ctx-size` in `scripts/run_nightly.sh`, or set `REWARD_GPU_UUID` / `FLOW_GPU_UUID`
in `.env` to split the two judges across cards — the P40 sits idle during evals.

**Stop Ollama before a run** if it is configured for all GPUs — it squats on the
3060 where the judges live, and a larger model loading mid-run will OOM them:

```bash
sudo systemctl stop ollama     # note: breaks bge-m3 embeddings for dependents
# ... run the eval ...
sudo systemctl start ollama
```

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

### Comparing two quants of the same model

When eval scores diverge between two GGUF quants (e.g. unsloth UD-Q4_K_S vs vanilla Q4_K_M of the same base), `compare_quants.sh` decomposes the gap into structural vs behavioral causes:

```bash
# Full comparison (~10 min): metadata diff + xxh64 + determinism + wikitext-2 perplexity
./scripts/compare_quants.sh

# Compare an arbitrary pair
./scripts/compare_quants.sh /path/to/A.gguf label_a /path/to/B.gguf label_b

# Skip the slow steps
./scripts/compare_quants.sh --skip-perplexity              # metadata + determinism only
./scripts/compare_quants.sh --skip-determinism             # metadata + perplexity only
./scripts/compare_quants.sh --system-suffix ""             # for non-thinking models
```

Outputs land in `results/quant-compare/<timestamp>/`. The metadata diff alone (Step 1) usually reveals the headline difference — different tensor-quant placements, missing imatrix calibration, divergent chat templates — before you spend time on perplexity. See the [Qwen 3.x family comparison](docs/comparisons/qwen3-family-2026-04-19.md) for a worked example.

`scripts/gguf_meta_diff.py` is callable standalone if all you want is the metadata diff:

```bash
uv run --with gguf python scripts/gguf_meta_diff.py A.gguf B.gguf --labels A B
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
| `qwen3.8-27b` | Qwen 3.8 27B | UD-Q4_K_XL | Hybrid attention+SSM (336 SSM tensors, 48 of 65 blocks). **Requires llama.cpp ≥ Aug 2026** — older builds fail with `missing tensor 'blk.64.ssm_conv1d.weight'` because they assume the final layer is an SSM block. **Needs `chat_template_kwargs: {reasoning_effort: medium}`** — its chat template has no `/no_think` branch, so that string is inert filler; the template defaults to `xhigh`. Without the override, on long prompts it exhausts the whole `max_tokens` budget inside `reasoning_content` and returns `finish_reason=length` with empty `content`. Observed on all 4 coding tasks in `run-20260829-040649` (11k–16k chars of reasoning, no answer). A short prompt returns `stop` normally, so this does not reproduce on a quick smoke test. Emits some tool calls as `{"function": ...}` instead of the Hermes `{"name": ...}`; the parser accepts both |

Add or replace models by editing `config/llama_swap_config.yaml` and the `models:` block in `config/eval_config.yaml`. The `models:` block accepts an optional `system_suffix` per model for chat-template triggers like `/no_think`.

## Coding tasks run for real

Coding tasks execute in a container built from `Dockerfile.sandbox-{go,python,deno}`:
no network, non-root, read-only root filesystem with a tmpfs workspace, capped
memory/CPU/PIDs, hard timeouts, and removal on exit. Files are streamed in over
stdin rather than bind-mounted, so nothing the model writes can reach a host path.

`test_pass_rate` is decided by a hidden suite under `tasks/coding/suites/<task_id>/`,
installed after the conversation ends. The model never sees it, and its own tests
are moved aside before scoring — the tasks ask the model to write tests, so
scoring those would let it grade itself. It still sees its own tests when it calls
`run_tests` during the conversation, which is the point of a real environment.

This requires each coding task to state an API contract in its prompt, since a
hidden suite has to compile against something. That trades API-design freedom for
measurability.

## Judges

Two judges with complementary biases, routed by scoring dimension:

| Judge | Params | Dimensions | Bias |
|-------|--------|------------|------|
| Flow-Judge (`:9092`) | 3.8B | `reasoning_quality`, `practical_output` | 5-bias (recognizes excellence) |
| RewardAnything (`:9091`) | 8B | everything else | 3-bias (conservative, accurate on average responses) |

Judges score on a 1–5 scale, averaged over three samples per criterion
(`evaluation.judge_averaging`). Averaging is worth its cost because the target
is essentially deterministic at `temperature: 0` — the same task reproduces to
within 0.1% of its latency and to identical scores — so run-to-run variance is
the judge's, and repeating target runs would buy nothing.

Until 2026-08-30 the judge prompt instructed *"Most responses deserve a 3... You
MUST pick exactly 1, 3, or 5"*, and 1579 of 2149 historical scores duly landed on
3.0 with 4.0 appearing twice. Nothing in the code enforced that, so scores from
before that date carry a quantization the current prompt does not.

Calibration: neither judge alone reaches Cohen's kappa > 0.6, but dimension routing exploits their complementary error profiles.

## Output

Results live in `results/runs/`:

- `eval_results.db` — SQLite with all results, scores, tool calls
- `run-YYYYMMDD-HHMMSS.md` — Markdown comparison report

### Failed measurements are visible, not scored

A task that could not be measured is recorded as an error with score 0 rather
than being scored on whatever partial text the model happened to emit. The
harness distinguishes:

| `error` prefix | Meaning |
|---|---|
| `truncated:` | Generation hit `max_tokens` with real content. Raise the task's `max_tokens`. |
| `degenerate_repetition:` | The model looped on a short substring until cut off. Raising `max_tokens` only lengthens the loop. |
| `unparsed_tool_call:` | Tool-call JSON did not parse after one corrective retry. Raw payload attached. |
| `task_timeout:` | Task exceeded its `timeout_seconds` wall-clock budget. |
| `synthesis nudge failed` | The final-answer request errored, so there is no answer to score. |

Previously all four were silent: the fragment became the "final answer" and a
judge scored it. Query them with `SELECT error FROM task_results WHERE error IS
NOT NULL`.

### Unmeasurable criteria are excluded, not faked

A scoring criterion with no implementation is dropped from the weighted average
rather than scored 0 or 1. `task_results.unscored_weight` records the fraction
of a task's declared weight that was excluded, and reports carry an `Unscored`
column plus a "Partially Scored Dimensions" section.

This matters when reading a score: 0.86 over 35% of a task's criteria is a
narrower claim than 0.86 over all of them, not a better result. Coding tasks
currently exclude 40-70% of their weight pending real test execution.

### Malformed tool calls are repaired and counted

Some models emit structurally broken tool-call JSON. Where the defect is
characterized and safely repairable, the parser fixes it rather than discarding
the call, and records the count in `task_results.repaired_tool_calls`. Reports
include a per-model repair rate. A high rate is a model-quality signal — see
`CLAUDE.md` for the qwen3.8 case (34% of coding tool calls).

## K8s deployment

For unattended operation on a single-node k3s homelab, the same pipeline runs as a CronJob. Three container images (`nite-eval-orchestrator`, `nite-eval-judge`, `nite-eval-target`) plus manifests in [`k8s/base/`](k8s/base/) reproduce the bare-metal layout: judges as long-running Deployments on the P40, target llama-swap as a Deployment on the 3090, orchestrator as a nightly Job. SQLite checkpoints persist on a PVC so a mid-run pod kill resumes cleanly.

See [`k8s/README.md`](k8s/README.md) for build, push, GPU-UUID config, apply, and ops procedures.

The bare-metal `scripts/run_nightly.sh` flow above remains the supported fallback path.

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
  sandbox.py        automated_scoring.py  gpu_check.py
  model_manager.py  ast_comparator.py
scripts/
  run_nightly.sh             # unattended runner
  smoke_test.py              # quick pipeline check
  validate_judge_pipeline.py # judge sanity check with synthetic responses
  run_calibration.py         # judge calibration against human scores
  compare_quants.sh          # compare two GGUF quants: metadata, determinism, perplexity
  gguf_meta_diff.py          # standalone GGUF metadata + per-tensor quant diff
docs/comparisons/            # writeups from past comparison runs
```

## License

[MIT](LICENSE)
