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

Run `run-20260901-043322` on the reference hardware, 7 models × 15 tasks,
llama.cpp `cd26896c1`. The first sweep where every number means what it says:
failed tasks score 0 in their dimension rather than being dropped, coding runs
at `max_tokens: 32768`, the fixture gaps that returned errors to well-formed
calls are closed, container timestamps no longer reach the model, and any call
the mocks cannot answer is counted and reported.

| Model | Research | Planning | Coding | Agentic | Composite | Tasks |
|-------|---------:|---------:|-------:|--------:|----------:|------:|
| **qwen3.8-27b** | 0.87 | 0.80 | 0.94 | 0.85 | **0.86** | 15/15 |
| ornith-1.5-35b-a3b | 0.80 | 0.79 | 0.47 | 0.76 | **0.70** | 15/15 |
| qwen3.6-35b-a3b (UD-Q4_K_S) | 0.75 | 0.76 | 0.28 | 0.75 | **0.63** | 13/15 |
| qwen3.5-9b | 0.77 | 0.73 | 0.15 | 0.71 | **0.59** | 12/15 |
| qwen3.6-35b-a3b-strix (Q4_K_M) | 0.80 | 0.49 | 0.28 | 0.77 | **0.59** | 12/15 |
| gemma4-26b-a4b | 0.68 | 0.68 | 0.17 | 0.81 | **0.58** | 13/15 |
| qwen3.5-27b | 0.86 | 0.70 | 0.04 | 0.72 | **0.58** | 12/15 |

### Reading this

**qwen3.8-27b wins outright** — first in all four dimensions, 0.16 clear of
second, and one of only two models to complete every task. Its coding at 0.94
against a field where nobody else passes 0.47 is the largest single gap here.

**The bottom four are a four-way tie.** 0.58, 0.58, 0.59, 0.59 — every adjacent
pair inside `scoring.min_detectable_difference`, and the report says so per run.
Ranking them is reading noise.

**Coding is what separates the field, and it is mostly a completion problem.**
13 of the 15 task failures are coding truncations. `coding_artemis_medium_01`
failed for five of seven models and `coding_mcp_hard_01` for four; qwen3.8 and
ornith are the only models that finish all four coding tasks. A low coding score
here usually means the model did not get to the end, not that the code was bad —
and coding carries its own 0.15 threshold, so gaps below that carry no
information at all.

**strix's 0.49 planning is one failed task, not a quant difference.** It scores
0.76 and 0.71 on the two planning tasks it completes and truncates
`planning_wine_easy_01`, which every other model passes at 0.75-0.78. The two
Qwen3.6 quants are otherwise equal — identical coding (0.28), close agentic
(0.77 vs 0.75), and statistically identical wikitext-2 perplexity. Do not read
the 0.59 vs 0.63 composite as a quant-quality signal.

**Latency does not track quality, and the spread is 8x.** qwen3.5-9b averages
18.6s per task and scores 0.59; qwen3.8-27b takes 145.8s for its 0.86; ornith
reaches 0.70 at 37.5s, four times faster than the winner.

Notes:

- One unanswered tool call across all 105 tasks, and it is a model defect rather than a fixture gap: gemma4 emitted `call_mcp_tool` with `server` nested one level too deep, which no mock can match. See "Unanswered Tool Calls" in the report.
- Repairs are effectively gone: 1 salvaged JSON element across the whole sweep, for qwen3.8. ornith needs none because it runs on the native tool-call path.
- `unscored_weight` is 0% everywhere: every declared criterion was measured.
- Each model has a different shape, which the composite flattens. gemma4 leads agentic at 0.81 while trailing everywhere else; qwen3.5-27b is second in research at 0.86 with the worst coding at 0.04.
- Qwen models use the standard Hermes tool-call format. Gemma 4 emits Harmony-style calls (`<|tool_call>call:FUNC{…}<tool_call|>`); the parser handles both. ornith uses the native `tools` API — see "Native tool calling is per-model".
- Thinking is configured per model and by measurement, not convention: Qwen 3.6 takes `/no_think`; qwen3.8 keeps `reasoning_effort: medium` because turning thinking off cost it research 0.80 → 0.63; ornith runs with `enable_thinking: false` — see below.

### ornith-1.5: coding, and why thinking is off

ornith is the one model where the thinking switch was decided by measurement
rather than convention, and the two questions are entangled.

Its template's only reasoning control is `enable_thinking` — no `/no_think`
branch, no `reasoning_effort` middle setting, so the choice is binary. Measured
both ways over 15 tasks on the native tool-call path:

| dimension | thinking on | thinking off |
|---|---:|---:|
| research | 0.80 | 0.81 |
| planning | 0.80 | 0.79 |
| coding | 0.22 | **0.49** |
| agentic | 0.80 | 0.72 |
| composite | 0.65 | **0.70** |

Coding has since read 0.33, 0.39 and 0.47 across further runs of that same
thinking-off configuration, against the 0.49 above. Four samples spanning 0.16
is the reproducibility problem described under "Known limitations", and it is
why coding carries a 0.15 threshold rather than the composite's 0.05. The
thinking-on and thinking-off coding figures differ by more than that spread,
which is what makes the comparison usable; the individual numbers are not.

**With thinking on it does not finish.** Reasoning and the tool call compete for
one contiguous `max_tokens`, and reasoning wins: `coding_artemis_medium_01` spent
its entire 32768-token budget on turn 1 and emitted zero tool calls. Raising the
budget only bought more of it — 89k chars at 24576, 121k at 32768, same turn,
still no call.

**With thinking off it finishes and the code does not compile.** On
`coding_mcp_easy_01` a judge scored the code 0.83 while the hidden suite failed
to build on `undefined: Load`, a function the task's contract requires. It writes
fluent, well-structured code that misses the interface.

So thinking off is the better setting — 15/15 tasks against 12/15, and 3-5x
faster on several tasks — but it trades a truncation failure for a correctness
one rather than fixing coding. That the reasoning is doing real work is
measurable: `coding_mcp_easy_01` is the one coding task ornith completed with
thinking on, and it scored 0.89 there against 0.25 with thinking off.

Giving reasoning its own turn was tried and rejected. A turn cut off mid-thought
was continued rather than failing the task; ornith then reasoned to exhaustion
four times in a row — 120871, 124269 and 117697 chars — after being told
explicitly not to restate its reasoning. It does not have a clipped thought, it
has a non-converging one. The implementation and its tests are preserved on the
`worktree-ornith-reasoning-continuation` branch, unmerged.

Note also that this is **not** qwen3.8's finding. Thinking off cost qwen3.8
research 0.80 → 0.63, which is why it keeps `reasoning_effort: medium`. ornith's
research did not move. Neither result generalises to the other model.

<details>
<summary>Superseded: <code>run-20260830-231628</code> (6 models, pre-2026-08-31 scoring)</summary>

**Not comparable to the table above.** Two changes on 2026-08-31 altered what
these numbers mean, for every model: a failed task now scores 0 in its dimension
rather than being excluded from the average, and coding `max_tokens` went from
24576 to 32768, and three fixture gaps that returned errors to well-formed tool
calls were closed. All seven models have since been re-run on the current
harness, so this is kept for provenance only — the table above supersedes it
entirely.

| Model | Research | Planning | Coding | Agentic | Composite | Tasks |
|-------|---------:|---------:|-------:|--------:|----------:|------:|
| **qwen3.8-27b** | 0.85 | 0.79 | 0.93 | 0.84 | **0.85** | 15/15 |
| qwen3.6-35b-a3b-strix (Q4_K_M) | 0.76 | 0.74 | 0.57 | 0.78 | 0.71 | 12/15 |
| qwen3.5-9b | 0.77 | 0.74 | 0.58 | 0.72 | 0.70 | 12/15 |
| qwen3.5-27b | 0.84 | 0.72 | 0.43 | 0.77 | 0.69 | 13/15 |
| qwen3.6-35b-a3b (UD-Q4_K_S) | 0.73 | 0.75 | 0.54 | 0.73 | 0.69 | 13/15 |
| gemma4-26b-a4b | 0.67 | 0.71 | 0.34 | 0.80 | 0.63 | 13/15 |

12 of 90 task runs failed: 11 truncations (10 coding, plus `planning_wine_easy_01`
for strix alone) and one unrepairable malformed tool call (gemma4 on
`coding_mcp_hard_01`). Because failed tasks were excluded from their dimension
average rather than scored 0, the dimension figures here flatter every model that
failed a task — which is the defect the 2026-08-31 change fixed.

</details>

<details>
<summary>Superseded baseline: <code>run-20260418-234519</code> (5 models, April harness)</summary>

**Do not cite these numbers.** Kept for provenance only. The harness was scoring
failures as answers: truncated generations were judged as complete, `automated`
criteria were hardcoded to `0.0`, `deterministic` criteria returned a free `1.0`,
checklists matched on single keywords, and the judge prompt capped scores at
1/3/5. Produced on llama.cpp build 8642 (`7c7d6ce5c`, 2026-04-03), which cannot
load `qwen3.8-27b`; sampler defaults, chat-template handling, and CUDA kernels
all changed before the current baseline.

| Model | Research | Planning | Coding | Agentic | Composite |
|-------|---------:|---------:|-------:|--------:|----------:|
| **qwen3.6-35b-a3b** (UD-Q4_K_S) | 0.82 | 0.90 | 0.28 | 0.78 | **0.70** |
| qwen3.6-35b-a3b-strix (Q4_K_M) | 0.77 | 0.77 | 0.21 | 0.77 | 0.63 |
| qwen3.5-27b | 0.75 | 0.69 | 0.21 | 0.82 | 0.62 |
| qwen3.5-9b | 0.75 | 0.69 | 0.28 | 0.74 | 0.62 |
| gemma4-26b-a4b | 0.68 | 0.69 | 0.28 | 0.67 | 0.58 |

Coding clustered at 0.15-0.42 for every model there. That was a harness defect,
not a rubric ceiling: `automated` criteria returned a hardcoded `0.0` at 40-70%
of each coding task's weight, and the mock `run_tests` reported success before
the model had written anything. Both are fixed.

</details>

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

Specs read from the GGUF headers on the reference host, not from model cards.
Parameter counts are summed over actual tensor elements, so they include the
embedding matrices and will differ slightly from the marketing number.

| Name | Arch | Params | Layers | Experts (active) | Quant | GGUF | VRAM @ 64k |
|------|------|-------:|-------:|------------------|-------|-----:|-----------:|
| `qwen3.5-27b` | `qwen35` dense | 26.90B | 64 | — | Q4_K_M | 15.4 GiB | not measured |
| `qwen3.5-9b` | `qwen35` dense | 8.95B | 32 | — | Q4_K_M | 5.3 GiB | not measured |
| `gemma4-26b-a4b` | `gemma4` MoE | 25.23B | 30 | 128 (8) | Q4_K_M | 15.6 GiB | 17853 MiB |
| `qwen3.6-35b-a3b` | `qwen35moe` MoE | 34.66B | 40 | 256 (8) | UD-Q4_K_S | 19.5 GiB | not measured |
| `qwen3.6-35b-a3b-strix` | `qwen35moe` MoE | 34.66B | 40 | 256 (8) | Q4_K_M | 19.7 GiB | 21393 MiB |
| `qwen3.8-27b` | `qwen35` hybrid | 27.32B | 65 | — | UD-Q4_K_XL | 16.4 GiB | 19211 MiB |
| `ornith-1.5-35b-a3b` | `qwen35moe` MoE | 35.51B | 41 | 256 (8) | Q4_K_M | 20.2 GiB | 21412 MiB |

Attention geometry, which is what determines how fast KV cache grows with context:

| Name | Q heads | KV heads | Key length | Embedding | Vocab | Trained ctx | Tensors |
|------|--------:|----------|-----------:|----------:|------:|------------:|--------:|
| `qwen3.5-27b` | 24 | 4 | 256 | 5120 | 248320 | 262144 | 851 |
| `qwen3.5-9b` | 16 | 4 | 256 | 4096 | 248320 | 262144 | 427 |
| `gemma4-26b-a4b` | 16 | 8, but 2 on every 6th layer | 512 | 2816 | 262144 | 262144 | 658 |
| `qwen3.6-35b-a3b` | 16 | 2 | 256 | 2048 | 248320 | 262144 | 733 |
| `qwen3.6-35b-a3b-strix` | 16 | 2 | 256 | 2048 | 248320 | 262144 | 733 |
| `qwen3.8-27b` | 24 | 4 | 256 | 5120 | 248320 | 262144 | 866 |
| `ornith-1.5-35b-a3b` | 16 | 2 | 256 | 2048 | 248320 | 262144 | 753 |

Every Qwen-derived model here shares the same 248320 vocabulary, including
ornith; only gemma4 differs at 262144. So a parameter-count difference between
two of them is architecture, not tokenizer — ornith's extra 0.85B over
qwen3.6 is its multi-token-prediction block, not a larger vocab.

All seven run under llama-swap with identical flags — `-ngl 999 --ctx-size 65536
-fa on --cache-type-k q8_0 --cache-type-v q8_0` — pinned to the 3090 by UUID and
in an `exclusive: true` group so only one is resident at a time. **Context is
65536, not the 262144 the models were trained for**; see the VRAM measurements in
`CLAUDE.md` before raising it. `ornith-1.5-35b-a3b` is now the binding model at
21412 MiB, a hair above `qwen3.6-35b-a3b-strix` at 21393, leaving ~3.1 GB
headroom on the 24 GB card.

Per-model notes:

- **`gemma4-26b-a4b`** emits tool calls in a Harmony-style format
  (`<|tool_call>call:FUNC{…}<tool_call|>`) rather than Hermes; the parser handles
  both. Its `key_length` of 512 is twice every other model's, so on paper its KV
  cache should be the largest here — in practice llama.cpp gives it
  sliding-window attention (the `2` KV heads on every 6th layer above), and
  doubling context cost it only ~500 MiB.
- **`qwen3.6-35b-a3b` and `-strix`** are the same base model at different quants
  (unsloth UD-Q4_K_S vs Sero/Strix Q4_K_M) — byte-for-byte identical metadata,
  733 tensors each, differing only in quantization. Both are MoE reasoning models
  and need `system_suffix: "/no_think"`, or they burn the whole token budget
  inside `<think>…</think>`.
- **`qwen3.8-27b`** reports `general.architecture = qwen35` but is not a plain
  dense Qwen: it is a hybrid, with 336 SSM tensors across 48 of its 65 blocks and
  conventional attention in only 17. **Requires llama.cpp ≥ Aug 2026** — older
  builds fail with `missing tensor 'blk.64.ssm_conv1d.weight'` because they assume
  the final layer is an SSM block. **Needs `chat_template_kwargs:
  {reasoning_effort: medium}`**: its chat template has no `/no_think` branch, so
  that string is inert filler and the template defaults to `xhigh`. Without the
  override it exhausts the whole `max_tokens` budget inside `reasoning_content`
  on long prompts and returns `finish_reason=length` with empty `content` —
  observed on all 4 coding tasks in `run-20260829-040649` (11k–16k chars of
  reasoning, no answer). A short prompt returns `stop` normally, so this does not
  reproduce on a quick smoke test. It also emits some tool calls as
  `{"function": ...}` instead of the Hermes `{"name": ...}`; the parser accepts both.

- **`ornith-1.5-35b-a3b`** (`ornith-ai/Ornith-1.5-35B-A3B-GGUF`, Q4_K_M) is
  structurally close to `qwen3.6-35b-a3b` — same `qwen35moe` arch, 256 experts
  with 8 active, 16/2 heads — but it is a hybrid: of its 40 transformer layers
  only 10 carry full attention (blocks 3, 7, … 39, every 4th) and the other 30
  are linear attention. `block_count` reads 41 because the GGUF also carries a
  multi-token-prediction block (`blk.40.nextn.*`); whether llama.cpp uses it is
  unverified. That extra block accounts for the parameter difference against
  qwen3.6 (35.51B vs 34.66B); the two share an identical 248320 vocabulary, so
  it is not that. It runs with `enable_thinking: false`
  and is the only model with `native_tools: true` — see "Native tool calling"
  below. The two go together: thinking off used to wreck its hand-written
  tool-call JSON (1/8 parseable against 8/8), which no longer matters once the
  server produces the call. Measured both ways on 15 tasks, thinking off is
  +0.05 composite and 15/15 tasks against 12/15, almost all of it coding
  (0.22 -> 0.49) against a smaller agentic loss (0.80 -> 0.72). This is not
  qwen3.8's finding, where thinking off cost research 0.80 -> 0.63; Ornith's
  research did not move. It
  needs `chat_template_kwargs: {enable_thinking: false}`; its template has no
  `/no_think` branch and no `reasoning_effort`, so the reasoning switch is
  binary. **Not yet validated on this harness:** at 20.2 GiB it is the largest
  target here and its VRAM at 64k is unmeasured, and it was post-trained on an
  XML tool-call format (`<function=NAME><parameter=KEY>…`) that
  `hermes_parser` cannot parse. nite-eval injects tool definitions as prompt
  text rather than via a `tools` field, so the model is instructed in Hermes
  JSON and that template branch never fires — whether it complies is untested.

Add or replace models by editing `config/llama_swap_config.yaml` and the `models:`
block in `config/eval_config.yaml`. The `models:` block accepts an optional
`system_suffix` per model for chat-template triggers like `/no_think`, and
`chat_template_kwargs` for templates that take structured options.

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
narrower claim than 0.86 over all of them, not a better result. As of
`run-20260830-231628` all 15 tasks score at 0% unscored weight, coding included,
so every criterion is measured. A task added without a hidden suite will sit
below 100% without failing loudly — check the column before quoting a number.

### Malformed tool calls are repaired and counted

Some models emit structurally broken tool-call JSON. Where the defect is
characterized and safely repairable, the parser fixes it rather than discarding
the call, and records the count in `task_results.repaired_tool_calls`. Reports
include a per-model repair rate. A high rate is a model-quality signal — see
`CLAUDE.md` for the qwen3.8 case (34% of coding tool calls).

Two repairs exist, for two one-character defects in different positions:

| Model | Defect | Repair |
|---|---|---|
| qwen3.8-27b | drops the opening quote of the **key** after the name — `{"name": "write_file",\narguments":` | `_repair_dropped_key_quote` |
| ornith-1.5-35b-a3b | drops the opening quote of the name's **value** — `{"name": search_thoughts",` | `_repair_dropped_name_value_quote` |

They run value-first: a dropped value quote leaves an unbalanced quote that
desynchronises the key repair's string tracking, which then rewrites
`"arguments"` to `""arguments"`.

### Native tool calling is per-model

`config/eval_config.yaml` accepts `native_tools: true` per model. With it, the
tool schemas go in the request and the server's structured `message.tool_calls`
are read back; the model's own chat template renders its tool instructions, so
the harness does not inject its own. Without it — the default — tool definitions
are pasted into the system prompt and the reply is parsed out of the text by
`hermes_parser`.

Only `ornith-1.5-35b-a3b` uses it, because that is the usage its model card
documents. Asked for tool calls in prose it emitted a different broken shape
nearly every turn: a dropped quote on the name value, `<function":` where
`{"function":` belongs, `<tool_call>` batches with one closing tag, payloads
opened in JSON and closed in XML, closing tags used as openers. Three rounds of
repairs took it from 8/15 to 12/15 tasks and still needed 52 repairs across 26
calls on one task. On the native path it needs **none** — repairs went to zero
across all 15 tasks — and `agentic_wine_medium_01` rose 0.56 to 0.75 because the
model finally receives the argument schema as a schema rather than as prose.

The flag is off for the other six deliberately. Switching a model changes what
it is asked to do, so its scores move and stop being comparable to earlier runs.

Sending the schemas is only half of it: the reply has to go back as an assistant
message carrying `tool_calls`, with each result as a `tool` message naming its
`tool_call_id`. Without that the model sees itself say nothing and tool results
arriving unprompted — `agentic_mcp_hard_01` made one call, stopped, and scored
0.40 against 0.93.

## Known limitations

Deliberate trade-offs, not bugs. Each is a thing a number from this harness does
not tell you.

**Coding tasks state an API contract.** A hidden test suite has to compile
against something, so each coding prompt specifies module, package, types and
signatures. That removes API-design freedom: a model is measured on implementing
a stated interface, not on choosing a good one. Standard for benchmarks of this
shape, but it makes the coding dimension easier than the prose of the task alone
suggests.

**Conversation history is edited.** Tool calls over 1500 characters are replaced
in history by a note saying what was called. Without it, a task writing sixteen
files exhausts the context — each file body otherwise appears twice, as the turn
that emitted it and again as history. The consequence is that a model wanting to
re-read its own earlier output must call `read_file`; it cannot scroll back. That
is closer to how agent harnesses actually behave, but it is a behavioural
difference between this harness and a plain chat loop.

**Nothing before 2026-08-30 is comparable.** The harness scored failures as
answers: truncated generations judged as complete, `automated` criteria hardcoded
to `0.0`, `deterministic` criteria returning a free `1.0`, checklists matching on
single keywords, a judge prompt capped at 1/3/5. Old runs remain in the database
for provenance. They are not a baseline.

**Coding is not reproducible, by construction.** Coding tasks run in a real
container and the container's output enters the conversation. An `ls -la` on
turn 1 returns the working directory's mtime, which is the container's creation
time, so two runs of the same model at `temperature: 0` see different history
from turn 2 and go on to write different code — ornith-1.5 emitted 65986, 66666
and 65604 bytes of tool-call arguments across three runs of one task. Where the
code it happened to write hit a real bug, both automated criteria scored 0
instead of 1.00 and 0.93, moving that task 0.62 to 0.00. Per-task swings of
0.1-0.6 and a dimension-level swing of 0.49 to 0.39 have been observed on
identical configuration, so `scoring.dimension_min_detectable_difference` sets
coding's threshold to 0.15 and reports print it. Timestamps are the cause found;
anything else varying per container — `find` ordering, hostnames, mtimes on
copied files — behaves the same way.

**One sample per task.** 15 tasks, each run once, judge scores averaged over
three. Composite gaps under 0.05 are inside judge variance; reports say so per
run. On mock-backed tasks the target is essentially deterministic at
`temperature: 0`, so repeat runs of the model buy nothing there — but see the
entry above, which makes that false for coding. More tasks or more judge samples
are what would
sharpen this.

**Per-task token budgets bound the coding scores.** Coding tasks cap generation
at `max_tokens: 32768` as of 2026-08-31, raised from 24576 (planning is at
`12288`). In `run-20260830-231628` five of
six models hit `finish_reason=length` on at least one coding task and scored
`0.00` there, which is most of the spread in that dimension. A low coding score
means "did not finish inside the budget" at least as often as it means "wrote bad
code", and raising the budget changes the numbers. Check the failure reasons
before attributing a coding gap to model quality.

**The scheduled path is unverified.** The k8s manifests account for GPU checks
and sandbox availability, but have not been deployed or run.

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
