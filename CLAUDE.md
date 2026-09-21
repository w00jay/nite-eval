# CLAUDE.md — nite-eval

Project-local context for Claude Code. Gitignored.

## What this project does

Autonomous overnight LLM evaluation pipeline for local GGUF models served via llama.cpp + llama-swap. Multi-turn agentic tasks, dimension-routed judges (RewardAnything + Flow-Judge), SQLite persistence, markdown reports.

Read the README first — it covers usage, hardware, models, and the dimension/judge layout.

## Where things live

- `config/eval_config.yaml` — model list (with optional `system_suffix` per model), judge URLs, scoring weights. Public, committed.
- `config/llama_swap_config.yaml` — local llama-swap config with absolute GGUF paths and CUDA UUIDs. **Gitignored**; copy from `*.example.yaml`.
- **Model files and `.jinja` templates live in `~/models`, never in `llama.cpp/build/bin`.** Binaries stay in `build/bin` because a binary is a build artifact; models are not. Until 2026-09-19 the GGUFs sat inside the build tree, where a `cmake -B build` or `git clean` in the llama.cpp checkout would have destroyed 37 GB of them.
- `config/templates/` — chat templates extracted from GGUFs for inspection / overrides. Generated; safe to delete and regenerate.
- `tasks/<dimension>/*.yaml` — task definitions per dimension (research/planning/coding/agentic).
- `src/nite_eval/` — orchestrator, conversation runner, scoring, judges, mock tools, parsers.
- `scripts/` — runners + utilities. See "Scripts" below.
- `results/` — **gitignored.** Raw eval outputs, SQLite DB, comparison artifacts.
- `docs/comparisons/` — durable summaries of past comparison runs. Committed.
- `.env` — paths, GPU UUIDs, ports. **Gitignored**; copy from `.env.example`.

## Scripts

| Script | Purpose |
|---|---|
| `scripts/run_nightly.sh` | Main runner. Starts target llama-swap + both judges, runs orchestrator, generates report, cleans up on exit. Honors `NITE_MODELS`, `NITE_DIMENSION`, `NITE_CONFIG`. |
| `scripts/smoke_test.py` | Quick end-to-end pipeline check on a single inline task. Use to verify the loop works after changing scoring/runner code. |
| `scripts/validate_judge_pipeline.py` | Sends synthetic responses (good/bad/refusal) through the judge to sanity-check rubric routing. |
| `scripts/run_calibration.py` + `score_calibration.py` + `generate_calibration_set.py` | Judge calibration against human scores. |
| `scripts/compare_quants.sh` | Compare two GGUF quants of the same base model: metadata diff, xxh64 hashes, determinism check (with `--system-suffix` for thinking models, default `/no_think`), wikitext-2 perplexity. Args: `PATH_A LABEL_A PATH_B LABEL_B` or no args (defaults to the two Qwen3.6 quants). Outputs to `results/quant-compare/<timestamp>/`. |
| `scripts/gguf_meta_diff.py` | Standalone GGUF metadata comparator (general.* / tokenizer.* / per-tensor quant breakdown / chat template hash). Run via `uv run --with gguf python scripts/gguf_meta_diff.py A.gguf B.gguf --labels A B`. The `gguf` package isn't a runtime dep so it's pulled ad-hoc. |

## GPU placement — check this every time

**Target model and judges must be on separate GPUs.** This is not a performance
preference; a shared GPU silently corrupts a run. The judges hold ~11GB, the
target needs 18-21GB, and when they contend the target's layers spill to host
memory. The eval still completes and still writes scores — only the latency
numbers are quietly garbage, and a large model may fail to load outright.

Assignment on this host:

| GPU | Device | Role | UUID prefix |
|---|---|---|---|
| 0 | RTX 3060 12GB | both judges (reward-anything + flow-judge) | `GPU-144ccc5f` |
| 1 | RTX 3090 24GB | model under evaluation | `GPU-d5346770` |
| 2 | Tesla P40 24GB | intentionally unused | `GPU-219f27f6` |

Where placement is actually decided (three places, all must agree):

1. `.env` — `TARGET_GPU_UUID` / `JUDGE_GPU_UUID`. Consumed by `run_nightly.sh`
   for the judge servers. **Source of truth.**
2. `config/llama_swap_config.yaml` — per-model `CUDA_VISIBLE_DEVICES=<uuid>`.
   Pins the target. Gitignored, so also update `*.example.yaml`.
3. `config/eval_config.yaml` `hardware:` block — **documentation only**, no code
   reads it. Keep it accurate but never rely on it.

Use UUIDs, never indices. Index order is not stable across driver reloads;
`CUDA_VISIBLE_DEVICES=1` can silently become a different card.

**Automated checks (added after a run was found using an 8192 context and
mis-sized budgets without anything noticing):**

- `scripts/run_nightly.sh` preflights before loading models: both UUIDs set,
  not equal, present on the host, and warns about stale llama processes.
- `src/nite_eval/gpu_check.py` — `preflight()` runs in the orchestrator after
  the health check. Verifies config pinning statically, then asks the driver
  which GPU each live `llama-server` actually occupies. Wrong GPU is a hard
  error that aborts the run; low VRAM headroom is a warning.
- Bypass with `--skip-gpu-check` only when you know why you are bypassing it.

**When changing anything GPU-related, verify with the driver, not the config:**

```bash
nvidia-smi --query-compute-apps=pid,gpu_uuid,used_memory --format=csv
```

Do not trust a config file to tell you where a model landed. Read the driver.

**Context is 65536** (`--ctx-size` in `config/llama_swap_config.yaml`). Measured
on the 3090 at 64k with `-fa on --cache-type-k q8_0 --cache-type-v q8_0`:

| model | VRAM at 64k | headroom |
|---|---|---|
| gemma4-26b-a4b | 17853 MiB | 6.7 GB |
| qwen3.8-27b | 19211 MiB | 5.2 GB |
| ornith-1.5-35b-a3b | 21412 MiB | **3.1 GB** |

ornith-1.5-35b-a3b is the binding model, inheriting that from
qwen3.6-35b-a3b-strix (21393 MiB) when strix was retired after
run-20260901-043322. qwen3.6-35b-a3b is not measured. gemma looks like the risk on paper
(`head_dim` 512, so its KV grows twice as fast) but llama.cpp uses sliding-window
attention for it, and doubling context cost it only 500 MiB. Re-measure before
going past 64k; do not extrapolate.

**Current known risk:** the two judges occupy ~11.4GB of the 3060's 12GB,
leaving under 1GB headroom. `gpu_check` warns on this. A larger judge model, or
raising judge `--ctx-size` above 4096, will not fit — move a judge to the P40
(index 2) rather than co-locating it with the target.

## Workflow notes

- **Report times in Pacific.** The host clock is UTC, and run IDs, log lines and
  the `started_at` / `finished_at` columns stay UTC — a run ID is an identifier,
  and rewriting it would break `--resume` and every cross-reference to a past
  run. Convert only when reporting a time to a person:

  ```bash
  TZ=America/Los_Angeles date                    # now, in Pacific
  TZ=America/Los_Angeles date -d '2026-09-01 03:13 UTC'   # convert a log time
  ```

  Do not hardcode an offset. Pacific is PDT (UTC-7) through early November and
  PST (UTC-8) after, so `run-20260901-015444` started at 18:54 PDT on Aug 31 —
  the date differs from the run ID's, which is exactly why the ID keeps UTC.
- **Adding a model:** edit `config/llama_swap_config.yaml` (cmd line for llama-server) **and** `config/eval_config.yaml` (`models:` block). If it is a reasoning model, **do not reach for `system_suffix: "/no_think"`** — that string is not a trigger in any Qwen template in this fleet (0 occurrences in qwen3.6, qwen3.8, ornith and qwopus), it is inert filler the model may or may not honour as plain text. Grep the model's own template and use the switch that is actually in it; `enable_thinking: false` is the one that closes the block where it exists, and it is a per-model judgement call: ornith adopted it (+0.05 composite, coding 0.22 -> 0.49), qwen3.8 rejected it (research 0.80 -> 0.63) and uses `reasoning_effort: medium`, qwen3.6 adopted it 2026-09-05 (composite 0.63 -> 0.64, inside noise — the reason is per-task, see below), and muse-glimmer has no `enable_thinking` in its template at all — its knob is `reasoning_strength`.
- **Stopping a run means killing the orchestrator, not `run_nightly.sh`.**
  SIGTERM to the wrapper does not propagate to its foreground child, so bash
  sits in `Ss` waiting while the orchestrator carries on running tasks — and
  anything that then kills llama-swap writes garbage failures into the live
  run. Kill the python process; `run_nightly.sh`'s `trap cleanup EXIT INT TERM`
  fires once it returns and tears down llama-swap and both judges:
  `pkill -f 'nite_eval\.orchestrator'`. Verify the GPU is clear with
  `nvidia-smi --query-compute-apps=pid,gpu_uuid,used_memory --format=csv`
  before starting anything else.
- **Resume after Ctrl-C:** `uv run python -m nite_eval.orchestrator --resume run-YYYYMMDD-HHMMSS`. The DB checkpoints per task.
- **Two judges, one routing rule:** Flow-Judge handles reasoning_quality + practical_output (5-bias, recognizes excellence); RewardAnything handles everything else (3-bias, conservative). Don't merge them — kappa drops without dimension routing.
- **Current baseline: `run-20260901-043322`** — 7 models x 15 tasks, the first
  sweep where every number means what it says. Quote from this, not from
  anything earlier.
- **Scores changed meaning again on 2026-08-31/09-01. A second boundary.** A
  failed task now scores 0 in its dimension instead of being dropped from the
  average, which had let failing a task raise the average of what remained;
  coding `max_tokens` went 24576 -> 32768; three fixture gaps that returned
  errors to well-formed tool calls were closed; and container timestamps are
  normalised out of sandbox output. Every dimension figure from before this is
  incomparable, not just coding.
- **Scores changed meaning on 2026-08-30 (Waves 1 and 2). Do not compare across that boundary.** Fixed since: truncation accepted as answers, `automated` scoring hardcoded to 0.0, `deterministic` returning free 1.0, checklist substring matching, a judge prompt that mandated a 1/3/5 scale, and mocks that reported unconditional success.
- **Speed is tok/s, not s/task, from run-20260902 onward.**
  `task_results.completion_tokens` / `prompt_tokens` hold the server's `usage`
  summed over every generation a task made, and the report's "Latency and
  throughput" table divides them by the same wall clock the latency column
  uses. Both are nullable: every run up to and including `run-20260901-043322`
  shows `—`, because 0 would read as a measurement. Confirmed working on
  `run-20260903-002243` — gemma4 recorded 109.8 tok/s (1036 gen / 4749 prompt
  tokens over 9.4s) on its first task. The figure includes prompt processing
  and tool round trips, so it is end-to-end task throughput, not decode speed.
- **Speculative decoding changes output, so MTP runs need their own baseline.**
  `--spec-type draft-mtp` is NOT exact at temperature 0 — verifying drafted
  tokens in one forward pass changes batch composition and the floating-point
  results, enough to flip a near-tie. Measured on 5 fixed prompts: qwopus
  diverges from its own non-speculative output on 2/5, qwen3.8 on 1/5, one case
  rewriting a whole answer. But the divergence is **deterministic** (same config
  twice = byte-identical) and **depth-independent** (`n-max 1` and `n-max 2` are
  byte-identical to each other, and diverge from no-spec on the same prompts).
  So output depends on one bit — spec on or off — and one re-baseline buys any
  depth. If MTP is ever adopted use depth >= 2; `n-max 1` is strictly worse,
  same divergence and 14% slower.
- **Every Qwen3.8-family GGUF here already carries an MTP draft head that is
  being skipped.** `block_count` 65 with `nextn_predict_layers` 1 means block 64
  is a NextN head (0.42B); without `--spec-type draft-mtp` llama.cpp logs
  `unused tensor blk.64.nextn.* -- ignoring`. Turning it on measured **+75% on
  qwen3.8-27b and +73% on qwopus** decode speed (41.2 -> 72.3 and 42.6 -> 73.6
  tok/s) on short prompts, and the gain **holds under eval conditions**: +69% at
  9k prompt tokens and +70% at 36k, with draft acceptance still 0.85 at 32k.
  Code drafts better than prose (+80% vs +52%), which is the workload that
  dominates here. **Only three fleet models have a head**: `qwen3.8-27b`,
  `ornith-1.5-35b-a3b` and `qwopus3.8-27b-*`. gemma4, both lfm2.5, muse-glimmer
  and — despite sharing ornith's `qwen35moe` arch — `qwen3.6-35b-a3b` do not, so
  it cannot be inferred from the arch string; check
  `<arch>.nextn_predict_layers` plus `blk.N.nextn.*` tensors per file. Enabling
  it is per-model in `llama_swap_config.yaml`, so only those three would need
  re-baselining. **The gain is not uniform**: dense `qwen3.8-27b` gets +75%,
  but MoE `ornith-1.5-35b-a3b` only +20-26% because it already decodes at
  121-156 tok/s against qwen3.8's 35-42 — speculation pays off when decode is
  bandwidth-bound, and a sparse MoE has less to recover. MTP also costs ~1.2 GB
  VRAM (ornith 21386 -> 22650 MiB at ctx 65536), which matters only on ornith
  since it is the binding model. Worth it on qwen3.8-27b, the slowest model in
  the fleet at 2305s per sweep; marginal on ornith at 521s.
- **`chat_template_kwargs` fails silently when the key is not in the template.**
  Jinja does not raise on an unused variable, so a knob copied from a sibling
  model looks correct and does nothing. `qwopus3.8-27b-*` dropped Qwen3.8's
  entire `reasoning_effort` machinery (0 occurrences in its template against 8
  in the base), so `reasoning_effort: medium` — which `qwen3.8-27b` depends on
  — is inert there; only `enable_thinking` acts. **Before setting any
  `chat_template_kwargs`, grep the model's own template for the key**, the way
  the `/no_think` note above was arrived at:
  `uv run --with gguf python -c "from gguf import GGUFReader; \
   print(GGUFReader('M.gguf').fields['tokenizer.chat_template'].contents())"`
- **tok/s cannot compare decode speed between models — it is confounded by turn
  count.** `Gen tok/s` divides generated tokens by wall clock *including prompt
  processing and tool round trips*, so a model that loops spends its time on
  prompt processing and scores low even if it decodes faster.
  `qwopus3.8-27b-q4km` read 37.4 against `qwen3.8-27b`'s 38.4 while using 2.7x
  the prompt tokens; that comparison is meaningless. **Compare on `Decode
  tok/s` instead** — `predicted_n` / `predicted_ms` from the server's `timings`
  block, recorded per task and excluding prompt processing entirely, so turn
  count cannot confound it. Where the two columns disagree, the gap is turn
  count, not speed. Runs before 2026-09-05 predate the column and print `—`;
  `run-20260905-174222` is the first carrying it (lfm2.5-2.6b, 270.7 decode
  against 258.3 gen).
- **Qwopus3.8-27B-Flash was evaluated 2026-09-05 and is not a replacement for
  qwen3.8-27b** — 0.69/0.62 composite against 0.86 in the same run
  (`run-20260905-063950`), with coding 0.50/0.37 against 0.90. Its
  characteristic failure is non-termination: it ends every turn with a tool
  call and never answers, or burns the budget inside `<think>`. Full writeup in
  `docs/comparisons/qwopus3.8-vs-qwen3.8-2026-09-05.md`.
- **A mock gap now records its arguments; the count never could explain
  itself.** `unmatched_mock_calls` conflates a fixture that was too narrow
  (harness at fault, score unfairly low) with a call nothing could match
  (model at fault, score correct) — gemma4 nesting `server` a level too deep
  inside `call_mcp_tool` is the second kind. `unmatched_mock_samples` stores
  the arguments and the report prints them; read those before blaming either
  side. Runs before 2026-09-02 say the arguments are unavailable.
- **Read `unscored_weight` before quoting any score.** Criteria with no implementation are excluded from the weighted average rather than faked, so a score may cover only part of its declared criteria. The report's "Partially Scored Dimensions" section lists any that do. As of 2026-08-30 all 15 tasks are at 0% — coding included, since B1 made `automated` criteria real — but a task added without a hidden suite will silently sit below 100% and the number will be a narrower claim than it looks.
- **qwen3.6 switched to `enable_thinking: false` on 2026-09-05, and its earlier
  runs are not comparable.** It carried `system_suffix: "/no_think"` until then,
  which was inert — `no_think` occurs 0 times in its template and 0 times in
  every other template in this fleet. The composite barely moved (0.63 -> 0.64,
  `run-20260902-045418` -> `run-20260905-235130`), so do not quote it as a win.
  What moved is which tasks finish, both ways: `coding_artemis_medium_01`
  completed for the first time in 7 attempts (0.00 -> 0.50, 1 turn -> 19), while
  `coding_wine_medium_01` regressed 0.86 -> 0.45, answering in one turn with 22
  characters and zero tool calls. `coding_mcp_hard_01` still fails but its
  overrun moved from `reasoning_content` to `content` (103437 chars), which is a
  `max_tokens` question now. n=1 on this config.
- **The coding judges never saw code until 2026-09-06, and every coding score
  before that is on the wrong side of a boundary.** `score_task` passed
  `model_response=conv.final_response`, and code is written through `write_file`
  calls, so `code_quality`, `error_handling`, `edge_case_handling`,
  `architecture` and `cache_design` were all scored from the model's closing
  prose — its summary of its own work. That is 30-60% of each coding task's
  weight (wine 0.60, mcp_hard 0.50, artemis 0.35, mcp_easy 0.30). Evidence: 977
  judge scores on code the automated criterion proved broken averaged 0.465 with
  maxes at 5/5, and nine tasks whose file did not exist scored a judge average of
  0.75 regardless of response length. Fixed by `evidence.build_code_evidence`
  plus an "Absent Work Scores 1" anchor. **Nothing has been re-run — the
  README's coding column predates the fix.**
- **An absent section is not a signal to a judge; state the absence.** First
  attempt at the above omitted the code block when there was no code. Measured
  against a live reward-anything on the real case: `code_quality` fell 4.00 ->
  1.67 but `error_handling` held at 3.67 and `edge_case_handling` at 4.00, the
  reasoning still describing an implementation that was never written — the
  judge fills a missing section from the task spec. With the absence stated
  explicitly all three go to 1.00, and qwen3.8's real code still scores
  4.00/4.00/3.67. Applies to any prompt block you are tempted to omit when empty.
- **The judge prompt has a hard 9600-char budget and the judges cannot be given
  more context.** Both share the 3060 with under 1GB headroom, so `--ctx-size
  4096` is fixed. The per-block caps were independent at 6000 chars each and
  could assemble past the window; `coding_mcp_hard_01` wrote 67181 chars in one
  run. `judge._fit_budget` trims the prose summary first and the artifacts last.
- **Grounding, measured by removing the tools (2026-09-09). Planning is the
  dimension-wide failure; research is one bad task.** The earlier note below is
  correlational — models that happened not to call tools. This is the
  controlled version: the same 6 tasks with `tools:` stripped, 4 models, run in
  a worktree so the artificial results stay out of the main DB.

  | task | grounded | tools removed | delta |
  |---|---|---|---|
  | research_finance_hard_01 | 0.674 | **0.787** | **+0.113** |
  | research_mcp_easy_01 | 0.746 | 0.385 | −0.361 |
  | research_wine_medium_01 | 0.846 | 0.517 | −0.329 |
  | planning_finance_hard_01 | 0.760 | 0.715 | −0.045 |
  | planning_mcp_medium_01 | 0.748 | 0.632 | −0.116 |
  | planning_wine_easy_01 | 0.718 | **0.767** | **+0.049** |

  Two research tasks work — take the tools away and the score craters by ~0.35.
  **`research_finance_hard_01` scores HIGHER with no tools**, so it measures
  recall of public knowledge, not research; a canary analysis reached the same
  verdict independently by showing its distinctive figures (TimesFM 200M
  params, Chronos 20M-710M, LoRA rank 8-16) are real public facts models
  reproduce without retrieval. The fix is a fixture edit — invented figures —
  not a scoring change.

  **All three planning tasks are near-indifferent to tools** (mean −0.037, one
  positive). That is the dimension to fix, and it matches planning's 0.131
  spread, the narrowest of the four.

  **`research_finance_hard_01` was fixed 2026-09-11 and research is now on a new
  boundary.** Its fixtures carry invented canaries (`SwingBench-24`, `FIN-27`,
  `0.847`, `11.3`) and a `grounding` criterion at weight 0.15 scored by
  `contains_check` — deterministic, no judge call, fractional. The other four
  weights were cut to keep the sum at 1.0, so **every model's research number
  moves and nothing has been re-run.** The real public figures (200M, 710M,
  rank 8-16) were left in on purpose: overwriting them would measure deference
  to a tool rather than research. `contains_check` had existed and been
  dispatched since before this, used by zero tasks — it is the cheap way to
  test grounding on any task, and every criterion must be quoted in the YAML or
  a bare `0.847` parses as a float and raises mid-run.

  Research is porous rather than blind: `wine_medium` with no tools ranges
  0.0-0.908, and the top of that range beats the grounded average of 0.846. A
  confident answer still slips through a rubric that otherwise catches this.
- **A canary checklist detects grounding deterministically, at zero judge
  cost.** Facts that exist only in a task's mock fixtures — `70M+ users`,
  `4.5M wines`, `wooj-brain` — cannot be produced from training knowledge, so
  citing one proves retrieval. Measured over runs since run-20260831: **203/250
  grounded responses cite ≥1, and 0/39 ungrounded ones do.** Design rule: a
  canary must be something the fixture author *invented*, not looked up —
  `rank 8-16` and `Postgres 15` were cited without retrieval 13 and 7 times.
  Report it as a flag, never as a composite term: `tool_calls.result` has
  always been stored, so the column is computable backwards over every run and
  moves no historical score. The strongest single case: gemma4 received all 5
  `research_wine_medium_01` canaries in its tool results, cited none, and
  scored 0.825 — reproducibly across 4 runs. That is why this beats checking
  `total_tool_calls > 0`.
- **Rubrics do not test grounding, and planning is the worst case.** Since
  2026-08-30, planning completed with zero tool calls 32 times (22.9%) scoring
  0.75 against 0.76 for tool-using runs; research scores *higher* without tools
  (0.85 vs 0.77). Only coding separates them, via its automated criteria. The
  report's "Tasks That Declared Tools, Used None" section (added 2026-09-06,
  needs `tools_declared`, NULL before then) flags which scores measure the answer
  rather than the work. The rubric fix is deliberately not done — it would move
  planning and research for every model.
- **`finish_reason=length` is a loop 89% of the time, not a budget shortfall.**
  Of 64 truncation failures in the whole DB, 57 are the model repeating itself,
  6 are genuinely long output and 1 is a reasoning overrun — across 11 models,
  so it is not one model's defect. The old error string ended `max_tokens=32768`
  and the log said "raise max_tokens", which is wrong advice in 89% of cases and
  is exactly what the 24576 -> 32768 raise already disproved (gemma4 went 64k ->
  85k chars and failed identically). `diagnose_truncation` now separates three
  shapes: `reasoning_overrun` (empty content — the knob is the reasoning switch,
  not the budget), `degenerate_repetition` (looped — a bigger budget buys a
  longer loop), and `truncated` (genuinely out of room, the only case where
  raising max_tokens is right).
- **Compression ratio detects loops that the suffix detector cannot.**
  `detect_degenerate_repetition` needs an exact, contiguous, suffix-anchored unit
  of ≤8 chars; real loops are a Go table row repeated with the name changing, a
  prose line repeated with others interleaved, or one runaway tool-call token
  filling an 85k-character line. Measured: truncation failures min 0.0031 /
  median 0.0402, against 287 distinct real written files min 0.1468 / median
  0.3205 — the populations do not overlap. Threshold is **0.13**, chosen over
  0.15 because it catches the same 57 loops while flagging 0/287 real files
  instead of 3. **Anything you generate from a template will trip it** — one
  sentence skeleton with the numbers changing measures 0.073 — so test fixtures
  for "healthy long output" must be real text, not generated filler.
- **A dimension that did not run prints `—`, not `0.00`.** The composite was
  always arithmetically right (`get_dimension_averages` omits dimensions with
  nothing terminal, `compute_composite` renormalises over the rest), but the
  table printed 0.00 for absent dimensions, so `--dimension coding` produced
  three columns of 0.00 beside a "Composite" that was only the coding score. The
  report now says which dimensions the composite covers, and warns when models
  were measured over different dimension sets — renormalising over fewer
  dimensions can lift a model above one measured on more.
- **n=15 is the noise floor.** Composite gaps under ~0.05 are inside judge-rubric variance. Need n≥40 or 3-run averages for confident model-vs-model claims.
- **Coding reproducibility was fixed 2026-09-09, and coding scores moved. A
  new comparability boundary.** Coding runs in a real container and the
  container's output enters the conversation, so any per-container variation
  makes two runs at `temperature: 0` diverge from the next turn and write
  different code. Eight surfaces have now been closed; the first commit only
  ever closed part of one:

  | surface | example |
  |---|---|
  | `ls -la` dates | handled since the original commit |
  | `find -ls` / `ls -li` / `ls -ls` dates | `33  4 -rw-r--r-- ... Sep  8 20:44` — the inode/block prefix moved the mode bits off line-start, so the original anchor never matched |
  | ASLR heap address | `<HorizonsClient object at 0x7038b21b87a0>` |
  | `go test` elapsed | `FAIL mcpconfig/config 0.008s` vs `0.007s` |
  | `deno test` elapsed | `... ok (5ms)` vs `(4ms)` |
  | pytest summary | `347 passed in 18.19s` |
  | Go `log` wall clock | `2026/09/06 18:57:14 INFO proxied call` |
  | container hostname | `HOSTNAME=02c32ab71364`, surfaced by `env` |

  ANSI CSI sequences are stripped first, because deno colourizes even with no
  TTY and the colour codes sat between `ok` and `(4ms)`, defeating that anchor.
  Assuming `docker exec` without `-t` meant no colour cost a gate run.

  **Measured before/after**, muse-glimmer, 3 back-to-back repeats each:

  | pair | before | after |
  |---|---|---|
  | artemis_medium | trace diverged | ok, spread 0.029 |
  | mcp_easy | ok | ok, 0.025 |
  | mcp_hard | diverged, `race_detector_clean` + `test_pass_rate` flipped, **spread 0.369** | **ok, 0.042** |
  | wine_medium | trace diverged | ok, 0.042 |

  **Two surfaces remain open and are not fixable by substitution.** Go
  randomizes map iteration order per process, so a table-driven test backed by
  a map permutes whole blocks of output (measured: ornith / mcp_easy, same four
  subtests in different order across runs). And a timestamp the model's *own*
  generated code prints is outside the harness's control. Both are why the gate
  keeps a documented-exception clause instead of demanding absolute byte
  identity.

  **Coding's threshold is 0.10 as of 2026-09-11**, lowered from 0.15 in
  `scoring.dimension_min_detectable_difference` once the confirming gate run
  settled (`run-20260909-030053` / `-034638` / `-043217`, 2 models x 4 coding
  tasks): G1 7/8 pairs byte-identical, G2 8/8, G3 8/8, worst G4 spread 0.042
  against the 0.09 bound. The one G1 failure is ornith / `coding_mcp_easy_01`,
  the Go map-iteration permutation above, and its score spread was 0.000 — the
  trace moved and the number did not. Not 0.05: the judge floor alone is 0.0875,
  so the composite's threshold is out of reach here no matter how clean the
  container gets.
- **`scripts/check_determinism_gate.py` decides whether coding is
  reproducible.** Give it the run IDs of N back-to-back repeats:
  `uv run python scripts/check_determinism_gate.py RUN_A RUN_B RUN_C`. Four
  conditions — G1 byte-identical `(turn, call_index, tool_name, arguments)`
  traces, G2 no completed/failed flips, G3 no deterministic criterion moves
  (anything that is not `judge_rubric`), G4 `weighted_score` spread within
  0.09. The thresholds are measured, not chosen: 0.09 is the worst spread
  (0.0875) across 37 groups / 99 runs whose model output was byte-identical,
  where every criterion that moved was the judge's. G1 carries the power — G2
  alone is underpowered at n=3, because the flip rate given a diverged trace is
  only ~26% per pair. On a G1 failure it prints the first differing byte of the
  *preceding* tool result, since that result is the leak.
  **Run it against the baseline before trusting a fix**: doing so found the
  `find -ls` leak and the ANSI gap, neither of which code review caught.

  **It tells a reordering apart from a leak.** When the two preceding results
  hold the same lines in a different order it says `line-permutation only: N
  lines, M positions reordered` and tags the failure `[line-permutation only]`
  — the Go map-iteration case. The old message was `volatile markers present:
  NONE — investigate, this may be real`, which pointed at the one surface
  already measured and declined, because the `VOLATILE_HINTS` scan only looks
  at a 60-char window around the first differing byte and the reordered lines
  carry no volatile token. It annotates, it does not excuse: a permutation is
  still a G1 failure and the gate still exits non-zero.
- **A failed task's tool calls are now recorded.** They used to be counted in
  `total_tool_calls` and never written, because the flattening sat after the
  failure branch's `return` — 32 failed coding rows had 168 calls and 0 stored.
  A task that fails in one run and completes in the next is exactly the trace a
  determinism check needs. Runs before 2026-09-09 cannot be checked for trace
  identity on their failed tasks.
- **Variance outside coding is the judge's, not the target's.** At
  `temperature: 0` the target is essentially deterministic. The often-quoted
  600163 / 599702 / 601768 ms triple with identical scores is real but was
  **qwen3.8-27b**, not ornith, and it is pre-2026-08-30, so it sits outside the
  comparable set — verified against the DB 2026-09-08. Judge averaging
  (`evaluation.judge_averaging`, on by default, 3 samples) attacks the real
  source for research, planning and agentic. Coding was the exception until
  2026-09-09 because its variance came from the container, not the judge;
  repeating target runs helped there and nowhere else.
- **The judge's 1/3/5 quantization was a prompt bug, not a model limit.** The prompt said "Most responses deserve a 3... You MUST pick exactly 1, 3, or 5." 1579 of 2149 historical scores landed on 3.0 and 4.0 appeared twice. The prompt now describes a real 1-5 scale.

## Concurrent work: run evals on main, build features in a worktree

A full run holds the GPU for hours. Feature work belongs in a worktree so the
two never touch:

```bash
git worktree add .claude/worktrees/<name> -b <branch>
cd .claude/worktrees/<name>
uv sync --all-extras --dev     # a fresh worktree has no .venv
```

**A running eval is immune to anything you merge.** The orchestrator imports its
code at launch, so edits to `main` — even merged ones — do not reach a run in
flight. That is the point: no mid-run behaviour change, and the run stays
internally consistent. The corollary bites though. `run-20260901-015444` was
launched minutes before the `unmatched_mock_calls` work merged and recorded none
of it, because its process predated the column. A fix only applies to runs
started after it lands.

**Gitignored files are absent from a worktree.** `.env`,
`config/llama_swap_config.yaml`, `results/`, `CLAUDE.md` and `TODO.md` are all
ignored, so a fresh worktree cannot run an eval until you copy the first two in
and `mkdir -p results/runs`. Do that only when you actually intend to run one
there — a worktree writes to its own `results/eval_results.db`, which is good
isolation for an experiment and useless for comparing against history. Compare
in the main checkout's DB.

**Worktrees branch from `origin/main` by default.** If the work belongs on top
of an open feature branch, `git reset --hard origin/<branch>` after creating it,
or the first commit silently drops everything that branch added.

Merge order matters when a run is going: land instrumentation that only records
(safe any time) separately from anything that changes a score or a fixture, and
hold the latter until the run finishes. Otherwise the run you are waiting on
becomes uncomparable to the one you are about to start.

## Known model defect: qwen3.8 drops a JSON key quote

qwen3.8-27b reproducibly omits the opening quote of the key following the tool
name, at `temperature: 0`, in runs weeks apart:

```
{"name": "write_file",
arguments": {"content": "package auth\n\nimport (..."}}
```

Everything else is correct — hundreds of lines of escaped source survive intact.
`hermes_parser._repair_dropped_key_quote()` restores the quote (string-aware, so
a `key":` sequence inside escaped source is never touched) and counts the repair.

**Measured rate is high and task-dependent** (`run-20260830-043419`):

| task | tool calls | repaired | rate |
|---|---|---|---|
| coding_mcp_hard_01 | 28 | 23 | **82%** |
| coding_artemis_medium_01 | 24 | 0 | 0% |
| coding_mcp_easy_01 | 18 | 0 | 0% |

It correlates with writing large files, not with task difficulty as such.

Before the repair existed, these calls were discarded and the model was scored on
whatever text fell out — `coding_mcp_hard_01` scored 0.25. With repair it scores
0.50. **Any historical coding comparison involving qwen3.8 is contaminated by
this.** Check the `repaired_tool_calls` column and the report's "Malformed Tool
Calls" section before comparing models; a model with a high repair rate was
previously being penalised for a one-character defect.

## Accepted limitations (do not "fix" these by accident)

- **Coding prompts state an API contract.** Required for hidden suites to
  compile. Measures implementing an interface, not designing one.
- **History is compacted.** Tool calls over 1500 chars become a note. A model
  must `read_file` rather than scroll back. Removing this reintroduces context
  exhaustion on `coding_mcp_hard_01`.
- **Pre-2026-08-30 runs are not a baseline.** They are kept for provenance only.
- **One sample per task.** Gaps under `scoring.min_detectable_difference`
  (0.05) are ties, and reports label them. Repeat target runs do not help — the
  target is deterministic at temperature 0 and the variance is the judge's.

## Things to verify before declaring "model X is better"

1. Both models completed all tasks (check `15/15` in the report).
2. Look for runaway tool calls in logs — gemma4 historically drops 96–183 tool calls per task on agentic items, hitting `max_tool_calls=20`. That depresses its scores artificially.
3. Look for empty content warnings — Qwen3-family thinking models emit empty `content` (everything goes into `reasoning_content`) when reasoning overruns `max_tokens`. Do **not** read that as a missing `system_suffix`: `/no_think` is inert in every template here. Check which reasoning switch the model's template actually has, and whether the budget is simply too small for the prompt.
4. For quant comparisons: run `scripts/compare_quants.sh` first. If perplexities are within CIs, eval-score gaps under ~0.05 composite are likely judge variance, not quant quality.

## Past comparisons

- `docs/comparisons/qwen3-family-2026-04-19.md` — Qwen 3.5 vs 3.6 family, plus deep-dive on unsloth UD-Q4_K_S vs Sero/Strix Q4_K_M (PPL identical, eval gap = judge noise; chat templates differ but not in the code paths nite-eval exercises).
- `docs/comparisons/qwen3.6-thinking-2026-09-05.md` — removing the inert `/no_think` from qwen3.6 and adopting `enable_thinking: false`. Composite 0.63 -> 0.64 (noise), but `coding_artemis_medium_01` completed for the first time in 7 attempts while `coding_wine_medium_01` regressed 0.86 -> 0.45.
- `docs/comparisons/bonsai2-vs-qwen3.8-vs-muse-2026-09-19.md` — first three-way with a ternary quant. qwen3.8 and muse tie; Ternary-Bonsai-2 ties on agentic and research and collapses on coding, and is the slowest of the three in wall clock despite the fastest decode.

## Frontier models, and the two knobs that are not about quality

Since PR #17 a model entry can name a `provider:` — `local` (the default), `anthropic`, `openai`,
or `openai_compatible` for any gateway. Local models keep llama-swap; API models are reached
through their own SDKs. Two things follow that are easy to misread as capability differences:

- **`native_tools` is a confound, not a setting.** Local models get tool definitions pasted into
  the system prompt (Hermes tags); native models get the server's or provider's own tool API. A
  score gap across formats mixes capability with format familiarity, so the format is recorded per
  task, shown as a `Fmt` column, and a run that mixes them carries a caveat in the report. Read
  that caveat before ranking anything. To separate the two effects, re-run one model under the
  other format — `native_tools: false` forces the Hermes path.
- **API spend is measured and capped.** Cost comes from each provider's usage fields, priced per
  model, and the run aborts (checkpointed, resumable) at `cost.max_usd` / `--max-cost`.
  `--dry-run` prints an upper bound with no API calls. Only Anthropic list prices ship as
  defaults; anything else reports as UNPRICED rather than silently counting as free.

Module map additions from that work: `src/nite_eval/providers/` (backends + registry),
`src/nite_eval/cost.py` (pricing, budget cap), and `ApiJudgeClient` in `judge.py`, which lets
`judge.frontier` route named rubric dimensions to an API judge.

## Cross-repo: `ops` owns the host, this repo owns the eval

The eval host's systemd units and setup scripts live in a **separate public repo, `w00jay/ops`**,
under `r730xd/`. The two repos point at the same binary and the same model files, so a change to
where models live has to land in both or one side breaks silently:

| Side | Files |
|---|---|
| nite-eval | `config/llama_swap_config.yaml` (gitignored) and both `*.example.yaml` |
| ops | `r730xd/setup-llm-service.sh` (`MODEL_DIR`), `setup-llm-gemma-service.sh`, `setup-llm-light-service.sh` |
| ops | `r730xd/README.md`, `r730xd/tailscale-local-ai-guide.md` — including their `hf download --local-dir` hints, which are how models ended up in `build/bin` in the first place |

Rules when touching it from here: it is its own repo with its own PRs, so do not write into it
without the user's explicit direction, and **check for uncommitted work first** — on 2026-09-19 it
held 136 lines of in-flight edits across three files, including the exact file a migration needed
to change. Those were committed separately so neither change was bundled under the other's message.

Latent issue there, not fixed: `setup-llm-service.sh` finds models with `find -type f`, which skips
symlinks. Most of `~/models` is HF-cache symlinks, so a future symlinked `Qwen*.gguf` would be
invisible to it.

## Ternary quants: what Bonsai 2 established

`bonsai2-27b-ternary` is the first non-stock-llama.cpp target. Details live in its
`config/eval_config.yaml` entry; the parts that generalise:

- **A vendor quant type may need a vendor fork.** Stock llama.cpp had zero occurrences of `PQ2_0`
  or `PTQ1_0`, so that entry — and only that entry — points at a fork build. Check the symbol in
  the stock checkout before believing a card either way.
- **Do not write a tool-call parser for a new format.** Bonsai's template emits
  `<function=NAME><parameter=KEY>…`, not Hermes. `native_tools: true` made the server's own
  template and grammar produce the call, exactly as it did for ornith. No parser was needed.
- **Decode speed is not throughput.** It decodes ~1.6x faster than either 17 GB rival and still
  took the most wall-clock time in the three-way, because it spends the speed on reasoning tokens.
- **Greedy decoding turns a long trace into a loop.** `degenerate_repetition` failures went 1, 1,
  0 across `reasoning_effort` medium / low / the card's temp 1.0 sampling. Looping is an artifact
  of the temp 0.0 house rule; over-thinking on hard tasks is the model, and neither the effort knob
  nor sampling removed it.
