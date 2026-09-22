# Ternary-Bonsai-2-27B vs qwen3.8-27b vs muse-glimmer-30b

`run-20260919-193418` — 3 models x 15 tasks, one sitting, same judges, temperature 0.0,
`max_tokens` 4096. Stock llama.cpp `cd26896c1` for qwen3.8 and muse; Ternary-Bonsai-2 runs on the
PrismML fork (`0.2.0-dev`, build 1) because stock rejects its quant type.

The question: **what does a 2.13 bpw ternary quant cost on our own tasks**, measured against the
Q4_K_XL of its own base model (Qwen3.8-27B) and against the other strong 17 GB target we run.

## Result

| Model | Size | Research | Planning | Coding | Agentic | Composite | Tasks |
|---|---|---:|---:|---:|---:|---:|---:|
| qwen3.8-27b | ~17 GB | 0.87 | **0.85** | **0.78** | 0.84 | **0.83** | 15/15 |
| muse-glimmer-30b | ~17 GB | 0.82 | 0.81 | 0.77 | **0.86** | **0.81** | 15/15 |
| bonsai2-27b-ternary | **6.8 GB** | 0.84 | 0.65 | 0.24 | 0.84 | **0.64** | **12/15** |

**qwen3.8 and muse are tied** — 0.018 apart, against a 0.05 resolution floor. They tie on coding
too. Agentic is a three-way tie. Research spans 0.05 across all three, so it is a tie as well.

The only dimensions that separate anything are **planning** and **coding**, and both separate the
ternary build downward.

## Read the format caveat before ranking

qwen3.8 ran **hermes**; bonsai2 and muse ran **native** tool calling. The report flags this
automatically, and it matters here: a gap across formats carries a format effect as well as a
capability one. qwen3.8's win is therefore not clean — it is the only model in the run being asked
to hand-write tool JSON in prose, which if anything handicaps it.

## Decode speed is not throughput

| Model | decode | wall clock, 15 tasks | coding s/task |
|---|---:|---:|---:|
| bonsai2-27b-ternary | **64.9 t/s** | 39.6 min | 457 s |
| qwen3.8-27b | 39.9 t/s | 36.8 min | 322 s |
| muse-glimmer-30b | 41.0 t/s | **25.9 min** | 236 s |

The fastest decoder was the slowest model. Bonsai decodes ~60% faster than either 17 GB rival and
still finished last, because it spends the advantage generating reasoning tokens nobody reads. If
the reason to want a small ternary quant is throughput, this run says tokens/sec was the wrong
thing to measure; time-to-finished-task inverted the ordering.

## The three coding failures are deterministic, not flaky

Bonsai failed the same three coding tasks it failed two days earlier, with the **same character
counts** — 117,243 / 113,649 / 105 — and the same classifications. At temperature 0 the target is
deterministic, so these are a fixed property of this model-config pair on these tasks, not luck.

One is a confirmed repetition loop (`degenerate_repetition`: the 117k-char trace compresses to
8.4% of its size, where real code never measures below 0.147). The other two are plain budget
exhaustion: ~113k chars of `reasoning_content` against a 32,768-token ceiling, `finish_reason=length`,
empty `content`. Raising `max_tokens` buys a longer loop, not an answer.

Two side runs (documented in the `bonsai2-27b-ternary` entry in `config/eval_config.yaml`)
separated the causes:

- `reasoning_effort: low` did not fix it — it relocated it. Same three failures, one coding task
  rescued and `planning_finance_hard_01` broken instead, and `coding_artemis` looped *harder*.
- The card's own sampling (`temp 1.0 / top_p 0.95 / top_k 20`) eliminated **every** repetition
  loop — 1, 1, 0 across medium / low / temp 1.0 — but two coding tasks still died of plain
  truncation.

A third side run (`run-20260922-045506`) tried `enable_thinking: false`, the switch that took
qwen3.6 from 0-for-6 to 0.50 on `coding_artemis_medium_01`. It did not transfer: coding stayed at
0.23 with the same three failures, and agentic fell 0.841 -> 0.721 for a net composite loss
(0.641 -> 0.622). The switch worked at the token level — the loop vanished and per-task tokens
roughly halved — but the model spent the savings on turns, going 63 -> 128 turns and 90 -> 163
tool calls across the sweep and exhausting the same budget in 22-23 short turns.

So looping is an artifact of our temperature 0.0 house rule, and over-thinking on hard tasks is
the model. Neither the effort knob, nor sampling, nor turning thinking off removed the second one.
`coding_artemis_medium_01` never once completed across all four configurations.

## What this does not answer

- **Not the vendor's benchmark claim.** The card's 98.2%-of-FP16 headline came from EvalScope +
  vLLM on an H100 against the unquantized model. Nothing here tests that, and these 15 tasks share
  no benchmarks with that list.
- **Not a format-matched comparison** — see the caveat above.
- **Not a verdict on the ternary method.** It is one build of one model on one rig.

## Verdict

At 40% the size it holds parity on agentic and research and is not usable on coding, which drags
the composite to 0.64 against a tied 0.83/0.81 pair. The speed advantage does not survive contact
with wall-clock time. Keep it as a target — it is cheap to run and the agentic/research parity at
6.8 GB is a real result — but it does not displace either 17 GB model.
