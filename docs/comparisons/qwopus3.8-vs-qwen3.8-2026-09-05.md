# Qwopus3.8-27B-Flash vs Qwen3.8-27B — 2026-09-05

`Jackrong/Qwopus3.8-27B-Flash-GGUF` is a fine-tune of `Qwen/Qwen3.8-27B`, the
model it is compared against here. Same architecture (`qwen35`), same geometry,
same tool-calling path — the fine-tune is the only variable, which makes this
the most controlled model-vs-model comparison run on this harness so far.

Two quants were carried (Q4_K_M and Q5_K_M) because neither matches the
baseline exactly: `qwen3.8-27b` runs unsloth's UD-Q4_K_XL, a mixed dynamic
quant with 191 tensors at Q5_K. Q5_K_M is the closer bit-width, Q4_K_M the
closer file size.

The two models are architecturally identical, measured from the GGUF headers:
both are 866 tensors and 27.32B parameters, both declare `block_count` 65 with
`nextn_predict_layers` 1 (so 26.90B is actually loaded and block 64 — the
0.42B MTP head — is skipped), and both share 24/4 attention heads, key length
256, embedding 5120 and the 248320 Qwen vocabulary. Only the weights and the
chat template differ.

## Reproduce

```bash
# The comparison run (3 models x 15 tasks, ~3h20m)
NITE_MODELS="qwopus3.8-27b-q4km qwopus3.8-27b-q5km qwen3.8-27b" \
    ./scripts/run_nightly.sh

# Chat-template / metadata diff against the base model
uv run --with gguf python scripts/gguf_meta_diff.py \
    /path/to/Qwen3.8-27B-UD-Q4_K_XL.gguf \
    /path/to/Qwopus3.8-27B-Flash-MTP-Q4_K_M.gguf \
    --labels qwen3.8 qwopus-q4km
```

Runs: `run-20260905-052814` (thinking ON, q4km only, aborted after 15 tasks by
design) and `run-20260905-063950` (thinking OFF, all three models, complete).

---

## TL;DR

- **The base model wins decisively.** Composite **0.86** vs **0.69** (q4km) and
  **0.62** (q5km). That gap is more than 3x the 0.05 noise floor — a
  measurement, not a direction.
- **Qwopus's characteristic failure is non-termination**, not weak capability.
  It fails by never stopping: burning the token budget inside `<think>`, or
  ending every turn with a tool call and never producing an answer. Where it
  terminates it is competitive, including three tasks where it ties or beats
  the base.
- **`reasoning_effort` is inert on Qwopus and fails silently.** The fine-tune
  dropped the base model's entire `reasoning_effort` machinery from its chat
  template. Copying the `qwen3.8-27b` config looks correct and does nothing.
  `enable_thinking` is the only live knob.
- **`enable_thinking: false` is the right setting, but it is not a fix.** It
  removed one whole class of failure (research 0.52 -> 0.77) and cost nothing
  measurable elsewhere, but non-termination still occurred three times.
- **Do not read the Q4-vs-Q5 gap as a quant finding.** The entire difference is
  one task where q5km hit non-termination and q4km did not.
- **The vendor's speed claim does not reproduce.** Measured with decode-only
  timings, Qwopus is **+3.4%** over the base without speculative decoding and
  **+1.8%** with it — not +12.8%. Their draft-acceptance gap does not reproduce
  either.
- **MTP is worth far more than the fine-tune: +73-75% decode speed on both
  models**, from a draft head that ships in the base Qwen3.8 and is skipped
  today. But it is **not output-preserving**, so enabling it invalidates
  comparison against every existing run.

---

## Composite scores (`run-20260905-063950`)

| Model | Research | Planning | Coding | Agentic | Composite | Tasks |
|---|---|---|---|---|---|---|
| **qwen3.8-27b** | 0.86 | 0.80 | 0.90 | 0.86 | **0.86** | 15/15 |
| qwopus3.8-27b-q4km | 0.77 | 0.71 | 0.50 | 0.79 | **0.69** | 15/15 |
| qwopus3.8-27b-q5km | 0.56 | 0.75 | 0.37 | 0.81 | **0.62** | 13/15 |

Agentic is close to a tie (0.79 / 0.81 vs 0.86). Coding is where the model
collapses, and coding's own threshold is 0.15 — a 0.40 gap clears it easily.

### Where Qwopus matches or beats the base

| Task | q4km | q5km | qwen3.8 |
|---|---|---|---|
| `agentic_mcp_hard_01` | 0.88 | 0.88 | 0.88 |
| `agentic_finance_hard_01` | 0.71 | 0.72 | 0.69 |
| `coding_wine_medium_01` | 0.94 | 0.85 | 0.85 |

The fine-tune did not break the model. It is fine on tasks short enough to
finish.

---

## The `reasoning_effort` trap

This is the durable finding, and it is the same shape as the `/no_think` trap
already documented for `qwen3.8-27b` in `config/eval_config.yaml` — in reverse.

Qwopus did **not** inherit the base chat template:

| | qwen3.8-27b | Qwopus |
|---|---|---|
| template length | 9993 chars | 4718 chars |
| `reasoning_effort` occurrences | **8** | **0** |
| `enable_thinking` occurrences | 4 | 2 |

The base template carries `reasoning_effort|default('xhigh')` and injects
*"Reasoning effort is set to xhigh. Please think carefully..."* into the system
prompt. Qwopus removed that mechanism outright. Jinja does not raise on an
unused variable, so `chat_template_kwargs: {reasoning_effort: medium}` — the
setting `qwen3.8-27b` depends on — silently does nothing.

Measured on the live server, `"What is 17*23?"` at temp 0, completion tokens /
`reasoning_content` length:

| variant | Q4_K_M | Q5_K_M |
|---|---|---|
| baseline (no kwargs) | 86 / 128 | 81 / 105 |
| `reasoning_effort: low` | 86 / 128 | 81 / 105 |
| `reasoning_effort: medium` | 86 / 128 | 81 / 105 |
| `enable_thinking: false` | 178 / 0 | 96 / 0 |

Byte-identical at every `reasoning_effort` setting. Only `enable_thinking` acts.

**This also explains the model's headline claim mechanically.** The removal of
the xhigh instruction — not just retrained weights — is a large part of why the
fine-tune stops burning its budget inside `<think>`.

---

## Thinking ON vs OFF (q4km, same tasks, same judges)

`run-20260905-052814` ran q4km with thinking at its template default (ON).
It was stopped after 15 tasks once the pattern was clear, and re-run with
`enable_thinking: false`.

| dimension | thinking ON | thinking OFF | delta | verdict |
|---|---|---|---|---|
| research | 0.52 | **0.77** | +0.245 | real |
| coding | 0.42 | 0.50 | +0.073 | inside coding's 0.15 floor |
| agentic | 0.76 | 0.79 | +0.028 | tie |
| planning | 0.76 | 0.71 | -0.047 | tie |
| hard failures | 1 | **0** | — | categorical |

The research gain is one task: `research_finance_hard_01` went **0.00 -> 0.80**.
With thinking on it made 14 tool calls across 6 turns and never answered; with
thinking off it took 15 turns, made the same 14 calls, and produced an 8474-char
answer. Deliberation was preventing it from committing.

Note this failure never involved `<think>` at all, yet disabling thinking fixed
it — worth remembering before assuming a reasoning knob only affects reasoning.

**Conclusion: `enable_thinking: false` is the correct configuration for this
model.** It is also what the author used for their own agentic battery.

---

## Non-termination is the defect

Three tasks in the thinking-OFF run still ended with no answer at all:

| model | task | turns | tool calls | result |
|---|---|---|---|---|
| q4km | `coding_mcp_hard_01` | 33 | 33 | 0.08 |
| q5km | `coding_artemis_medium_01` | 25 | 24 | 0.62 |
| q5km | `research_finance_hard_01` | 5 | 14 | 0.00 |

All three recorded *"No final answer produced: every turn ended in a tool call."*

With thinking ON the same defect appeared as reasoning overflow instead —
`coding_artemis_medium_01` emitted 63528 chars of `reasoning_content` then an
empty answer with zero tool calls; `coding_mcp_hard_01` truncated at 109526
chars. Turning thinking off changed the model from *thinking forever and doing
nothing* to *doing a great deal and achieving nothing*.

**This is independently corroborated by the model card.** The author's single
miss in their 14-task agentic battery (T07) is described as *"a termination and
verbosity failure, not necessarily an inability to implement the parser"* —
the agent had already written a passing solution and kept iterating past its
time limit. We reproduced that failure mode three times on a different harness.

---

## The quant question: unresolved, and this run cannot resolve it

q4km 0.69 vs q5km 0.62 clears 0.05 arithmetically. **Do not read it as a quant
result.** The entire gap is `research_finance_hard_01`, where q5km hit
non-termination and q4km did not — which side of a coin-flip each landed on.

Agentic, the dimension with no container nondeterminism, is a flat tie:

| task | q4km | q5km |
|---|---|---|
| `agentic_artemis_medium_01` | 0.79 | 0.78 |
| `agentic_brain_easy_01` | 0.90 | 0.95 |
| `agentic_finance_hard_01` | 0.71 | 0.72 |
| `agentic_mcp_hard_01` | 0.88 | 0.88 |
| `agentic_wine_medium_01` | 0.69 | 0.73 |

Coding diverged wildly between quants (`artemis` 0.10 vs 0.62 with *identical*
25 turns / 24 tool calls), which is the documented container-timestamp
nondeterminism, not quant quality. Settling the quant question needs
`scripts/compare_quants.sh` plus 3-run coding averages, not one run each.

---

## Speed: measured properly, the +12.8% claim does not reproduce

The original run measured throughput with a metric that could not compare
models. `Gen tok/s` divides generated tokens by wall clock *including prompt
processing and tool round trips*, so it is confounded by turn count — and
Qwopus used 2.7x the prompt tokens because it loops:

| Model | Gen tok/s | Avg ms/task | Avg prompt tok |
|---|---|---|---|
| qwopus3.8-27b-q4km | 37.4 | 136946 | **53205** |
| qwopus3.8-27b-q5km | 34.2 | 219093 | 30925 |
| qwen3.8-27b | 38.4 | 149551 | 19591 |

That table says nothing about either decoder. `predicted_n` / `predicted_ms`
from the server's `timings` block is now captured (see the run TODO below), and
a direct probe settles the question — 5 fixed prompts, temp 0, max_tokens 512,
same GPU, Q4_K_M against the baseline's UD-Q4_K_XL:

| | no spec decoding | `--spec-type draft-mtp --spec-draft-n-max 2` | MTP gain |
|---|---|---|---|
| `qwen3.8-27b` (base) | 41.2 tok/s | 72.3 tok/s | **+75%** |
| `qwopus3.8-27b-q4km` | 42.6 tok/s | 73.6 tok/s | **+73%** |
| **Qwopus over base** | **+3.4%** | **+1.8%** | — |

**The author's +12.8% decode-throughput claim does not reproduce.** Qwopus is
2-3% faster, which on 5 prompts is not distinguishable from nothing. Their
draft-acceptance claim does not reproduce either: they report the base at 66.1%
against Qwopus's 80.7%, but the base measured 0.73-0.93 here.

Both are measured in a different regime from theirs (they used `-np 10` at
327680 context on MMLU-Pro, Q5_K_M for both), so this is a disagreement between
setups rather than proof of an error. What it does establish is that **on this
harness, at this context, there is no meaningful decode-speed advantage.**

**MTP itself is worth far more than the fine-tune.** +73-75% decode speed on
both models, from a draft head that ships in the base Qwen3.8 and is being
skipped today. That is the finding with practical value here.

### MTP is not output-preserving, so MTP runs are not score-comparable

The earlier assumption in this document — that speculative decoding is exact at
temperature 0, so scores stay comparable — is **wrong in practice**:

| model | prompts identical with MTP on |
|---|---|
| `qwopus3.8-27b-q4km` | **3 / 5** |
| `qwen3.8-27b` | **4 / 5** |

Greedy speculative decoding is exact in theory. It is not exact here because
verifying several draft tokens in one forward pass changes batch composition,
and therefore the floating-point results, by enough to flip a near-tie between
candidate tokens. One divergence rewrote a whole answer (438 vs 485 chars).

Both models diverge, so this is a property of llama.cpp's speculative decoding
rather than of either model.

**But the divergence is deterministic, and there are exactly two output
regimes.** Measured on qwopus q4km, same 5 prompts, temp 0:

| comparison | identical |
|---|---|
| MTP n-max 2 vs MTP n-max 2 (rerun) | **5 / 5** |
| MTP n-max 1 vs MTP n-max 2 | **5 / 5** |
| no-spec vs MTP n-max 1 | 3 / 5 |
| no-spec vs MTP n-max 2 | 3 / 5 |

Two theories were tested here. The first — that the divergence is deterministic
batch composition rather than randomness — **holds**: the same configuration run
twice is byte-identical, and every MTP run produced exactly 784 predicted tokens
against no-spec's 765.

The second — that speculating fewer tokens would reduce or remove the
divergence — **fails**. `--spec-draft-n-max 1` diverges from no-spec on exactly
the same 2 of 5 prompts as n-max 2, and is 14% slower (63.5 vs 73.9 tok/s).
Verifying even one drafted token alongside the real one is enough to change the
batch shape; after that, depth costs nothing further in fidelity.

So the rule is simple, and better than expected:

- **Output depends on one bit — speculative decoding on or off** — not on
  depth. Any `--spec-draft-n-max` gives the same tokens.
- **MTP output is reproducible**, so an MTP baseline is as stable as the
  current one.
- **There is no setting that is both faster and comparable to existing runs.**
  The choice is binary.
- If MTP is adopted, **use a depth of at least 2**: n-max 1 is strictly worse,
  slower for identical output.

**Consequence: enabling MTP for an eval run would invalidate comparison against
every existing run** — but it would need re-baselining exactly once, after
which depth can be tuned freely for speed.

## Recommendation

**Do not replace `qwen3.8-27b` with Qwopus.** It is worse on every dimension,
badly so on coding, and its failure mode (never terminating) is the expensive
kind — 33 turns and 662s to score 0.08.

It may still be worth watching for short agentic work, where it ties the base
and the tasks are short enough that non-termination rarely triggers.

## TODO / open questions

- ~~**Capture decode speed.**~~ Done. `predicted_ms` / `predicted_n` from the
  server's `timings` block are recorded per task and reported as "Decode
  tok/s". Verified returned by default on llama-server build 10553; no request
  flag needed.
- ~~**Re-run with `--spec-type draft-mtp` to test the +12.8% claim.**~~ Done as
  a direct probe rather than a full eval, because MTP turned out not to be
  output-preserving — a scored MTP run would not have been comparable to
  anything. See "Speed" above.
- ~~**A no-answer diagnostic in the report.**~~ Done. Tasks where every turn
  ended in a tool call now get their own section. It immediately corrected a
  reading of this very run: q5km scored **0.62** on `coding_artemis_medium_01`
  *while never producing an answer*, earning partial credit from automated
  criteria on files it wrote. That 0.62 had been read here as a success.
- **Settle the quant question properly** with `compare_quants.sh` (perplexity +
  determinism) if it ever matters. On this evidence it does not.
- **Consider running the whole fleet with MTP.** +73-75% decode speed on both
  models tested, from a head that ships in the base model and is skipped today.
  It would require re-baselining every model, since MTP changes output — but
  the run-time saving across a 7-model sweep is large.
- ~~**Does `--spec-draft-n-max 1` preserve output?**~~ Tested: no. It diverges
  from no-spec on exactly the same prompts as n-max 2 while being 14% slower.
  All MTP depths produce identical output to each other, so the only boundary
  is spec on/off — one re-baseline buys any depth.
