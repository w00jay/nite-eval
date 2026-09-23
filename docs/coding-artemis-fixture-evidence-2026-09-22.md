# Is `coding_artemis_medium_01` a gate or a hard task?

Evidence for the open decision in `TODO.md` ("may be measuring budget, not coding", found
2026-09-03). Pulled from the results DB on 2026-09-22, after Ternary-Bonsai-2 became the sixth
model that cannot complete it.

## The claim that holds

> Six of ten models have never once completed it since 2026-08-30.

Confirmed, and Bonsai makes it six of thirteen in that window:

| never completes since 2026-08-30 | attempts |
|---|---|
| bonsai2-27b-ternary | 0/6 |
| gemma4-26b-a4b | 0/8 |
| lfm2.5-2.6b | 0/4 |
| qwen3.5-27b | 0/4 |
| qwen3.5-9b | 0/4 |
| qwen3.6-35b-a3b-strix | 0/4 |

Bonsai failed it under **four** separate configurations — `reasoning_effort: medium`, `low`, the
model card's `temp 1.0 / top_p 0.95 / top_k 20` sampling, and `enable_thinking: false` — plus the
doubled-budget run. It has never once completed it.

## The claim that does not hold

> The models that finish score *well* — 0.92 and 0.96 — while the models that fail score nothing
> at all. That bimodality is the tell.

Completed scores in the same window are **not** bimodal: 11 at 0.90+, 34 between 0.50 and 0.89,
15 below 0.50. Two models complete it every time and score near the floor:

| model | completions | average when completed |
|---|---|---|
| lfm2.5-8b-a1b | 4/4 | **0.044** |
| qwopus3.8-27b-q4km | 3/3 | **0.289** |
| qwen3.6-35b-a3b | 3/10 | 0.399 |
| ornith-1.5-35b-a3b | 15/21 | 0.579 |
| qwopus3.8-27b-q5km | 2/3 | 0.690 |
| qwen3.8-27b | 17/23 | 0.726 |
| muse-glimmer-30b | 16/17 | 0.807 |

A gate implies finishing predicts scoring well. Here finishing predicts nothing — the range among
finishers is 0.044 to 0.807. That is a difficulty gradient with a high failure rate.

## It is not an outlier among coding tasks

Full history, all models:

| task | attempts | failed | fail rate |
|---|---|---|---|
| coding_mcp_hard_01 | 200 | 39 | 19.5% |
| **coding_artemis_medium_01** | 200 | 38 | **19.0%** |
| coding_wine_medium_01 | 200 | 10 | 5.0% |
| coding_mcp_easy_01 | 200 | 10 | 5.0% |

It fails at the same rate as `coding_mcp_hard_01`, which has never been flagged. The two hard
coding tasks fail about four times as often as the two easy ones, which is what "hard" should
look like.

## What the evidence actually points at

The problem looks like **aggregation, not the fixture**. A failed task enters the dimension
average as `0.0`, so "ran out of budget" and "wrote bad code" are indistinguishable once averaged.
That is why six models read as gated: their zeros come from non-completion, while
`lfm2.5-8b-a1b`'s 0.044 comes from finishing badly. Those are different facts.

If that reading is right, the fix is to report completion rate alongside the dimension score
rather than to change or drop the task — and it would move every model's coding number, not only
the six.

## Limits of this evidence

- Only **8 attempts** exist since the determinism-gate fix on 2026-09-11, five of them Bonsai's
  failures. The post-fix window is too thin to carry an argument; everything above rests on the
  2026-08-30 window.
- Full-history numbers (200 attempts) mix pre- and post-fixture-fix eras and are shown only for
  the cross-task failure-rate comparison, where both tasks span the same eras.
