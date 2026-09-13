# Planning dimension redesign — spec v2 (post devil's-advocate)

## What changed from v1

- **Three prerequisites v1 didn't know it had**, all in the judge plumbing.
  Without them `constraint_handling` is scored by a judge that cannot see the
  constraint. One is already fixed.
- **Dropped the claim that more tasks buys spread.** Task count reduces variance
  of a model's average; it does not separate models. New tasks are justified by
  coverage only (planning lacks artemis-tracker and wooj-brain).
- **Staged into four phases with a kill point**, because the full sweep is ~8.5h
  and v1 changed weights, fixtures and task count at once — if spread had not
  moved, nothing would have been attributable.
- **Canary false-positive worry retired.** Measured 0/145 hits across historical
  planning responses for every proposed token, `74` and `112` included.

## Root cause (unchanged, still the core of this)

1. Every `check_*` mock returns `compatible: true`; every estimate fits. Nothing
   any tool returns would change a plan written without calling it.
2. The mocks hand over **conclusions, not facts** — "Fits within limit",
   "recommend text-only", "OAuth2 refresh flow is the hard part". A model that
   does call the tools is transcribing a verdict, not planning.

## Phase 0 — prerequisites (no scoring change, nothing re-baselines)

| # | item | state |
|---|---|---|
| P1 | `build_tool_evidence` states the absence when no tools were called | **DONE** — 6 tests, `NO_TOOLS_CALLED` |
| P2 | Add a `constraint_handling` rubric to `rubrics.py` | **DONE** — PR #14 |
| P3 | Add `constraint_handling` to `EVIDENCE_DIMENSIONS` | **DONE** — PR #14, `orchestrator.py` |
| P4 | Validate the rubric on a live judge before it carries weight | **PASSED** 2026-09-13 — A 4.67 / B 3.00 / C 1.67 / D 1.00, A−D +3.67 against a 1.0 margin (3 samples) |

P2 exists because `get_rubric()` falls back to `"Rate constraint_handling from 1
(poor) to 5 (excellent)"` — the task YAML's `criteria:` text never reaches the
judge. P3 because `EVIDENCE_DIMENSIONS` is only
`{no_hallucination, data_accuracy, data_threading}`.

The rubric is shared across all five tasks, so it must be constraint-agnostic:
*"Did the plan identify the specific limit the tools reported and route around
it, or does it assume the default path works?"* — with the tool evidence
supplying what the limit was.

**P4 is the gate.** `scripts/validate_constraint_rubric.py` sends four fixed
responses for one task: handles-the-constraint (A), mentions-it-then-ignores-it
(B), ignores-it (C), and confidently-ignores-it-with-no-tool-calls (D). If the
confident ignorer does not
score clearly below the handler, the criterion does not work and no weight goes
on it. This is the method that proved the code-evidence anchor.

## Phase 1 — pilot on one task

Rewrite `planning_wine_easy_01` only. Smallest task, clearest constraint.

- Edge fn wall clock **9.5s**; Claude Vision on 4032x3024 labels **14.2s p50 /
  31.4s p95**. Measurements, no recommendation.
- Default plan puts the scan inline in the edge function and is wrong; a
  grounded plan downscales client-side or goes async.
- Canaries `9.5`, `14.2`, `31.4` (`contains_check`, quoted).

Then a **cheap** run: 3 models x planning only, not the fleet.
`run-20260913-182729`: lfm2.5-8b-a1b, qwen3.6-35b-a3b, ornith-1.5-35b-a3b.

**Kill point.** If ungrounded runs do not lose on wine_easy, stop. The other
four rewrites are wasted work and the approach needs rethinking.

**Pass bar, fixed 2026-09-13 before ornith's result was in.** Ungrounded means
the `grounding` criterion scored 0. Failing either condition is a kill:

1. ornith's wine_easy score beats the best ungrounded wine_easy score in the
   run by **>= 0.15**.
2. qwen3.6 moves the right way against its 0.53 on the old task
   (`run-20260908-012542`): higher if it grounds, lower if it does not.

If ornith does not ground, condition 1 cannot be tested and the pilot is
inconclusive, not a pass.

Why a numeric bar: a no-tools plan can still reach ~0.65 here, because
`completeness` (checklist, 0.20), `dependency_correctness`, `risk_awareness` and
`specificity` do not depend on retrieval. lfm2.5 with zero tool calls scored
0.47, down from 0.59-0.63 on the old task — a loss, but not proof of
separation.

## Phase 2 — roll out to the existing three

| task | planted constraint | default plan gets it wrong by | canaries |
|---|---|---|---|
| mcp_medium | Gmail OAuth2 refresh token bound to client redirect URI — not proxyable without a broker | proxying all 5 servers symmetrically | broker name, `wooj-brain` |
| finance_hard | Storage needs **112GB** against a **100GB** plan; earnings audio unavailable from every listed source | ingesting all 4 sources as-is | `112`, `74` |

Measure planning spread across the three rewritten tasks before going further.
This is the last point where the change is still attributable to the rubric
rather than to the task set.

## Phase 3 — coverage

Add `planning_artemis_medium_01` and `planning_brain_hard_01`, giving planning
one task per source project (n=5, matching agentic). Justified by coverage, not
by spread.

## Scoring (applies from Phase 1, same shape on every task)

| criterion | weight | method |
|---|---|---|
| dependency_correctness / architecture_quality | 0.20 | judge_rubric |
| completeness | 0.20 | checklist |
| specificity / feasibility | 0.10 | judge_rubric |
| risk_awareness | 0.15 | judge_rubric |
| **grounding** | **0.15** | **contains_check** |
| **constraint_handling** | **0.20** | **judge_rubric** |

The 0.20 on `constraint_handling` was provisional until P4; P4 passed, so it
stands.

## Expected effect, and how it can be wrong

A zero-tool plan forfeits `grounding` outright and should lose most of
`constraint_handling` — ~0.35 of the weight, taking today's 0.750 zero-tool
average toward 0.49.

That projection assumed the judge penalises an ungrounded plan. **P1 is what
makes the assumption true**; before it, the judge received an empty section and
backfilled from the task spec. P4 measured it on fixed responses (the no-tools
plan scored 1.00), and the pilot's first live case agrees: lfm2.5, zero tool
calls, got `constraint_handling` 1.00 and `grounding` 0, and scored 0.47 overall.

Second failure mode: if the constraint is too obvious, every model handles it
and the dimension re-saturates at a higher number. The pilot is what detects
this, at 3 models rather than 9.

## Boundaries

- Planning re-baselines for every model (Phase 1 onward).
- Phase 3 takes tasks 15 -> 17, which moves the **composite** as well.
- Research is already un-re-baselined from PR #13. One sweep should close both.
