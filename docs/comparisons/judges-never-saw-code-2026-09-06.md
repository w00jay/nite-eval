# The coding judges had never seen code

**Date:** 2026-09-06
**Found via:** `run-20260905-235130`, `coding_wine_medium_01`, qwen3.6-35b-a3b
**Affects:** every coding score in this repo, on both sides of the 2026-08-30 boundary

## What was observed

A 28-character response scored 0.45:

    /app/scanlabel/scan_label.ts

One turn, zero tool calls, 1311ms. The three judge criteria each returned 4/5 at
confidence 1.0, with reasoning describing an implementation in specific detail:

> "It correctly handles error cases as specified, such as returning 200 for
> Claude failures instead of 500."
> "UploadImage errors are handled by continuing execution."
> "catches extractLabel errors to return 200 with manual_entry_required"

The automated criterion in the same row recorded:

    Module not found "file:///app/scanlabel/scan_label.ts"

The file did not exist. `0.75 × 0.60 + 0.0 × 0.40 = 0.45`.

## Root cause

`orchestrator.score_task` passed `model_response=conv.final_response`, and gave
`evidence` only to `no_hallucination`, `data_accuracy` and `data_threading`.
Code is written through `write_file` tool calls, which appear in neither field.

So every coding criterion — `code_quality`, `error_handling`,
`edge_case_handling`, `architecture`, `cache_design` — was scored from the
model's closing prose. Not only on failures. qwen3.8's winning run (0.94-0.98)
was judged on this:

> "The implementation is complete and type-checks successfully. Here's a summary
> of what was built..."

**The judges were scoring self-description.** How much of the dimension:

| task | judge_rubric (blind) | automated (real) |
|---|---|---|
| `coding_wine_medium_01` | 0.60 | 0.40 |
| `coding_mcp_hard_01` | 0.50 | 0.50 |
| `coding_artemis_medium_01` | 0.35 | 0.65 |
| `coding_mcp_easy_01` | 0.30 | 0.70 |

Corroborating counts from the DB:

- 977 judge scores on code the automated criterion proved broken: mean 0.465, max 1.0.
- 9 tasks where the file did not exist at all; **7 scored exactly 0.75**, and
  response length was irrelevant — gemma4's 4593 characters and qwen3.6's 28
  both landed there.

## The fix

`src/nite_eval/evidence.py`:

- `build_code_evidence` reconstructs the final contents of each file written
  (last write per path, original position kept) and is passed to **every**
  `judge_rubric` criterion. Deliberately not a named subset: a hardcoded list is
  what caused this, since `EVIDENCE_DIMENSIONS` silently excluded every coding
  criterion and would have excluded any new one too. It is self-limiting — a task
  that writes nothing produces nothing.
- The judge prompt gained an **"Absent Work Scores 1"** anchor stating that the
  task description is not evidence that any of it was done.
- When a file-writing tool was offered and nothing was written, the prompt says
  so explicitly.

### The first attempt failed, and the measurement is why we know

Omitting the code section when there was no code is not enough. **An absent
section is not a signal** — the judge fills it from the task specification.
Against a live reward-anything, on the real recorded cases, 3-sample averaging:

| case | error_handling | code_quality | edge_case_handling |
|---|---|---|---|
| non-answer, section omitted | 3.67 | 1.67 | 4.00 |
| **non-answer, absence stated** | **1.00** | **1.00** | **1.00** |
| qwen3.8 real code | 4.00 | 4.00 | 3.67 |

With the absence stated, the judge's reasoning becomes correct:

> "The model did not generate any code files as required. The task explicitly
> asked for implementation in scan_label.ts, but the response contains no code.
> This absence of work means all error handling and functionality criteria are
> unmet."

The legitimate winner is undamaged. `coding_wine_medium_01` for qwen3.6 moves
0.45 → 0.00.

Worth noting honestly: showing qwen3.8 its real code barely moved its scores
(4.00/4.00/3.67 against 4.00/4.00/4.00 on prose alone). On this evidence the
anchor and the stated absence are doing most of the work, not the code block
itself — but the code block is what makes the absence meaningful, and a judge
scoring an artifact it cannot see was indefensible regardless.

## Prompt budget

The judges run at `--ctx-size 4096` and cannot be given more: both share the
3060, which has under 1GB of headroom. The per-block caps were independent at
6000 chars each, so task + evidence + code + response could assemble past the
window — `coding_mcp_hard_01` wrote 67181 characters of file content in one run.

`judge._fit_budget` now enforces a 9600-character total, trimming the prose
summary first, then tool results, then code, and the task description only as a
last resort. A real coding prompt with code included measures 9200 chars
(~2875 tokens) against a ~3072-token budget.

## Consequence

**This is a new comparison boundary for every coding score in the repo**, larger
than the qwen3.6 reasoning-switch one, because it changes what 30-60% of each
coding task measures. Nothing has been re-run. The README's Coding column, and
every coding figure in `docs/comparisons/`, predates it.

---

## Re-baseline: `run-20260906-181136`

7 models × 4 coding tasks, run on the fixed harness. 23 completed, 5 failed,
2h09m.

| Model | prev | prev run | **new** | delta | tasks |
|---|---:|---|---:|---:|---:|
| qwen3.8-27b | 0.90 | `…063950` | **0.78** | −0.13 | 4/4 |
| muse-glimmer-30b | 0.70 | `…002243` | **0.61** | −0.09 | 3/4 |
| ornith-1.5-35b-a3b | 0.48 | `…002243` | **0.42** | −0.06 | 4/4 |
| gemma4-26b-a4b | 0.17 | `…002243` | **0.16** | −0.01 | 2/4 |
| qwen3.6-35b-a3b | 0.31 | `…235130` | **0.15** | −0.17 | 3/4 |
| lfm2.5-2.6b | 0.23 | `…002243` | **0.12** | −0.11 | 3/4 |
| lfm2.5-8b-a1b | 0.11 | `…002243` | **0.06** | −0.05 | 4/4 |

**Every model dropped**, which is the expected direction: the old numbers
included credit for describing work. The ordering at the top is unchanged —
qwen3.8 > muse-glimmer > ornith — so the fix did not reshuffle the fleet, it
lowered it. gemma4's −0.01 is not resilience; it was already near the floor.

`coding_wine_medium_01` for qwen3.6 landed at **0.00**, matching the offline
validation exactly.

### What the new report section immediately exposed

`lfm2.5-8b-a1b` has **never made a single tool call on any coding task**, in any
run. Four tasks, one turn each, 2505-10607 characters of prose, byte-identical
across `run-20260903-002243` and this one:

| task | turns | tool calls | response chars | old | new |
|---|---:|---:|---:|---:|---:|
| `coding_artemis_medium_01` | 1 | 0 | 8604 | 0.09 | 0.00 |
| `coding_mcp_easy_01` | 1 | 0 | 5621 | 0.20 | 0.20 |
| `coding_mcp_hard_01` | 1 | 0 | 10607 | 0.02 | 0.00 |
| `coding_wine_medium_01` | 1 | 0 | 2505 | 0.14 | 0.04 |

Its entire coding score was credit for prose. This was invisible before the
"Declared Tools, Used None" section existed.

### One residual case, and it is not the same bug

`coding_mcp_easy_01` still scores 0.20 for `lfm2.5-8b-a1b` with zero tool calls.
That is **not** the judge hallucinating from the spec. The model wrote real Go
inline in the chat — six fenced blocks, `package config`, imports, structs — and
the judge scored the code it was actually shown:

> "The code implements the required structs and validation logic, with proper
> error handling and YAML parsing. Tests cover key validation scenarios, though
> they use a non-standard 'tmpfile' package..."

That is a correct reading of real code. The model produced code but never
delivered it through the tool, so nothing reached disk and both automated
criteria scored 0 — 70% of the task's weight. `0.67 × 0.30 = 0.20` is the
defensible remainder.

Worth knowing for future prompt work: the absence sentinel says "every criterion
about the code scores 1", and the judge overrode it because it could see code.
Judges weigh what is in front of them over an instruction about what is not.

### A reporting wart this run exposed

The summary table prints Research, Planning and Agentic as 0.00 for a
dimension-filtered run and then computes a Composite from them, so qwen3.8's
"Composite 0.78" is just its coding score wearing the wrong label. The composite
should exclude dimensions that were not run. Not fixed here.
