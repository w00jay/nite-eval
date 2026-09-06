# qwen3.6-35b-a3b: `/no_think` removed, `enable_thinking: false` adopted

**Date:** 2026-09-05 (probe), run `run-20260905-235130` (sweep, 2026-09-05 23:51 UTC / 16:51 PDT)
**Model:** `qwen3.6-35b-a3b`, Qwen3.6-35B-A3B-UD-Q4_K_S.gguf
**Compared against:** `run-20260902-045418` (the run the README quoted) and
`run-20260901-043322` (the 7-model baseline). Both agree.

## TL;DR

`system_suffix: "/no_think"` never did anything, on any task, in any run — it is
not a trigger in this model's template, or in any template in this fleet. It has
been removed, and `chat_template_kwargs: {enable_thinking: false}` set instead.

**The composite did not meaningfully move: 0.63 -> 0.64.** That is inside the
0.05 noise floor and should not be quoted as an improvement. What moved is
*which* tasks the model can finish, and that moved in both directions.

| | prev (`…045418`) | new (`…235130`) | delta |
|---|---|---|---|
| research | 0.747 | 0.783 | +0.036 |
| planning | 0.769 | 0.706 | −0.063 |
| coding | 0.278 | 0.313 | +0.035 |
| agentic | 0.722 | 0.778 | +0.056 |
| **composite** | **0.63** | **0.64** | **+0.016** |
| tasks completed | 13/15 | 14/15 | +1 |

Every dimension delta is at or inside its threshold (0.05 composite, 0.15 for
coding). Treat all of them as ties. The real findings are per-task.

## Why `/no_think` was removed

Measured on the live server, this exact GGUF, temperature 0, `"What is 17*23?"`,
max_tokens 2048:

| variant | completion | reasoning_len | content_len |
|---|---|---|---|
| baseline | 380 | 872 | 17 |
| `system_suffix "/no_think"` | 381 | 865 | 17 |
| `reasoning_effort: "low"` | 380 | 872 | 17 |
| **`enable_thinking: false`** | **105** | **0** | **211** |

`/no_think` lands one token off baseline; `reasoning_effort` is byte-identical to
it. Both are inert, exactly as their 0 occurrences in the embedded template
predict. `no_think` appears **0 times in every template in this fleet** —
gemma4, qwen3.6, qwen3.8, ornith, muse-glimmer and qwopus alike.

The config comment claimed the opposite ("Qwen3-family chat-template trigger
that disables `<think>` output ... without this, qwen3.6 burns the entire
max_tokens budget on reasoning"). The symptom was real; the stated cause was
not. Removing the suffix is score-neutral for that reason.

## What actually changed, per task

| task | prev | new | note |
|---|---|---|---|
| coding_artemis_medium_01 | 0.00 (1t/0tc, failed) | **0.50 (19t/18tc)** | first completion ever |
| coding_mcp_easy_01 | 0.25 (8t/10tc) | 0.30 (17t/16tc) | tie |
| coding_mcp_hard_01 | 0.00 (1t/0tc, failed) | 0.00 (8t/7tc, failed) | still fails, new failure mode |
| **coding_wine_medium_01** | **0.86 (5t/4tc)** | **0.45 (1t/0tc)** | **regression** |
| planning_wine_easy_01 | 0.78 (1t/0tc) | 0.59 (5t/4tc) | regression |
| research_finance_hard_01 | 0.80 (5t/14tc) | 0.80 (1t/0tc) | same score, no tools used |
| research_wine_medium_01 | 0.82 (5t/10tc) | 0.82 (1t/0tc) | same score, no tools used |
| agentic (5 tasks) | 0.722 avg | 0.778 avg | tie, all 5 complete both ways |

### The win: `coding_artemis_medium_01` completes for the first time

qwen3.6 had failed this task **6 times out of 6** since 2026-08-30, always the
same way — the whole `max_tokens` budget spent inside `reasoning_content`, no
`content`, `finish_reason=length`, zero tool calls. Reproduced in the probe at
turn 1 against the task's real brief:

| | completion | reasoning_len | content_len | finish | tool calls |
|---|---|---|---|---|---|
| thinking ON | 32768 | 120110 | 0 | length | 0 |
| thinking OFF | 27 | 0 | 96 | stop | 1 clean `run_code` |

In the sweep it then ran 19 turns / 18 tool calls and scored 0.50. This is the
single clearest effect of the switch.

### The loss: `coding_wine_medium_01` stops using tools entirely

0.86 (5 turns, 4 tool calls, 94s) -> 0.45 (1 turn, 0 tool calls, **1.3s**). The
entire final response is 22 characters:

    /app/scanlabel/scan_label.ts

The model emitted a file path and stopped. This is a real regression caused by
the switch, not judge variance — the drop is 0.41 against a 0.15 coding
threshold. Thinking-off appears to remove whatever was driving it to explore
before answering.

**It still scored 0.45.** A 22-character non-answer taking a third of the
dimension's tasks with it is a scoring gap worth its own entry — see TODO.md.
The "Tasks That Produced No Answer" section added 2026-09-05 does not catch it,
because that detects the *opposite* failure (every turn ends in a tool call and
no answer is ever emitted). Terminating instantly with a non-answer is not
covered.

### `coding_mcp_hard_01` still fails, but differently

The failure converted from a reasoning overrun to a content overrun:

    prev: 1 turn,  0 tool calls, reasoning_content overruns 32768
    new:  8 turns, 7 tool calls, content_len=103437, finish=length

It now gets seven tool calls into the task before running out of budget — the
same shape as gemma4's artemis failure, which is a `max_tokens` question, not a
reasoning-switch one. Not fixed here.

### Research holds its score while dropping every tool call

`research_finance_hard_01` went 14 tool calls -> 0, and
`research_wine_medium_01` 10 -> 0, both with **identical scores** (0.80, 0.82).
The model now answers from parametric knowledge and the judge cannot tell. That
is a statement about the research rubric, not about this switch, but it is worth
recording: those two scores are not measuring tool use.

## The risk that did not materialise

qwen3.8 rejected this same switch because it cost research 0.80 -> 0.63, and a
turn-1 probe cannot see answer quality. Watched for here: research went **up**
(0.747 -> 0.783). Planning went down 0.063, driven almost entirely by
`planning_wine_easy_01` (0.78 -> 0.59) — which, oddly, is a task where the model
started *using* tools (0tc -> 4tc) and scored worse for it.

## Status

Adopted. `config/eval_config.yaml` carries the measurements inline.

**This opens a new comparison boundary for qwen3.6.** Every earlier qwen3.6
number was produced with inert filler in the system prompt and full reasoning
on; those runs are not comparable to this one. n=1 on the new config, so the
per-task findings above are the durable part, not the dimension means.
