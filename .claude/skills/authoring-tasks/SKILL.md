---
name: authoring-tasks
description: >
  Write or review a nite-eval task YAML — a new evaluation scenario under
  tasks/<dimension>/. Use when adding a task, adding tools or mocks to an
  existing one, changing scoring criteria or max_tokens, or diagnosing a task
  that scores oddly for every model. Encodes the failure modes that have
  actually shipped, not general advice.
---

# Authoring a nite-eval task

Every rule here comes from a bug that reached a scored run. The failure mode is
always the same shape: **the task measures the harness rather than the model**,
and nothing in the report says so.

Read an existing task in the same dimension first. `tasks/agentic/agentic_brain_easy_01.yaml`
is the smallest complete example; `tasks/coding/coding_mcp_easy_01.yaml` shows
the sandbox variant.

## Before you write anything

Decide what the task is supposed to discriminate. If every model will score
0.75–0.80 on it, it costs 15 minutes of GPU per model and tells you nothing —
`planning` currently does this: 0.80 / 0.79 / 0.76 / 0.73 / 0.70 / 0.68 across
seven models, with every per-task gap at or below 0.04.

## Mocks: the model will not phrase calls the way you did

**Every tool needs a catch-all**, listed last, because the matcher takes the
first mock that fits.

```yaml
  - match: {}          # keep last
    response:
      content:
        bottles: []
        note: no bottles match those filters
```

Without one, a reasonable call returns an *error*, which reads to the model as a
broken tool and invites it to retry until it burns its turn budget. Real cases:

| What the model sent | Why it missed |
|---|---|
| `query_inventory({"filters": {"wine_type": "sparkling"}})` | fixture had white, rosé, red |
| `query_inventory({})` | every mock required a `filters` key |
| `search_news({"query": "Nvidia stock analyst..."})` | fixture matched `query_contains: NVDA` only |
| `fetch_url({"url": ...})` | declared in `tools:`, never mocked |

An empty result set is the truthful answer when the fixture has no such data.
An error is not.

**Match on the loosest thing that is still correct.** `query_contains` beats an
exact string; a ticker and the company name should both hit. Nested dicts
already match on the keys the mock declares, so `{filters: {wine_type: red}}`
matches a call that also passes `region` — you do not need to enumerate
combinations.

**A tool with no mocks must be a scored distractor.** `tests/test_task_fixture_coverage.py`
enforces this: an unmocked tool has to be named in a `scoring` criterion, which
is what separates a designed trap (`send_email`, carrying a `tool_absence`
criterion worth 0.2) from one that was forgotten. Run that test after editing
any task.

## Scoring: what the judge actually sees

`judge_rubric` criteria are scored against **`final_response`** — the model's
closing prose, not the work it did. A model that spends its last turn calling a
tool produces no closing message, and the judge is handed:

```
[No final answer produced: every one of 25 turns ended in a tool call ...]
```

It scores 0. In `run-20260901-043322`, ornith's code passed 86% of the hidden
tests on `coding_artemis_medium_01` while `code_quality` and `cache_design` both
scored 0.00 — 35% of that task's weight, structurally unreachable. qwen3.8 scored
1.00 and 0.83 on the same criteria because it writes a 1764-char summary.

So: **if a criterion is about the artifact, weight it `automated`, not
`judge_rubric`** — or accept that models which do not summarise cannot score it.
This is a known open defect (`TODO.md`); until it is fixed, a coding task whose
judge weight exceeds ~30% will rank models by verbosity.

**Judge routing is by criterion name.** `reasoning_quality` and
`practical_output` go to Flow-Judge; everything else to RewardAnything
(`judge.py:414`). Naming a criterion one of those two changes which model grades
it — do not rename for style.

**`automated` criteria need a hidden suite that compiles.** They score from the
exit code of `environment.hidden_test_cmd`, so the task prompt must state the
exact API contract — module, package, types, signatures. A criterion with no
implementation is excluded from the weighted average and shows in
`unscored_weight`; the target is 0%.

## max_tokens: sized to the answer, not guessed

The global default is 4096 (`evaluation.max_tokens`). Override per task when the
answer is large:

| dimension | current | why |
|---|---|---|
| coding | 32768 | whole source files inside a JSON string |
| planning / research | 12288 | long structured prose |
| agentic | 8192 | |

Getting this wrong produces `truncated: finish_reason=length`, which fails the
task outright — 13 of the 15 failures in the last sweep were coding truncations.
**But raising it is not always the fix.** A model that reasons without
converging will fill any budget: ornith emitted 89k chars at 24576 and 121k at
32768 on the same task, same turn, still without a tool call. Check whether the
output is work or deliberation before raising.

Context is 65536, so a task's `max_tokens` plus its history must fit. Half the
window is the practical ceiling.

## Sandbox tasks

Coding tasks add an `environment:` block and get no `mock_responses` — their
tools run against a real container.

- `hidden_test_cmd` scores; `test_cmd` is what the model sees when it calls
  `run_tests`. Keep them different, and `isolate_globs` moves the model's own
  test files aside so its helpers cannot collide with the hidden suite at
  compile time.
- **Container output is not reproducible.** `ls -la` returns the container's
  creation time; timestamps are normalised out (`sandbox._normalize_volatile`)
  but hostnames, `find` ordering and `setup_cmd` mtimes are not. Two runs of one
  model at `temperature: 0` can write different code, so coding carries its own
  0.15 threshold in `scoring.dimension_min_detectable_difference`. Do not read a
  single coding run as a measurement.

## Before committing

```bash
uv run pytest tests/test_task_fixture_coverage.py -q   # mocks + distractor rule
uv run python -c "from pathlib import Path; from nite_eval.task_loader import load_tasks; print(len(load_tasks(Path('tasks'))))"
uv run python scripts/smoke_test.py --model <model> --skip-judge   # needs llama-swap
```

Then run the task for one model and read the report, not just the score:

- `unscored_weight` must be 0%.
- "Unanswered Tool Calls" must be empty — if not, the fixture is short, unless
  the call itself is malformed.
- Check `Turns` and `TCs`: a task finishing in 1 turn with 0 tool calls is
  usually truncation, not success.

**A new task changes what a composite means.** Adding one to a dimension shifts
that dimension for every model, so the fleet needs re-running before old and new
numbers are compared. Say so in the commit.
