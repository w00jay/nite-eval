# Qwen 3.x family comparison — 2026-04-19

Snapshot of nite-eval results plus a deeper investigation of the two Qwen3.6
quants (unsloth UD-Q4_K_S vs Sero/Strix Q4_K_M).

## Reproduce

```bash
# Full nite-eval run across the family (target: ~25 min, depends on GPU)
NITE_MODELS="qwen3.5-9b qwen3.5-27b gemma4-26b-a4b qwen3.6-35b-a3b qwen3.6-35b-a3b-strix" \
    ./scripts/run_nightly.sh

# Two-quant deep-dive (metadata + determinism + wikitext-2 perplexity, ~10 min)
./scripts/compare_quants.sh
```

Raw artifacts (eval logs, perplexity output, extracted Jinja templates) land in `results/runs/` and `results/quant-compare/<timestamp>/`. Both are gitignored — this document is the durable summary.

---

## TL;DR

- **Best overall in this run:** `qwen3.6-35b-a3b` (unsloth UD-Q4_K_S) at composite **0.70**, ~11% ahead of the Qwen3.5 pair (0.62) and the Strix Q4_K_M variant (0.63).
- **The two Qwen3.6 quants are functionally equivalent on language modeling.** Wikitext PPL: 5.9118 vs 5.9243, well inside ±0.037 CIs. The 0.07 nite-eval composite gap between them is **not** a quant-quality gap; it's most likely judge-rubric variance on n=15.
- **Coding scores cap out for every model** (0.15 / 0.25 / 0.42 across all 5 models on the same tasks). The discriminator is broken in coding tasks — they're not measuring model capability.
- **qwen3.5-27b is not earning its size.** 3× the latency of qwen3.5-9b for an identical 0.62 composite.

---

## Composite scores (run-20260418-234519)

| Model | Research | Planning | Coding | Agentic | Composite | Latency (avg ms) |
|---|---|---|---|---|---|---|
| qwen3.5-9b | 0.75 | 0.69 | 0.28 | 0.74 | **0.62** | 31,799 |
| qwen3.5-27b | 0.75 | 0.69 | 0.21 | 0.82 | **0.62** | 95,350 |
| gemma4-26b-a4b | 0.68 | 0.69 | 0.28 | 0.67 | **0.58** | 41,468 |
| qwen3.6-35b-a3b (UD-Q4_K_S) | 0.82 | 0.90 | 0.28 | 0.78 | **0.70** | 32,137 |
| qwen3.6-35b-a3b-strix (Q4_K_M) | 0.77 | 0.77 | 0.21 | 0.77 | **0.63** | 30,965 |

All models completed 15/15 tasks. Run config: temp=0, max_tokens=4096, ctx=8192, `-fa on`, KV cache q8_0.

---

## Qwen 3.5 vs Qwen 3.6 family

### Where Qwen3.6 (unsloth) wins
- **Planning: 0.90** vs 0.69 for both Qwen3.5 variants — biggest single dimension gap.
- **Research: 0.82** vs 0.75.
- **Latency on par with qwen3.5-9b** (32s vs 32s avg) despite being a 35B-A3B MoE — the active-parameter advantage shows up.

### Where Qwen3.5 holds its own
- **Agentic: qwen3.5-27b leads at 0.82** (vs 0.78 for unsloth Qwen3.6) — Qwen3.5-27B's dense weights still win on multi-turn tool-using tasks.
- **Coding is identical across all** — task ceiling, not a model gap.

### Where Qwen3.5 loses
- **qwen3.5-27b is 3× slower than qwen3.5-9b for the same composite (0.62).** Use qwen3.5-9b instead unless the agentic +0.08 is worth ~15 minutes per run.
- **Empty-content turns:** qwen3.5-9b emitted ~10 empty responses across the run (see `eval-run.md` log evidence). Likely needs `system_suffix: "/no_think"` like Qwen3.6 has — see TODO below.

### Recommendation
- **Default model:** qwen3.6-35b-a3b (unsloth UD-Q4_K_S). Best composite, fast latency, no `<think>` runaway with the existing `/no_think` suffix.
- **Backup / for agentic-heavy work:** qwen3.5-27b only when the +0.04 agentic delta justifies 3× the wall time. Otherwise qwen3.5-9b for parity.
- **Skip:** gemma4-26b-a4b (lowest composite, plus persistent runaway tool-call behavior — dropped 96–183 tool calls per task on multiple agentic/research items).

---

## Qwen 3.6: unsloth UD-Q4_K_S vs Strix Q4_K_M

### Quick verdict

| Test | unsloth UD-Q4_K_S | strix Q4_K_M | Conclusion |
|---|---|---|---|
| Wikitext-2 PPL | **5.9118** ± 0.037 | **5.9243** ± 0.037 | Statistically identical |
| nite-eval composite | **0.70** | 0.63 | Real but small (n=15 noise) |
| Determinism (temp=0, 2 runs) | identical | identical | Both reproducible |
| File size | 19.9 GB | 20.2 GB | Comparable |
| Imatrix calibration | yes (76 chunks, 510 entries) | none | unsloth advantage |
| Attention precision | Q8_0 (251 tensors, 1.94 GB) | Q4_K (uniform) | unsloth keeps attention 8-bit |
| Chat template | upstream Qwen + Unsloth fixes | upstream Qwen vanilla | Differ — see below |

### Why perplexity is equal but eval composites differ

Perplexity confirms the two quants are **equally good language models**. But nite-eval scores diverge by 0.07 composite. Three possible drivers, in order of likelihood:

1. **Judge variance on n=15.** Even with deterministic outputs, two quants generate *different but equally valid* responses (different bit patterns of the same model). The judge then scores those slightly differently. ±0.05 composite from 15 tasks is structural — not noise from sampling, but per-task variance in judge rubrics. **Most likely driver.**

2. **Per-quant generation differences amplified by single-shot scoring.** The biggest gap is on planning tasks (0.90 vs 0.77) — all single-turn, 0 tool calls. Templates render identical prompts here (see below), so the model genuinely produces different responses. With n=3 planning tasks, one judge call can swing the per-dimension score by 0.10+.

3. **Chat template differences.** Diffed but unlikely to be the main driver — see next section.

### Chat template diff (see `templates/template.diff`)

Two real categories of difference:

**(a) Strix is strict-mode; unsloth is permissive.**
Strix raises exceptions on: multiple system messages, system not at index 0, missing user query, unexpected role (e.g., `developer`). Unsloth silently accepts all of these. Comment in unsloth's template: `{#- Unsloth fixes - developer role, tool calling #}`.

For nite-eval's request shape (single system → single user → optional tool/assistant turns), neither code path is hit. Both quants completed 15/15 tasks — strix isn't actually raising in practice.

**(b) Tool-call argument re-serialization is slightly different.**
- Strix: `args_value | string if args_value is string else args_value | tojson | safe`
- Unsloth: `args_value | tojson | safe if (mapping or sequence) else args_value | string`

Diverges only on **booleans** (strix: `true`, unsloth: `True`) and **nulls** (strix: `null`, unsloth: `None`). Only matters when re-rendering prior tool calls into the next prompt. Doesn't apply to planning tasks (0 tool calls), which is where the eval gap actually lives.

### Other GGUF metadata differences

Both files are the same Qwen3.6 base model (same vocab sha256, same architecture/dimensions). Other notable differences:

| Field | unsloth | Strix |
|---|---|---|
| `general.file_type` | 14 (Q4_K_S) | 15 (Q4_K_M) |
| `tokenizer.ggml.padding_token_id` | 248055 (correct) | 248044 (BOS reused) |
| Tensor quant placement | 251 Q8_0 + 117 Q4_K + 4 Q6_K | 0 Q8_0 + 371 Q4_K + 61 Q6_K |
| `quantize.imatrix.*` | populated | absent |
| Ancestry metadata | `general.base_model.*`, `quantized_by=Unsloth`, `repo_url` | absent |

The padding-token point is worth flagging: Strix reuses the BOS token id (248044) as the pad token. For pure inference this rarely matters (no batching of variable-length sequences in chat), but it's technically wrong.

### Practical conclusion

- **Use unsloth UD-Q4_K_S** for any production-leaning workload. Imatrix-calibrated, attention preserved at Q8_0, correct padding token, actively maintained.
- **The Strix Q4_K_M is not "broken"** — it's a competent vanilla Q4_K_M quant. PPL is within noise. Use it as a sanity-check baseline; don't expect it to beat unsloth on instruction-following.
- **Don't infer a quant-quality verdict from a single 15-task nite-eval run.** Composite gaps under ~0.05 are inside judge noise.

---

## TODO / open questions

1. **qwen3.5-9b empty content:** add `system_suffix: "/no_think"` to its eval_config entry (it's also a Qwen3-family thinking model). Then re-baseline.
2. **Coding task ceiling:** `coding_mcp_easy_01` returns 0.15 for every model; `coding_mcp_hard_01` returns 0.25 for every model. Investigate whether the rubric or the mock harness is capping these. Current data is not measuring coding capability.
3. **gemma4 runaway tool calls:** drops 96–183 tool calls per task, hitting `max_tool_calls=20` repeatedly. Either tighten gemma's prompting or drop it from the lineup.
4. **qwen3.5-27b cost/benefit:** decide whether the +0.04 agentic delta over qwen3.5-9b is worth 3× the wall time.
5. **n=15 is too few.** Bump task count toward 40+ before drawing strong cross-model conclusions; per-quant variance currently dominates.
6. **(Optional) Settle the template question:** force both Qwen3.6 quants onto the unsloth template, re-run. If composites converge, template was the driver; if not, judge variance is.

---

## Reproduce / extend

```bash
# Quant comparison (metadata + determinism + perplexity, ~10 min)
./scripts/compare_quants.sh

# Skip the slow steps if you only want metadata/templates
./scripts/compare_quants.sh --skip-perplexity --skip-determinism

# Targeted re-run of just the two Qwen3.6 quants
NITE_MODELS="qwen3.6-35b-a3b qwen3.6-35b-a3b-strix" ./scripts/run_nightly.sh

# Just the planning dimension across all models (where the gap lives)
NITE_DIMENSION=planning ./scripts/run_nightly.sh
```
