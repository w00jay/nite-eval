# BFCL Spike Results (2026-04-05)

## Decision: Skip BFCL, build custom Hermes AST comparator

## Key Findings

**Install method:** No `pip install bfcl-eval`. Real install is clone `github.com/ShishirPatil/gorilla` + `pip install -e .` from subdirectory. Latest confirmed version: v3 (not v4).

**Local endpoint support:** BFCL uses a handler class per model provider (~80-150 lines each). No generic "point at URL" option. A llama.cpp handler would need to be written from scratch.

**Hermes format:** Zero awareness of `<tool_call>` XML tags. Expects OpenAI JSON format. Translation shim needed: ~100-200 lines on top of the handler class.

**Test corpus:** ~2000+ cases across 8 categories:

| Category | Approx Count |
|----------|-------------|
| Simple (single function) | ~400 |
| Multiple functions | ~200 |
| Parallel functions | ~200 |
| Parallel multiple | ~200 |
| Java/JavaScript/Python | ~600 |
| Multi-turn | ~200 |
| Relevance detection | ~200 |
| REST API | ~100 |

## Integration Effort: BFCL vs Custom

| Dimension | BFCL Integration | Custom AST Comparator |
|-----------|-----------------|----------------------|
| Test corpus | 2000+ curated | Start with converted BFCL cases |
| Multi-turn | Built-in | Defer to Inspect AI (Layer 2) |
| Irrelevance detection | Built-in (~200 cases) | Build later if needed |
| Hermes support | None — need shim | Native |
| Time to first result | 2-3 days | 0.5-1 day |
| Maintenance | Track upstream monorepo | Self-contained |

## Rationale

1. Handler class + Hermes shim + debugging = 2-3 days minimum, with ongoing maintenance risk
2. We already have hermes_parser, mock_tools, scoring — AST comparator slots in directly
3. BFCL test cases are JSON on HuggingFace — steal the corpus, skip the framework
4. Multi-turn is better handled by Inspect AI (Layer 2)

## Action Items

- Build custom Hermes AST comparator (~200 lines)
- Download 200-500 BFCL test cases from HuggingFace, convert to task YAML format
- Add irrelevance detection and multi-turn tests in Phase 2 if needed
