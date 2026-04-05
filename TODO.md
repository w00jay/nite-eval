# nite-eval TODO

## Phase 1: Foundation + Spikes (Weeks 1-2)

### Done
- [x] Install llama-swap + llama-server, verify GPU isolation
- [x] Download Selene-1-Mini Q6_K judge model
- [x] Write `hermes_parser.py` — parse/validate `<tool_call>` tags (17 tests)
- [x] Write `model_manager.py` — start/stop/health-check for llama-server + llama-swap
- [x] Write `judge.py` — judge client with 3x averaging and JSON parsing
- [x] Write `mock_tools.py` — deterministic tool responses from YAML (9 tests)
- [x] Write `conversation_runner.py` — multi-turn Hermes agent loop
- [x] Write `scoring.py` — sequence match, subset, checklist, composite (13 tests)
- [x] End-to-end smoke test passing (model → tool calls → mock responses → judge → score)
- [x] llama-swap config with 3 target models (Qwen 3.5 27B/9B, Gemma 4 26B)

### Remaining
- [ ] **BFCL spike** — clone Gorilla repo, confirm version, test with llama.cpp endpoint, assess Hermes handler effort. Go/no-go after 2 days. Fallback: custom Hermes AST comparator (~200 lines)
- [ ] **Judge calibration shootout** — hand-score 50 examples across scoring dimensions, run Selene + Flow-Judge + RewardAnything, measure Cohen's κ per dimension, require κ > 0.6
- [ ] Download remaining judge candidates (Flow-Judge v0.1, RewardAnything-8B)

## Phase 2: Benchmark Integration + Custom Tasks (Weeks 3-5)

- [ ] **Task loader** — parse YAML task definitions from `tasks/` dir, wire into conversation runner
- [ ] **Split task definitions** — move 15 tasks from `PLANS/nite-eval-tasks.yaml` into individual files under `tasks/{dimension}/`
- [ ] **SQLite storage** — results DB schema, checkpointing after each prompt, resume logic with `(model, task_id, prompt_id)` dedup
- [ ] **Orchestrator** — main loop: iterate models × layers × tasks, store results, handle model swapping via llama-swap
- [ ] **Integrate lm-evaluation-harness** — configure MMLU-Pro, HumanEval, GPQA via YAML, pipe results to SQLite
- [ ] **Integrate BFCL or build custom AST comparator** — based on spike results
- [ ] **Judge prompt templates** — write rubric prompts for research quality, plan quality, code quality, tool-calling
- [ ] **Write 5-15 more custom tasks** — target 20-30 total (5-8 per dimension, 3 difficulty tiers)
- [ ] **Nightly runner script** — `scripts/run_eval.sh` with nohup, PID tracking, log rotation
- [ ] **Overnight test run** — 2 models across Layers 1, 3, 4, verify checkpoint/resume works

## Phase 3: Agentic Eval + Polish (Weeks 6-8)

- [ ] **Integrate Inspect AI** — SWE-bench Lite, optionally GAIA, Docker sandboxing for code tasks
- [ ] **Report generator** — Markdown comparison tables, per-dimension breakdowns, leaderboard JSON
- [ ] **vLLM backend support** — detect backend from config, spawn vLLM serve, poll `/v1/models` for readiness
- [ ] **Contamination awareness** — private test cases unique to real workloads, weight higher than public benchmarks
- [ ] **Full overnight eval** — 4+ models, validate stability across repeated runs (σ < 0.5 on 5-point scale)
- [ ] **Docker sandbox for coding tasks** — safe execution of model-generated code (Go, Python, TypeScript)

## Future: Head-to-Head Comparison

- [ ] **Arena-Hard-Auto v2.0** — deferred until viable judge strategy exists (stronger local judge, or API-judge budget)
- [ ] **τ-bench** — realistic agentic tool use with pass^k consistency metric

## Known Issues

- BFCL claims in original plan were unverified (`pip install bfcl-eval` likely wrong, Hermes format not OOTB)
- Judge calibration shootout blocked on hand-scoring 50 examples
- Mock tool matching uses fuzzy `_contains` matching — needs clear precedence/fallback strategy for edge cases
- Coding tasks need a real sandbox (Docker or route through Inspect AI's sandbox)
