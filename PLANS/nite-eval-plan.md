# Autonomous LLM evaluation suite for local models

**A fully autonomous overnight evaluation pipeline can be built from existing open-source components — no custom framework from scratch required.** The architecture pairs UK AISI's Inspect AI for agentic evaluations with EleutherAI's lm-evaluation-harness for static benchmarks, orchestrated by a Python pipeline that swaps models on your RTX 3090 via llama-swap while a persistent judge model runs on your RTX 3060. The key insight: rather than building one monolithic tool, compose proven frameworks into a layered evaluation stack, each covering a distinct capability dimension with appropriate scoring methodology.

This plan covers concrete tool selection, directory structure, task definitions, scoring rubrics, Hermes format handling, and a phased implementation roadmap — everything needed to go from zero to overnight autonomous evaluation runs.

> **Validation notes (2026-04-05):** Two layers require pre-commitment spikes before
> building. Layer 3 (BFCL) has unverified installation, version, and Hermes-format
> compatibility claims — a spike is required before committing to the integration.
> Layer 5 (Arena-Hard-Auto) depends on a frontier-class judge for its published
> accuracy metrics; a 7B local judge degrades reliability significantly. Layer 5 is
> deferred to a future phase pending either a stronger local judge or an API-judge
> budget. See details inline.

---

## The five-layer evaluation architecture

The system decomposes into five layers, each backed by a proven framework and targeting specific capabilities. Every layer connects to the same model endpoints via OpenAI-compatible APIs, enabling uniform orchestration.

**Layer 1 — Static benchmarks** uses EleutherAI's lm-evaluation-harness to run MMLU, HumanEval, GPQA, GSM8K, and other standard benchmarks. These establish baseline reasoning and knowledge. The harness natively supports vLLM (`--model vllm`) and llama.cpp via `local-chat-completions` pointed at `http://localhost:8080/v1`. It handles all scoring internally with deterministic metrics.

**Layer 2 — Agentic and multi-turn evaluation** uses UK AISI's Inspect AI, the strongest open-source framework for agent evaluation. Inspect provides a Dataset → Task → Solver → Scorer pipeline with built-in ReAct agents, Docker sandboxing for code execution, and native support for llama.cpp and vLLM model providers. It ships with **100+ pre-built evaluations** including SWE-Bench, GAIA, and Cybench. Custom tasks for Hermes-format tool calling slot directly into its `@task` decorator system. Crucially, Inspect supports separate model roles — the target model generates while the judge model scores, each on different endpoints.

**Layer 3 — Function calling validation** uses AST-based evaluation — comparing the structural tree of generated function calls against ground truth — so **no LLM judge is needed** for scoring. The primary candidate is Berkeley Function Calling Leaderboard (BFCL), which covers single-call, parallel-call, multi-turn, and irrelevance-detection scenarios.

> **⚠ Spike required before committing.** Validation found several unconfirmed claims:
> (1) The `pip install bfcl-eval` install path is likely wrong — BFCL installs from the
> Gorilla monorepo via `pip install -e .`. (2) "BFCL v4" is unverified; latest confirmed
> version is v3. (3) Connecting to a bare llama.cpp endpoint requires writing a custom
> model handler, not just pointing at a URL. (4) Hermes `<tool_call>` XML format is not
> supported out of the box — a shim translating to OpenAI function-call format is needed.
>
> **Spike plan (1–2 days, Phase 1):** Clone `github.com/ShishirPatil/gorilla`, confirm
> current version, test with a llama.cpp endpoint, and assess Hermes handler effort.
> **If spike succeeds:** integrate BFCL as planned.
> **If spike fails (>2 days of handler work):** build a lightweight custom Hermes AST
> comparator that validates `<tool_call>` output against gold-standard sequences. The
> core logic (parse JSON from tags, compare function name + args structure) is ~200 lines
> and already partially prototyped in `hermes_parser.py`. This fallback still covers
> single-call, parallel-call, and argument validation — just without BFCL's multi-turn
> and irrelevance-detection test corpus.

**Layer 4 — Custom practical tasks** uses a purpose-built harness (detailed below) for Hermes-format tool calling, research synthesis, planning, and coding tasks that reflect real usage patterns rather than academic benchmarks. These tasks use the Prometheus 2 judge on the RTX 3060 for rubric-based scoring.

**Layer 5 — Head-to-head comparison (DEFERRED).** Arena-Hard-Auto v2.0 provides pairwise model ranking via 500 challenging real-world prompts with Bradley-Terry scoring. It achieves **87.4% separability** and **90.8% agreement** with human preferences — but these metrics were measured with GPT-4-Turbo as the judge. A 7B local judge exhibits severe position bias and substantially lower separability, making the rankings unreliable.

> **Deferred to future phase.** Layer 5 is excluded from the MVP. It will be added when
> one of: (a) a stronger local judge (14B+) fits on the RTX 3060 and is validated for
> pairwise comparison, (b) an API-judge budget is allocated for this layer only, or
> (c) composite scores from Layers 1–4 prove insufficient for model ranking. In the
> interim, cross-model comparison uses the composite score from Layers 1–4 directly —
> this sacrifices the open-ended quality signal but avoids producing misleading rankings.

| Layer | Framework | What it measures | Scoring method | Judge needed? | Status |
|-------|-----------|-----------------|----------------|---------------|--------|
| 1 | lm-evaluation-harness | Knowledge, reasoning, code | Deterministic metrics | No | MVP |
| 2 | Inspect AI | Agentic tasks, SWE-bench, GAIA | Task-specific + model-graded | Yes (RTX 3060) | MVP |
| 3 | BFCL or custom AST harness | Tool-calling accuracy | AST matching | No | MVP (spike first) |
| 4 | Custom harness | Research, planning, coding, Hermes tools | Rubric + deterministic | Yes (RTX 3060) | MVP |
| 5 | Arena-Hard-Auto | Overall quality ranking | Pairwise Bradley-Terry | Yes (frontier-class) | DEFERRED |

---

## Hardware orchestration and model serving

The dual-GPU setup splits cleanly: **GPU 0 (RTX 3090)** runs target models sequentially, while **GPU 1 (RTX 3060)** runs the judge model persistently throughout the entire evaluation run. GPU isolation uses `CUDA_VISIBLE_DEVICES` per-process, and all communication goes through OpenAI-compatible HTTP APIs.

**llama.cpp is recommended over vLLM for this setup.** The reasons are decisive: startup time is seconds versus 30–120 seconds for vLLM's CUDA graph compilation; GGUF quantization fits larger models in 24GB (a Q4_K_M 30B+ model fits easily, versus FP16 maxing at ~13B on vLLM); and llama-swap provides automatic model hot-swapping that vLLM lacks entirely. vLLM's continuous batching advantage is irrelevant for sequential evaluation where you send one prompt at a time.

The recommended model-swapping solution is **llama-swap** (`github.com/mostlygeek/llama-swap`), a lightweight Go proxy that automatically starts and stops llama-server instances based on the `model` field in API requests. Define all target models in a single YAML config, and the orchestrator simply changes the model name in its API calls — llama-swap handles graceful shutdown, startup, and health checking transparently.

```yaml
# llama-swap config.yaml
models:
  "qwen3.5-30b-a3b":
    cmd: >
      llama-server -m /models/qwen3.5-30b-a3b-Q4_K_M.gguf
      --port ${PORT} -ngl 999 --ctx-size 8192
      -fa on --cache-type-k q8_0 --cache-type-v q8_0 --no-webui --metrics
    checkEndpoint: "http://127.0.0.1:${PORT}/health"
    group: "target-gpu"
  "gemma4-12b":
    cmd: >
      llama-server -m /models/gemma-4-12b-it-Q6_K.gguf
      --port ${PORT} -ngl 999 --ctx-size 8192
      -fa on --cache-type-k q8_0 --cache-type-v q8_0 --no-webui --metrics
    checkEndpoint: "http://127.0.0.1:${PORT}/health"
    group: "target-gpu"
groups:
  "target-gpu":
    exclusive: true  # Only one target model loaded at a time
```

Start the judge model directly, since it never changes:

```bash
CUDA_VISIBLE_DEVICES=1 llama-server \
  -m /models/selene-1-mini-llama-3.1-8b-Q6_K.gguf \
  --port 8081 -ngl 999 --ctx-size 8192 \
  -fa on --no-webui
```

For vLLM support (needed for some models with custom architectures), the orchestrator detects the backend from the eval config and spawns either `llama-server` or `vllm serve` via subprocess. The readiness check differs: poll `/health` for llama.cpp, but poll `/v1/models` for vLLM (since vLLM's `/health` returns 200 before the model is loaded). The key constraint with vLLM is that stopping a model requires killing the entire process — there is no hot-swap API.

---

## The judge model on RTX 3060

### Primary recommendation: Atla Selene-1-Mini (8B)

**Atla Selene-1-Mini** (`AtlaAI/Selene-1-Mini-Llama-3.1-8B`) is the recommended judge model. It is the **#1 ranked 8B generative model on RewardBench**, outperforming GPT-4o on RewardBench, EvalBiasBench, and AutoJ benchmarks. At Q6_K quantization it requires ~6.5GB VRAM, leaving ample room for context on the RTX 3060's 12GB.

Key advantages over the previous candidate (Prometheus 2):
- **Stronger base model**: Llama 3.1 8B vs Mistral 7B v0.2 (two generations behind)
- **All three evaluation modes**: absolute rubric scoring, binary classification, AND pairwise comparison — Prometheus 2 lacks classification
- **128K context window**: vs ~8K for Prometheus 2 — critical for evaluating long model outputs
- **14 GGUF quantizations** available (vs 2 community quants for Prometheus 2)
- **Official prompt templates** at `github.com/atla-ai/selene-mini` for all evaluation modes

### Calibration candidates

Before locking in the judge, Phase 1 includes a calibration shootout with 50 hand-scored examples across 3 candidates:

| Candidate | Params | VRAM (Q6_K) | Strength | Weakness |
|-----------|--------|-------------|----------|----------|
| **Selene-1-Mini** | 8B | ~6.5GB | Best overall; all 3 modes; #1 RewardBench 8B | Newer, less community validation |
| **Flow-Judge v0.1** | 3.8B | ~2.5GB | 0.919 Pearson on 5-Likert rubrics; tiny | Not trained on code/math eval; 8K context |
| **RewardAnything-8B** | 8B | ~6.7GB | Define criteria in natural language; Qwen3 base | Reward scorer, not traditional rubric judge |

Measure Cohen's κ per scoring dimension. Selene is expected to win overall, but Flow-Judge may outperform on text-focused rubrics (research/planning quality) despite being 2x smaller. If Flow-Judge proves strong on text dimensions, consider running both: Flow-Judge for research/planning rubrics (~2.5GB) and Selene for code quality and general scoring (~6.5GB) — both fit on the RTX 3060 simultaneously if needed.

### Other models evaluated and rejected

| Model | Why rejected |
|-------|-------------|
| Prometheus 2 (7B) | Outdated base model (Mistral 7B v0.2), not competitive on RewardBench, leniency bias on 5-point scales |
| Skywork-Critic-8B | Pairwise comparison ONLY — cannot do rubric scoring |
| JudgeLRM-7B/8B | Zero documentation, unvalidated benchmarks |
| Auto-J (13B) | Abandoned (41 downloads/mo), LLaMA 1 base, no GGUF |
| OffsetBias-8B | Pairwise binary output only |
| RISE-Judge-7B | Pairwise only, Chinese-prompt-oriented |
| Self-Taught Evaluator (70B) | Too large for RTX 3060 |

### Design principles for small-model judging

These principles apply regardless of which judge model wins the calibration:

- **Use binary or ternary scales, not 10-point scales.** Score compression and leniency bias are severe with small models on fine-grained scales. The RESEARCHRUBRICS benchmark (Scale AI, 2025) found ternary grading (Yes/Partial/No) improves agreement with human evaluators over binary.
- **Decompose multi-dimensional evaluation into separate single-criterion judge calls.** A single prompt asking the judge to rate "correctness, completeness, and style" simultaneously degrades accuracy. Three separate judge calls, each with one rubric, produce far better results.
- **Require chain-of-thought reasoning before the score.** The judge must output its reasoning before generating a numeric score — this is critical for small models and produces measurably better calibration.
- **Run evaluations 3x and average.** At temperature 0.1–0.2, three runs with majority voting for binary decisions or averaging for numeric scores reduces variance below 0.5 standard deviation.
- **Extract log probabilities when possible.** Low-entropy (high-confidence) judgments approach near-perfect accuracy; high-entropy judgments should be flagged for review.
- **Stay at Q4_K_M or above.** Quantization below Q4 degrades judge quality by 8-15+ points. Q6_K degradation is ~2-5 points — tolerable.

### Where 8B judges work vs. fail

Research from RewardBench, JudgeBench, and CriticBench shows clear boundaries:

**8B judges are reliable for:** format compliance (80-88%), safety/toxicity detection, obvious quality differences, rubric dimensions that are surface-level (conciseness, structure, tone), instruction following.

**8B judges struggle with:** code correctness evaluation (45-60%), mathematical verification, distinguishing "good" from "great" (fine-grained scoring), subtle logical errors, factual accuracy in specialized domains.

This reinforces the plan's core design: **deterministic scoring for code and tool-calling** (test suites, AST matching), reserving the judge only for subjective dimensions (research quality, plan completeness, recommendation quality) where no deterministic alternative exists.

### Judge prompt templates

Use Selene's official prompt templates from `github.com/atla-ai/selene-mini`. For rubric-based absolute scoring, the template follows this structure:

```
You are evaluating {dimension_name}. Analyze step by step, then score.

## Rubric
1 (Poor): {concrete behavioral description}
3 (Acceptable): {concrete behavioral description}  
5 (Excellent): {concrete behavioral description}

## Input
Task: {task_description}
Response: {model_response}

Output ONLY: {"reasoning": "...", "score": N}
```

> **Note:** Adapt to Selene's official template format during Phase 1. The structure above
> is illustrative — Selene has specific prompt templates for absolute, classification,
> and pairwise modes that should be used for best results.

---

## Hermes tool-calling format in the eval harness

All target models use the NousResearch Hermes function-calling format, which wraps tool definitions in `<tools></tools>` XML tags within the ChatML system prompt, model outputs in `<tool_call></tool_call>` tags, and tool responses in `<tool_response></tool_response>` tags with a `tool` role message. The exact format:

**Tool definitions** go in the system prompt as a JSON array of OpenAI-schema tool objects inside `<tools>` tags. **Model tool calls** appear as `<tool_call>{"name": "func", "arguments": {...}}</tool_call>`. **Tool responses** feed back as `<tool_response>{"name": "func", "content": {...}}</tool_response>`. Hermes 3 additionally supports a `<scratch_pad>` reasoning block before tool calls, which is useful for evaluating the model's planning process alongside its tool use.

The eval harness needs a robust parser that handles edge cases discovered in vLLM's Hermes tool parser implementation: arguments may appear before the name key in JSON, arguments may be empty for zero-arg functions, whitespace varies inside tags, and models occasionally produce malformed JSON with trailing commas.

```python
import re, json

TOOL_CALL_RE = re.compile(r'<tool_call>\s*(.*?)\s*</tool_call>', re.DOTALL)

def extract_and_validate(response: str, available_tools: list[dict]):
    """Extract tool calls, validate against schema, return structured results."""
    matches = TOOL_CALL_RE.findall(response)
    results = []
    for match in matches:
        try:
            call = json.loads(match.strip())
        except json.JSONDecodeError:
            results.append({"valid": False, "error": "malformed_json", "raw": match})
            continue
        # Validate function exists
        tool_def = next((t for t in available_tools 
                        if t["function"]["name"] == call.get("name")), None)
        if not tool_def:
            results.append({"valid": False, "error": "unknown_function", "call": call})
            continue
        # Validate required params
        required = tool_def["function"]["parameters"].get("required", [])
        missing = [r for r in required if r not in call.get("arguments", {})]
        if missing:
            results.append({"valid": False, "error": "missing_params", 
                          "missing": missing, "call": call})
            continue
        results.append({"valid": True, "call": call})
    return results
```

The mock tool environment uses deterministic responses keyed on `(function_name, canonical_args)` tuples for reproducibility. Each eval task definition includes both the tool schemas and a response map, enabling the multi-turn conversation loop: send user query → model emits `<tool_call>` → harness validates and returns `<tool_response>` → model continues until it produces a final text answer or exhausts max turns.

---

## Practical task design across four dimensions

Rather than relying solely on academic benchmarks, the suite includes custom tasks that mirror real agent workloads. Each task has three difficulty tiers (easy, medium, hard) and a deterministic scoring path.

**Research tasks** test multi-step information gathering and synthesis. Example: "Research the current state of WebAssembly for server-side applications, covering runtime implementations, production use cases, performance vs native, security model, and ecosystem maturity." The model is given `web_search`, `fetch_url`, and `summarize_text` tools with mock responses returning realistic but fixed content. Scoring decomposes into five binary/ternary criteria: coverage (all aspects addressed?), accuracy (claims match ground truth?), synthesis (connections drawn across aspects?), sources (cited and credible?), and structure (well-organized?). The judge evaluates each criterion independently. Difficulty scales by number of aspects (2 → 5 → 5 plus cross-technology comparison).

**Planning tasks** test dependency-aware task decomposition. Example: "Create a migration plan for moving a PostgreSQL-backed Django app from EC2 to Kubernetes on GKE." Tools include `get_current_infrastructure`, `check_compatibility`, and `estimate_effort`. Scoring checks completeness (all phases covered?), dependency correctness (ordering valid?), risk mitigation (rollback strategy present?), feasibility (realistic timelines?), and specificity (steps actionable, not vague?).

**Coding tasks** use automated test suites as the primary metric, supplemented by judge-scored quality. Example: "Implement a REST API that validates CSV uploads against a JSON schema and returns column statistics." The model gets `read_file`, `write_file`, `run_tests`, and `run_code` tools. **pass@1 on the test suite** is the hard metric (0–5 points), with code quality (type hints, error handling, documentation) adding 0–3 points from the judge, and edge case handling worth 0–2 points. Go tasks additionally run `go vet` and the race detector as automated quality gates.

**Agentic tool-calling tasks** test the full agent loop with Hermes-format tools. Ten task archetypes cover the critical scenarios:

- **Multi-step data gathering**: 3–5 tool calls in sequence, synthesizing results across calls
- **Error recovery**: First tool call returns an error; model must retry or adapt
- **Context maintenance**: Information from early turns must be correctly referenced in later turns
- **Conditional branching**: Tool result determines which path to take next
- **Parallel tool calls**: Multiple independent lookups that could be batched
- **Distractor avoidance**: 6 tools available, only 1 needed — model must not call irrelevant ones
- **Complex argument construction**: Tools with nested parameters and multiple required fields
- **Long-horizon tasks**: 8+ turns with progressive state building
- **Ambiguity handling**: Insufficient information — model should request clarification rather than hallucinate
- **Result chaining**: Output of one tool feeds as input to the next, requiring correct data transformation

Each agentic task scores on four axes: tool selection correctness (right tool?), argument accuracy (right parameters?), sequence efficiency (minimal unnecessary calls?), and final answer correctness (synthesized results accurately?). These are all deterministic — comparing against the gold-standard tool call sequence and expected output — so no judge model is needed for most agentic scoring.

---

## Scoring methodology and composite metrics

The suite produces three levels of scoring: per-task raw scores, per-dimension aggregates, and a single composite score for cross-model comparison.

**Per-task scoring** uses the most reliable method for each task type. Code tasks use pass@k on test suites (deterministic). Tool-calling tasks use AST matching against gold-standard calls (deterministic). Research and planning tasks use rubric-based ternary grading via the Prometheus 2 judge (semi-automated). Standard benchmarks use their built-in metrics (deterministic).

**Per-dimension aggregation** normalizes each task score to 0–1 and computes the mean within each dimension. The four primary dimensions and their component benchmarks:

- **Research**: Custom research tasks (judge-scored) + GAIA subset (exact-match)
- **Planning**: Custom planning tasks (judge-scored) + AgentBoard progress rate
- **Coding**: HumanEval pass@1 + SWE-bench Lite resolved% + custom coding tasks (test pass rate)
- **Agentic**: BFCL accuracy (or custom AST comparator) + custom Hermes tool-calling tasks (deterministic) + τ-bench pass^k

**Composite scoring** uses a weighted average with configurable weights, defaulting to equal weighting across dimensions. The formula: `composite = 0.25 × research + 0.25 × planning + 0.25 × coding + 0.25 × agentic`. Results are stored with full provenance — every composite score traces back to individual task results, judge outputs, and raw model responses.

For **cross-model comparison** in the MVP, models are ranked by their composite scores across Layers 1–4. This lacks the open-ended quality signal that pairwise comparison provides, but avoids producing misleading rankings from a weak judge. Arena-Hard-Auto pairwise ranking is deferred (see Layer 5 and Future Phase).

**Reproducibility safeguards**: temperature 0.0 for all evaluation inference, fixed random seeds, version-controlled prompts and rubrics, deterministic mock tool responses, and the requirement that standard deviation across 3 runs stays below 0.5 on a 5-point scale. All judge prompts, raw outputs, and intermediate results are logged to SQLite alongside the final scores.

---

## Proposed directory structure

```
llm-eval-suite/
├── config/
│   ├── eval_config.yaml          # Master config: models, GPUs, backends, timeouts
│   ├── llama_swap_config.yaml    # llama-swap model definitions
│   ├── models.yaml               # Model registry (name, path, format, backend)
│   └── scoring_weights.yaml      # Dimension weights for composite score
├── tasks/
│   ├── research/
│   │   ├── tech_landscape.yaml   # Task definition + tools + mock responses + rubric
│   │   ├── competitive_analysis.yaml
│   │   └── regulatory_impact.yaml
│   ├── planning/
│   │   ├── cloud_migration.yaml
│   │   ├── incident_response.yaml
│   │   └── data_pipeline.yaml
│   ├── coding/
│   │   ├── api_implementation/   # Each task: prompt.md + tests/ + reference_solution/
│   │   ├── bug_fix/
│   │   ├── go_cli_tool/
│   │   └── concurrent_processor/
│   ├── agentic/
│   │   ├── multi_step_gather.yaml
│   │   ├── error_recovery.yaml
│   │   ├── context_maintenance.yaml
│   │   ├── conditional_branch.yaml
│   │   └── distractor_avoidance.yaml
│   └── benchmarks/
│       ├── lm_eval_tasks.yaml    # Tasks for lm-evaluation-harness
│       ├── bfcl_config.yaml      # BFCL evaluation config
│       └── arena_hard_config.yaml
├── harness/
│   ├── orchestrator.py           # Main overnight loop: model swap → eval → score → report
│   ├── model_manager.py          # Start/stop/health-check for llama.cpp and vLLM
│   ├── hermes_parser.py          # Parse/validate Hermes-format tool calls
│   ├── mock_tools.py             # Deterministic tool response simulator
│   ├── conversation_runner.py    # Multi-turn agent loop with tool execution
│   ├── judge.py                  # Judge client: send to Selene-1-Mini on port 8081
│   ├── scoring.py                # Aggregate scores, compute composite, normalize
│   └── report_generator.py       # Markdown/HTML comparison reports
├── judges/
│   ├── prompts/
│   │   ├── research_quality.txt
│   │   ├── plan_quality.txt
│   │   ├── code_quality.txt
│   │   └── tool_calling.txt
│   └── calibration/
│       ├── calibration_set.jsonl # 50+ examples with human-labeled scores
│       └── calibrate.py          # Validate judge agreement before deployment
├── results/
│   ├── runs/                     # Timestamped run directories
│   │   └── 2026-04-04_2200/
│   │       ├── eval_results.db   # SQLite: all raw results + checkpoints
│   │       ├── results.json      # Exported JSON for portability
│   │       ├── comparison.md     # Generated comparison report
│   │       └── logs/
│   └── leaderboard.json          # Running cross-run leaderboard
├── scripts/
│   ├── run_eval.sh               # nohup wrapper for overnight runs
│   ├── setup_env.sh              # Install dependencies, download judge model
│   └── download_models.sh        # Fetch GGUF models from HuggingFace
├── docker/
│   ├── docker-compose.yaml       # Optional: containerized eval environment
│   └── Dockerfile.sandbox        # Code execution sandbox for coding tasks
├── requirements.txt
└── README.md
```

---

## Example task definition format

Each custom task is a self-contained YAML file with everything needed for reproducible evaluation:

```yaml
# tasks/agentic/error_recovery.yaml
id: agentic_error_recovery_01
dimension: agentic
difficulty: medium
description: "DNS lookup with transient failure requiring retry"

system_prompt: |
  You are a helpful assistant with access to network diagnostic tools.
  
tools:
  - type: function
    function:
      name: dns_lookup
      description: "Resolve a domain name to its IP address"
      parameters:
        type: object
        properties:
          domain: {type: string, description: "Domain to resolve"}
        required: [domain]
  - type: function
    function:
      name: port_check
      description: "Check if a port is open on an IP address"
      parameters:
        type: object
        properties:
          ip: {type: string}
          port: {type: integer}
        required: [ip, port]
  - type: function  # Distractor
    function:
      name: send_email
      description: "Send an email notification"
      parameters:
        type: object
        properties:
          to: {type: string}
          subject: {type: string}
          body: {type: string}
        required: [to, subject, body]

user_message: "Check if example.com has port 443 open."

mock_responses:
  dns_lookup:
    - match: {domain: "example.com"}
      sequence:  # First call fails, second succeeds
        - {error: "DNS timeout, please retry"}
        - {content: {ip: "93.184.216.34", ttl: 3600}}
  port_check:
    - match: {ip: "93.184.216.34", port: 443}
      response: {content: {open: true, service: "https", latency_ms: 12}}

expected_tool_sequence:
  - {name: dns_lookup, args: {domain: "example.com"}}  # Fails
  - {name: dns_lookup, args: {domain: "example.com"}}  # Retries
  - {name: port_check, args: {ip: "93.184.216.34", port: 443}}

scoring:
  tool_selection:
    weight: 0.3
    method: sequence_match  # Deterministic
    criteria: "Called dns_lookup and port_check, not send_email"
  error_recovery:
    weight: 0.3
    method: deterministic
    criteria: "Retried dns_lookup after error without hallucinating an IP"
  argument_accuracy:
    weight: 0.2
    method: exact_match
    criteria: "Correct domain and IP/port in all calls"
  final_answer:
    weight: 0.2
    method: contains_check
    criteria: "States port 443 is open on 93.184.216.34"

max_turns: 8
timeout_seconds: 60
```

---

## Overnight orchestration pipeline flow

The orchestrator is a single Python script that runs unattended. Its execution flow for one complete evaluation run:

**Phase 0 — Startup.** Verify both GPUs are available via `nvidia-smi`. Start the judge model on GPU 1 (port 8081) and confirm readiness by polling `/health`. Start llama-swap on GPU 0 (port 8080) with the model configuration. Log system state (GPU memory, driver version, model checksums) for reproducibility.

**Phase 1 — Per-model evaluation loop.** For each model in the evaluation queue: send a request to llama-swap with the model name in the API call (llama-swap handles loading). Confirm the model is serving by sending a simple test prompt. Run the evaluation layers sequentially (static benchmarks → agentic → BFCL/AST → custom tasks). After all layers complete, the orchestrator checks GPU 0 memory usage — if it exceeds 95%, it triggers a forced model unload before proceeding.

**Phase 2 — Failure recovery.** The pipeline checkpoints progress to SQLite after every prompt. If the process crashes (OOM, power loss, GPU hang), restarting the script automatically resumes from the last checkpoint. Each result row includes a unique `(model_name, task_id, prompt_id)` key, so duplicate detection prevents re-scoring completed prompts. Server crashes are detected by polling `/health` every 30 seconds during inference; if the server dies, the orchestrator waits 10 seconds for GPU memory release, then restarts the server and retries the failed prompt up to twice.

**Phase 3 — Scoring and reporting.** After all models complete all tasks, the orchestrator aggregates scores per dimension, computes composite scores, generates a Markdown comparison report with tables and per-dimension breakdowns, and updates the running leaderboard JSON file. The report includes confidence intervals from 3-run averaging and flags any tasks where the judge's score variance exceeded 0.5.

A simplified version of the core loop:

```python
def run_evaluation(config_path: str):
    config = load_config(config_path)
    db = init_database(config.results_dir)
    judge = JudgeClient(base_url="http://localhost:8081/v1")
    target = TargetClient(base_url="http://localhost:8080/v1")
    
    for model in config.models:
        checkpoint = db.get_checkpoint(model.name)
        if checkpoint.status == "complete":
            continue
        
        # llama-swap auto-loads on first request with model name
        target.test_connection(model_name=model.name)
        
        for layer in [static_benchmarks, agentic_eval, bfcl_eval, 
                       custom_tasks]:  # arena_hard deferred pending frontier judge
            layer.run(
                target=target, judge=judge, model=model,
                db=db, resume_from=checkpoint.get(layer.name, 0)
            )
        
        db.mark_complete(model.name)
    
    generate_comparison_report(db, config.results_dir)
```

---

## Which existing benchmarks to run locally

Not all benchmarks deserve runtime in an overnight evaluation. Based on practical feasibility on RTX 3090, relevance to the four capability dimensions, and setup complexity, the recommended subset:

**Must-run** (high signal, low setup friction): lm-evaluation-harness for MMLU-Pro, HumanEval, and GPQA — establishes baseline reasoning and coding ability with zero custom setup. BFCL for tool calling (AST-scored, directly measures function-calling competency) — **contingent on spike results**; if BFCL integration proves too costly, the custom Hermes AST comparator covers the core signal. Together these cover static knowledge, code generation, and function calling in under 2 hours per model.

**Should-run** (high signal, moderate setup): SWE-bench Lite (300 instances) for end-to-end coding ability — requires Docker but produces the most realistic coding evaluation available. τ-bench for realistic agentic tool use with its unique pass^k consistency metric — the caveat is it requires two simultaneous models, solvable by using historical trajectories for the user simulator or running a tiny model on CPU.

**Nice-to-have** (diminishing returns for this use case): AgentBoard for multi-domain agent analysis (heavy Docker setup), Terminal-Bench 2.0 for CLI skills (newer, less established), WebArena for web navigation (requires six Docker containers).

---

## Implementation roadmap in three phases (+ future phase)

**Phase 1 (Week 1–2): Foundation + spikes.** Install llama-swap and llama-server. Set up GPU isolation. Download judge candidates (Selene-1-Mini Q6_K, Flow-Judge v0.1 bf16, RewardAnything-8B Q6_K) to RTX 3060. Write `model_manager.py` (start/stop/health-check) and `hermes_parser.py` (extract and validate tool calls). Validate the basic loop: load model → send prompt → get response → parse tool calls → send to judge → store result.

**Judge calibration shootout (days 1–2 of Phase 1):** Hand-score 50 examples across all scoring dimensions (research quality, plan quality, code quality, tool-calling correctness). Run each judge candidate on the same 50 examples. Measure Cohen's κ per dimension — require κ > 0.6 overall. Expected outcome: Selene-1-Mini wins overall; Flow-Judge may win on text-focused rubrics. Lock in the judge (or judges, if dimension-specific routing proves worthwhile) before proceeding.

**BFCL spike (days 3–4 of Phase 1):** Clone `github.com/ShishirPatil/gorilla`, confirm current version and install method, test with a llama.cpp `/v1/chat/completions` endpoint, and assess effort to add a Hermes-format handler. **Go/no-go decision at end of day 4:** if BFCL integrates in ≤2 days of handler work, proceed with it in Phase 2. If not, build the custom Hermes AST comparator (~200 lines, validates `<tool_call>` output against gold-standard sequences using the parser from `hermes_parser.py`).

**Phase 2 (Week 3–5): Benchmark integration + custom tasks.** Integrate lm-evaluation-harness for static benchmarks (configure via YAML, pipe results to SQLite). Integrate BFCL or build custom AST comparator (based on spike results). Set up the SQLite schema, checkpointing, and resume logic. Write 20–30 custom task definitions across all four dimensions (5–8 per dimension, three difficulty tiers each). Build `mock_tools.py` and `conversation_runner.py` for multi-turn Hermes tool calling. Write judge prompt templates for research, planning, and code quality scoring. Build the nightly runner script with nohup and PID tracking. Test overnight run with 2 models across Layers 1, 3, and 4.

**Phase 3 (Week 6–8): Agentic evaluation + polish.** Integrate Inspect AI for SWE-bench Lite and optionally GAIA. Build the scoring aggregation pipeline and composite score computation. Build the report generator (Markdown tables, per-dimension breakdowns). Add vLLM backend support as an alternative to llama.cpp. Implement contamination awareness: add private test cases unique to your use cases that cannot have leaked into training data. Run full overnight evaluation across 4+ models and validate that results are stable across repeated runs. Weight private custom tasks more heavily than public benchmarks in the composite score.

**Future phase: Head-to-head comparison.** Integrate Arena-Hard-Auto v2.0 for pairwise ranking when a viable judge strategy is in place. Options: (a) a stronger local judge (14B+) that fits on RTX 3060 with validated pairwise accuracy, (b) API-judge budget for this layer only (Arena-Hard is 500 prompts × 2 position swaps = 1000 judge calls per model pair — cost is bounded), or (c) evidence that composite scores from Layers 1–4 are insufficient for model ranking decisions. Until then, cross-model ranking uses the Layer 1–4 composite score directly.

The Karpathy autoresearch insight applies here as a meta-principle: **lock the evaluation, iterate on the models.** Once the suite is built and validated, it becomes the fixed "prepare.py" — the immutable ground truth against which every new model release is measured overnight, producing a morning-ready comparison report that immediately answers "is this model better for my agentic workloads?"

---

## Conclusion: key design decisions and their rationale

The most consequential decision in this architecture is **composing existing frameworks rather than building from scratch**. Inspect AI alone would cover 70% of needs, but combining it with lm-evaluation-harness for static benchmarks and AST-based tool-call validation produces better coverage with less custom code. The second key decision is **deterministic scoring wherever possible** — AST matching for tool calls, test suite pass rates for code, exact match for factual queries — reserving the LLM judge only for subjective dimensions like research quality and plan completeness where no deterministic alternative exists.

The third key decision, added during validation, is **not trusting frameworks at face value**. BFCL's plug-and-play claims and Arena-Hard-Auto's accuracy metrics both required verification against the actual constraints (Hermes format, local judge). The spike-before-committing approach for BFCL and the deferral of Arena-Hard-Auto avoid sinking weeks into integrations that may not deliver their promised value under the fully-local architecture.

The architecture deliberately avoids two common pitfalls. First, it does not attempt to run the judge and target on the same GPU — even with quantized models, context switching between inference sessions on a single GPU introduces latency spikes, VRAM fragmentation, and the risk of OOM during the judge call when the target model's KV cache is still resident. Second, it does not use the target model as its own judge — self-enhancement bias (where LLMs preferentially rate their own outputs higher) is well-documented and would invalidate cross-model comparisons.

The final and perhaps most important insight: **private, domain-specific tasks are more valuable than public benchmarks for practical model selection.** Every public benchmark is a contamination risk — models may have trained on it, inflating scores. The custom Hermes tool-calling tasks, coded to your specific agent patterns with your specific tool schemas, produce signal that no public leaderboard can match. Build 30 high-quality custom tasks that reflect your actual agentic workloads, and those alone will tell you more about which model to deploy than any combination of academic benchmarks.