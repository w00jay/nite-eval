#!/usr/bin/env python3
"""End-to-end smoke test for the nite-eval pipeline.

Tests the full loop: load a task → run conversation → score results.
Requires: judge model on :9091, llama-swap on :8080 with at least one target model.

Usage:
    # 1. Start judge on GPU 1:
    CUDA_VISIBLE_DEVICES=1 /home/woojay/P/llama.cpp/build/bin/llama-server \
        -m <judge-model-path>.gguf \
        --port 9091 -ngl 999 --ctx-size 8192 -fa on --no-webui

    # 2. Start llama-swap on GPU 0:
    CUDA_VISIBLE_DEVICES=0 /home/woojay/T/llama-swap/llama-swap \
        --config config/llama_swap_config.yaml --listen :8080

    # 3. Run this script:
    uv run python scripts/smoke_test.py [--model qwen3.5-9b]
"""

import argparse
import sys
import time

from nite_eval.conversation_runner import run_conversation
from nite_eval.judge import JudgeClient
from nite_eval.mock_tools import MockToolEnv
from nite_eval.model_manager import check_health, warm_up_model
from nite_eval.scoring import (
    score_distractor_avoidance,
    score_subset_match,
)

TARGET_URL = "http://127.0.0.1:8080"
JUDGE_URL = "http://127.0.0.1:9091"

# Inline smoke test task (the simple wooj-brain thought capture from tasks YAML)
SMOKE_TASK = {
    "id": "smoke_test_agentic_01",
    "dimension": "agentic",
    "system_prompt": (
        "You are a personal knowledge assistant. You have access to the user's "
        "thought capture system (Wooj-brain) for saving and searching ideas."
    ),
    "tools": [
        {
            "type": "function",
            "function": {
                "name": "capture_thought",
                "description": "Save a new thought to Wooj-brain",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "content": {"type": "string", "description": "The thought content"},
                        "type": {
                            "type": "string",
                            "enum": ["idea", "observation", "question", "decision", "reference"],
                        },
                        "topics": {"type": "array", "items": {"type": "string"}, "description": "Topic tags"},
                    },
                    "required": ["content", "type"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "search_thoughts",
                "description": "Search captured thoughts by meaning",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Semantic search query"},
                        "limit": {"type": "integer", "default": 5},
                    },
                    "required": ["query"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "send_email",
                "description": "Send an email",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "to": {"type": "string"},
                        "subject": {"type": "string"},
                        "body": {"type": "string"},
                    },
                    "required": ["to", "subject", "body"],
                },
            },
        },
    ],
    "user_message": (
        "I just realized that the autoresearch pattern could be applied to my "
        "MCP gateway — the eval metric would be tool-call routing latency. "
        "Save this insight and then find any related thoughts I've had about "
        "evaluation or testing patterns."
    ),
    "mock_responses": {
        "capture_thought": [
            {
                "match": {"content_contains": "autoresearch"},
                "response": {"content": {"id": "thought_123", "status": "saved"}},
            },
            {
                "match": {"content_contains": "any"},
                "response": {"content": {"id": "thought_124", "status": "saved"}},
            },
        ],
        "search_thoughts": [
            {
                "match": {"query_contains": "eval"},
                "response": {
                    "content": {
                        "results": [
                            {
                                "id": "thought_089",
                                "content": "Need a consistent way to benchmark local LLM performance",
                                "type": "question",
                            },
                            {
                                "id": "thought_045",
                                "content": "Karpathy's key insight: the fixed eval is what makes the agent loop work",
                                "type": "observation",
                            },
                        ]
                    }
                },
            },
            {
                "match": {"query_contains": "any"},
                "response": {"content": {"results": []}},
            },
        ],
    },
    "max_turns": 8,
}


def check_services() -> bool:
    """Verify both servers are up."""
    print("Checking services...")
    target_ok = check_health(TARGET_URL)
    judge_ok = check_health(JUDGE_URL)
    print(f"  Target ({TARGET_URL}): {'OK' if target_ok else 'DOWN'}")
    print(f"  Judge  ({JUDGE_URL}): {'OK' if judge_ok else 'DOWN'}")
    return target_ok and judge_ok


def run_smoke_test(model_name: str) -> bool:
    """Run the full smoke test pipeline."""
    task = SMOKE_TASK

    # --- Step 1: Warm up model ---
    print(f"\n[1/4] Warming up model: {model_name}")
    if not warm_up_model(TARGET_URL, model_name, timeout=120.0):
        print("  FAIL: Model warm-up failed")
        return False
    print("  OK")

    # --- Step 2: Run conversation ---
    print("\n[2/4] Running conversation...")
    mock_env = MockToolEnv.from_task_yaml(task["mock_responses"])

    start = time.monotonic()
    result = run_conversation(
        base_url=TARGET_URL,
        model_name=model_name,
        system_prompt=task["system_prompt"],
        tools=task["tools"],
        user_message=task["user_message"],
        mock_env=mock_env,
        max_turns=task["max_turns"],
    )
    elapsed = time.monotonic() - start

    print(f"  Turns: {len(result.turns)}")
    print(f"  Tool calls: {result.total_tool_calls}")
    print(f"  Reached max turns: {result.reached_max_turns}")
    print(f"  Elapsed: {elapsed:.1f}s")
    if result.error:
        print(f"  ERROR: {result.error}")
        return False

    # Show tool call log
    call_log = mock_env.get_call_log()
    print("  Call log:")
    for call in call_log:
        print(f"    [{call['call_number']}] {call['name']}({call['arguments']})")

    # --- Step 3: Deterministic scoring ---
    print("\n[3/4] Deterministic scoring...")
    tool_coverage = score_subset_match(call_log, ["capture_thought", "search_thoughts"])
    distractor = score_distractor_avoidance(call_log, ["send_email"])
    print(f"  Tool coverage (capture + search): {tool_coverage:.2f}")
    print(f"  Distractor avoidance (no send_email): {distractor:.2f}")

    # --- Step 4: Judge scoring ---
    print("\n[4/4] Judge scoring...")
    judge = JudgeClient(base_url=f"{JUDGE_URL}/v1")
    judge_result = judge.evaluate(
        dimension="synthesis",
        rubric=(
            "1 (Poor): Does not connect search results to the user's insight.\n"
            "3 (Acceptable): Mentions the search results but connection is superficial.\n"
            "5 (Excellent): Draws meaningful connections between the search results "
            "and the user's autoresearch/MCP gateway insight."
        ),
        task_description="Capture a thought and find related thoughts about evaluation patterns.",
        model_response=result.final_response,
    )
    print(f"  Judge result: {judge_result}")
    judge.close()

    # --- Summary ---
    print("\n" + "=" * 60)
    print("SMOKE TEST RESULTS")
    print("=" * 60)
    print(f"  Model:             {model_name}")
    print(f"  Turns used:        {len(result.turns)} / {task['max_turns']}")
    print(f"  Tool calls:        {result.total_tool_calls}")
    print(f"  Tool coverage:     {tool_coverage:.0%}")
    print(f"  Distractor clean:  {distractor:.0%}")
    print(f"  Total time:        {elapsed:.1f}s")

    from nite_eval.judge import JudgeResult

    if isinstance(judge_result, JudgeResult):
        print(f"  Judge score:       {judge_result.score}/5")
        print(f"  Judge reasoning:   {judge_result.reasoning[:200]}")
    else:
        print(f"  Judge error:       {judge_result.error}")

    print()
    print("Final response (first 500 chars):")
    print("-" * 40)
    print(result.final_response[:500])

    return tool_coverage > 0.5 and distractor == 1.0


def main():
    parser = argparse.ArgumentParser(description="nite-eval smoke test")
    parser.add_argument("--model", default="qwen3.5-9b", help="Target model name (must match llama-swap config)")
    parser.add_argument("--skip-judge", action="store_true", help="Skip judge scoring (if judge not running)")
    args = parser.parse_args()

    if not check_services():
        if args.skip_judge:
            print("\nJudge not required (--skip-judge). Checking target only...")
            if not check_health(TARGET_URL):
                print("Target server not running. Start llama-swap first.")
                sys.exit(1)
        else:
            print("\nBoth servers must be running. Start them with:")
            print("  # GPU 1 - Judge:")
            print("  CUDA_VISIBLE_DEVICES=1 /home/woojay/P/llama.cpp/build/bin/llama-server \\")
            print("    -m <judge>.gguf --port 9091 -ngl 999 --ctx-size 8192 -fa on --no-webui")
            print("  # GPU 0 - Target:")
            print("  CUDA_VISIBLE_DEVICES=0 /home/woojay/T/llama-swap/llama-swap \\")
            print("    --config config/llama_swap_config.yaml --listen :8080")
            sys.exit(1)

    success = run_smoke_test(args.model)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
