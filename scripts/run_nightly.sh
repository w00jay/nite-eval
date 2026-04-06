#!/usr/bin/env bash
# Nightly evaluation runner for nite-eval
#
# Starts judge servers, runs the full eval pipeline, generates report.
# Designed for unattended overnight execution via nohup or cron.
#
# Usage:
#   # Interactive (foreground):
#   ./scripts/run_nightly.sh
#
#   # Unattended (background):
#   nohup ./scripts/run_nightly.sh > results/nightly.log 2>&1 &
#
#   # With specific models:
#   NITE_MODELS="qwen3.5-9b" ./scripts/run_nightly.sh
#
# Prerequisites:
#   - Target llama-swap must be running on :9080 (GPU 1 / RTX 3090)
#   - UV environment set up (uv sync)
#
# Environment variables:
#   NITE_MODELS       Space-separated model list (default: all from config)
#   NITE_DIMENSION    Filter to one dimension (default: all)
#   NITE_CONFIG       Config path (default: config/eval_config.yaml)
#   NITE_JUDGE_GPU    GPU ID for judge servers (default: 2)
#   NITE_TARGET_URL   Target server URL (default: http://127.0.0.1:9080)

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_DIR"

# Config
JUDGE_GPU="${NITE_JUDGE_GPU:-2}"
TARGET_URL="${NITE_TARGET_URL:-http://127.0.0.1:9080}"
CONFIG="${NITE_CONFIG:-config/eval_config.yaml}"
LLAMA_SERVER="/home/woojay/P/llama.cpp/build/bin/llama-server"
REWARD_MODEL="/home/woojay/models/RewardAnything-8B-v1.Q6_K.gguf"
FLOW_MODEL="/home/woojay/models/Flow-Judge-v0.1.Q6_K.gguf"
REWARD_PORT=9091
FLOW_PORT=9092

# PID tracking for cleanup
REWARD_PID=""
FLOW_PID=""

cleanup() {
    echo ""
    echo "=== Cleaning up ==="
    if [ -n "$REWARD_PID" ] && kill -0 "$REWARD_PID" 2>/dev/null; then
        echo "Stopping reward-anything (PID $REWARD_PID)"
        kill "$REWARD_PID" 2>/dev/null || true
        wait "$REWARD_PID" 2>/dev/null || true
    fi
    if [ -n "$FLOW_PID" ] && kill -0 "$FLOW_PID" 2>/dev/null; then
        echo "Stopping flow-judge (PID $FLOW_PID)"
        kill "$FLOW_PID" 2>/dev/null || true
        wait "$FLOW_PID" 2>/dev/null || true
    fi
    echo "Cleanup complete"
}
trap cleanup EXIT INT TERM

echo "=== nite-eval nightly runner ==="
echo "Time: $(date -u '+%Y-%m-%d %H:%M UTC')"
echo "Project: $PROJECT_DIR"
echo "Config: $CONFIG"
echo ""

# Check target server
echo "Checking target server at $TARGET_URL..."
if ! curl -sf "$TARGET_URL/health" > /dev/null 2>&1; then
    echo "ERROR: Target server not responding at $TARGET_URL"
    echo "Start with: CUDA_VISIBLE_DEVICES=1 /home/woojay/T/llama-swap/llama-swap --config config/llama_swap_config.yaml --listen :9080"
    exit 1
fi
echo "Target server OK"

# Start a judge server if not already running
start_judge() {
    local model_path="$1"
    local port="$2"
    local name="$3"

    if curl -sf "http://127.0.0.1:$port/health" > /dev/null 2>&1; then
        echo "$name already running on :$port"
        return 0
    fi

    echo "Starting $name on :$port (GPU $JUDGE_GPU)..."
    CUDA_VISIBLE_DEVICES="$JUDGE_GPU" "$LLAMA_SERVER" \
        -m "$model_path" \
        --port "$port" \
        -ngl 999 --ctx-size 4096 -np 1 --no-webui \
        > "results/${name}.log" 2>&1 &

    local pid=$!
    echo "$name started (PID $pid)"

    # Wait for ready
    local deadline=$((SECONDS + 120))
    while [ $SECONDS -lt $deadline ]; do
        if curl -sf "http://127.0.0.1:$port/health" > /dev/null 2>&1; then
            echo "$name ready"
            return 0
        fi
        sleep 1
    done

    echo "ERROR: $name failed to start within 120s"
    kill "$pid" 2>/dev/null || true
    exit 1
}

mkdir -p results

start_judge "$REWARD_MODEL" "$REWARD_PORT" "reward-anything"
REWARD_PID=$(pgrep -f "llama-server.*--port $REWARD_PORT" || true)

start_judge "$FLOW_MODEL" "$FLOW_PORT" "flow-judge"
FLOW_PID=$(pgrep -f "llama-server.*--port $FLOW_PORT" || true)

echo ""
echo "=== Starting evaluation ==="
echo ""

# Build orchestrator args
ORCH_ARGS=(--config "$CONFIG")
if [ -n "${NITE_MODELS:-}" ]; then
    # shellcheck disable=SC2086
    ORCH_ARGS+=(--models $NITE_MODELS)
fi
if [ -n "${NITE_DIMENSION:-}" ]; then
    ORCH_ARGS+=(--dimension "$NITE_DIMENSION")
fi

# Run evaluation
uv run python -m nite_eval.orchestrator "${ORCH_ARGS[@]}"
EXIT_CODE=$?

echo ""
echo "=== Evaluation complete ==="
echo "Exit code: $EXIT_CODE"
echo "Time: $(date -u '+%Y-%m-%d %H:%M UTC')"

# List generated reports
echo ""
echo "Reports:"
ls -la results/runs/*.md 2>/dev/null || echo "  (none)"

exit $EXIT_CODE
