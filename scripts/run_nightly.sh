#!/usr/bin/env bash
# Nightly evaluation runner for nite-eval
#
# Starts all servers (target + judges), runs the eval pipeline, generates
# report, and cleans up all servers on exit.
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
#   - UV environment set up (uv sync)
#   - GGUF models in expected paths
#
# Environment variables:
#   NITE_MODELS       Space-separated model list (default: all from config)
#   NITE_DIMENSION    Filter to one dimension (default: all)
#   NITE_CONFIG       Config path (default: config/eval_config.yaml)
#   NITE_TARGET_GPU   GPU ID for target llama-swap (default: 1)
#   NITE_JUDGE_GPU    GPU ID for judge servers (default: 2)

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_DIR"

# Load .env if present (paths, GPUs, ports)
if [ -f "$PROJECT_DIR/.env" ]; then
    set -a
    # shellcheck disable=SC1091
    . "$PROJECT_DIR/.env"
    set +a
fi

# Config
TARGET_GPU="${NITE_TARGET_GPU:-${TARGET_GPU:-1}}"
JUDGE_GPU="${NITE_JUDGE_GPU:-${JUDGE_GPU:-2}}"
CONFIG="${NITE_CONFIG:-config/eval_config.yaml}"
LLAMA_SERVER="${LLAMA_SERVER_BIN:?LLAMA_SERVER_BIN not set — copy .env.example to .env}"
LLAMA_SWAP="${LLAMA_SWAP_BIN:?LLAMA_SWAP_BIN not set — copy .env.example to .env}"
LLAMA_SWAP_CONFIG="config/llama_swap_config.yaml"
JUDGE_MODEL_DIR="${JUDGE_MODEL_DIR:?JUDGE_MODEL_DIR not set — copy .env.example to .env}"
REWARD_MODEL="${JUDGE_MODEL_DIR%/}/${REWARD_MODEL_FILE:-RewardAnything-8B-v1.Q6_K.gguf}"
FLOW_MODEL="${JUDGE_MODEL_DIR%/}/${FLOW_MODEL_FILE:-Flow-Judge-v0.1.Q6_K.gguf}"
TARGET_PORT="${TARGET_PORT:-9070}"
REWARD_PORT="${REWARD_PORT:-9091}"
FLOW_PORT="${FLOW_PORT:-9092}"

if [ ! -f "$LLAMA_SWAP_CONFIG" ]; then
    echo "ERROR: $LLAMA_SWAP_CONFIG not found."
    echo "Copy config/llama_swap_config.example.yaml to $LLAMA_SWAP_CONFIG and edit paths."
    exit 1
fi

# Track what we started (only clean up what we own)
STARTED_TARGET=""
STARTED_REWARD=""
STARTED_FLOW=""

cleanup() {
    echo ""
    echo "=== Cleaning up ==="
    for desc_pid in \
        "llama-swap:$STARTED_TARGET" \
        "reward-anything:$STARTED_REWARD" \
        "flow-judge:$STARTED_FLOW"; do
        local desc="${desc_pid%%:*}"
        local pid="${desc_pid##*:}"
        if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
            echo "Stopping $desc (PID $pid)"
            kill "$pid" 2>/dev/null || true
            wait "$pid" 2>/dev/null || true
        fi
    done
    echo "Cleanup complete"
}
trap cleanup EXIT INT TERM

echo "=== nite-eval nightly runner ==="
echo "Time: $(date -u '+%Y-%m-%d %H:%M UTC')"
echo "Project: $PROJECT_DIR"
echo "Config: $CONFIG"
echo ""

mkdir -p results

# --- Verify GPU assignment before loading anything ---
# A stale llama-server from a previous run holding the target GPU, or judges
# pinned to the same card as the model under evaluation, silently ruins a run:
# it completes, writes scores, and the latency numbers are garbage.
echo "Checking GPU assignment..."
if [ -z "${TARGET_GPU_UUID:-}" ] || [ -z "${JUDGE_GPU_UUID:-}" ]; then
    echo "ERROR: TARGET_GPU_UUID and JUDGE_GPU_UUID must be set in .env"
    echo "       Available GPUs:"
    nvidia-smi -L | sed 's/^/         /'
    exit 1
fi
if [ "$TARGET_GPU_UUID" = "$JUDGE_GPU_UUID" ]; then
    echo "ERROR: TARGET_GPU_UUID and JUDGE_GPU_UUID are the same GPU."
    echo "       Judges would contend with the model under evaluation for VRAM."
    exit 1
fi
for uuid in "$TARGET_GPU_UUID" "$JUDGE_GPU_UUID"; do
    if ! nvidia-smi -L | grep -q "$uuid"; then
        echo "ERROR: GPU $uuid not present on this host."
        nvidia-smi -L | sed 's/^/         /'
        exit 1
    fi
done
# Warn about pre-existing llama processes — they hold VRAM we are about to need.
# -x matches the process NAME; -f would also match this script's own shell
# wrapper, whose command line contains the pattern string, so every run would
# warn about a stale process that does not exist.
# pgrep -c prints the count AND exits 1 when it is zero, hence the `|| true`.
STALE_SERVER=$(pgrep -xc llama-server 2>/dev/null) || true
STALE_SWAP=$(pgrep -xc llama-swap 2>/dev/null) || true
STALE=$(( ${STALE_SERVER:-0} + ${STALE_SWAP:-0} ))
if [ "$STALE" -gt 0 ]; then
    echo "WARNING: $STALE llama-server/llama-swap process(es) already running."
    echo "         They hold VRAM and may be pinned to the wrong GPU:"
    nvidia-smi --query-compute-apps=pid,gpu_uuid,used_memory --format=csv,noheader | sed 's/^/         /'
fi
echo "GPU assignment OK (target=${TARGET_GPU_UUID}, judge=${JUDGE_GPU_UUID})"

# --- Start target llama-swap if not running ---
echo "Checking target server on :$TARGET_PORT..."
if curl -sf "http://127.0.0.1:$TARGET_PORT/health" > /dev/null 2>&1; then
    echo "Target llama-swap already running on :$TARGET_PORT"
else
    echo "Starting llama-swap on :$TARGET_PORT..."
    "$LLAMA_SWAP" \
        --config "$LLAMA_SWAP_CONFIG" --listen ":$TARGET_PORT" \
        > "results/llama-swap.log" 2>&1 &
    STARTED_TARGET=$!
    echo "llama-swap started (PID $STARTED_TARGET)"

    # Wait for ready
    deadline=$((SECONDS + 30))
    while [ $SECONDS -lt $deadline ]; do
        if curl -sf "http://127.0.0.1:$TARGET_PORT/health" > /dev/null 2>&1; then
            echo "Target llama-swap ready"
            break
        fi
        sleep 1
    done
    if ! curl -sf "http://127.0.0.1:$TARGET_PORT/health" > /dev/null 2>&1; then
        echo "ERROR: llama-swap failed to start within 30s"
        exit 1
    fi
fi

# --- Start judge servers if not running ---
start_judge() {
    local model_path="$1"
    local port="$2"
    local name="$3"

    if curl -sf "http://127.0.0.1:$port/health" > /dev/null 2>&1; then
        echo "$name already running on :$port"
        _STARTED_PID=""
        return 0
    fi

    echo "Starting $name on :$port (GPU $JUDGE_GPU)..."
    CUDA_VISIBLE_DEVICES="${JUDGE_GPU_UUID:-${NITE_JUDGE_UUID:-$JUDGE_GPU}}" "$LLAMA_SERVER" \
        -m "$model_path" \
        --port "$port" \
        -ngl 999 --ctx-size 4096 -np 1 --no-webui \
        > "results/${name}.log" 2>&1 &

    _STARTED_PID=$!
    echo "$name started (PID $_STARTED_PID)"

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
    kill "$_STARTED_PID" 2>/dev/null || true
    _STARTED_PID=""
    exit 1
}

start_judge "$REWARD_MODEL" "$REWARD_PORT" "reward-anything"
STARTED_REWARD="$_STARTED_PID"

start_judge "$FLOW_MODEL" "$FLOW_PORT" "flow-judge"
STARTED_FLOW="$_STARTED_PID"

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
