#!/usr/bin/env bash
# Compare two GGUF quants of the same base model.
#
# Steps:
#   1. Dump + diff GGUF metadata (chat template, tokenizer hash, per-tensor quant types)
#   2. xxh64 file hash for each
#   3. Determinism check: 2 identical chat completions at temp=0, diff outputs
#   4. Perplexity benchmark on wikitext-2-raw (downloads if missing)
#
# Usage:
#   scripts/compare_quants.sh                                # compares the two Qwen3.6 quants
#   scripts/compare_quants.sh --skip-determinism             # skip step 3
#   scripts/compare_quants.sh --skip-perplexity              # skip step 4
#   scripts/compare_quants.sh --system-suffix ""             # for non-thinking models
#   scripts/compare_quants.sh PATH_A LABEL_A PATH_B LABEL_B  # compare arbitrary pair
#
# Defaults: system-suffix is "/no_think" (correct for the default Qwen3.6 pair).
# Override to "" for non-thinking models, or to any other chat-template trigger.
#
# Requires: $LLAMA_SERVER_BIN and $TARGET_GPU_UUID from .env.
# Stops anything it starts on exit.

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_DIR"

if [ -f .env ]; then
    set -a
    # shellcheck disable=SC1091
    . .env
    set +a
fi

LLAMA_SERVER="${LLAMA_SERVER_BIN:?LLAMA_SERVER_BIN not set in .env}"
LLAMA_BIN_DIR="$(dirname "$LLAMA_SERVER")"
GPU_UUID="${TARGET_GPU_UUID:?TARGET_GPU_UUID not set in .env}"
GGUF_DIR="${GGUF_DIR:-$LLAMA_BIN_DIR}"

# --- Defaults: the two Qwen3.6 quants ---
MODEL_A="$GGUF_DIR/Qwen3.6-35B-A3B-UD-Q4_K_S.gguf"
LABEL_A="unsloth-UD-Q4_K_S"
MODEL_B="$GGUF_DIR/Qwen3.6-35B-A3B-Q4_K_M.gguf"
LABEL_B="strix-Q4_K_M"

SKIP_DET=0
SKIP_PPL=0
SYSTEM_SUFFIX="/no_think"  # appended to the determinism-check system prompt
POSITIONAL=()
while [ $# -gt 0 ]; do
    case "$1" in
        --skip-determinism) SKIP_DET=1; shift ;;
        --skip-perplexity)  SKIP_PPL=1; shift ;;
        --system-suffix)    SYSTEM_SUFFIX="$2"; shift 2 ;;
        -h|--help)
            sed -n '2,21p' "$0"; exit 0 ;;
        *) POSITIONAL+=("$1"); shift ;;
    esac
done

if [ "${#POSITIONAL[@]}" -eq 4 ]; then
    MODEL_A="${POSITIONAL[0]}"; LABEL_A="${POSITIONAL[1]}"
    MODEL_B="${POSITIONAL[2]}"; LABEL_B="${POSITIONAL[3]}"
elif [ "${#POSITIONAL[@]}" -ne 0 ]; then
    echo "ERROR: pass either 0 args (use defaults) or 4 (PATH_A LABEL_A PATH_B LABEL_B)"
    exit 2
fi

for f in "$MODEL_A" "$MODEL_B"; do
    if [ ! -f "$f" ]; then
        echo "ERROR: model file not found: $f"
        exit 1
    fi
done

OUT="results/quant-compare/$(date -u +%Y%m%d-%H%M%S)"
mkdir -p "$OUT"
echo "=== Output: $OUT ==="
echo

# ────────────────────────────────────────────────────────────
# Step 1: GGUF metadata
# ────────────────────────────────────────────────────────────
echo "=== Step 1: GGUF metadata ==="
# `gguf` is not in nite-eval's runtime deps — pull it ad-hoc for this script.
uv run --with gguf python scripts/gguf_meta_diff.py "$MODEL_A" "$MODEL_B" \
    --labels "$LABEL_A" "$LABEL_B" 2>&1 | tee "$OUT/01-metadata.txt"

# ────────────────────────────────────────────────────────────
# Step 2: file-level xxh64 hash (sanity, not strictly needed)
# ────────────────────────────────────────────────────────────
echo
echo "=== Step 2: file hashes (xxh64, no per-layer) ==="
{
    for pair in "$MODEL_A:$LABEL_A" "$MODEL_B:$LABEL_B"; do
        m="${pair%%:*}"; l="${pair##*:}"
        echo "--- $l: $(basename "$m") ---"
        "$LLAMA_BIN_DIR/llama-gguf-hash" --xxh64 --no-layer "$m" 2>&1 || true
    done
} | tee "$OUT/02-hashes.txt"

# ────────────────────────────────────────────────────────────
# Step 3: determinism check
# ────────────────────────────────────────────────────────────
DET_PORT="${DET_PORT:-9078}"

start_server() {
    local model="$1" label="$2"
    if curl -sf "http://127.0.0.1:$DET_PORT/health" >/dev/null 2>&1; then
        echo "ERROR: port $DET_PORT already in use; set DET_PORT=<free port> and retry"
        return 1
    fi
    env CUDA_VISIBLE_DEVICES="$GPU_UUID" "$LLAMA_SERVER" \
        -m "$model" --port "$DET_PORT" -ngl 999 --ctx-size 4096 \
        -fa on --cache-type-k q8_0 --cache-type-v q8_0 \
        --no-webui > "$OUT/03-server-$label.log" 2>&1 &
    SERVER_PID=$!
    local deadline=$((SECONDS + 180))
    while [ $SECONDS -lt $deadline ]; do
        if curl -sf "http://127.0.0.1:$DET_PORT/health" >/dev/null 2>&1; then
            return 0
        fi
        sleep 1
    done
    echo "ERROR: server failed to become healthy within 180s"
    kill "$SERVER_PID" 2>/dev/null || true
    return 1
}

stop_server() {
    if [ -n "${SERVER_PID:-}" ] && kill -0 "$SERVER_PID" 2>/dev/null; then
        kill "$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
    fi
    SERVER_PID=""
}
trap 'stop_server' EXIT INT TERM

determinism_run() {
    local model="$1" label="$2"
    echo "--- determinism: $label (system-suffix='$SYSTEM_SUFFIX') ---"
    start_server "$model" "$label" || return 1

    # Build payload via python so SYSTEM_SUFFIX gets escaped correctly.
    # Reasoning models like Qwen3.x emit empty `content` (everything goes into
    # `reasoning_content`) unless a chat-template trigger like /no_think is in
    # the system prompt — that's why this isn't a literal heredoc.
    local payload
    payload="$(SUFFIX="$SYSTEM_SUFFIX" python3 -c '
import json, os
sys_prompt = ("You are a helpful assistant. " + os.environ["SUFFIX"]).strip()
print(json.dumps({
    "model": "x",
    "temperature": 0.0,
    "top_p": 1.0,
    "seed": 1,
    "max_tokens": 256,
    "messages": [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": "List the first 12 prime numbers separated by commas. No other text."},
    ],
}))')"

    for i in 1 2; do
        curl -sf "http://127.0.0.1:$DET_PORT/v1/chat/completions" \
            -H 'Content-Type: application/json' -d "$payload" \
            | python3 -c 'import json,sys; print(json.load(sys.stdin)["choices"][0]["message"]["content"])' \
            > "$OUT/03-det-$label-run$i.txt"
    done

    if diff -q "$OUT/03-det-$label-run1.txt" "$OUT/03-det-$label-run2.txt" >/dev/null; then
        echo "  DETERMINISTIC (identical across 2 runs at temp=0)"
        cat "$OUT/03-det-$label-run1.txt" | head -3
    else
        echo "  NON-DETERMINISTIC — outputs diverge:"
        diff -u "$OUT/03-det-$label-run1.txt" "$OUT/03-det-$label-run2.txt" | head -30
    fi
    stop_server
}

if [ "$SKIP_DET" = "0" ]; then
    echo
    echo "=== Step 3: Determinism check (temp=0, seed=1, same prompt twice) ==="
    determinism_run "$MODEL_A" "$LABEL_A"
    determinism_run "$MODEL_B" "$LABEL_B"
else
    echo
    echo "=== Step 3: Determinism — SKIPPED (--skip-determinism) ==="
fi

# ────────────────────────────────────────────────────────────
# Step 4: perplexity
# ────────────────────────────────────────────────────────────
WIKI_DIR="results/quant-compare/_data"
WIKI="$WIKI_DIR/wiki.test.raw"

ensure_wiki() {
    if [ -s "$WIKI" ]; then
        echo "Reusing cached $WIKI ($(wc -c < "$WIKI") bytes)"
        return 0
    fi
    mkdir -p "$WIKI_DIR"
    echo "Fetching wikitext-2-raw test split via HuggingFace datasets..."
    if uv run --with 'datasets>=2.14' --quiet python - "$WIKI" <<'PY'
import sys
from datasets import load_dataset
out = sys.argv[1]
ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
with open(out, "w", encoding="utf-8") as f:
    for row in ds:
        f.write(row["text"])
print(f"  wrote {len(ds)} rows -> {out}")
PY
    then
        echo "  ok ($(wc -c < "$WIKI") bytes)"
        return 0
    fi
    echo "ERROR: could not fetch wikitext-2-raw; place it at $WIKI manually"
    return 1
}

perplexity_run() {
    local model="$1" label="$2"
    echo "--- perplexity: $label ---"
    env CUDA_VISIBLE_DEVICES="$GPU_UUID" "$LLAMA_BIN_DIR/llama-perplexity" \
        -m "$model" -f "$WIKI" \
        -c 2048 -b 512 -ngl 999 \
        -fa on --cache-type-k q8_0 --cache-type-v q8_0 \
        2>&1 | tee "$OUT/04-ppl-$label.log" \
        | grep --line-buffered -E '\[[0-9]+\]|^Final estimate|perplexity:'
}

if [ "$SKIP_PPL" = "0" ]; then
    echo
    echo "=== Step 4: Perplexity (wikitext-2-raw, ctx=2048, batch=512) ==="
    if ensure_wiki; then
        perplexity_run "$MODEL_A" "$LABEL_A"
        perplexity_run "$MODEL_B" "$LABEL_B"
    fi
else
    echo
    echo "=== Step 4: Perplexity — SKIPPED (--skip-perplexity) ==="
fi

# ────────────────────────────────────────────────────────────
# Final summary
# ────────────────────────────────────────────────────────────
echo
echo "=== Final summary ==="
{
    echo "[metadata diff]"
    grep -E '^!! ' "$OUT/01-metadata.txt" || echo "  (no metadata differences)"
    if [ "$SKIP_PPL" = "0" ]; then
        echo
        echo "[perplexity]"
        for l in "$LABEL_A" "$LABEL_B"; do
            f="$OUT/04-ppl-$l.log"
            if [ -f "$f" ]; then
                line="$(grep -E '^Final estimate|perplexity: ' "$f" | tail -1)"
                printf "  %-20s  %s\n" "$l" "${line:-<not found>}"
            fi
        done
    fi
} | tee "$OUT/SUMMARY.txt"

echo
echo "Done. Full artifacts in $OUT"
