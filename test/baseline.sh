#!/usr/bin/env bash
# Baseline comparison: run the SAME prompt through FLASH_ATTN and TURBOQUANT
# backends and print both completions side by side.
#
# Only run this AFTER you have stopped any existing server on the target ports.
# Uses port 8010 for the fp baseline (FLASH_ATTN) and 8009 for TurboQuant.
#
# Usage:
#   bash test/baseline.sh                         # defaults
#   bash test/baseline.sh "Hello" 4 3             # prompt, max_tokens, gpu

set -u

MODEL="${MODEL:-/mnt/nvme3n1/g00872988/models/Qwen3-0.6B}"
PROMPT="${1:-Hello}"
MAX_TOKENS="${2:-4}"
GPU="${3:-3}"
MAX_LEN="${MAX_LEN:-4096}"

FP_PORT=8010
TQ_PORT=8009

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
FP_LOG="$ROOT_DIR/baseline_fp.log"
TQ_LOG="$ROOT_DIR/baseline_tq.log"

# --- cleanup: kill any zombie engine cores from previous runs ------------
kill_zombies() {
    pkill -9 -f 'vllm serve'    2>/dev/null || true
    pkill -9 -f 'VllmWorker'    2>/dev/null || true
    pkill -9 -f 'EngineCore'    2>/dev/null || true
    # Give file descriptors a moment to drop before we rebind ports.
    sleep 2
}

# --- cleanup on exit -----------------------------------------------------
cleanup() {
    local code=$?
    if [ -n "${FP_PID:-}" ]; then
        echo "[baseline] stopping FLASH_ATTN serve pid=$FP_PID"
        kill "$FP_PID" 2>/dev/null || true
    fi
    if [ -n "${TQ_PID:-}" ]; then
        echo "[baseline] stopping TURBOQUANT serve pid=$TQ_PID"
        kill "$TQ_PID" 2>/dev/null || true
    fi
    # Make sure nothing lingers across reruns.
    pkill -9 -f 'vllm serve' 2>/dev/null || true
    pkill -9 -f 'EngineCore' 2>/dev/null || true
    exit "$code"
}
trap cleanup EXIT INT TERM

# --- wait-for-health helper ---------------------------------------------
# $1: port, $2: log path, $3: pid (to detect early exit)
wait_healthy() {
    local port="$1" log="$2" pid="$3"
    for _ in $(seq 1 72); do   # up to 6 minutes
        if curl -sf "http://localhost:${port}/health" >/dev/null 2>&1; then
            return 0
        fi
        if ! kill -0 "$pid" 2>/dev/null; then
            echo "[baseline] server on port $port exited early; tail of log:" >&2
            tail -n 40 "$log" >&2
            return 1
        fi
        sleep 5
    done
    echo "[baseline] server on port $port never became healthy" >&2
    tail -n 40 "$log" >&2
    return 1
}

# --- query helper --------------------------------------------------------
query_completion() {
    local port="$1"
    curl -s "http://localhost:${port}/v1/completions" \
        -H 'Content-Type: application/json' \
        -d "{\"model\":\"${MODEL}\",\"prompt\":\"${PROMPT}\",\"max_tokens\":${MAX_TOKENS},\"temperature\":0}"
}

# --- proxy unset (localhost must not go through corp proxy) --------------
unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY

kill_zombies

export CUDA_VISIBLE_DEVICES="$GPU"
echo "[baseline] model=$MODEL prompt='$PROMPT' max_tokens=$MAX_TOKENS gpu=$GPU"
echo "[baseline] fp log -> $FP_LOG"
echo "[baseline] tq log -> $TQ_LOG"
: > "$FP_LOG"
: > "$TQ_LOG"

# ========== 1) FLASH_ATTN baseline =======================================
echo ""
echo "[baseline] starting FLASH_ATTN serve on port $FP_PORT ..."
(
    cd "$ROOT_DIR"
    vllm serve "$MODEL" \
        --port "$FP_PORT" \
        --enforce-eager \
        --attention-backend FLASH_ATTN \
        --max-model-len "$MAX_LEN" \
        --gpu-memory-utilization 0.3 \
        >>"$FP_LOG" 2>&1
) &
FP_PID=$!
echo "[baseline] FLASH_ATTN pid=$FP_PID"
wait_healthy "$FP_PORT" "$FP_LOG" "$FP_PID" || exit 1

echo "[baseline] FLASH_ATTN query:"
FP_OUT=$(query_completion "$FP_PORT")
echo "$FP_OUT"
FP_TEXT=$(echo "$FP_OUT" | python3 -c 'import sys,json;print(json.load(sys.stdin)["choices"][0]["text"])' 2>/dev/null || echo "<parse failed>")

echo "[baseline] stopping FLASH_ATTN"
kill "$FP_PID" 2>/dev/null || true
wait "$FP_PID" 2>/dev/null || true
FP_PID=""
# Make sure the subprocess tree is gone before we spin up the next backend.
pkill -9 -f 'EngineCore' 2>/dev/null || true
sleep 3

# ========== 2) TURBOQUANT =================================================
echo ""
echo "[baseline] starting TURBOQUANT serve on port $TQ_PORT ..."
(
    cd "$ROOT_DIR"
    vllm serve "$MODEL" \
        --port "$TQ_PORT" \
        --enforce-eager \
        --attention-backend TURBOQUANT \
        --max-model-len "$MAX_LEN" \
        --gpu-memory-utilization 0.3 \
        >>"$TQ_LOG" 2>&1
) &
TQ_PID=$!
echo "[baseline] TURBOQUANT pid=$TQ_PID"
wait_healthy "$TQ_PORT" "$TQ_LOG" "$TQ_PID" || exit 1

echo "[baseline] TURBOQUANT query:"
TQ_OUT=$(query_completion "$TQ_PORT")
echo "$TQ_OUT"
TQ_TEXT=$(echo "$TQ_OUT" | python3 -c 'import sys,json;print(json.load(sys.stdin)["choices"][0]["text"])' 2>/dev/null || echo "<parse failed>")

echo ""
echo "========================== SIDE-BY-SIDE =========================="
echo "prompt:       ${PROMPT}"
echo "max_tokens:   ${MAX_TOKENS}  (temperature=0, greedy decoding)"
echo "FLASH_ATTN:   ${FP_TEXT}"
echo "TURBOQUANT:   ${TQ_TEXT}"
echo "=================================================================="
echo "[baseline] full logs at: $FP_LOG  $TQ_LOG"
