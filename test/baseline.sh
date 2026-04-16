#!/usr/bin/env bash
# Baseline comparison: run the SAME prompt through FLASH_ATTN and TURBOQUANT
# backends and print both completions side by side.
#
# This script never kills any process. It only starts two background vllm
# servers (FLASH_ATTN on $FP_PORT, TURBOQUANT on $TQ_PORT) and queries both.
# Both servers stay alive after the script exits -- clean them up yourself
# when you are done. Their PIDs are printed so you can find them.
#
# Prerequisites (YOU manage these):
#   - ports $FP_PORT (8010) and $TQ_PORT (8009) are free
#   - GPU is free of stale engines (prior EngineCore processes killed)
#   - proxy env vars unset if localhost must bypass corp proxy
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

# --- wait-for-health helper ---------------------------------------------
# $1: port, $2: log path, $3: pid (for liveness check via ps, no signals)
wait_healthy() {
    local port="$1" log="$2" pid="$3"
    for _ in $(seq 1 72); do   # up to 6 minutes
        if curl -sf "http://localhost:${port}/health" >/dev/null 2>&1; then
            return 0
        fi
        if ! ps -p "$pid" >/dev/null 2>&1; then
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

export CUDA_VISIBLE_DEVICES="$GPU"
echo "[baseline] model=$MODEL prompt='$PROMPT' max_tokens=$MAX_TOKENS gpu=$GPU"
echo "[baseline] fp log -> $FP_LOG"
echo "[baseline] tq log -> $TQ_LOG"
: > "$FP_LOG"
: > "$TQ_LOG"

# ========== 1) FLASH_ATTN baseline (port 8010) ===========================
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
echo "[baseline] FLASH_ATTN pid=$FP_PID  (NOT auto-stopped; you manage it)"
wait_healthy "$FP_PORT" "$FP_LOG" "$FP_PID" || exit 1

echo "[baseline] FLASH_ATTN query:"
FP_OUT=$(query_completion "$FP_PORT")
echo "$FP_OUT"
FP_TEXT=$(echo "$FP_OUT" | python3 -c 'import sys,json;print(json.load(sys.stdin)["choices"][0]["text"])' 2>/dev/null || echo "<parse failed>")

# ========== 2) TURBOQUANT (port 8009) ====================================
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
echo "[baseline] TURBOQUANT pid=$TQ_PID  (NOT auto-stopped; you manage it)"
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
echo "[baseline] both servers still running:"
echo "[baseline]   FLASH_ATTN  pid=$FP_PID  port=$FP_PORT"
echo "[baseline]   TURBOQUANT  pid=$TQ_PID  port=$TQ_PORT"
