#!/usr/bin/env bash
# Start a TurboQuant server and fire a single small NIAH trial.
# Usage:
#   bash test/eval_debug.sh                     # algo=prod bits=8 ctx=64
#   bash test/eval_debug.sh mse 8 64            # algo, bits, ctx
#   bash test/eval_debug.sh prod 4 128
#
# Server is NOT auto-stopped. Its PID is printed; clean it up yourself.

set -u

MODEL="${MODEL:-/mnt/nvme3n1/g00872988/models/Llama-3.1-8B-Instruct}"
GPU="${GPU:-3}"
MAX_LEN="${MAX_LEN:-4096}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.3}"
PORT="${PORT:-8009}"

ALGO="${ALGO:-${1:-prod}}"
BITS="${BITS:-${2:-8}}"
CTX="${CTX:-${3:-64}}"

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
TS="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="$ROOT_DIR/logs/eval_debug_$TS"
mkdir -p "$LOG_DIR"
ln -sfn "$LOG_DIR" "$ROOT_DIR/logs/eval_debug_latest"
SERVE_LOG="$LOG_DIR/server.log"
DBG_LOG="$LOG_DIR/turboquant_debug.log"

wait_healthy() {
    local port="$1" log="$2" pid="$3"
    for _ in $(seq 1 72); do
        if curl -sf "http://localhost:${port}/health" >/dev/null 2>&1; then
            return 0
        fi
        if ! ps -p "$pid" >/dev/null 2>&1; then
            echo "[debug] server exited early; tail of log:" >&2
            tail -n 40 "$log" >&2
            return 1
        fi
        sleep 5
    done
    echo "[debug] server never became healthy" >&2
    tail -n 40 "$log" >&2
    return 1
}

echo "[debug] model=$MODEL algo=$ALGO bits=$BITS ctx=$CTX gpu=$GPU port=$PORT"
echo "[debug] server log: $SERVE_LOG"
: > "$SERVE_LOG"

CUDA_VISIBLE_DEVICES="$GPU" \
TURBOQUANT_ALGO="$ALGO" \
TURBOQUANT_BITS="$BITS" \
TURBOQUANT_DEBUG_LOG="$DBG_LOG" \
setsid vllm serve "$MODEL" \
    --port "$PORT" \
    --enforce-eager \
    --attention-backend TURBOQUANT \
    --max-model-len "$MAX_LEN" \
    --gpu-memory-utilization "$GPU_MEM_UTIL" \
    >>"$SERVE_LOG" 2>&1 &
SERVE_PID=$!
echo "[debug] server pid=$SERVE_PID (NOT auto-stopped)"

if ! wait_healthy "$PORT" "$SERVE_LOG" "$SERVE_PID"; then
    exit 1
fi
echo "[debug] server healthy"

python3 "$ROOT_DIR/test/eval_niah.py" \
    --endpoint "http://localhost:${PORT}" \
    --model "$MODEL" \
    --ctx-lens "$CTX" \
    --positions 0.5 \
    --trials 1 \
    --max-tokens 16 \
    --request-timeout 1800 \
    --tag "tq_${ALGO}_b${BITS}" \
    2>&1 | tee -a "$SERVE_LOG"

echo ""
echo "[debug] done. server still running; stop with: kill -TERM -$SERVE_PID"
