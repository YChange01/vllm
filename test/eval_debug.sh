#!/usr/bin/env bash
# Start a TURBOQUANT b=8 server with TQ_DEBUG=1, fire one small NIAH query
# to flush the instrumentation, print the first ~80 [TQ_ATTN] lines.
#
# This script does NOT kill any process -- the vllm serve it launches is
# left running. Clean it up yourself when you are done. The PID is printed
# at the end.
#
# Prerequisite (user manages): port 8009 must be free and no stale
# EngineCore holding GPU memory.
#
# Usage:
#   bash test/eval_debug.sh                    # defaults: bits=8, ctx=64
#   bash test/eval_debug.sh 4                  # bits=4
#   bash test/eval_debug.sh 8 128              # bits=8, ctx=128
#   BITS=4 CTX=128 bash test/eval_debug.sh     # same via env

set -u

MODEL="${MODEL:-/mnt/nvme3n1/g00872988/models/Llama-3.1-8B-Instruct}"
GPU="${GPU:-3}"
MAX_LEN="${MAX_LEN:-4096}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.3}"
PORT="${PORT:-8009}"

BITS="${BITS:-${1:-8}}"
CTX="${CTX:-${2:-64}}"

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
SERVE_LOG="$ROOT_DIR/tq_debug_server.log"
DBG_LOG="$ROOT_DIR/tq_debug_attn.log"

# ---- wait-for-health helper (ps-based, no signals) ----------------------
wait_healthy() {
    local port="$1" log="$2" pid="$3"
    for _ in $(seq 1 72); do   # up to 6 minutes
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

echo "[debug] model=$MODEL bits=$BITS ctx=$CTX gpu=$GPU port=$PORT"
echo "[debug] server log: $SERVE_LOG"
echo "[debug] filtered : $DBG_LOG"
: > "$SERVE_LOG"
: > "$DBG_LOG"

# ---- launch TURBOQUANT server with TQ_DEBUG=1 ---------------------------
# setsid so the process is its own session leader (makes it easy to track
# via pid / kill -TERM -$pid when you manually clean up).
CUDA_VISIBLE_DEVICES="$GPU" \
TQ_DEBUG=1 \
TQ_VERIFY=1 \
TQ_KSTATS=1 \
TURBOQUANT_BITS="$BITS" \
setsid vllm serve "$MODEL" \
    --port "$PORT" \
    --enforce-eager \
    --attention-backend TURBOQUANT \
    --max-model-len "$MAX_LEN" \
    --gpu-memory-utilization "$GPU_MEM_UTIL" \
    >>"$SERVE_LOG" 2>&1 &
SERVE_PID=$!
echo "[debug] server pid=$SERVE_PID  (NOT auto-stopped; you manage it)"

if ! wait_healthy "$PORT" "$SERVE_LOG" "$SERVE_PID"; then
    echo "[debug] server startup failed; see $SERVE_LOG" >&2
    exit 1
fi
echo "[debug] server healthy"

# ---- fire one small NIAH trial to flush the instrumentation -------------
python3 "$ROOT_DIR/test/eval_niah.py" \
    --endpoint "http://localhost:${PORT}" \
    --model "$MODEL" \
    --ctx-lens "$CTX" \
    --positions 0.5 \
    --trials 1 \
    --max-tokens 16 \
    --request-timeout 1800 \
    --tag "tq_b${BITS}_dbg" \
    2>&1 | tee -a "$SERVE_LOG"

# ---- extract TQ_ATTN debug lines ----------------------------------------
echo ""
echo "=========================== TQ_ATTN debug =========================="
grep -E '\[TQ_ATTN|\[TQ_VERIFY|\[TQ_KSTATS' "$SERVE_LOG" > "$DBG_LOG" || true
if [ -s "$DBG_LOG" ]; then
    head -n 80 "$DBG_LOG"
else
    echo "[debug] no [TQ_ATTN] lines found; search tail of $SERVE_LOG for errors" >&2
    tail -n 40 "$SERVE_LOG" >&2
fi
echo "===================================================================="

echo ""
echo "[debug] done."
echo "[debug] full server log: $SERVE_LOG"
echo "[debug] filtered TQ_ATTN: $DBG_LOG"
echo "[debug] server still running:"
echo "[debug]   pid=$SERVE_PID  port=$PORT"
echo "[debug] stop it yourself when finished (kill -TERM -$SERVE_PID for the group)."
