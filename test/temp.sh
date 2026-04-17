#!/usr/bin/env bash
# One-shot diagnostic scratchpad. Overwritten every time.
#
# Current probe: localize bug between Triton kernel impl vs algorithm.
# Run mse b=8 with TURBOQUANT_PYREF=1 (pure-PyTorch fp32 ref attend
# kernel) on the long prompt. Compare text output to the broken
# Triton-kernel run.
#
#   PYREF correct + Triton broken  -> bug in Triton kernel implementation
#                                     (memory layout, static_range gating,
#                                     flash accumulation, dtype handling)
#   PYREF broken too               -> bug in algorithm itself (rotation,
#                                     codebook reconstruction, inv_d)
#
# NOTE: PYREF is SLOW (Python loops). Expect ~minutes for 26-token prefill
# with 32 layers. Only use for diagnostic, not production.

set -u

cd "$(dirname "$0")/.."

GPU="${GPU:-3}"
PROMPT="${PROMPT:-Machine learning has transformed many fields over the past decade with deep neural networks achieving remarkable performance on natural language understanding and speech synthesis.}"
MAX_TOKENS="${MAX_TOKENS:-4}"

ROOT_DIR="$(pwd)"
TS="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="$ROOT_DIR/logs/pyref_$TS"
mkdir -p "$LOG_DIR"
ln -sfn "$LOG_DIR" "$ROOT_DIR/logs/pyref_latest"
SERVE_LOG="$LOG_DIR/server.log"
DBG_LOG="$LOG_DIR/turboquant_debug.log"

echo "[temp] GPU=$GPU MAX_TOKENS=$MAX_TOKENS"
echo "[temp] PROMPT=\"$PROMPT\""
echo "[temp] log dir: $LOG_DIR"
echo ""

PORT=8009
CURRENT_PGID=""
cleanup() {
    local code=$?
    if [ -n "$CURRENT_PGID" ]; then
        echo "[temp] cleanup pgid $CURRENT_PGID"
        kill -TERM -"$CURRENT_PGID" 2>/dev/null || true
        wait "$CURRENT_PGID" 2>/dev/null || true
    fi
    exit "$code"
}
trap cleanup EXIT INT TERM

echo "=========================================================="
echo "Launching mse b=8 with TURBOQUANT_PYREF=1"
echo "=========================================================="
: > "$SERVE_LOG"
env CUDA_VISIBLE_DEVICES="$GPU" \
    TURBOQUANT_ALGO=mse \
    TURBOQUANT_BITS=8 \
    TURBOQUANT_PYREF=1 \
    TURBOQUANT_DEBUG_LOG="$DBG_LOG" \
    setsid vllm serve /mnt/nvme3n1/g00872988/models/Llama-3.1-8B-Instruct \
        --port "$PORT" \
        --enforce-eager \
        --attention-backend TURBOQUANT \
        --max-model-len 4096 \
        --gpu-memory-utilization 0.3 \
        >>"$SERVE_LOG" 2>&1 &
CURRENT_PGID=$!

echo "[temp] waiting for /health (PYREF takes a while to compile)..."
for _ in $(seq 1 96); do
    if curl -sf "http://localhost:${PORT}/health" >/dev/null 2>&1; then
        echo "[temp] healthy"
        break
    fi
    if ! ps -p "$CURRENT_PGID" >/dev/null 2>&1; then
        echo "[temp] server exited early; tail of log:" >&2
        tail -n 60 "$SERVE_LOG" >&2
        exit 1
    fi
    sleep 5
done

echo ""
echo "[temp] sending completion request (this is SLOW under PYREF)..."
RESP=$(curl -s "http://localhost:${PORT}/v1/completions" \
    -H 'Content-Type: application/json' \
    -d "{\"model\":\"/mnt/nvme3n1/g00872988/models/Llama-3.1-8B-Instruct\",\"prompt\":\"${PROMPT}\",\"max_tokens\":${MAX_TOKENS},\"temperature\":0}" \
    --max-time 1800)
echo "[temp] raw: $RESP"
TEXT=$(echo "$RESP" | python3 -c 'import sys,json;print(json.load(sys.stdin)["choices"][0]["text"])' 2>/dev/null || echo "<parse failed>")

kill -TERM -"$CURRENT_PGID" 2>/dev/null || true
wait "$CURRENT_PGID" 2>/dev/null || true
CURRENT_PGID=""

echo ""
echo "=========================================================="
echo "RESULT"
echo "=========================================================="
echo "PROMPT:               $PROMPT"
echo "FLASH_ATTN expected:  However, the field"
echo "TRITON kernel saw:    ://owowow                          (broken)"
echo "PYREF mse b=8 saw:    $TEXT"
echo ""
echo "If PYREF text is reasonable -> Triton kernel bug"
echo "If PYREF text is also gibberish -> algorithm bug"
