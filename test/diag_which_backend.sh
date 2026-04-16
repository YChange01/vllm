#!/usr/bin/env bash
# What attention backend does vllm ACTUALLY use when we ask for TURBOQUANT?
#
# Module-load marker + stderr fallback never fired in diag_bypass.sh,
# but diag_install.sh shows the module imports fine and file location
# matches the repo. So the server process is selecting a different
# backend (silent fallback) despite our --attention-backend TURBOQUANT.
# Grep the server startup log for the backend-selection line(s).

set -u

GPU="${GPU:-3}"
MODEL="${MODEL:-/mnt/nvme3n1/g00872988/models/Llama-3.1-8B-Instruct}"
PORT="${PORT:-8009}"
ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
LOG="$ROOT_DIR/diag_which_backend.log"

CURRENT_PGID=""
cleanup() {
    local code=$?
    if [ -n "$CURRENT_PGID" ]; then
        kill -TERM -"$CURRENT_PGID" 2>/dev/null || true
        wait "$CURRENT_PGID" 2>/dev/null || true
    fi
    exit "$code"
}
trap cleanup EXIT INT TERM

: > "$LOG"
echo "[which] launching vllm serve --attention-backend TURBOQUANT ..."
echo "[which] log: $LOG"

CUDA_VISIBLE_DEVICES="$GPU" \
TURBOQUANT_BYPASS=1 \
setsid vllm serve "$MODEL" \
    --port "$PORT" \
    --enforce-eager \
    --attention-backend TURBOQUANT \
    --max-model-len 4096 \
    --gpu-memory-utilization 0.3 \
    >"$LOG" 2>&1 &
CURRENT_PGID=$!

# Wait up to ~60s for startup lines to land.
for _ in $(seq 1 30); do
    if grep -q "Selected attention\|Using .* backend\|attention.*backend\|TURBOQUANT_DBG" "$LOG" 2>/dev/null; then
        break
    fi
    if ! ps -p "$CURRENT_PGID" >/dev/null 2>&1; then
        break
    fi
    sleep 2
done

# Give it a couple more seconds for the logs to flush.
sleep 3

kill -TERM -"$CURRENT_PGID" 2>/dev/null || true
wait "$CURRENT_PGID" 2>/dev/null || true
CURRENT_PGID=""

echo ""
echo "==================================================================="
echo "[which] grepping backend / TURBOQUANT_DBG mentions in $LOG:"
echo "==================================================================="
grep -a -nE "TURBOQUANT_DBG|[Aa]ttention backend|[Ss]elected.*ttn|Using.*ackend|Fallback|backend=|FLASH_ATTN|TurboQuant" "$LOG" \
    | head -40 || echo "(no matches)"

echo ""
echo "==================================================================="
echo "[which] last 40 lines of log (context for any error):"
echo "==================================================================="
tail -40 "$LOG"
