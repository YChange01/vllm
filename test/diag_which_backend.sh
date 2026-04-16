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

# Wait up to ~6 minutes for the server to finish initializing. We want
# ALL attention-related log lines, which only land after model load.
echo "[which] waiting for /health ..."
for _ in $(seq 1 72); do
    if curl -sf "http://localhost:${PORT}/health" >/dev/null 2>&1; then
        echo "[which] server healthy"
        break
    fi
    if ! ps -p "$CURRENT_PGID" >/dev/null 2>&1; then
        echo "[which] server exited early"
        break
    fi
    sleep 5
done

# Let any final initialization messages flush.
sleep 5

kill -TERM -"$CURRENT_PGID" 2>/dev/null || true
wait "$CURRENT_PGID" 2>/dev/null || true
CURRENT_PGID=""

# Output tailored to stay short. Each grep capped at 200 chars * 10 lines.
_short() { cut -c1-200 | head -"${1:-10}"; }

echo ""
echo "==================================================================="
echo "[which] log lines: $(wc -l < "$LOG")  bytes: $(wc -c < "$LOG")"
echo "==================================================================="

echo ""
echo "-- TURBOQUANT_DBG lines --"
grep -na "TURBOQUANT_DBG" "$LOG" | _short 10
echo "count: $(grep -c "TURBOQUANT_DBG" "$LOG" || true)"

echo ""
echo "-- backend / attention selection --"
grep -na -iE "attention backend|selected attn|using.*backend|attn backend|init attention" "$LOG" | _short 10

echo ""
echo "-- Turbo/Quant mentions --"
grep -na -i "turbo\|quant" "$LOG" | _short 10

echo ""
echo "-- Errors / Warnings / Fallbacks --"
grep -na -iE "error|warn|fail|traceback|fallback|notimplemented" "$LOG" | _short 20

echo ""
echo "-- Log path (upload if sections above are inconclusive) --"
echo "$LOG"
