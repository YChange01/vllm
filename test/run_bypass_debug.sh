#!/usr/bin/env bash
# Run the BYPASS diagnostic with TURBOQUANT_DEBUG=1 and dump the log.
#
# Logs layer-0 first store + first forward payload (slot_mapping, seq_lens,
# query_start_loc, block_table, scale, output.shape, q first values) so we
# can tell which vLLM-side metadata disagrees with our assumptions.
#
# Usage:
#   bash test/run_bypass_debug.sh             # default gpu=3
#   GPU=0 bash test/run_bypass_debug.sh
#   bash test/run_bypass_debug.sh "Hello" 4   # prompt, max_tokens

set -u

GPU="${GPU:-3}"
PROMPT="${1:-Hello}"
MAX_TOKENS="${2:-4}"
DEBUG_LOG="${TURBOQUANT_DEBUG_LOG:-/tmp/turboquant_debug.log}"

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

echo "[run-bypass] repo: $ROOT_DIR"
echo "[run-bypass] pulling latest ..."
git pull --ff-only
echo ""
echo "[run-bypass] HEAD:"
git log --oneline -3
echo ""

# Clear any previous debug log so we only see this run's output.
for p in "$DEBUG_LOG" \
         /tmp/turboquant_debug.log \
         /mnt/nvme3n1/g00872988/turboquant/turboquant_debug.log \
         "$ROOT_DIR/turboquant_debug.log"; do
    rm -f "$p" 2>/dev/null || true
done
echo "[run-bypass] cleared all candidate debug log paths"

# Run BYPASS with debug logging on.
echo "[run-bypass] launching diag_bypass.sh with TURBOQUANT_DEBUG=1 ..."
CUDA_VISIBLE_DEVICES="$GPU" \
TURBOQUANT_BYPASS=1 \
TURBOQUANT_DEBUG=1 \
TURBOQUANT_DEBUG_LOG="$DEBUG_LOG" \
    bash test/diag_bypass.sh "$PROMPT" "$MAX_TOKENS" "$GPU"

echo ""
echo "==================================================================="
echo "[run-bypass] candidate debug log paths:"
for p in "$DEBUG_LOG" \
         /tmp/turboquant_debug.log \
         /mnt/nvme3n1/g00872988/turboquant/turboquant_debug.log \
         "$ROOT_DIR/turboquant_debug.log"; do
    if [ -s "$p" ]; then
        echo ""
        echo "--- $p ($(wc -l < "$p") lines) ---"
        head -60 "$p"
    fi
done
echo ""
echo "==================================================================="
echo "[run-bypass] grep 'TURBOQUANT_DBG' from bypass server log"
echo "==================================================================="
BP_LOG="$ROOT_DIR/diag_bypass_tq.log"
if [ -f "$BP_LOG" ]; then
    grep -a "TURBOQUANT_DBG\|TurboQuant" "$BP_LOG" | head -40 || echo "(no match in $BP_LOG)"
else
    echo "(server log $BP_LOG does not exist)"
fi
