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
rm -f "$DEBUG_LOG"
echo "[run-bypass] cleared $DEBUG_LOG"

# Run BYPASS with debug logging on.
echo "[run-bypass] launching diag_bypass.sh with TURBOQUANT_DEBUG=1 ..."
CUDA_VISIBLE_DEVICES="$GPU" \
TURBOQUANT_BYPASS=1 \
TURBOQUANT_DEBUG=1 \
TURBOQUANT_DEBUG_LOG="$DEBUG_LOG" \
    bash test/diag_bypass.sh "$PROMPT" "$MAX_TOKENS" "$GPU"

echo ""
echo "==================================================================="
echo "[run-bypass] first 60 lines of $DEBUG_LOG"
echo "==================================================================="
if [ -s "$DEBUG_LOG" ]; then
    head -60 "$DEBUG_LOG"
else
    echo "(debug log is empty — the backend did not hit the instrumented path)"
fi
