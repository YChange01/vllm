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

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
# This wrapper delegates to diag_bypass.sh, which creates its own
# timestamped subdir. We only set TURBOQUANT_DEBUG_LOG if the caller
# explicitly wants to override where the python side writes.
LOG_DIR="$ROOT_DIR/logs"
mkdir -p "$LOG_DIR"
cd "$ROOT_DIR"

echo "[run-bypass] repo: $ROOT_DIR"
echo "[run-bypass] pulling latest ..."
git pull --ff-only
echo ""
echo "[run-bypass] HEAD:"
git log --oneline -3
echo ""

# Run BYPASS; diag_bypass.sh writes into logs/diag_bypass_<ts>/ .
echo "[run-bypass] launching diag_bypass.sh with TURBOQUANT_BYPASS=1 ..."
CUDA_VISIBLE_DEVICES="$GPU" \
TURBOQUANT_BYPASS=1 \
    bash test/diag_bypass.sh "$PROMPT" "$MAX_TOKENS" "$GPU"

# diag_bypass.sh points this symlink at the run it just created.
RUN_DIR="$(readlink -f "$LOG_DIR/diag_bypass_latest" 2>/dev/null || echo "")"

echo ""
echo "==================================================================="
if [ -n "$RUN_DIR" ] && [ -d "$RUN_DIR" ]; then
    echo "[run-bypass] this run's logs:"
    echo "  $RUN_DIR"
    echo ""
    for p in "$RUN_DIR"/*.log; do
        [ -s "$p" ] || continue
        echo "--- $p ($(wc -l < "$p") lines) ---"
        head -60 "$p"
        echo ""
    done
    echo "==================================================================="
    echo "[run-bypass] grep 'TURBOQUANT_DBG' from bypass server log"
    echo "==================================================================="
    BP_LOG="$RUN_DIR/tq_bypass_server.log"
    if [ -f "$BP_LOG" ]; then
        grep -a "TURBOQUANT_DBG\|TurboQuant" "$BP_LOG" | head -40 || echo "(no match in $BP_LOG)"
    else
        echo "(server log $BP_LOG does not exist)"
    fi
else
    echo "[run-bypass] could not resolve diag_bypass_latest symlink"
fi
