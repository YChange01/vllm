#!/usr/bin/env bash
# One-shot diagnostic scratchpad. Overwritten every time.
#
# K store rewritten in pure PyTorch (no Triton kernel) -- multi-token
# Triton writes were corrupting the cache. Triton attend kernel is
# unchanged (single-program-per-query, never showed the bug).
#
# Run mse b=8 first (algorithm 1: Lloyd-Max only). If the text comes
# out reasonable, prod (algorithm 2: + QJL residual) should give an
# even more accurate output.
#
# Expectation: text close to FLASH_ATTN's "However, the field".
# If still gibberish: K Lloyd-Max accuracy on real Llama K is the
# limit and we'd need attention-sink protection or stronger algo.

set -u

cd "$(dirname "$0")/.."

GPU="${GPU:-3}"
PROMPT="${PROMPT:-Machine learning has transformed many fields over the past decade with deep neural networks achieving remarkable performance on natural language understanding and speech synthesis.}"
MAX_TOKENS="${MAX_TOKENS:-4}"

echo "[temp] GPU=$GPU MAX_TOKENS=$MAX_TOKENS  algo=mse  store=python"
echo ""

echo "=========================================================="
echo "Run mse b=8 baseline (Python K store + Triton attend)"
echo "=========================================================="
TURBOQUANT_ALGO=mse TURBOQUANT_BITS=8 \
    bash test/baseline.sh "$PROMPT" "$MAX_TOKENS" "$GPU" || true

echo ""
echo "=========================================================="
echo "[verify L0..L4]  V should still match exactly (max_abs_diff=0)"
echo "=========================================================="
DBG="logs/baseline_latest/turboquant_debug_tq8.log"
if [ -f "$DBG" ]; then
    grep "^\[verify L[0-4]\]" "$DBG" | head -5
else
    echo "(missing $DBG)"
fi

echo ""
echo "=========================================================="
echo "[verifyK L0..L4]  stored k_norm should equal input ||k||"
echo "=========================================================="
if [ -f "$DBG" ]; then
    grep "^\[verifyK L[0-4]\]" "$DBG" | head -5
fi

echo ""
echo "=========================================================="
echo "[out L0..L4 call=0]   |attn|.mean / |attn|.max"
echo "Reference -- BYPASS L0 c0 = 0.0040 / 0.248"
echo "             FLASH_ATTN text = ' However, the field'"
echo "=========================================================="
if [ -f "$DBG" ]; then
    grep "^\[out   L[0-4]\] call=0" "$DBG" | head -5
fi
