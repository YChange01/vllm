#!/usr/bin/env bash
# One-shot diagnostic scratchpad. Overwritten every time.
#
# Current probe: V is now stored entirely in Python (do_kv_cache_update),
# K via Triton kernel only. The Triton store had a num_tokens-dependent
# corruption of cache_v_fp -- single-token decode worked, multi-token
# prefill silently overwrote V with K-derived values. Going around it
# in Python should round-trip V perfectly.
#
# Expectation:
#   * verify L*  -> max_abs_diff = 0  on EVERY layer for both prefill
#                   and decode (Python copy is bit-exact)
#   * baseline.sh text output -> "However, the field" or similar
#                                 (matching FLASH_ATTN), since K quant
#                                 has 0.5% logit error per the (4) probe

set -u

cd "$(dirname "$0")/.."

GPU="${GPU:-3}"
PROMPT="${PROMPT:-Machine learning has transformed many fields over the past decade with deep neural networks achieving remarkable performance on natural language understanding and speech synthesis.}"
MAX_TOKENS="${MAX_TOKENS:-4}"

echo "[temp] GPU=$GPU MAX_TOKENS=$MAX_TOKENS"
echo ""

echo "=========================================================="
echo "Run mse b=8 baseline -- V via Python copy, K via Triton"
echo "=========================================================="
TURBOQUANT_ALGO=mse TURBOQUANT_BITS=8 \
    bash test/baseline.sh "$PROMPT" "$MAX_TOKENS" "$GPU" || true

echo ""
echo "=========================================================="
echo "[verify L0..L4] (max_abs_diff should be 0 now)"
echo "=========================================================="
DBG="logs/baseline_latest/turboquant_debug_tq8.log"
if [ -f "$DBG" ]; then
    grep "^\[verify L[0-4]\]" "$DBG"
else
    echo "(missing $DBG)"
fi
