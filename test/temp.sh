#!/usr/bin/env bash
# One-shot diagnostic scratchpad. Overwritten every time.
#
# Current probe: PYREF |attn|.max=5.25 > max|v|=0.482 is mathematically
# only possible if cache_v_fp contains values larger than the input V
# bound. Triton store kernel must be writing wrong values to cache_v_fp
# (BYPASS works because it skips Triton store and uses direct python copy).
#
# Added [verify L*] post-store debug log that compares stored cache_v_fp
# against input V at the just-written slots, plus dtype info for
# kv_cache, _v_fp, _k_idx, _k_norm. This run launches mse b=8 (TRITON
# attend, NOT pyref -- we don't need pyref because the corruption is
# in the store path) and shows the first verify entries.

set -u

cd "$(dirname "$0")/.."

GPU="${GPU:-3}"
PROMPT="${PROMPT:-Machine learning has transformed many fields over the past decade with deep neural networks achieving remarkable performance on natural language understanding and speech synthesis.}"
MAX_TOKENS="${MAX_TOKENS:-4}"

echo "[temp] GPU=$GPU MAX_TOKENS=$MAX_TOKENS"
echo ""

echo "=========================================================="
echo "Run mse b=8 (Triton, NOT pyref) just to populate cache and log [verify L*]"
echo "=========================================================="
TURBOQUANT_ALGO=mse TURBOQUANT_BITS=8 \
    bash test/baseline.sh "$PROMPT" "$MAX_TOKENS" "$GPU" || true

echo ""
echo "=========================================================="
echo "[verify L*] entries  (one per layer per call, up to 3 each)"
echo "----------------------------------------------------------"
echo "Look for max_abs_diff. If 0 -> store works, bug elsewhere."
echo "If non-zero -> Triton store corrupting V."
echo "Also check input_v|.|max vs stored_v|.|max."
echo "=========================================================="
DBG="logs/baseline_latest/turboquant_debug_tq8.log"
if [ -f "$DBG" ]; then
    grep "^\[verify L" "$DBG" | head -10
    echo ""
    echo "----- [verifyK L*] (K cache sanity) -----"
    grep "^\[verifyK L" "$DBG" | head -10
else
    echo "(missing $DBG)"
fi

echo ""
echo "=========================================================="
echo "Reminder: full V layout for first written slot of L0"
echo "=========================================================="
echo "Look at the first L0 [verify] entry's stored_v[0,0,:4] vs"
echo "first_input_slot_v[0,0,:4] -- they must match exactly."
