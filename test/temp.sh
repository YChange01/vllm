#!/usr/bin/env bash
# One-shot diagnostic scratchpad. Overwritten every time.
#
# Current probe: V cache is now correct (V copied in Python, not Triton).
# mse b=8 still gives gibberish ("AAAA..."), with |attn|.max < max|v|
# (math is sound) but K quant noise still distorts attention sink and
# topples the softmax ordering.
#
# Try paper's Algorithm 2 (Q_prod = b-1 bit MSE + 1-bit QJL on the
# residual). The QJL part gives an unbiased inner-product estimator
# (Lemma 4) and should be more precise than raw MSE.
#
# Expectation: if prod text approaches "However, the field", TurboQuant
# K-only quant works on Llama. If still gibberish, attention sink needs
# explicit protection (a separate fix).

set -u

cd "$(dirname "$0")/.."

GPU="${GPU:-3}"
PROMPT="${PROMPT:-Machine learning has transformed many fields over the past decade with deep neural networks achieving remarkable performance on natural language understanding and speech synthesis.}"
MAX_TOKENS="${MAX_TOKENS:-4}"

echo "[temp] GPU=$GPU MAX_TOKENS=$MAX_TOKENS  algo=prod"
echo ""

echo "=========================================================="
echo "Run prod b=8 baseline (Algorithm 2: 7-bit MSE + 1-bit QJL)"
echo "=========================================================="
TURBOQUANT_ALGO=prod TURBOQUANT_BITS=8 \
    bash test/baseline.sh "$PROMPT" "$MAX_TOKENS" "$GPU" || true

echo ""
echo "=========================================================="
echo "[verify L0..L4]  V should still match exactly (max_abs_diff=0)"
echo "=========================================================="
DBG="logs/baseline_latest/turboquant_debug_tq8.log"
if [ -f "$DBG" ]; then
    grep "^\[verify L[0-4]\]" "$DBG" | head -10
else
    echo "(missing $DBG)"
fi

echo ""
echo "=========================================================="
echo "[out L0..L4 call=0]   |attn|.mean / |attn|.max"
echo "Reference -- BYPASS L0 c0 = 0.0040 / 0.248"
echo "             BYPASS L1 c0 = 0.0066 / 0.309"
echo "             FLASH_ATTN text = ', However, the field'"
echo "=========================================================="
if [ -f "$DBG" ]; then
    grep "^\[out   L[0-4]\] call=0" "$DBG" | head -10
fi
