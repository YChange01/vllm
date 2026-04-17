#!/usr/bin/env bash
# End-to-end probe for the TurboQuant Triton-kernel bug on Llama.
#
# Runs three checks in sequence and extracts the three comparison cuts
# that localise the bug in the store/attend kernel math.
#
#   1. BYPASS baseline           -> logs/diag_bypass_<ts>/
#        Confirms vLLM plumbing still works (no quant). Output should
#        match FLASH_ATTN.
#   2. mse b=8 via baseline.sh   -> logs/baseline_<ts>/
#        Main failing case. TURBOQUANT_ALGO=mse TURBOQUANT_BITS=8.
#   3. varlen kernel sanity (synthetic Gaussian)
#        If this fails, the Triton kernel itself has regressed and the
#        bug is not distribution-specific.
#
# Extracts printed after runs:
#   (a) BYPASS  per-layer  |attn|.mean   (reference magnitude)
#   (b) mse b=8 per-layer  |attn|.mean   (KEY DIFF vs BYPASS)
#   (c) mse b=8 layer-0 full store/fwd trajectory
#
# Usage:
#   bash test/diag_mse.sh                 # gpu=3, prompt="Hello", max_tokens=4
#   GPU=0 bash test/diag_mse.sh           # override gpu via env
#   GPU=0 bash test/diag_mse.sh Hello 4   # override prompt / max_tokens

set -u

GPU="${GPU:-3}"
PROMPT="${1:-Hello}"
MAX_TOKENS="${2:-4}"
ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

section() {
    echo ""
    echo "=========================================================="
    echo "[diag-mse] $*"
    echo "=========================================================="
}

# ---- 1/3  BYPASS ---------------------------------------------------------
section "1/3  BYPASS (no quant)"
bash test/diag_bypass.sh "$PROMPT" "$MAX_TOKENS" "$GPU" || true

# ---- 2/3  mse b=8 --------------------------------------------------------
section "2/3  TURBOQUANT_ALGO=mse  TURBOQUANT_BITS=8"
TURBOQUANT_ALGO=mse TURBOQUANT_BITS=8 \
    bash test/baseline.sh "$PROMPT" "$MAX_TOKENS" "$GPU" || true

# ---- 3/3  varlen kernel sanity ------------------------------------------
section "3/3  varlen kernel sanity (random Gaussian)"
CUDA_VISIBLE_DEVICES="$GPU" python3 test/test_varlen_kernel.py --bits 8 || true

# ---- Extracts -----------------------------------------------------------
BP_DIR="$ROOT_DIR/logs/diag_bypass_latest"
BASE_DIR="$ROOT_DIR/logs/baseline_latest"

section "EXTRACTS"

echo ""
echo "(a) BYPASS per-layer |attn|.mean  --  reference magnitude"
echo "----------------------------------------------------------"
if [ -f "$BP_DIR/tq_bypass_server.log" ]; then
    grep "^\[TURBOQUANT_DBG\] \[out " "$BP_DIR/tq_bypass_server.log" \
        || echo "(no [out] lines found)"
else
    echo "(missing: $BP_DIR/tq_bypass_server.log)"
fi

echo ""
echo "(b) mse b=8 per-layer |attn|.mean  --  KEY DIFF vs (a)"
echo "----------------------------------------------------------"
if [ -f "$BASE_DIR/tq8_server.log" ]; then
    grep "^\[TURBOQUANT_DBG\] \[out " "$BASE_DIR/tq8_server.log" \
        || echo "(no [out] lines found)"
else
    echo "(missing: $BASE_DIR/tq8_server.log)"
fi

echo ""
echo "(c) mse b=8 layer-0 full store/fwd trajectory"
echo "----------------------------------------------------------"
if [ -f "$BASE_DIR/tq8_server.log" ]; then
    grep "^\[TURBOQUANT_DBG\]" "$BASE_DIR/tq8_server.log" | grep "L0\]" \
        || echo "(no L0 lines found)"
else
    echo "(missing: $BASE_DIR/tq8_server.log)"
fi

echo ""
echo "=========================================================="
echo "[diag-mse] done. Run dirs:"
echo "  BYPASS:  $BP_DIR"
echo "  mse b=8: $BASE_DIR"
echo "=========================================================="
