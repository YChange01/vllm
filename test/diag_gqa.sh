#!/usr/bin/env bash
# Diagnostic: run the varlen kernel smoke test with Llama-3.1-8B config
# (num_heads_q=32, num_heads_kv=8, head_size=128, gqa=4).
#
# Goal: isolate whether the `://24` baseline bug is in the Triton kernel
# itself or in the vLLM integration layer.
#
# Interpretation
# --------------
#   mse_rel < ~3% across all num_tokens   -> kernel handles GQA=4 correctly;
#                                            bug is in vLLM integration (output
#                                            layout, scale wiring, buffer per
#                                            layer, etc.).
#   mse_rel >>10% / NaN / huge mse_max    -> kernel has a GQA=4 specific bug
#                                            (stride / head mapping / Hadamard).
#
# For comparison we also run the original gqa=2 config (heads_q=16) which
# we already know works (`mse_rel≈1.76%` on num_tokens=128).
#
# Usage:
#   bash test/diag_gqa.sh            # defaults (both configs, bits=8)
#   BITS=4 bash test/diag_gqa.sh     # try b=4 too
#   GPU=3 bash test/diag_gqa.sh

set -u

GPU="${GPU:-3}"
BITS="${BITS:-8}"
ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
PY="$ROOT_DIR/test/test_varlen_kernel.py"

echo "==================================================================="
echo "[diag_gqa] BASELINE  heads_q=16 heads_kv=8 gqa=2  (known-good)"
echo "==================================================================="
CUDA_VISIBLE_DEVICES="$GPU" python3 "$PY" \
    --bits "$BITS" \
    --num-heads-q 16 --num-heads-kv 8 --head-size 128

echo ""
echo "==================================================================="
echo "[diag_gqa] LLAMA-3.1-8B  heads_q=32 heads_kv=8 gqa=4  (suspect)"
echo "==================================================================="
CUDA_VISIBLE_DEVICES="$GPU" python3 "$PY" \
    --bits "$BITS" \
    --num-heads-q 32 --num-heads-kv 8 --head-size 128

echo ""
echo "==================================================================="
echo "[diag_gqa] DONE. If Llama-config mse_rel is in the same ballpark as"
echo "[diag_gqa] the baseline (<~3%), the kernel is clean and the bug is"
echo "[diag_gqa] in the vLLM integration layer. Otherwise the kernel has"
echo "[diag_gqa] a GQA-4-specific issue."
echo "==================================================================="
