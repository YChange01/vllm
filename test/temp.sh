#!/usr/bin/env bash
# Scratchpad for ad-hoc turboquant kernel checks, no vLLM serve.
# Run env_test.sh first to rule out driver/torch/triton issues.

set -u
cd "$(dirname "$0")/.."

GPU="${GPU:-3}"
export CUDA_VISIBLE_DEVICES="$GPU"
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

echo "=========================================================="
echo "1) Standalone varlen kernel (store + base attend on random K/V/Q)"
echo "=========================================================="
python3 test/test_varlen_kernel.py --num-tokens 64 || true

echo ""
echo "=========================================================="
echo "2) Base attend vs LUT attend numerical agreement"
echo "=========================================================="
python3 test/test_lut_vs_base.py || true

echo ""
echo "[temp] kernel checks done. If everything looks OK, try:"
echo "    GPU=$GPU bash test/baseline.sh                 # FLASH / base / LUT"
echo "    GPU=$GPU TURBOQUANT_ALGO=prod bash test/baseline.sh"
