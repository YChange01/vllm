#!/usr/bin/env bash
# Validate + benchmark the A+B+C+D changes on turboquant-lut:
# Triton store kernel, bf16 Hadamard, no CPU-GPU sync, no nonzero filter.
#
# Stages run in order. If a stage fails hard the script keeps going so
# you see all the output -- but cross-check stage 1 first (numerical
# equivalence). If that breaks, skip the rest and paste the error.

set -u
cd "$(dirname "$0")/.."

GPU="${GPU:-3}"
TS="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="logs/temp_$TS"
mkdir -p "$LOG_DIR"
ln -sfn "temp_$TS" "logs/temp_latest"

export CUDA_VISIBLE_DEVICES="$GPU"
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

echo "=========================================================="
echo "1) Kernel-level numerical check (no vLLM)"
echo "   test_varlen_kernel: store + base attend on random K/V/Q"
echo "   test_lut_vs_base:   base vs LUT numerical agreement"
echo "=========================================================="
python3 test/test_varlen_kernel.py --num-tokens 64 2>&1 | tee "$LOG_DIR/test_varlen.log" || true
echo ""
python3 test/test_lut_vs_base.py 2>&1 | tee "$LOG_DIR/test_lut_vs_base.log" || true

echo ""
echo "=========================================================="
echo "2) vLLM text-level agreement: FLASH vs base vs LUT"
echo "   mse first, then prod"
echo "=========================================================="
GPU="$GPU" bash test/baseline.sh 2>&1 | tee "$LOG_DIR/baseline_mse.log" || true
echo ""
GPU="$GPU" TURBOQUANT_ALGO=prod bash test/baseline.sh 2>&1 | tee "$LOG_DIR/baseline_prod.log" || true

echo ""
echo "=========================================================="
echo "3) Profile per-decode-step latency breakdown"
echo "   drives store+attend directly, no vLLM subprocess"
echo "=========================================================="
GPU="$GPU" TURBOQUANT_ALGO=mse TURBOQUANT_BITS=4 \
    python3 test/profile_attend.py 2>&1 | tee "$LOG_DIR/profile_mse_base.log" || true
echo ""
GPU="$GPU" TURBOQUANT_ALGO=mse TURBOQUANT_BITS=4 TURBOQUANT_USE_LUT=1 \
    python3 test/profile_attend.py 2>&1 | tee "$LOG_DIR/profile_mse_lut.log" || true

echo ""
echo "=========================================================="
echo "4) End-to-end throughput A/B"
echo "   FLASH_ATTN + four TURBOQUANT variants"
echo "=========================================================="
GPU="$GPU" STAGES="FLASH_ATTN TURBOQUANT_mse_b4 TURBOQUANT_mse_b4_lut TURBOQUANT_prod_b4 TURBOQUANT_prod_b4_lut" \
    bash test/throughput.sh 2>&1 | tee "$LOG_DIR/throughput.log" || true

echo ""
echo "[temp] done. Per-stage logs: $LOG_DIR/"
