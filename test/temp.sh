#!/usr/bin/env bash
# Validate the Stage 1 CUDA attend kernel on turboquant-cuda branch.
# First invocation triggers a JIT compile (~30-60s nvcc). Subsequent
# runs hit the torch extension cache.

set -u
cd "$(dirname "$0")/.."

GPU="${GPU:-3}"
TS="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="logs/cuda_$TS"
mkdir -p "$LOG_DIR"
ln -sfn "cuda_$TS" "logs/cuda_latest"

export CUDA_VISIBLE_DEVICES="$GPU"
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

echo "=========================================================="
echo "1) CUDA kernel numerical check vs Triton TC"
echo "   First run JIT-compiles the CUDA extension (~30-60s)."
echo "=========================================================="
TURBOQUANT_CUDA_VERBOSE=1 \
    python3 test/test_cuda_vs_tc.py 2>&1 | tee "$LOG_DIR/test_cuda_vs_tc.log"

echo ""
echo "=========================================================="
echo "2) vLLM text-level agreement: FLASH vs Triton TC vs CUDA"
echo "   mse path only (CUDA raises NotImplementedError on prod)"
echo "=========================================================="
GPU="$GPU" TURBOQUANT_USE_CUDA=1 \
    bash test/baseline.sh 2>&1 | tee "$LOG_DIR/baseline_cuda.log"

echo ""
echo "=========================================================="
echo "3) Throughput A/B: FLASH, Triton TC (default), CUDA (WMMA)"
echo "   FLASH_ATTN is the reference; TURBOQUANT_mse_b4 runs with the"
echo "   default Triton TC kernel; TURBOQUANT_mse_b4_cuda uses CUDA."
echo "=========================================================="
GPU="$GPU" \
    STAGES="FLASH_ATTN TURBOQUANT_mse_b4 TURBOQUANT_mse_b4_cuda" \
    bash test/throughput.sh 2>&1 | tee "$LOG_DIR/throughput.log" || true

echo ""
echo "[temp] done. Per-stage logs under $LOG_DIR/"
