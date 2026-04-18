#!/usr/bin/env bash
# Validate the CUDA attend kernel on turboquant-cuda branch.
# First invocation triggers a JIT compile (~30-60s nvcc). The torch
# extension cache key is based on source checksum, but it has been
# known to hold onto stale artifacts when e.g. a prior compile failed,
# so we nuke the build dir up front. If that matters to you (you're
# iterating on the same code without changes), comment the rm out.

set -u
cd "$(dirname "$0")/.."

GPU="${GPU:-3}"
TS="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="logs/cuda_$TS"
mkdir -p "$LOG_DIR"
ln -sfn "cuda_$TS" "logs/cuda_latest"

# Clear any stale JIT build artifacts.
rm -rf "${TURBOQUANT_CUDA_BUILD_DIR:-$HOME/.cache/torch_extensions/turboquant_cuda}"

export CUDA_VISIBLE_DEVICES="$GPU"
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

# CUTLASS discovery. The extension builds FA3 path by default; point this at
# a CUTLASS checkout (e.g. vllm's build/_deps/cutlass-src) or set
# TURBOQUANT_BUILD_FA3=0 to fall back to WMMA-only build.
: "${VLLM_CUTLASS_SRC_DIR:=}"
if [[ -z "$VLLM_CUTLASS_SRC_DIR" ]]; then
    for guess in \
        "$PWD/build/_deps/cutlass-src" \
        "$PWD/build"/cp*/_deps/cutlass-src \
        "/usr/local/cutlass"; do
        if [[ -f "$guess/include/cutlass/cutlass.h" ]]; then
            export VLLM_CUTLASS_SRC_DIR="$guess"
            break
        fi
    done
fi
echo "[temp] VLLM_CUTLASS_SRC_DIR=${VLLM_CUTLASS_SRC_DIR:-<unset>}"

echo ""
echo "=========================================================="
echo "0) FA3 build probe (JIT compile with CUTLASS, verify version)"
echo "=========================================================="
TURBOQUANT_CUDA_VERBOSE=1 \
    python3 test/test_fa3_build.py 2>&1 | tee "$LOG_DIR/fa3_build.log"

echo ""
echo "=========================================================="
echo "1) CUDA kernel numerical check vs Triton TC"
echo "   First run JIT-compiles the CUDA extension (~30-60s)."
echo "=========================================================="
TURBOQUANT_CUDA_VERBOSE=1 \
    python3 test/test_cuda_vs_tc.py 2>&1 | tee "$LOG_DIR/test_cuda_vs_tc.log"

echo ""
echo "=========================================================="
echo "2) vLLM text-level agreement: FLASH vs Triton TC vs CUDA (mse)"
echo "=========================================================="
GPU="$GPU" TURBOQUANT_USE_CUDA=1 \
    bash test/baseline.sh 2>&1 | tee "$LOG_DIR/baseline_cuda.log"

echo ""
echo "=========================================================="
echo "3) Throughput A/B: FLASH, Triton TC (default), CUDA (WMMA+cp.async)"
echo "=========================================================="
GPU="$GPU" \
    STAGES="FLASH_ATTN TURBOQUANT_mse_b4 TURBOQUANT_mse_b4_cuda" \
    bash test/throughput.sh 2>&1 | tee "$LOG_DIR/throughput.log" || true

echo ""
echo "=========================================================="
echo "4) Kernel-level profile (mse, CUDA path) -- directly drives"
echo "   store + attend with synthetic tensors; 32 layers × BATCH=16."
echo "=========================================================="
GPU="$GPU" TURBOQUANT_USE_CUDA=1 TURBOQUANT_ALGO=mse \
    python3 test/profile_attend.py 2>&1 | tee "$LOG_DIR/profile_mse.log"

echo ""
echo "=========================================================="
echo "5) Kernel-level profile (prod, CUDA path) -- same harness,"
echo "   prod path adds Sq matmul + QJL wmma + 1-bit decode."
echo "=========================================================="
GPU="$GPU" TURBOQUANT_USE_CUDA=1 TURBOQUANT_ALGO=prod \
    python3 test/profile_attend.py 2>&1 | tee "$LOG_DIR/profile_prod.log"

echo ""
echo "[temp] done. Per-stage logs under $LOG_DIR/"
