#!/usr/bin/env bash
# Profile the two TURBOQUANT b=4 kernel paths back-to-back, so we can see
# where the ~90 ms mse_b4_lut decode step actually goes (attend kernel vs
# Python store vs pre/post-rotate matmul vs vLLM overhead).
#
# FLASH_ATTN reference is not profiled here -- the throughput.sh run
# already gives the reference number (ITL ~14 ms), and this machine's
# default FLASHINFER backend hits a GLIBCXX_3.4.32 mismatch which is
# unrelated to our work.

set -u
cd "$(dirname "$0")/.."

GPU="${GPU:-3}"
TS="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="logs/profile_$TS"
mkdir -p "$LOG_DIR"
ln -sfn "profile_$TS" "logs/profile_latest"

echo "=========================================================="
echo "1) TURBOQUANT mse b=4 (base kernel, CUDA core)"
echo "=========================================================="
GPU="$GPU" TURBOQUANT_ALGO=mse TURBOQUANT_BITS=4 \
    python3 test/profile_attend.py 2>&1 | tee "$LOG_DIR/mse_b4_base.log"

echo ""
echo "=========================================================="
echo "2) TURBOQUANT mse b=4 (LUT kernel, tensor core)"
echo "=========================================================="
GPU="$GPU" TURBOQUANT_ALGO=mse TURBOQUANT_BITS=4 TURBOQUANT_USE_LUT=1 \
    python3 test/profile_attend.py 2>&1 | tee "$LOG_DIR/mse_b4_lut.log"

echo ""
echo "[temp] done. Per-run outputs under $LOG_DIR/"
echo "[temp] Category breakdown is at the bottom of each .log"
echo "[temp] Chrome trace: $LOG_DIR/../profile_latest/trace_*.json"
