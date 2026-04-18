#!/usr/bin/env bash
# Scratchpad: profile latency breakdown across three configs so we can see
# where the ~90 ms mse_b4_lut decode step actually goes.

set -u
cd "$(dirname "$0")/.."

GPU="${GPU:-3}"
TS="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="logs/profile_$TS"
mkdir -p "$LOG_DIR"
ln -sfn "profile_$TS" "logs/profile_latest"

echo "=========================================================="
echo "1) FLASH_ATTN (reference)"
echo "=========================================================="
GPU="$GPU" ATTN_BACKEND=FLASH_ATTN \
    python3 test/profile_attend.py 2>&1 | tee "$LOG_DIR/flash_attn.log"

echo ""
echo "=========================================================="
echo "2) TURBOQUANT mse b=4 (base kernel, CUDA core)"
echo "=========================================================="
GPU="$GPU" ATTN_BACKEND=TURBOQUANT TURBOQUANT_ALGO=mse TURBOQUANT_BITS=4 \
    python3 test/profile_attend.py 2>&1 | tee "$LOG_DIR/mse_b4_base.log"

echo ""
echo "=========================================================="
echo "3) TURBOQUANT mse b=4 (LUT kernel, tensor core)"
echo "=========================================================="
GPU="$GPU" ATTN_BACKEND=TURBOQUANT TURBOQUANT_ALGO=mse TURBOQUANT_BITS=4 \
    TURBOQUANT_USE_LUT=1 \
    python3 test/profile_attend.py 2>&1 | tee "$LOG_DIR/mse_b4_lut.log"

echo ""
echo "[temp] done. Per-run outputs under $LOG_DIR/"
echo "[temp] Chrome traces: $LOG_DIR/../profile_latest/trace_*.json"
echo "[temp] scp those to your laptop, open in https://ui.perfetto.dev/"
