#!/usr/bin/env bash
# Quick correctness probe after the K-store-in-Python rewrite.
# Compares baseline.sh outputs (FLASH_ATTN vs TURBOQUANT mse b=8 vs
# TURBOQUANT mse b=4) on a 26-token prompt.

set -u

cd "$(dirname "$0")/.."

GPU="${GPU:-3}"
PROMPT="${PROMPT:-Machine learning has transformed many fields over the past decade with deep neural networks achieving remarkable performance on natural language understanding and speech synthesis.}"
MAX_TOKENS="${MAX_TOKENS:-4}"

echo "=========================================================="
echo "1) mse  b=8 vs b=4"
echo "=========================================================="
TURBOQUANT_ALGO=mse TURBOQUANT_BITS=8 \
    bash test/baseline.sh "$PROMPT" "$MAX_TOKENS" "$GPU" || true

echo ""
echo "=========================================================="
echo "2) prod b=8 vs b=4"
echo "=========================================================="
TURBOQUANT_ALGO=prod TURBOQUANT_BITS=8 \
    bash test/baseline.sh "$PROMPT" "$MAX_TOKENS" "$GPU" || true
