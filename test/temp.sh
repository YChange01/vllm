#!/usr/bin/env bash
# Verify K + V quantization (mse and prod) all still produce correct
# greedy text after V was switched from raw to Q_mse quantization.

set -u
cd "$(dirname "$0")/.."

GPU="${GPU:-3}"
PROMPT="${PROMPT:-Machine learning has transformed many fields over the past decade with deep neural networks achieving remarkable performance on natural language understanding and speech synthesis.}"
MAX_TOKENS="${MAX_TOKENS:-4}"

echo "=========================================================="
echo "1) mse  K+V  b=8 vs b=4"
echo "=========================================================="
TURBOQUANT_ALGO=mse TURBOQUANT_BITS=8 \
    bash test/baseline.sh "$PROMPT" "$MAX_TOKENS" "$GPU" || true

echo ""
echo "=========================================================="
echo "2) prod K, mse V  b=8 vs b=4"
echo "=========================================================="
TURBOQUANT_ALGO=prod TURBOQUANT_BITS=8 \
    bash test/baseline.sh "$PROMPT" "$MAX_TOKENS" "$GPU" || true
