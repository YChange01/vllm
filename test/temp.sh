#!/usr/bin/env bash
# Quick greedy-text sanity for the turboquant-lut branch:
# run baseline.sh twice, once with algo=mse and once with algo=prod.
# Each invocation compares FLASH_ATTN vs base-kernel b=4 vs LUT-kernel b=4.

set -u
cd "$(dirname "$0")/.."

GPU="${GPU:-3}"
PROMPT="${PROMPT:-Machine learning has transformed many fields over the past decade with deep neural networks achieving remarkable performance on natural language understanding and speech synthesis.}"
MAX_TOKENS="${MAX_TOKENS:-4}"

echo "=========================================================="
echo "1) mse b=4:  FLASH_ATTN  vs  base kernel  vs  LUT kernel"
echo "=========================================================="
TURBOQUANT_ALGO=mse \
    bash test/baseline.sh "$PROMPT" "$MAX_TOKENS" "$GPU" || true

echo ""
echo "=========================================================="
echo "2) prod b=4:  FLASH_ATTN  vs  base kernel  vs  LUT kernel"
echo "=========================================================="
TURBOQUANT_ALGO=prod \
    bash test/baseline.sh "$PROMPT" "$MAX_TOKENS" "$GPU" || true
