#!/usr/bin/env bash
# Start vLLM with TurboQuant attention backend.
# Usage:
#   bash test/serve.sh                          # default model, card 0
#   bash test/serve.sh /path/to/model 8000 1    # custom model, port, card

MODEL="${1:-/mnt/nvme3n1/g00872988/models/Qwen3-0.6B}"
PORT="${2:-8009}"
GPU="${3:-1}"

if [ -n "$GPU" ]; then
    export CUDA_VISIBLE_DEVICES="$GPU"
fi

echo "model: $MODEL"
echo "port:  $PORT"
echo "gpu:   ${CUDA_VISIBLE_DEVICES:-all}"
echo ""

MAX_LEN="${4:-4096}"
# Limit KV cache blocks so that our side buffers (_k_idx + _v_cache +
# _k_norms per layer) still fit in the remaining GPU memory.
# 1024 blocks × 16 tokens/block = 16K cache tokens, plenty for testing.
NUM_BLOCKS="${5:-1024}"

vllm serve "$MODEL" \
    --port "$PORT" \
    --enforce-eager \
    --attention-backend TURBOQUANT \
    --max-model-len "$MAX_LEN" \
    --gpu-memory-utilization 0.5 \
    --num-gpu-blocks-override "$NUM_BLOCKS"
