#!/usr/bin/env bash
# Start vLLM with TurboQuant attention backend.
# Usage:
#   bash test/serve.sh                          # default model, card 0
#   bash test/serve.sh /path/to/model 8000 1    # custom model, port, card

MODEL="${1:-/mnt/nvme3n1/g00872988/models/Qwen3-0.6B}"
PORT="${2:-8009}"
GPU="${3:-}"

if [ -n "$GPU" ]; then
    export CUDA_VISIBLE_DEVICES="$GPU"
fi

echo "model: $MODEL"
echo "port:  $PORT"
echo "gpu:   ${CUDA_VISIBLE_DEVICES:-all}"
echo ""

vllm serve "$MODEL" --port "$PORT" --enforce-eager --attention-backend TURBOQUANT
