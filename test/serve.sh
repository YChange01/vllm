#!/usr/bin/env bash
# Start vLLM with TurboQuant attention backend.
# Usage:
#   bash test/serve.sh                          # default model, card 0
#   bash test/serve.sh /path/to/model 8000 1    # custom model, port, card
#   CUDA_VISIBLE_DEVICES=2 bash test/serve.sh   # specify card via env

MODEL="${1:-/mnt/nvme3n1/g00872988/models/Qwen3-0.6B}"
PORT="${2:-8000}"
GPU="${3:-}"

# If GPU arg provided, set CUDA_VISIBLE_DEVICES
if [ -n "$GPU" ]; then
    export CUDA_VISIBLE_DEVICES="$GPU"
fi

echo "model: $MODEL"
echo "port:  $PORT"
echo "gpu:   ${CUDA_VISIBLE_DEVICES:-all}"

vllm serve "$MODEL" --port "$PORT" --enforce-eager --attention-backend TURBOQUANT
