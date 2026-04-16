#!/usr/bin/env bash
# Start vLLM with TurboQuant attention backend.
# Usage: bash test/serve.sh [MODEL_PATH] [PORT]

MODEL="${1:-/mnt/nvme3n1/g00872988/models/Qwen3-0.6B}"
PORT="${2:-8000}"

vllm serve "$MODEL" --port "$PORT" --enforce-eager --attention-backend TURBOQUANT
