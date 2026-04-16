#!/usr/bin/env bash
# Start vLLM with TurboQuant attention backend.
# Usage:
#   bash test/serve.sh                          # default model, card 0
#   bash test/serve.sh /path/to/model 8000 1    # custom model, port, card

MODEL="${1:-/mnt/nvme3n1/g00872988/models/Qwen3-0.6B}"
PORT="${2:-8000}"
GPU="${3:-}"

if [ -n "$GPU" ]; then
    export CUDA_VISIBLE_DEVICES="$GPU"
fi

cleanup() {
    echo ""
    echo "cleaning up GPU processes..."
    # Kill all child processes (EngineCore etc.)
    pkill -P $$ 2>/dev/null
    # Kill any leftover python using our port
    lsof -ti:${PORT} 2>/dev/null | xargs kill -9 2>/dev/null
    echo "done"
}
trap cleanup EXIT INT TERM

echo "model: $MODEL"
echo "port:  $PORT"
echo "gpu:   ${CUDA_VISIBLE_DEVICES:-all}"
echo "Ctrl+C to stop (will auto-cleanup GPU memory)"
echo ""

vllm serve "$MODEL" --port "$PORT" --enforce-eager --attention-backend TURBOQUANT
