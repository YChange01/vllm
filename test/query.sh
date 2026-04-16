#!/usr/bin/env bash
# Send a test query to the running vLLM server.
# Usage: bash test/query.sh [PROMPT] [PORT]

MODEL="/mnt/nvme3n1/g00872988/models/Qwen3-0.6B"
PROMPT="${1:-Hello}"
PORT="${2:-8000}"

echo "[health check]"
curl -s http://localhost:${PORT}/health && echo ""

echo "[models]"
curl -s http://localhost:${PORT}/v1/models | python3 -m json.tool 2>/dev/null || echo "failed"

echo "[completions]"
curl -s http://localhost:${PORT}/v1/completions \
  -H 'Content-Type: application/json' \
  -d "{\"model\":\"${MODEL}\",\"prompt\":\"${PROMPT}\",\"max_tokens\":30}" | python3 -m json.tool 2>/dev/null || echo "failed or timeout"
