#!/usr/bin/env bash
# Diagnostic: run the TurboQuant backend with TURBOQUANT_BYPASS=1.
#
# Under BYPASS the backend stores original bf16 K/V in a parallel buffer
# and does plain FP softmax(q @ k^T * scale) @ v at attend time. NO quant,
# NO Hadamard, NO Triton store/attend kernel.
#
# Interpretation
# --------------
#   bypass output == FLASH_ATTN output  -> vLLM plumbing is correct; the
#                                          `://24` bug lives in the quant
#                                          Triton kernel (or its numerics).
#   bypass output still `://24`-like    -> the bug is NOT in the quant
#                                          kernel. It is in how this
#                                          backend plugs into vLLM (output
#                                          tensor, slot semantics, scale
#                                          wiring, etc).
#
# Usage:
#   bash test/diag_bypass.sh
#   bash test/diag_bypass.sh "Hello" 4 3   # prompt, max_tokens, gpu

set -u

MODEL="${MODEL:-/mnt/nvme3n1/g00872988/models/Llama-3.1-8B-Instruct}"
PROMPT="${1:-Hello}"
MAX_TOKENS="${2:-4}"
GPU="${3:-3}"
MAX_LEN="${MAX_LEN:-4096}"

FP_PORT=8010
TQ_PORT=8009

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
TS="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="$ROOT_DIR/logs/diag_bypass_$TS"
mkdir -p "$LOG_DIR"
ln -sfn "$LOG_DIR" "$ROOT_DIR/logs/diag_bypass_latest"
FP_LOG="$LOG_DIR/fp_server.log"
BP_LOG="$LOG_DIR/tq_bypass_server.log"
DBG_LOG="$LOG_DIR/turboquant_debug.log"

CURRENT_PGID=""
cleanup() {
    local code=$?
    if [ -n "$CURRENT_PGID" ]; then
        kill -TERM -"$CURRENT_PGID" 2>/dev/null || true
        wait "$CURRENT_PGID" 2>/dev/null || true
    fi
    exit "$code"
}
trap cleanup EXIT INT TERM

wait_healthy() {
    local port="$1" log="$2" pid="$3"
    for _ in $(seq 1 72); do
        if curl -sf "http://localhost:${port}/health" >/dev/null 2>&1; then
            return 0
        fi
        if ! ps -p "$pid" >/dev/null 2>&1; then
            echo "[bypass] server on port $port exited early; tail of log:" >&2
            tail -n 40 "$log" >&2
            return 1
        fi
        sleep 5
    done
    echo "[bypass] server on port $port never became healthy" >&2
    tail -n 40 "$log" >&2
    return 1
}

query_completion() {
    local port="$1"
    curl -s "http://localhost:${port}/v1/completions" \
        -H 'Content-Type: application/json' \
        -d "{\"model\":\"${MODEL}\",\"prompt\":\"${PROMPT}\",\"max_tokens\":${MAX_TOKENS},\"temperature\":0}"
}

parse_text() {
    python3 -c 'import sys,json;print(json.load(sys.stdin)["choices"][0]["text"])' 2>/dev/null \
        || echo "<parse failed>"
}

run_backend() {
    local label="$1" port="$2" log="$3" backend="$4" extra_env="$5" out_var="$6"
    echo ""
    echo "[bypass] starting $label on port $port ..."
    : > "$log"
    if [ -n "$extra_env" ]; then
        env $extra_env setsid vllm serve "$MODEL" \
            --port "$port" \
            --enforce-eager \
            --attention-backend "$backend" \
            --max-model-len "$MAX_LEN" \
            --gpu-memory-utilization 0.3 \
            >>"$log" 2>&1 &
    else
        setsid vllm serve "$MODEL" \
            --port "$port" \
            --enforce-eager \
            --attention-backend "$backend" \
            --max-model-len "$MAX_LEN" \
            --gpu-memory-utilization 0.3 \
            >>"$log" 2>&1 &
    fi
    CURRENT_PGID=$!
    if ! wait_healthy "$port" "$log" "$CURRENT_PGID"; then
        exit 1
    fi
    local out text
    out=$(query_completion "$port")
    text=$(echo "$out" | parse_text)
    echo "[bypass] $label raw: $out"
    echo "[bypass] $label text: $text"
    printf -v "$out_var" '%s' "$text"
    kill -TERM -"$CURRENT_PGID" 2>/dev/null || true
    wait "$CURRENT_PGID" 2>/dev/null || true
    CURRENT_PGID=""
    sleep 3
}

export CUDA_VISIBLE_DEVICES="$GPU"
echo "[bypass] model=$MODEL prompt='$PROMPT' max_tokens=$MAX_TOKENS gpu=$GPU"
echo "[bypass] logs dir: $LOG_DIR  (symlink: $ROOT_DIR/logs/diag_bypass_latest)"

FP_TEXT=""
BP_TEXT=""

run_backend "FLASH_ATTN"           "$FP_PORT" "$FP_LOG" FLASH_ATTN ""                                                          FP_TEXT
run_backend "TURBOQUANT BYPASS=1"  "$TQ_PORT" "$BP_LOG" TURBOQUANT "TURBOQUANT_BYPASS=1 TURBOQUANT_DEBUG_LOG=$DBG_LOG"         BP_TEXT

echo ""
echo "========================== SIDE-BY-SIDE =========================="
echo "prompt:          ${PROMPT}"
echo "max_tokens:      ${MAX_TOKENS}  (temperature=0, greedy decoding)"
echo "FLASH_ATTN:          ${FP_TEXT}"
echo "TURBOQUANT BYPASS=1: ${BP_TEXT}"
echo "=================================================================="
echo ""
echo "If the two outputs MATCH (or close):"
echo "  -> backend plumbing is correct; bug is in the quant Triton kernel."
echo "If the BYPASS output is still gibberish like FLASH_ATTN:"
echo "  -> bug is in vLLM integration (output shape, slot_mapping,"
echo "     block_table interpretation, scale wiring)."
