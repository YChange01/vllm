#!/usr/bin/env bash
# Baseline comparison (turboquant-lut branch -- b=4 only).
#
# Runs the same prompt through three configurations:
#   1. FLASH_ATTN                   (fp reference)
#   2. TURBOQUANT b=4 + LUT   (Triton tensor-core attend, TURBOQUANT_USE_LUT=1)
#   3. TURBOQUANT b=4 + CUDA  (raw-CUDA WMMA attend, TURBOQUANT_USE_CUDA=1)
#
# Each server is started with setsid so it has its own process group; at
# the end of that stage we kill the WHOLE group (signal -TERM to -PGID)
# so stray EngineCore children cannot leak. We never pkill anything else.
#
# Prerequisite (user manages these):
#   - ports 8009 and 8010 must be free
#   - any prior zombies must be cleaned up already
#
# Usage:
#   bash test/baseline.sh                              # defaults
#   bash test/baseline.sh "Hello" 4 3                  # prompt, max_tokens, gpu
#   GPU=2 bash test/baseline.sh                        # via env
#   TURBOQUANT_ALGO=prod bash test/baseline.sh         # flip algo for TQ stages

set -u

MODEL="${MODEL:-/mnt/nvme3n1/g00872988/models/Llama-3.1-8B-Instruct}"
PROMPT="${1:-Hello}"
MAX_TOKENS="${2:-4}"
GPU="${GPU:-${3:-3}}"
MAX_LEN="${MAX_LEN:-4096}"
ALGO="${TURBOQUANT_ALGO:-mse}"

FP_PORT=8010
TQ_PORT=8009

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
TS="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="$ROOT_DIR/logs/baseline_$TS"
mkdir -p "$LOG_DIR"
ln -sfn "$LOG_DIR" "$ROOT_DIR/logs/baseline_latest"
FP_LOG="$LOG_DIR/fp_server.log"
TQ_LUT_LOG="$LOG_DIR/tq_lut_server.log"
TQ_CUDA_LOG="$LOG_DIR/tq_cuda_server.log"

CURRENT_PGID=""

cleanup() {
    local code=$?
    if [ -n "$CURRENT_PGID" ]; then
        echo "[baseline] interrupted; stopping process group $CURRENT_PGID"
        kill -TERM -"$CURRENT_PGID" 2>/dev/null || true
        wait "$CURRENT_PGID" 2>/dev/null || true
    fi
    exit "$code"
}
trap cleanup EXIT INT TERM

wait_healthy() {
    local port="$1" log="$2" pid="$3"
    for _ in $(seq 1 72); do   # up to 6 minutes
        if curl -sf "http://localhost:${port}/health" >/dev/null 2>&1; then
            return 0
        fi
        if ! ps -p "$pid" >/dev/null 2>&1; then
            echo "[baseline] server on port $port exited early; tail of log:" >&2
            tail -n 40 "$log" >&2
            return 1
        fi
        sleep 5
    done
    echo "[baseline] server on port $port never became healthy" >&2
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

# $1 label, $2 port, $3 log, $4 backend, $5 extra env, $6 out-var
run_backend() {
    local label="$1" port="$2" log="$3" backend="$4" extra_env="$5" out_var="$6"

    echo ""
    echo "[baseline] starting $label on port $port ..."
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
    echo "[baseline] $label raw: $out"
    echo "[baseline] $label text: $text"
    printf -v "$out_var" '%s' "$text"

    echo "[baseline] stopping $label process group $CURRENT_PGID"
    kill -TERM -"$CURRENT_PGID" 2>/dev/null || true
    wait "$CURRENT_PGID" 2>/dev/null || true
    CURRENT_PGID=""
    sleep 3
}

# ---- main ---------------------------------------------------------------
export CUDA_VISIBLE_DEVICES="$GPU"
echo "[baseline] model=$MODEL prompt='$PROMPT' max_tokens=$MAX_TOKENS gpu=$GPU"
echo "[baseline] algo=$ALGO (bits=4, LUT branch)"
echo "[baseline] logs dir: $LOG_DIR  (symlink: $ROOT_DIR/logs/baseline_latest)"

FP_TEXT=""
TQ_LUT_TEXT=""
TQ_CUDA_TEXT=""

run_backend "FLASH_ATTN" "$FP_PORT" "$FP_LOG" FLASH_ATTN "" FP_TEXT
run_backend "TURBOQUANT b=4 (LUT)" "$TQ_PORT" "$TQ_LUT_LOG" TURBOQUANT \
    "TURBOQUANT_ALGO=$ALGO TURBOQUANT_BITS=4 TURBOQUANT_USE_LUT=1" TQ_LUT_TEXT
run_backend "TURBOQUANT b=4 (CUDA)" "$TQ_PORT" "$TQ_CUDA_LOG" TURBOQUANT \
    "TURBOQUANT_ALGO=$ALGO TURBOQUANT_BITS=4 TURBOQUANT_USE_CUDA=1" TQ_CUDA_TEXT

echo ""
echo "========================== SIDE-BY-SIDE =========================="
echo "prompt:                    ${PROMPT}"
echo "max_tokens:                ${MAX_TOKENS}  (temperature=0, greedy)"
echo "algo:                      ${ALGO}"
echo "FLASH_ATTN:                ${FP_TEXT}"
echo "TURBOQUANT b=4 (LUT):      ${TQ_LUT_TEXT}"
echo "TURBOQUANT b=4 (CUDA):     ${TQ_CUDA_TEXT}"
echo "=================================================================="
echo "[baseline] done; all three servers have been stopped."
