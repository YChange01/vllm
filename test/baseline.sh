#!/usr/bin/env bash
# Baseline comparison: same prompt through three configurations.
#
#   1. FLASH_ATTN           (fp reference)
#   2. TURBOQUANT  b=8      (8-bit Lloyd-Max, default)
#   3. TURBOQUANT  b=4      (4-bit Lloyd-Max, TURBOQUANT_BITS=4)
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
#   bash test/baseline.sh                         # defaults
#   bash test/baseline.sh "Hello" 4 3             # prompt, max_tokens, gpu

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
LOG_DIR="$ROOT_DIR/logs/baseline_$TS"
mkdir -p "$LOG_DIR"
ln -sfn "$LOG_DIR" "$ROOT_DIR/logs/baseline_latest"
FP_LOG="$LOG_DIR/fp_server.log"
TQ8_LOG="$LOG_DIR/tq8_server.log"
TQ4_LOG="$LOG_DIR/tq4_server.log"
DBG8_LOG="$LOG_DIR/turboquant_debug_tq8.log"
DBG4_LOG="$LOG_DIR/turboquant_debug_tq4.log"

# Track the running server's process-group id so cleanup can kill the
# whole group on early exit / Ctrl-C. Empty string = nothing to kill.
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

# ---- wait for /health, using ps to probe liveness (no signals) ----------
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

# ---- POST one completion ------------------------------------------------
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

# ---- run one backend end-to-end -----------------------------------------
# Globals read:  MODEL MAX_LEN CURRENT_PGID
# Globals written: CURRENT_PGID, <OUT_VAR>
# $1 label, $2 port, $3 log, $4 backend, $5 extra env (e.g. TURBOQUANT_BITS=4), $6 out-var name
run_backend() {
    local label="$1" port="$2" log="$3" backend="$4" extra_env="$5" out_var="$6"

    echo ""
    echo "[baseline] starting $label on port $port ..."
    : > "$log"

    # setsid makes the subprocess a new session leader (PID == PGID), so
    # killing -PGID later reliably takes out the EngineCore children too.
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
        # trap cleanup will kill the group on exit
        exit 1
    fi

    local out text
    out=$(query_completion "$port")
    text=$(echo "$out" | parse_text)
    echo "[baseline] $label raw: $out"
    echo "[baseline] $label text: $text"
    printf -v "$out_var" '%s' "$text"

    # Stop this server's entire process group so the next stage starts clean.
    echo "[baseline] stopping $label process group $CURRENT_PGID"
    kill -TERM -"$CURRENT_PGID" 2>/dev/null || true
    wait "$CURRENT_PGID" 2>/dev/null || true
    CURRENT_PGID=""
    # Give the driver a moment to free KV cache / allocator state.
    sleep 3
}

# ---- main ---------------------------------------------------------------
export CUDA_VISIBLE_DEVICES="$GPU"
echo "[baseline] model=$MODEL prompt='$PROMPT' max_tokens=$MAX_TOKENS gpu=$GPU"
echo "[baseline] logs dir: $LOG_DIR  (symlink: $ROOT_DIR/logs/baseline_latest)"

FP_TEXT=""
TQ8_TEXT=""
TQ4_TEXT=""

run_backend "FLASH_ATTN"        "$FP_PORT" "$FP_LOG"  FLASH_ATTN  ""                                                        FP_TEXT
run_backend "TURBOQUANT b=8"    "$TQ_PORT" "$TQ8_LOG" TURBOQUANT  "TURBOQUANT_BITS=8 TURBOQUANT_DEBUG_LOG=$DBG8_LOG"         TQ8_TEXT
run_backend "TURBOQUANT b=4"    "$TQ_PORT" "$TQ4_LOG" TURBOQUANT  "TURBOQUANT_BITS=4 TURBOQUANT_DEBUG_LOG=$DBG4_LOG"         TQ4_TEXT

echo ""
echo "========================== SIDE-BY-SIDE =========================="
echo "prompt:          ${PROMPT}"
echo "max_tokens:      ${MAX_TOKENS}  (temperature=0, greedy decoding)"
echo "FLASH_ATTN:      ${FP_TEXT}"
echo "TURBOQUANT b=8:  ${TQ8_TEXT}"
echo "TURBOQUANT b=4:  ${TQ4_TEXT}"
echo "=================================================================="
echo "[baseline] done; all three servers have been stopped."
