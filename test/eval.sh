#!/usr/bin/env bash
# NIAH (Needle-in-a-Haystack) eval across two backends:
#   - FLASH_ATTN       (fp reference)
#   - TURBOQUANT b=4   (this branch is b=4 only)
#
# Same prompt set, temperature=0, so any accuracy drop is attributable to the
# quantization path. Each server is started via setsid in its own process
# group; at the end of its stage we kill -TERM -$PGID to take the whole
# group (parent + EngineCore) down together. No other processes are touched.
#
# Prerequisite (user manages these): ports 8009 / 8010 free, no zombies.
#
# Usage:
#   bash test/eval.sh
#   CTX=512,2048,8192 POSITIONS=0.1,0.5,0.9 TRIALS=5 bash test/eval.sh
#   MAX_TOKENS=24 bash test/eval.sh

set -u

MODEL="${MODEL:-/mnt/nvme3n1/g00872988/models/Llama-3.1-8B-Instruct}"
GPU="${GPU:-3}"
# NOTE: TurboQuant currently allocates a SEPARATE uint8 _k_idx buffer per
# attention layer on top of vLLM's native bf16 KV cache. That roughly
# doubles KV-cache memory for this backend. Keep max_model_len and
# gpu_memory_utilization modest so the extra buffers fit -- override with
# MAX_LEN=... GPU_MEM_UTIL=... when running at longer contexts.
MAX_LEN="${MAX_LEN:-4096}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.3}"

CTX="${CTX:-512,2048,4096}"
POSITIONS="${POSITIONS:-0.1,0.5,0.9}"
TRIALS="${TRIALS:-3}"
SEED="${SEED:-42}"
MAX_TOKENS="${MAX_TOKENS:-16}"
REQUEST_TIMEOUT="${REQUEST_TIMEOUT:-1800}"  # seconds; first Triton JIT on a
                                            # long ctx can take many minutes
                                            # for the TurboQuant kernel today.

FP_PORT=8010
TQ_PORT=8009

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
TS="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="$ROOT_DIR/logs/eval_$TS"
mkdir -p "$LOG_DIR"
ln -sfn "$LOG_DIR" "$ROOT_DIR/logs/eval_latest"

CURRENT_PGID=""

cleanup() {
    local code=$?
    if [ -n "$CURRENT_PGID" ]; then
        echo "[eval] interrupted; stopping process group $CURRENT_PGID"
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
            echo "[eval] server on port $port exited early; tail of log:" >&2
            tail -n 40 "$log" >&2
            return 1
        fi
        sleep 5
    done
    echo "[eval] server on port $port never became healthy" >&2
    tail -n 40 "$log" >&2
    return 1
}

# $1 tag, $2 port, $3 server-log, $4 eval-log, $5 backend, $6 env (e.g. TURBOQUANT_BITS=4)
run_stage() {
    local tag="$1" port="$2" slog="$3" elog="$4" backend="$5" extra_env="$6"

    echo ""
    echo "==================================================================="
    echo "[eval] stage: $tag  (port=$port backend=$backend env='$extra_env')"
    echo "[eval] server log: $slog"
    echo "[eval] eval   log: $elog"
    echo "==================================================================="
    : > "$slog"
    : > "$elog"

    if [ -n "$extra_env" ]; then
        env $extra_env setsid vllm serve "$MODEL" \
            --port "$port" \
            --enforce-eager \
            --attention-backend "$backend" \
            --max-model-len "$MAX_LEN" \
            --gpu-memory-utilization "$GPU_MEM_UTIL" \
            >>"$slog" 2>&1 &
    else
        setsid vllm serve "$MODEL" \
            --port "$port" \
            --enforce-eager \
            --attention-backend "$backend" \
            --max-model-len "$MAX_LEN" \
            --gpu-memory-utilization "$GPU_MEM_UTIL" \
            >>"$slog" 2>&1 &
    fi
    CURRENT_PGID=$!

    if ! wait_healthy "$port" "$slog" "$CURRENT_PGID"; then
        exit 1
    fi

    python3 "$ROOT_DIR/test/eval_niah.py" \
        --endpoint "http://localhost:${port}" \
        --model "$MODEL" \
        --ctx-lens "$CTX" \
        --positions "$POSITIONS" \
        --trials "$TRIALS" \
        --seed "$SEED" \
        --max-tokens "$MAX_TOKENS" \
        --request-timeout "$REQUEST_TIMEOUT" \
        --tag "$tag" \
        2>&1 | tee "$elog"

    echo "[eval] stopping $tag process group $CURRENT_PGID"
    kill -TERM -"$CURRENT_PGID" 2>/dev/null || true
    wait "$CURRENT_PGID" 2>/dev/null || true
    CURRENT_PGID=""
    sleep 3
}

export CUDA_VISIBLE_DEVICES="$GPU"
echo "[eval] model=$MODEL gpu=$GPU"
echo "[eval] ctx=$CTX positions=$POSITIONS trials=$TRIALS max_tokens=$MAX_TOKENS seed=$SEED"

run_stage "FLASH_ATTN"     "$FP_PORT" "$LOG_DIR/fp_server.log"   "$LOG_DIR/fp_eval.log"   FLASH_ATTN  ""
run_stage "TURBOQUANT_b4"  "$TQ_PORT" "$LOG_DIR/tq4_server.log"  "$LOG_DIR/tq4_eval.log"  TURBOQUANT  "TURBOQUANT_ALGO=prod TURBOQUANT_BITS=4"

# Optional third stage: paper §4.3 2.25-bit split config (32@b=3 + 96@b=2).
# Enabled when OUTLIER_MASK is set to an existing .pt from calibrate.sh.
STAGES_RUN=("FLASH_ATTN" "TURBOQUANT_b4")
if [ -n "${OUTLIER_MASK:-}" ]; then
    if [ ! -f "$OUTLIER_MASK" ]; then
        echo "[eval] OUTLIER_MASK set to $OUTLIER_MASK but file not found" >&2
        exit 1
    fi
    SPLIT_ENV="TURBOQUANT_ALGO=prod TURBOQUANT_OUTLIER_MASK=$OUTLIER_MASK"
    SPLIT_ENV="$SPLIT_ENV TURBOQUANT_BITS_OUTLIER=3 TURBOQUANT_BITS_REGULAR=2"
    run_stage "TURBOQUANT_split_2_25bit" "$TQ_PORT" \
        "$LOG_DIR/tq_split_server.log" "$LOG_DIR/tq_split_eval.log" \
        TURBOQUANT "$SPLIT_ENV"
    STAGES_RUN+=("TURBOQUANT_split_2_25bit")
fi

echo ""
echo "=========================== AGGREGATE ============================"
for tag in "${STAGES_RUN[@]}"; do
    # Map stage name -> eval log basename.
    case "$tag" in
        FLASH_ATTN)                 log="$LOG_DIR/fp_eval.log" ;;
        TURBOQUANT_b4)              log="$LOG_DIR/tq4_eval.log" ;;
        TURBOQUANT_split_2_25bit)   log="$LOG_DIR/tq_split_eval.log" ;;
        *) continue ;;
    esac
    if [ -f "$log" ]; then
        grep -E '=== NIAH grid|ctx|pos=|overall' "$log" | tail -n 20 || true
        echo ""
    fi
done
echo "=================================================================="
echo "[eval] done; all server groups have been stopped."
echo "[eval] per-stage logs under $LOG_DIR/"
