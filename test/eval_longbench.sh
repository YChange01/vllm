#!/usr/bin/env bash
# LongBench evaluation harness. For each stage: start a vllm server,
# run test/eval_longbench.py against every task in --data-dir, tear
# the server down, move on. Produces one eval log per stage with a
# per-task score table.
#
# Default stages mirror eval.sh defaults (paper-faithful KV-quant
# configurations).
#
# Usage:
#   bash test/eval_longbench.sh                       # all defaults
#   STAGES="TURBOQUANT_b4 TURBOQUANT_split_3_5bit" \
#     bash test/eval_longbench.sh
#   TASKS="narrativeqa,qasper" MAX_SAMPLES=20 \
#     bash test/eval_longbench.sh                     # smoke run
#   DATA_DIR=calib_data/longbench_v1 \
#     SUBSET=mini bash test/eval_longbench.sh         # paper-like subset

set -u

MODEL="${MODEL:-/mnt/nvme3n1/g00872988/models/Llama-3.1-8B-Instruct}"
GPU="${GPU:-2}"
MAX_LEN="${MAX_LEN:-32768}"     # LongBench max context is ~32k
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.5}"
OUTLIER_MASK="${OUTLIER_MASK:-/tmp/outliers_llama-3_1-8b_32.pt}"

# Data + task selection
DATA_DIR="${DATA_DIR:-calib_data/longbench_v1}"
TASKS="${TASKS:-}"               # empty -> all tasks in DATA_DIR
SUBSET="${SUBSET:-}"             # used when TASKS empty (informational)
MAX_SAMPLES="${MAX_SAMPLES:-}"   # empty -> all

# Stage list -- see vllm/turboquant/stages.py for names.
STAGES="${STAGES:-FLASH_ATTN TURBOQUANT_b4 TURBOQUANT_split_3_5bit}"

REQUEST_TIMEOUT="${REQUEST_TIMEOUT:-1800}"
SEED="${SEED:-42}"

FP_PORT="${FP_PORT:-8010}"
TQ_PORT="${TQ_PORT:-8009}"

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
TS="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="$ROOT_DIR/logs/longbench_$TS"
mkdir -p "$LOG_DIR"
ln -sfn "$LOG_DIR" "$ROOT_DIR/logs/longbench_latest"

CURRENT_PGID=""

cleanup() {
    local code=$?
    if [ -n "$CURRENT_PGID" ]; then
        echo "[lb] interrupted; stopping process group $CURRENT_PGID"
        kill -TERM -"$CURRENT_PGID" 2>/dev/null || true
        wait "$CURRENT_PGID" 2>/dev/null || true
    fi
    exit "$code"
}
trap cleanup EXIT INT TERM

wait_healthy() {
    local port="$1" log="$2" pid="$3"
    # LongBench prompts can be 32k tokens; JIT compile may take a while
    # on the first call. 10 minutes is the startup budget (the first
    # inference gets the full REQUEST_TIMEOUT window on top).
    for _ in $(seq 1 120); do
        if curl -sf "http://localhost:${port}/health" >/dev/null 2>&1; then
            return 0
        fi
        if ! ps -p "$pid" >/dev/null 2>&1; then
            echo "[lb] server on port $port exited early; tail of log:" >&2
            tail -n 40 "$log" >&2
            return 1
        fi
        sleep 5
    done
    echo "[lb] server on port $port never became healthy" >&2
    tail -n 40 "$log" >&2
    return 1
}

stage_args() {
    # Bridge to scripts/stage_env.py for env vars; FLASH_ATTN special.
    local tag="$1"
    if [ "$tag" = "FLASH_ATTN" ]; then
        echo "FLASH_ATTN  "
        return 0
    fi
    local extra_env
    extra_env=$(python3 "$ROOT_DIR/scripts/stage_env.py" "$tag" \
                ${OUTLIER_MASK:+--outlier-mask "$OUTLIER_MASK"})
    if [ $? -ne 0 ]; then return 1; fi
    echo "TURBOQUANT  $extra_env"
}

run_stage() {
    local tag="$1" port="$2" slog="$3" elog="$4" backend="$5" extra_env="$6"

    echo ""
    echo "==================================================================="
    echo "[lb] stage: $tag  (port=$port backend=$backend env='$extra_env')"
    echo "[lb] server log: $slog"
    echo "[lb] eval   log: $elog"
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

    local details="$LOG_DIR/${tag}_details.json"
    # Unbuffer Python stdout: when piped to tee (non-TTY), block-
    # buffering can swallow progress prints for minutes at a time.
    PYTHONUNBUFFERED=1 \
    python3 "$ROOT_DIR/test/eval_longbench.py" \
        --endpoint "http://localhost:${port}" \
        --model "$MODEL" \
        --data-dir "$DATA_DIR" \
        --tag "$tag" \
        --request-timeout "$REQUEST_TIMEOUT" \
        --max-context "$MAX_LEN" \
        ${TASKS:+--tasks "$TASKS"} \
        ${MAX_SAMPLES:+--max-samples "$MAX_SAMPLES"} \
        --save-details "$details" \
        2>&1 | tee "$elog"

    echo "[lb] stopping $tag process group $CURRENT_PGID"
    kill -TERM -"$CURRENT_PGID" 2>/dev/null || true
    wait "$CURRENT_PGID" 2>/dev/null || true
    CURRENT_PGID=""
    sleep 3
}

export CUDA_VISIBLE_DEVICES="$GPU"
echo "[lb] model=$MODEL gpu=$GPU max_len=$MAX_LEN"
echo "[lb] data_dir=$DATA_DIR tasks='${TASKS:-all}' max_samples='${MAX_SAMPLES:-all}'"
echo "[lb] stages: $STAGES"

if [ ! -d "$DATA_DIR" ]; then
    echo "[lb] data dir not found: $DATA_DIR" >&2
    echo "     Run: python3 scripts/dump_longbench.py --output $DATA_DIR" >&2
    exit 1
fi

STAGES_RUN=()
for tag in $STAGES; do
    cfg="$(stage_args "$tag")" || exit 1
    read -r backend extra_env <<< "$cfg"
    port="$TQ_PORT"
    [ "$tag" = "FLASH_ATTN" ] && port="$FP_PORT"
    short=$(echo "$tag" | sed 's/^TURBOQUANT_/tq_/' | tr '[:upper:]' '[:lower:]')
    run_stage "$tag" "$port" \
        "$LOG_DIR/${short}_server.log" \
        "$LOG_DIR/${short}_eval.log" \
        "$backend" "$extra_env"
    STAGES_RUN+=("$tag:$LOG_DIR/${short}_eval.log")
done

echo ""
echo "=========================== AGGREGATE ============================"
for entry in "${STAGES_RUN[@]}"; do
    tag="${entry%%:*}"
    log="${entry#*:}"
    if [ -f "$log" ]; then
        echo "--- $tag ---"
        grep -E 'OVERALL|LongBench results' "$log" | tail -n 5 || true
        echo ""
    fi
done
echo "=================================================================="
echo "[lb] done; per-stage logs + detail JSONs under $LOG_DIR/"
