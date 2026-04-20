#!/usr/bin/env bash
# Needle-In-A-Haystack eval reproducing the setup of Figure 4 in the
# TurboQuant paper (arXiv:2504.19874).
#
# Default 9-stage sweep (see STAGES= below):
#   FLASH_ATTN + {b3, b4, b4_t, b4_tfu, b5}
#              + {split_2_25bit, split_3_5bit, split_3_5bit_fu}
# Override via STAGES="..." to pick a subset. See vllm/turboquant/stages.py
# for every registered name; old _r / _rr suffixes still resolve with a
# deprecation warning.
#
# Grid matches Figure 4:
#   - 15 context lengths, log-spaced from 4k to 104k tokens:
#       4k, 5k, 6k, 8k, 10k, 13k, 16k, 20k, 26k, 32k, 41k, 51k, 65k, 82k, 104k
#   - 10 depth positions, uniformly from 0 to 1:
#       0, 0.111, ..., 0.889, 1.0
#   - 1 trial per (ctx, depth) cell  (150 requests per config; 450 total)
#
# Note: 104k requires max_model_len >= 104000. Set MAX_LEN and
# GPU_MEM_UTIL high enough to fit the longest KV cache. Our quant cache
# saves memory but vLLM still allocates its own bf16 kv_cache that we
# ignore; that buffer still counts against GPU_MEM_UTIL.
#
# Usage:
#   bash test/eval_figure4.sh
#   MAX_LEN=131072 GPU_MEM_UTIL=0.85 bash test/eval_figure4.sh
#   STAGES="FLASH_ATTN TURBOQUANT_prod_b4" bash test/eval_figure4.sh
#   # To probe up to 32k first (cheaper smoke test):
#   CTX=4096,5161,6500,8192,10321,13000,16385,20642,26000,32770 \
#     MAX_LEN=40960 bash test/eval_figure4.sh

set -u

MODEL="${MODEL:-/mnt/nvme3n1/g00872988/models/Llama-3.1-8B-Instruct}"
GPU="${GPU:-2}"
MAX_LEN="${MAX_LEN:-131072}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.85}"
OUTLIER_MASK="${OUTLIER_MASK:-/tmp/outliers_llama-3_1-8b_32.pt}"

# 15 log-spaced context lengths from 4k to 104k. Approximate integer token
# counts matching the paper's visible x-axis labels (4/8/10/16/26/41/65/104)
# at every-other position.
CTX="${CTX:-4096,5161,6500,8192,10321,13000,16385,20642,26000,32770,41285,52000,65540,82570,104000}"
# 10 depth positions from 0 to 1.
POSITIONS="${POSITIONS:-0,0.111,0.222,0.333,0.444,0.556,0.667,0.778,0.889,1.0}"
TRIALS="${TRIALS:-1}"
SEED="${SEED:-42}"
MAX_TOKENS="${MAX_TOKENS:-16}"
# 104k prefill + first-call Triton JIT can take minutes; bump generously.
REQUEST_TIMEOUT="${REQUEST_TIMEOUT:-3600}"

# Default 9-stage sweep (matches test/eval_longbench.sh):
# bit-width spectrum (b3/b4/b5) + tight-pack variants (b4_t / b4_tfu)
# + both split configs with fu variant.
STAGES="${STAGES:-FLASH_ATTN \
TURBOQUANT_b3 \
TURBOQUANT_b4 \
TURBOQUANT_b4_t \
TURBOQUANT_b4_tfu \
TURBOQUANT_b5 \
TURBOQUANT_split_2_25bit \
TURBOQUANT_split_3_5bit \
TURBOQUANT_split_3_5bit_fu}"

FP_PORT="${FP_PORT:-8010}"
TQ_PORT="${TQ_PORT:-8009}"

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
TS="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="$ROOT_DIR/logs/eval_figure4_$TS"
mkdir -p "$LOG_DIR"
ln -sfn "$LOG_DIR" "$ROOT_DIR/logs/eval_figure4_latest"

CURRENT_PGID=""

cleanup() {
    local code=$?
    if [ -n "$CURRENT_PGID" ]; then
        echo "[fig4] interrupted; stopping process group $CURRENT_PGID"
        kill -TERM -"$CURRENT_PGID" 2>/dev/null || true
        wait "$CURRENT_PGID" 2>/dev/null || true
    fi
    exit "$code"
}
trap cleanup EXIT INT TERM

wait_healthy() {
    local port="$1" log="$2" pid="$3"
    # 104k loads take a while; allow up to 20 minutes before giving up.
    for _ in $(seq 1 240); do
        if curl -sf "http://localhost:${port}/health" >/dev/null 2>&1; then
            return 0
        fi
        if ! ps -p "$pid" >/dev/null 2>&1; then
            echo "[fig4] server on port $port exited early; tail of log:" >&2
            tail -n 40 "$log" >&2
            return 1
        fi
        sleep 5
    done
    echo "[fig4] server on port $port never became healthy" >&2
    tail -n 40 "$log" >&2
    return 1
}

# $1 tag, $2 port, $3 server-log, $4 eval-log, $5 backend, $6 env
run_stage() {
    local tag="$1" port="$2" slog="$3" elog="$4" backend="$5" extra_env="$6"

    echo ""
    echo "==================================================================="
    echo "[fig4] stage: $tag  (port=$port backend=$backend env='$extra_env')"
    echo "[fig4] server log: $slog"
    echo "[fig4] eval   log: $elog"
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

    # Unbuffer Python stdout so per-cell progress lines stream in
    # real time when piped to tee (non-TTY default is 4 KB block buf).
    PYTHONUNBUFFERED=1 \
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

    echo "[fig4] stopping $tag process group $CURRENT_PGID"
    kill -TERM -"$CURRENT_PGID" 2>/dev/null || true
    wait "$CURRENT_PGID" 2>/dev/null || true
    CURRENT_PGID=""
    sleep 3
}

stage_args() {
    # Bridge to vllm/turboquant/stages.py via scripts/stage_env.py.
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

export CUDA_VISIBLE_DEVICES="$GPU"
echo "[fig4] model=$MODEL gpu=$GPU"
echo "[fig4] max_len=$MAX_LEN gpu_mem_util=$GPU_MEM_UTIL"
echo "[fig4] ctx=$CTX"
echo "[fig4] positions=$POSITIONS trials=$TRIALS"
echo "[fig4] stages: $STAGES"

for tag in $STAGES; do
    read -r backend extra_env < <(stage_args "$tag")
    port="$TQ_PORT"
    [ "$tag" = "FLASH_ATTN" ] && port="$FP_PORT"
    run_stage "$tag" "$port" \
        "$LOG_DIR/${tag}_server.log" \
        "$LOG_DIR/${tag}_eval.log" \
        "$backend" "$extra_env"
done

echo ""
echo "=========================== AGGREGATE ============================"
for tag in $STAGES; do
    elog="$LOG_DIR/${tag}_eval.log"
    if [ -f "$elog" ]; then
        echo "--- $tag ---"
        grep -E '=== NIAH grid|ctx|pos=|overall' "$elog" | tail -n 40 || true
        echo ""
    fi
done
echo "=================================================================="
echo "[fig4] done; all server groups stopped."
echo "[fig4] per-stage logs under $LOG_DIR/"
