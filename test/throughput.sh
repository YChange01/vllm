#!/usr/bin/env bash
# Online throughput benchmark (``vllm bench serve``) across backends.
# Default stages (b=4 paper-faithful comparison):
#   - FLASH_ATTN                (bf16 reference, vLLM's own FA backend)
#   - TURBOQUANT_prod_b4        (homog Q_prod, ~5-bit storage)
#   - TURBOQUANT_split_3_5bit   (paper 4.3 split, 3.5-bit storage)
# Other available stages (add to STAGES=...):
#   - TURBOQUANT_mse_b4         (homog Q_mse, no QJL)
#   - TURBOQUANT_split_2_25bit  (paper 4.3 literal '2.5-bit')
#   - TURBOQUANT_mse_b4_cuda    (raw-CUDA WMMA attend kernel)
#
# Uses the ``random`` dataset with fixed input/output lengths. Each backend
# is served by its own setsid'd vllm process group; we ``kill -TERM -$PGID``
# at the end of each stage so the EngineCore goes with it.
#
# Usage:
#   bash test/throughput.sh
#   INPUT_LEN=2048 OUTPUT_LEN=256 NUM_PROMPTS=200 CONCURRENCY=32 bash test/throughput.sh
#   # request-rate=inf (default) means all requests fire at t=0 (max load).

set -u

MODEL="${MODEL:-/mnt/nvme3n1/g00872988/models/Llama-3.1-8B-Instruct}"
GPU="${GPU:-2}"
MAX_LEN="${MAX_LEN:-4096}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.3}"
OUTLIER_MASK="${OUTLIER_MASK:-/tmp/outliers_llama-3_1-8b_32.pt}"

INPUT_LEN="${INPUT_LEN:-1024}"
OUTPUT_LEN="${OUTPUT_LEN:-128}"
NUM_PROMPTS="${NUM_PROMPTS:-100}"
CONCURRENCY="${CONCURRENCY:-16}"
REQUEST_RATE="${REQUEST_RATE:-inf}"
NUM_WARMUPS="${NUM_WARMUPS:-5}"
SEED="${SEED:-42}"

# Which stages to run. Space-separated tags.
STAGES="${STAGES:-FLASH_ATTN TURBOQUANT_prod_b4 TURBOQUANT_split_3_5bit}"

FP_PORT=8010
TQ_PORT=8009

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
TS="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="$ROOT_DIR/logs/throughput_$TS"
mkdir -p "$LOG_DIR"
ln -sfn "$LOG_DIR" "$ROOT_DIR/logs/throughput_latest"

CURRENT_PGID=""

cleanup() {
    local code=$?
    if [ -n "$CURRENT_PGID" ]; then
        echo "[bench] interrupted; stopping process group $CURRENT_PGID"
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
            echo "[bench] server on port $port exited early; tail of log:" >&2
            tail -n 40 "$log" >&2
            return 1
        fi
        sleep 5
    done
    echo "[bench] server on port $port never became healthy" >&2
    tail -n 40 "$log" >&2
    return 1
}

# $1 tag, $2 port, $3 server-log, $4 bench-log, $5 backend, $6 env
run_stage() {
    local tag="$1" port="$2" slog="$3" blog="$4" backend="$5" extra_env="$6"

    echo ""
    echo "==================================================================="
    echo "[bench] stage: $tag  (port=$port backend=$backend env='$extra_env')"
    echo "[bench] server log: $slog"
    echo "[bench] bench  log: $blog"
    echo "==================================================================="
    : > "$slog"
    : > "$blog"

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

    vllm bench serve \
        --host 127.0.0.1 \
        --port "$port" \
        --model "$MODEL" \
        --dataset-name random \
        --random-input-len "$INPUT_LEN" \
        --random-output-len "$OUTPUT_LEN" \
        --num-prompts "$NUM_PROMPTS" \
        --max-concurrency "$CONCURRENCY" \
        --request-rate "$REQUEST_RATE" \
        --num-warmups "$NUM_WARMUPS" \
        --seed "$SEED" \
        --label "$tag" \
        --save-result \
        --result-dir "$LOG_DIR" \
        --result-filename "${tag}.json" \
        2>&1 | tee "$blog"

    echo "[bench] stopping $tag process group $CURRENT_PGID"
    kill -TERM -"$CURRENT_PGID" 2>/dev/null || true
    wait "$CURRENT_PGID" 2>/dev/null || true
    CURRENT_PGID=""
    sleep 3
}

stage_args() {
    # echo "backend env" for a given tag
    case "$1" in
        FLASH_ATTN)              echo "FLASH_ATTN  " ;;
        TURBOQUANT_mse_b4)       echo "TURBOQUANT  TURBOQUANT_ALGO=mse TURBOQUANT_BITS=4" ;;
        TURBOQUANT_prod_b4)      echo "TURBOQUANT  TURBOQUANT_ALGO=prod TURBOQUANT_BITS=4" ;;
        TURBOQUANT_prod_b4_r)    echo "TURBOQUANT  TURBOQUANT_ALGO=prod TURBOQUANT_BITS=4 TURBOQUANT_TIGHT_PACK=1" ;;
        TURBOQUANT_mse_b4_cuda)  echo "TURBOQUANT  TURBOQUANT_ALGO=mse TURBOQUANT_BITS=4 TURBOQUANT_USE_CUDA=1" ;;
        TURBOQUANT_prod_b4_cuda) echo "TURBOQUANT  TURBOQUANT_ALGO=prod TURBOQUANT_BITS=4 TURBOQUANT_USE_CUDA=1" ;;
        TURBOQUANT_split_3_5bit)
            if [ ! -f "${OUTLIER_MASK:-}" ]; then
                echo "[bench] stage $1 needs OUTLIER_MASK at $OUTLIER_MASK" >&2
                return 1
            fi
            echo "TURBOQUANT  TURBOQUANT_ALGO=prod TURBOQUANT_OUTLIER_MASK=$OUTLIER_MASK TURBOQUANT_BITS_OUTLIER=5 TURBOQUANT_BITS_REGULAR=3"
            ;;
        TURBOQUANT_split_3_5bit_r)
            if [ ! -f "${OUTLIER_MASK:-}" ]; then
                echo "[bench] stage $1 needs OUTLIER_MASK at $OUTLIER_MASK" >&2
                return 1
            fi
            echo "TURBOQUANT  TURBOQUANT_ALGO=prod TURBOQUANT_OUTLIER_MASK=$OUTLIER_MASK TURBOQUANT_BITS_OUTLIER=5 TURBOQUANT_BITS_REGULAR=3 TURBOQUANT_FP16_NORMS=1"
            ;;
        TURBOQUANT_split_3_5bit_rr)
            if [ ! -f "${OUTLIER_MASK:-}" ]; then
                echo "[bench] stage $1 needs OUTLIER_MASK at $OUTLIER_MASK" >&2
                return 1
            fi
            echo "TURBOQUANT  TURBOQUANT_ALGO=prod TURBOQUANT_OUTLIER_MASK=$OUTLIER_MASK TURBOQUANT_BITS_OUTLIER=5 TURBOQUANT_BITS_REGULAR=3 TURBOQUANT_FP16_NORMS=1 TURBOQUANT_UINT8_RNORM=1"
            ;;
        TURBOQUANT_split_2_25bit)
            if [ ! -f "${OUTLIER_MASK:-}" ]; then
                echo "[bench] stage $1 needs OUTLIER_MASK at $OUTLIER_MASK" >&2
                return 1
            fi
            echo "TURBOQUANT  TURBOQUANT_ALGO=prod TURBOQUANT_OUTLIER_MASK=$OUTLIER_MASK TURBOQUANT_BITS_OUTLIER=3 TURBOQUANT_BITS_REGULAR=2"
            ;;
        *) echo "[bench] unknown stage tag: $1" >&2; return 1 ;;
    esac
}

export CUDA_VISIBLE_DEVICES="$GPU"
echo "[bench] model=$MODEL gpu=$GPU"
echo "[bench] input=$INPUT_LEN output=$OUTPUT_LEN prompts=$NUM_PROMPTS "\
"concurrency=$CONCURRENCY rate=$REQUEST_RATE warmups=$NUM_WARMUPS"
echo "[bench] stages: $STAGES"

for tag in $STAGES; do
    read -r backend extra_env < <(stage_args "$tag")
    port="$TQ_PORT"
    [ "$tag" = "FLASH_ATTN" ] && port="$FP_PORT"
    run_stage "$tag" "$port" \
        "$LOG_DIR/${tag}_server.log" \
        "$LOG_DIR/${tag}_bench.log" \
        "$backend" "$extra_env"
done

echo ""
echo "=========================== AGGREGATE ============================"
printf "%-22s %12s %14s %14s %12s %12s\n" \
    "stage" "req/s" "input_tok/s" "output_tok/s" "mean_ttft" "mean_itl"
echo "------------------------------------------------------------------"
for tag in $STAGES; do
    blog="$LOG_DIR/${tag}_bench.log"
    [ -f "$blog" ] || continue
    # `vllm bench serve` prints a summary table with these lines:
    #   Request throughput (req/s):            X
    #   Input  token throughput (tok/s):       X
    #   Output token throughput (tok/s):       X
    #   Mean TTFT (ms):                        X
    #   Mean ITL  (ms):                        X
    rps=$(grep -E 'Request throughput'       "$blog" | tail -n1 | awk '{print $(NF)}')
    itps=$(grep -E 'Input token throughput'  "$blog" | tail -n1 | awk '{print $(NF)}')
    otps=$(grep -E 'Output token throughput' "$blog" | tail -n1 | awk '{print $(NF)}')
    ttft=$(grep -E 'Mean TTFT'               "$blog" | tail -n1 | awk '{print $(NF)}')
    itl=$(grep  -E 'Mean ITL'                "$blog" | tail -n1 | awk '{print $(NF)}')
    printf "%-22s %12s %14s %14s %12s %12s\n" \
        "$tag" "${rps:-?}" "${itps:-?}" "${otps:-?}" "${ttft:-?}" "${itl:-?}"
done
echo "=================================================================="
echo "[bench] done; all server groups stopped."
echo "[bench] per-stage logs + JSON under $LOG_DIR/"
