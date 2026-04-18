#!/usr/bin/env bash
# GSM8K accuracy eval across FLASH_ATTN / TurboQuant (TC) / TurboQuant (CUDA).
# Starts a vLLM server per backend, runs test/eval_gsm8k.py against it,
# tears it down, prints side-by-side accuracy.
#
# Usage:
#   bash test/eval_gsm8k.sh                                # defaults
#   N=200 bash test/eval_gsm8k.sh                          # more samples
#   TURBOQUANT_ALGO=prod N=100 bash test/eval_gsm8k.sh     # prod path
#   GPU=2 STAGES="FLASH_ATTN TURBOQUANT_CUDA" bash test/eval_gsm8k.sh
#
# Stages recognized (whitespace-separated list):
#   FLASH_ATTN          -- vLLM native FA backend (fp reference)
#   TURBOQUANT_TC       -- Triton tensor-core attend
#   TURBOQUANT_CUDA     -- raw CUDA WMMA + cp.async attend
#
# Default STAGES runs all three.

set -u

MODEL="${MODEL:-/mnt/nvme3n1/g00872988/models/Llama-3.1-8B-Instruct}"
N="${N:-100}"
MAX_TOKENS="${MAX_TOKENS:-512}"
CONCURRENCY="${CONCURRENCY:-16}"
GPU="${GPU:-3}"
MAX_LEN="${MAX_LEN:-4096}"
ALGO="${TURBOQUANT_ALGO:-mse}"
STAGES="${STAGES:-FLASH_ATTN TURBOQUANT_TC TURBOQUANT_CUDA}"

FP_PORT=8010
TQ_PORT=8009

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
TS="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="$ROOT_DIR/logs/gsm8k_$TS"
mkdir -p "$LOG_DIR"
ln -sfn "gsm8k_$TS" "$ROOT_DIR/logs/gsm8k_latest"

CURRENT_PGID=""
cleanup() {
    local code=$?
    if [ -n "$CURRENT_PGID" ]; then
        echo "[gsm8k] interrupted; stopping process group $CURRENT_PGID"
        kill -TERM -"$CURRENT_PGID" 2>/dev/null || true
        wait "$CURRENT_PGID" 2>/dev/null || true
    fi
    exit "$code"
}
trap cleanup EXIT INT TERM

wait_healthy() {
    local port="$1" log="$2" pid="$3"
    for _ in $(seq 1 72); do       # up to 6 min for first-time JIT
        if curl -sf "http://localhost:${port}/health" >/dev/null 2>&1; then
            return 0
        fi
        if ! ps -p "$pid" >/dev/null 2>&1; then
            echo "[gsm8k] server on port $port exited early; tail of log:" >&2
            tail -n 40 "$log" >&2
            return 1
        fi
        sleep 5
    done
    echo "[gsm8k] server on port $port never became healthy" >&2
    tail -n 40 "$log" >&2
    return 1
}

# $1 label, $2 port, $3 server log, $4 backend, $5 extra env, $6 eval log
run_stage() {
    local label="$1" port="$2" srv_log="$3" backend="$4" extra_env="$5" eval_log="$6"

    echo ""
    echo "[gsm8k] starting $label on port $port ..."
    : > "$srv_log"

    if [ -n "$extra_env" ]; then
        env $extra_env setsid vllm serve "$MODEL" \
            --port "$port" \
            --enforce-eager \
            --attention-backend "$backend" \
            --max-model-len "$MAX_LEN" \
            --gpu-memory-utilization 0.3 \
            >>"$srv_log" 2>&1 &
    else
        setsid vllm serve "$MODEL" \
            --port "$port" \
            --enforce-eager \
            --attention-backend "$backend" \
            --max-model-len "$MAX_LEN" \
            --gpu-memory-utilization 0.3 \
            >>"$srv_log" 2>&1 &
    fi
    CURRENT_PGID=$!

    if ! wait_healthy "$port" "$srv_log" "$CURRENT_PGID"; then
        exit 1
    fi

    echo "[gsm8k] $label server healthy; running eval_gsm8k.py n=$N"
    python3 "$ROOT_DIR/test/eval_gsm8k.py" \
        --port "$port" \
        --model "$MODEL" \
        --n "$N" \
        --max_tokens "$MAX_TOKENS" \
        --concurrency "$CONCURRENCY" \
        2>&1 | tee "$eval_log"

    echo "[gsm8k] stopping $label process group $CURRENT_PGID"
    kill -TERM -"$CURRENT_PGID" 2>/dev/null || true
    wait "$CURRENT_PGID" 2>/dev/null || true
    CURRENT_PGID=""
    sleep 3
}

# ---- main ----------------------------------------------------------------
export CUDA_VISIBLE_DEVICES="$GPU"
echo "[gsm8k] model=$MODEL n=$N max_tokens=$MAX_TOKENS concurrency=$CONCURRENCY"
echo "[gsm8k] gpu=$GPU algo=$ALGO stages=$STAGES"
echo "[gsm8k] logs dir: $LOG_DIR"

declare -A STAGE_LOGS

for stage in $STAGES; do
    eval_log="$LOG_DIR/${stage,,}_eval.log"
    srv_log="$LOG_DIR/${stage,,}_server.log"
    STAGE_LOGS["$stage"]="$eval_log"

    case "$stage" in
        FLASH_ATTN)
            run_stage "$stage" "$FP_PORT" "$srv_log" FLASH_ATTN "" "$eval_log"
            ;;
        TURBOQUANT_TC)
            run_stage "$stage" "$TQ_PORT" "$srv_log" TURBOQUANT \
                "TURBOQUANT_ALGO=$ALGO TURBOQUANT_BITS=4" "$eval_log"
            ;;
        TURBOQUANT_CUDA)
            run_stage "$stage" "$TQ_PORT" "$srv_log" TURBOQUANT \
                "TURBOQUANT_ALGO=$ALGO TURBOQUANT_BITS=4 TURBOQUANT_USE_CUDA=1" \
                "$eval_log"
            ;;
        *)
            echo "[gsm8k] unknown stage '$stage'; skipping" >&2
            ;;
    esac
done

echo ""
echo "========================== AGGREGATE =========================="
echo "n=$N  algo=$ALGO  concurrency=$CONCURRENCY"
printf "%-22s  %-s\n" "stage" "accuracy"
printf "%-22s  %-s\n" "----------------------" "----------"
for stage in $STAGES; do
    log="${STAGE_LOGS[$stage]:-}"
    if [[ -n "$log" && -f "$log" ]]; then
        acc_line=$(grep '^  accuracy' "$log" | tail -1 | sed 's/^  accuracy *: *//')
        printf "%-22s  %s\n" "$stage" "${acc_line:-<parse failed>}"
    else
        printf "%-22s  <no log>\n" "$stage"
    fi
done
echo "==============================================================="
echo "[gsm8k] done. Per-stage logs under $LOG_DIR/"
