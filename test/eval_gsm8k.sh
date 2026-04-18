#!/usr/bin/env bash
# GSM8K accuracy eval via lm-evaluation-harness across FLASH_ATTN /
# TurboQuant (TC) / TurboQuant (CUDA). Starts a vLLM server per backend,
# runs `lm_eval --model local-completions --tasks gsm8k`, tears down.
#
# Mirrors the pattern in tests/entrypoints/openai/correctness/test_lmeval.py
# (the vLLM-upstream reference for lm-eval integration).
#
# Prerequisite: `pip install lm-eval` in the active env.
#
# Usage:
#   bash test/eval_gsm8k.sh                                # defaults: N=100, mse
#   N=200 bash test/eval_gsm8k.sh                          # more samples
#   N=-1 bash test/eval_gsm8k.sh                           # full 1319 test set
#   TURBOQUANT_ALGO=prod bash test/eval_gsm8k.sh           # prod path
#   STAGES="FLASH_ATTN TURBOQUANT_CUDA" bash test/eval_gsm8k.sh  # subset
#
# Stage labels:
#   FLASH_ATTN        FP reference (vLLM native FA3)
#   TURBOQUANT_TC     Triton tensor-core attend
#   TURBOQUANT_CUDA   raw CUDA WMMA + cp.async attend

set -u

MODEL="${MODEL:-/mnt/nvme3n1/g00872988/models/Llama-3.1-8B-Instruct}"
N="${N:-100}"               # <=0 -> full test set (1319); lm_eval --limit
NUM_FEWSHOT="${NUM_FEWSHOT:-5}"
NUM_CONCURRENT="${NUM_CONCURRENT:-64}"
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

# lm_eval's --limit expects a positive integer; translate N<=0 to full set.
if [[ "$N" -le 0 ]]; then
    LIMIT_ARG=""
else
    LIMIT_ARG="--limit $N"
fi

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

# $1 label, $2 port, $3 server log, $4 backend, $5 extra env, $6 eval out dir
run_stage() {
    local label="$1" port="$2" srv_log="$3" backend="$4" extra_env="$5" out_dir="$6"

    echo ""
    echo "[gsm8k] starting $label on port $port ..."
    : > "$srv_log"
    mkdir -p "$out_dir"

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

    echo "[gsm8k] $label server healthy; running lm_eval (gsm8k, ${NUM_FEWSHOT}-shot, limit=${N})"
    local url="http://localhost:${port}/v1/completions"
    local model_args="model=${MODEL},base_url=${url},num_concurrent=${NUM_CONCURRENT},tokenized_requests=False"

    # lm_eval writes results JSON under output_path
    lm_eval \
        --model local-completions \
        --model_args "$model_args" \
        --tasks gsm8k \
        --num_fewshot "$NUM_FEWSHOT" \
        $LIMIT_ARG \
        --output_path "$out_dir" \
        2>&1 | tee "$out_dir/stdout.log"

    echo "[gsm8k] stopping $label process group $CURRENT_PGID"
    kill -TERM -"$CURRENT_PGID" 2>/dev/null || true
    wait "$CURRENT_PGID" 2>/dev/null || true
    CURRENT_PGID=""
    sleep 3
}

# Parse lm_eval output dir for exact_match score. lm_eval writes results
# under <out_dir>/<something>/results_*.json; just walk the tree.
parse_acc() {
    local out_dir="$1"
    local json
    json=$(find "$out_dir" -name 'results_*.json' 2>/dev/null | head -1)
    if [[ -z "$json" || ! -f "$json" ]]; then
        # Fallback: grep the stdout table ("exact_match")
        grep -E '\|gsm8k\|.*exact_match' "$out_dir/stdout.log" 2>/dev/null \
            | head -1 \
            | awk -F'|' '{print $7}' \
            | xargs \
            || echo "<no result>"
        return
    fi
    python3 -c "
import json, sys
d = json.load(open('$json'))
r = d['results']['gsm8k']
# report strict-match exact_match (gsm8k's primary metric)
for k, v in r.items():
    if k.startswith('exact_match') and 'strict' in k:
        print(f'{v:.4f}'); sys.exit(0)
for k, v in r.items():
    if k.startswith('exact_match'):
        print(f'{v:.4f}'); sys.exit(0)
print('<no exact_match>')
"
}

# ---- main ----------------------------------------------------------------
export CUDA_VISIBLE_DEVICES="$GPU"
echo "[gsm8k] model=$MODEL  N=$N  num_fewshot=$NUM_FEWSHOT  num_concurrent=$NUM_CONCURRENT"
echo "[gsm8k] gpu=$GPU algo=$ALGO stages=$STAGES"
echo "[gsm8k] logs dir: $LOG_DIR"

declare -A STAGE_OUT

for stage in $STAGES; do
    out_dir="$LOG_DIR/${stage,,}"
    srv_log="$LOG_DIR/${stage,,}_server.log"
    STAGE_OUT["$stage"]="$out_dir"

    case "$stage" in
        FLASH_ATTN)
            run_stage "$stage" "$FP_PORT" "$srv_log" FLASH_ATTN "" "$out_dir"
            ;;
        TURBOQUANT_TC)
            run_stage "$stage" "$TQ_PORT" "$srv_log" TURBOQUANT \
                "TURBOQUANT_ALGO=$ALGO TURBOQUANT_BITS=4" "$out_dir"
            ;;
        TURBOQUANT_CUDA)
            run_stage "$stage" "$TQ_PORT" "$srv_log" TURBOQUANT \
                "TURBOQUANT_ALGO=$ALGO TURBOQUANT_BITS=4 TURBOQUANT_USE_CUDA=1" "$out_dir"
            ;;
        *)
            echo "[gsm8k] unknown stage '$stage'; skipping" >&2
            ;;
    esac
done

echo ""
echo "========================== AGGREGATE =========================="
echo "N=$N  algo=$ALGO  num_fewshot=$NUM_FEWSHOT  (task=gsm8k, strict-match)"
printf "%-22s  %s\n" "stage" "exact_match"
printf "%-22s  %s\n" "----------------------" "-----------"
for stage in $STAGES; do
    out_dir="${STAGE_OUT[$stage]:-}"
    if [[ -n "$out_dir" && -d "$out_dir" ]]; then
        acc=$(parse_acc "$out_dir")
        printf "%-22s  %s\n" "$stage" "$acc"
    else
        printf "%-22s  <no output>\n" "$stage"
    fi
done
echo "==============================================================="
echo "[gsm8k] done. Per-stage logs + JSON under $LOG_DIR/"
