#!/usr/bin/env bash
# Outlier channel calibration for TurboQuant split mode (paper §4.3).
#
# Prereq:
#   scripts/dump_wikitext2.py on a connected machine -> wikitext2_calib.jsonl
#   Upload that file to $TEXTS_FILE below.
#
# Usage:
#   bash scripts/calibrate.sh                      # defaults
#   GPU=2 NUM_OUTLIERS=32 bash scripts/calibrate.sh
#   MODEL=/path/to/other/model bash scripts/calibrate.sh

set -euo pipefail

MODEL="${MODEL:-/mnt/nvme3n1/g00872988/models/Llama-3.1-8B-Instruct}"
GPU="${GPU:-2}"
NUM_SAMPLES="${NUM_SAMPLES:-128}"
SEQ_LEN="${SEQ_LEN:-2048}"
NUM_OUTLIERS="${NUM_OUTLIERS:-32}"

# Default location for the pre-downloaded text samples. Generated on a
# connected machine via scripts/dump_wikitext2.py; user uploads manually.
TEXTS_FILE="${TEXTS_FILE:-calib_data/wikitext2_calib.jsonl}"

# Tag the output by model basename + num_outliers so repeat runs with
# different mask sizes don't clobber each other.
MODEL_TAG="$(basename "$MODEL" | tr '[:upper:]' '[:lower:]' \
             | sed 's/-instruct$//' | tr '.' '_')"
OUTPUT="${OUTPUT:-/tmp/outliers_${MODEL_TAG}_${NUM_OUTLIERS}.pt}"

if [ ! -f "$TEXTS_FILE" ]; then
    echo "[calibrate] texts file not found: $TEXTS_FILE" >&2
    echo "[calibrate] Generate it on a connected machine:" >&2
    echo "            python3 scripts/dump_wikitext2.py" >&2
    echo "[calibrate] Then upload to this host at $TEXTS_FILE" >&2
    exit 1
fi

echo "[calibrate] model=$MODEL gpu=$GPU"
echo "[calibrate] texts=$TEXTS_FILE  samples=$NUM_SAMPLES  seq_len=$SEQ_LEN"
echo "[calibrate] num_outliers=$NUM_OUTLIERS"
echo "[calibrate] output=$OUTPUT"

CUDA_VISIBLE_DEVICES="$GPU" python3 scripts/calibrate_outliers.py \
    --model "$MODEL" \
    --texts-file "$TEXTS_FILE" \
    --num-samples "$NUM_SAMPLES" \
    --seq-len "$SEQ_LEN" \
    --num-outliers "$NUM_OUTLIERS" \
    --device cuda:0 \
    --output "$OUTPUT"

echo ""
echo "[calibrate] done -> $OUTPUT"
ls -la "$OUTPUT"
