#!/usr/bin/env bash
# Full comparison of TurboQuant stages vs FLASH_ATTN:
#   compression ratio + bytes/token + bytes/128tok
#   + req/s + output/total token throughput + mean TTFT/ITL
#
# Runs test/throughput.sh for the chosen STAGES (each stage gets its
# own vllm serve + vllm bench serve), then aggregates the results into
# a single table via scripts/compare_stages.py.
#
# Usage:
#   bash test/compare.sh                    # default: 12-stage sweep
#   STAGES="FLASH_ATTN TURBOQUANT_b4" \
#     bash test/compare.sh                  # pick your own subset
#   INPUT_LEN=2048 OUTPUT_LEN=256 NUM_PROMPTS=200 \
#     bash test/compare.sh                  # longer workload

set -u

# Default sweep: the 4 axes the user wanted to compare.
#   b=2,3,4,5 homog
#   b=4 with each flag combination
#   split 2.25 / 3.5 with each flag combination
#   FLASH_ATTN baseline
export STAGES="${STAGES:-FLASH_ATTN \
TURBOQUANT_b2 \
TURBOQUANT_b3 \
TURBOQUANT_b4 \
TURBOQUANT_b4_t \
TURBOQUANT_b4_tf \
TURBOQUANT_b4_tfu \
TURBOQUANT_b5 \
TURBOQUANT_split_2_25bit \
TURBOQUANT_split_3_5bit \
TURBOQUANT_split_3_5bit_f \
TURBOQUANT_split_3_5bit_fu}"

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

echo "[compare] running throughput.sh over ${STAGES}"
bash "$ROOT_DIR/test/throughput.sh"

echo ""
echo "[compare] aggregating into unified table"
python3 "$ROOT_DIR/scripts/compare_stages.py" \
    --log-dir "$ROOT_DIR/logs/throughput_latest" \
    --stages  "$STAGES" \
    --csv     "$ROOT_DIR/logs/throughput_latest/compare.csv"
