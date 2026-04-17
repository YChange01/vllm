#!/usr/bin/env bash
# One-shot diagnostic scratchpad. Overwritten every time.
#
# Current probe: verify the fix "V stored raw in bf16 (no int8 quant)"
# resolves the long-prompt explosion. Runs BYPASS + mse b=8 on the long
# prompt and shows per-layer BYPASS-vs-mse |attn|.mean ratios.
#   - Before fix: L0 ratio ~100x, median ~7x, max ~123x
#   - If fix works: ratios should drop dramatically (< 2x across all
#     layers) and output text should match FLASH_ATTN.

set -u

cd "$(dirname "$0")/.."

GPU="${GPU:-3}"
PROMPT="${PROMPT:-Machine learning has transformed many fields over the past decade with deep neural networks achieving remarkable performance on natural language understanding and speech synthesis.}"
MAX_TOKENS="${MAX_TOKENS:-4}"

echo "[temp] GPU=$GPU  MAX_TOKENS=$MAX_TOKENS"
echo "[temp] PROMPT=\"$PROMPT\""
echo ""

echo "=========================================================="
echo "1/2  BYPASS (long prompt)"
echo "=========================================================="
bash test/diag_bypass.sh "$PROMPT" "$MAX_TOKENS" "$GPU" || true

echo ""
echo "=========================================================="
echo "2/2  TURBOQUANT_ALGO=mse  TURBOQUANT_BITS=8  (long prompt)"
echo "=========================================================="
TURBOQUANT_ALGO=mse TURBOQUANT_BITS=8 \
    bash test/baseline.sh "$PROMPT" "$MAX_TOKENS" "$GPU" || true

BP="logs/diag_bypass_latest/turboquant_debug.log"
MSE="logs/baseline_latest/turboquant_debug_tq8.log"

echo ""
echo "=========================================================="
echo "Per-layer |attn|.mean/max   BYPASS vs mse b=8   call=0"
echo "(baseline for reference: before fix L0 ratio was 100x)"
echo "=========================================================="
python3 - "$BP" "$MSE" <<'PYEOF'
import re, sys
from statistics import median

def parse(path):
    rows = {}
    try:
        with open(path) as fh:
            for line in fh:
                m = re.search(
                    r"\[out\s+L(\d+)\] call=(\d+).*"
                    r"\|attn\|\.mean=([\d.eE+-]+) \|attn\|\.max=([\d.eE+-]+)",
                    line,
                )
                if m:
                    rows[(int(m.group(1)), int(m.group(2)))] = (
                        float(m.group(3)), float(m.group(4)))
    except FileNotFoundError:
        print(f"MISSING: {path}")
    return rows

bp = parse(sys.argv[1])
mse = parse(sys.argv[2])

print(f"{'L':>3}  {'BP mean':>9}  {'MSE mean':>9}  {'ratio':>7}   {'BP max':>7}  {'MSE max':>7}")
print("-" * 62)
ratios = {0: [], 1: [], 2: []}
for L in range(32):
    for c in (0, 1, 2):
        bp_v = bp.get((L, c))
        ms_v = mse.get((L, c))
        if bp_v is None or ms_v is None:
            continue
        bp_m, bp_max = bp_v
        ms_m, ms_max = ms_v
        r = ms_m / max(bp_m, 1e-12)
        ratios[c].append(r)
        if c == 0:
            tag = " EXPL" if r >= 10 else (" bad" if r >= 2 else "")
            print(f"L{L:<2}  {bp_m:9.4f}  {ms_m:9.4f}  {r:6.2f}x  {bp_max:7.3f}  {ms_max:7.3f}{tag}")

print()
print("=" * 62)
print("Summary across all 32 layers:")
for c in sorted(ratios):
    if ratios[c]:
        print(f"  call={c}  median={median(ratios[c]):6.2f}x   "
              f"max={max(ratios[c]):6.2f}x   n={len(ratios[c])}")
print()
print("Fix PASS criterion: all medians < 2x and all max < 5x")
PYEOF
