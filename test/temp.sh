#!/usr/bin/env bash
# One-shot diagnostic scratchpad. Overwritten every time.
#
# Current probe: verify the "attention sink breaks mse quant" hypothesis.
# Run BYPASS and mse b=8 with a LONG prompt (~30 tokens). If the bug is
# attention-sink-driven, the |attn|.mean ratio should drop substantially
# vs the short "Hello" case (which was ~22x at L1 prefill). A longer
# prompt dilutes the sink-dominance of each query because every query
# token now attends to many non-BOS keys.

set -u

cd "$(dirname "$0")/.."

GPU="${GPU:-3}"
# ~30 Llama-3 tokens. One sentence, no BOS-specific nuances.
PROMPT="${PROMPT:-Machine learning has transformed many fields over the past decade with deep neural networks achieving remarkable performance on natural language understanding and speech synthesis.}"
MAX_TOKENS="${MAX_TOKENS:-4}"

echo "[temp] GPU=$GPU  MAX_TOKENS=$MAX_TOKENS"
echo "[temp] PROMPT=\"$PROMPT\""
echo ""

echo "=========================================================="
echo "1/2  BYPASS baseline"
echo "=========================================================="
bash test/diag_bypass.sh "$PROMPT" "$MAX_TOKENS" "$GPU" || true

echo ""
echo "=========================================================="
echo "2/2  TURBOQUANT_ALGO=mse  TURBOQUANT_BITS=8"
echo "=========================================================="
TURBOQUANT_ALGO=mse TURBOQUANT_BITS=8 \
    bash test/baseline.sh "$PROMPT" "$MAX_TOKENS" "$GPU" || true

echo ""
echo "=========================================================="
echo "PER-(layer,call) |attn|.mean  BYPASS vs mse b=8"
echo "(long-prompt run; compare ratio to the short 'Hello' run)"
echo "=========================================================="

BP="logs/diag_bypass_latest/turboquant_debug.log"
MSE="logs/baseline_latest/turboquant_debug_tq8.log"

python3 - "$BP" "$MSE" <<'PYEOF'
import re, sys

def parse(path):
    rows = {}
    try:
        with open(path) as fh:
            for line in fh:
                m = re.match(
                    r"\[out\s+L(\d+)\]\s+call=(\d+).*?\|attn\|\.mean=([\d.]+)",
                    line,
                )
                if m:
                    rows[(int(m.group(1)), int(m.group(2)))] = float(m.group(3))
    except FileNotFoundError:
        print(f"MISSING: {path}")
    return rows

bp = parse(sys.argv[1])
mse = parse(sys.argv[2])

# Summary: median ratio by call (prefill vs decodes)
from statistics import median
by_call: dict[int, list[float]] = {}

print(f"{'layer':>6} {'call':>5} {'BYPASS':>10} {'mse_b8':>10} {'ratio':>8}")
print("-" * 44)
for key in sorted(set(bp) | set(mse)):
    L, c = key
    b = bp.get(key)
    m = mse.get(key)
    ratio = (m / b) if (b and m is not None and b > 1e-9) else None
    if ratio is not None:
        by_call.setdefault(c, []).append(ratio)
    b_s = f"{b:.4f}" if b is not None else "---"
    m_s = f"{m:.4f}" if m is not None else "---"
    r_s = f"{ratio:6.2f}x" if ratio is not None else "---"
    marker = ""
    if ratio is not None:
        if ratio >= 10:
            marker = "  <-- EXPLOSION"
        elif ratio >= 2:
            marker = "  <-- bad"
    print(f"L{L:<5} c{c:<4} {b_s:>10} {m_s:>10} {r_s:>8}{marker}")

print()
print("=" * 44)
print("Median ratio per call (across all 32 layers):")
for c in sorted(by_call):
    vals = by_call[c]
    print(f"  call={c}  n={len(vals):<2} median={median(vals):.2f}x  max={max(vals):.2f}x")
print()
print("Short 'Hello' run for reference:")
print("  call=0 (prefill=2)   median ~ 7-8x   max ~ 22x")
print("  call=1 (decode seq_len=3)  median ~ 5x")
print("  call=2 (decode seq_len=4)  median ~ 5-7x")
print()
print("If long-prompt median drops to ~1-2x -> sink hypothesis confirmed.")
print("If long-prompt still shows large ratios -> something else is wrong.")
PYEOF
