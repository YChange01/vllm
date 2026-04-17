#!/usr/bin/env bash
# One-shot diagnostic scratchpad. Overwritten every time.
#
# Current probe: L0 is fine, L1 onwards explodes. We want:
#   (1) per-(layer,call) |attn|.mean side-by-side BYPASS vs mse_b8
#   (2) full L1 trajectory (store+fwd+out) from both runs for direct compare

set -u

cd "$(dirname "$0")/.."

BP="logs/diag_bypass_latest/turboquant_debug.log"
MSE="logs/baseline_latest/turboquant_debug_tq8.log"

echo "============================================================"
echo "(1) per-(layer,call) |attn|.mean  BYPASS vs mse b=8"
echo "    ratio > 2  = bad; ratio >= 10 = explosion"
echo "============================================================"
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

print(f"{'layer':>6} {'call':>5} {'BYPASS':>10} {'mse_b8':>10} {'ratio':>8}")
print("-" * 44)
for key in sorted(set(bp) | set(mse)):
    L, c = key
    b = bp.get(key)
    m = mse.get(key)
    if b is not None and m is not None and b > 1e-9:
        ratio = m / b
    else:
        ratio = None
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
PYEOF

echo ""
echo "============================================================"
echo "(2) L1 full trajectory  BYPASS"
echo "============================================================"
grep " L1\] " "$BP" 2>/dev/null || echo "(nothing)"

echo ""
echo "============================================================"
echo "(2) L1 full trajectory  mse b=8"
echo "============================================================"
grep " L1\] " "$MSE" 2>/dev/null || echo "(nothing)"

echo ""
echo "============================================================"
echo "(3) L0 store K magnitudes vs L1 store K magnitudes (mse b=8)"
echo "    are L1 K values much larger / outlier-y than L0?"
echo "============================================================"
grep -E "^\[store L[01]\] " "$MSE" 2>/dev/null
