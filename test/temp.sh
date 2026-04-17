#!/usr/bin/env bash
# One-shot diagnostic scratchpad. Overwritten every time.
#
# Current probe: long prompt made L0 mse explode 100x (not attention sink).
# With enhanced logging (||k||, ||v||, ||q|| min/max/mean per layer, plus
# |attn|.max and NaN/Inf flag) we can now see whether the explosion is:
#   - at input  (||k|| or ||v|| has outliers that mismatch codebook)
#   - at output (|attn|.max >> bypass, or NaN/Inf in accumulation)

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
echo "L0 / L1 / L15 / L31 store+fwd+out  BYPASS vs mse b=8"
echo "=========================================================="

for L in 0 1 15 31; do
    echo ""
    echo "-- Layer $L  BYPASS --"
    grep " L$L\] " "$BP" 2>/dev/null
    echo ""
    echo "-- Layer $L  mse b=8 --"
    grep " L$L\] " "$MSE" 2>/dev/null
done

echo ""
echo "=========================================================="
echo "Summary: per-layer ||k||/||v||/|attn| stats (call=0 only)"
echo "=========================================================="
python3 - "$BP" "$MSE" <<'PYEOF'
import re, sys

def parse(path):
    rows = {}  # (layer, kind) -> dict of stats
    try:
        with open(path) as fh:
            for line in fh:
                m_store = re.search(
                    r"\[store L(\d+)\].*num_tokens=(\d+).*"
                    r"\|\|k\|\|=\(min=([\d.eE+-]+),max=([\d.eE+-]+),mean=([\d.eE+-]+)\).*"
                    r"\|\|v\|\|=\(min=([\d.eE+-]+),max=([\d.eE+-]+),mean=([\d.eE+-]+)\)",
                    line,
                )
                if m_store:
                    L = int(m_store.group(1))
                    key = (L, "store", int(m_store.group(2)))
                    rows[key] = dict(
                        k_min=float(m_store.group(3)),
                        k_max=float(m_store.group(4)),
                        k_mean=float(m_store.group(5)),
                        v_min=float(m_store.group(6)),
                        v_max=float(m_store.group(7)),
                        v_mean=float(m_store.group(8)),
                    )
                m_out = re.search(
                    r"\[out\s+L(\d+)\] call=(\d+).*"
                    r"\|attn\|\.mean=([\d.eE+-]+) \|attn\|\.max=([\d.eE+-]+)",
                    line,
                )
                if m_out:
                    L = int(m_out.group(1))
                    c = int(m_out.group(2))
                    key = (L, "out", c)
                    rows[key] = dict(
                        attn_mean=float(m_out.group(3)),
                        attn_max=float(m_out.group(4)),
                    )
    except FileNotFoundError:
        print(f"MISSING: {path}")
    return rows

bp = parse(sys.argv[1])
mse = parse(sys.argv[2])

# Table: per layer, show ||k|| and |attn| from both runs
print(f"{'L':>3} "
      f"{'BP |attn|mean':>14} {'MSE |attn|mean':>15} "
      f"{'BP |attn|max':>13} {'MSE |attn|max':>14} "
      f"{'BP ||k||max':>12} {'MSE ||k||max':>13}")
print("-" * 90)
for L in range(32):
    bp_out = bp.get((L, "out", 0), {})
    ms_out = mse.get((L, "out", 0), {})
    bp_st = next((v for k, v in bp.items() if k[0] == L and k[1] == "store"), {})
    ms_st = next((v for k, v in mse.items() if k[0] == L and k[1] == "store"), {})
    def fmt(x, w=10, p=3):
        return f"{x:{w}.{p}f}" if x is not None else "-" * w
    print(
        f"L{L:<2} "
        f"{fmt(bp_out.get('attn_mean'), 14, 4)} "
        f"{fmt(ms_out.get('attn_mean'), 15, 4)} "
        f"{fmt(bp_out.get('attn_max'), 13, 3)} "
        f"{fmt(ms_out.get('attn_max'), 14, 3)} "
        f"{fmt(bp_st.get('k_max'), 12, 3)} "
        f"{fmt(ms_st.get('k_max'), 13, 3)}"
    )
PYEOF
