#!/usr/bin/env bash
# One-shot diagnostic scratchpad. Overwritten every time. Do NOT rely
# on stable contents across sessions.
#
# Current probe: extract the three comparison cuts from the PYTHON-SIDE
# debug files (turboquant_debug*.log), which are the clean source --
# no vllm "(EngineCore pid=xxx)" prefix, no "[TURBOQUANT_DBG]" prefix.

set -u

cd "$(dirname "$0")/.."

BP_DBG="logs/diag_bypass_latest/turboquant_debug.log"
TQ8_DBG="logs/baseline_latest/turboquant_debug_tq8.log"

echo "=========================================================="
echo "(a) BYPASS per-layer |attn|.mean  --  reference"
echo "=========================================================="
if [ -s "$BP_DBG" ]; then
    grep "^\[out " "$BP_DBG"
else
    echo "(missing $BP_DBG)"
fi

echo ""
echo "=========================================================="
echo "(b) mse b=8 per-layer |attn|.mean  --  KEY DIFF vs (a)"
echo "=========================================================="
if [ -s "$TQ8_DBG" ]; then
    grep "^\[out " "$TQ8_DBG"
else
    echo "(missing $TQ8_DBG)"
fi

echo ""
echo "=========================================================="
echo "(c) mse b=8 layer-0 full store/fwd/out trajectory"
echo "=========================================================="
if [ -s "$TQ8_DBG" ]; then
    grep " L0\] " "$TQ8_DBG"
else
    echo "(missing $TQ8_DBG)"
fi

echo ""
echo "=========================================================="
echo "(d) module-load marker (confirm algo/bits)"
echo "=========================================================="
grep -H "^# module" "$BP_DBG" "$TQ8_DBG" 2>/dev/null
