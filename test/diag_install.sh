#!/usr/bin/env bash
# Find out whether `vllm serve` is using our git-modified source or
# an installed site-packages version.
#
# If turboquant_attn.__file__ does NOT resolve to this repo, every
# change we push is being ignored by the running server.

set -u

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
echo "[diag-install] repo root: $ROOT_DIR"
echo ""
echo "[diag-install] which python3:"
which python3
echo ""
echo "[diag-install] which vllm:"
which vllm || true
echo ""
echo "[diag-install] python sys.path (first 10):"
python3 -c "import sys; [print(p) for p in sys.path[:10]]"
echo ""
echo "[diag-install] location of vllm package:"
python3 -c "import vllm; print(vllm.__file__)"
echo ""
echo "[diag-install] location of TurboQuant backend module:"
python3 -c "
import vllm.v1.attention.backends.turboquant_attn as m
print('file:', m.__file__)
print('has _dbg_write:', hasattr(m, '_dbg_write'))
print('has TURBOQUANT_BYPASS:', hasattr(m, 'TURBOQUANT_BYPASS'))
print('has _DEBUG_CANDIDATE_PATHS:', hasattr(m, '_DEBUG_CANDIDATE_PATHS'))
"
echo ""
echo "[diag-install] are we picking up our repo code?"
EXPECTED="$ROOT_DIR/vllm/v1/attention/backends/turboquant_attn.py"
ACTUAL=$(python3 -c "import vllm.v1.attention.backends.turboquant_attn as m; print(m.__file__)")
echo "  expected: $EXPECTED"
echo "  actual:   $ACTUAL"
if [ "$EXPECTED" = "$ACTUAL" ]; then
    echo "  -> MATCH: server should be using your edits."
else
    echo "  -> MISMATCH: vllm is loading from site-packages."
    echo "     Reinstall in editable mode:"
    echo "       VLLM_USE_PRECOMPILED=1 pip install -e ."
fi
