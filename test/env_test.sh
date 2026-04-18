#!/usr/bin/env bash
# Environment diagnostic for turboquant-lut: versions + a tiny CUDA matmul.
# No vLLM / no turboquant kernels here -- if a stage below fails, the
# problem is in the driver / torch / triton layer, not in our code.

set -u
cd "$(dirname "$0")/.."

GPU="${GPU:-3}"
export CUDA_VISIBLE_DEVICES="$GPU"

echo "=========================================================="
echo "1) NVIDIA driver"
echo "=========================================================="
nvidia-smi | head -3

echo ""
echo "=========================================================="
echo "2) Python package versions"
echo "=========================================================="
python3 -c "import torch; print(f'torch         : {torch.__version__}')"
python3 -c "import torch; print(f'torch.cuda    : {torch.version.cuda}')"
python3 -c "import torch; print(f'cudnn         : {torch.backends.cudnn.version()}')"
python3 -c "import torch; print(f'device        : {torch.cuda.get_device_name(0)}')"
python3 -c "import torch; cap = torch.cuda.get_device_capability(0); print(f'compute cap   : sm_{cap[0]}{cap[1]}')"
python3 -c "import triton; print(f'triton        : {triton.__version__}')" 2>&1

echo ""
echo "=========================================================="
echo "3) Tiny bf16 CUDA matmul"
echo "   If this crashes -> driver / torch mismatch, not vLLM."
echo "=========================================================="
python3 -c "
import torch
x = torch.randn(1024, 1024, device='cuda', dtype=torch.bfloat16)
y = x @ x
torch.cuda.synchronize()
print(f'matmul OK, y.sum() = {y.float().sum().item():.3f}')
"

echo ""
echo "=========================================================="
echo "4) Tiny Triton kernel smoke (no turboquant, no vLLM)"
echo "   @triton.jit needs a real source file -- use a temp .py"
echo "=========================================================="
TMP_SMOKE="$(mktemp --suffix=.py)"
trap 'rm -f "$TMP_SMOKE"' EXIT
cat > "$TMP_SMOKE" <<'PY'
import torch, triton
import triton.language as tl

@triton.jit
def _smoke(x_ptr, y_ptr, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    off = pid * BLOCK + tl.arange(0, BLOCK)
    mask = off < N
    x = tl.load(x_ptr + off, mask=mask)
    tl.store(y_ptr + off, x * 2.0, mask=mask)

N = 1024
x = torch.randn(N, device='cuda', dtype=torch.float32)
y = torch.empty_like(x)
_smoke[(N // 64,)](x, y, N, BLOCK=64)
torch.cuda.synchronize()
print(f'triton smoke OK, rel_err = {(y - 2*x).abs().max().item():.2e}')
PY
python3 "$TMP_SMOKE"

echo ""
echo "[env_test] all stages above should print 'OK'. If anything failed,"
echo "           the failing line localizes the layer -- driver, torch,"
echo "           triton, or CUDA matmul itself."
