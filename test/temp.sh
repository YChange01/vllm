#!/usr/bin/env bash
# Diagnostic + kernel-level sanity for turboquant-lut, bypassing vLLM serve.
# Use when vllm serve throws "driver too old" or other env errors -- this
# script narrows down whether the failure is in the CUDA/torch layer, the
# Triton kernels, or the vLLM integration.

set -u
cd "$(dirname "$0")/.."

GPU="${GPU:-3}"
export CUDA_VISIBLE_DEVICES="$GPU"
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

echo "=========================================================="
echo "1) Environment versions"
echo "=========================================================="
nvidia-smi | head -3
echo ""
python3 - <<'PY'
import torch
print(f"torch          : {torch.__version__}")
print(f"torch.cuda     : {torch.version.cuda}")
print(f"cudnn          : {torch.backends.cudnn.version()}")
print(f"device         : {torch.cuda.get_device_name(0)}")
print(f"compute cap    : sm_{''.join(str(x) for x in torch.cuda.get_device_capability(0))}")
try:
    import triton
    print(f"triton         : {triton.__version__}")
except Exception as e:
    print(f"triton         : FAILED TO IMPORT: {e}")
PY

echo ""
echo "=========================================================="
echo "2) Torch CUDA sanity (tiny bf16 matmul)"
echo "   If this crashes with driver/runtime errors, issue is NOT vLLM."
echo "=========================================================="
python3 - <<'PY' || true
import torch
x = torch.randn(1024, 1024, device='cuda', dtype=torch.bfloat16)
y = x @ x
torch.cuda.synchronize()
print(f"matmul OK, y.sum() = {y.float().sum().item():.3f}")
PY

echo ""
echo "=========================================================="
echo "3) Standalone Triton kernel test (no vLLM runtime)"
echo "   Validates turboquant store + attend kernels on random data."
echo "=========================================================="
python3 test/test_varlen_kernel.py --num-tokens 64 || true

echo ""
echo "=========================================================="
echo "4) Numerical equivalence: base attend vs LUT attend"
echo "=========================================================="
python3 test/test_lut_vs_base.py || true

echo ""
echo "=========================================================="
echo "5) (Only try if 1-4 all pass) vLLM serve baseline"
echo "=========================================================="
echo "# To run it manually after this script:"
echo "#     GPU=$GPU bash test/baseline.sh"
