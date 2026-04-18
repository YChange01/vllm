# TurboQuant CUDA attend extension

Raw-CUDA / C++ implementation of the turboquant attend kernel with inline
dequant. Purpose: provide a starting point for **FA3-style** optimization
(Hopper/Blackwell tensor cores, TMA, warp specialization) that the Triton
path can't reach.

## Stages

| Stage | Contents | Status |
|---|---|---|
| **1** | Scalar fp32 matmul baseline. mse path only. Builds via JIT (`cpp_extension.load`). | **this PR** |
| 1.5 | WMMA tensor cores (`mma.m16n8k16`) for Q·K^T and P·V. Ampere+. | next |
| 2 | prod / QJL path (fused S·r_unit matvec, sign bit-pack load). | follow-up |
| 3 | wgmma (Hopper+): larger matmul tiles, higher tensor-core throughput. | follow-up |
| 4 | Warp specialization + TMA (true FA3 architecture: async pipeline). | follow-up |

## Files

```
csrc/
  attend_cuda.cu    - kernel + launch entry (attend_mse_launch)
  binding.cpp       - pybind11 module (exposes attend_mse)
  README.md         - this file
```

Python side:
```
vllm/turboquant/attend_cuda.py   - JIT loader + Python wrapper
```

## How it's wired

Set `TURBOQUANT_USE_CUDA=1` at module import time. Backend's attend
dispatch picks `turboquant_paged_attention_cuda` over the Triton LUT /
base paths. First use triggers a one-time JIT compile of the CUDA
extension (~30s on B200 with nvcc, cached thereafter).

The precedence is CUDA > LUT > base, so CUDA overrides LUT if both env
vars are set.

## Build (manual)

The extension is JIT-compiled on first use. No manual build is needed
for normal operation. If you want to pre-build or troubleshoot:

```bash
# Set build directory explicitly
export TURBOQUANT_CUDA_BUILD_DIR=/tmp/turboquant_cuda_build

# Restrict to a single arch for faster nvcc
export TURBOQUANT_CUDA_ARCH=90    # H100 only

# Verbose nvcc output
export TURBOQUANT_CUDA_VERBOSE=1

# Trigger compile
python3 -c "from vllm.turboquant.attend_cuda import _ext; _ext()"
```

## Correctness smoke test

```bash
GPU=3 python3 test/test_cuda_vs_triton.py
```

Compares this kernel's output vs the Triton `attend.py` base kernel on
random Q/K/V. Expected: `max_abs_diff` around bf16 precision (~1.6e-2),
`mean_rel_diff` under 1%.

## Known limitations (Stage 1)

* **No tensor core yet.** The kernel uses scalar fp32 multiply-adds.
  Expect it to be ~same speed as the Triton base kernel, slower than
  the Triton LUT kernel. This is a correctness baseline; Stage 1.5 adds
  WMMA and should leapfrog both.
* **mse only.** prod/QJL is a `NotImplementedError` in the Python wrapper.
* **head_size = 128 hardcoded.** Change `HEAD_SIZE` in `attend_cuda.cu`
  if you need a different head dim.
* **Single warpgroup.** No producer/consumer warp specialization yet.
* **No TMA.** Uses plain shared-memory loads.
