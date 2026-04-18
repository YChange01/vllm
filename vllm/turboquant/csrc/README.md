# TurboQuant CUDA attend extension

Raw-CUDA / C++ implementation of the turboquant attend kernel with inline
dequant and WMMA tensor cores. Purpose: provide a starting point for
**FA3-style** optimization (Hopper/Blackwell `wgmma`, TMA, warp
specialization) that the Triton path can't fully reach.

## What's here today

* **Tensor-core matmul** via `nvcuda::wmma` (`mma.m16n8k16`) for both
  `Q @ K^T` and `P @ V`. Ampere+.
* **Inline dequant**: load 4-bit nibble-packed K/V idx, codebook
  gather to bf16, scale by per-slot norm, feed WMMA fragment.
* **mse path only** for now.

## Roadmap

| Stage | Contents | Status |
|---|---|---|
| **Today** | WMMA (mma.m16n8k16), mse only. JIT-compiled via `cpp_extension.load`. | shipped |
| Next | **prod / QJL** path (S·r_unit matvec, 1-bit qjl_sign bit-pack load) | follow-up |
| Next | **wgmma** on Hopper/Blackwell: larger `m64n256k16` tiles | follow-up |
| Final | **Warp specialization + TMA**: producer/consumer async pipeline (true FA3 architecture) | follow-up |

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

Set `TURBOQUANT_USE_CUDA=1`. The backend's attend dispatch replaces the
default Triton TC kernel with `turboquant_paged_attention_cuda`. First
use triggers a one-time JIT compile (~30-60s nvcc on B200, cached
thereafter).

## Build (manual)

The extension is JIT-compiled on first use. No manual build needed for
normal operation. For troubleshooting:

```bash
# Custom build cache
export TURBOQUANT_CUDA_BUILD_DIR=/tmp/turboquant_cuda_build

# Restrict to a single arch for faster nvcc
export TURBOQUANT_CUDA_ARCH=90           # e.g. H100 only

# Verbose nvcc output
export TURBOQUANT_CUDA_VERBOSE=1

# Trigger compile eagerly
python3 -c "from vllm.turboquant.attend_cuda import _ext; _ext()"
```

## Correctness smoke test

```bash
GPU=3 python3 test/test_cuda_vs_tc.py
```

Compares this kernel's output against the Triton TC kernel on random
Q/K/V. Expected: `max_abs_diff` around bf16 precision (~1.6e-2),
`mean_rel_diff` under 1%.

## Known limitations

* **mse only.** prod path raises `NotImplementedError` in the Python
  wrapper.
* **head_size = 128 hardcoded.** Change `HEAD_SIZE` in `attend_cuda.cu`
  if you need a different head dim.
* **BLOCK_N = 32** to stay under the default 48 KB static shared memory
  limit. Reaching 64 requires switching to dynamic shared memory +
  `cudaFuncSetAttribute(..., cudaFuncAttributeMaxDynamicSharedMemorySize)`.
* **No warp specialization** (producer/consumer). Single warp group,
  compute and memory serialized per tile.
* **No TMA.** Uses plain shared-memory loads.
