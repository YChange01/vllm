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
| 1.5 | WMMA (mma.m16n8k16), mse only. JIT-compiled. 65ms ITL (slower than TC, see throughput log 2026-04-18) | shipped |
| **3A** | **wgmma** replaces WMMA. Single warpgroup, sync cp.async loads. Target: ≤40ms (TC parity). | **next** |
| 3B | cp.async double-buffer loads, still single warpgroup. Target: ≤30ms. | planned |
| 3C | Producer/consumer warp specialization. TMA for Q + idx/norm arrays, producer decodes bf16 K/V into smem staging, consumer wgmma. + prod/QJL merged. Target: ≤22ms. | planned |

## Stage 3C design (full FA3 form, dequant-aware)

Reference: `vllm-project/flash-attention@f5bc33c` `hopper/mainloop_fwd_sm90_tma_gmma_ws.hpp`
— we mirror its structure but cannot reuse it directly: vanilla FA3 TMAs bf16 K/V
straight from gmem to smem, our K/V live as 4-bit idx + fp32 norm + codebook
gather. TMA cannot gather, so the load→dequant→mma pipeline gains a middle stage.

### Warp group layout (Hopper sm_90a, 3 warpgroups × 128 threads = 384)

| WG | Role | Work |
|---|---|---|
| 0 | **Producer** | TMA load Q (once), paged TMA load of K/V idx+norm tiles, decode idx+norm → bf16 K/V tile in smem staging, signal consumer via `mbarrier` |
| 1 | **Consumer A** | wgmma QK on (Q, decoded K) → scores; online softmax running max/sum; wgmma PV on (P, decoded V) → accumulator |
| 2 | **Consumer B** | Same as A but on next tile (pingpong: while A is in softmax, B runs wgmma and vice versa) |

### Shared memory layout (per SM, ≤228 KB dynamic smem on Hopper)

```
smem {
  Q_tile          [BLOCK_M × HEAD_SIZE] bf16     # 16 × 128 × 2 =  4 KB, 1 stage
  K_staging       [STAGES × BLOCK_N × HEAD_SIZE] bf16   # 2 × 64 × 128 × 2 = 32 KB
  V_staging       [STAGES × BLOCK_N × HEAD_SIZE] bf16   # same = 32 KB
  K_idx_raw       [STAGES × BLOCK_N × HEAD_SIZE/8] uint32  # 4-bit nibbles packed
  K_norm_raw      [STAGES × BLOCK_N × HEAD_SIZE/G] fp32
  V_idx_raw       same shape
  V_norm_raw      same shape
  codebook        [2**b × HEAD_SIZE] bf16  # persistent, reused all blocks
  softmax_scratch [BLOCK_M × 2] fp32       # max, sum per row
  mbar_load_k[STAGES], mbar_decode_k[STAGES], ...
}
```

### Pipeline (per KV tile of BLOCK_N tokens)

```
producer (WG0)                     consumer A (WG1)            consumer B (WG2)
---------------                    ------------------          ------------------
TMA load idx/norm tile i   ---->   ~wait on decode_k[i]~       ~running on tile i-1~
decode K[i], V[i]          ---->   wgmma QK                    softmax pass
mbarrier.arrive decode[i]          pingpong swap
advance stage                      wgmma PV
...
```

### Dequant math (in producer)

mse: `K_bf16[n,d] = codebook[K_idx[n,d]] * K_norm[n,d/G]`
prod: `K_bf16[n,d] = codebook[K_idx[n,d]] * K_norm[n,d/G] + (sqrt(π/2)/d) * dot(S[d,:], qjl_sign[n,:])`

Both compile from the same kernel via `enum Algo { MSE, PROD }` template parameter — no runtime branch.

### What we don't get that vanilla FA3 has

- TMA cannot stream bf16 K/V directly (we gather). Producer WG is doing real work, not just issuing TMA descriptors. This caps our speedup below vanilla FA3.
- Can't reuse FA3's `CollectiveMainloopFwdSm90` template — it assumes contiguous K/V. We use CUTLASS pipeline + mbarrier primitives directly.

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
