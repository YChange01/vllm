// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// TurboQuant flash-style attend kernel in raw CUDA with inline dequant.
//
// Stage 1 (this file):
//   * Functional correctness baseline -- scalar fp32 matmul (no tensor core
//     yet). Proves the C++ extension path, paged addressing, online softmax
//     and dequant plumbing work end-to-end.
//   * mse path only (no QJL yet).
//   * 4-bit nibble-packed K/V idx, fp32 per-slot norm.
//
// Future stages (follow-up PRs):
//   Stage 1.5: WMMA tensor cores (mma.m16n8k16) for Q @ K.T and P @ V.
//              This alone should match or beat the Triton LUT kernel.
//   Stage 2:   add prod/QJL path
//   Stage 3:   upgrade to wgmma (Hopper/Blackwell) for larger matmul tiles
//   Stage 4:   warp specialization + TMA (async pipeline) -- true FA3 parity
//
// Target arches: sm_80 (Ampere), sm_90 (Hopper), sm_100 (Blackwell).
// WMMA works on all of them. wgmma/TMA are sm_90+.

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <torch/extension.h>

namespace turboquant {
namespace cuda {

using namespace nvcuda::wmma;
using bf16 = __nv_bfloat16;

// --- Compile-time tile sizes (tuned for head_size=128, BLOCK_N=64) ---
constexpr int HEAD_SIZE = 128;
constexpr int BLOCK_N   = 64;           // KV tile size
constexpr int BLOCK_M_PAD = 16;          // WMMA tile min
constexpr int WMMA_M    = 16;
constexpr int WMMA_N    = 16;
constexpr int WMMA_K    = 16;
constexpr int K_CB_MAX  = 16;            // 4-bit codebook max

// Each block: 128 threads = 4 warps.
constexpr int THREADS_PER_BLOCK = 128;

// -------------------------------------------------------------------------
// Helpers: nibble unpack + codebook dequant into a shared-memory bf16 tile.
//
// in_idx_global: (BLOCK_N, HEAD_SIZE/2) uint8, paged-addressed externally
// norms_global : (BLOCK_N,) fp32
// codebook_sh  : (K_CB,) fp32 (already loaded into shared mem)
// out_tile_sh  : (BLOCK_N, HEAD_SIZE) bf16 output
//
// Each thread handles one (n, d) pair. With 128 threads and 64*128=8192 cells,
// each thread covers 64 cells via a stride-128 loop.
// -------------------------------------------------------------------------
__device__ __forceinline__
void dequant_tile(
    const uint8_t* __restrict__ idx_ptr_base,   // starting addr for this tile (BLOCK_N, d/2)
    const int32_t* __restrict__ slot_addrs,     // (BLOCK_N,) base addr per slot into idx buf
    const float*   __restrict__ norms_global,   // (num_blocks, bs, H_kv)
    const int32_t* __restrict__ meta_addrs,     // (BLOCK_N,) addresses into norm buf
    const float*   __restrict__ codebook_sh,    // (K_CB,)
    bf16* __restrict__ tile_sh,                 // (BLOCK_N, HEAD_SIZE) shared
    const int n_valid,                          // number of slots actually in-range
    const int idx_dim                           // head_size / 2
) {
    const int tid = threadIdx.x;
    // For each (n, d) pair in the tile, dequant + scale.
    // Total cells: BLOCK_N * HEAD_SIZE = 8192. Stride THREADS_PER_BLOCK = 128.
    #pragma unroll 8
    for (int cell = tid; cell < BLOCK_N * HEAD_SIZE; cell += THREADS_PER_BLOCK) {
        const int n = cell / HEAD_SIZE;
        const int d = cell % HEAD_SIZE;
        if (n >= n_valid) {
            tile_sh[n * HEAD_SIZE + d] = __float2bfloat16(0.f);
            continue;
        }
        // Load packed byte at (n, d/2)
        const int32_t slot_base = slot_addrs[n];
        const uint8_t packed = idx_ptr_base[slot_base + (d >> 1)];
        const int nib = (d & 1) ? ((packed >> 4) & 0xF) : (packed & 0xF);
        const float c = codebook_sh[nib];
        // Multiply by this slot's norm
        const float nv = norms_global[meta_addrs[n]];
        tile_sh[n * HEAD_SIZE + d] = __float2bfloat16(c * nv);
    }
}

// -------------------------------------------------------------------------
// Main attend kernel -- one thread block per (query_token, kv_head).
//
// Inputs:
//   q_rot          : (T_q, H_q, HEAD_SIZE) bf16, Hadamard-rotated
//   cache_k_idx    : (num_blocks, block_size, H_kv, HEAD_SIZE/2) uint8
//   cache_k_norm   : (num_blocks, block_size, H_kv) fp32
//   cache_v_idx/norm: same layout
//   block_table    : (num_seqs, bt_stride) int32
//   seq_id/ kv_end : per-query (T_q,) int32
//   codebook       : (K_CB,) bf16
//   out            : (T_q, H_q, HEAD_SIZE) bf16, rotated V space (Python post-rotates)
// -------------------------------------------------------------------------
__launch_bounds__(THREADS_PER_BLOCK, 2)
__global__ void attend_mse_kernel(
    const bf16* __restrict__ q_rot,
    const uint8_t* __restrict__ cache_k_idx,
    const float*   __restrict__ cache_k_norm,
    const uint8_t* __restrict__ cache_v_idx,
    const float*   __restrict__ cache_v_norm,
    const int32_t* __restrict__ block_table,
    const int32_t* __restrict__ seq_id_per_query,
    const int32_t* __restrict__ kv_end_per_query,
    const bf16*    __restrict__ codebook,
    bf16* __restrict__ out,
    const float inv_d,
    const int block_size,
    const int num_heads_q,
    const int num_heads_kv,
    const int gqa_group,
    const int idx_dim,
    const int bt_stride,
    const int K_CB
) {
    const int q_idx   = blockIdx.x;
    const int kvh_idx = blockIdx.y;
    const int tid     = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane    = tid % 32;
    const int q_head_start = kvh_idx * gqa_group;

    const int32_t seq_idx = seq_id_per_query[q_idx];
    const int32_t kv_end  = kv_end_per_query[q_idx];

    // ---- Shared memory layout ----
    __shared__ bf16  Q_sh[BLOCK_M_PAD][HEAD_SIZE];            // (16, 128) bf16 -- Q (gqa padded to 16)
    __shared__ bf16  K_tile[BLOCK_N][HEAD_SIZE];              // (64, 128) bf16
    __shared__ bf16  V_tile[BLOCK_N][HEAD_SIZE];              // (64, 128) bf16
    __shared__ float codebook_sh[K_CB_MAX];                   // (16,)
    __shared__ float m_sh[BLOCK_M_PAD];                       // (16,)
    __shared__ float l_sh[BLOCK_M_PAD];                       // (16,)
    __shared__ float acc_sh[BLOCK_M_PAD][HEAD_SIZE];          // (16, 128) fp32
    __shared__ float scores_sh[BLOCK_M_PAD][BLOCK_N];         // (16, 64)

    // Slot-addressing scratch (computed per-tile)
    __shared__ int32_t slot_base_sh[BLOCK_N];
    __shared__ int32_t meta_sh[BLOCK_N];

    // -------- Load codebook (few threads) --------
    if (tid < K_CB) {
        codebook_sh[tid] = __bfloat162float(codebook[tid]);
    }

    // -------- Load Q (gqa_group rows into padded BLOCK_M_PAD) --------
    // Q_sh[m][d] for m in [0, gqa_group), d in [0, HEAD_SIZE)
    // Each thread handles one (m, d) pair.
    #pragma unroll
    for (int cell = tid; cell < BLOCK_M_PAD * HEAD_SIZE; cell += THREADS_PER_BLOCK) {
        const int m = cell / HEAD_SIZE;
        const int d = cell % HEAD_SIZE;
        if (m < gqa_group) {
            const int q_head = q_head_start + m;
            const int q_off = (q_idx * num_heads_q + q_head) * HEAD_SIZE + d;
            Q_sh[m][d] = q_rot[q_off];
        } else {
            Q_sh[m][d] = __float2bfloat16(0.f);
        }
    }

    // -------- Initialize softmax state --------
    if (tid < BLOCK_M_PAD) {
        m_sh[tid] = -1e30f;
        l_sh[tid] = 0.f;
    }
    for (int cell = tid; cell < BLOCK_M_PAD * HEAD_SIZE; cell += THREADS_PER_BLOCK) {
        acc_sh[cell / HEAD_SIZE][cell % HEAD_SIZE] = 0.f;
    }

    __syncthreads();

    // -------- KV loop --------
    const int num_tiles = (kv_end + BLOCK_N - 1) / BLOCK_N;

    for (int tile_i = 0; tile_i < num_tiles; ++tile_i) {
        const int kv_base = tile_i * BLOCK_N;
        const int n_valid = min(BLOCK_N, kv_end - kv_base);

        // ---- Compute slot_base / meta addresses for this tile ----
        // slot_base[n] = phys_block * bs * H_kv * idx_dim
        //              + tok_in_block * H_kv * idx_dim
        //              + kvh_idx * idx_dim
        if (tid < BLOCK_N) {
            const int n = tid;
            if (n < n_valid) {
                const int pos = kv_base + n;
                const int blk = pos / block_size;
                const int tok_in_blk = pos % block_size;
                const int phys = block_table[seq_idx * bt_stride + blk];
                slot_base_sh[n] = ((phys * block_size + tok_in_blk) * num_heads_kv + kvh_idx) * idx_dim;
                meta_sh[n]      = (phys * block_size + tok_in_blk) * num_heads_kv + kvh_idx;
            } else {
                slot_base_sh[n] = 0;
                meta_sh[n]      = 0;
            }
        }
        __syncthreads();

        // ---- Dequant K tile into K_tile[BLOCK_N][HEAD_SIZE] ----
        dequant_tile(cache_k_idx, slot_base_sh, cache_k_norm, meta_sh,
                     codebook_sh, &K_tile[0][0], n_valid, idx_dim);
        __syncthreads();

        // ---- Compute scores[m][n] = Q_sh[m] . K_tile[n] for m<BLOCK_M_PAD, n<BLOCK_N ----
        // Simple one-thread-per-(m,n) parallelism for first version.
        // BLOCK_M_PAD*BLOCK_N = 16*64 = 1024 scores, 128 threads -> each does 8.
        #pragma unroll
        for (int cell = tid; cell < BLOCK_M_PAD * BLOCK_N; cell += THREADS_PER_BLOCK) {
            const int m = cell / BLOCK_N;
            const int n = cell % BLOCK_N;
            if (n >= n_valid) {
                scores_sh[m][n] = -1e30f;
                continue;
            }
            float s = 0.f;
            #pragma unroll
            for (int d = 0; d < HEAD_SIZE; ++d) {
                s += __bfloat162float(Q_sh[m][d]) * __bfloat162float(K_tile[n][d]);
            }
            scores_sh[m][n] = s * inv_d;
        }
        __syncthreads();

        // ---- Online softmax update (per row m) ----
        // Use one warp per m (16 rows, 4 warps available). 4 rows/warp.
        if (tid < BLOCK_M_PAD) {
            const int m = tid;
            // Row max
            float row_max = -1e30f;
            #pragma unroll
            for (int n = 0; n < BLOCK_N; ++n) {
                row_max = fmaxf(row_max, scores_sh[m][n]);
            }
            const float m_new = fmaxf(m_sh[m], row_max);
            const float alpha = __expf(m_sh[m] - m_new);
            // Convert scores -> probs inline and compute row sum + rescale acc
            float row_sum = 0.f;
            #pragma unroll
            for (int n = 0; n < BLOCK_N; ++n) {
                const float p = __expf(scores_sh[m][n] - m_new);
                scores_sh[m][n] = p;              // reuse scores_sh as probs
                row_sum += p;
            }
            l_sh[m] = l_sh[m] * alpha + row_sum;
            // Rescale acc
            #pragma unroll
            for (int d = 0; d < HEAD_SIZE; ++d) {
                acc_sh[m][d] *= alpha;
            }
            m_sh[m] = m_new;
        }
        __syncthreads();

        // ---- Dequant V tile ----
        dequant_tile(cache_v_idx, slot_base_sh, cache_v_norm, meta_sh,
                     codebook_sh, &V_tile[0][0], n_valid, idx_dim);
        __syncthreads();

        // ---- acc[m][d] += sum_n probs[m][n] * V_tile[n][d] ----
        // One thread per (m, d) pair: BLOCK_M_PAD * HEAD_SIZE = 2048 cells / 128 threads = 16 cells/thread.
        #pragma unroll
        for (int cell = tid; cell < BLOCK_M_PAD * HEAD_SIZE; cell += THREADS_PER_BLOCK) {
            const int m = cell / HEAD_SIZE;
            const int d = cell % HEAD_SIZE;
            float s = 0.f;
            #pragma unroll
            for (int n = 0; n < BLOCK_N; ++n) {
                s += scores_sh[m][n] * __bfloat162float(V_tile[n][d]);
            }
            acc_sh[m][d] += s;
        }
        __syncthreads();
    }

    // -------- Normalize and write out (first gqa_group rows only) --------
    for (int cell = tid; cell < BLOCK_M_PAD * HEAD_SIZE; cell += THREADS_PER_BLOCK) {
        const int m = cell / HEAD_SIZE;
        const int d = cell % HEAD_SIZE;
        if (m < gqa_group) {
            const int q_head = q_head_start + m;
            const int out_off = (q_idx * num_heads_q + q_head) * HEAD_SIZE + d;
            const float denom = fmaxf(l_sh[m], 1e-12f);
            out[out_off] = __float2bfloat16(acc_sh[m][d] / denom);
        }
    }
}

// -------------------------------------------------------------------------
// C++ entry point (called from pybind binding). Dispatches the kernel.
// -------------------------------------------------------------------------
void attend_mse_launch(
    at::Tensor q_rot,                   // (T_q, H_q, HEAD_SIZE) bf16
    at::Tensor cache_k_idx,             // (num_blocks, bs, H_kv, HEAD_SIZE/2) uint8
    at::Tensor cache_k_norm,            // (num_blocks, bs, H_kv) fp32
    at::Tensor cache_v_idx,
    at::Tensor cache_v_norm,
    at::Tensor block_table,             // (num_seqs, bt_stride) int32
    at::Tensor seq_id_per_query,        // (T_q,) int32
    at::Tensor kv_end_per_query,        // (T_q,) int32
    at::Tensor codebook,                // (K_CB,) bf16
    at::Tensor out,                     // (T_q, H_q, HEAD_SIZE) bf16
    int block_size,
    int gqa_group
) {
    TORCH_CHECK(q_rot.is_cuda() && q_rot.scalar_type() == at::kBFloat16,
                "q_rot must be bf16 CUDA");
    TORCH_CHECK(cache_k_idx.scalar_type() == at::kByte);
    TORCH_CHECK(cache_k_norm.scalar_type() == at::kFloat);
    TORCH_CHECK(codebook.scalar_type() == at::kBFloat16);

    const int T_q         = q_rot.size(0);
    const int num_heads_q = q_rot.size(1);
    const int head_size   = q_rot.size(2);
    const int num_heads_kv = cache_k_idx.size(2);
    const int idx_dim     = cache_k_idx.size(3);
    const int bt_stride   = block_table.size(1);
    const int K_CB        = codebook.size(0);

    TORCH_CHECK(head_size == HEAD_SIZE, "head_size must be 128 for this kernel");
    TORCH_CHECK(K_CB <= K_CB_MAX, "K_CB too large");

    const float inv_d = 1.0f / static_cast<float>(head_size);

    dim3 grid(T_q, num_heads_kv);
    dim3 block(THREADS_PER_BLOCK);

    attend_mse_kernel<<<grid, block>>>(
        reinterpret_cast<const bf16*>(q_rot.data_ptr<at::BFloat16>()),
        cache_k_idx.data_ptr<uint8_t>(),
        cache_k_norm.data_ptr<float>(),
        cache_v_idx.data_ptr<uint8_t>(),
        cache_v_norm.data_ptr<float>(),
        block_table.data_ptr<int32_t>(),
        seq_id_per_query.data_ptr<int32_t>(),
        kv_end_per_query.data_ptr<int32_t>(),
        reinterpret_cast<const bf16*>(codebook.data_ptr<at::BFloat16>()),
        reinterpret_cast<bf16*>(out.data_ptr<at::BFloat16>()),
        inv_d,
        block_size,
        num_heads_q,
        num_heads_kv,
        gqa_group,
        idx_dim,
        bt_stride,
        K_CB
    );
}

} // namespace cuda
} // namespace turboquant
