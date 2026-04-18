// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// TurboQuant flash-style attend kernel in raw CUDA with WMMA tensor cores.
//
// Pipeline (per thread block, one per (query_token, kv_head) pair):
//   1. Load Q into shared memory; dequant 16-entry codebook
//   2. For each KV tile of BLOCK_N=64 slots:
//        a. Build paged slot addresses
//        b. Dequant K idx (4-bit) + codebook gather + norm scale -> K_sh bf16
//        c. WMMA: scores = Q_sh @ K_sh.T  (tensor core)
//        d. Mask invalid slots; scale by 1/d
//        e. Online softmax: row max -> alpha -> probs (bf16 P_sh); update l
//        f. Rescale acc_sh *= alpha
//        g. Dequant V similarly -> V_sh bf16
//        h. WMMA: acc_sh += P_sh @ V_sh  (tensor core)
//   3. Normalize acc_sh / l_sh and write out (rotated V space).
//
// Uses nvcuda::wmma (mma.m16n8k16 family) for both matmuls. Requires sm_80+.
// Hopper/Blackwell also execute this fine, but the big wgmma tiles and TMA
// asynchronous loads are left for a follow-up PR.
//
// Scope: mse path only. prod/QJL is a follow-up.

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <mma.h>
#include <torch/extension.h>

namespace turboquant {
namespace cuda {

using namespace nvcuda::wmma;
using bf16 = __nv_bfloat16;

// --- Compile-time tile sizes (tuned for head_size=128, BLOCK_N=64) ---
constexpr int HEAD_SIZE = 128;
constexpr int BLOCK_N   = 64;
constexpr int BLOCK_M_PAD = 16;             // WMMA min
constexpr int WMMA_M    = 16;
constexpr int WMMA_N    = 16;
constexpr int WMMA_K    = 16;
constexpr int K_CB_MAX  = 16;
constexpr int THREADS   = 128;              // 4 warps
constexpr int WARPS     = THREADS / 32;     // 4

// ---------------------------------------------------------------------------
// Dequant one KV tile into shared memory as bf16.
// ---------------------------------------------------------------------------
__device__ __forceinline__
void dequant_tile(
    const uint8_t* __restrict__ idx_buf,
    const int32_t* __restrict__ slot_base_sh,
    const float*   __restrict__ norms_buf,
    const int32_t* __restrict__ meta_sh,
    const float*   __restrict__ codebook_sh,
    bf16* __restrict__ tile_sh,
    const int n_valid
) {
    const int tid = threadIdx.x;
    // 64 * 128 = 8192 cells; 128 threads -> 64 cells/thread.
    #pragma unroll 8
    for (int cell = tid; cell < BLOCK_N * HEAD_SIZE; cell += THREADS) {
        const int n = cell / HEAD_SIZE;
        const int d = cell % HEAD_SIZE;
        if (n >= n_valid) {
            tile_sh[n * HEAD_SIZE + d] = __float2bfloat16(0.f);
            continue;
        }
        const int32_t slot_off = slot_base_sh[n] + (d >> 1);
        const uint8_t packed = idx_buf[slot_off];
        const int nib = (d & 1) ? ((packed >> 4) & 0xF) : (packed & 0xF);
        const float c = codebook_sh[nib];
        const float nv = norms_buf[meta_sh[n]];
        tile_sh[n * HEAD_SIZE + d] = __float2bfloat16(c * nv);
    }
}

// ---------------------------------------------------------------------------
// Main attend kernel (WMMA tensor cores for both QK and PV matmuls).
// ---------------------------------------------------------------------------
__launch_bounds__(THREADS, 2)
__global__ void attend_mse_tc_kernel(
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
    const int q_head_start = kvh_idx * gqa_group;

    const int32_t seq_idx = seq_id_per_query[q_idx];
    const int32_t kv_end  = kv_end_per_query[q_idx];

    // --- Shared memory (16-byte aligned for WMMA) ---
    __shared__ __align__(16) bf16  Q_sh    [BLOCK_M_PAD][HEAD_SIZE];       // (16, 128)
    __shared__ __align__(16) bf16  K_sh    [BLOCK_N][HEAD_SIZE];           // (64, 128)
    __shared__ __align__(16) bf16  V_sh    [BLOCK_N][HEAD_SIZE];           // (64, 128)
    __shared__ __align__(16) bf16  P_sh    [BLOCK_M_PAD][BLOCK_N];         // (16, 64)
    __shared__ float scores_sh[BLOCK_M_PAD][BLOCK_N];                      // (16, 64)
    __shared__ float codebook_sh[K_CB_MAX];
    __shared__ float m_sh   [BLOCK_M_PAD];
    __shared__ float l_sh   [BLOCK_M_PAD];
    __shared__ float alpha_sh[BLOCK_M_PAD];
    __shared__ float acc_sh [BLOCK_M_PAD][HEAD_SIZE];                      // (16, 128)
    __shared__ int32_t slot_base_sh[BLOCK_N];
    __shared__ int32_t meta_sh     [BLOCK_N];

    // --- Load codebook (tiny) ---
    if (tid < K_CB) codebook_sh[tid] = __bfloat162float(codebook[tid]);

    // --- Load Q (gqa_group rows into padded 16) ---
    for (int cell = tid; cell < BLOCK_M_PAD * HEAD_SIZE; cell += THREADS) {
        const int m = cell / HEAD_SIZE;
        const int d = cell % HEAD_SIZE;
        if (m < gqa_group) {
            const int q_head = q_head_start + m;
            const int q_off  = (q_idx * num_heads_q + q_head) * HEAD_SIZE + d;
            Q_sh[m][d] = q_rot[q_off];
        } else {
            Q_sh[m][d] = __float2bfloat16(0.f);
        }
    }

    // --- Init softmax state + acc ---
    if (tid < BLOCK_M_PAD) {
        m_sh[tid] = -1e30f;
        l_sh[tid] = 0.f;
    }
    for (int cell = tid; cell < BLOCK_M_PAD * HEAD_SIZE; cell += THREADS) {
        acc_sh[cell / HEAD_SIZE][cell % HEAD_SIZE] = 0.f;
    }
    __syncthreads();

    // ========================= KV LOOP =========================
    const int num_tiles = (kv_end + BLOCK_N - 1) / BLOCK_N;

    for (int tile = 0; tile < num_tiles; ++tile) {
        const int kv_base = tile * BLOCK_N;
        const int n_valid = min(BLOCK_N, kv_end - kv_base);

        // --- Paged addresses ---
        if (tid < BLOCK_N) {
            const int n = tid;
            if (n < n_valid) {
                const int pos = kv_base + n;
                const int blk = pos / block_size;
                const int tok = pos % block_size;
                const int phys = block_table[seq_idx * bt_stride + blk];
                slot_base_sh[n] = ((phys * block_size + tok) * num_heads_kv + kvh_idx) * idx_dim;
                meta_sh[n]      = (phys * block_size + tok) * num_heads_kv + kvh_idx;
            } else {
                slot_base_sh[n] = 0;
                meta_sh[n]      = 0;
            }
        }
        __syncthreads();

        // --- Dequant K ---
        dequant_tile(cache_k_idx, slot_base_sh, cache_k_norm, meta_sh,
                     codebook_sh, &K_sh[0][0], n_valid);
        __syncthreads();

        // --- WMMA Q @ K.T -> scores_sh ---
        // Each warp owns one (16x16) tile of scores:
        //   warp w -> scores[0:16, 16*w : 16*(w+1)]
        // M = BLOCK_M_PAD = 16, N = BLOCK_N = 64, K = HEAD_SIZE = 128.
        {
            fragment<matrix_a, WMMA_M, WMMA_N, WMMA_K, bf16, row_major> a_frag;
            fragment<matrix_b, WMMA_M, WMMA_N, WMMA_K, bf16, col_major> b_frag;
            fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;
            fill_fragment(c_frag, 0.f);

            const int n_start = warp_id * WMMA_N;   // 0, 16, 32, 48
            #pragma unroll
            for (int k_off = 0; k_off < HEAD_SIZE; k_off += WMMA_K) {
                // A tile: Q_sh[0:16, k_off:k_off+16] row-major, ld=HEAD_SIZE.
                load_matrix_sync(a_frag, &Q_sh[0][k_off], HEAD_SIZE);
                // B tile (col-major): K_sh is (BLOCK_N, HEAD_SIZE) row-major;
                // viewed as (HEAD_SIZE, BLOCK_N) col-major with ld=HEAD_SIZE
                // the 16-col tile starts at offset (n_start * HEAD_SIZE + k_off).
                load_matrix_sync(b_frag,
                                 &K_sh[n_start][k_off],
                                 HEAD_SIZE);
                mma_sync(c_frag, a_frag, b_frag, c_frag);
            }
            store_matrix_sync(&scores_sh[0][n_start], c_frag,
                              BLOCK_N, mem_row_major);
        }
        __syncthreads();

        // --- Mask invalid slots + scale by 1/d ---
        for (int cell = tid; cell < BLOCK_M_PAD * BLOCK_N; cell += THREADS) {
            const int m = cell / BLOCK_N;
            const int n = cell % BLOCK_N;
            scores_sh[m][n] = (n < n_valid)
                ? (scores_sh[m][n] * inv_d)
                : -1e30f;
        }
        __syncthreads();

        // --- Online softmax (one thread per M row) ---
        // 16 M rows; threads 0..15 each do one row.
        if (tid < BLOCK_M_PAD) {
            const int m = tid;
            float row_max = -1e30f;
            #pragma unroll
            for (int n = 0; n < BLOCK_N; ++n) {
                row_max = fmaxf(row_max, scores_sh[m][n]);
            }
            const float m_new = fmaxf(m_sh[m], row_max);
            alpha_sh[m] = __expf(m_sh[m] - m_new);
            m_sh[m] = m_new;

            float row_sum = 0.f;
            #pragma unroll
            for (int n = 0; n < BLOCK_N; ++n) {
                const float p = (n < n_valid) ? __expf(scores_sh[m][n] - m_new) : 0.f;
                P_sh[m][n] = __float2bfloat16(p);
                row_sum += p;
            }
            l_sh[m] = l_sh[m] * alpha_sh[m] + row_sum;
        }
        __syncthreads();

        // --- Rescale acc_sh *= alpha (parallel over (m, d)) ---
        for (int cell = tid; cell < BLOCK_M_PAD * HEAD_SIZE; cell += THREADS) {
            const int m = cell / HEAD_SIZE;
            const int d = cell % HEAD_SIZE;
            acc_sh[m][d] *= alpha_sh[m];
        }
        __syncthreads();

        // --- Dequant V ---
        dequant_tile(cache_v_idx, slot_base_sh, cache_v_norm, meta_sh,
                     codebook_sh, &V_sh[0][0], n_valid);
        __syncthreads();

        // --- WMMA P @ V -> acc_sh (accumulate into existing) ---
        // M = 16, N = HEAD_SIZE = 128, K = BLOCK_N = 64.
        // Each of 4 warps handles 2 consecutive N-tiles (16 cols each).
        {
            fragment<matrix_a, WMMA_M, WMMA_N, WMMA_K, bf16, row_major> a_frag;
            fragment<matrix_b, WMMA_M, WMMA_N, WMMA_K, bf16, row_major> b_frag;
            fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;

            #pragma unroll
            for (int sub = 0; sub < 2; ++sub) {
                const int n_start = (warp_id * 2 + sub) * WMMA_N;  // [0..112] step 16

                // Preload current acc tile into c_frag so mma_sync keeps adding.
                load_matrix_sync(c_frag,
                                 &acc_sh[0][n_start],
                                 HEAD_SIZE, mem_row_major);

                #pragma unroll
                for (int k_off = 0; k_off < BLOCK_N; k_off += WMMA_K) {
                    // A: P_sh[0:16, k_off:k_off+16] row-major, ld=BLOCK_N.
                    load_matrix_sync(a_frag, &P_sh[0][k_off], BLOCK_N);
                    // B: V_sh[k_off:k_off+16, n_start:n_start+16] row-major, ld=HEAD_SIZE.
                    load_matrix_sync(b_frag, &V_sh[k_off][n_start], HEAD_SIZE);
                    mma_sync(c_frag, a_frag, b_frag, c_frag);
                }
                store_matrix_sync(&acc_sh[0][n_start], c_frag,
                                  HEAD_SIZE, mem_row_major);
            }
        }
        __syncthreads();
    }

    // ========================= FINAL NORMALIZE + STORE =========================
    for (int cell = tid; cell < BLOCK_M_PAD * HEAD_SIZE; cell += THREADS) {
        const int m = cell / HEAD_SIZE;
        const int d = cell % HEAD_SIZE;
        if (m < gqa_group) {
            const int q_head  = q_head_start + m;
            const int out_off = (q_idx * num_heads_q + q_head) * HEAD_SIZE + d;
            const float denom = fmaxf(l_sh[m], 1e-12f);
            out[out_off] = __float2bfloat16(acc_sh[m][d] / denom);
        }
    }
}

// ---------------------------------------------------------------------------
// C++ launch entry
// ---------------------------------------------------------------------------
void attend_mse_launch(
    at::Tensor q_rot,
    at::Tensor cache_k_idx,
    at::Tensor cache_k_norm,
    at::Tensor cache_v_idx,
    at::Tensor cache_v_norm,
    at::Tensor block_table,
    at::Tensor seq_id_per_query,
    at::Tensor kv_end_per_query,
    at::Tensor codebook,
    at::Tensor out,
    int block_size,
    int gqa_group
) {
    TORCH_CHECK(q_rot.is_cuda() && q_rot.scalar_type() == at::kBFloat16,
                "q_rot must be bf16 CUDA");
    TORCH_CHECK(cache_k_idx.scalar_type() == at::kByte);
    TORCH_CHECK(cache_k_norm.scalar_type() == at::kFloat);
    TORCH_CHECK(codebook.scalar_type() == at::kBFloat16);

    const int T_q          = q_rot.size(0);
    const int num_heads_q  = q_rot.size(1);
    const int head_size    = q_rot.size(2);
    const int num_heads_kv = cache_k_idx.size(2);
    const int idx_dim      = cache_k_idx.size(3);
    const int bt_stride    = block_table.size(1);
    const int K_CB         = codebook.size(0);

    TORCH_CHECK(head_size == HEAD_SIZE,
                "head_size must be 128 for this WMMA kernel");
    TORCH_CHECK(K_CB <= K_CB_MAX, "K_CB too large");
    TORCH_CHECK(gqa_group <= BLOCK_M_PAD,
                "gqa_group must be <= 16 (WMMA tile)");

    const float inv_d = 1.0f / static_cast<float>(head_size);

    dim3 grid(T_q, num_heads_kv);
    dim3 block(THREADS);

    attend_mse_tc_kernel<<<grid, block>>>(
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
