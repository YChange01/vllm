// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// TurboQuant flash-style attend kernel in raw CUDA with WMMA tensor cores.
//
// Pipeline (per thread block, one per (query_token, kv_head) pair):
//   1. Load Q into shared memory; dequant 16-entry codebook
//   2. Prefetch raw K/V idx+norm for tile 0 via cp.async (stage 0)
//   3. For each KV tile:
//        a. Prefetch raw for tile t+1 into the other stage via cp.async
//        b. Wait on tile t's async (keep t+1 in-flight)
//        c. Dequant K from smem staging -> K_sh bf16
//        d. WMMA: scores = Q_sh @ K_sh.T
//        e. Mask + scale + online softmax -> P_sh, alpha, m, l
//        f. Rescale acc_sh *= alpha
//        g. Dequant V from smem staging -> V_sh bf16
//        h. WMMA: acc_sh += P_sh @ V_sh
//   4. Normalize acc_sh / l_sh and write out (rotated V space).
//
// Uses nvcuda::wmma (mma.m16n8k16) for matmuls, cp.async for overlapping
// tile t+1's gmem load with tile t's compute. Requires sm_80+.
//
// Scope: mse path only. prod/QJL is a follow-up.

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_pipeline.h>
#include <mma.h>
#include <torch/extension.h>

namespace turboquant {
namespace cuda {

using namespace nvcuda::wmma;
using bf16 = __nv_bfloat16;

// --- Compile-time tile sizes ---
// BLOCK_N = 32 keeps static shared memory under the 48 KB default limit.
// With cp.async double-buffer staging we sit at ~41 KB total smem.
constexpr int HEAD_SIZE = 128;
constexpr int BLOCK_N   = 32;
constexpr int BLOCK_M_PAD = 16;             // WMMA min
constexpr int WMMA_M    = 16;
constexpr int WMMA_N    = 16;
constexpr int WMMA_K    = 16;
constexpr int K_CB_MAX  = 16;
constexpr int THREADS   = 128;              // 4 warps
constexpr int IDX_BYTES = HEAD_SIZE / 2;    // 4-bit packed -> 64 bytes/slot
constexpr int STAGES    = 2;                // cp.async double buffer

// ---------------------------------------------------------------------------
// Dequant one KV tile (raw idx+norm already staged in smem) into bf16 smem.
// All sources are now smem; the gmem load happened asynchronously earlier.
// ---------------------------------------------------------------------------
__device__ __forceinline__
void dequant_tile_from_stage(
    const uint8_t* __restrict__ idx_stg,    // smem: BLOCK_N * IDX_BYTES
    const float*   __restrict__ norm_stg,   // smem: BLOCK_N floats
    const float*   __restrict__ codebook_sh,
    bf16* __restrict__ tile_sh,
    const int n_valid
) {
    const int tid = threadIdx.x;
    #pragma unroll 8
    for (int cell = tid; cell < BLOCK_N * HEAD_SIZE; cell += THREADS) {
        const int n = cell / HEAD_SIZE;
        const int d = cell % HEAD_SIZE;
        if (n >= n_valid) {
            tile_sh[n * HEAD_SIZE + d] = __float2bfloat16(0.f);
            continue;
        }
        const uint8_t packed = idx_stg[n * IDX_BYTES + (d >> 1)];
        const int nib = (d & 1) ? ((packed >> 4) & 0xF) : (packed & 0xF);
        const float c = codebook_sh[nib];
        const float nv = norm_stg[n];
        tile_sh[n * HEAD_SIZE + d] = __float2bfloat16(c * nv);
    }
}

// ---------------------------------------------------------------------------
// Compute paged slot_base + meta arrays for one tile (gmem block_table read).
// Fills slot_base_stg[stage] and meta_stg[stage] for slots [0, BLOCK_N).
// Writes 0 for slots past n_valid (they won't be dequanted).
// ---------------------------------------------------------------------------
__device__ __forceinline__
void compute_slot_meta(
    const int32_t* __restrict__ block_table,
    const int seq_idx,
    const int kvh_idx,
    const int kv_base,
    const int kv_end,
    const int block_size,
    const int num_heads_kv,
    const int idx_dim,
    const int bt_stride,
    int32_t* __restrict__ slot_base_stg_row,
    int32_t* __restrict__ meta_stg_row
) {
    const int tid = threadIdx.x;
    if (tid < BLOCK_N) {
        const int pos = kv_base + tid;
        if (pos < kv_end) {
            const int blk = pos / block_size;
            const int tok = pos % block_size;
            const int phys = block_table[seq_idx * bt_stride + blk];
            slot_base_stg_row[tid] =
                ((phys * block_size + tok) * num_heads_kv + kvh_idx) * idx_dim;
            meta_stg_row[tid] =
                (phys * block_size + tok) * num_heads_kv + kvh_idx;
        } else {
            slot_base_stg_row[tid] = 0;
            meta_stg_row[tid] = 0;
        }
    }
}

// ---------------------------------------------------------------------------
// Issue cp.async loads of raw K/V idx + norm for one tile into smem stage.
// Caller must __pipeline_commit() after. 128 threads each issue one 16-byte
// cp.async for K idx, one for V idx; 32 threads also issue one 4-byte cp.async
// for K norm and one for V norm.
// ---------------------------------------------------------------------------
__device__ __forceinline__
void issue_prefetch(
    const uint8_t* __restrict__ cache_k_idx_g,
    const uint8_t* __restrict__ cache_v_idx_g,
    const float*   __restrict__ cache_k_norm_g,
    const float*   __restrict__ cache_v_norm_g,
    const int32_t* __restrict__ slot_base_stg_row,  // smem, BLOCK_N
    const int32_t* __restrict__ meta_stg_row,       // smem, BLOCK_N
    uint8_t* __restrict__ k_idx_stg_row,            // smem, BLOCK_N*IDX_BYTES
    uint8_t* __restrict__ v_idx_stg_row,
    float*   __restrict__ k_norm_stg_row,           // smem, BLOCK_N
    float*   __restrict__ v_norm_stg_row
) {
    const int tid = threadIdx.x;
    // 32 slots × 64 bytes / 16-byte cp.async = 128 transfers, one per thread.
    constexpr int COPIES_PER_IDX_TILE = BLOCK_N * IDX_BYTES / 16;
    static_assert(COPIES_PER_IDX_TILE == THREADS,
                  "expect 1 idx 16B cp.async per thread");
    const int slot = tid / (IDX_BYTES / 16);
    const int col16 = (tid % (IDX_BYTES / 16)) * 16;
    const int32_t soff = slot_base_stg_row[slot] + col16;
    __pipeline_memcpy_async(
        &k_idx_stg_row[slot * IDX_BYTES + col16],
        &cache_k_idx_g[soff], 16);
    __pipeline_memcpy_async(
        &v_idx_stg_row[slot * IDX_BYTES + col16],
        &cache_v_idx_g[soff], 16);
    if (tid < BLOCK_N) {
        __pipeline_memcpy_async(
            &k_norm_stg_row[tid],
            &cache_k_norm_g[meta_stg_row[tid]], 4);
        __pipeline_memcpy_async(
            &v_norm_stg_row[tid],
            &cache_v_norm_g[meta_stg_row[tid]], 4);
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

    // --- Shared memory (16-byte aligned for WMMA / cp.async) ---
    __shared__ __align__(16) bf16  Q_sh    [BLOCK_M_PAD][HEAD_SIZE];       // (16, 128)
    __shared__ __align__(16) bf16  K_sh    [BLOCK_N][HEAD_SIZE];           // (32, 128)
    __shared__ __align__(16) bf16  V_sh    [BLOCK_N][HEAD_SIZE];           // (32, 128)
    __shared__ __align__(16) bf16  P_sh    [BLOCK_M_PAD][BLOCK_N];         // (16, 32)
    __shared__ float scores_sh[BLOCK_M_PAD][BLOCK_N];                      // (16, 32)
    __shared__ float codebook_sh[K_CB_MAX];
    __shared__ float m_sh   [BLOCK_M_PAD];
    __shared__ float l_sh   [BLOCK_M_PAD];
    __shared__ float alpha_sh[BLOCK_M_PAD];
    __shared__ float acc_sh [BLOCK_M_PAD][HEAD_SIZE];                      // (16, 128)

    // cp.async double-buffered raw staging. Tile t uses stage t & 1; tile t+1
    // is loading into stage (t+1) & 1 concurrently with tile t's compute.
    __shared__ __align__(16) uint8_t K_idx_stg [STAGES][BLOCK_N * IDX_BYTES];  // 2×2KB
    __shared__ __align__(16) uint8_t V_idx_stg [STAGES][BLOCK_N * IDX_BYTES];  // 2×2KB
    __shared__ __align__(16) float   K_norm_stg[STAGES][BLOCK_N];              // 2×128B
    __shared__ __align__(16) float   V_norm_stg[STAGES][BLOCK_N];
    __shared__ __align__(16) int32_t slot_base_stg[STAGES][BLOCK_N];           // 2×128B
    __shared__ __align__(16) int32_t meta_stg     [STAGES][BLOCK_N];

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

    // --- Prefetch tile 0 into stage 0 ---
    if (num_tiles > 0) {
        compute_slot_meta(block_table, seq_idx, kvh_idx,
                          /*kv_base=*/0, kv_end, block_size, num_heads_kv,
                          idx_dim, bt_stride,
                          slot_base_stg[0], meta_stg[0]);
        __syncthreads();
        issue_prefetch(cache_k_idx, cache_v_idx, cache_k_norm, cache_v_norm,
                       slot_base_stg[0], meta_stg[0],
                       K_idx_stg[0], V_idx_stg[0],
                       K_norm_stg[0], V_norm_stg[0]);
        __pipeline_commit();
    }

    for (int tile = 0; tile < num_tiles; ++tile) {
        const int kv_base = tile * BLOCK_N;
        const int n_valid = min(BLOCK_N, kv_end - kv_base);
        const int s       = tile & 1;

        // --- Kick off tile t+1 prefetch into the other stage ---
        const bool has_next = (tile + 1 < num_tiles);
        if (has_next) {
            const int next_s  = (tile + 1) & 1;
            compute_slot_meta(block_table, seq_idx, kvh_idx,
                              /*kv_base=*/(tile + 1) * BLOCK_N, kv_end,
                              block_size, num_heads_kv, idx_dim, bt_stride,
                              slot_base_stg[next_s], meta_stg[next_s]);
            __syncthreads();
            issue_prefetch(cache_k_idx, cache_v_idx,
                           cache_k_norm, cache_v_norm,
                           slot_base_stg[next_s], meta_stg[next_s],
                           K_idx_stg[next_s], V_idx_stg[next_s],
                           K_norm_stg[next_s], V_norm_stg[next_s]);
            __pipeline_commit();
        }

        // Wait for THIS tile's cp.async (keep next tile's in-flight if any).
        __pipeline_wait_prior(has_next ? 1 : 0);
        __syncthreads();

        // --- Dequant K from smem stage ---
        dequant_tile_from_stage(K_idx_stg[s], K_norm_stg[s],
                                codebook_sh, &K_sh[0][0], n_valid);
        __syncthreads();

        // --- WMMA Q @ K.T -> scores_sh ---
        // M = BLOCK_M_PAD = 16, N = BLOCK_N = 32, K = HEAD_SIZE = 128.
        // Two 16x16 N-tiles total; warps 0 and 1 handle them. Warps 2/3
        // are idle during QK (they come back to work for PV below).
        if (warp_id < 2) {
            fragment<matrix_a, WMMA_M, WMMA_N, WMMA_K, bf16, row_major> a_frag;
            fragment<matrix_b, WMMA_M, WMMA_N, WMMA_K, bf16, col_major> b_frag;
            fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;
            fill_fragment(c_frag, 0.f);

            const int n_start = warp_id * WMMA_N;   // 0, 16
            #pragma unroll
            for (int k_off = 0; k_off < HEAD_SIZE; k_off += WMMA_K) {
                // A tile: Q_sh[0:16, k_off:k_off+16] row-major, ld=HEAD_SIZE.
                load_matrix_sync(a_frag, &Q_sh[0][k_off], HEAD_SIZE);
                // B tile (col-major): K_sh row-major (BLOCK_N, HEAD_SIZE)
                // is bit-identical to K^T col-major (HEAD_SIZE, BLOCK_N)
                // with leading dim = HEAD_SIZE.
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

        // --- Dequant V from same smem stage ---
        dequant_tile_from_stage(V_idx_stg[s], V_norm_stg[s],
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
    TORCH_CHECK(idx_dim == IDX_BYTES,
                "idx_dim must equal HEAD_SIZE/2 (4-bit nibble packing); "
                "cp.async prefetch is hard-wired to 16-byte transfers");

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
