# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tensor-core attend kernel for TurboQuant.

Replaces the per-slot fp32 scalar inner-product of ``attend.py`` with a
tiled matmul formulation so the attention compute runs on tensor cores:

    for each KV tile of BLOCK_N slots:
        K_tile = codebook[k_idx[tile]] * k_norm[tile]      # (BLOCK_N, d) bf16
        qk     = tl.dot(Q, K_tile.T)                       # (BLOCK_M, BLOCK_N)
        V_tile = codebook[v_idx[tile]] * v_norm[tile]      # (BLOCK_N, d) bf16
        acc    = acc * rescale + tl.dot(softmax(qk), V_tile)   # (BLOCK_M, d)

Dequantization happens inline per tile: load nibble-packed idx, unpack
to int32, gather the (small, cached) bf16 codebook, scale by the
per-slot ``norm``. K/V dequant each produce a (BLOCK_N, d) bf16 tile --
exactly the operand shape ``tl.dot`` wants.

Grid: ``(num_query_tokens, num_heads_kv)``. Each program handles one
GQA group (``gqa_group`` q_heads sharing one kv_head). BLOCK_M is
padded to 16 (Triton tensor-core tile minimum) even when gqa_group < 16
-- for Llama-3 8B gqa_group=4, so 3/4 of the matmul rows are masked
waste, but tensor cores are ~50x faster than fp32 scalar so net gain
is still ~10x per tile.

QJL path (prod): the ``Sq @ qjl_sign.T`` term is also turned into a
``tl.dot`` by unpacking 1-bit qjl signs into a (BLOCK_N, d) bf16
``{+1, -1}`` tile.

Post-rotation of V (``(out @ H.T) * signs / sqrt(d)``) still runs in
Python on the kernel output -- fusing it is a separate follow-up.
Store path (K/V quantization write) also stays in Python for now.

Scope: b=4 only (K_CB <= 16, 4-bit nibble-packed K/V idx, 1-bit-packed
QJL sign for prod).
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from vllm.triton_utils import tl, triton

if TYPE_CHECKING:
    from vllm.turboquant.codebook import QuantState


_NEG_LARGE = tl.constexpr(-1.0e30)


# Tensor-core tl.dot wants M, N, K multiples of 16. Our head_size = 128
# is already a power of 2. BLOCK_M is forced to at least 16.
_BLOCK_M_MIN = 16


# ---------------------------------------------------------------------------
# Autotuned tensor-core attend kernel
# ---------------------------------------------------------------------------
_AUTOTUNE_CONFIGS = [
    triton.Config({"BLOCK_N": 32}, num_warps=2, num_stages=2),
    triton.Config({"BLOCK_N": 32}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_N": 64}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_N": 64}, num_warps=4, num_stages=3),
    triton.Config({"BLOCK_N": 64}, num_warps=8, num_stages=2),
    triton.Config({"BLOCK_N": 128}, num_warps=8, num_stages=2),
    triton.Config({"BLOCK_N": 128}, num_warps=8, num_stages=3),
]


@triton.autotune(
    configs=_AUTOTUNE_CONFIGS,
    key=["head_size", "K_CB", "USE_QJL", "BLOCK_M", "GQA_GROUP"],
)
@triton.jit
def _tc_attend_kernel(
    q_rotated_ptr,                  # (T_q, H_q, d) bf16/fp16
    Sq_ptr,                         # (T_q, H_q, d) bf16/fp16  prod only
    cache_k_idx_ptr,                # (num_blocks, bs, H_kv, d/2) uint8
    cache_k_norm_ptr,               # (num_blocks, bs, H_kv) fp32
    cache_v_idx_ptr,                # (num_blocks, bs, H_kv, d/2) uint8
    cache_v_norm_ptr,               # (num_blocks, bs, H_kv) fp32
    cache_k_qjl_sign_ptr,           # (num_blocks, bs, H_kv, d/8) uint8 prod
    cache_k_rnorm_ptr,              # (num_blocks, bs, H_kv) fp32  prod
    block_table_ptr,                # (num_seqs, stride) int32
    seq_id_per_query_ptr,           # (T_q,) int32
    kv_end_per_query_ptr,           # (T_q,) int32
    codebook_ptr,                   # (K_CB,) bf16 (same dtype as Q)
    out_ptr,                        # (T_q, H_q, d) OUT_DTYPE
    inv_d,
    qjl_coef,
    block_table_stride,
    OUT_DTYPE: tl.constexpr,
    COMPUTE_DTYPE: tl.constexpr,    # tl.bfloat16 or tl.float16
    num_heads_q: tl.constexpr,
    num_heads_kv: tl.constexpr,
    head_size: tl.constexpr,
    idx_dim: tl.constexpr,
    qjl_dim: tl.constexpr,
    block_size: tl.constexpr,
    BLOCK_M: tl.constexpr,          # >= 16 (tensor-core min), >= GQA_GROUP
    BLOCK_N: tl.constexpr,          # autotuned KV-tile
    BLOCK_D: tl.constexpr,          # next_power_of_2(head_size)
    K_CB: tl.constexpr,
    USE_QJL: tl.constexpr,
    GQA_GROUP: tl.constexpr,
):
    q_idx = tl.program_id(0)
    kvh_idx = tl.program_id(1)
    q_head_start = kvh_idx * GQA_GROUP

    seq_idx = tl.load(seq_id_per_query_ptr + q_idx)
    kv_end = tl.load(kv_end_per_query_ptr + q_idx)

    m_off = tl.arange(0, BLOCK_M)
    n_off = tl.arange(0, BLOCK_N)
    d_off = tl.arange(0, BLOCK_D)
    mask_m = m_off < GQA_GROUP
    mask_d = d_off < head_size

    # Load Q tile (BLOCK_M, BLOCK_D) in compute dtype (bf16/fp16).
    q_base = q_idx * num_heads_q * head_size
    q_offs_2d = (
        q_base
        + (q_head_start + m_off[:, None]) * head_size
        + d_off[None, :]
    )
    q_rot = tl.load(
        q_rotated_ptr + q_offs_2d,
        mask=mask_m[:, None] & mask_d[None, :],
        other=0.0,
    )

    if USE_QJL:
        sq = tl.load(
            Sq_ptr + q_offs_2d,
            mask=mask_m[:, None] & mask_d[None, :],
            other=0.0,
        )

    m_i = tl.full((BLOCK_M,), _NEG_LARGE, dtype=tl.float32)
    l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)

    # 4-bit nibble-pack layout: two indices per byte.
    d_pack = d_off // 2
    is_high = (d_off % 2) == 1

    num_tiles = (kv_end + BLOCK_N - 1) // BLOCK_N

    for tile_i in range(0, num_tiles):
        kv_start = tile_i * BLOCK_N
        n_pos = kv_start + n_off             # (BLOCK_N,)
        mask_n = n_pos < kv_end

        # Paged addressing for this KV tile.
        block_idx = n_pos // block_size
        tok_in_block = n_pos % block_size

        phys_blocks = tl.load(
            block_table_ptr + seq_idx * block_table_stride + block_idx,
            mask=mask_n, other=0,
        )                                    # (BLOCK_N,) int32

        base_idx = (
            phys_blocks * (block_size * num_heads_kv * idx_dim)
            + tok_in_block * (num_heads_kv * idx_dim)
            + kvh_idx * idx_dim
        )                                    # (BLOCK_N,) int32
        meta_addrs = (
            phys_blocks * (block_size * num_heads_kv)
            + tok_in_block * num_heads_kv
            + kvh_idx
        )                                    # (BLOCK_N,)

        # ---- K dequant tile (BLOCK_N, BLOCK_D) bf16 ----
        addrs_k = base_idx[:, None] + d_pack[None, :]
        packed_k = tl.load(
            cache_k_idx_ptr + addrs_k,
            mask=mask_n[:, None] & mask_d[None, :],
            other=0,
        ).to(tl.uint8)
        k_low = packed_k & 0xF
        k_high = (packed_k >> 4) & 0xF
        k_idx_full = tl.where(is_high[None, :], k_high, k_low).to(tl.int32)
        # Gather from codebook (tiny, cached in L1/const).
        k_tile = tl.load(codebook_ptr + k_idx_full)   # (BLOCK_N, BLOCK_D) bf16

        # Scale by k_norm (broadcast along last dim). Going through fp32
        # keeps the scale-then-tensor-core sequence numerically clean.
        k_norm = tl.load(cache_k_norm_ptr + meta_addrs, mask=mask_n, other=0.0)
        k_tile_scaled = (
            k_tile.to(tl.float32) * k_norm[:, None]
        ).to(COMPUTE_DTYPE)

        # ---- QK via tensor core ----
        # (BLOCK_M, BLOCK_D) x (BLOCK_D, BLOCK_N) -> (BLOCK_M, BLOCK_N)
        qk = tl.dot(q_rot, tl.trans(k_tile_scaled))   # fp32 accumulator
        # qk == (sum_d q_rot[m,d] * codebook[k_idx[n,d]] * k_norm[n])
        #     == main_dot_{m,n} * k_norm[n]

        if USE_QJL:
            # ---- QJL sign tile (BLOCK_N, BLOCK_D) bf16 with +/-1 ----
            bit_pos = d_off % 8
            byte_pos = d_off // 8
            base_qjl = (
                phys_blocks * (block_size * num_heads_kv * qjl_dim)
                + tok_in_block * (num_heads_kv * qjl_dim)
                + kvh_idx * qjl_dim
            )
            addrs_qjl = base_qjl[:, None] + byte_pos[None, :]
            qjl_byte = tl.load(
                cache_k_qjl_sign_ptr + addrs_qjl,
                mask=mask_n[:, None] & mask_d[None, :],
                other=0,
            ).to(tl.int32)
            bit = (qjl_byte >> bit_pos[None, :]) & 1
            qjl_sign_tile = (
                1.0 - 2.0 * bit.to(tl.float32)
            ).to(COMPUTE_DTYPE)

            r_norm = tl.load(
                cache_k_rnorm_ptr + meta_addrs, mask=mask_n, other=0.0
            )
            # Sq @ qjl_sign.T via tensor core
            qjl_dot = tl.dot(sq, tl.trans(qjl_sign_tile))  # (BLOCK_M, BLOCK_N) fp32

            # logit = (main_dot + qjl_coef * r_norm * qjl_dot) * k_norm * inv_d
            #       = (qk + qjl_coef * r_norm * qjl_dot * k_norm) * inv_d
            # (qk already includes k_norm via the scaled K tile.)
            logit = (
                qk + qjl_coef * r_norm[None, :] * qjl_dot * k_norm[None, :]
            ) * inv_d
        else:
            logit = qk * inv_d

        # Mask out-of-range slots before softmax.
        logit = tl.where(mask_n[None, :], logit, _NEG_LARGE)

        # ---- Online softmax update ----
        m_new = tl.maximum(m_i, tl.max(logit, axis=1))
        alpha = tl.exp(m_i - m_new)
        probs = tl.exp(logit - m_new[:, None])        # (BLOCK_M, BLOCK_N) fp32
        l_i = l_i * alpha + tl.sum(probs, axis=1)

        # ---- V dequant tile (BLOCK_N, BLOCK_D) bf16 ----
        addrs_v = base_idx[:, None] + d_pack[None, :]
        packed_v = tl.load(
            cache_v_idx_ptr + addrs_v,
            mask=mask_n[:, None] & mask_d[None, :],
            other=0,
        ).to(tl.uint8)
        v_low = packed_v & 0xF
        v_high = (packed_v >> 4) & 0xF
        v_idx_full = tl.where(is_high[None, :], v_high, v_low).to(tl.int32)
        v_tile = tl.load(codebook_ptr + v_idx_full)    # (BLOCK_N, BLOCK_D) bf16

        v_norm = tl.load(cache_v_norm_ptr + meta_addrs, mask=mask_n, other=0.0)
        v_tile_scaled = (
            v_tile.to(tl.float32) * v_norm[:, None]
        ).to(COMPUTE_DTYPE)

        # ---- acc update: acc * alpha + probs @ V via tensor core ----
        probs_cast = probs.to(COMPUTE_DTYPE)
        acc = acc * alpha[:, None] + tl.dot(probs_cast, v_tile_scaled)

        m_i = m_new

    # Normalize V accumulator and write out.
    out = acc / tl.maximum(l_i[:, None], 1e-12)
    out_offs = (
        q_base
        + (q_head_start + m_off[:, None]) * head_size
        + d_off[None, :]
    )
    tl.store(
        out_ptr + out_offs,
        out.to(OUT_DTYPE),
        mask=mask_m[:, None] & mask_d[None, :],
    )


# ---------------------------------------------------------------------------
# Python wrapper
# ---------------------------------------------------------------------------
def turboquant_paged_attention_lut(
    q: torch.Tensor,
    cache_k_idx: torch.Tensor,
    cache_k_norm: torch.Tensor,
    cache_v_idx: torch.Tensor,
    cache_v_norm: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    query_start_loc: torch.Tensor,
    state: "QuantState",
    cache_k_qjl_sign: torch.Tensor | None = None,
    cache_k_rnorm: torch.Tensor | None = None,
) -> torch.Tensor:
    num_query_tokens, num_heads_q, head_size = q.shape
    _, block_size, num_heads_kv, idx_dim = cache_k_idx.shape
    num_seqs = int(seq_lens.shape[0])
    assert query_start_loc.shape[0] == num_seqs + 1
    assert block_table.shape[0] >= num_seqs

    use_qjl = state.algo == "prod"
    if use_qjl:
        assert cache_k_qjl_sign is not None and cache_k_rnorm is not None

    K_CB = int(state.codebook.shape[0])
    assert K_CB <= 16, (
        f"TC kernel is b=4 only (K_CB <= 16); got K_CB={K_CB}"
    )
    assert idx_dim == head_size // 2, (
        f"cache_k_idx last dim {idx_dim} != head_size/2 -- b=4 only"
    )
    assert q.dtype in (torch.bfloat16, torch.float16), (
        f"TC kernel requires bf16 or fp16 queries for tensor-core dot; "
        f"got {q.dtype}"
    )

    dev = q.device

    qsl = query_start_loc.to(device=dev, dtype=torch.int64)
    query_lens = qsl[1:] - qsl[:-1]
    seq_ids = torch.arange(num_seqs, dtype=torch.int64, device=dev)
    seq_id_per_query_i64 = torch.repeat_interleave(seq_ids, query_lens)
    q_pos_per_query = (
        torch.arange(num_query_tokens, dtype=torch.int64, device=dev)
        - qsl[:-1][seq_id_per_query_i64]
    )
    prefix_len = seq_lens.to(device=dev, dtype=torch.int64) - query_lens
    kv_end_per_query_i64 = (
        prefix_len[seq_id_per_query_i64] + q_pos_per_query + 1
    )
    seq_id_per_query = seq_id_per_query_i64.to(torch.int32).contiguous()
    kv_end_per_query = kv_end_per_query_i64.to(torch.int32).contiguous()

    # Pre-rotate Q in Python, in the query's native dtype so the matmul
    # hits tensor cores (cuBLAS sgemm_f32 was dominating the prior
    # profile). H is symmetric so H.T == H.
    q_signed = q * state.signs                                       # bf16 elementwise
    q_rotated_k = (
        q_signed.reshape(-1, head_size) @ state.H
    ).view(num_query_tokens, num_heads_q, head_size).contiguous()    # bf16

    if use_qjl:
        Sq_k = (
            q_rotated_k.reshape(-1, head_size) @ state.S.T
        ).view(num_query_tokens, num_heads_q, head_size).contiguous()
    else:
        Sq_k = q_rotated_k

    qjl_dim = head_size // 8 if use_qjl else 1
    if use_qjl:
        qjl_sign_buf = cache_k_qjl_sign
        assert qjl_sign_buf.dtype == torch.uint8
        assert qjl_sign_buf.shape[-1] == qjl_dim
    else:
        qjl_sign_buf = cache_k_idx
    rnorm_buf = cache_k_rnorm if use_qjl else cache_k_norm

    # Codebook in the query's compute dtype (bf16 or fp16) so the
    # gather inside the kernel directly produces tensor-core operands.
    codebook_ct = state.codebook.to(q.dtype)

    gqa_group = num_heads_q // num_heads_kv
    BLOCK_M = max(gqa_group, _BLOCK_M_MIN)
    # Round BLOCK_M up to next power of two to keep tl.dot happy.
    BLOCK_M = 1 << (BLOCK_M - 1).bit_length()
    BLOCK_D = triton.next_power_of_2(head_size)

    out = torch.empty_like(q)

    OUT_DTYPE = tl.bfloat16 if q.dtype == torch.bfloat16 else tl.float16
    COMPUTE_DTYPE = OUT_DTYPE

    inv_d = 1.0 / float(head_size)
    qjl_coef = math.sqrt(math.pi / 2.0) / float(head_size)
    block_table_stride = int(block_table.shape[1])

    grid = (num_query_tokens, num_heads_kv)
    _tc_attend_kernel[grid](
        q_rotated_k,
        Sq_k,
        cache_k_idx,
        cache_k_norm,
        cache_v_idx,
        cache_v_norm,
        qjl_sign_buf,
        rnorm_buf,
        block_table,
        seq_id_per_query,
        kv_end_per_query,
        codebook_ct,
        out,
        inv_d,
        qjl_coef,
        block_table_stride,
        OUT_DTYPE=OUT_DTYPE,
        COMPUTE_DTYPE=COMPUTE_DTYPE,
        num_heads_q=num_heads_q,
        num_heads_kv=num_heads_kv,
        head_size=head_size,
        idx_dim=idx_dim,
        qjl_dim=qjl_dim,
        block_size=block_size,
        BLOCK_M=BLOCK_M,
        BLOCK_D=BLOCK_D,
        K_CB=K_CB,
        USE_QJL=use_qjl,
        GQA_GROUP=gqa_group,
    )

    # Post-rotate V back to original space in bf16 (tensor core matmul).
    # H symmetric so H.T == H. Do it in place in the query's dtype.
    inv_sqrt_d = 1.0 / math.sqrt(float(head_size))
    output = (
        out.reshape(-1, head_size) @ state.H
    ).view(num_query_tokens, num_heads_q, head_size)
    output = output * state.signs * inv_sqrt_d
    return output.contiguous()
