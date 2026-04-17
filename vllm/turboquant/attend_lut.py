# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Triton attend kernel with in-kernel LUT for K inner product.

Variant of ``attend.py`` that precomputes the per-(query, head) table

    LUT[j, c] = q_rot[j] * codebook[c]     j in [d],  c in [K_CB]

once in registers, then gathers ``sum_j LUT[j, k_idx[j]]`` per KV slot
via mask-sum. Goal: replace the per-slot codebook dequant

    rk = codebook[k_idx]               # d scalar gathers
    main_dot = sum(q_rot * rk)         # d fp32 muls + reduce

with a register-resident table lookup that removes the d multiplies.

Trade-off: mask-sum gather does K_CB x more register arithmetic per
slot (one compare + one select per (j, c) pair), so the net win is
hardware-dependent. Kept as an A/B alternative to ``attend.py``; switch
via ``TURBOQUANT_USE_LUT=1``.

V path is unchanged -- V is reconstructed as a d-dim vector per slot
(not an inner product with Q), so a LUT gives no benefit there. QJL
path (prod) is also unchanged.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from vllm.triton_utils import tl, triton

if TYPE_CHECKING:
    from vllm.turboquant.codebook import QuantState


_NEG_LARGE = tl.constexpr(-1.0e30)


@triton.jit
def _attend_kernel_lut(
    q_rotated_ptr,                  # (T_q, H_q, d)      fp16/bf16
    Sq_ptr,                         # (T_q, H_q, d)      fp16/bf16  prod only
    cache_k_idx_ptr,                # (num_blocks, bs, H_kv, idx_dim) uint8
    cache_k_norm_ptr,               # (num_blocks, bs, H_kv)          fp32
    cache_v_idx_ptr,                # (num_blocks, bs, H_kv, idx_dim) uint8
    cache_v_norm_ptr,               # (num_blocks, bs, H_kv)          fp32
    cache_k_qjl_sign_ptr,           # (num_blocks, bs, H_kv, d/8) uint8 prod
    cache_k_rnorm_ptr,              # (num_blocks, bs, H_kv)      fp32  prod
    block_table_ptr,                # (num_seqs, block_table_stride) int32
    seq_id_per_query_ptr,           # (T_q,) int32
    kv_end_per_query_ptr,           # (T_q,) int32
    codebook_ptr,                   # (K_CB,) fp32
    out_ptr,                        # (T_q, H_q, d) fp16/bf16 rotated space
    inv_d,
    qjl_coef,
    max_blocks_per_seq,
    block_table_stride,
    OUT_DTYPE: tl.constexpr,
    num_heads_q: tl.constexpr,
    num_heads_kv: tl.constexpr,
    head_size: tl.constexpr,
    idx_dim: tl.constexpr,
    qjl_dim: tl.constexpr,
    block_size: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_BS: tl.constexpr,
    K_CB: tl.constexpr,
    USE_QJL: tl.constexpr,
    USE_4BIT_PACK: tl.constexpr,
):
    q_idx = tl.program_id(0)
    qh_idx = tl.program_id(1)

    gqa_group = num_heads_q // num_heads_kv
    kvh_idx = qh_idx // gqa_group

    seq_idx = tl.load(seq_id_per_query_ptr + q_idx)
    kv_end = tl.load(kv_end_per_query_ptr + q_idx)
    num_blocks_q = (kv_end + block_size - 1) // block_size

    d_idx = tl.arange(0, BLOCK_D)
    c_idx = tl.arange(0, K_CB)
    mask_d = d_idx < head_size

    q_off = (q_idx * num_heads_q + qh_idx) * head_size + d_idx
    q_rot = tl.load(q_rotated_ptr + q_off, mask=mask_d, other=0.0).to(tl.float32)
    if USE_QJL:
        sq = tl.load(Sq_ptr + q_off, mask=mask_d, other=0.0).to(tl.float32)

    # Codebook is small (<= 256 scalars). Load once, keep in registers.
    codebook_vec = tl.load(codebook_ptr + c_idx).to(tl.float32)   # (K_CB,)

    # Build per-(query, head) LUT in registers:
    #     LUT[j, c] = q_rot[j] * codebook[c]
    # For head_size=128, K_CB=16 -> 2048 fp32 = 8 KB per program.
    LUT = q_rot[:, None] * codebook_vec[None, :]                  # (BLOCK_D, K_CB)

    m_i = tl.full((), _NEG_LARGE, dtype=tl.float32)
    l_i = tl.zeros((), dtype=tl.float32)
    acc = tl.zeros((BLOCK_D,), dtype=tl.float32)

    if USE_4BIT_PACK:
        d_pack = d_idx // 2
        is_high = (d_idx % 2) == 1

    for block_i in range(0, max_blocks_per_seq):
        if block_i < num_blocks_q:
            phys_block = tl.load(
                block_table_ptr + seq_idx * block_table_stride + block_i
            )
            for tok_in_block in tl.static_range(0, BLOCK_BS):
                abs_pos = block_i * block_size + tok_in_block
                if abs_pos < kv_end:
                    base_idx = (
                        phys_block * block_size * num_heads_kv * idx_dim
                        + tok_in_block * num_heads_kv * idx_dim
                        + kvh_idx * idx_dim
                    )
                    meta = (
                        phys_block * block_size * num_heads_kv
                        + tok_in_block * num_heads_kv
                        + kvh_idx
                    )

                    # K idx load (packed or unpacked).
                    if USE_4BIT_PACK:
                        packed_k = tl.load(
                            cache_k_idx_ptr + base_idx + d_pack,
                            mask=mask_d, other=0,
                        ).to(tl.uint8)
                        k_low = packed_k & 0xF
                        k_high = (packed_k >> 4) & 0xF
                        k_idx_u8 = tl.where(is_high, k_high, k_low).to(tl.int32)
                    else:
                        k_idx_u8 = tl.load(
                            cache_k_idx_ptr + base_idx + d_idx,
                            mask=mask_d, other=0,
                        ).to(tl.int32)

                    # ===== LUT gather via mask-sum =====
                    # one_hot[j, c] = (k_idx_u8[j] == c), one 1 per row.
                    one_hot = (k_idx_u8[:, None] == c_idx[None, :])   # (BLOCK_D, K_CB)
                    main_dot = tl.sum(tl.where(one_hot, LUT, 0.0))    # scalar

                    k_norm = tl.load(cache_k_norm_ptr + meta)

                    if USE_QJL:
                        base_qjl = (
                            phys_block * block_size * num_heads_kv * qjl_dim
                            + tok_in_block * num_heads_kv * qjl_dim
                            + kvh_idx * qjl_dim
                        )
                        bit_pos = d_idx % 8
                        byte_pos = d_idx // 8
                        qjl_byte = tl.load(
                            cache_k_qjl_sign_ptr + base_qjl + byte_pos,
                            mask=mask_d, other=0,
                        ).to(tl.int32)
                        bit = ((qjl_byte >> bit_pos) & 1).to(tl.float32)
                        qjl_sign = 1.0 - 2.0 * bit
                        qjl_dot = tl.sum(sq * qjl_sign)
                        r_norm = tl.load(cache_k_rnorm_ptr + meta)
                        logit = (main_dot + qjl_coef * r_norm * qjl_dot) \
                                * k_norm * inv_d
                    else:
                        logit = main_dot * k_norm * inv_d

                    # V dequant: per-slot codebook lookup (unchanged).
                    if USE_4BIT_PACK:
                        packed_v = tl.load(
                            cache_v_idx_ptr + base_idx + d_pack,
                            mask=mask_d, other=0,
                        ).to(tl.uint8)
                        v_low = packed_v & 0xF
                        v_high = (packed_v >> 4) & 0xF
                        v_idx_u8 = tl.where(is_high, v_high, v_low).to(tl.int32)
                    else:
                        v_idx_u8 = tl.load(
                            cache_v_idx_ptr + base_idx + d_idx,
                            mask=mask_d, other=0,
                        ).to(tl.int32)

                    rv = tl.load(codebook_ptr + v_idx_u8).to(tl.float32)
                    v_norm = tl.load(cache_v_norm_ptr + meta)
                    v_vec = rv * v_norm

                    # Flash softmax accumulation.
                    m_new = tl.maximum(m_i, logit)
                    alpha = tl.exp(m_i - m_new)
                    beta = tl.exp(logit - m_new)
                    l_i = l_i * alpha + beta
                    acc = acc * alpha + beta * v_vec
                    m_i = m_new

    out_vec = acc / tl.maximum(l_i, 1e-12)
    out_off = (q_idx * num_heads_q + qh_idx) * head_size + d_idx
    tl.store(out_ptr + out_off, out_vec.to(OUT_DTYPE), mask=mask_d)


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
    """LUT-based paged attention with on-the-fly K dequant (via register
    LUT) and per-slot V dequant.

    Same interface and output as ``turboquant_paged_attention`` in
    ``attend.py``. Use via ``TURBOQUANT_USE_LUT=1``.
    """
    num_query_tokens, num_heads_q, head_size = q.shape
    _, block_size, num_heads_kv, idx_dim = cache_k_idx.shape
    num_seqs = int(seq_lens.shape[0])
    assert query_start_loc.shape[0] == num_seqs + 1
    assert block_table.shape[0] >= num_seqs

    use_qjl = state.algo == "prod"
    if use_qjl:
        assert cache_k_qjl_sign is not None and cache_k_rnorm is not None

    codebook_f = state.codebook.to(torch.float32)
    K_CB = int(codebook_f.shape[0])
    use_4bit_pack = K_CB <= 16
    expected_idx_dim = head_size // 2 if use_4bit_pack else head_size
    assert idx_dim == expected_idx_dim, (
        f"cache_k_idx last dim {idx_dim} != expected {expected_idx_dim} "
        f"(K_CB={K_CB}, head_size={head_size})"
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

    # Pre-rotate Q (same as non-LUT path).
    H_f = state.H.to(torch.float32)
    signs_f = state.signs.to(torch.float32)
    q_f = q.float()
    q_rotated = (q_f * signs_f) @ H_f.T
    q_rotated_k = q_rotated.to(q.dtype).contiguous()

    if use_qjl:
        S_f = state.S.to(torch.float32)
        Sq = q_rotated @ S_f.T
        Sq_k = Sq.to(q.dtype).contiguous()
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

    out = torch.empty_like(q)
    grid = (num_query_tokens, num_heads_q)
    BLOCK_D = triton.next_power_of_2(head_size)
    BLOCK_BS = block_size
    OUT_DTYPE = tl.bfloat16 if q.dtype == torch.bfloat16 else tl.float16

    inv_d = 1.0 / float(head_size)
    qjl_coef = math.sqrt(math.pi / 2.0) / float(head_size)
    actual_max_blocks = (int(seq_lens.max().item()) + block_size - 1) // block_size
    block_table_stride = int(block_table.shape[1])

    _attend_kernel_lut[grid](
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
        codebook_f,
        out,
        inv_d,
        qjl_coef,
        actual_max_blocks,
        block_table_stride,
        OUT_DTYPE=OUT_DTYPE,
        num_heads_q=num_heads_q,
        num_heads_kv=num_heads_kv,
        head_size=head_size,
        idx_dim=idx_dim,
        qjl_dim=qjl_dim,
        block_size=block_size,
        BLOCK_D=BLOCK_D,
        BLOCK_BS=BLOCK_BS,
        K_CB=K_CB,
        USE_QJL=use_qjl,
        USE_4BIT_PACK=use_4bit_pack,
    )

    out_f = out.float()
    output_f = (out_f @ H_f.T) * signs_f / math.sqrt(float(head_size))
    return output_f.to(out.dtype)
