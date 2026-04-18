# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Triton attend kernel for TurboQuant (paper arXiv:2504.19874).

Reads the paged K cache populated by ``vllm.turboquant.store`` (K idx +
||k||, optionally K QJL sign + ||r||) and the V cache (V idx + ||v||).
Runs causal varlen flash-style attention with on-the-fly K and V
dequantization. One Triton program per (query_token, q_head) -- this is
the hot path on every decode step.

Logit (attention score) per (q, k):

    Algorithm 1 (mse):
        k_rot ≈ codebook[idx]
        logit = <q_rot, k_rot> * ||k|| / d        # = <q, k> / sqrt(d)

    Algorithm 2 (prod): + 1-bit QJL on the residual
        k_rot ≈ codebook[idx] + ||r|| * sqrt(pi/2)/d * S^T @ qjl_sign
        logit = (<q_rot, codebook[idx]> +
                 sqrt(pi/2)/d * ||r|| * <S @ q_rot, qjl_sign>) * ||k|| / d

V dequant (always Q_mse):

    v_rot ≈ codebook[v_idx] * ||v||
    v     ≈ signs * (H @ v_rot) / sqrt(d)

The kernel accumulates ``acc = sum_i w_i * v_rot_i`` in rotated /
||·||=sqrt(d)-scaled space; the final inverse-Hadamard + signs +
1/sqrt(d) is applied once per (query, head) in the Python wrapper
(small matmul on top of the kernel output).
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
def _attend_kernel(
    q_rotated_ptr,                  # (T_q, H_q, d)  fp16/bf16  H @ (signs * q)
    Sq_ptr,                         # (T_q, H_q, d)  fp16/bf16  S @ q_rotated  (prod only)
    cache_k_idx_ptr,                # (num_blocks, bs, H_kv, idx_dim) uint8
    cache_k_norm_ptr,               # (num_blocks, bs, H_kv)          fp32
    cache_v_idx_ptr,                # (num_blocks, bs, H_kv, idx_dim) uint8
    cache_v_norm_ptr,               # (num_blocks, bs, H_kv)          fp32
    cache_k_qjl_sign_ptr,           # (num_blocks, bs, H_kv, d/8) uint8 (prod only)
                                    # 8 sign bits per byte, bit_j = (sign_j < 0)
    cache_k_rnorm_ptr,              # (num_blocks, bs, H_kv)    fp32 (prod only)
    block_table_ptr,                # (num_seqs, block_table_stride) int32
    seq_id_per_query_ptr,           # (T_q,) int32
    kv_end_per_query_ptr,           # (T_q,) int32  one-past-last attended kv pos
    codebook_ptr,                   # (K_CB,) fp32  shared by K and V
    out_ptr,                        # (T_q, H_q, d) fp16/bf16  in rotated V space
    inv_d,                          # 1/d
    qjl_coef,                       # sqrt(pi/2)/d  (prod only)
    max_blocks_per_seq,             # int loop bound
    block_table_stride,             # int row stride
    OUT_DTYPE: tl.constexpr,
    num_heads_q: tl.constexpr,
    num_heads_kv: tl.constexpr,
    head_size: tl.constexpr,        # logical head dim
    idx_dim: tl.constexpr,          # physical last-dim of cache_k_idx /
                                    # cache_v_idx (head_size if unpacked,
                                    # head_size // 2 if 4-bit packed)
    qjl_dim: tl.constexpr,          # physical last-dim of cache_k_qjl_sign
                                    # (head_size // 8 when prod, else unused)
    block_size: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_BS: tl.constexpr,
    USE_QJL: tl.constexpr,
    USE_4BIT_PACK: tl.constexpr,    # True iff K_CB <= 16 (idx fits in nibble)
):
    q_idx = tl.program_id(0)
    qh_idx = tl.program_id(1)

    gqa_group = num_heads_q // num_heads_kv
    kvh_idx = qh_idx // gqa_group

    seq_idx = tl.load(seq_id_per_query_ptr + q_idx)
    kv_end = tl.load(kv_end_per_query_ptr + q_idx)
    num_blocks_q = (kv_end + block_size - 1) // block_size

    d_idx = tl.arange(0, BLOCK_D)
    mask_d = d_idx < head_size

    q_off = (q_idx * num_heads_q + qh_idx) * head_size + d_idx
    q_rot = tl.load(q_rotated_ptr + q_off, mask=mask_d, other=0.0).to(tl.float32)
    if USE_QJL:
        sq = tl.load(Sq_ptr + q_off, mask=mask_d, other=0.0).to(tl.float32)

    m_i = tl.full((), _NEG_LARGE, dtype=tl.float32)
    l_i = tl.zeros((), dtype=tl.float32)
    acc = tl.zeros((BLOCK_D,), dtype=tl.float32)

    # When 4-bit packed, two coords share a byte. d_pack picks the byte,
    # is_high picks the nibble. Computed once outside the loop.
    if USE_4BIT_PACK:
        d_pack = d_idx // 2                  # (BLOCK_D,)
        is_high = (d_idx % 2) == 1           # (BLOCK_D,)

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

                    # K idx load: packed or unpacked.
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

                    rk = tl.load(codebook_ptr + k_idx_u8).to(tl.float32)
                    main_dot = tl.sum(q_rot * rk)
                    k_norm = tl.load(cache_k_norm_ptr + meta)

                    if USE_QJL:
                        # QJL sign cache is bit-packed uint8 (8 signs/byte).
                        # bit_j = (sign_j < 0); inverse: sign = 1 - 2 * bit.
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

                    # V dequant in rotated space (||·|| = sqrt(d) per slot).
                    # Post-rotation is applied once per (query, head) in
                    # the Python wrapper, not per slot here.
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


def turboquant_paged_attention(
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
    """Paged attention with on-the-fly K and V dequantization.

    Pre-rotates Q (and Sq for prod) on-device via torch.matmul so the
    Triton kernel itself stays free of Hadamard multiplies. The kernel
    accumulates V in the rotated/sqrt(d)-normalized space; this wrapper
    applies the inverse rotation + signs + 1/sqrt(d) once per
    (query, head) on the kernel's output to recover the true
    softmax-weighted V sum.
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

    # Per-query metadata on-device (vectorized).
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

    # Pre-rotate Q (and Sq for prod) in native bf16/fp16 for tensor-core
    # matmul. H is symmetric so H.T == H.
    q_signed = q * state.signs
    q_rotated_k = (
        q_signed.reshape(-1, head_size) @ state.H
    ).view(num_query_tokens, num_heads_q, head_size).contiguous()

    if use_qjl:
        Sq_k = (
            q_rotated_k.reshape(-1, head_size) @ state.S.T
        ).view(num_query_tokens, num_heads_q, head_size).contiguous()
    else:
        # Unused when USE_QJL=False, but the kernel still needs a valid
        # pointer of the right dtype.
        Sq_k = q_rotated_k

    out = torch.empty_like(q)
    grid = (num_query_tokens, num_heads_q)
    BLOCK_D = triton.next_power_of_2(head_size)
    BLOCK_BS = block_size
    OUT_DTYPE = tl.bfloat16 if q.dtype == torch.bfloat16 else tl.float16

    inv_d = 1.0 / float(head_size)
    qjl_coef = math.sqrt(math.pi / 2.0) / float(head_size)

    # Use block_table.shape[1] as the outer loop bound. It is always >=
    # the max-required number of blocks (block_table is pre-allocated
    # large enough), and the kernel short-circuits per-block via
    # ``if block_i < num_blocks_q`` so extra iterations cost nothing.
    # This avoids a seq_lens.max().item() -> CPU sync per layer.
    block_table_stride = int(block_table.shape[1])
    actual_max_blocks = block_table_stride

    qjl_dim = head_size // 8 if use_qjl else 1
    if use_qjl:
        qjl_sign_buf = cache_k_qjl_sign
        assert qjl_sign_buf.dtype == torch.uint8, (
            f"cache_k_qjl_sign must be uint8 (bit-packed), "
            f"got {qjl_sign_buf.dtype}"
        )
        assert qjl_sign_buf.shape[-1] == qjl_dim, (
            f"cache_k_qjl_sign last dim {qjl_sign_buf.shape[-1]} != "
            f"expected {qjl_dim} (head_size // 8)"
        )
    else:
        # Kernel still needs a valid uint8 pointer even when USE_QJL=False.
        # Reuse cache_k_idx which is already uint8.
        qjl_sign_buf = cache_k_idx
    rnorm_buf = cache_k_rnorm if use_qjl else cache_k_norm

    _attend_kernel[grid](
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
        USE_QJL=use_qjl,
        USE_4BIT_PACK=use_4bit_pack,
    )

    # Post-rotate V back to original space in native dtype (tensor core).
    # H symmetric so H.T == H.
    inv_sqrt_d = 1.0 / math.sqrt(float(head_size))
    output = (
        out.reshape(-1, head_size) @ state.H
    ).view(num_query_tokens, num_heads_q, head_size)
    output = output * state.signs * inv_sqrt_d
    return output.contiguous()
