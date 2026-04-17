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
    cache_k_idx_ptr,                # (num_blocks, bs, H_kv, d) uint8
    cache_k_norm_ptr,               # (num_blocks, bs, H_kv)    fp32
    cache_v_idx_ptr,                # (num_blocks, bs, H_kv, d) uint8
    cache_v_norm_ptr,               # (num_blocks, bs, H_kv)    fp32
    cache_k_qjl_sign_ptr,           # (num_blocks, bs, H_kv, d) int8 (prod only)
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
    head_size: tl.constexpr,
    block_size: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_BS: tl.constexpr,
    USE_QJL: tl.constexpr,
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

    for block_i in range(0, max_blocks_per_seq):
        if block_i < num_blocks_q:
            phys_block = tl.load(
                block_table_ptr + seq_idx * block_table_stride + block_i
            )
            for tok_in_block in tl.static_range(0, BLOCK_BS):
                abs_pos = block_i * block_size + tok_in_block
                if abs_pos < kv_end:
                    base = (
                        phys_block * block_size * num_heads_kv * head_size
                        + tok_in_block * num_heads_kv * head_size
                        + kvh_idx * head_size
                    )
                    meta = (
                        phys_block * block_size * num_heads_kv
                        + tok_in_block * num_heads_kv
                        + kvh_idx
                    )

                    # K dequant + logit.
                    k_idx_u8 = tl.load(
                        cache_k_idx_ptr + base + d_idx, mask=mask_d, other=0
                    ).to(tl.int32)
                    rk = tl.load(codebook_ptr + k_idx_u8).to(tl.float32)
                    main_dot = tl.sum(q_rot * rk)
                    k_norm = tl.load(cache_k_norm_ptr + meta)

                    if USE_QJL:
                        qjl_sign = tl.load(
                            cache_k_qjl_sign_ptr + base + d_idx,
                            mask=mask_d, other=0,
                        ).to(tl.float32)
                        qjl_dot = tl.sum(sq * qjl_sign)
                        r_norm = tl.load(cache_k_rnorm_ptr + meta)
                        logit = (main_dot + qjl_coef * r_norm * qjl_dot) \
                                * k_norm * inv_d
                    else:
                        logit = main_dot * k_norm * inv_d

                    # V dequant in rotated space (||·|| = sqrt(d) per slot).
                    # The post-rotation is applied once per (query, head)
                    # in the Python wrapper, not per slot here.
                    v_idx_u8 = tl.load(
                        cache_v_idx_ptr + base + d_idx, mask=mask_d, other=0
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
    _, block_size, num_heads_kv, _ = cache_k_idx.shape
    num_seqs = int(seq_lens.shape[0])
    assert query_start_loc.shape[0] == num_seqs + 1
    assert block_table.shape[0] >= num_seqs

    use_qjl = state.algo == "prod"
    if use_qjl:
        assert cache_k_qjl_sign is not None and cache_k_rnorm is not None

    codebook_f = state.codebook.to(torch.float32)

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

    # Pre-rotate Q (and Sq for prod). H is symmetric, so H.T == H.
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

    actual_max_blocks = (int(seq_lens.max().item()) + block_size - 1) // block_size
    block_table_stride = int(block_table.shape[1])

    qjl_sign_buf = cache_k_qjl_sign if use_qjl else cache_k_idx
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
        block_size=block_size,
        BLOCK_D=BLOCK_D,
        BLOCK_BS=BLOCK_BS,
        USE_QJL=use_qjl,
    )

    # Post-rotate V back to original space:
    #   output[d] = signs[d] * (H @ acc)[d] / sqrt(d)
    #            = signs[d] * (acc @ H.T)[d] / sqrt(d)        (H symmetric)
    out_f = out.float()
    output_f = (out_f @ H_f.T) * signs_f / math.sqrt(float(head_size))
    return output_f.to(out.dtype)
