# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Triton kernels for TurboQuant quantized paged attention.

Two kernels:
  1. ``_quantize_and_store_kernel``: L2-normalizes K, rotates, quantizes,
     stores uint8 idx + fp32 norm per key.
  2. ``_dequant_and_attend_kernel``: loads idx + norm, dequantizes with
     norm restoration, runs flash-style attention.

Per-key norm preservation:
  Store saves ``||k||`` as a separate fp32 scalar per (block, slot, head).
  Attend loads it and scales the dequantized unit-vector by ``||k|| / sqrt(d)``
  to recover the original magnitude. This is critical for attention logits.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from vllm.triton_utils import tl, triton

if TYPE_CHECKING:
    from vllm.turboquant.codebook import GaussianCodebook


_NEG_LARGE = tl.constexpr(-1.0e30)


@triton.jit
def _quantize_and_store_kernel(
    new_k_ptr,           # (num_tokens, num_heads_kv, head_size) fp16/bf16
    new_v_ptr,           # (num_tokens, num_heads_kv, head_size) fp16/bf16
    cache_k_ptr,         # (num_blocks, block_size, num_heads_kv, head_size) uint8
    cache_v_ptr,         # (num_blocks, block_size, num_heads_kv, head_size) fp16/bf16
    cache_k_norm_ptr,    # (num_blocks, block_size, num_heads_kv) fp32
    slot_mapping_ptr,    # (num_tokens,) int64
    hadamard_ptr,        # (head_size, head_size) fp16/bf16
    signs_ptr,           # (head_size,) fp16/bf16
    boundaries_ptr,      # (K_CB - 1,) fp32
    num_heads_kv: tl.constexpr,
    head_size: tl.constexpr,
    block_size: tl.constexpr,
    K_CB: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """One program = one (token, kv_head) pair."""
    tok = tl.program_id(0)
    head = tl.program_id(1)

    slot = tl.load(slot_mapping_ptr + tok)
    if slot < 0:
        return

    block_idx = slot // block_size
    off_in_block = slot % block_size

    d_idx = tl.arange(0, BLOCK_D)
    mask_d = d_idx < head_size

    kv_off = (tok * num_heads_kv + head) * head_size + d_idx
    k_vec = tl.load(new_k_ptr + kv_off, mask=mask_d, other=0.0).to(tl.float32)
    v_vec = tl.load(new_v_ptr + kv_off, mask=mask_d, other=0.0)

    # Compute and save L2 norm BEFORE normalization.
    k_norm_sq = tl.sum(k_vec * k_vec)
    k_norm = tl.sqrt(tl.maximum(k_norm_sq, 1e-12))

    # Store norm to separate buffer: cache_k_norm[block, slot, head].
    norm_off = block_idx * block_size * num_heads_kv + off_in_block * num_heads_kv + head
    tl.store(cache_k_norm_ptr + norm_off, k_norm)

    # Normalize to ||k|| = sqrt(head_size) for codebook compatibility.
    k_scale = tl.sqrt(float(head_size)) / k_norm
    k_normed = k_vec * k_scale

    # Apply signs: sk = diag(s) * k.
    signs = tl.load(signs_ptr + d_idx, mask=mask_d, other=1.0).to(tl.float32)
    sk = k_normed * signs

    # Rotate: rotated = H @ sk using a full (d, d) tile.
    row_idx = d_idx[:, None]
    col_idx = d_idx[None, :]
    H_mask = (row_idx < head_size) & (col_idx < head_size)
    H_off = row_idx * head_size + col_idx
    H_tile = tl.load(hadamard_ptr + H_off, mask=H_mask, other=0.0).to(tl.float32)
    rotated = tl.sum(H_tile * sk[None, :], axis=1)

    # Bucketize -> 0..K_CB-1.
    idx = tl.zeros((BLOCK_D,), dtype=tl.int32)
    for k in tl.static_range(K_CB - 1):
        b = tl.load(boundaries_ptr + k).to(tl.float32)
        idx += (rotated > b).to(tl.int32)

    # cache_k is uint8, so idx (in [0, K_CB-1]) fits directly for K_CB <= 256.
    # Do NOT mask with 0x0F: that would truncate the top 4 bits whenever
    # TURBOQUANT_BITS > 4, aliasing every 8-bit index onto the first 16
    # codebook entries (the extreme-negative tail of the Lloyd-Max table).
    idx_u8 = idx.to(tl.uint8)

    cache_off = (
        block_idx * block_size * num_heads_kv * head_size
        + off_in_block * num_heads_kv * head_size
        + head * head_size
        + d_idx
    )
    tl.store(cache_k_ptr + cache_off, idx_u8, mask=mask_d)
    tl.store(cache_v_ptr + cache_off, v_vec, mask=mask_d)


@triton.jit
def _dequant_and_attend_kernel(
    q_ptr,                # (num_seqs, num_heads_q, head_size) fp16/bf16
    cache_k_ptr,          # (num_blocks, block_size, num_heads_kv, head_size) uint8
    cache_v_ptr,          # (num_blocks, block_size, num_heads_kv, head_size) fp16/bf16
    cache_k_norm_ptr,     # (num_blocks, block_size, num_heads_kv) fp32
    block_table_ptr,      # (num_seqs, max_blocks_per_seq) int32
    seq_lens_ptr,         # (num_seqs,) int32
    codebook_ptr,         # (K_CB,) fp32
    hadamard_ptr,         # (head_size, head_size) fp16/bf16
    signs_ptr,            # (head_size,) fp16/bf16
    out_ptr,              # (num_seqs, num_heads_q, head_size) fp16/bf16
    scale,                # fp32 scalar, 1 / sqrt(head_size)
    inv_sqrt_d,           # fp32 scalar, 1 / sqrt(head_size) for norm restoration
    OUT_DTYPE: tl.constexpr,
    num_heads_q: tl.constexpr,
    num_heads_kv: tl.constexpr,
    head_size: tl.constexpr,
    block_size: tl.constexpr,
    max_blocks_per_seq: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_BS: tl.constexpr,
):
    """One program = one (seq, q_head)."""
    seq_idx = tl.program_id(0)
    qh_idx = tl.program_id(1)

    gqa_group = num_heads_q // num_heads_kv
    kvh_idx = qh_idx // gqa_group

    seq_len = tl.load(seq_lens_ptr + seq_idx)
    num_blocks = (seq_len + block_size - 1) // block_size

    d_idx = tl.arange(0, BLOCK_D)
    mask_d = d_idx < head_size

    q_off = (seq_idx * num_heads_q + qh_idx) * head_size + d_idx
    q = tl.load(q_ptr + q_off, mask=mask_d, other=0.0).to(tl.float32)

    signs = tl.load(signs_ptr + d_idx, mask=mask_d, other=1.0).to(tl.float32)

    row_idx = d_idx[:, None]
    col_idx = d_idx[None, :]
    H_mask = (row_idx < head_size) & (col_idx < head_size)
    H_off = row_idx * head_size + col_idx
    H_tile = tl.load(hadamard_ptr + H_off, mask=H_mask, other=0.0).to(tl.float32)

    m_i = tl.full((), _NEG_LARGE, dtype=tl.float32)
    l_i = tl.zeros((), dtype=tl.float32)
    acc = tl.zeros((BLOCK_D,), dtype=tl.float32)

    for block_i in tl.static_range(0, max_blocks_per_seq):
        if block_i < num_blocks:
            phys_block = tl.load(
                block_table_ptr + seq_idx * max_blocks_per_seq + block_i
            )

            for tok_in_block in tl.static_range(0, BLOCK_BS):
                abs_pos = block_i * block_size + tok_in_block
                if abs_pos < seq_len:
                    base = (
                        phys_block * block_size * num_heads_kv * head_size
                        + tok_in_block * num_heads_kv * head_size
                        + kvh_idx * head_size
                    )
                    k_idx_u8 = tl.load(
                        cache_k_ptr + base + d_idx, mask=mask_d, other=0
                    ).to(tl.int32)
                    rk = tl.load(codebook_ptr + k_idx_u8).to(tl.float32)
                    k_unrot = tl.sum(H_tile * rk[None, :], axis=1)
                    k_unit = k_unrot * signs

                    # Restore original norm: k = k_unit * (k_norm / sqrt(d)).
                    norm_off = (
                        phys_block * block_size * num_heads_kv
                        + tok_in_block * num_heads_kv
                        + kvh_idx
                    )
                    k_norm = tl.load(cache_k_norm_ptr + norm_off)
                    k_vec = k_unit * (k_norm * inv_sqrt_d)

                    qk = tl.sum(q * k_vec) * scale

                    v_vec = tl.load(
                        cache_v_ptr + base + d_idx, mask=mask_d, other=0.0
                    ).to(tl.float32)

                    m_new = tl.maximum(m_i, qk)
                    alpha = tl.exp(m_i - m_new)
                    beta = tl.exp(qk - m_new)
                    l_i = l_i * alpha + beta
                    acc = acc * alpha + beta * v_vec
                    m_i = m_new

    out_vec = acc / tl.maximum(l_i, 1e-12)
    out_off = (seq_idx * num_heads_q + qh_idx) * head_size + d_idx
    tl.store(out_ptr + out_off, out_vec.to(OUT_DTYPE), mask=mask_d)


def turboquant_store_kv(
    new_k: torch.Tensor,
    new_v: torch.Tensor,
    cache_k: torch.Tensor,
    cache_v: torch.Tensor,
    cache_k_norm: torch.Tensor,
    slot_mapping: torch.Tensor,
    codebook: GaussianCodebook,
    block_size: int,
) -> None:
    """Quantize ``new_k``, store idx + norm into paged K cache, copy V."""
    num_tokens, num_heads_kv, head_size = new_k.shape
    K_CB = int(codebook.codebook.shape[0])

    boundaries = codebook.boundaries
    if boundaries.dtype != torch.float32:
        boundaries = boundaries.to(torch.float32)

    grid = (num_tokens, num_heads_kv)
    BLOCK_D = triton.next_power_of_2(head_size)

    _quantize_and_store_kernel[grid](
        new_k,
        new_v,
        cache_k,
        cache_v,
        cache_k_norm,
        slot_mapping,
        codebook.H,
        codebook.signs,
        boundaries,
        num_heads_kv=num_heads_kv,
        head_size=head_size,
        block_size=block_size,
        K_CB=K_CB,
        BLOCK_D=BLOCK_D,
    )


def turboquant_paged_attention(
    q: torch.Tensor,
    cache_k: torch.Tensor,
    cache_v: torch.Tensor,
    cache_k_norm: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    codebook: GaussianCodebook,
    scale: float | None = None,
) -> torch.Tensor:
    num_seqs, num_heads_q, head_size = q.shape
    _, block_size, num_heads_kv, _ = cache_k.shape
    if scale is None:
        scale = 1.0 / (head_size ** 0.5)
    inv_sqrt_d = 1.0 / (head_size ** 0.5)

    codebook_f32 = codebook.codebook
    if codebook_f32.dtype != torch.float32:
        codebook_f32 = codebook_f32.to(torch.float32)

    out = torch.empty_like(q)
    grid = (num_seqs, num_heads_q)
    BLOCK_D = triton.next_power_of_2(head_size)
    BLOCK_BS = block_size
    OUT_DTYPE = tl.bfloat16 if q.dtype == torch.bfloat16 else tl.float16

    actual_max_seq = int(seq_lens.max().item())
    actual_max_blocks = (actual_max_seq + block_size - 1) // block_size

    _dequant_and_attend_kernel[grid](
        q,
        cache_k,
        cache_v,
        cache_k_norm,
        block_table,
        seq_lens,
        codebook_f32,
        codebook.H,
        codebook.signs,
        out,
        scale,
        inv_sqrt_d,
        OUT_DTYPE=OUT_DTYPE,
        num_heads_q=num_heads_q,
        num_heads_kv=num_heads_kv,
        head_size=head_size,
        block_size=block_size,
        max_blocks_per_seq=actual_max_blocks,
        BLOCK_D=BLOCK_D,
        BLOCK_BS=BLOCK_BS,
    )
    return out
