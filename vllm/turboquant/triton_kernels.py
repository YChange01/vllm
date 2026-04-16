# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Triton kernels for TurboQuant quantized paged attention.

Two kernels:
  1. ``_quantize_and_store_kernel``: rotates + quantizes + stores a new K slot,
     and copies V unchanged.
  2. ``_dequant_and_attend_kernel``: per-(seq, q_head) flash-style attention on
     paged quantized K and fp16/bf16 V.

Design notes:
  - Hadamard multiplication uses a ``(d, d)`` tile and ``tl.sum`` broadcast,
    so no scalar indexing into tiles (which Triton forbids).
  - Codebook lookup uses pointer arithmetic (``tl.load(cb_ptr + idx)``)
    instead of ``tl.gather`` (not available in all Triton versions).
  - Per-row L2 normalization: each new K vector is normalized to
    ``||k|| = sqrt(d)`` BEFORE quantization. Dequantization multiplies by the
    same scalar ``sqrt(d) / d = 1 / sqrt(d)``. This keeps the Lloyd-Max
    codebook valid without storing a per-key scale.
  - Online softmax uses ``m_init = -1e30`` rather than ``-inf`` so the first
    iteration's ``exp(m_init - m_new)`` evaluates to (approximately) zero
    without producing NaN.

The MVP stores one uint8 per coordinate (low nibble carries the 4-bit idx).
A follow-up can pack two nibbles per byte to realize the full 4x memory
saving.
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
    slot_mapping_ptr,    # (num_tokens,) int64
    hadamard_ptr,        # (head_size, head_size) fp16/bf16 (symmetric, normalized)
    signs_ptr,           # (head_size,) fp16/bf16 (+/- 1)
    boundaries_ptr,      # (K_CB - 1,) fp32 codebook decision boundaries
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

    # Load k and v for this (token, head).
    kv_off = (tok * num_heads_kv + head) * head_size + d_idx
    k_vec = tl.load(new_k_ptr + kv_off, mask=mask_d, other=0.0).to(tl.float32)
    v_vec = tl.load(new_v_ptr + kv_off, mask=mask_d, other=0.0)

    # Per-row L2 normalize k to ||k|| = sqrt(head_size). Dequant multiplies by
    # inv_sqrt_d, so the final reconstructed scale matches the input direction
    # up to a shared constant absorbed by softmax (attention is scale-invariant
    # when all K share the same scaling).
    k_norm_sq = tl.sum(k_vec * k_vec)
    k_inv_norm = 1.0 / tl.sqrt(tl.maximum(k_norm_sq, 1e-12))
    # Target norm: sqrt(head_size), so scale by sqrt(d) / ||k||.
    k_scale = k_inv_norm * tl.sqrt(float(head_size))
    k_normed = k_vec * k_scale

    # Apply signs: sk = diag(s) * k.
    signs = tl.load(signs_ptr + d_idx, mask=mask_d, other=1.0).to(tl.float32)
    sk = k_normed * signs

    # Rotate: rotated[i] = sum_j sk[j] * H[i, j] using a full (d, d) tile.
    row_idx = d_idx[:, None]
    col_idx = d_idx[None, :]
    H_mask = (row_idx < head_size) & (col_idx < head_size)
    H_off = row_idx * head_size + col_idx
    H_tile = tl.load(hadamard_ptr + H_off, mask=H_mask, other=0.0).to(tl.float32)
    rotated = tl.sum(H_tile * sk[None, :], axis=1)

    # Bucketize against codebook boundaries -> 0..K_CB-1.
    idx = tl.zeros((BLOCK_D,), dtype=tl.int32)
    for k in tl.static_range(K_CB - 1):
        b = tl.load(boundaries_ptr + k).to(tl.float32)
        idx += (rotated > b).to(tl.int32)

    idx_u8 = idx.to(tl.uint8) & 0x0F

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
    block_table_ptr,      # (num_seqs, max_blocks_per_seq) int32
    seq_lens_ptr,         # (num_seqs,) int32
    codebook_ptr,         # (K_CB,) fp32
    hadamard_ptr,         # (head_size, head_size) fp16/bf16
    signs_ptr,            # (head_size,) fp16/bf16
    out_ptr,              # (num_seqs, num_heads_q, head_size) fp16/bf16
    scale,                # fp32 scalar, 1 / sqrt(head_size)
    OUT_DTYPE: tl.constexpr,  # tl.float16 or tl.bfloat16
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

    # Running flash-style softmax accumulators.
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
                    # Codebook lookup via pointer arithmetic (no tl.gather).
                    rk = tl.load(codebook_ptr + k_idx_u8).to(tl.float32)
                    # Inverse rotation: k = signs * (H @ rk).
                    k_unrot = tl.sum(H_tile * rk[None, :], axis=1)
                    k_vec = k_unrot * signs

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
    slot_mapping: torch.Tensor,
    codebook: GaussianCodebook,
    block_size: int,
) -> None:
    """Quantize ``new_k`` and store it into the paged K cache as uint8 idx,
    and copy ``new_v`` into the paged V cache.

    MVP: one uint8 per coordinate (the low nibble carries the 4-bit idx). A
    follow-up should pack two nibbles per byte for the full 4x saving.
    """
    num_tokens, num_heads_kv, head_size = new_k.shape
    K_CB = int(codebook.codebook.shape[0])

    # Prepare fp32 boundaries tensor if not already fp32.
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
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    codebook: GaussianCodebook,
    scale: float | None = None,
) -> torch.Tensor:
    num_seqs, num_heads_q, head_size = q.shape
    _, block_size, num_heads_kv, _ = cache_k.shape
    max_blocks_per_seq = block_table.shape[1]
    if scale is None:
        scale = 1.0 / (head_size ** 0.5)

    # The codebook is used via pointer arithmetic; cast to fp32 for numerical
    # consistency with the normalized store path.
    codebook_f32 = codebook.codebook
    if codebook_f32.dtype != torch.float32:
        codebook_f32 = codebook_f32.to(torch.float32)

    out = torch.empty_like(q)
    grid = (num_seqs, num_heads_q)
    BLOCK_D = triton.next_power_of_2(head_size)
    BLOCK_BS = block_size
    OUT_DTYPE = tl.bfloat16 if q.dtype == torch.bfloat16 else tl.float16

    # Use actual max blocks from seq_lens, not the pre-allocated block_table
    # width (which can be huge, e.g. 2560 for max_seq_len=40960). Using the
    # allocation size as tl.static_range bound causes Triton to unroll 40K+
    # iterations and hang during compilation.
    actual_max_seq = int(seq_lens.max().item())
    actual_max_blocks = (actual_max_seq + block_size - 1) // block_size

    _dequant_and_attend_kernel[grid](
        q,
        cache_k,
        cache_v,
        block_table,
        seq_lens,
        codebook_f32,
        codebook.H,
        codebook.signs,
        out,
        scale,
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
