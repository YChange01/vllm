# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Triton kernels for TurboQuant quantized paged attention.

Two kernels:
  1. ``_quantize_and_store_kernel``: rotates + quantizes + stores a new K slot.
  2. ``_dequant_and_attend_kernel``: per-(seq, head) flash-style attention on
     paged quantized K and fp16/bf16 V.

Approach: dequantize K inside the tile (via codebook lookup) then run standard
attention, keeping the implementation simple. A future optimization can fuse
the query-side LUT (see ``tqlite/tqlite/mse.py::inner_product``) for extra
speed.

The current MVP assumes ``head_size`` is a power of 2 and ``bits == 4``. Called
from ``vllm/v1/attention/backends/turboquant_attn.py``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from vllm.triton_utils import tl, triton

if TYPE_CHECKING:
    from vllm.turboquant.codebook import GaussianCodebook


# ---------------------------------------------------------------------------
# Kernel 1: reshape + rotate + quantize + pack + store
# ---------------------------------------------------------------------------
@triton.jit
def _quantize_and_store_kernel(
    new_k_ptr,           # (num_tokens, num_heads_kv, head_size) fp16
    cache_k_ptr,         # (num_blocks, block_size, num_heads_kv, head_size/2) uint8
    slot_mapping_ptr,    # (num_tokens,) int64
    hadamard_ptr,        # (head_size, head_size) fp16
    signs_ptr,           # (head_size,) fp16
    boundaries_ptr,      # (K-1,) fp16, K = 2^bits
    num_heads_kv: tl.constexpr,
    head_size: tl.constexpr,
    block_size: tl.constexpr,
    BITS: tl.constexpr,
    K_CB: tl.constexpr,  # 2^BITS
    BLOCK_D: tl.constexpr,
):
    """One program handles one (token, kv_head) pair."""
    tok = tl.program_id(0)
    head = tl.program_id(1)

    slot = tl.load(slot_mapping_ptr + tok)
    if slot < 0:
        return

    block_idx = slot // block_size
    off_in_block = slot % block_size

    # Load new_k[tok, head, :].
    d_idx = tl.arange(0, BLOCK_D)
    mask_d = d_idx < head_size
    k_off = (tok * num_heads_kv + head) * head_size + d_idx
    k_vec = tl.load(new_k_ptr + k_off, mask=mask_d, other=0.0).to(tl.float32)

    # Apply signs (diag(s) * x).
    signs = tl.load(signs_ptr + d_idx, mask=mask_d, other=1.0).to(tl.float32)
    sk = k_vec * signs

    # Multiply by H^T: rotated[i] = sum_j sk[j] * H[i, j].
    rotated = tl.zeros((BLOCK_D,), dtype=tl.float32)
    for j in range(head_size):
        h_row = tl.load(
            hadamard_ptr + j * head_size + d_idx, mask=mask_d, other=0.0
        ).to(tl.float32)
        rotated += sk[j] * h_row

    # Bucketize against codebook decision boundaries.
    idx = tl.zeros((BLOCK_D,), dtype=tl.int32)
    for k in range(K_CB - 1):
        b = tl.load(boundaries_ptr + k).to(tl.float32)
        idx += (rotated > b).to(tl.int32)

    # MVP: store one uint8 per coordinate (holds a 4-bit idx in the low nibble).
    # A follow-up should pack two nibbles per byte to get the true 4x saving.
    idx_u8 = idx.to(tl.uint8) & 0x0F
    cache_off = (
        block_idx * block_size * num_heads_kv * head_size
        + off_in_block * num_heads_kv * head_size
        + head * head_size
        + d_idx
    )
    tl.store(cache_k_ptr + cache_off, idx_u8, mask=mask_d)


# ---------------------------------------------------------------------------
# Kernel 2: paged attention with quantized K cache
# ---------------------------------------------------------------------------
@triton.jit
def _dequant_and_attend_kernel(
    q_ptr,                # (num_seqs, num_heads_q, head_size) fp16
    cache_k_ptr,          # (num_blocks, block_size, num_heads_kv, head_size) uint8 idx
    cache_v_ptr,          # (num_blocks, block_size, num_heads_kv, head_size) fp16
    block_table_ptr,      # (num_seqs, max_blocks_per_seq) int32
    seq_lens_ptr,         # (num_seqs,) int32
    codebook_ptr,         # (K_CB,) fp16
    hadamard_ptr,         # (head_size, head_size) fp16
    signs_ptr,            # (head_size,) fp16
    out_ptr,              # (num_seqs, num_heads_q, head_size) fp16
    scale: tl.constexpr,  # 1 / sqrt(head_size)
    num_heads_q: tl.constexpr,
    num_heads_kv: tl.constexpr,
    head_size: tl.constexpr,
    block_size: tl.constexpr,
    max_blocks_per_seq: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_BS: tl.constexpr,
):
    """One program = one (seq, q_head).

    Per program:
      1. Load q[seq, q_head, :].
      2. Walk block_table[seq, :num_blocks].
      3. For every block, for every slot:
         a. Load k_idx (head_size,) uint8.
         b. rk = codebook[k_idx]  (fp32 via gather).
         c. k  = (rk @ H) * signs  (inverse rotation Pi^T).
         d. qk = (q . k) * scale.
         e. Update running flash-style softmax + v accumulator.
      4. Store acc / l_i as the attention output.
    """
    seq_idx = tl.program_id(0)
    qh_idx = tl.program_id(1)

    gqa_group = num_heads_q // num_heads_kv
    kvh_idx = qh_idx // gqa_group

    seq_len = tl.load(seq_lens_ptr + seq_idx)
    num_blocks = (seq_len + block_size - 1) // block_size

    d_idx = tl.arange(0, BLOCK_D)
    mask_d = d_idx < head_size
    bs_idx = tl.arange(0, BLOCK_BS)

    # load q
    q_off = (seq_idx * num_heads_q + qh_idx) * head_size + d_idx
    q = tl.load(q_ptr + q_off, mask=mask_d, other=0.0).to(tl.float32)

    # running softmax accumulators
    m_i = -float("inf")
    l_i = 0.0
    acc = tl.zeros((BLOCK_D,), dtype=tl.float32)

    # codebook into regs (K_CB small, assume <= 16)
    # 假设 K_CB = 16
    cb_idx = tl.arange(0, 16)
    cb_vals = tl.load(codebook_ptr + cb_idx).to(tl.float32)

    for block_i in range(0, max_blocks_per_seq):
        if block_i >= num_blocks:
            break
        phys_block = tl.load(block_table_ptr + seq_idx * max_blocks_per_seq + block_i)

        # One slot at a time inside the block (no intra-block tiling yet).
        for tok_in_block in range(0, BLOCK_BS):
            abs_pos = block_i * block_size + tok_in_block
            if abs_pos >= seq_len:
                break

            # load k_idx 和 v
            base = (
                phys_block * block_size * num_heads_kv * head_size
                + tok_in_block * num_heads_kv * head_size
                + kvh_idx * head_size
            )
            k_idx_u8 = tl.load(cache_k_ptr + base + d_idx, mask=mask_d, other=0).to(
                tl.int32
            )
            # codebook lookup
            rk = tl.gather(cb_vals, k_idx_u8, axis=0)
            # 反旋转: k = (rk @ H) * signs
            # 先算 rk @ H: out[i] = sum_j rk[j] * H[j, i]
            k_unrot = tl.zeros((BLOCK_D,), dtype=tl.float32)
            for j in range(head_size):
                h_col = tl.load(
                    hadamard_ptr + j * head_size + d_idx, mask=mask_d, other=0.0
                ).to(tl.float32)
                k_unrot += rk[j] * h_col
            signs = tl.load(signs_ptr + d_idx, mask=mask_d, other=1.0).to(tl.float32)
            k_vec = k_unrot * signs

            # q @ k scalar
            qk = tl.sum(q * k_vec) * scale

            # load v (fp16 cache, not quantized)
            v_off = base + d_idx
            v_vec = tl.load(cache_v_ptr + v_off, mask=mask_d, other=0.0).to(tl.float32)

            # running softmax update
            m_new = tl.maximum(m_i, qk)
            alpha = tl.exp(m_i - m_new)
            beta = tl.exp(qk - m_new)
            l_i = l_i * alpha + beta
            acc = acc * alpha + beta * v_vec
            m_i = m_new

    out = acc / l_i
    out_off = (seq_idx * num_heads_q + qh_idx) * head_size + d_idx
    tl.store(out_ptr + out_off, out.to(tl.float16), mask=mask_d)


# ---------------------------------------------------------------------------
# Python entrypoints
# ---------------------------------------------------------------------------
def turboquant_store_kv(
    new_k: torch.Tensor,         # (num_tokens, num_heads_kv, head_size) fp16
    new_v: torch.Tensor,         # (num_tokens, num_heads_kv, head_size) fp16
    cache_k: torch.Tensor,       # (num_blocks, block_size, num_heads_kv, head_size) uint8
    cache_v: torch.Tensor,       # (num_blocks, block_size, num_heads_kv, head_size) fp16
    slot_mapping: torch.Tensor,  # (num_tokens,) int64
    codebook: GaussianCodebook,
    block_size: int,
) -> None:
    """Quantize ``new_k`` and store it into the paged K cache as uint8 idx.

    MVP: one uint8 per coordinate (4 bits of payload, 4 bits wasted). A
    follow-up should pack two nibbles per byte to realize the 4x saving.
    """
    num_tokens, num_heads_kv, head_size = new_k.shape
    bits = codebook.bits
    K_CB = 2 ** bits

    # store V 直接 reshape + scatter
    for i in range(num_tokens):
        slot = int(slot_mapping[i].item())
        if slot < 0:
            continue
        block_idx = slot // block_size
        off = slot % block_size
        cache_v[block_idx, off] = new_v[i]

    # quantize + store K via kernel
    grid = (num_tokens, num_heads_kv)
    BLOCK_D = max(16, triton.next_power_of_2(head_size))
    _quantize_and_store_kernel[grid](
        new_k,
        cache_k,
        slot_mapping,
        codebook.H,
        codebook.signs,
        codebook.boundaries,
        num_heads_kv=num_heads_kv,
        head_size=head_size,
        block_size=block_size,
        BITS=bits,
        K_CB=K_CB,
        BLOCK_D=BLOCK_D,
    )


def turboquant_paged_attention(
    q: torch.Tensor,             # (num_seqs, num_heads_q, head_size) fp16
    cache_k: torch.Tensor,       # (num_blocks, block_size, num_heads_kv, head_size) uint8
    cache_v: torch.Tensor,       # (num_blocks, block_size, num_heads_kv, head_size) fp16
    block_table: torch.Tensor,   # (num_seqs, max_blocks_per_seq) int32
    seq_lens: torch.Tensor,      # (num_seqs,) int32
    codebook: GaussianCodebook,
    scale: float | None = None,
) -> torch.Tensor:
    num_seqs, num_heads_q, head_size = q.shape
    _, block_size, num_heads_kv, _ = cache_k.shape
    max_blocks_per_seq = block_table.shape[1]
    if scale is None:
        scale = 1.0 / (head_size ** 0.5)

    out = torch.empty_like(q)
    grid = (num_seqs, num_heads_q)
    BLOCK_D = max(16, triton.next_power_of_2(head_size))
    BLOCK_BS = block_size

    _dequant_and_attend_kernel[grid](
        q,
        cache_k,
        cache_v,
        block_table,
        seq_lens,
        codebook.codebook,
        codebook.H,
        codebook.signs,
        out,
        scale=scale,
        num_heads_q=num_heads_q,
        num_heads_kv=num_heads_kv,
        head_size=head_size,
        block_size=block_size,
        max_blocks_per_seq=max_blocks_per_seq,
        BLOCK_D=BLOCK_D,
        BLOCK_BS=BLOCK_BS,
    )
    return out
