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
    q_ptr,                  # (num_query_tokens, num_heads_q, head_size) fp16/bf16
    cache_k_ptr,            # (num_blocks, block_size, num_heads_kv, head_size) uint8
    cache_v_ptr,            # (num_blocks, block_size, num_heads_kv, head_size) fp16/bf16
    cache_k_norm_ptr,       # (num_blocks, block_size, num_heads_kv) fp32
    block_table_ptr,        # (num_seqs, block_table_stride) int32
    seq_id_per_query_ptr,   # (num_query_tokens,) int32 : which seq each q belongs to
    kv_end_per_query_ptr,   # (num_query_tokens,) int32 : causal kv upper bound for each q
    codebook_ptr,           # (K_CB,) fp32
    hadamard_ptr,           # (head_size, head_size) fp16/bf16
    signs_ptr,              # (head_size,) fp16/bf16
    out_ptr,                # (num_query_tokens, num_heads_q, head_size) fp16/bf16
    scale,                  # fp32 scalar, 1 / sqrt(head_size)
    inv_sqrt_d,             # fp32 scalar, 1 / sqrt(head_size) for norm restoration
    max_blocks_per_seq,     # runtime int: loop bound; keep SMALL for speed
    block_table_stride,     # runtime int: real row stride of block_table
    OUT_DTYPE: tl.constexpr,
    num_heads_q: tl.constexpr,
    num_heads_kv: tl.constexpr,
    head_size: tl.constexpr,
    block_size: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_BS: tl.constexpr,
):
    """One program = one (query_token, q_head).

    Varlen-aware: grid.x spans ALL query tokens across ALL sequences in the
    batch (num_actual_tokens from vLLM metadata). Each program:
      1. reads its sequence id from ``seq_id_per_query``
      2. reads its causal kv upper bound from ``kv_end_per_query`` (for
         pure decode this equals seq_lens[seq]; for prefill position p in
         seq s it equals prefix_len[s] + p + 1)
      3. indexes block_table at row seq_idx, stride block_table_stride
      4. attends only to cache positions [0, kv_end)

    Earlier revisions treated ``program_id(0)`` as "sequence index" and
    pulled ``seq_lens[seq_idx]``, which was correct for decode (num_seqs==1
    case) but broken for multi-token prefill (all query tokens live in the
    same sequence so seq_idx > 0 overran seq_lens and skipped the causal
    mask). This version is the first to handle prefill properly.

    ``max_blocks_per_seq`` is kept as a runtime int so the outer loop
    stays a real loop -- unrolling at ctx=512 is what made Triton JIT
    hang for tens of minutes.
    """
    q_idx = tl.program_id(0)
    qh_idx = tl.program_id(1)

    gqa_group = num_heads_q // num_heads_kv
    kvh_idx = qh_idx // gqa_group

    seq_idx = tl.load(seq_id_per_query_ptr + q_idx)
    kv_end = tl.load(kv_end_per_query_ptr + q_idx)
    num_blocks = (kv_end + block_size - 1) // block_size

    d_idx = tl.arange(0, BLOCK_D)
    mask_d = d_idx < head_size

    q_off = (q_idx * num_heads_q + qh_idx) * head_size + d_idx
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

    for block_i in range(0, max_blocks_per_seq):
        if block_i < num_blocks:
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
    out_off = (q_idx * num_heads_q + qh_idx) * head_size + d_idx
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
    query_start_loc: torch.Tensor,
    codebook: GaussianCodebook,
    scale: float | None = None,
) -> torch.Tensor:
    """Paged attention with dequant-on-the-fly K cache, varlen-aware.

    q shape: (num_query_tokens, num_heads_q, head_size)
        num_query_tokens is the SUM of query lengths across all sequences
        in the batch (== vLLM's num_actual_tokens). For pure decode this
        equals num_seqs; for prefill it can be arbitrarily larger.

    Varlen metadata:
        seq_lens:        (num_seqs,)    int, total context length per seq
        query_start_loc: (num_seqs+1,)  int, cumulative query-token offsets

    We derive per-query-token tensors on-device:
        seq_id_per_query[i]   = which sequence query token i belongs to
        kv_end_per_query[i]   = causal upper bound, i.e. prefix_len[s] + p + 1
                                where s = seq_id_per_query[i] and p is the
                                within-sequence position of token i.
    """
    num_query_tokens, num_heads_q, head_size = q.shape
    _, block_size, num_heads_kv, _ = cache_k.shape
    num_seqs = int(seq_lens.shape[0])

    if scale is None:
        scale = 1.0 / (head_size ** 0.5)
    inv_sqrt_d = 1.0 / (head_size ** 0.5)

    codebook_f32 = codebook.codebook
    if codebook_f32.dtype != torch.float32:
        codebook_f32 = codebook_f32.to(torch.float32)

    # --- per-query metadata (all on-device, vectorised) ------------------
    # query_lens[s] = #query tokens contributed by sequence s in this batch
    qsl = query_start_loc.to(torch.int32)
    query_lens = qsl[1:] - qsl[:-1]                                # (num_seqs,)
    seq_ids = torch.arange(num_seqs, dtype=torch.int32, device=q.device)
    seq_id_per_query = torch.repeat_interleave(seq_ids, query_lens)  # (num_q,)
    # Position WITHIN sequence for each query token:
    q_pos_per_query = (
        torch.arange(num_query_tokens, dtype=torch.int32, device=q.device)
        - qsl[:-1][seq_id_per_query]
    )
    # prefix_len[s] = tokens already in cache BEFORE this batch's queries.
    prefix_len_per_seq = seq_lens.to(torch.int32) - query_lens     # (num_seqs,)
    kv_end_per_query = (
        prefix_len_per_seq[seq_id_per_query] + q_pos_per_query + 1
    ).to(torch.int32)

    out = torch.empty_like(q)
    grid = (num_query_tokens, num_heads_q)
    BLOCK_D = triton.next_power_of_2(head_size)
    BLOCK_BS = block_size
    OUT_DTYPE = tl.bfloat16 if q.dtype == torch.bfloat16 else tl.float16

    actual_max_seq = int(seq_lens.max().item())
    actual_max_blocks = (actual_max_seq + block_size - 1) // block_size
    # block_table is laid out with the scheduler's full per-seq stride
    # (typically max_model_len // block_size). The kernel must use THIS
    # stride to index rows; it must use actual_max_blocks only as the
    # runtime loop bound so Triton does not unroll the entire stride.
    block_table_stride = int(block_table.shape[1])

    _dequant_and_attend_kernel[grid](
        q,
        cache_k,
        cache_v,
        cache_k_norm,
        block_table,
        seq_id_per_query,
        kv_end_per_query,
        codebook_f32,
        codebook.H,
        codebook.signs,
        out,
        scale,
        inv_sqrt_d,
        actual_max_blocks,
        block_table_stride,
        OUT_DTYPE=OUT_DTYPE,
        num_heads_q=num_heads_q,
        num_heads_kv=num_heads_kv,
        head_size=head_size,
        block_size=block_size,
        BLOCK_D=BLOCK_D,
        BLOCK_BS=BLOCK_BS,
    )
    return out
