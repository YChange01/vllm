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
    cache_v_ptr,         # (num_blocks, block_size, num_heads_kv, head_size) int8   <-- Step 1
    cache_k_norm_ptr,    # (num_blocks, block_size, num_heads_kv) fp32
    cache_v_scale_ptr,   # (num_blocks, block_size, num_heads_kv) fp32              <-- Step 1
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
    """One program = one (token, kv_head) pair.

    Step 1 adds V quantization on the store path: for each (token, kv_head)
    we compute scale = max(|v|) / 127 and store the per-element int8 value
    plus the scale. Dequant in the attend kernel is `int8 * scale`. Per-head
    symmetric int8 is a first pass -- cheap, no rotation, no codebook, good
    enough to prove the pipeline and cut V memory roughly in half.
    """
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
    v_vec = tl.load(new_v_ptr + kv_off, mask=mask_d, other=0.0).to(tl.float32)

    # --- K quantization (unchanged from before Step 1) ---
    k_norm_sq = tl.sum(k_vec * k_vec)
    k_norm = tl.sqrt(tl.maximum(k_norm_sq, 1e-12))

    # Per-(slot, head) offset for scalar metadata buffers (k_norm, v_scale).
    meta_off = block_idx * block_size * num_heads_kv + off_in_block * num_heads_kv + head
    tl.store(cache_k_norm_ptr + meta_off, k_norm)

    k_scale = tl.sqrt(float(head_size)) / k_norm
    k_normed = k_vec * k_scale

    signs = tl.load(signs_ptr + d_idx, mask=mask_d, other=1.0).to(tl.float32)
    sk = k_normed * signs

    row_idx = d_idx[:, None]
    col_idx = d_idx[None, :]
    H_mask = (row_idx < head_size) & (col_idx < head_size)
    H_off = row_idx * head_size + col_idx
    H_tile = tl.load(hadamard_ptr + H_off, mask=H_mask, other=0.0).to(tl.float32)
    rotated = tl.sum(H_tile * sk[None, :], axis=1)

    idx = tl.zeros((BLOCK_D,), dtype=tl.int32)
    for k in tl.static_range(K_CB - 1):
        b = tl.load(boundaries_ptr + k).to(tl.float32)
        idx += (rotated > b).to(tl.int32)
    idx_u8 = idx.to(tl.uint8)

    cache_off = (
        block_idx * block_size * num_heads_kv * head_size
        + off_in_block * num_heads_kv * head_size
        + head * head_size
        + d_idx
    )
    tl.store(cache_k_ptr + cache_off, idx_u8, mask=mask_d)

    # --- V quantization (Step 1) ---
    # Per-(token, head) symmetric int8: scale = max|v| / 127, stored as fp32.
    v_abs = tl.where(mask_d, tl.abs(v_vec), 0.0)
    v_max = tl.max(v_abs, axis=0)
    # Guard against a fully-zero vector: we would divide by 0.
    v_scale = tl.maximum(v_max / 127.0, 1e-12)
    v_int = (v_vec / v_scale).to(tl.int8)
    tl.store(cache_v_scale_ptr + meta_off, v_scale)
    tl.store(cache_v_ptr + cache_off, v_int, mask=mask_d)


@triton.jit
def _dequant_and_attend_kernel(
    q_ptr,                  # (num_query_tokens, num_heads_q, head_size) fp16/bf16
    cache_k_ptr,            # (num_blocks, block_size, num_heads_kv, head_size) uint8
    cache_v_ptr,            # (num_blocks, block_size, num_heads_kv, head_size) int8
    cache_k_norm_ptr,       # (num_blocks, block_size, num_heads_kv) fp32
    cache_v_scale_ptr,      # (num_blocks, block_size, num_heads_kv) fp32 (Step 1)
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

                    # V dequant (Step 1): int8 per-(slot, head) * per-(slot, head) fp32 scale.
                    v_int8 = tl.load(
                        cache_v_ptr + base + d_idx, mask=mask_d, other=0
                    ).to(tl.float32)
                    v_scale = tl.load(cache_v_scale_ptr + norm_off)
                    v_vec = v_int8 * v_scale

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
    cache_v_scale: torch.Tensor,
    slot_mapping: torch.Tensor,
    codebook: GaussianCodebook,
    block_size: int,
) -> None:
    """Quantize ``new_k`` (Lloyd-Max + Hadamard) and ``new_v`` (per-(slot,head)
    symmetric int8) into the paged K/V caches.

    Step 1 extends this to also produce V scales. ``cache_v`` is now an int8
    tensor (same shape as cache_k) and ``cache_v_scale`` is a per-(slot,head)
    fp32 scale tensor parallel to ``cache_k_norm``.
    """
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
        cache_v_scale,
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


_TQ_ATTN_DEBUG_COUNTER = [0]


def turboquant_paged_attention(
    q: torch.Tensor,
    cache_k: torch.Tensor,
    cache_v: torch.Tensor,
    cache_k_norm: torch.Tensor,
    cache_v_scale: torch.Tensor,
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
    import os
    _tq_debug = int(os.environ.get("TQ_DEBUG", "0"))

    num_query_tokens, num_heads_q, head_size = q.shape
    _, block_size, num_heads_kv, _ = cache_k.shape
    num_seqs = int(seq_lens.shape[0])

    if scale is None:
        scale = 1.0 / (head_size ** 0.5)
    inv_sqrt_d = 1.0 / (head_size ** 0.5)

    # Shape contract sanity.  Any mismatch here means vLLM fed us metadata
    # whose layout we do NOT understand; continuing silently gives garbage
    # attention (seen in the NIAH eval).  Fail loudly instead.
    assert query_start_loc.shape[0] == num_seqs + 1, (
        f"query_start_loc length {query_start_loc.shape[0]} != num_seqs+1 "
        f"({num_seqs + 1}); shapes q={tuple(q.shape)} "
        f"seq_lens={tuple(seq_lens.shape)} "
        f"qsl={tuple(query_start_loc.shape)}"
    )
    assert block_table.shape[0] >= num_seqs, (
        f"block_table rows {block_table.shape[0]} < num_seqs {num_seqs}"
    )

    if _tq_debug and _TQ_ATTN_DEBUG_COUNTER[0] < 8:
        _TQ_ATTN_DEBUG_COUNTER[0] += 1
        import torch as _t
        _qsl_cpu = query_start_loc.detach().to("cpu").tolist()
        _sl_cpu = seq_lens.detach().to("cpu").tolist()
        _bt_row0_head = block_table[0, :8].detach().to("cpu").tolist() \
            if block_table.numel() > 0 else []
        print(
            f"[TQ_ATTN #{_TQ_ATTN_DEBUG_COUNTER[0]}] "
            f"q.shape={tuple(q.shape)} q.dtype={q.dtype} "
            f"num_query_tokens={num_query_tokens} num_seqs={num_seqs} "
            f"qsl={_qsl_cpu} qsl.dtype={query_start_loc.dtype} "
            f"seq_lens={_sl_cpu} seq_lens.dtype={seq_lens.dtype} "
            f"block_table.shape={tuple(block_table.shape)} "
            f"block_table.stride={tuple(block_table.stride())} "
            f"block_table[0,:8]={_bt_row0_head} "
            f"cache_k.shape={tuple(cache_k.shape)}",
            flush=True,
        )

    codebook_f32 = codebook.codebook
    if codebook_f32.dtype != torch.float32:
        codebook_f32 = codebook_f32.to(torch.float32)

    # --- per-query metadata (all on-device, vectorised) ------------------
    # Everything downstream needs to live on the same device as q and use
    # int64 for index ops / int32 only for the final kernel-arg tensors.
    # torch.repeat_interleave requires the `repeats` tensor to be Long, and
    # advanced indexing with int32 indices has been quirky across torch
    # versions -- use int64 throughout the host-side math, cast to int32
    # only when handing tensors to the kernel.
    dev = q.device
    qsl = query_start_loc.to(device=dev, dtype=torch.int64)
    query_lens = qsl[1:] - qsl[:-1]                                # (num_seqs,)
    seq_ids = torch.arange(num_seqs, dtype=torch.int64, device=dev)
    seq_id_per_query_i64 = torch.repeat_interleave(seq_ids, query_lens)  # (num_q,)
    q_pos_per_query_i64 = (
        torch.arange(num_query_tokens, dtype=torch.int64, device=dev)
        - qsl[:-1][seq_id_per_query_i64]
    )
    prefix_len_per_seq_i64 = seq_lens.to(device=dev, dtype=torch.int64) - query_lens
    kv_end_per_query_i64 = (
        prefix_len_per_seq_i64[seq_id_per_query_i64] + q_pos_per_query_i64 + 1
    )

    # Kernel expects int32 pointers.
    seq_id_per_query = seq_id_per_query_i64.to(torch.int32).contiguous()
    kv_end_per_query = kv_end_per_query_i64.to(torch.int32).contiguous()

    if _tq_debug and _TQ_ATTN_DEBUG_COUNTER[0] <= 8:
        _kv_end_cpu = kv_end_per_query.detach().to("cpu").tolist()
        _seq_id_cpu = seq_id_per_query.detach().to("cpu").tolist()
        print(
            f"[TQ_ATTN #{_TQ_ATTN_DEBUG_COUNTER[0]} derived] "
            f"query_lens={query_lens.detach().to('cpu').tolist()} "
            f"prefix_len={prefix_len_per_seq_i64.detach().to('cpu').tolist()} "
            f"seq_id_per_query[:16]={_seq_id_cpu[:16]} "
            f"kv_end_per_query[:16]={_kv_end_cpu[:16]} "
            f"kv_end_per_query[-1]={_kv_end_cpu[-1] if _kv_end_cpu else None}",
            flush=True,
        )

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
        cache_v_scale,
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

    # --- optional inline verification against a pure-PyTorch reference ----
    # Enable with TQ_VERIFY=1 in the server env. This is slow (O(N^2) per
    # call) but answers the last remaining question: given the EXACT tensors
    # vLLM hands us, does the Triton kernel output match the mathematical
    # reference? If diff is tiny, the bug is downstream of us.
    if int(os.environ.get("TQ_VERIFY", "0")) and num_query_tokens <= 128:
        with torch.no_grad():
            cb_ref = codebook.codebook.to(torch.float32)
            H_ref = codebook.H.to(torch.float32)
            signs_ref = codebook.signs.to(torch.float32)
            max_seq_for_ref = int(seq_lens.max().item())
            # Dequantise K for sequence 0 up to max_seq_for_ref.
            k_ref = torch.zeros(
                max_seq_for_ref, num_heads_kv, head_size,
                dtype=torch.float32, device=q.device,
            )
            v_ref = torch.zeros_like(k_ref)
            for pos in range(max_seq_for_ref):
                bi = pos // block_size
                toff = pos % block_size
                phys = int(block_table[0, bi].item())
                for h in range(num_heads_kv):
                    idx = cache_k[phys, toff, h].long()
                    rk = cb_ref[idx]
                    k_unrot = rk @ H_ref
                    k_unit = k_unrot * signs_ref
                    k_norm_h = float(cache_k_norm[phys, toff, h].item())
                    k_ref[pos, h] = k_unit * (k_norm_h * inv_sqrt_d)
                    v_scale_h = float(cache_v_scale[phys, toff, h].item())
                    v_ref[pos, h] = cache_v[phys, toff, h].float() * v_scale_h

            # Pure-PyTorch causal attention
            gqa = num_heads_q // num_heads_kv
            ref_out = torch.zeros_like(q, dtype=torch.float32)
            for qi in range(num_query_tokens):
                kv_end_i = int(kv_end_per_query[qi].item())
                for h in range(num_heads_q):
                    kh = h // gqa
                    qv = q[qi, h].float()
                    kk = k_ref[:kv_end_i, kh]
                    vv = v_ref[:kv_end_i, kh]
                    sc = (qv @ kk.T) * scale
                    w = torch.softmax(sc, dim=-1)
                    ref_out[qi, h] = w @ vv

            diff = (out.float() - ref_out).abs()
            ref_abs_mean = ref_out.abs().mean().clamp(min=1e-6)
            print(
                f"[TQ_VERIFY] num_q={num_query_tokens} "
                f"max_abs_err={float(diff.max().item()):.4f} "
                f"mean_abs_err={float(diff.mean().item()):.4f} "
                f"rel_err={float((diff.mean() / ref_abs_mean).item()):.4%} "
                f"out_std={float(out.float().std().item()):.4f} "
                f"ref_std={float(ref_out.std().item()):.4f} "
                f"q_std={float(q.float().std().item()):.4f}",
                flush=True,
            )
    return out
