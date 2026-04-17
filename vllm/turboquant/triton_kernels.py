# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Triton kernels for TurboQuant Algorithm 1 (Q_mse) and Algorithm 2 (Q_prod).

Paper: Zandieh et al., arXiv:2504.19874.

Storage layout per (slot, head)
-------------------------------
Common to both algorithms:
    cache_k_idx    : uint8  (head_dim,)       -- MSE bucket index
    cache_k_norm   : fp32   ()                -- ||k||
    cache_v_fp     : fp16/bf16 (head_dim,)    -- V raw (no quant)

Only for Q_prod:
    cache_k_qjl_sign : int8 (head_dim,)       -- sign(S @ r_unit)
    cache_k_rnorm    : fp32 ()                -- ||r||

V rationale: paper only compresses K. int8 V with per-slot-per-head
scale was previously used but produced outputs exceeding max|v| in
long-prefill on Llama-3.1-8B (|attn|.max > max|v| is impossible for
correct softmax-weighted sum of V). Keeping V raw fixed this.

Attention (both algorithms work in rotated space)
-------------------------------------------------
Because Pi = H * diag(signs) is orthogonal:

    <q, k> / sqrt(d)
      = <H * diag(signs) * q, H * diag(signs) * k> / sqrt(d)
      = <q_rot, k_rot> / sqrt(d)

For a K vector stored as ``rotated = H * diag(signs) * (k * sqrt(d)/||k||)``
this reduces to (k_norm/d) * <q_rot, rotated>. Q_prod replaces ``rotated``
with ``codebook[idx] + r_norm * sqrt(pi/2)/d * S^T @ qjl_sign`` which is
an unbiased inner-product reconstruction (paper Lemma 4).
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from vllm.triton_utils import tl, triton

if TYPE_CHECKING:
    from vllm.turboquant.codebook import QuantState


_NEG_LARGE = tl.constexpr(-1.0e30)


# =========================================================================
# Store kernel: quantize K (mse or prod), store V as-is in bf16/fp16
# V is kept in the input dtype -- paper only quantizes K, and int8 V
# produced outputs exceeding max|v| in long-prefill cases (|attn|.max >
# max|v| is mathematically impossible for correct softmax-weighted sum).
# =========================================================================
@triton.jit
def _store_kernel(
    new_k_ptr,                      # (T, H_kv, d) fp16/bf16
    new_v_ptr,                      # (T, H_kv, d) fp16/bf16
    cache_k_idx_ptr,                # (num_blocks, bs, H_kv, d) uint8
    cache_k_norm_ptr,               # (num_blocks, bs, H_kv) fp32
    cache_v_fp_ptr,                 # (num_blocks, bs, H_kv, d) fp16/bf16 -- V raw
    cache_k_qjl_sign_ptr,           # int8 (prod only; unused if USE_QJL=False)
    cache_k_rnorm_ptr,              # fp32 (prod only)
    slot_mapping_ptr,               # (T,) int64
    codebook_ptr,                   # (K_CB,) fp32
    boundaries_ptr,                 # (K_CB - 1,) fp32
    hadamard_ptr,                   # (d, d) fp16/bf16
    signs_ptr,                      # (d,) fp16/bf16
    qjl_matrix_ptr,                 # (d, d) fp16/bf16 (prod only)
    num_heads_kv: tl.constexpr,
    head_size: tl.constexpr,
    block_size: tl.constexpr,
    K_CB: tl.constexpr,
    BLOCK_D: tl.constexpr,
    USE_QJL: tl.constexpr,
    V_DTYPE: tl.constexpr,
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

    # Per-dim and scalar-meta offsets into the paged cache.
    cache_off = (
        block_idx * block_size * num_heads_kv * head_size
        + off_in_block * num_heads_kv * head_size
        + head * head_size
        + d_idx
    )
    meta_off = (
        block_idx * block_size * num_heads_kv
        + off_in_block * num_heads_kv
        + head
    )

    # Load K, V and rotation primitives.
    kv_off = (tok * num_heads_kv + head) * head_size + d_idx
    k_vec = tl.load(new_k_ptr + kv_off, mask=mask_d, other=0.0).to(tl.float32)
    v_vec = tl.load(new_v_ptr + kv_off, mask=mask_d, other=0.0).to(tl.float32)

    signs = tl.load(signs_ptr + d_idx, mask=mask_d, other=1.0).to(tl.float32)
    row = d_idx[:, None]
    col = d_idx[None, :]
    mat_mask = (row < head_size) & (col < head_size)
    H_tile = tl.load(
        hadamard_ptr + row * head_size + col, mask=mat_mask, other=0.0
    ).to(tl.float32)

    # --- K: norm + rotation ---
    k_norm = tl.sqrt(tl.maximum(tl.sum(k_vec * k_vec), 1e-12))
    tl.store(cache_k_norm_ptr + meta_off, k_norm)
    k_normed = k_vec * (tl.sqrt(float(head_size)) / k_norm)
    rotated = tl.sum(H_tile * (k_normed * signs)[None, :], axis=1)

    # --- K: Lloyd-Max MSE quantization ---
    idx = tl.zeros((BLOCK_D,), dtype=tl.int32)
    for i in tl.static_range(K_CB - 1):
        b = tl.load(boundaries_ptr + i).to(tl.float32)
        idx += (rotated > b).to(tl.int32)
    tl.store(cache_k_idx_ptr + cache_off, idx.to(tl.uint8), mask=mask_d)

    # --- K: QJL residual (Algorithm 2 only) ---
    if USE_QJL:
        rk_dq = tl.load(codebook_ptr + idx).to(tl.float32)
        r = tl.where(mask_d, rotated - rk_dq, 0.0)
        r_norm = tl.sqrt(tl.maximum(tl.sum(r * r), 1e-12))
        r_unit = r / r_norm
        S_tile = tl.load(
            qjl_matrix_ptr + row * head_size + col, mask=mat_mask, other=0.0
        ).to(tl.float32)
        qjl_raw = tl.sum(S_tile * r_unit[None, :], axis=1)
        qjl_sign = tl.where(qjl_raw >= 0.0, 1.0, -1.0)
        tl.store(cache_k_rnorm_ptr + meta_off, r_norm)
        tl.store(
            cache_k_qjl_sign_ptr + cache_off,
            qjl_sign.to(tl.int8),
            mask=mask_d,
        )

    # --- V: stored raw in input dtype (no quantization) ---
    tl.store(cache_v_fp_ptr + cache_off, v_vec.to(V_DTYPE), mask=mask_d)


# =========================================================================
# Attend kernel: rotated-space logit + flash-style accumulation
# =========================================================================
@triton.jit
def _attend_kernel(
    q_rotated_ptr,                  # (T_q, H_q, d) fp16/bf16 -- H @ (signs * q)
    Sq_ptr,                         # (T_q, H_q, d) fp16/bf16 -- S @ q_rotated (prod only)
    cache_k_idx_ptr,                # (num_blocks, bs, H_kv, d) uint8
    cache_k_norm_ptr,               # (num_blocks, bs, H_kv) fp32
    cache_v_fp_ptr,                 # (num_blocks, bs, H_kv, d) fp16/bf16 -- V raw
    cache_k_qjl_sign_ptr,           # int8 (prod only)
    cache_k_rnorm_ptr,              # fp32 (prod only)
    block_table_ptr,                # (num_seqs, block_table_stride) int32
    seq_id_per_query_ptr,           # (T_q,) int32
    kv_end_per_query_ptr,           # (T_q,) int32
    codebook_ptr,                   # (K_CB,) fp32
    out_ptr,                        # (T_q, H_q, d) fp16/bf16
    inv_d,                          # fp32 1/d
    qjl_coef,                       # fp32 sqrt(pi/2)/d (prod only)
    max_blocks_per_seq,             # runtime int loop bound
    block_table_stride,             # runtime int row stride
    OUT_DTYPE: tl.constexpr,
    num_heads_q: tl.constexpr,
    num_heads_kv: tl.constexpr,
    head_size: tl.constexpr,
    block_size: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_BS: tl.constexpr,
    USE_QJL: tl.constexpr,
):
    """One program = one (query_token, q_head). Varlen + causal via
    ``seq_id_per_query`` and ``kv_end_per_query``."""
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

                    # MSE main term (always).
                    k_idx_u8 = tl.load(
                        cache_k_idx_ptr + base + d_idx, mask=mask_d, other=0
                    ).to(tl.int32)
                    rk = tl.load(codebook_ptr + k_idx_u8).to(tl.float32)
                    main_dot = tl.sum(q_rot * rk)

                    k_norm = tl.load(cache_k_norm_ptr + meta)

                    # Optional QJL correction (Algorithm 2).
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

                    # V raw (no dequant needed) + flash accumulation.
                    v_vec = tl.load(
                        cache_v_fp_ptr + base + d_idx, mask=mask_d, other=0.0
                    ).to(tl.float32)

                    m_new = tl.maximum(m_i, logit)
                    alpha = tl.exp(m_i - m_new)
                    beta = tl.exp(logit - m_new)
                    l_i = l_i * alpha + beta
                    acc = acc * alpha + beta * v_vec
                    m_i = m_new

    out_vec = acc / tl.maximum(l_i, 1e-12)
    out_off = (q_idx * num_heads_q + qh_idx) * head_size + d_idx
    tl.store(out_ptr + out_off, out_vec.to(OUT_DTYPE), mask=mask_d)


# =========================================================================
# Python wrappers
# =========================================================================
def turboquant_store_kv(
    new_k: torch.Tensor,
    new_v: torch.Tensor,
    cache_k_idx: torch.Tensor,
    cache_k_norm: torch.Tensor,
    cache_v_fp: torch.Tensor,
    slot_mapping: torch.Tensor,
    state: "QuantState",
    block_size: int,
    cache_k_qjl_sign: torch.Tensor | None = None,
    cache_k_rnorm: torch.Tensor | None = None,
) -> None:
    """Quantize K (Lloyd-Max) and copy V as-is into the paged cache.

    ``cache_k_qjl_sign`` and ``cache_k_rnorm`` are required when
    ``state.algo == "prod"`` and must be ``None`` for ``"mse"``.
    """
    num_tokens, num_heads_kv, head_size = new_k.shape
    K_CB = int(state.codebook.shape[0])

    boundaries = state.boundaries
    if boundaries.dtype != torch.float32:
        boundaries = boundaries.to(torch.float32)
    codebook = state.codebook
    if codebook.dtype != torch.float32:
        codebook = codebook.to(torch.float32)

    use_qjl = state.algo == "prod"
    if use_qjl:
        assert cache_k_qjl_sign is not None and cache_k_rnorm is not None, (
            "prod requires cache_k_qjl_sign and cache_k_rnorm"
        )
        qjl_matrix = state.S
        qjl_sign_buf = cache_k_qjl_sign
        rnorm_buf = cache_k_rnorm
    else:
        # Unused by kernel but must be valid pointers; reuse existing buffers.
        qjl_matrix = state.H
        qjl_sign_buf = cache_k_idx
        rnorm_buf = cache_k_norm

    v_tl_dtype = tl.bfloat16 if cache_v_fp.dtype == torch.bfloat16 else tl.float16

    grid = (num_tokens, num_heads_kv)
    BLOCK_D = triton.next_power_of_2(head_size)
    _store_kernel[grid](
        new_k,
        new_v,
        cache_k_idx,
        cache_k_norm,
        cache_v_fp,
        qjl_sign_buf,
        rnorm_buf,
        slot_mapping,
        codebook,
        boundaries,
        state.H,
        state.signs,
        qjl_matrix,
        num_heads_kv=num_heads_kv,
        head_size=head_size,
        block_size=block_size,
        K_CB=K_CB,
        BLOCK_D=BLOCK_D,
        USE_QJL=use_qjl,
        V_DTYPE=v_tl_dtype,
    )


def turboquant_paged_attention(
    q: torch.Tensor,
    cache_k_idx: torch.Tensor,
    cache_k_norm: torch.Tensor,
    cache_v_fp: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    query_start_loc: torch.Tensor,
    state: "QuantState",
    cache_k_qjl_sign: torch.Tensor | None = None,
    cache_k_rnorm: torch.Tensor | None = None,
) -> torch.Tensor:
    """Paged attention with dequant-on-the-fly K cache, varlen + causal.

    Per-query pre-rotations happen here on-device via torch.matmul so the
    attend kernel stays free of Hadamard multiplies. V is stored raw in
    ``cache_v_fp`` (bf16/fp16) and loaded directly without dequant.
    """
    num_query_tokens, num_heads_q, head_size = q.shape
    _, block_size, num_heads_kv, _ = cache_k_idx.shape
    num_seqs = int(seq_lens.shape[0])

    assert query_start_loc.shape[0] == num_seqs + 1
    assert block_table.shape[0] >= num_seqs

    codebook = state.codebook
    if codebook.dtype != torch.float32:
        codebook = codebook.to(torch.float32)

    use_qjl = state.algo == "prod"
    if use_qjl:
        assert cache_k_qjl_sign is not None and cache_k_rnorm is not None

    # --- Per-query metadata (on-device, vectorised) ----------------------
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

    # --- Pre-rotate Q (and Sq for prod) ----------------------------------
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
        Sq_k = q_rotated_k  # unused by kernel but must be a valid pointer

    out = torch.empty_like(q)
    grid = (num_query_tokens, num_heads_q)
    BLOCK_D = triton.next_power_of_2(head_size)
    BLOCK_BS = block_size
    OUT_DTYPE = tl.bfloat16 if q.dtype == torch.bfloat16 else tl.float16

    inv_d = 1.0 / float(head_size)
    qjl_coef = math.sqrt(math.pi / 2.0) / float(head_size)

    actual_max_seq = int(seq_lens.max().item())
    actual_max_blocks = (actual_max_seq + block_size - 1) // block_size
    block_table_stride = int(block_table.shape[1])

    qjl_sign_buf = cache_k_qjl_sign if use_qjl else cache_k_idx
    rnorm_buf = cache_k_rnorm if use_qjl else cache_k_norm

    _attend_kernel[grid](
        q_rotated_k,
        Sq_k,
        cache_k_idx,
        cache_k_norm,
        cache_v_fp,
        qjl_sign_buf,
        rnorm_buf,
        block_table,
        seq_id_per_query,
        kv_end_per_query,
        codebook,
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
    return out


# =========================================================================
# Pure PyTorch reference for `turboquant_paged_attention`.
# Same algorithm as the Triton kernel, but in fp32 with no flash trick.
# Used to localize bugs: if this gives correct output but the Triton
# kernel does not, the bug is in the kernel (memory layout, accumulation,
# static_range gating). If both give the same wrong output, the bug is
# in the algorithm (reconstruction math, inv_d, rotations).
#
# Eats the SAME cache buffers populated by `turboquant_store_kv`.
# Algorithm 2 (USE_QJL=True) is supported via the same Sq/qjl_sign/r_norm
# fields. Slow (Python loops); intended for diagnostic A/B only.
# =========================================================================
def python_paged_attention(
    q: torch.Tensor,
    cache_k_idx: torch.Tensor,
    cache_k_norm: torch.Tensor,
    cache_v_fp: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    query_start_loc: torch.Tensor,
    state: "QuantState",
    cache_k_qjl_sign: torch.Tensor | None = None,
    cache_k_rnorm: torch.Tensor | None = None,
) -> torch.Tensor:
    T_q, H_q, d = q.shape
    num_blocks, block_size, H_kv, _ = cache_k_idx.shape
    num_seqs = int(seq_lens.shape[0])
    gqa = H_q // H_kv
    use_qjl = state.algo == "prod"
    dev = q.device

    # Per-query metadata (mirrors the wrapper above).
    qsl = query_start_loc.to(device=dev, dtype=torch.int64)
    query_lens = qsl[1:] - qsl[:-1]
    seq_ids = torch.arange(num_seqs, dtype=torch.int64, device=dev)
    seq_id_per_q = torch.repeat_interleave(seq_ids, query_lens)
    q_pos = (
        torch.arange(T_q, dtype=torch.int64, device=dev)
        - qsl[:-1][seq_id_per_q]
    )
    prefix_len = seq_lens.to(device=dev, dtype=torch.int64) - query_lens
    kv_end_per_q = prefix_len[seq_id_per_q] + q_pos + 1

    # Pre-rotate Q (and Sq for prod) -- same as Triton wrapper.
    H_f = state.H.to(torch.float32)
    signs_f = state.signs.to(torch.float32)
    q_f = q.float()
    q_rot = (q_f * signs_f) @ H_f.T              # (T_q, H_q, d)
    if use_qjl:
        S_f = state.S.to(torch.float32)
        sq_full = q_rot @ S_f.T                  # (T_q, H_q, d)
    codebook = state.codebook.to(torch.float32)  # (K_CB,)

    inv_d = 1.0 / float(d)
    qjl_coef = math.sqrt(math.pi / 2.0) / float(d)

    out = torch.empty_like(q)
    for qi in range(T_q):
        seq = int(seq_id_per_q[qi].item())
        end = int(kv_end_per_q[qi].item())
        # Gather (end, H_kv, d) reconstructed K and raw V for the prefix.
        rk_seq = []
        knorm_seq = []
        v_seq = []
        if use_qjl:
            qjl_sign_seq = []
            rnorm_seq = []
        for pos in range(end):
            blk_local = pos // block_size
            tok = pos % block_size
            phys = int(block_table[seq, blk_local].item())
            k_idx = cache_k_idx[phys, tok].long()  # (H_kv, d)
            rk = codebook[k_idx]                   # (H_kv, d) fp32
            rk_seq.append(rk)
            knorm_seq.append(cache_k_norm[phys, tok].float())   # (H_kv,)
            v_seq.append(cache_v_fp[phys, tok].float())         # (H_kv, d)
            if use_qjl:
                qjl_sign_seq.append(
                    cache_k_qjl_sign[phys, tok].float()         # (H_kv, d)
                )
                rnorm_seq.append(
                    cache_k_rnorm[phys, tok].float()            # (H_kv,)
                )
        rk_t = torch.stack(rk_seq, dim=0)        # (end, H_kv, d)
        knorm_t = torch.stack(knorm_seq, dim=0)  # (end, H_kv)
        v_t = torch.stack(v_seq, dim=0)          # (end, H_kv, d)
        if use_qjl:
            qjl_sign_t = torch.stack(qjl_sign_seq, dim=0)   # (end, H_kv, d)
            rnorm_t = torch.stack(rnorm_seq, dim=0)         # (end, H_kv)

        for h in range(H_q):
            kh = h // gqa
            qh = q_rot[qi, h]                             # (d,)
            main_dot = (qh.unsqueeze(0) * rk_t[:, kh]).sum(dim=-1)  # (end,)
            if use_qjl:
                sqh = sq_full[qi, h]                      # (d,)
                qjl_dot = (sqh.unsqueeze(0) * qjl_sign_t[:, kh]).sum(dim=-1)
                logits = (
                    main_dot + qjl_coef * rnorm_t[:, kh] * qjl_dot
                ) * knorm_t[:, kh] * inv_d
            else:
                logits = main_dot * knorm_t[:, kh] * inv_d  # (end,)
            weights = torch.softmax(logits, dim=-1)         # (end,)
            out[qi, h] = (
                weights.unsqueeze(-1) * v_t[:, kh]          # (end, d)
            ).sum(dim=0).to(q.dtype)
    return out
