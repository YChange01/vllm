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


# Note: the previous Triton store kernel (_store_kernel) was removed.
# Multi-token grids on B200 produced corrupted V (and apparently QJL)
# writes -- mse and prod gave bit-identical |attn|.mean across layers,
# which was a tell that the QJL residual writes were also garbage.
# The Python store implementation below uses standard PyTorch ops on
# the GPU (cuBLAS-backed matmul, fused element-wise, scatter via
# advanced indexing); no custom Triton kernel for store at all.
# Triton is still used for the attend kernel, which is the hot path
# during decode and has not shown the same multi-program issue.


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
    cache_k_idx: torch.Tensor,
    cache_k_norm: torch.Tensor,
    slot_mapping: torch.Tensor,
    state: "QuantState",
    block_size: int,
    cache_k_qjl_sign: torch.Tensor | None = None,
    cache_k_rnorm: torch.Tensor | None = None,
) -> None:
    """Quantize K (Lloyd-Max + Hadamard) into the paged cache.

    Pure PyTorch implementation. Runs on the same device as ``new_k``
    via standard tensor ops (matmul -> cuBLAS, searchsorted, scatter).
    V is stored separately in Python by the caller (do_kv_cache_update).

    ``cache_k_qjl_sign`` and ``cache_k_rnorm`` are required when
    ``state.algo == "prod"`` and must be ``None`` for ``"mse"``.
    """
    if new_k.shape[0] == 0:
        return

    T, H_kv, d = new_k.shape
    K_CB = int(state.codebook.shape[0])
    use_qjl = state.algo == "prod"
    if use_qjl:
        assert cache_k_qjl_sign is not None and cache_k_rnorm is not None, (
            "prod requires cache_k_qjl_sign and cache_k_rnorm"
        )

    # All compute in fp32 on GPU for numerical safety.
    k_f = new_k.float()
    H_f = state.H.to(torch.float32)
    signs_f = state.signs.to(torch.float32)
    boundaries_f = state.boundaries.to(torch.float32)

    # ||k|| per (token, head); clamp away the singular case.
    k_norm = k_f.norm(dim=-1).clamp_min(1e-6)            # (T, H_kv)

    # k_normed has ||.|| == sqrt(d) per (token, head).
    inv_norm = (math.sqrt(float(d)) / k_norm).unsqueeze(-1)  # (T, H_kv, 1)
    k_normed = k_f * inv_norm                             # (T, H_kv, d)

    # rotated[..., i] = sum_j H[i, j] * (signs * k_normed)[..., j]
    #                 = (H @ (signs * k_normed))[..., i]
    # In row-vec form on the last dim: (k_normed * signs) @ H.T
    rotated = (k_normed * signs_f) @ H_f.T                # (T, H_kv, d)

    # Lloyd-Max bucket via binary search; matches Triton's
    # idx = sum((rotated > b) for b in boundaries).
    idx32 = torch.searchsorted(boundaries_f.contiguous(),
                               rotated.contiguous())      # (T, H_kv, d) int64
    idx_uint8 = idx32.clamp(0, K_CB - 1).to(torch.uint8)  # (T, H_kv, d)

    if use_qjl:
        codebook_f = state.codebook.to(torch.float32)
        S_f = state.S.to(torch.float32)
        rk = codebook_f[idx32.clamp(0, K_CB - 1)]         # (T, H_kv, d)
        r = rotated - rk                                  # (T, H_kv, d)
        r_norm = r.norm(dim=-1).clamp_min(1e-6)           # (T, H_kv)
        r_unit = r / r_norm.unsqueeze(-1)                 # (T, H_kv, d)
        # qjl_raw[..., i] = sum_j S[i, j] * r_unit[..., j]
        qjl_raw = r_unit @ S_f.T                          # (T, H_kv, d)
        qjl_sign = torch.where(
            qjl_raw >= 0,
            torch.ones((), dtype=torch.float32, device=qjl_raw.device),
            -torch.ones((), dtype=torch.float32, device=qjl_raw.device),
        ).to(torch.int8)                                  # (T, H_kv, d)

    # Scatter into paged cache via slot_mapping. Skip padded (-1) tokens.
    valid = slot_mapping >= 0
    if not bool(valid.any()):
        return
    slots = slot_mapping[valid].to(torch.int64)
    b_idx = slots // block_size
    off = slots % block_size

    cache_k_idx[b_idx, off] = idx_uint8[valid]
    cache_k_norm[b_idx, off] = k_norm[valid]
    if use_qjl:
        cache_k_qjl_sign[b_idx, off] = qjl_sign[valid]
        cache_k_rnorm[b_idx, off] = r_norm[valid]


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
