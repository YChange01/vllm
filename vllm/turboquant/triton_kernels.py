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
    new_k_ptr,                  # (num_tokens, num_heads_kv, head_size) fp16/bf16
    new_v_ptr,                  # (num_tokens, num_heads_kv, head_size) fp16/bf16
    cache_k_ptr,                # (num_blocks, block_size, num_heads_kv, head_size) uint8
    cache_v_ptr,                # (num_blocks, block_size, num_heads_kv, head_size) int8
    cache_k_norm_ptr,           # (num_blocks, block_size, num_heads_kv) fp32
    cache_v_scale_ptr,          # (num_blocks, block_size, num_heads_kv) fp32
    cache_k_qjl_sign_ptr,       # (num_blocks, block_size, num_heads_kv, head_size) int8  <-- Algo 2 Stage 2
    cache_k_rnorm_ptr,          # (num_blocks, block_size, num_heads_kv) fp32             <-- Algo 2 Stage 2
    slot_mapping_ptr,           # (num_tokens,) int64
    codebook_ptr,               # (K_CB,) fp32, MSE codebook with 2^(bits-1) entries
    hadamard_ptr,               # (head_size, head_size) fp16/bf16, main rotation
    qjl_matrix_ptr,             # (head_size, head_size) fp16/bf16, QJL iid-Gaussian S
    signs_ptr,                  # (head_size,) fp16/bf16
    boundaries_ptr,             # (K_CB - 1,) fp32
    num_heads_kv: tl.constexpr,
    head_size: tl.constexpr,
    block_size: tl.constexpr,
    K_CB: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """One program = one (token, kv_head) pair.

    Implements the TurboQuant Algorithm 2 store path (paper §1.3, §2.2):

      Stage 0 (norm)   : k_norm = ||k||, k_normed = k * sqrt(d) / k_norm
      Stage 1 (rotate) : rotated = H @ (signs * k_normed)   -- paper Pi
      Stage 2 (MSE)    : idx = bucketize(rotated, boundaries)  using
                         a (b-1)-bit Lloyd-Max codebook
      Stage 3 (resid)  : r = rotated - codebook[idx]
                         r_norm = ||r||, r_unit = r / r_norm
      Stage 4 (QJL)    : qjl_sign = sign(S @ r_unit),  S iid N(0,1)
                         [paper Definition 1]

    At attend time the reconstruction is
        rotated_approx = codebook[idx] + r_norm * sqrt(pi/2)/d * S^T @ qjl_sign
    which gives an unbiased inner-product estimator (Lemma 4).

    V uses per-(slot, head) symmetric int8 (scale = max(|v|)/127); that path
    is untouched by the paper and kept from the earlier commit.
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

    # --- K norm + rotation --------------------------------------------------
    k_norm_sq = tl.sum(k_vec * k_vec)
    k_norm = tl.sqrt(tl.maximum(k_norm_sq, 1e-12))

    meta_off = block_idx * block_size * num_heads_kv + off_in_block * num_heads_kv + head
    tl.store(cache_k_norm_ptr + meta_off, k_norm)

    k_scale = tl.sqrt(float(head_size)) / k_norm
    k_normed = k_vec * k_scale

    signs = tl.load(signs_ptr + d_idx, mask=mask_d, other=1.0).to(tl.float32)
    sk = k_normed * signs

    row_idx = d_idx[:, None]
    col_idx = d_idx[None, :]
    mat_mask = (row_idx < head_size) & (col_idx < head_size)
    mat_off = row_idx * head_size + col_idx
    H_tile = tl.load(hadamard_ptr + mat_off, mask=mat_mask, other=0.0).to(tl.float32)
    rotated = tl.sum(H_tile * sk[None, :], axis=1)

    # --- MSE quantization (Algorithm 2 Stage 2, b-1 bits) -------------------
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

    # --- QJL on residual (Algorithm 2 Stage 4) ------------------------------
    # Dequant the freshly-picked index, form residual in rotated space, split
    # magnitude (r_norm) from direction (r_unit), then QJL-quantize the unit
    # vector. The r_norm scalar stays in fp32 since QJL reconstruction is
    # unbiased only on the unit sphere.
    rk_dq = tl.load(codebook_ptr + idx).to(tl.float32)
    residual = rotated - rk_dq
    r_masked = tl.where(mask_d, residual, 0.0)
    r_norm_sq = tl.sum(r_masked * r_masked)
    r_norm = tl.sqrt(tl.maximum(r_norm_sq, 1e-12))
    r_unit = r_masked / r_norm

    S_tile = tl.load(qjl_matrix_ptr + mat_off, mask=mat_mask, other=0.0).to(tl.float32)
    qjl_raw = tl.sum(S_tile * r_unit[None, :], axis=1)
    qjl_sign = tl.where(qjl_raw >= 0.0, 1.0, -1.0)

    tl.store(cache_k_rnorm_ptr + meta_off, r_norm)
    tl.store(cache_k_qjl_sign_ptr + cache_off, qjl_sign.to(tl.int8), mask=mask_d)

    # --- V quantization (Step 1, unchanged) ---------------------------------
    v_abs = tl.where(mask_d, tl.abs(v_vec), 0.0)
    v_max = tl.max(v_abs, axis=0)
    v_scale = tl.maximum(v_max / 127.0, 1e-12)
    v_int = (v_vec / v_scale).to(tl.int8)
    tl.store(cache_v_scale_ptr + meta_off, v_scale)
    tl.store(cache_v_ptr + cache_off, v_int, mask=mask_d)


@triton.jit
def _dequant_and_attend_kernel(
    q_rotated_ptr,              # (num_query_tokens, num_heads_q, head_size) fp16/bf16 -- H @ (signs * q)
    Sq_ptr,                     # (num_query_tokens, num_heads_q, head_size) fp16/bf16 -- S @ q_rotated
    cache_k_ptr,                # (num_blocks, block_size, num_heads_kv, head_size) uint8
    cache_v_ptr,                # (num_blocks, block_size, num_heads_kv, head_size) int8
    cache_k_norm_ptr,           # (num_blocks, block_size, num_heads_kv) fp32
    cache_v_scale_ptr,          # (num_blocks, block_size, num_heads_kv) fp32
    cache_k_qjl_sign_ptr,       # (num_blocks, block_size, num_heads_kv, head_size) int8
    cache_k_rnorm_ptr,          # (num_blocks, block_size, num_heads_kv) fp32
    block_table_ptr,            # (num_seqs, block_table_stride) int32
    seq_id_per_query_ptr,       # (num_query_tokens,) int32
    kv_end_per_query_ptr,       # (num_query_tokens,) int32
    codebook_ptr,               # (K_CB,) fp32
    out_ptr,                    # (num_query_tokens, num_heads_q, head_size) fp16/bf16
    inv_d,                      # fp32 scalar, 1/head_size
    qjl_coef,                   # fp32 scalar, sqrt(pi/2)/head_size
    max_blocks_per_seq,         # runtime int
    block_table_stride,         # runtime int
    OUT_DTYPE: tl.constexpr,
    num_heads_q: tl.constexpr,
    num_heads_kv: tl.constexpr,
    head_size: tl.constexpr,
    block_size: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_BS: tl.constexpr,
):
    """One program = one (query_token, q_head).

    Pure-rotated-space attend, following TurboQuant Algorithm 2.

    Pre-rotated inputs (prepared by the Python wrapper once per query head):
        q_rotated = H @ (signs * q)           -- same rotation as Pi on K
        Sq        = S @ q_rotated             -- the iid-Gaussian QJL rotation

    Because the rotation is orthogonal, inner products are preserved:
        <q, k> / sqrt(d)  =  <q_rotated, k_rotated> * k_norm / d
    where k_rotated is the pre-quantization rotated vector stored per key.
    We reconstruct it as
        k_rotated  ≈  codebook[idx] + r_norm * sqrt(pi/2)/d * S^T @ qjl_sign
    and split the inner product into two cheap dot products:
        <q_rotated, codebook[idx]>                          (BLOCK_D muls)
        <Sq, qjl_sign>  (qjl_sign is ±1, signed sum)        (BLOCK_D add/sub)

    This avoids the d×d un-rotation Hadamard multiply we used to do per KV
    token in the earlier kernel, so the QJL addition is actually faster
    than the original Algorithm-1 implementation.

    Varlen semantics (seq_id_per_query / kv_end_per_query / block_table_stride)
    are unchanged from the previous revision.
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
    q_rot = tl.load(q_rotated_ptr + q_off, mask=mask_d, other=0.0).to(tl.float32)
    sq = tl.load(Sq_ptr + q_off, mask=mask_d, other=0.0).to(tl.float32)

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
                    norm_off = (
                        phys_block * block_size * num_heads_kv
                        + tok_in_block * num_heads_kv
                        + kvh_idx
                    )

                    k_idx_u8 = tl.load(
                        cache_k_ptr + base + d_idx, mask=mask_d, other=0
                    ).to(tl.int32)
                    rk_main = tl.load(codebook_ptr + k_idx_u8).to(tl.float32)
                    qjl_sign = tl.load(
                        cache_k_qjl_sign_ptr + base + d_idx,
                        mask=mask_d, other=0,
                    ).to(tl.float32)

                    main_dot = tl.sum(q_rot * rk_main)
                    qjl_dot = tl.sum(sq * qjl_sign)
                    r_norm = tl.load(cache_k_rnorm_ptr + norm_off)
                    k_norm = tl.load(cache_k_norm_ptr + norm_off)

                    # <q, k> / sqrt(d) = <q_rotated, rotated_approx> * k_norm / d
                    logit = (main_dot + qjl_coef * r_norm * qjl_dot) * k_norm * inv_d

                    # V dequant (Step 1): int8 per-(slot, head) * per-(slot, head) fp32 scale.
                    v_int8 = tl.load(
                        cache_v_ptr + base + d_idx, mask=mask_d, other=0
                    ).to(tl.float32)
                    v_scale = tl.load(cache_v_scale_ptr + norm_off)
                    v_vec = v_int8 * v_scale

                    m_new = tl.maximum(m_i, logit)
                    alpha = tl.exp(m_i - m_new)
                    beta = tl.exp(logit - m_new)
                    l_i = l_i * alpha + beta
                    acc = acc * alpha + beta * v_vec
                    m_i = m_new

    out_vec = acc / tl.maximum(l_i, 1e-12)
    out_off = (q_idx * num_heads_q + qh_idx) * head_size + d_idx
    tl.store(out_ptr + out_off, out_vec.to(OUT_DTYPE), mask=mask_d)


_TQ_KSTATS_COUNTER = [0]


def _tq_kstats_dump(
    new_k: torch.Tensor,
    codebook: GaussianCodebook,
) -> None:
    """Print per-dim outlier statistics for the incoming K tensor.

    Emits three per-dim signals, averaged over (token, kv_head):
      1. raw_max    -- max(|K|) per dim: identifies outlier channels in the
                       original (pre-rotation, pre-norm) K.
      2. rot_max    -- max(|H @ (k_normed * signs)|) per dim: if Hadamard
                       equalizes as the paper assumes, this should be flat.
      3. resid_abs  -- |rotated - codebook[idx]| per element: if one dim
                       dominates, the per-head mean(|r|) scale is bogus.

    The ``outlier_ratio`` is top1_dim_max / mean_dim_max. For healthy
    N(0, 1) data (what the smoke test uses) it should be ~3x. For a real
    LLM K tensor with an outlier channel it is typically 10x-100x. A high
    ratio AFTER rotation is the smoking gun for Step 2's failure mode.
    """
    with torch.no_grad():
        k_f = new_k.detach().float()                        # (T, H, D)
        head_size = k_f.shape[-1]
        signs = codebook.signs.float()
        H = codebook.H.float()
        cb = codebook.codebook.float()
        boundaries = codebook.boundaries.float()

        # 1) raw K per-dim outliers
        raw_abs = k_f.abs()
        raw_dim_max = raw_abs.amax(dim=(0, 1))              # (D,)
        raw_dim_mean = raw_abs.mean(dim=(0, 1))             # (D,)
        raw_top = float(raw_dim_max.max().item())
        raw_avg = float(raw_dim_max.mean().item())

        # 2) rotated K per-dim outliers (matches what store kernel sees)
        k_norm = k_f.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        k_scaled = k_f * ((head_size ** 0.5) / k_norm)
        rot = (k_scaled * signs) @ H.T                      # (T, H, D)
        rot_abs = rot.abs()
        rot_dim_max = rot_abs.amax(dim=(0, 1))
        rot_top = float(rot_dim_max.max().item())
        rot_avg = float(rot_dim_max.mean().item())

        # 3) residual distribution after Lloyd-Max quant
        idx = torch.bucketize(rot.contiguous(), boundaries)
        rk_dq = cb[idx]
        resid = (rot - rk_dq).abs()
        r_max = float(resid.max().item())
        r_mean = float(resid.mean().item())
        r_p99 = float(resid.reshape(-1).quantile(0.99).item())

        # top 3 outlier dims by raw max -- useful for locating channels
        top_dims = raw_dim_max.topk(3).indices.tolist()
        top_vals = raw_dim_max.topk(3).values.tolist()

        print(
            f"[TQ_KSTATS #{_TQ_KSTATS_COUNTER[0]}] "
            f"T={k_f.shape[0]} H={k_f.shape[1]} D={head_size} | "
            f"raw: top_max={raw_top:.3f} avg_max={raw_avg:.3f} "
            f"ratio={raw_top / max(raw_avg, 1e-6):.1f}x "
            f"top3_dims={top_dims} top3_vals={[f'{v:.2f}' for v in top_vals]} | "
            f"rot: top_max={rot_top:.3f} avg_max={rot_avg:.3f} "
            f"ratio={rot_top / max(rot_avg, 1e-6):.1f}x | "
            f"resid: max={r_max:.4f} mean={r_mean:.4f} p99={r_p99:.4f} "
            f"max/mean={r_max / max(r_mean, 1e-6):.1f}x",
            flush=True,
        )


def turboquant_store_kv(
    new_k: torch.Tensor,
    new_v: torch.Tensor,
    cache_k: torch.Tensor,
    cache_v: torch.Tensor,
    cache_k_norm: torch.Tensor,
    cache_v_scale: torch.Tensor,
    cache_k_qjl_sign: torch.Tensor,
    cache_k_rnorm: torch.Tensor,
    slot_mapping: torch.Tensor,
    codebook: GaussianCodebook,
    block_size: int,
) -> None:
    """Store K via TurboQuant Algorithm 2 (paper §2.2) and V via int8.

    Algorithm 2 stores per (slot, head):
      - ``cache_k``: (b-1)-bit MSE index  (uint8)
      - ``cache_k_norm``: fp32 ||k|| (stage 0 norm)
      - ``cache_k_qjl_sign``: int8 ±1 per dim, sign(S @ r_unit)  (stage 4)
      - ``cache_k_rnorm``: fp32 ||residual||
    V keeps the per-(slot, head) symmetric int8 from Step 1.
    """
    import os
    if int(os.environ.get("TQ_KSTATS", "0")) and _TQ_KSTATS_COUNTER[0] < 8:
        _TQ_KSTATS_COUNTER[0] += 1
        _tq_kstats_dump(new_k, codebook)

    num_tokens, num_heads_kv, head_size = new_k.shape
    K_CB = int(codebook.codebook.shape[0])

    boundaries = codebook.boundaries
    if boundaries.dtype != torch.float32:
        boundaries = boundaries.to(torch.float32)

    codebook_f32 = codebook.codebook
    if codebook_f32.dtype != torch.float32:
        codebook_f32 = codebook_f32.to(torch.float32)

    grid = (num_tokens, num_heads_kv)
    BLOCK_D = triton.next_power_of_2(head_size)

    _quantize_and_store_kernel[grid](
        new_k,
        new_v,
        cache_k,
        cache_v,
        cache_k_norm,
        cache_v_scale,
        cache_k_qjl_sign,
        cache_k_rnorm,
        slot_mapping,
        codebook_f32,
        codebook.H,
        codebook.S,
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
    cache_k_qjl_sign: torch.Tensor,
    cache_k_rnorm: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    query_start_loc: torch.Tensor,
    codebook: GaussianCodebook,
    scale: float | None = None,
) -> torch.Tensor:
    """Paged attention on a TurboQuant-Algorithm-2 K cache.

    Compared to the earlier implementation this version does attention
    entirely in the rotated space, so the kernel never has to un-rotate K.
    We pay a one-shot pre-rotation of Q on the host-side (torch mm) in
    exchange for killing a d×d Hadamard multiply per KV token.

    q shape: (num_query_tokens, num_heads_q, head_size)
    Varlen metadata: same seq_lens / query_start_loc contract as before.
    """
    import os
    _tq_debug = int(os.environ.get("TQ_DEBUG", "0"))

    num_query_tokens, num_heads_q, head_size = q.shape
    _, block_size, num_heads_kv, _ = cache_k.shape
    num_seqs = int(seq_lens.shape[0])

    inv_d = 1.0 / float(head_size)
    import math as _math
    qjl_coef = _math.sqrt(_math.pi / 2.0) / float(head_size)

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

    # --- per-query metadata (on-device, vectorised) ----------------------
    dev = q.device
    qsl = query_start_loc.to(device=dev, dtype=torch.int64)
    query_lens = qsl[1:] - qsl[:-1]
    seq_ids = torch.arange(num_seqs, dtype=torch.int64, device=dev)
    seq_id_per_query_i64 = torch.repeat_interleave(seq_ids, query_lens)
    q_pos_per_query_i64 = (
        torch.arange(num_query_tokens, dtype=torch.int64, device=dev)
        - qsl[:-1][seq_id_per_query_i64]
    )
    prefix_len_per_seq_i64 = seq_lens.to(device=dev, dtype=torch.int64) - query_lens
    kv_end_per_query_i64 = (
        prefix_len_per_seq_i64[seq_id_per_query_i64] + q_pos_per_query_i64 + 1
    )

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

    # --- Pre-rotate Q on the host side -----------------------------------
    # q_rotated = H @ (signs * q).  In torch this is (q * signs) @ H.T.
    # Sq        = S @ q_rotated.   One fp32 mm per query-head block; the
    # cost is amortised over ALL KV tokens we will compare against.
    H_f = codebook.H.to(torch.float32)
    S_f = codebook.S.to(torch.float32)
    signs_f = codebook.signs.to(torch.float32)
    q_f = q.float()
    q_rotated = (q_f * signs_f) @ H_f.T            # (T, H_q, d)
    Sq = q_rotated @ S_f.T                         # (T, H_q, d)

    # Kernel expects the same dtype as q for q_rotated / Sq, to keep the
    # load dtypes matched.  (The kernel casts to fp32 internally.)
    q_rotated_k = q_rotated.to(q.dtype).contiguous()
    Sq_k = Sq.to(q.dtype).contiguous()

    out = torch.empty_like(q)
    grid = (num_query_tokens, num_heads_q)
    BLOCK_D = triton.next_power_of_2(head_size)
    BLOCK_BS = block_size
    OUT_DTYPE = tl.bfloat16 if q.dtype == torch.bfloat16 else tl.float16

    actual_max_seq = int(seq_lens.max().item())
    actual_max_blocks = (actual_max_seq + block_size - 1) // block_size
    block_table_stride = int(block_table.shape[1])

    _dequant_and_attend_kernel[grid](
        q_rotated_k,
        Sq_k,
        cache_k,
        cache_v,
        cache_k_norm,
        cache_v_scale,
        cache_k_qjl_sign,
        cache_k_rnorm,
        block_table,
        seq_id_per_query,
        kv_end_per_query,
        codebook_f32,
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
    )

    # --- optional PyTorch verification (TQ_VERIFY=1) ---------------------
    # Reconstruct K the same way the kernel does and run plain softmax
    # attention; the diff isolates Triton arithmetic from algorithmic bugs.
    if int(os.environ.get("TQ_VERIFY", "0")) and num_query_tokens <= 128:
        with torch.no_grad():
            cb_ref = codebook.codebook.to(torch.float32)
            H_ref = H_f
            S_ref = S_f
            signs_ref = signs_f
            max_seq_for_ref = int(seq_lens.max().item())
            k_ref = torch.zeros(
                max_seq_for_ref, num_heads_kv, head_size,
                dtype=torch.float32, device=q.device,
            )
            v_ref = torch.zeros_like(k_ref)
            qjl_scale_const = _math.sqrt(_math.pi / 2.0) / float(head_size)
            inv_sqrt_d = 1.0 / (head_size ** 0.5)
            for pos in range(max_seq_for_ref):
                bi = pos // block_size
                toff = pos % block_size
                phys = int(block_table[0, bi].item())
                for h in range(num_heads_kv):
                    idx = cache_k[phys, toff, h].long()
                    rk_main = cb_ref[idx]
                    qjl_sign = cache_k_qjl_sign[phys, toff, h].float()
                    r_norm_h = float(cache_k_rnorm[phys, toff, h].item())
                    # Paper: r_unit_approx = sqrt(pi/2)/d * S^T @ qjl_sign
                    r_unit_approx = qjl_scale_const * (S_ref.T @ qjl_sign)
                    rotated_approx = rk_main + r_norm_h * r_unit_approx
                    k_unrot = rotated_approx @ H_ref
                    k_unit = k_unrot * signs_ref
                    k_norm_h = float(cache_k_norm[phys, toff, h].item())
                    k_ref[pos, h] = k_unit * (k_norm_h * inv_sqrt_d)
                    v_scale_h = float(cache_v_scale[phys, toff, h].item())
                    v_ref[pos, h] = cache_v[phys, toff, h].float() * v_scale_h

            gqa = num_heads_q // num_heads_kv
            ref_out = torch.zeros_like(q, dtype=torch.float32)
            sqrt_scale = 1.0 / (head_size ** 0.5)
            for qi in range(num_query_tokens):
                kv_end_i = int(kv_end_per_query[qi].item())
                for h in range(num_heads_q):
                    kh = h // gqa
                    qv = q[qi, h].float()
                    kk = k_ref[:kv_end_i, kh]
                    vv = v_ref[:kv_end_i, kh]
                    sc = (qv @ kk.T) * sqrt_scale
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
