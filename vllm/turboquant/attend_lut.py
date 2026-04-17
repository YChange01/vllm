# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Flash-decoding + LUT-gather Triton attend kernel for TurboQuant.

Two-kernel pipeline:

  1. ``_flash_lut_attend_kernel``  grid = (T_q, H_q, K_SPLIT)
     - Each program owns one (query, head) and one K-split chunk of the
       KV sequence. Per-program register LUT
           LUT[j, c] = q_rot[j] * codebook[c]      (BLOCK_D x K_CB fp32)
       is built once and reused for all slots in the chunk. Per slot,
       ``main_dot = sum_j LUT[j, k_idx[j]]`` is computed via mask-sum
       gather over the register LUT. V is dequantized per slot via the
       small (K_CB <= 16) shared codebook.
     - Writes partial (m_i, l_i, acc) for this split to scratch.

  2. ``_combine_kernel``  grid = (T_q, H_q)
     - Online-softmax-merges the K_SPLIT partials into the final
       softmax-weighted V sum in rotated space. Single pass.

Motivations:

  * Occupancy. Base kernel grid was (T_q, H_q). At decode T_q=1 and
    H_q=32 that is 32 programs -- B200 has 108 SMs, so occupancy
    capped at ~30%. Splitting the KV dim gives (1, 32, K_SPLIT) =
    32 * K_SPLIT programs. For K_SPLIT >= 4 we saturate SMs.

  * LUT. One Python matmul (q * signs @ H.T) per layer is kept; the
    kernel's per-slot K dequant becomes a register-LUT gather. V
    dequant is per-slot codebook lookup -- V is reconstructed as a
    full d-dim vector, so a per-(j,c) LUT gives no benefit.

  * Autotune. Tune (num_warps, num_stages) at JIT time keyed by the
    head-size / K_CB / K_SPLIT / USE_QJL quadruple.

Scope: b=4 only (K_CB <= 16). K/V idx caches are always 4-bit
nibble-packed (two indices per uint8). QJL sign cache is 1-bit packed
(8 signs per byte). Post-rotation of the V accumulator back to the
original space is done in Python on the combine kernel's output --
not fused here, to keep the combine kernel simple.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from vllm.triton_utils import tl, triton

if TYPE_CHECKING:
    from vllm.turboquant.codebook import QuantState


_NEG_LARGE = tl.constexpr(-1.0e30)


# ---------------------------------------------------------------------------
# K_SPLIT policy
# ---------------------------------------------------------------------------
# Pick K_SPLIT based on max seq len so that each split processes roughly
# ~_KV_CHUNK_TARGET slots. Quantized to a small set of power-of-two values
# to keep the Triton JIT cache small. Must cover decode (T_q=1, need high
# K_SPLIT) and prefill (T_q large, K_SPLIT=1 is fine because the (T_q, H_q)
# grid already has plenty of programs).
_KV_CHUNK_TARGET = 256
_K_SPLIT_CHOICES = (1, 4, 8, 16, 32, 64)


def _pick_k_split(max_seq_len: int, num_query_tokens: int,
                  num_heads_q: int) -> int:
    """Choose K_SPLIT balancing per-program work against grid size."""
    # Prefill: grid (T_q, H_q) already large, don't split.
    if num_query_tokens > 16:
        return 1
    # Decode: want K_SPLIT * H_q >= ~256 programs to saturate SMs.
    want = max(
        1,
        (max_seq_len + _KV_CHUNK_TARGET - 1) // _KV_CHUNK_TARGET,
    )
    want = max(want, 256 // max(num_heads_q, 1))
    for choice in _K_SPLIT_CHOICES:
        if choice >= want:
            return choice
    return _K_SPLIT_CHOICES[-1]


# ---------------------------------------------------------------------------
# Autotuned attend kernel
# ---------------------------------------------------------------------------
_AUTOTUNE_CONFIGS = [
    triton.Config({}, num_warps=2, num_stages=2),
    triton.Config({}, num_warps=4, num_stages=2),
    triton.Config({}, num_warps=4, num_stages=3),
    triton.Config({}, num_warps=8, num_stages=2),
    triton.Config({}, num_warps=8, num_stages=3),
]


@triton.autotune(
    configs=_AUTOTUNE_CONFIGS,
    key=["head_size", "K_CB", "K_SPLIT", "USE_QJL"],
)
@triton.jit
def _flash_lut_attend_kernel(
    q_rotated_ptr,                  # (T_q, H_q, d) fp16/bf16
    Sq_ptr,                         # (T_q, H_q, d) fp16/bf16  prod only
    cache_k_idx_ptr,                # (num_blocks, bs, H_kv, d/2) uint8
    cache_k_norm_ptr,               # (num_blocks, bs, H_kv)     fp32
    cache_v_idx_ptr,                # (num_blocks, bs, H_kv, d/2) uint8
    cache_v_norm_ptr,               # (num_blocks, bs, H_kv)     fp32
    cache_k_qjl_sign_ptr,           # (num_blocks, bs, H_kv, d/8) uint8 prod
    cache_k_rnorm_ptr,              # (num_blocks, bs, H_kv)     fp32  prod
    block_table_ptr,                # (num_seqs, stride) int32
    seq_id_per_query_ptr,           # (T_q,) int32
    kv_end_per_query_ptr,           # (T_q,) int32  one-past-last kv pos
    codebook_ptr,                   # (K_CB,) fp32
    partial_m_ptr,                  # (T_q, H_q, K_SPLIT) fp32
    partial_l_ptr,                  # (T_q, H_q, K_SPLIT) fp32
    partial_acc_ptr,                # (T_q, H_q, K_SPLIT, d) fp32
    inv_d,
    qjl_coef,
    block_table_stride,
    num_heads_q: tl.constexpr,
    num_heads_kv: tl.constexpr,
    head_size: tl.constexpr,
    idx_dim: tl.constexpr,
    qjl_dim: tl.constexpr,
    block_size: tl.constexpr,
    BLOCK_D: tl.constexpr,
    K_CB: tl.constexpr,
    USE_QJL: tl.constexpr,
    K_SPLIT: tl.constexpr,
):
    q_idx = tl.program_id(0)
    qh_idx = tl.program_id(1)
    split_idx = tl.program_id(2)

    gqa_group = num_heads_q // num_heads_kv
    kvh_idx = qh_idx // gqa_group

    seq_idx = tl.load(seq_id_per_query_ptr + q_idx)
    kv_end = tl.load(kv_end_per_query_ptr + q_idx)

    # Per-split KV range (ceil-div so the last split handles the tail).
    kv_per_split = (kv_end + K_SPLIT - 1) // K_SPLIT
    kv_lo = split_idx * kv_per_split
    kv_hi = tl.minimum((split_idx + 1) * kv_per_split, kv_end)

    d_idx = tl.arange(0, BLOCK_D)
    c_idx = tl.arange(0, K_CB)
    mask_d = d_idx < head_size

    # Output offsets for partials.
    partial_scalar_off = (q_idx * num_heads_q + qh_idx) * K_SPLIT + split_idx
    partial_acc_off = (
        ((q_idx * num_heads_q + qh_idx) * K_SPLIT + split_idx) * head_size
        + d_idx
    )

    # Load Q rotated (and Sq for prod) into registers.
    q_off = (q_idx * num_heads_q + qh_idx) * head_size + d_idx
    q_rot = tl.load(q_rotated_ptr + q_off, mask=mask_d, other=0.0).to(tl.float32)
    if USE_QJL:
        sq = tl.load(Sq_ptr + q_off, mask=mask_d, other=0.0).to(tl.float32)

    # Small codebook (K_CB <= 16 scalars). Load once per program.
    codebook_vec = tl.load(codebook_ptr + c_idx).to(tl.float32)

    # Per-(query, head) LUT in registers:
    #     LUT[j, c] = q_rot[j] * codebook[c]
    # For head_size=128, K_CB=16 -> (128, 16) = 2048 fp32 = 8 KB. Compiler
    # decides between registers and shared memory.
    LUT = q_rot[:, None] * codebook_vec[None, :]

    m_i = tl.full((), _NEG_LARGE, dtype=tl.float32)
    l_i = tl.zeros((), dtype=tl.float32)
    acc = tl.zeros((BLOCK_D,), dtype=tl.float32)

    # 4-bit nibble-pack layout (this branch is b=4 only).
    d_pack = d_idx // 2
    is_high = (d_idx % 2) == 1

    block_lo = kv_lo // block_size
    block_hi = (kv_hi + block_size - 1) // block_size

    for block_i in range(block_lo, block_hi):
        phys_block = tl.load(
            block_table_ptr + seq_idx * block_table_stride + block_i
        )
        for tok_in_block in tl.static_range(0, block_size):
            abs_pos = block_i * block_size + tok_in_block
            if abs_pos >= kv_lo and abs_pos < kv_hi:
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

                # --- K dequant + main_dot via LUT gather ---
                packed_k = tl.load(
                    cache_k_idx_ptr + base_idx + d_pack,
                    mask=mask_d, other=0,
                ).to(tl.uint8)
                k_low = packed_k & 0xF
                k_high = (packed_k >> 4) & 0xF
                k_idx_u8 = tl.where(is_high, k_high, k_low).to(tl.int32)

                one_hot = (k_idx_u8[:, None] == c_idx[None, :])
                main_dot = tl.sum(tl.where(one_hot, LUT, 0.0))

                k_norm = tl.load(cache_k_norm_ptr + meta)

                if USE_QJL:
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

                # --- V dequant (per-slot, shared small codebook) ---
                packed_v = tl.load(
                    cache_v_idx_ptr + base_idx + d_pack,
                    mask=mask_d, other=0,
                ).to(tl.uint8)
                v_low = packed_v & 0xF
                v_high = (packed_v >> 4) & 0xF
                v_idx_u8 = tl.where(is_high, v_high, v_low).to(tl.int32)

                rv = tl.load(codebook_ptr + v_idx_u8).to(tl.float32)
                v_norm = tl.load(cache_v_norm_ptr + meta)
                v_vec = rv * v_norm

                # --- Flash softmax online accumulation ---
                m_new = tl.maximum(m_i, logit)
                alpha = tl.exp(m_i - m_new)
                beta = tl.exp(logit - m_new)
                l_i = l_i * alpha + beta
                acc = acc * alpha + beta * v_vec
                m_i = m_new

    tl.store(partial_m_ptr + partial_scalar_off, m_i)
    tl.store(partial_l_ptr + partial_scalar_off, l_i)
    tl.store(partial_acc_ptr + partial_acc_off, acc, mask=mask_d)


# ---------------------------------------------------------------------------
# Combine kernel: merge K_SPLIT partials into final output (rotated space)
# ---------------------------------------------------------------------------
@triton.jit
def _combine_kernel(
    partial_m_ptr,                  # (T_q, H_q, K_SPLIT) fp32
    partial_l_ptr,                  # (T_q, H_q, K_SPLIT) fp32
    partial_acc_ptr,                # (T_q, H_q, K_SPLIT, d) fp32
    out_ptr,                        # (T_q, H_q, d) fp16/bf16 rotated space
    num_heads_q: tl.constexpr,
    head_size: tl.constexpr,
    BLOCK_D: tl.constexpr,
    K_SPLIT: tl.constexpr,
    OUT_DTYPE: tl.constexpr,
):
    q_idx = tl.program_id(0)
    qh_idx = tl.program_id(1)

    d_idx = tl.arange(0, BLOCK_D)
    mask_d = d_idx < head_size

    base_scalar = (q_idx * num_heads_q + qh_idx) * K_SPLIT
    base_acc = (q_idx * num_heads_q + qh_idx) * K_SPLIT * head_size

    m_final = tl.full((), _NEG_LARGE, dtype=tl.float32)
    l_final = tl.zeros((), dtype=tl.float32)
    acc_final = tl.zeros((BLOCK_D,), dtype=tl.float32)

    for split_idx in tl.static_range(0, K_SPLIT):
        m = tl.load(partial_m_ptr + base_scalar + split_idx)
        l = tl.load(partial_l_ptr + base_scalar + split_idx)
        a = tl.load(
            partial_acc_ptr + base_acc + split_idx * head_size + d_idx,
            mask=mask_d, other=0.0,
        )

        m_new = tl.maximum(m_final, m)
        alpha = tl.exp(m_final - m_new)
        beta = tl.exp(m - m_new)
        l_final = l_final * alpha + beta * l
        acc_final = acc_final * alpha + beta * a
        m_final = m_new

    out_vec = acc_final / tl.maximum(l_final, 1e-12)
    out_off = (q_idx * num_heads_q + qh_idx) * head_size + d_idx
    tl.store(out_ptr + out_off, out_vec.to(OUT_DTYPE), mask=mask_d)


def turboquant_paged_attention_lut(
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
    """Flash-decoding + LUT paged attention.

    Same interface as ``turboquant_paged_attention``. Decides K_SPLIT
    from ``seq_lens.max()`` and T_q, launches the split attend kernel
    plus the combine kernel, then Python-side post-rotates V.
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
    assert K_CB <= 16, (
        f"LUT kernel is b=4 only (K_CB <= 16); got K_CB={K_CB}"
    )
    assert idx_dim == head_size // 2, (
        f"cache_k_idx last dim {idx_dim} != head_size/2; "
        f"LUT kernel requires 4-bit nibble pack"
    )

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

    # Pre-rotate Q (and Sq for prod) in Python. Hadamard is symmetric
    # so H.T == H; keeping the .T for explicitness in code reading.
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
        Sq_k = q_rotated_k

    qjl_dim = head_size // 8 if use_qjl else 1
    if use_qjl:
        qjl_sign_buf = cache_k_qjl_sign
        assert qjl_sign_buf.dtype == torch.uint8
        assert qjl_sign_buf.shape[-1] == qjl_dim
    else:
        qjl_sign_buf = cache_k_idx
    rnorm_buf = cache_k_rnorm if use_qjl else cache_k_norm

    max_seq_len = int(seq_lens.max().item())
    K_SPLIT = _pick_k_split(max_seq_len, num_query_tokens, num_heads_q)

    # Scratch buffers for partials.
    partial_shape_scalar = (num_query_tokens, num_heads_q, K_SPLIT)
    partial_shape_acc = (num_query_tokens, num_heads_q, K_SPLIT, head_size)
    partial_m = torch.empty(partial_shape_scalar, dtype=torch.float32, device=dev)
    partial_l = torch.empty(partial_shape_scalar, dtype=torch.float32, device=dev)
    partial_acc = torch.empty(partial_shape_acc, dtype=torch.float32, device=dev)

    out = torch.empty_like(q)
    BLOCK_D = triton.next_power_of_2(head_size)
    OUT_DTYPE = tl.bfloat16 if q.dtype == torch.bfloat16 else tl.float16

    inv_d = 1.0 / float(head_size)
    qjl_coef = math.sqrt(math.pi / 2.0) / float(head_size)
    block_table_stride = int(block_table.shape[1])

    # 1) Attend kernel — fills partials.
    grid_attend = (num_query_tokens, num_heads_q, K_SPLIT)
    _flash_lut_attend_kernel[grid_attend](
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
        partial_m,
        partial_l,
        partial_acc,
        inv_d,
        qjl_coef,
        block_table_stride,
        num_heads_q=num_heads_q,
        num_heads_kv=num_heads_kv,
        head_size=head_size,
        idx_dim=idx_dim,
        qjl_dim=qjl_dim,
        block_size=block_size,
        BLOCK_D=BLOCK_D,
        K_CB=K_CB,
        USE_QJL=use_qjl,
        K_SPLIT=K_SPLIT,
    )

    # 2) Combine kernel — online-softmax merge, writes `out` in rotated
    # V space.
    grid_combine = (num_query_tokens, num_heads_q)
    _combine_kernel[grid_combine](
        partial_m,
        partial_l,
        partial_acc,
        out,
        num_heads_q=num_heads_q,
        head_size=head_size,
        BLOCK_D=BLOCK_D,
        K_SPLIT=K_SPLIT,
        OUT_DTYPE=OUT_DTYPE,
    )

    # Post-rotate V back to original space.
    out_f = out.float()
    output_f = (out_f @ H_f.T) * signs_f / math.sqrt(float(head_size))
    return output_f.to(out.dtype)
