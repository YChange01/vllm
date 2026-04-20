# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tensor-core attend kernel for TurboQuant (paper-faithful).

Paper arXiv:2504.19874, Algorithm 2 applied uniformly to K and V in
the KV cache.

Data layout per stored (token, kv_head) slot:
    cache_k_idx        uint8 (d/2)       4-bit nibble-packed Lloyd-Max idx
    cache_k_norm       fp32            ||k||
    cache_k_qjl_sign   uint8 (d/8)       1-bit-packed sign(S @ r_unit)  (prod)
    cache_k_rnorm      fp32            ||r_k||                        (prod)
    same fields for V (cache_v_*).

Reconstruction, in rotated-unit-sphere space:
    y_approx = codebook[idx] + (sqrt(pi/2)/d) * ||r|| * S^T @ qjl_sign

For the Q @ K inner product we never need to materialize the QJL
contribution to K in full d-dim; we contract it against the
pre-projected Sq = Q_rot @ S^T as in the main-text formula:

    <q, y_k_approx> = <q, codebook[idx]>
                    + (sqrt(pi/2)/d) * ||r_k|| * <Sq, qjl_k>

For the weighted-sum output with V the paper requires the full
reconstruction. Doing the d x d matvec per slot inside the kernel is
too expensive, so we split the V accumulator into two halves --
``acc_main`` from ``codebook[v_idx]`` and ``acc_qjl`` from the 1-bit
signs -- and apply ``S^T`` once in Python post-kernel. The Python
post-pass also folds in the inverse rotation ``Pi^T``; both fit into
a single cuBLAS bf16 matmul.

Scope: b=4 only on this branch (K_CB <= 16, 4-bit nibble-packed idx,
1-bit-packed QJL sign). Stage 2 will add variable bit widths.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from vllm.triton_utils import tl, triton

if TYPE_CHECKING:
    from vllm.turboquant.codebook import QuantState


_NEG_LARGE = tl.constexpr(-1.0e30)
_BLOCK_M_MIN = 16


_AUTOTUNE_CONFIGS = [
    triton.Config({"BLOCK_N": 32}, num_warps=2, num_stages=2),
    triton.Config({"BLOCK_N": 32}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_N": 64}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_N": 64}, num_warps=4, num_stages=3),
    triton.Config({"BLOCK_N": 64}, num_warps=8, num_stages=2),
    triton.Config({"BLOCK_N": 128}, num_warps=8, num_stages=2),
    triton.Config({"BLOCK_N": 128}, num_warps=8, num_stages=3),
]


@triton.autotune(
    configs=_AUTOTUNE_CONFIGS,
    key=["head_size", "K_CB", "PACK_BITS", "USE_QJL", "USE_TIGHT",
         "USE_UINT8_RNORM", "BLOCK_M", "GQA_GROUP"],
)
@triton.jit
def _tc_attend_kernel(
    q_rotated_ptr,                  # (T_q, H_q, d) bf16/fp16
    Sq_ptr,                         # (T_q, H_q, d) bf16/fp16  prod only
    cache_k_idx_ptr,                # (num_blocks, bs, H_kv, d/2) uint8
    cache_k_norm_ptr,               # (num_blocks, bs, H_kv) fp32
    cache_v_idx_ptr,                # (num_blocks, bs, H_kv, d/2) uint8
    cache_v_norm_ptr,               # (num_blocks, bs, H_kv) fp32
    cache_k_qjl_sign_ptr,           # (num_blocks, bs, H_kv, d/8) uint8 prod
    cache_k_rnorm_ptr,              # (num_blocks, bs, H_kv) fp32  prod
    cache_v_qjl_sign_ptr,           # (num_blocks, bs, H_kv, d/8) uint8 prod
    cache_v_rnorm_ptr,              # (num_blocks, bs, H_kv) fp32  prod
    block_table_ptr,                # (num_seqs, stride) int32
    seq_id_per_query_ptr,           # (T_q,) int32
    kv_end_per_query_ptr,           # (T_q,) int32
    codebook_ptr,                   # (K_CB,) bf16 (same dtype as Q)
    out_main_ptr,                   # (T_q, H_q, d) bf16  main V accumulator
    out_qjl_ptr,                    # (T_q, H_q, d) bf16  QJL V accumulator (prod)
    inv_sqrt_d,
    qjl_coef,
    block_table_stride,
    OUT_DTYPE: tl.constexpr,
    COMPUTE_DTYPE: tl.constexpr,    # tl.bfloat16 or tl.float16
    num_heads_q: tl.constexpr,
    num_heads_kv: tl.constexpr,
    head_size: tl.constexpr,
    idx_dim: tl.constexpr,
    qjl_dim: tl.constexpr,
    block_size: tl.constexpr,
    BLOCK_M: tl.constexpr,          # >= 16 (tensor-core min), >= GQA_GROUP
    BLOCK_N: tl.constexpr,          # autotuned KV-tile
    BLOCK_D: tl.constexpr,          # next_power_of_2(head_size)
    K_CB: tl.constexpr,
    PACK_BITS: tl.constexpr,        # 1, 2, 4, or 8
    USE_QJL: tl.constexpr,
    USE_TIGHT: tl.constexpr,        # b=4 prod tight nibble (idx+qjl)
    USE_UINT8_RNORM: tl.constexpr,  # rnorm cache stored as uint8 in [0, RNORM_MAX]
    GQA_GROUP: tl.constexpr,
):
    q_idx = tl.program_id(0)
    kvh_idx = tl.program_id(1)
    q_head_start = kvh_idx * GQA_GROUP
    # Must match store-side scaling. Hardcoded to 2.0; ||r|| typically
    # < 1 for randn data, 2x headroom for outlier-amplified.
    RNORM_MAX: tl.constexpr = 2.0
    RNORM_DEQUANT: tl.constexpr = 2.0 / 255.0

    seq_idx = tl.load(seq_id_per_query_ptr + q_idx)
    kv_end = tl.load(kv_end_per_query_ptr + q_idx)

    m_off = tl.arange(0, BLOCK_M)
    n_off = tl.arange(0, BLOCK_N)
    d_off = tl.arange(0, BLOCK_D)
    mask_m = m_off < GQA_GROUP
    mask_d = d_off < head_size

    # Load Q tile (BLOCK_M, BLOCK_D) in compute dtype.
    q_base = q_idx * num_heads_q * head_size
    q_offs_2d = (
        q_base
        + (q_head_start + m_off[:, None]) * head_size
        + d_off[None, :]
    )
    q_rot = tl.load(
        q_rotated_ptr + q_offs_2d,
        mask=mask_m[:, None] & mask_d[None, :],
        other=0.0,
    )

    if USE_QJL:
        sq = tl.load(
            Sq_ptr + q_offs_2d,
            mask=mask_m[:, None] & mask_d[None, :],
            other=0.0,
        )

    m_i = tl.full((BLOCK_M,), _NEG_LARGE, dtype=tl.float32)
    l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)
    acc_main = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)
    if USE_QJL:
        acc_qjl = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)

    # Variable-bit unpack layout. PACK_BITS in {1, 2, 4, 8}.
    N_PER_BYTE: tl.constexpr = 8 // PACK_BITS
    UNPACK_MASK: tl.constexpr = (1 << PACK_BITS) - 1
    # In tight mode the high bit of each 4-bit nibble is the QJL sign;
    # the low 3 bits are the Lloyd-Max idx. Otherwise full nibble = idx.
    IDX_MASK: tl.constexpr = 7 if USE_TIGHT else UNPACK_MASK
    d_pack = d_off // N_PER_BYTE
    d_bit_shift = (d_off % N_PER_BYTE) * PACK_BITS

    num_tiles = (kv_end + BLOCK_N - 1) // BLOCK_N

    for tile_i in range(0, num_tiles):
        kv_start = tile_i * BLOCK_N
        n_pos = kv_start + n_off
        mask_n = n_pos < kv_end

        block_idx = n_pos // block_size
        tok_in_block = n_pos % block_size

        phys_blocks = tl.load(
            block_table_ptr + seq_idx * block_table_stride + block_idx,
            mask=mask_n, other=0,
        )

        base_idx = (
            phys_blocks * (block_size * num_heads_kv * idx_dim)
            + tok_in_block * (num_heads_kv * idx_dim)
            + kvh_idx * idx_dim
        )
        meta_addrs = (
            phys_blocks * (block_size * num_heads_kv)
            + tok_in_block * num_heads_kv
            + kvh_idx
        )

        # ------------------------------------------------------------------
        # K dequant tile: codebook[idx_k] (unit-sphere rotated), scaled by ||k||
        # ------------------------------------------------------------------
        addrs_k = base_idx[:, None] + d_pack[None, :]
        packed_k = tl.load(
            cache_k_idx_ptr + addrs_k,
            mask=mask_n[:, None] & mask_d[None, :],
            other=0,
        ).to(tl.uint8)
        # In tight mode the nibble = (qjl_bit << 3) | idx; we need both.
        nibble_k = (packed_k.to(tl.int32) >> d_bit_shift[None, :]) & UNPACK_MASK
        k_idx_full = nibble_k & IDX_MASK
        k_tile = tl.load(codebook_ptr + k_idx_full)

        # Promote to fp32 for downstream math; cache may be fp16 when
        # FP16_NORMS is enabled.
        k_norm = tl.load(
            cache_k_norm_ptr + meta_addrs, mask=mask_n, other=0.0
        ).to(tl.float32)
        # k_tile_scaled approximates ||k|| * (Pi @ k/||k||) = (Pi @ k);
        # the extra Pi^T in recovery cancels in <q_rot, k_tile_scaled>.
        k_tile_scaled = (
            k_tile.to(tl.float32) * k_norm[:, None]
        ).to(COMPUTE_DTYPE)

        # Q @ K^T (tensor core)
        qk = tl.dot(q_rot, tl.trans(k_tile_scaled))

        if USE_QJL:
            # QJL on K: unpack 1-bit signs into +/-1 bf16 tile.
            if USE_TIGHT:
                # Sign was packed at bit 3 of each idx nibble; reuse
                # nibble_k (already loaded above) instead of a second
                # buffer load.
                bit_k = (nibble_k >> 3) & 1
            else:
                bit_pos = d_off % 8
                byte_pos = d_off // 8
                base_qjl = (
                    phys_blocks * (block_size * num_heads_kv * qjl_dim)
                    + tok_in_block * (num_heads_kv * qjl_dim)
                    + kvh_idx * qjl_dim
                )
                addrs_qjl_k = base_qjl[:, None] + byte_pos[None, :]
                qjl_byte_k = tl.load(
                    cache_k_qjl_sign_ptr + addrs_qjl_k,
                    mask=mask_n[:, None] & mask_d[None, :],
                    other=0,
                ).to(tl.int32)
                bit_k = (qjl_byte_k >> bit_pos[None, :]) & 1
            qjl_sign_tile_k = (
                1.0 - 2.0 * bit_k.to(tl.float32)
            ).to(COMPUTE_DTYPE)

            r_norm_k = tl.load(
                cache_k_rnorm_ptr + meta_addrs, mask=mask_n, other=0.0
            ).to(tl.float32)
            if USE_UINT8_RNORM:
                r_norm_k = r_norm_k * RNORM_DEQUANT
            # Sq @ qjl_k^T via tensor core -> (BLOCK_M, BLOCK_N) fp32.
            qjl_dot_k = tl.dot(sq, tl.trans(qjl_sign_tile_k))

            # logit = (<q_rot, k_tile> + (sqrt(pi/2)/d) * r_norm_k * <Sq,qjl_k>)
            #         * k_norm ... but k_norm is already in k_tile_scaled, so:
            #         = (qk + qjl_coef * r_norm_k * qjl_dot_k * k_norm) / sqrt(d)
            logit = (
                qk
                + qjl_coef * r_norm_k[None, :] * qjl_dot_k * k_norm[None, :]
            ) * inv_sqrt_d
        else:
            logit = qk * inv_sqrt_d

        logit = tl.where(mask_n[None, :], logit, _NEG_LARGE)

        # Online softmax update.
        m_new = tl.maximum(m_i, tl.max(logit, axis=1))
        alpha = tl.exp(m_i - m_new)
        probs = tl.exp(logit - m_new[:, None])
        l_i = l_i * alpha + tl.sum(probs, axis=1)
        acc_main = acc_main * alpha[:, None]
        if USE_QJL:
            acc_qjl = acc_qjl * alpha[:, None]

        # ------------------------------------------------------------------
        # V dequant main: codebook[v_idx] * v_norm (tensor-core P @ V_main)
        # ------------------------------------------------------------------
        addrs_v = base_idx[:, None] + d_pack[None, :]
        packed_v = tl.load(
            cache_v_idx_ptr + addrs_v,
            mask=mask_n[:, None] & mask_d[None, :],
            other=0,
        ).to(tl.uint8)
        nibble_v = (packed_v.to(tl.int32) >> d_bit_shift[None, :]) & UNPACK_MASK
        v_idx_full = nibble_v & IDX_MASK
        v_tile = tl.load(codebook_ptr + v_idx_full)

        v_norm = tl.load(
            cache_v_norm_ptr + meta_addrs, mask=mask_n, other=0.0
        ).to(tl.float32)
        v_tile_scaled = (
            v_tile.to(tl.float32) * v_norm[:, None]
        ).to(COMPUTE_DTYPE)

        probs_cast = probs.to(COMPUTE_DTYPE)
        acc_main = acc_main + tl.dot(probs_cast, v_tile_scaled)

        if USE_QJL:
            # V QJL: 1-bit signs scaled by ||r_v|| * ||v||. The Python
            # post-pass applies Sv^T and then Pi^T to acc_qjl.
            if USE_TIGHT:
                bit_v = (nibble_v >> 3) & 1
            else:
                addrs_qjl_v = base_qjl[:, None] + byte_pos[None, :]
                qjl_byte_v = tl.load(
                    cache_v_qjl_sign_ptr + addrs_qjl_v,
                    mask=mask_n[:, None] & mask_d[None, :],
                    other=0,
                ).to(tl.int32)
                bit_v = (qjl_byte_v >> bit_pos[None, :]) & 1
            qjl_sign_tile_v = (
                1.0 - 2.0 * bit_v.to(tl.float32)
            ).to(COMPUTE_DTYPE)

            r_norm_v = tl.load(
                cache_v_rnorm_ptr + meta_addrs, mask=mask_n, other=0.0
            ).to(tl.float32)
            if USE_UINT8_RNORM:
                r_norm_v = r_norm_v * RNORM_DEQUANT
            v_qjl_scaled = (
                qjl_sign_tile_v.to(tl.float32)
                * r_norm_v[:, None]
                * v_norm[:, None]
            ).to(COMPUTE_DTYPE)
            acc_qjl = acc_qjl + tl.dot(probs_cast, v_qjl_scaled)

        m_i = m_new

    # Normalize and write out. The caller handles Pi^T and (for prod)
    # the Sv^T matmul on acc_qjl.
    inv_l = 1.0 / tl.maximum(l_i[:, None], 1e-12)
    out_offs = (
        q_base
        + (q_head_start + m_off[:, None]) * head_size
        + d_off[None, :]
    )
    tl.store(
        out_main_ptr + out_offs,
        (acc_main * inv_l).to(OUT_DTYPE),
        mask=mask_m[:, None] & mask_d[None, :],
    )
    if USE_QJL:
        tl.store(
            out_qjl_ptr + out_offs,
            (acc_qjl * inv_l).to(OUT_DTYPE),
            mask=mask_m[:, None] & mask_d[None, :],
        )


def turboquant_paged_attention_tc(
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
    cache_v_qjl_sign: torch.Tensor | None = None,
    cache_v_rnorm: torch.Tensor | None = None,
) -> torch.Tensor:
    num_query_tokens, num_heads_q, head_size = q.shape
    _, block_size, num_heads_kv, idx_dim = cache_k_idx.shape
    num_seqs = int(seq_lens.shape[0])
    assert query_start_loc.shape[0] == num_seqs + 1
    assert block_table.shape[0] >= num_seqs

    use_qjl = state.algo == "prod"
    use_tight = bool(getattr(state, "tight_pack", False))
    use_uint8_rnorm = (
        cache_k_rnorm is not None and cache_k_rnorm.dtype == torch.uint8
    )
    if use_qjl and not use_tight:
        assert cache_k_qjl_sign is not None and cache_k_rnorm is not None
        assert cache_v_qjl_sign is not None and cache_v_rnorm is not None
    if use_tight:
        # Tight: qjl bits are merged into idx nibbles; only need rnorm.
        assert cache_k_rnorm is not None and cache_v_rnorm is not None

    K_CB = int(state.codebook.shape[0])
    pack_bits = state.pack_bits
    expected_idx_dim = head_size * pack_bits // 8
    assert idx_dim == expected_idx_dim, (
        f"cache_k_idx last dim {idx_dim} != head_size*pack_bits/8 = "
        f"{expected_idx_dim} (head_size={head_size}, pack_bits={pack_bits})"
    )
    assert q.dtype in (torch.bfloat16, torch.float16), (
        f"TC kernel requires bf16 or fp16 queries for tensor-core dot; "
        f"got {q.dtype}"
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

    # Pre-rotate Q via Pi (bf16 cuBLAS). Pi is not symmetric -- always
    # multiply on the right.
    q_rotated = (
        q.reshape(-1, head_size) @ state.Pi
    ).view(num_query_tokens, num_heads_q, head_size).contiguous()

    if use_qjl:
        # Sq = q_rotated @ S^T matches <q_rot, S^T @ qjl> = <Sq, qjl>
        # used inside the kernel.
        Sq = (
            q_rotated.reshape(-1, head_size) @ state.S.t()
        ).view(num_query_tokens, num_heads_q, head_size).contiguous()
    else:
        Sq = q_rotated  # unused by the kernel when USE_QJL=False

    qjl_dim = head_size // 8 if (use_qjl and not use_tight) else 1
    if use_tight:
        # Kernel reads qjl from idx nibble; the qjl ptr is unused but
        # needs to be a valid same-dtype tensor for the signature.
        k_qjl_buf = cache_k_idx
        v_qjl_buf = cache_v_idx
        k_rnorm_buf = cache_k_rnorm
        v_rnorm_buf = cache_v_rnorm
    elif use_qjl:
        assert cache_k_qjl_sign.dtype == torch.uint8
        assert cache_k_qjl_sign.shape[-1] == qjl_dim
        assert cache_v_qjl_sign.shape[-1] == qjl_dim
        k_qjl_buf = cache_k_qjl_sign
        v_qjl_buf = cache_v_qjl_sign
        k_rnorm_buf = cache_k_rnorm
        v_rnorm_buf = cache_v_rnorm
    else:
        k_qjl_buf = cache_k_idx
        v_qjl_buf = cache_k_idx
        k_rnorm_buf = cache_k_norm
        v_rnorm_buf = cache_k_norm

    codebook_ct = state.codebook.to(q.dtype)

    gqa_group = num_heads_q // num_heads_kv
    BLOCK_M = max(gqa_group, _BLOCK_M_MIN)
    BLOCK_M = 1 << (BLOCK_M - 1).bit_length()
    BLOCK_D = triton.next_power_of_2(head_size)

    out_main = torch.empty_like(q)
    if use_qjl:
        out_qjl = torch.empty_like(q)
    else:
        out_qjl = out_main  # unused

    OUT_DTYPE = tl.bfloat16 if q.dtype == torch.bfloat16 else tl.float16
    COMPUTE_DTYPE = OUT_DTYPE

    inv_sqrt_d = 1.0 / math.sqrt(float(head_size))
    qjl_coef = math.sqrt(math.pi / 2.0) / float(head_size)
    block_table_stride = int(block_table.shape[1])

    grid = (num_query_tokens, num_heads_kv)
    _tc_attend_kernel[grid](
        q_rotated,
        Sq,
        cache_k_idx,
        cache_k_norm,
        cache_v_idx,
        cache_v_norm,
        k_qjl_buf,
        k_rnorm_buf,
        v_qjl_buf,
        v_rnorm_buf,
        block_table,
        seq_id_per_query,
        kv_end_per_query,
        codebook_ct,
        out_main,
        out_qjl,
        inv_sqrt_d,
        qjl_coef,
        block_table_stride,
        OUT_DTYPE=OUT_DTYPE,
        COMPUTE_DTYPE=COMPUTE_DTYPE,
        num_heads_q=num_heads_q,
        num_heads_kv=num_heads_kv,
        head_size=head_size,
        idx_dim=idx_dim,
        qjl_dim=qjl_dim,
        block_size=block_size,
        BLOCK_M=BLOCK_M,
        BLOCK_D=BLOCK_D,
        K_CB=K_CB,
        PACK_BITS=pack_bits,
        USE_QJL=use_qjl,
        USE_TIGHT=use_tight,
        USE_UINT8_RNORM=use_uint8_rnorm,
        GQA_GROUP=gqa_group,
    )

    # Post-process: un-rotate with Pi^T. For prod, also resolve the V
    # QJL residual via S^T before adding to the main output. Both
    # reductions are one bf16 matmul each via cuBLAS.
    if use_qjl:
        qjl_contribution = (
            out_qjl.reshape(-1, head_size) @ state.S
        ) * qjl_coef
        combined = out_main.reshape(-1, head_size) + qjl_contribution
        output = combined @ state.Pi_T
    else:
        output = out_main.reshape(-1, head_size) @ state.Pi_T
    return output.view(num_query_tokens, num_heads_q, head_size).contiguous()
