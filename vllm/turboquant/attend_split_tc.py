# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Split-slice Triton tensor-core attend kernel for TurboQuant.

Paper §4.3 applies two independent TurboQuant instances to a single
head: one to an outlier-channel slice (smaller d_out, higher bit
width) and one to the remaining regular channels (d_reg, lower bit
width). The attention inner product factorizes across slices:

    <q, k> = <q_out, k_out> + <q_reg, k_reg>

so a single softmax over the summed pre-scale QK produces the correct
attention weights. V is reconstructed per slice and scattered back to
the full head dimension by the Python wrapper.

Each kernel program handles one (query_token, kv_head) pair and
maintains four accumulators in prod mode:

    acc_out_main : codebook[v_out_idx] contribution, d_out wide
    acc_out_qjl  : 1-bit sign contribution for V outlier, d_out wide
    acc_reg_main : codebook[v_reg_idx] contribution, d_reg wide
    acc_reg_qjl  : 1-bit sign contribution for V regular, d_reg wide

Post-kernel Python resolves each slice's QJL via
``acc_qjl @ S_{slice}`` and un-rotates with ``Pi_T_{slice}``, then
scatters back to full head_dim.

Homogeneous (non-split) attention continues to use ``attend_tc.py``;
this file is only loaded when ``TURBOQUANT_OUTLIER_MASK`` is set.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from vllm.triton_utils import tl, triton

if TYPE_CHECKING:
    from vllm.turboquant.outlier import SplitQuantState


_NEG_LARGE = tl.constexpr(-1.0e30)
_BLOCK_M_MIN = 16


_AUTOTUNE_CONFIGS = [
    triton.Config({"BLOCK_N": 32}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_N": 64}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_N": 64}, num_warps=8, num_stages=2),
    triton.Config({"BLOCK_N": 128}, num_warps=8, num_stages=2),
]


# Kernel argument count is large but mechanical: each slice contributes
# four K buffers, four V buffers, a codebook pointer, and the slice-
# specific constexprs (D, BLOCK_D, IDX_DIM, QJL_DIM, PACK_BITS, K_CB).
@triton.autotune(
    configs=_AUTOTUNE_CONFIGS,
    key=[
        "D_OUT", "D_REG",
        "K_CB_OUT", "K_CB_REG",
        "PACK_BITS_OUT", "PACK_BITS_REG",
        "USE_QJL", "BLOCK_M", "GQA_GROUP",
    ],
)
@triton.jit
def _split_attend_kernel(
    # ---- Query, pre-rotated per slice (BLOCK_D padded) ----
    q_rot_out_ptr,                  # (T_q, H_q, D_OUT) bf16
    Sq_out_ptr,                     # (T_q, H_q, D_OUT) bf16  prod only
    q_rot_reg_ptr,                  # (T_q, H_q, D_REG) bf16
    Sq_reg_ptr,                     # (T_q, H_q, D_REG) bf16  prod only
    # ---- K outlier slice ----
    cache_k_idx_out_ptr,            # (nb, bs, H_kv, IDX_DIM_OUT) uint8
    cache_k_norm_out_ptr,           # (nb, bs, H_kv) fp32
    cache_k_qjl_sign_out_ptr,       # (nb, bs, H_kv, QJL_DIM_OUT) uint8 prod
    cache_k_rnorm_out_ptr,          # (nb, bs, H_kv) fp32                prod
    # ---- K regular slice ----
    cache_k_idx_reg_ptr,
    cache_k_norm_reg_ptr,
    cache_k_qjl_sign_reg_ptr,
    cache_k_rnorm_reg_ptr,
    # ---- V outlier slice ----
    cache_v_idx_out_ptr,
    cache_v_norm_out_ptr,
    cache_v_qjl_sign_out_ptr,
    cache_v_rnorm_out_ptr,
    # ---- V regular slice ----
    cache_v_idx_reg_ptr,
    cache_v_norm_reg_ptr,
    cache_v_qjl_sign_reg_ptr,
    cache_v_rnorm_reg_ptr,
    # ---- Shared metadata ----
    block_table_ptr,
    seq_id_per_query_ptr,
    kv_end_per_query_ptr,
    codebook_out_ptr,               # (K_CB_OUT,) bf16
    codebook_reg_ptr,               # (K_CB_REG,) bf16
    # ---- Outputs, one per slice (plus qjl if prod) ----
    out_main_out_ptr,               # (T_q, H_q, D_OUT) bf16
    out_qjl_out_ptr,                # (T_q, H_q, D_OUT) bf16 prod only
    out_main_reg_ptr,               # (T_q, H_q, D_REG) bf16
    out_qjl_reg_ptr,                # (T_q, H_q, D_REG) bf16 prod only
    # ---- Scalars ----
    inv_sqrt_d_full,
    qjl_coef_out,                   # sqrt(pi/2) / d_out
    qjl_coef_reg,                   # sqrt(pi/2) / d_reg
    block_table_stride,
    # ---- Constexprs ----
    OUT_DTYPE: tl.constexpr,
    COMPUTE_DTYPE: tl.constexpr,
    num_heads_q: tl.constexpr,
    num_heads_kv: tl.constexpr,
    D_OUT: tl.constexpr,
    D_REG: tl.constexpr,
    BLOCK_D_OUT: tl.constexpr,      # next_pow2(D_OUT)
    BLOCK_D_REG: tl.constexpr,      # next_pow2(D_REG)
    IDX_DIM_OUT: tl.constexpr,
    IDX_DIM_REG: tl.constexpr,
    QJL_DIM_OUT: tl.constexpr,
    QJL_DIM_REG: tl.constexpr,
    PACK_BITS_OUT: tl.constexpr,
    PACK_BITS_REG: tl.constexpr,
    K_CB_OUT: tl.constexpr,
    K_CB_REG: tl.constexpr,
    block_size: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    USE_QJL: tl.constexpr,
    GQA_GROUP: tl.constexpr,
):
    q_idx = tl.program_id(0)
    kvh_idx = tl.program_id(1)
    q_head_start = kvh_idx * GQA_GROUP

    seq_idx = tl.load(seq_id_per_query_ptr + q_idx)
    kv_end = tl.load(kv_end_per_query_ptr + q_idx)

    m_off = tl.arange(0, BLOCK_M)
    n_off = tl.arange(0, BLOCK_N)
    d_off_out = tl.arange(0, BLOCK_D_OUT)
    d_off_reg = tl.arange(0, BLOCK_D_REG)
    mask_m = m_off < GQA_GROUP
    mask_d_out = d_off_out < D_OUT
    mask_d_reg = d_off_reg < D_REG

    # -----------------------------------------------------------------
    # Load Q tiles (pre-rotated per slice, full D in the slice dim)
    # -----------------------------------------------------------------
    q_base = q_idx * num_heads_q
    q_off_out = (
        (q_base + q_head_start + m_off[:, None]) * D_OUT
        + d_off_out[None, :]
    )
    q_rot_out = tl.load(
        q_rot_out_ptr + q_off_out,
        mask=mask_m[:, None] & mask_d_out[None, :],
        other=0.0,
    )

    q_off_reg = (
        (q_base + q_head_start + m_off[:, None]) * D_REG
        + d_off_reg[None, :]
    )
    q_rot_reg = tl.load(
        q_rot_reg_ptr + q_off_reg,
        mask=mask_m[:, None] & mask_d_reg[None, :],
        other=0.0,
    )

    if USE_QJL:
        sq_out = tl.load(
            Sq_out_ptr + q_off_out,
            mask=mask_m[:, None] & mask_d_out[None, :],
            other=0.0,
        )
        sq_reg = tl.load(
            Sq_reg_ptr + q_off_reg,
            mask=mask_m[:, None] & mask_d_reg[None, :],
            other=0.0,
        )

    # Accumulators
    m_i = tl.full((BLOCK_M,), _NEG_LARGE, dtype=tl.float32)
    l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)
    acc_out_main = tl.zeros((BLOCK_M, BLOCK_D_OUT), dtype=tl.float32)
    acc_reg_main = tl.zeros((BLOCK_M, BLOCK_D_REG), dtype=tl.float32)
    if USE_QJL:
        acc_out_qjl = tl.zeros((BLOCK_M, BLOCK_D_OUT), dtype=tl.float32)
        acc_reg_qjl = tl.zeros((BLOCK_M, BLOCK_D_REG), dtype=tl.float32)

    # Packing constants per slice
    N_PER_BYTE_OUT: tl.constexpr = 8 // PACK_BITS_OUT
    UNPACK_MASK_OUT: tl.constexpr = (1 << PACK_BITS_OUT) - 1
    d_pack_out = d_off_out // N_PER_BYTE_OUT
    d_shift_out = (d_off_out % N_PER_BYTE_OUT) * PACK_BITS_OUT

    N_PER_BYTE_REG: tl.constexpr = 8 // PACK_BITS_REG
    UNPACK_MASK_REG: tl.constexpr = (1 << PACK_BITS_REG) - 1
    d_pack_reg = d_off_reg // N_PER_BYTE_REG
    d_shift_reg = (d_off_reg % N_PER_BYTE_REG) * PACK_BITS_REG

    # QJL bit-position tables per slice
    if USE_QJL:
        bit_pos_out = d_off_out % 8
        byte_pos_out = d_off_out // 8
        bit_pos_reg = d_off_reg % 8
        byte_pos_reg = d_off_reg // 8

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

        meta_addrs = (
            phys_blocks * (block_size * num_heads_kv)
            + tok_in_block * num_heads_kv
            + kvh_idx
        )

        # ================================================================
        # K outlier slice: dequant and QK contribution
        # ================================================================
        base_idx_out = (
            phys_blocks * (block_size * num_heads_kv * IDX_DIM_OUT)
            + tok_in_block * (num_heads_kv * IDX_DIM_OUT)
            + kvh_idx * IDX_DIM_OUT
        )
        addrs_k_out = base_idx_out[:, None] + d_pack_out[None, :]
        packed_k_out = tl.load(
            cache_k_idx_out_ptr + addrs_k_out,
            mask=mask_n[:, None] & mask_d_out[None, :],
            other=0,
        ).to(tl.uint8)
        k_idx_out = (
            (packed_k_out.to(tl.int32) >> d_shift_out[None, :])
            & UNPACK_MASK_OUT
        )
        k_tile_out = tl.load(codebook_out_ptr + k_idx_out)

        k_norm_out = tl.load(
            cache_k_norm_out_ptr + meta_addrs, mask=mask_n, other=0.0
        ).to(tl.float32)
        k_tile_out_scaled = (
            k_tile_out.to(tl.float32) * k_norm_out[:, None]
        ).to(COMPUTE_DTYPE)
        qk_out = tl.dot(q_rot_out, tl.trans(k_tile_out_scaled))

        if USE_QJL:
            base_qjl_out = (
                phys_blocks * (block_size * num_heads_kv * QJL_DIM_OUT)
                + tok_in_block * (num_heads_kv * QJL_DIM_OUT)
                + kvh_idx * QJL_DIM_OUT
            )
            addrs_qjl_k_out = base_qjl_out[:, None] + byte_pos_out[None, :]
            qjl_byte_k_out = tl.load(
                cache_k_qjl_sign_out_ptr + addrs_qjl_k_out,
                mask=mask_n[:, None] & mask_d_out[None, :],
                other=0,
            ).to(tl.int32)
            bit_k_out = (qjl_byte_k_out >> bit_pos_out[None, :]) & 1
            qjl_sign_k_out = (
                1.0 - 2.0 * bit_k_out.to(tl.float32)
            ).to(COMPUTE_DTYPE)

            r_norm_k_out = tl.load(
                cache_k_rnorm_out_ptr + meta_addrs, mask=mask_n, other=0.0
            ).to(tl.float32)
            qjl_dot_k_out = tl.dot(sq_out, tl.trans(qjl_sign_k_out))
            qk_out = qk_out + (
                qjl_coef_out
                * r_norm_k_out[None, :]
                * qjl_dot_k_out
                * k_norm_out[None, :]
            )

        # ================================================================
        # K regular slice: dequant and QK contribution
        # ================================================================
        base_idx_reg = (
            phys_blocks * (block_size * num_heads_kv * IDX_DIM_REG)
            + tok_in_block * (num_heads_kv * IDX_DIM_REG)
            + kvh_idx * IDX_DIM_REG
        )
        addrs_k_reg = base_idx_reg[:, None] + d_pack_reg[None, :]
        packed_k_reg = tl.load(
            cache_k_idx_reg_ptr + addrs_k_reg,
            mask=mask_n[:, None] & mask_d_reg[None, :],
            other=0,
        ).to(tl.uint8)
        k_idx_reg = (
            (packed_k_reg.to(tl.int32) >> d_shift_reg[None, :])
            & UNPACK_MASK_REG
        )
        k_tile_reg = tl.load(codebook_reg_ptr + k_idx_reg)

        k_norm_reg = tl.load(
            cache_k_norm_reg_ptr + meta_addrs, mask=mask_n, other=0.0
        ).to(tl.float32)
        k_tile_reg_scaled = (
            k_tile_reg.to(tl.float32) * k_norm_reg[:, None]
        ).to(COMPUTE_DTYPE)
        qk_reg = tl.dot(q_rot_reg, tl.trans(k_tile_reg_scaled))

        if USE_QJL:
            base_qjl_reg = (
                phys_blocks * (block_size * num_heads_kv * QJL_DIM_REG)
                + tok_in_block * (num_heads_kv * QJL_DIM_REG)
                + kvh_idx * QJL_DIM_REG
            )
            addrs_qjl_k_reg = base_qjl_reg[:, None] + byte_pos_reg[None, :]
            qjl_byte_k_reg = tl.load(
                cache_k_qjl_sign_reg_ptr + addrs_qjl_k_reg,
                mask=mask_n[:, None] & mask_d_reg[None, :],
                other=0,
            ).to(tl.int32)
            bit_k_reg = (qjl_byte_k_reg >> bit_pos_reg[None, :]) & 1
            qjl_sign_k_reg = (
                1.0 - 2.0 * bit_k_reg.to(tl.float32)
            ).to(COMPUTE_DTYPE)

            r_norm_k_reg = tl.load(
                cache_k_rnorm_reg_ptr + meta_addrs, mask=mask_n, other=0.0
            ).to(tl.float32)
            qjl_dot_k_reg = tl.dot(sq_reg, tl.trans(qjl_sign_k_reg))
            qk_reg = qk_reg + (
                qjl_coef_reg
                * r_norm_k_reg[None, :]
                * qjl_dot_k_reg
                * k_norm_reg[None, :]
            )

        # ================================================================
        # Combined logit: <q, k> = <q_out, k_out> + <q_reg, k_reg>
        # Divide by sqrt(head_dim_full) for standard attention scale.
        # ================================================================
        logit = (qk_out + qk_reg) * inv_sqrt_d_full
        logit = tl.where(mask_n[None, :], logit, _NEG_LARGE)

        # ---- Online softmax ----
        m_new = tl.maximum(m_i, tl.max(logit, axis=1))
        alpha = tl.exp(m_i - m_new)
        probs = tl.exp(logit - m_new[:, None])
        l_i = l_i * alpha + tl.sum(probs, axis=1)

        acc_out_main = acc_out_main * alpha[:, None]
        acc_reg_main = acc_reg_main * alpha[:, None]
        if USE_QJL:
            acc_out_qjl = acc_out_qjl * alpha[:, None]
            acc_reg_qjl = acc_reg_qjl * alpha[:, None]

        probs_cast = probs.to(COMPUTE_DTYPE)

        # ================================================================
        # V outlier slice: main + (prod) qjl accumulation
        # ================================================================
        addrs_v_out = base_idx_out[:, None] + d_pack_out[None, :]
        packed_v_out = tl.load(
            cache_v_idx_out_ptr + addrs_v_out,
            mask=mask_n[:, None] & mask_d_out[None, :],
            other=0,
        ).to(tl.uint8)
        v_idx_out = (
            (packed_v_out.to(tl.int32) >> d_shift_out[None, :])
            & UNPACK_MASK_OUT
        )
        v_tile_out = tl.load(codebook_out_ptr + v_idx_out)
        v_norm_out = tl.load(
            cache_v_norm_out_ptr + meta_addrs, mask=mask_n, other=0.0
        ).to(tl.float32)
        v_tile_out_scaled = (
            v_tile_out.to(tl.float32) * v_norm_out[:, None]
        ).to(COMPUTE_DTYPE)
        acc_out_main = acc_out_main + tl.dot(probs_cast, v_tile_out_scaled)

        if USE_QJL:
            addrs_qjl_v_out = base_qjl_out[:, None] + byte_pos_out[None, :]
            qjl_byte_v_out = tl.load(
                cache_v_qjl_sign_out_ptr + addrs_qjl_v_out,
                mask=mask_n[:, None] & mask_d_out[None, :],
                other=0,
            ).to(tl.int32)
            bit_v_out = (qjl_byte_v_out >> bit_pos_out[None, :]) & 1
            qjl_sign_v_out = (
                1.0 - 2.0 * bit_v_out.to(tl.float32)
            ).to(COMPUTE_DTYPE)
            r_norm_v_out = tl.load(
                cache_v_rnorm_out_ptr + meta_addrs, mask=mask_n, other=0.0
            ).to(tl.float32)
            v_qjl_tile_out = (
                qjl_sign_v_out.to(tl.float32)
                * r_norm_v_out[:, None]
                * v_norm_out[:, None]
            ).to(COMPUTE_DTYPE)
            acc_out_qjl = acc_out_qjl + tl.dot(probs_cast, v_qjl_tile_out)

        # ================================================================
        # V regular slice: main + (prod) qjl accumulation
        # ================================================================
        addrs_v_reg = base_idx_reg[:, None] + d_pack_reg[None, :]
        packed_v_reg = tl.load(
            cache_v_idx_reg_ptr + addrs_v_reg,
            mask=mask_n[:, None] & mask_d_reg[None, :],
            other=0,
        ).to(tl.uint8)
        v_idx_reg = (
            (packed_v_reg.to(tl.int32) >> d_shift_reg[None, :])
            & UNPACK_MASK_REG
        )
        v_tile_reg = tl.load(codebook_reg_ptr + v_idx_reg)
        v_norm_reg = tl.load(
            cache_v_norm_reg_ptr + meta_addrs, mask=mask_n, other=0.0
        ).to(tl.float32)
        v_tile_reg_scaled = (
            v_tile_reg.to(tl.float32) * v_norm_reg[:, None]
        ).to(COMPUTE_DTYPE)
        acc_reg_main = acc_reg_main + tl.dot(probs_cast, v_tile_reg_scaled)

        if USE_QJL:
            addrs_qjl_v_reg = base_qjl_reg[:, None] + byte_pos_reg[None, :]
            qjl_byte_v_reg = tl.load(
                cache_v_qjl_sign_reg_ptr + addrs_qjl_v_reg,
                mask=mask_n[:, None] & mask_d_reg[None, :],
                other=0,
            ).to(tl.int32)
            bit_v_reg = (qjl_byte_v_reg >> bit_pos_reg[None, :]) & 1
            qjl_sign_v_reg = (
                1.0 - 2.0 * bit_v_reg.to(tl.float32)
            ).to(COMPUTE_DTYPE)
            r_norm_v_reg = tl.load(
                cache_v_rnorm_reg_ptr + meta_addrs, mask=mask_n, other=0.0
            ).to(tl.float32)
            v_qjl_tile_reg = (
                qjl_sign_v_reg.to(tl.float32)
                * r_norm_v_reg[:, None]
                * v_norm_reg[:, None]
            ).to(COMPUTE_DTYPE)
            acc_reg_qjl = acc_reg_qjl + tl.dot(probs_cast, v_qjl_tile_reg)

        m_i = m_new

    # ================================================================
    # Normalize and store. Four outputs (two if not USE_QJL); Python
    # post-pass does acc_qjl @ S + Pi^T per slice and scatters back.
    # ================================================================
    inv_l = 1.0 / tl.maximum(l_i[:, None], 1e-12)

    out_off_out = (
        (q_base + q_head_start + m_off[:, None]) * D_OUT
        + d_off_out[None, :]
    )
    tl.store(
        out_main_out_ptr + out_off_out,
        (acc_out_main * inv_l).to(OUT_DTYPE),
        mask=mask_m[:, None] & mask_d_out[None, :],
    )
    if USE_QJL:
        tl.store(
            out_qjl_out_ptr + out_off_out,
            (acc_out_qjl * inv_l).to(OUT_DTYPE),
            mask=mask_m[:, None] & mask_d_out[None, :],
        )

    out_off_reg = (
        (q_base + q_head_start + m_off[:, None]) * D_REG
        + d_off_reg[None, :]
    )
    tl.store(
        out_main_reg_ptr + out_off_reg,
        (acc_reg_main * inv_l).to(OUT_DTYPE),
        mask=mask_m[:, None] & mask_d_reg[None, :],
    )
    if USE_QJL:
        tl.store(
            out_qjl_reg_ptr + out_off_reg,
            (acc_reg_qjl * inv_l).to(OUT_DTYPE),
            mask=mask_m[:, None] & mask_d_reg[None, :],
        )


def turboquant_paged_attention_split_tc(
    q: torch.Tensor,
    # Outlier buffers (K + V)
    cache_k_idx_out: torch.Tensor,
    cache_k_norm_out: torch.Tensor,
    cache_v_idx_out: torch.Tensor,
    cache_v_norm_out: torch.Tensor,
    cache_k_qjl_sign_out: torch.Tensor | None,
    cache_k_rnorm_out: torch.Tensor | None,
    cache_v_qjl_sign_out: torch.Tensor | None,
    cache_v_rnorm_out: torch.Tensor | None,
    # Regular buffers (K + V)
    cache_k_idx_reg: torch.Tensor,
    cache_k_norm_reg: torch.Tensor,
    cache_v_idx_reg: torch.Tensor,
    cache_v_norm_reg: torch.Tensor,
    cache_k_qjl_sign_reg: torch.Tensor | None,
    cache_k_rnorm_reg: torch.Tensor | None,
    cache_v_qjl_sign_reg: torch.Tensor | None,
    cache_v_rnorm_reg: torch.Tensor | None,
    # Shared attention metadata
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    query_start_loc: torch.Tensor,
    # Split state (K side carries outlier_idx / regular_idx; V side
    # passed separately because its indices may differ).
    state_k: "SplitQuantState",
    state_v: "SplitQuantState",
) -> torch.Tensor:
    """Split-slice paged attention for the outlier-split configuration.

    ``state_k.outlier_idx`` selects the K-slice channel positions; the
    Python wrapper gathers those positions from Q to form the outlier
    Q slice and their complement to form the regular Q slice. The
    kernel produces four d-dim outputs per (query_token, head) which
    we recombine via one cuBLAS matmul per slice and scatter back into
    the full head_dim output.
    """
    num_query_tokens, num_heads_q, head_size = q.shape
    _, block_size, num_heads_kv, idx_dim_out = cache_k_idx_out.shape
    _, _, _, idx_dim_reg = cache_k_idx_reg.shape
    num_seqs = int(seq_lens.shape[0])

    use_qjl = state_k.algo == "prod"
    if use_qjl:
        assert state_v.algo == "prod"
        for buf in (
            cache_k_qjl_sign_out, cache_k_rnorm_out,
            cache_v_qjl_sign_out, cache_v_rnorm_out,
            cache_k_qjl_sign_reg, cache_k_rnorm_reg,
            cache_v_qjl_sign_reg, cache_v_rnorm_reg,
        ):
            assert buf is not None, "prod split attend requires all 8 QJL buffers"

    dev = q.device

    # --- Attention metadata scaffolds (same pattern as attend_tc) ---
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

    # --- Gather Q into outlier / regular slices, then rotate per-slice ---
    # K-side slice indices drive Q's slicing at QK time; V-side slice
    # indices are only used at V output scatter time. K and V slice
    # layouts may differ so Q gets rotated once for K paths.
    outlier_idx_k = state_k.outlier_idx
    regular_idx_k = state_k.regular_idx
    d_out_k = outlier_idx_k.numel()
    d_reg_k = regular_idx_k.numel()

    q_flat = q.reshape(-1, head_size)
    q_out_k = q_flat.index_select(dim=-1, index=outlier_idx_k)
    q_reg_k = q_flat.index_select(dim=-1, index=regular_idx_k)
    q_rot_out = (q_out_k @ state_k.state_out.Pi).view(
        num_query_tokens, num_heads_q, d_out_k
    ).contiguous()
    q_rot_reg = (q_reg_k @ state_k.state_reg.Pi).view(
        num_query_tokens, num_heads_q, d_reg_k
    ).contiguous()

    if use_qjl:
        Sq_out = (
            q_rot_out.reshape(-1, d_out_k) @ state_k.state_out.S.t()
        ).view(num_query_tokens, num_heads_q, d_out_k).contiguous()
        Sq_reg = (
            q_rot_reg.reshape(-1, d_reg_k) @ state_k.state_reg.S.t()
        ).view(num_query_tokens, num_heads_q, d_reg_k).contiguous()
    else:
        Sq_out = q_rot_out
        Sq_reg = q_rot_reg

    # --- Kernel output buffers (in rotated unit-sphere space per slice) ---
    out_main_out = torch.empty_like(q_rot_out)
    out_main_reg = torch.empty_like(q_rot_reg)
    if use_qjl:
        out_qjl_out = torch.empty_like(q_rot_out)
        out_qjl_reg = torch.empty_like(q_rot_reg)
        k_qjl_out_buf = cache_k_qjl_sign_out
        k_qjl_reg_buf = cache_k_qjl_sign_reg
        v_qjl_out_buf = cache_v_qjl_sign_out
        v_qjl_reg_buf = cache_v_qjl_sign_reg
        k_rn_out_buf = cache_k_rnorm_out
        k_rn_reg_buf = cache_k_rnorm_reg
        v_rn_out_buf = cache_v_rnorm_out
        v_rn_reg_buf = cache_v_rnorm_reg
    else:
        out_qjl_out = out_main_out
        out_qjl_reg = out_main_reg
        # Dummy pointers for the USE_QJL=False branch. The kernel
        # never dereferences these under constexpr-guarded ifs.
        k_qjl_out_buf = cache_k_idx_out
        k_qjl_reg_buf = cache_k_idx_reg
        v_qjl_out_buf = cache_v_idx_out
        v_qjl_reg_buf = cache_v_idx_reg
        k_rn_out_buf = cache_k_norm_out
        k_rn_reg_buf = cache_k_norm_reg
        v_rn_out_buf = cache_v_norm_out
        v_rn_reg_buf = cache_v_norm_reg

    codebook_out_ct = state_k.state_out.codebook.to(q.dtype).contiguous()
    codebook_reg_ct = state_k.state_reg.codebook.to(q.dtype).contiguous()

    gqa_group = num_heads_q // num_heads_kv
    BLOCK_M = max(gqa_group, _BLOCK_M_MIN)
    BLOCK_M = 1 << (BLOCK_M - 1).bit_length()
    BLOCK_D_OUT = triton.next_power_of_2(d_out_k)
    BLOCK_D_REG = triton.next_power_of_2(d_reg_k)

    OUT_DTYPE = tl.bfloat16 if q.dtype == torch.bfloat16 else tl.float16
    COMPUTE_DTYPE = OUT_DTYPE

    inv_sqrt_d_full = 1.0 / math.sqrt(float(head_size))
    qjl_coef_out = math.sqrt(math.pi / 2.0) / float(d_out_k)
    qjl_coef_reg = math.sqrt(math.pi / 2.0) / float(d_reg_k)
    qjl_dim_out = d_out_k // 8 if use_qjl else 1
    qjl_dim_reg = d_reg_k // 8 if use_qjl else 1
    block_table_stride = int(block_table.shape[1])

    grid = (num_query_tokens, num_heads_kv)
    _split_attend_kernel[grid](
        q_rot_out, Sq_out,
        q_rot_reg, Sq_reg,
        cache_k_idx_out, cache_k_norm_out,
        k_qjl_out_buf, k_rn_out_buf,
        cache_k_idx_reg, cache_k_norm_reg,
        k_qjl_reg_buf, k_rn_reg_buf,
        cache_v_idx_out, cache_v_norm_out,
        v_qjl_out_buf, v_rn_out_buf,
        cache_v_idx_reg, cache_v_norm_reg,
        v_qjl_reg_buf, v_rn_reg_buf,
        block_table,
        seq_id_per_query, kv_end_per_query,
        codebook_out_ct, codebook_reg_ct,
        out_main_out, out_qjl_out,
        out_main_reg, out_qjl_reg,
        inv_sqrt_d_full, qjl_coef_out, qjl_coef_reg,
        block_table_stride,
        OUT_DTYPE=OUT_DTYPE,
        COMPUTE_DTYPE=COMPUTE_DTYPE,
        num_heads_q=num_heads_q,
        num_heads_kv=num_heads_kv,
        D_OUT=d_out_k,
        D_REG=d_reg_k,
        BLOCK_D_OUT=BLOCK_D_OUT,
        BLOCK_D_REG=BLOCK_D_REG,
        IDX_DIM_OUT=idx_dim_out,
        IDX_DIM_REG=idx_dim_reg,
        QJL_DIM_OUT=qjl_dim_out,
        QJL_DIM_REG=qjl_dim_reg,
        PACK_BITS_OUT=state_k.state_out.pack_bits,
        PACK_BITS_REG=state_k.state_reg.pack_bits,
        K_CB_OUT=int(state_k.state_out.codebook.shape[0]),
        K_CB_REG=int(state_k.state_reg.codebook.shape[0]),
        block_size=block_size,
        BLOCK_M=BLOCK_M,
        USE_QJL=use_qjl,
        GQA_GROUP=gqa_group,
    )

    # --- Post-pass: resolve QJL via S_v, apply Pi_T per slice, scatter ---
    # V slice indices may differ from K slice indices. acc_out/acc_reg
    # are in the K state's rotated unit-sphere space for each slice --
    # but the Q pre-rotation used K's Pi, so the output "scale" is in
    # the K state's space. Since K and V share the same attention
    # weights, the natural place to apply V's Pi^T is indeed V's state.
    # However, attention semantics say the V output is in the ORIGINAL
    # unrotated v-space, so post-rotate via V's Pi_T per slice.
    # NOTE: if K and V slice layouts differ (different outlier_idx for
    # K vs V), we cannot use the same kernel output shapes. Caller
    # must ensure state_k.outlier_idx == state_v.outlier_idx for now.
    assert state_k.outlier_idx.equal(state_v.outlier_idx), (
        "This branch requires state_k.outlier_idx == state_v.outlier_idx; "
        "per-head independent K/V splits would need a dual-shape kernel."
    )
    assert state_k.regular_idx.equal(state_v.regular_idx)

    out_flat_out = out_main_out.reshape(-1, d_out_k)
    out_flat_reg = out_main_reg.reshape(-1, d_reg_k)
    if use_qjl:
        out_flat_out = out_flat_out + qjl_coef_out * (
            out_qjl_out.reshape(-1, d_out_k) @ state_v.state_out.S
        )
        out_flat_reg = out_flat_reg + qjl_coef_reg * (
            out_qjl_reg.reshape(-1, d_reg_k) @ state_v.state_reg.S
        )
    slice_out_v = (out_flat_out @ state_v.state_out.Pi_T).view(
        num_query_tokens, num_heads_q, d_out_k
    )
    slice_reg_v = (out_flat_reg @ state_v.state_reg.Pi_T).view(
        num_query_tokens, num_heads_q, d_reg_k
    )

    # Scatter slices back to full head_dim.
    output = torch.empty(
        num_query_tokens, num_heads_q, head_size,
        dtype=q.dtype, device=dev,
    )
    output.index_copy_(-1, state_v.outlier_idx, slice_out_v)
    output.index_copy_(-1, state_v.regular_idx, slice_reg_v)
    return output
