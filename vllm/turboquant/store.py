# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Triton-backed K/V quantization store for TurboQuant.

Paper-faithful reading of Algorithm 1 / Algorithm 2 in
arXiv:2504.19874:

    y        = Pi @ (x / ||x||)          # unit-norm rotated vector
    idx_j    = argmin_k |y_j - c_k|      # Lloyd-Max per coord
    r        = y - codebook[idx]         # residual in rotated space
    qjl_sign = sign(S @ (r / ||r||))     # 1-bit QJL of unit residual

We store (idx, ||x||, qjl_sign, ||r||) per (token, kv_head); dequant
reconstructs y_approx = codebook[idx] + (sqrt(pi/2)/d) * ||r|| * S^T @ qjl
in rotated-unit-sphere space, then multiplies by ||x|| to recover the
original-scale vector after applying Pi^T.

Both K and V follow the same quantizer (Q_mse or Q_prod) determined
by ``state.algo``; the paper does not separate K and V treatment.

The Python wrapper performs the rotation (bf16 matmul -> tensor core)
and unit-norm scaling; the Triton kernel does the scalar quantizer,
nibble/bit packing, and the per-slot paged scatter.

Scope: b=4 only in this first cut (K_CB <= 16, nibble-packed idx).
Stage 2 will add variable bit-widths.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from vllm.triton_utils import tl, triton

if TYPE_CHECKING:
    from vllm.turboquant.codebook import QuantState


@triton.jit
def _store_quant_kernel(
    rotated_ptr,            # (T, H_kv, d) bf16/fp16  unit-norm rotated input
    x_norm_ptr,             # (T, H_kv)    fp32       ||x|| per (tok, head)
    slot_mapping_ptr,       # (T,)         int64
    cache_idx_ptr,          # (num_blocks, bs, H_kv, idx_dim) uint8
    cache_norm_ptr,         # (num_blocks, bs, H_kv)          fp32
    cache_qjl_sign_ptr,     # (num_blocks, bs, H_kv, d/8)     uint8 prod only
    cache_rnorm_ptr,        # (num_blocks, bs, H_kv)          fp32  prod only
    boundaries_ptr,         # (K_CB - 1,) fp32
    codebook_ptr,           # (K_CB,)     fp32  (we cast inside)
    S_ptr,                  # (d, d)      bf16/fp16    prod only
    block_size,
    num_heads_kv,
    head_size,
    idx_dim,
    qjl_dim,
    BLOCK_D: tl.constexpr,           # next_pow2(head_size)
    BLOCK_IDX_DIM: tl.constexpr,     # next_pow2(idx_dim)
    BLOCK_QJL_DIM: tl.constexpr,     # next_pow2(qjl_dim) or 1 when !USE_QJL
    K_CB: tl.constexpr,
    PACK_BITS: tl.constexpr,         # 1, 2, 4, or 8
    USE_QJL: tl.constexpr,
):
    tok = tl.program_id(0)
    kvh = tl.program_id(1)

    slot = tl.load(slot_mapping_ptr + tok)
    if slot < 0:
        return

    phys = slot // block_size
    off = slot % block_size

    d_idx = tl.arange(0, BLOCK_D)
    mask_d = d_idx < head_size

    # --- 1) Load rotated value (unit-norm), promote to fp32 ---
    rot_base = (tok * num_heads_kv + kvh) * head_size
    rot_ct = tl.load(rotated_ptr + rot_base + d_idx, mask=mask_d, other=0.0)
    rot = rot_ct.to(tl.float32)

    # --- 2) Scalar Lloyd-Max: searchsorted against K_CB-1 boundaries ---
    idx = tl.zeros((BLOCK_D,), dtype=tl.int32)
    for c in tl.static_range(0, K_CB - 1):
        boundary = tl.load(boundaries_ptr + c)
        idx += (rot >= boundary).to(tl.int32)
    idx = tl.minimum(idx, K_CB - 1)

    # --- 3) Pack PACK_BITS-bit indices into uint8 via mask-sum ---
    # PACK_BITS in {1, 2, 4, 8} => N_PER_BYTE in {8, 4, 2, 1}.
    # For a coord j landing in byte B=j // N_PER_BYTE at bit offset
    # o=(j % N_PER_BYTE) * PACK_BITS, contrib = idx[j] << o; packed
    # byte = sum of contribs from all j mapping to B (non-overlapping
    # bits by construction so the sum equals the bitwise OR).
    N_PER_BYTE: tl.constexpr = 8 // PACK_BITS
    d_byte_pos = d_idx // N_PER_BYTE
    d_bit_shift = (d_idx % N_PER_BYTE) * PACK_BITS
    idx_shifted = idx << d_bit_shift
    pair_pos = tl.arange(0, BLOCK_IDX_DIM)
    pack_match = d_byte_pos[:, None] == pair_pos[None, :]
    pack_contrib = tl.where(pack_match, idx_shifted[:, None], 0)
    packed = tl.sum(pack_contrib, axis=0).to(tl.uint8)

    mask_pair = pair_pos < idx_dim
    idx_out = (
        phys * (block_size * num_heads_kv * idx_dim)
        + off * (num_heads_kv * idx_dim)
        + kvh * idx_dim
    )
    tl.store(cache_idx_ptr + idx_out + pair_pos, packed, mask=mask_pair)

    # --- 4) Write per-(tok, head) norm ---
    norm_val = tl.load(x_norm_ptr + tok * num_heads_kv + kvh)
    meta_out = (
        phys * (block_size * num_heads_kv)
        + off * num_heads_kv
        + kvh
    )
    tl.store(cache_norm_ptr + meta_out, norm_val)

    # --- 5) Prod path: QJL on residual r = rot - codebook[idx] ---
    if USE_QJL:
        rk = tl.load(codebook_ptr + idx).to(tl.float32)
        r = rot - rk

        r_norm_sq = tl.sum(r * r)
        r_norm = tl.sqrt(r_norm_sq)
        r_norm = tl.maximum(r_norm, 1e-6)
        r_unit = r / r_norm

        tl.store(cache_rnorm_ptr + meta_out, r_norm)

        # qjl_raw = S @ r_unit (d-dim matvec, on-chip fp32).
        S_tile = tl.load(
            S_ptr + d_idx[:, None] * head_size + d_idx[None, :],
            mask=mask_d[:, None] & mask_d[None, :],
            other=0.0,
        ).to(tl.float32)
        qjl_raw = tl.sum(S_tile * r_unit[None, :], axis=1)

        # Bit-pack: bit = (qjl_raw < 0); 8 bits per byte.
        qjl_bit = (qjl_raw < 0).to(tl.int32)
        d_bit_pos = d_idx % 8
        d_byte_pos_qjl = d_idx // 8
        bit_weight = 1 << d_bit_pos
        bit_contrib = qjl_bit * bit_weight

        byte_pos_qjl = tl.arange(0, BLOCK_QJL_DIM)
        qjl_match = d_byte_pos_qjl[:, None] == byte_pos_qjl[None, :]
        qjl_contrib_2d = tl.where(qjl_match, bit_contrib[:, None], 0)
        packed_qjl = tl.sum(qjl_contrib_2d, axis=0).to(tl.uint8)

        mask_qjl = byte_pos_qjl < qjl_dim
        qjl_out = (
            phys * (block_size * num_heads_kv * qjl_dim)
            + off * (num_heads_kv * qjl_dim)
            + kvh * qjl_dim
        )
        tl.store(cache_qjl_sign_ptr + qjl_out + byte_pos_qjl,
                 packed_qjl, mask=mask_qjl)


def _rotate_and_store(
    new_x: torch.Tensor,
    cache_idx: torch.Tensor,
    cache_norm: torch.Tensor,
    cache_qjl_sign: torch.Tensor | None,
    cache_rnorm: torch.Tensor | None,
    slot_mapping: torch.Tensor,
    state: "QuantState",
    block_size: int,
    use_qjl: bool,
) -> None:
    """Paper-faithful quant store for one tensor (K or V).

    Steps:
      1. norm   = ||x||                 # fp32 reduction
      2. scaled = x / norm              # unit-sphere rescale, bf16
      3. rotated = scaled @ Pi          # bf16 matmul (tensor core)
      4. Triton kernel: searchsorted -> idx, pack, store; and (prod)
         residual -> QJL sign-pack.
    """
    if new_x.shape[0] == 0:
        return
    T, H_kv, d = new_x.shape
    device = new_x.device
    dtype = new_x.dtype

    # ||x|| in fp32 for numerical stability. clamp_min matches the
    # previous implementation's eps handling and guards against zero
    # V rows at the prefill boundary.
    x_norm = (
        new_x.float().pow(2).sum(dim=-1).clamp_min(1e-12).sqrt().clamp_min(1e-6)
    )  # (T, H_kv)

    # Unit-sphere rescale + random orthogonal rotation. Pi is not
    # symmetric, unlike the old Hadamard path; we always multiply on
    # the right here.
    inv_norm = (1.0 / x_norm).to(dtype)                            # (T, H_kv)
    x_scaled = new_x * inv_norm.unsqueeze(-1)                      # bf16 unit norm
    rotated = (
        x_scaled.reshape(T * H_kv, d) @ state.Pi
    ).view(T, H_kv, d).contiguous()                                # bf16 on S^{d-1}

    boundaries_f = state.boundaries.to(torch.float32)
    codebook_f = state.codebook.to(torch.float32)

    K_CB = int(state.codebook.shape[0])
    pack_bits = state.pack_bits
    assert (d * pack_bits) % 8 == 0, (
        f"head_dim * pack_bits must be byte-aligned; got "
        f"head_dim={d} pack_bits={pack_bits}"
    )
    expected_idx_dim = d * pack_bits // 8
    idx_dim = int(cache_idx.shape[-1])
    assert idx_dim == expected_idx_dim, (
        f"cache_idx last dim {idx_dim} != expected {expected_idx_dim} "
        f"(head_dim={d}, pack_bits={pack_bits})"
    )
    qjl_dim = int(cache_qjl_sign.shape[-1]) if use_qjl else 1

    BLOCK_D = triton.next_power_of_2(d)
    BLOCK_IDX_DIM = triton.next_power_of_2(idx_dim)
    BLOCK_QJL_DIM = triton.next_power_of_2(qjl_dim) if use_qjl else 1

    qjl_sign_buf = cache_qjl_sign if use_qjl else cache_idx
    rnorm_buf = cache_rnorm if use_qjl else cache_norm
    S_buf = state.S if use_qjl else state.Pi  # any same-dtype tensor works

    grid = (T, H_kv)
    _store_quant_kernel[grid](
        rotated,
        x_norm,
        slot_mapping,
        cache_idx,
        cache_norm,
        qjl_sign_buf,
        rnorm_buf,
        boundaries_f,
        codebook_f,
        S_buf,
        block_size,
        H_kv,
        d,
        idx_dim,
        qjl_dim,
        BLOCK_D=BLOCK_D,
        BLOCK_IDX_DIM=BLOCK_IDX_DIM,
        BLOCK_QJL_DIM=BLOCK_QJL_DIM,
        K_CB=K_CB,
        PACK_BITS=pack_bits,
        USE_QJL=use_qjl,
    )


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
    """Quantize K into the paged caches via the shared store kernel."""
    use_qjl = state.algo == "prod"
    if use_qjl:
        assert cache_k_qjl_sign is not None and cache_k_rnorm is not None, (
            "prod requires cache_k_qjl_sign and cache_k_rnorm"
        )
    _rotate_and_store(
        new_k,
        cache_k_idx,
        cache_k_norm,
        cache_k_qjl_sign,
        cache_k_rnorm,
        slot_mapping,
        state,
        block_size,
        use_qjl=use_qjl,
    )


def turboquant_store_v(
    new_v: torch.Tensor,
    cache_v_idx: torch.Tensor,
    cache_v_norm: torch.Tensor,
    slot_mapping: torch.Tensor,
    state: "QuantState",
    block_size: int,
    cache_v_qjl_sign: torch.Tensor | None = None,
    cache_v_rnorm: torch.Tensor | None = None,
) -> None:
    """Quantize V with the same quantizer as K (paper Section 4.2 applies
    TurboQuant uniformly to the KV cache). prod mode stores (idx, norm,
    qjl_sign, r_norm) just like K."""
    use_qjl = state.algo == "prod"
    if use_qjl:
        assert cache_v_qjl_sign is not None and cache_v_rnorm is not None, (
            "prod requires cache_v_qjl_sign and cache_v_rnorm"
        )
    _rotate_and_store(
        new_v,
        cache_v_idx,
        cache_v_norm,
        cache_v_qjl_sign,
        cache_v_rnorm,
        slot_mapping,
        state,
        block_size,
        use_qjl=use_qjl,
    )


def turboquant_store_split(
    new_x: torch.Tensor,
    state_split,  # SplitQuantState
    cache_idx_out: torch.Tensor,
    cache_norm_out: torch.Tensor,
    cache_qjl_sign_out: torch.Tensor | None,
    cache_rnorm_out: torch.Tensor | None,
    cache_idx_reg: torch.Tensor,
    cache_norm_reg: torch.Tensor,
    cache_qjl_sign_reg: torch.Tensor | None,
    cache_rnorm_reg: torch.Tensor | None,
    slot_mapping: torch.Tensor,
    block_size: int,
) -> None:
    """Quantize one tensor (K or V) by gathering outlier vs regular
    channels and running two independent TurboQuant instances.

    Applies paper §4.3's "two independent instances of TurboQuant"
    split. Each slice is normalized independently -- its ``||x||`` is
    the L2 norm of its own channel subset, not the whole head. This
    matches the paper's Algorithm 2 applied to each slice.
    """
    if new_x.shape[0] == 0:
        return
    use_qjl = state_split.algo == "prod"
    if use_qjl:
        assert cache_qjl_sign_out is not None and cache_rnorm_out is not None
        assert cache_qjl_sign_reg is not None and cache_rnorm_reg is not None

    # Gather the two channel subsets along the last dim.
    x_out = new_x.index_select(dim=-1, index=state_split.outlier_idx)
    x_reg = new_x.index_select(dim=-1, index=state_split.regular_idx)

    _rotate_and_store(
        x_out,
        cache_idx_out, cache_norm_out,
        cache_qjl_sign_out, cache_rnorm_out,
        slot_mapping,
        state_split.state_out,
        block_size,
        use_qjl=use_qjl,
    )
    _rotate_and_store(
        x_reg,
        cache_idx_reg, cache_norm_reg,
        cache_qjl_sign_reg, cache_rnorm_reg,
        slot_mapping,
        state_split.state_reg,
        block_size,
        use_qjl=use_qjl,
    )
