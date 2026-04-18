# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Triton-backed K/V quantization store for TurboQuant.

Pipeline per store call (one Triton kernel per ``(token, kv_head)``
program, one extra cuBLAS matmul for the Hadamard rotation):

    Python (per call):
        1. norm   = ||x||                 # fp32 reduction
        2. scaled = x * (sqrt(d) / norm)  # bf16 elementwise
        3. signed = scaled * signs        # bf16 elementwise
        4. rotated = signed @ H           # bf16 matmul (tensor core)
    Triton (one program per (token, kv_head)):
        5. searchsorted rotated through Lloyd-Max boundaries -> idx
        6. 4-bit nibble-pack idx and write cache_idx at paged slot
        7. write cache_norm (fp32) at meta slot
        8. (prod only) residual r = rotated - codebook[idx]; r_norm;
           qjl_raw = S @ r_unit; qjl_bit = (qjl_raw < 0); bit-pack 8
           signs/byte and write cache_qjl_sign + cache_k_rnorm.

This replaces the previous pure-PyTorch path that launched ~11 kernels
per store call and did a fp32 SIMT SGEMM for the Hadamard. Concretely,
the old version spent ~3-4 ms/step in a ``slot_mapping >= 0`` nonzero/
select filter that was never needed for decode (slot_mapping has no
padding at decode). The Triton kernel uses ``if slot < 0: return``
per-program instead, with no CPU-sync roundtrip.

Scope: b=4 only. ``K_CB`` must be <= 16 (4-bit nibble packing).
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from vllm.triton_utils import tl, triton

if TYPE_CHECKING:
    from vllm.turboquant.codebook import QuantState


@triton.jit
def _store_quant_kernel(
    rotated_ptr,            # (T, H_kv, d) bf16/fp16   Hadamard-rotated input
    x_norm_ptr,             # (T, H_kv)    fp32        ||x|| per (tok, head)
    slot_mapping_ptr,       # (T,)         int64
    cache_idx_ptr,          # (num_blocks, bs, H_kv, d/2) uint8
    cache_norm_ptr,         # (num_blocks, bs, H_kv)      fp32
    cache_qjl_sign_ptr,     # (num_blocks, bs, H_kv, d/8) uint8 prod only
    cache_rnorm_ptr,        # (num_blocks, bs, H_kv)      fp32  prod only
    boundaries_ptr,         # (K_CB - 1,) fp32
    codebook_ptr,           # (K_CB,)     fp32 or bf16 (we cast inside)
    S_ptr,                  # (d, d)      bf16/fp16    prod only
    block_size,
    num_heads_kv,
    head_size,
    idx_dim,
    qjl_dim,
    BLOCK_D: tl.constexpr,           # next_pow2(head_size)
    BLOCK_IDX_DIM: tl.constexpr,     # next_pow2(idx_dim), == BLOCK_D // 2
    BLOCK_QJL_DIM: tl.constexpr,     # next_pow2(qjl_dim) or 1 when !USE_QJL
    K_CB: tl.constexpr,
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

    # --- 1) Load rotated value, promote to fp32 for comparisons ---
    rot_base = (tok * num_heads_kv + kvh) * head_size
    rot_ct = tl.load(rotated_ptr + rot_base + d_idx, mask=mask_d, other=0.0)
    rot = rot_ct.to(tl.float32)

    # --- 2) Searchsorted via K_CB-1 branchless comparisons ---
    idx = tl.zeros((BLOCK_D,), dtype=tl.int32)
    for c in tl.static_range(0, K_CB - 1):
        boundary = tl.load(boundaries_ptr + c)
        idx += (rot >= boundary).to(tl.int32)
    idx = tl.minimum(idx, K_CB - 1)

    # --- 3) Pack 4-bit indices via mask-sum: two 4-bit idx per byte ---
    # Contribution of d_idx -> byte (d // 2), nibble (d % 2):
    #   contrib[d] = idx[d] << (4 * (d % 2))
    #   packed[p]  = sum_{d : d//2 == p} contrib[d]
    d_byte_pos = d_idx // 2
    d_nibble_shift = (d_idx % 2) * 4
    idx_shifted = idx << d_nibble_shift
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

    # --- 5) Prod path: QJL on residual ---
    if USE_QJL:
        # r = rotated - codebook[idx]    (fp32)
        rk = tl.load(codebook_ptr + idx).to(tl.float32)
        r = rot - rk

        r_norm_sq = tl.sum(r * r)
        r_norm = tl.sqrt(r_norm_sq)
        r_norm = tl.maximum(r_norm, 1e-6)
        r_unit = r / r_norm

        tl.store(cache_rnorm_ptr + meta_out, r_norm)

        # qjl_raw = S @ r_unit (d-dim matvec).
        # Load S tile (BLOCK_D, BLOCK_D) and broadcast-multiply-sum.
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
    """Common core for K and V: Hadamard rotate (Python) + Triton store."""
    if new_x.shape[0] == 0:
        return
    T, H_kv, d = new_x.shape
    device = new_x.device
    dtype = new_x.dtype

    # ||x||: use fp32 for the reduction (Hadamard-invariant, so we
    # read from the original x).
    x_norm = new_x.float().pow(2).sum(dim=-1).clamp_min(1e-12).sqrt()
    # ``clamp_min(1e-6)`` on the norm itself rather than on the squared
    # sum preserves the eps semantics from the previous implementation.
    x_norm = x_norm.clamp_min(1e-6)  # (T, H_kv) fp32

    # Scale + signs + Hadamard: bf16 matmul -> tensor core.
    scale = (math.sqrt(float(d)) / x_norm).to(dtype)               # (T, H_kv)
    x_scaled = new_x * scale.unsqueeze(-1)                         # bf16
    x_signed = x_scaled * state.signs                              # bf16
    # H is symmetric so H.T == H; one matmul.
    rotated = (
        x_signed.reshape(T * H_kv, d) @ state.H
    ).view(T, H_kv, d).contiguous()                                # bf16

    # Boundaries stay fp32 (they are tiny and used directly by searchsorted).
    boundaries_f = state.boundaries.to(torch.float32)
    codebook_f = state.codebook.to(torch.float32)

    K_CB = int(state.codebook.shape[0])
    assert K_CB <= 16, (
        f"Triton store kernel is b=4 only (K_CB<=16); got K_CB={K_CB}"
    )
    idx_dim = int(cache_idx.shape[-1])
    qjl_dim = int(cache_qjl_sign.shape[-1]) if use_qjl else 1

    BLOCK_D = triton.next_power_of_2(d)
    BLOCK_IDX_DIM = triton.next_power_of_2(idx_dim)
    BLOCK_QJL_DIM = triton.next_power_of_2(qjl_dim) if use_qjl else 1

    # Dummy pointers for the USE_QJL=False case; the kernel never
    # dereferences them (constexpr-guarded branch).
    qjl_sign_buf = cache_qjl_sign if use_qjl else cache_idx
    rnorm_buf = cache_rnorm if use_qjl else cache_norm
    S_buf = state.S if use_qjl else state.H  # any same-dtype tensor works

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
    """Quantize K into the paged caches via a single Triton kernel."""
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
) -> None:
    """Quantize V via Q_mse only (no QJL). Same kernel, USE_QJL=False."""
    _rotate_and_store(
        new_v,
        cache_v_idx,
        cache_v_norm,
        None,
        None,
        slot_mapping,
        state,
        block_size,
        use_qjl=False,
    )
