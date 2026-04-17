# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""K/V quantization store for TurboQuant.

Pure PyTorch on the same device as the input. Implements the storage
side of paper arXiv:2504.19874 (Zandieh et al.):

    Algorithm 1 (Q_mse)  : rotated = H @ diag(signs) @ x_normed
                           idx     = Lloyd-Max bucket of rotated
                           stored  = (idx, ||x||)

    Algorithm 2 (Q_prod) : on top of Q_mse, also store the 1-bit QJL
                           sign of the per-coord residual:
                               r        = rotated - codebook[idx]
                               qjl_sign = sign(S @ r_unit)
                               stored   = (idx, ||x||, qjl_sign, ||r||)

K is quantized via the chosen algorithm (mse or prod). V is always
quantized via Q_mse only -- QJL targets unbiased inner-product
estimates, but attention's V step is a weighted sum where direct
reconstruction matters and the extra QJL bit doesn't help.

K and V share the same ``QuantState`` (same H, signs, codebook). This
saves memory and keeps reconstruction symmetric; per-vector Lloyd-Max
quantization decorrelates the noise within each layer regardless of
whether K and V share the rotation.

Why pure PyTorch and not Triton: a Triton store kernel was previously
used here, but on B200 multi-token grids it silently corrupted the V
cache (and likely the QJL caches) -- prefill (num_tokens > 1) wrote
K-derived values into V slots while decode (num_tokens == 1) wrote
correctly. PyTorch tensor ops on GPU avoid the issue at the cost of a
slightly slower prefill; attend (the hot decode path) is still Triton.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from vllm.turboquant.codebook import QuantState


def _quantize_lloyd_max(
    x: torch.Tensor, state: "QuantState"
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run Q_mse on ``x``: norm + Hadamard + Lloyd-Max bucket.

    Returns
    -------
    idx_uint8 : (..., d) uint8     Lloyd-Max bucket per coord
    x_norm    : (...,)   fp32      ||x|| per (token, head)
    rotated   : (..., d) fp32      H @ (signs * x_normed)
    idx32     : (..., d) int64     same as idx_uint8 but int64 (for
                                   downstream codebook gather, e.g. QJL)
    """
    d = x.shape[-1]
    K_CB = int(state.codebook.shape[0])

    x_f = x.float()
    H_f = state.H.to(torch.float32)
    signs_f = state.signs.to(torch.float32)
    boundaries_f = state.boundaries.to(torch.float32).contiguous()

    x_norm = x_f.norm(dim=-1).clamp_min(1e-6)
    x_normed = x_f * (math.sqrt(float(d)) / x_norm.unsqueeze(-1))
    rotated = (x_normed * signs_f) @ H_f.T

    idx32 = torch.searchsorted(boundaries_f, rotated.contiguous())
    idx32 = idx32.clamp(0, K_CB - 1)
    idx_uint8 = idx32.to(torch.uint8)
    return idx_uint8, x_norm, rotated, idx32


def _scatter_paged(
    cache: torch.Tensor,
    values: torch.Tensor,
    slot_mapping: torch.Tensor,
    block_size: int,
) -> None:
    """Scatter ``values`` into ``cache`` via paged ``slot_mapping``.

    ``cache``  : (num_blocks, block_size, ...) tensor.
    ``values`` : (T, ...) where T == slot_mapping.shape[0].
    Skips ``slot_mapping[i] < 0`` (padding).
    """
    valid = slot_mapping >= 0
    if not bool(valid.any()):
        return
    slots = slot_mapping[valid].to(torch.int64)
    b_idx = slots // block_size
    off = slots % block_size
    cache[b_idx, off] = values[valid]


def _maybe_pack_4bit(idx_uint8: torch.Tensor, K_CB: int) -> torch.Tensor:
    """Pack two 4-bit indices per byte iff ``K_CB <= 16``.

    Packs consecutive even/odd entries along the last dim:

        packed[..., i] = idx[..., 2i] | (idx[..., 2i + 1] << 4)

    Returns the input unchanged when ``K_CB > 16``.
    """
    if K_CB > 16:
        return idx_uint8
    assert idx_uint8.shape[-1] % 2 == 0, (
        f"last dim {idx_uint8.shape[-1]} must be even for 4-bit packing"
    )
    low = idx_uint8[..., 0::2]
    high = idx_uint8[..., 1::2]
    return (low | (high << 4)).to(torch.uint8)


def _pack_qjl_sign(qjl_sign: torch.Tensor) -> torch.Tensor:
    """Pack 8 QJL sign bits per byte along the last dim.

    Encoding: bit_j = (sign_j < 0). So +1 -> 0, -1 -> 1. Inverse in attend:
    ``sign = 1 - 2 * bit``. Last dim must be divisible by 8.

    Returns a (..., d / 8) uint8 tensor.
    """
    d = qjl_sign.shape[-1]
    assert d % 8 == 0, f"last dim {d} must be divisible by 8 for QJL bit-pack"
    bits = (qjl_sign < 0).to(torch.int32)
    prefix = bits.shape[:-1]
    bits = bits.view(*prefix, d // 8, 8)
    weights = 1 << torch.arange(8, device=bits.device, dtype=torch.int32)
    return (bits * weights).sum(dim=-1).to(torch.uint8)


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
    """Quantize one batch of K vectors into the paged caches.

    Parameters
    ----------
    new_k : (T, H_kv, d) fp16/bf16
        Per-token per-kv-head K vectors from this forward call.
    cache_k_idx : (num_blocks, block_size, H_kv, d) uint8
        Lloyd-Max bucket indices, written via slot_mapping.
    cache_k_norm : (num_blocks, block_size, H_kv) fp32
        Per (slot, head) ||k|| used to rescale at attend time.
    slot_mapping : (T,) int64
        Physical slot index per token; -1 means "skip" (padding).
    state : QuantState
        Holds the codebook, boundaries, Hadamard H, signs, and QJL S.
    block_size : int
        Tokens per block in the paged cache.
    cache_k_qjl_sign, cache_k_rnorm : optional
        Required when ``state.algo == "prod"``; ignored for "mse".
    """
    if new_k.shape[0] == 0:
        return

    use_qjl = state.algo == "prod"
    if use_qjl:
        assert cache_k_qjl_sign is not None and cache_k_rnorm is not None, (
            "prod requires cache_k_qjl_sign and cache_k_rnorm"
        )

    K_CB = int(state.codebook.shape[0])
    idx_uint8, k_norm, rotated, idx32 = _quantize_lloyd_max(new_k, state)
    idx_packed = _maybe_pack_4bit(idx_uint8, K_CB)
    _scatter_paged(cache_k_idx, idx_packed, slot_mapping, block_size)
    _scatter_paged(cache_k_norm, k_norm, slot_mapping, block_size)

    if use_qjl:
        codebook_f = state.codebook.to(torch.float32)
        S_f = state.S.to(torch.float32)
        # QJL is computed against the *unpacked* idx, so use the original
        # int64 idx32 we got from _quantize_lloyd_max.
        rk = codebook_f[idx32]                                  # (T, H_kv, d)
        r = rotated - rk
        r_norm = r.norm(dim=-1).clamp_min(1e-6)
        r_unit = r / r_norm.unsqueeze(-1)
        # qjl_raw[..., i] = sum_j S[i, j] * r_unit[..., j] = (S @ r_unit)[..., i]
        qjl_raw = r_unit @ S_f.T
        qjl_sign = torch.where(
            qjl_raw >= 0,
            qjl_raw.new_ones(()),
            -qjl_raw.new_ones(()),
        ).to(torch.int8)
        qjl_packed = _pack_qjl_sign(qjl_sign)
        _scatter_paged(cache_k_qjl_sign, qjl_packed, slot_mapping, block_size)
        _scatter_paged(cache_k_rnorm, r_norm, slot_mapping, block_size)


def turboquant_store_v(
    new_v: torch.Tensor,
    cache_v_idx: torch.Tensor,
    cache_v_norm: torch.Tensor,
    slot_mapping: torch.Tensor,
    state: "QuantState",
    block_size: int,
) -> None:
    """Quantize one batch of V vectors via Q_mse only (no QJL).

    Same H/signs/codebook as K (shared ``state``). Stored per-coord
    Lloyd-Max bucket + per (token, head) ||v||. Reconstruction in attend:
    ``v ≈ signs * (H @ codebook[idx]) * ||v|| / sqrt(d)`` -- the post-
    rotation is applied once per (query, head) in the attend wrapper,
    not per attended KV slot in the kernel.
    """
    if new_v.shape[0] == 0:
        return
    K_CB = int(state.codebook.shape[0])
    idx_uint8, v_norm, _rotated, _idx32 = _quantize_lloyd_max(new_v, state)
    idx_packed = _maybe_pack_4bit(idx_uint8, K_CB)
    _scatter_paged(cache_v_idx, idx_packed, slot_mapping, block_size)
    _scatter_paged(cache_v_norm, v_norm, slot_mapping, block_size)
