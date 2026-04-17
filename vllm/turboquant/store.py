# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""K quantization store for TurboQuant.

Pure PyTorch on the same device as the input. Implements the storage
side of paper arXiv:2504.19874 (Zandieh et al.):

    Algorithm 1 (Q_mse):  rotated = H @ diag(signs) @ k_normed
                          idx     = Lloyd-Max bucket of rotated
                          stored  = (idx, ||k||)

    Algorithm 2 (Q_prod): on top of Q_mse, also store the 1-bit QJL
                          sign of the per-coord residual:
                          r        = rotated - codebook[idx]
                          qjl_sign = sign(S @ r_unit)
                          stored   = (idx, ||k||, qjl_sign, ||r||)

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

    T, H_kv, d = new_k.shape
    K_CB = int(state.codebook.shape[0])

    # Compute in fp32 for numerical safety, on the K device.
    k_f = new_k.float()
    H_f = state.H.to(torch.float32)
    signs_f = state.signs.to(torch.float32)
    boundaries_f = state.boundaries.to(torch.float32).contiguous()

    # ||k|| per (token, head); clamp away the singular case.
    k_norm = k_f.norm(dim=-1).clamp_min(1e-6)              # (T, H_kv)

    # k_normed has ||.|| == sqrt(d) per (token, head). The codebook is
    # trained on N(0, 1) per coord, which matches per-coord variance ≈ 1
    # of a sqrt(d)-norm vector after Hadamard rotation.
    k_normed = k_f * (math.sqrt(float(d)) / k_norm.unsqueeze(-1))

    # rotated[..., i] = sum_j H[i, j] * (signs * k_normed)[..., j]
    #                 = (H @ (signs * k_normed))[..., i]
    # In row-vec form on the last dim: (k_normed * signs) @ H.T
    rotated = (k_normed * signs_f) @ H_f.T                  # (T, H_kv, d)

    # Lloyd-Max bucket: idx = #(boundaries < rotated). Equivalent to
    # torch.searchsorted with default right=False on a sorted sequence.
    idx32 = torch.searchsorted(boundaries_f, rotated.contiguous())
    idx_uint8 = idx32.clamp(0, K_CB - 1).to(torch.uint8)    # (T, H_kv, d)

    if use_qjl:
        codebook_f = state.codebook.to(torch.float32)
        S_f = state.S.to(torch.float32)
        rk = codebook_f[idx32.clamp(0, K_CB - 1)]           # (T, H_kv, d)
        r = rotated - rk
        r_norm = r.norm(dim=-1).clamp_min(1e-6)             # (T, H_kv)
        r_unit = r / r_norm.unsqueeze(-1)
        # qjl_raw[..., i] = sum_j S[i, j] * r_unit[..., j] = (S @ r_unit)[..., i]
        qjl_raw = r_unit @ S_f.T                            # (T, H_kv, d)
        qjl_sign = torch.where(
            qjl_raw >= 0,
            qjl_raw.new_ones(()),
            -qjl_raw.new_ones(()),
        ).to(torch.int8)                                    # (T, H_kv, d)

    # Scatter into paged caches via slot_mapping. Skip padded (-1) slots.
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
