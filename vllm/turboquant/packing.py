# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Int4 nibble packing helpers.

Two 4-bit indices are packed into one uint8 byte:
    low nibble  = idx[even positions]
    high nibble = idx[odd positions]

A key of head_dim=128 therefore compresses to 64 bytes (vs 256 bytes for fp16),
giving a 4x memory saving once the K cache is reshaped accordingly.
"""

from __future__ import annotations

import torch


def pack_int4(idx: torch.Tensor) -> torch.Tensor:
    """idx: (..., d) uint8 ∈ [0, 16) -> packed (..., d/2) uint8.

    要求 d 是偶数. 低 nibble = idx[2i], 高 nibble = idx[2i+1].
    """
    if idx.dtype != torch.uint8:
        idx = idx.to(torch.uint8)
    if idx.shape[-1] % 2 != 0:
        raise ValueError(f"last dim must be even, got {idx.shape[-1]}")
    idx = idx & 0x0F  # 安全裁剪
    low = idx[..., 0::2]
    high = idx[..., 1::2]
    return low | (high << 4)


def unpack_int4(packed: torch.Tensor, d: int) -> torch.Tensor:
    """packed: (..., d/2) uint8 -> (..., d) uint8. 逆操作."""
    if packed.shape[-1] * 2 != d:
        raise ValueError(
            f"packed last dim {packed.shape[-1]} != d/2 for d={d}"
        )
    low = packed & 0x0F
    high = (packed >> 4) & 0x0F
    out_shape = packed.shape[:-1] + (d,)
    out = torch.empty(out_shape, dtype=torch.uint8, device=packed.device)
    out[..., 0::2] = low
    out[..., 1::2] = high
    return out


def cache_bytes_per_key(head_dim: int, bits: int) -> int:
    """每个 key 的存储字节数."""
    bits_total = head_dim * bits
    if bits_total % 8 != 0:
        raise ValueError(f"head_dim * bits must be multiple of 8")
    return bits_total // 8
