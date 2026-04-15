# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Lloyd-Max codebook and Hadamard rotation utilities for TurboQuant.

Data structures are designed to live on-device so Triton kernels can read them
directly. The codebook and rotation are built once at model startup, once per
attention layer (distinct seeds).
"""

from __future__ import annotations

import math

import numpy as np
import torch


def _lloyd_max_gaussian_numpy(
    bits: int, n_iter: int = 100, n_samples: int = 200_000, seed: int = 0
) -> np.ndarray:
    """Lloyd-Max 迭代, 返回 2^bits 个 MSE 最优码字, 升序排列."""
    K = 2 ** bits
    rng = np.random.default_rng(seed)
    samples = rng.standard_normal(n_samples)
    edges = np.quantile(samples, np.linspace(0, 1, K + 1))
    c = 0.5 * (edges[:-1] + edges[1:])
    for _ in range(n_iter):
        boundaries = 0.5 * (c[:-1] + c[1:])
        idx = np.digitize(samples, boundaries)
        for k in range(K):
            mask = idx == k
            if mask.any():
                c[k] = samples[mask].mean()
    return np.sort(c)


def build_codebook(
    bits: int, dtype: torch.dtype = torch.float16, device: str | torch.device = "cpu"
) -> torch.Tensor:
    """返回形状 (K,) 的码本张量."""
    c = _lloyd_max_gaussian_numpy(bits)
    return torch.tensor(c, dtype=dtype, device=device)


def _hadamard_matrix(d: int, dtype: torch.dtype) -> torch.Tensor:
    if d & (d - 1) != 0:
        raise ValueError(f"head_dim must be power of 2, got {d}")
    H = torch.tensor([[1.0]], dtype=dtype)
    while H.shape[0] < d:
        H = torch.cat([torch.cat([H, H], dim=1), torch.cat([H, -H], dim=1)], dim=0)
    return H / math.sqrt(d)


def build_rotation(
    d: int,
    seed: int = 0,
    dtype: torch.dtype = torch.float16,
    device: str | torch.device = "cpu",
) -> tuple[torch.Tensor, torch.Tensor]:
    """返回 (H, signs).

    - H: 归一化 Hadamard 矩阵, shape (d, d)
    - signs: {±1}^d, shape (d,)

    完整旋转 Π x = (signs ⊙ x) @ H^T
    反旋转  Π^T x = (x @ H) ⊙ signs
    """
    g = torch.Generator().manual_seed(seed)
    H = _hadamard_matrix(d, dtype).to(device)
    signs = (torch.randint(0, 2, (d,), generator=g) * 2 - 1).to(dtype).to(device)
    return H, signs


class GaussianCodebook:
    """Per-layer quantization state (rotation + codebook).

    Holds on-device tensors that can be passed directly to Triton kernels.
    """

    def __init__(
        self,
        head_dim: int,
        bits: int,
        seed: int,
        dtype: torch.dtype,
        device: str | torch.device,
    ) -> None:
        self.head_dim = head_dim
        self.bits = bits
        self.dtype = dtype
        self.device = torch.device(device)

        self.codebook = build_codebook(bits, dtype=dtype, device=device)
        self.H, self.signs = build_rotation(head_dim, seed=seed, dtype=dtype, device=device)

        cb = self.codebook
        self.boundaries = 0.5 * (cb[:-1] + cb[1:])

    def rotate(self, x: torch.Tensor) -> torch.Tensor:
        """``x: (..., d) -> Pi x`` (keeps dtype and device)."""
        return (x * self.signs) @ self.H.T

    def quantize_to_idx(self, x: torch.Tensor) -> torch.Tensor:
        """``x: (..., d) -> idx: (..., d) uint8`` via rotate-then-bucketize."""
        rx = self.rotate(x)
        idx = torch.bucketize(rx.contiguous(), self.boundaries)
        return idx.to(torch.uint8)


class RandomRotation:
    """独立的随机旋转, 供 Triton kernel 和 PyTorch 验证共享."""

    def __init__(self, d: int, seed: int, dtype: torch.dtype, device: str | torch.device):
        self.d = d
        self.H, self.signs = build_rotation(d, seed=seed, dtype=dtype, device=device)

    def apply(self, x: torch.Tensor) -> torch.Tensor:
        return (x * self.signs) @ self.H.T

    def apply_inv(self, x: torch.Tensor) -> torch.Tensor:
        return (x @ self.H) * self.signs
