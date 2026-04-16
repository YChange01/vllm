# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Lloyd-Max codebook, Hadamard rotation and QJL matrix utilities.

This file implements the primitives for TurboQuant Algorithm 2
(``TurboQuant_prod``) from Zandieh et al., arXiv:2504.19874:

  1. random orthogonal rotation Pi (Hadamard with random ±1 signs) that
     makes coordinates approximately iid Gaussian in high d;
  2. an MSE-optimal scalar quantizer of bit-width ``b - 1`` per coordinate;
  3. a 1-bit QJL (Quantized Johnson-Lindenstrauss) quantizer applied to
     the residual. QJL is a separate d×d iid-Gaussian matrix ``S``;
     sign(S @ r_unit) is an unbiased 1-bit estimator of inner products.

All tensors are built on-device so Triton kernels can read them directly.
Each attention layer owns its own ``GaussianCodebook`` with a distinct seed,
so that H, signs, S, and the codebook differ per layer.
"""

from __future__ import annotations

import math

import numpy as np
import torch


def _lloyd_max_gaussian_numpy(
    bits: int, n_iter: int = 100, n_samples: int = 200_000, seed: int = 0
) -> np.ndarray:
    """Lloyd-Max iteration. Returns 2^bits MSE-optimal codewords, sorted.

    Samples from a standard normal because, after a random orthogonal
    rotation of a vector scaled to ||k_normed|| = sqrt(d), each coordinate
    has std ≈ 1. (The paper works with ||x||=1 and N(0, 1/d) per-coord;
    we absorb the sqrt(d) into the per-key norm so the codebook domain is
    unit-variance.)
    """
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
    """Return a ``(K,)`` codebook tensor on the given device."""
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
    """Return ``(H, signs)`` for the random orthogonal rotation Pi.

    - H: normalized Hadamard matrix, shape ``(d, d)``
    - signs: ``{+1, -1}^d``, shape ``(d,)``

    Forward:  ``Pi x  = (signs * x) @ H.T``
    Inverse:  ``Pi^T x = (x @ H) * signs``
    """
    g = torch.Generator().manual_seed(seed)
    H = _hadamard_matrix(d, dtype).to(device)
    signs = (torch.randint(0, 2, (d,), generator=g) * 2 - 1).to(dtype).to(device)
    return H, signs


def build_qjl_matrix(
    d: int,
    seed: int,
    dtype: torch.dtype,
    device: str | torch.device,
) -> torch.Tensor:
    """Return the QJL projection matrix S: (d, d) with iid N(0, 1) entries.

    TurboQuant paper Definition 1 / Lemma 4: ``Q_qjl(x) = sign(S @ x)`` is
    an unbiased 1-bit estimator of inner products when S is iid standard
    normal. We use the paper's exact construction -- dense Gaussian, not
    the Fast-JL / structured-H substitute -- so the unbiasedness guarantee
    applies directly to Qwen3 K vectors.

    Memory is negligible: for d=128 this is 128² * 4 bytes = 64 KiB per
    layer, allocated once at startup.
    """
    # Derive a distinct subgenerator seed so S is independent of H / signs.
    g = torch.Generator().manual_seed(seed ^ 0x51E_E5B0)
    S = torch.randn(d, d, generator=g).to(dtype).to(device)
    return S


class GaussianCodebook:
    """Per-layer quantization state for TurboQuant Algorithm 2.

    Parameters
    ----------
    head_dim : int
        Head dimension d. Must be a power of two (for Hadamard).
    bits : int
        TOTAL bit budget per coordinate. One bit is spent on QJL so the
        MSE codebook has ``2^(bits - 1)`` levels.
    seed : int
        Layer-distinct seed. Used to derive H / signs / S independently.

    Attributes (all on ``device``)
    -----------------------------
    codebook     : (2^main_bits,) fp16/bf16
    boundaries   : (2^main_bits - 1,) fp16/bf16  -- midpoints for bucketize
    H            : (d, d) fp16/bf16   -- normalized Hadamard
    signs        : (d,)    fp16/bf16  -- ±1, first-stage rotation signs
    S            : (d, d) fp16/bf16   -- QJL iid-Gaussian projection
    """

    def __init__(
        self,
        head_dim: int,
        bits: int,
        seed: int,
        dtype: torch.dtype,
        device: str | torch.device,
    ) -> None:
        if bits < 2:
            raise ValueError(
                f"TurboQuant Algorithm 2 needs bits >= 2 (one reserved for QJL); "
                f"got bits={bits}"
            )

        self.head_dim = head_dim
        self.bits = bits
        self.main_bits = bits - 1   # (b - 1) bits for MSE, 1 bit for QJL
        self.dtype = dtype
        self.device = torch.device(device)

        self.codebook = build_codebook(self.main_bits, dtype=dtype, device=device)
        self.H, self.signs = build_rotation(
            head_dim, seed=seed, dtype=dtype, device=device
        )
        self.S = build_qjl_matrix(head_dim, seed=seed, dtype=dtype, device=device)

        cb = self.codebook
        self.boundaries = 0.5 * (cb[:-1] + cb[1:])

    def rotate(self, x: torch.Tensor) -> torch.Tensor:
        """``x: (..., d) -> Pi x`` (keeps dtype and device)."""
        return (x * self.signs) @ self.H.T

    def quantize_to_idx(self, x: torch.Tensor) -> torch.Tensor:
        """``x: (..., d) -> idx: (..., d) uint8`` via rotate-then-bucketize.

        Indices are in [0, 2^main_bits - 1] so they fit in uint8 for any
        ``bits <= 9``.
        """
        rx = self.rotate(x)
        idx = torch.bucketize(rx.contiguous(), self.boundaries)
        return idx.to(torch.uint8)


class RandomRotation:
    """Standalone random rotation, shared by Triton kernels and PyTorch reference."""

    def __init__(self, d: int, seed: int, dtype: torch.dtype, device: str | torch.device):
        self.d = d
        self.H, self.signs = build_rotation(d, seed=seed, dtype=dtype, device=device)

    def apply(self, x: torch.Tensor) -> torch.Tensor:
        return (x * self.signs) @ self.H.T

    def apply_inv(self, x: torch.Tensor) -> torch.Tensor:
        return (x @ self.H) * self.signs
