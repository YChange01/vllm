# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""TurboQuant codebook, Hadamard rotation, and QJL matrix.

Implements the primitives of Zandieh et al., arXiv:2504.19874:

    Q_mse  (Algorithm 1): b-bit Lloyd-Max quantizer applied to each
                          coordinate of ``H * diag(signs) * x``.
    Q_prod (Algorithm 2): (b-1)-bit Q_mse + 1-bit QJL on the residual,
                          yielding an unbiased inner-product estimator
                          (Lemma 4).
"""

from __future__ import annotations

import math

import numpy as np
import torch


# -------------------------------------------------------------------------
# Lloyd-Max codebook
# -------------------------------------------------------------------------
def _lloyd_max_gaussian(
    bits: int,
    n_iter: int = 100,
    n_samples: int = 200_000,
    seed: int = 0,
) -> np.ndarray:
    """Lloyd-Max iteration. Returns 2^bits MSE-optimal centroids, sorted.

    Samples are drawn from N(0, 1). After Hadamard rotation of a vector
    scaled to ||k_normed|| = sqrt(d), each rotated coordinate has
    variance ~1, so the N(0, 1) codebook matches the quantization
    domain.
    """
    K = 2**bits
    rng = np.random.default_rng(seed)
    samples = rng.standard_normal(n_samples)
    edges = np.quantile(samples, np.linspace(0, 1, K + 1))
    centroids = 0.5 * (edges[:-1] + edges[1:])
    for _ in range(n_iter):
        boundaries = 0.5 * (centroids[:-1] + centroids[1:])
        idx = np.digitize(samples, boundaries)
        for k in range(K):
            mask = idx == k
            if mask.any():
                centroids[k] = samples[mask].mean()
    return np.sort(centroids)


# -------------------------------------------------------------------------
# Hadamard rotation
# -------------------------------------------------------------------------
def _hadamard_matrix(d: int, dtype: torch.dtype) -> torch.Tensor:
    """Normalized Walsh-Hadamard matrix of shape (d, d). Symmetric."""
    if d & (d - 1) != 0:
        raise ValueError(f"head_dim must be a power of 2, got {d}")
    H = torch.tensor([[1.0]], dtype=dtype)
    while H.shape[0] < d:
        H = torch.cat(
            [torch.cat([H, H], dim=1), torch.cat([H, -H], dim=1)], dim=0
        )
    return H / math.sqrt(d)


# -------------------------------------------------------------------------
# Per-layer state
# -------------------------------------------------------------------------
class QuantState:
    """All tensors needed to quantize / dequantize one attention layer.

    Parameters
    ----------
    algo : "mse" | "prod"
        Algorithm 1 or 2 from the paper.
    bits : int
        TOTAL bit budget per coordinate. For ``prod``, 1 bit is reserved
        for the QJL residual so the Lloyd-Max codebook has 2^(bits-1)
        entries. For ``mse`` the codebook has 2^bits entries.
    head_dim : int
        Head dimension d. Must be a power of 2.
    seed : int
        Layer-distinct seed used for H/signs/S so rotations differ
        across attention layers.
    """

    def __init__(
        self,
        algo: str,
        bits: int,
        head_dim: int,
        seed: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        if algo not in ("mse", "prod"):
            raise ValueError(f"algo must be 'mse' or 'prod', got {algo!r}")
        if algo == "prod" and bits < 2:
            raise ValueError(
                f"prod requires bits >= 2 (1 bit reserved for QJL); got {bits}"
            )

        self.algo = algo
        self.bits = bits
        self.head_dim = head_dim
        self.main_bits = bits - 1 if algo == "prod" else bits

        # Lloyd-Max codebook + half-point boundaries.
        centroids = _lloyd_max_gaussian(self.main_bits)
        self.codebook = torch.tensor(centroids, dtype=dtype, device=device)
        self.boundaries = 0.5 * (self.codebook[:-1] + self.codebook[1:])

        # Random orthogonal rotation Pi = H @ diag(signs).
        self.H = _hadamard_matrix(head_dim, dtype).to(device)
        g = torch.Generator().manual_seed(seed)
        self.signs = (
            (torch.randint(0, 2, (head_dim,), generator=g) * 2 - 1)
            .to(dtype)
            .to(device)
        )

        # QJL projection (iid Gaussian) -- only needed by Algorithm 2.
        if algo == "prod":
            g_qjl = torch.Generator().manual_seed(seed ^ 0x51EE5B0)
            self.S: torch.Tensor | None = (
                torch.randn(head_dim, head_dim, generator=g_qjl)
                .to(dtype)
                .to(device)
            )
        else:
            self.S = None
