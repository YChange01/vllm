# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""TurboQuant codebook, random orthogonal rotation, and QJL projection.

Paper-faithful implementation of Zandieh et al., arXiv:2504.19874:

    Q_mse  (Algorithm 1): b-bit Lloyd-Max quantizer applied to each
                          coordinate of ``Pi @ (x / ||x||)``, where
                          Pi is a uniformly random orthogonal matrix
                          generated via QR decomposition of an iid
                          Gaussian matrix (paper Section 3.1).

    Q_prod (Algorithm 2): (b-1)-bit Q_mse + 1-bit QJL on the residual,
                          yielding an unbiased inner-product estimator
                          (paper Theorem 2 / Lemma 4).

The input vector is normalized to the unit sphere before rotation so
that each rotated coordinate follows the Beta distribution of Lemma 1,
f_X(x) = Gamma(d/2) / (sqrt(pi) * Gamma((d-1)/2)) * (1 - x^2)^((d-3)/2),
on support [-1, 1]. Lloyd-Max centroids are trained on samples from
this exact Beta distribution -- sampled as the first coordinate of a
uniformly random unit vector in R^d -- so they inherit the paper's
per-dimension scale (|c_k| ~ 1/sqrt(d) for the moderately-high-d
regime highlighted in Section 3.1).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch

if TYPE_CHECKING:
    pass


# -------------------------------------------------------------------------
# Lloyd-Max codebook trained on the Lemma 1 Beta distribution
# -------------------------------------------------------------------------
def _sample_unit_sphere_coord(
    head_dim: int, n_samples: int, rng: np.random.Generator
) -> np.ndarray:
    """Draw n_samples from the paper's f_X distribution (Lemma 1).

    f_X is the distribution of a single coordinate of a uniformly
    random point on the unit hypersphere S^{d-1}. We realize it by
    sampling z ~ N(0, I_d), normalizing to unit length, and taking
    the first coordinate. No Gaussian approximation is used here --
    for d=32 this is noticeably different from N(0, 1/d).
    """
    z = rng.standard_normal(size=(n_samples, head_dim))
    z /= np.linalg.norm(z, axis=1, keepdims=True)
    return z[:, 0]


def _lloyd_max_beta(
    bits: int,
    head_dim: int,
    n_iter: int = 200,
    n_samples: int = 400_000,
    seed: int = 0,
) -> np.ndarray:
    """Return 2**bits MSE-optimal centroids for the Lemma 1 Beta density.

    The paper's Algorithm 1 trains the scalar quantizer by solving the
    continuous k-means problem (Eq. 4) over f_X. We approximate that
    integral with a Monte Carlo sample and run the standard Lloyd
    iteration. 400k samples + 200 iterations is empirically enough to
    converge the centroids to within 1e-5 on [-1, 1].
    """
    K = 2**bits
    rng = np.random.default_rng(seed)
    samples = _sample_unit_sphere_coord(head_dim, n_samples, rng)

    # Initialize centroids at the quantile partition of the samples.
    edges = np.quantile(samples, np.linspace(0.0, 1.0, K + 1))
    centroids = 0.5 * (edges[:-1] + edges[1:])

    for _ in range(n_iter):
        boundaries = 0.5 * (centroids[:-1] + centroids[1:])
        idx = np.digitize(samples, boundaries)
        new_centroids = centroids.copy()
        for k in range(K):
            mask = idx == k
            if mask.any():
                new_centroids[k] = samples[mask].mean()
        # Stop on convergence; avoids drift from empty bins.
        if np.allclose(new_centroids, centroids, atol=1e-7):
            centroids = new_centroids
            break
        centroids = new_centroids

    return np.sort(centroids)


# -------------------------------------------------------------------------
# Random orthogonal matrix via QR decomposition of a Gaussian matrix
# -------------------------------------------------------------------------
def _random_orthogonal(
    d: int, dtype: torch.dtype, device: torch.device, seed: int
) -> torch.Tensor:
    """Paper Section 3.1: Pi = Q from QR(randn(d, d)).

    The resulting Q is uniformly distributed on O(d) (Haar measure);
    we fix the column signs by the diagonal of R so the distribution
    is uniform on SO(d) -- equivalent up to a deterministic reflection
    and keeps the rotation reproducible across re-invocations.
    """
    gen = torch.Generator().manual_seed(seed)
    M = torch.randn(d, d, generator=gen, dtype=torch.float32)
    Q, R = torch.linalg.qr(M)
    sign = torch.sign(torch.diagonal(R))
    # Zero diagonals are impossible for almost-surely non-singular M,
    # but guard anyway.
    sign = torch.where(sign == 0, torch.ones_like(sign), sign)
    Q = Q * sign.unsqueeze(0)
    return Q.to(dtype=dtype, device=device)


# -------------------------------------------------------------------------
# Per-layer state
# -------------------------------------------------------------------------
class QuantState:
    """Per-attention-layer rotations, codebook, and QJL projection.

    Parameters
    ----------
    algo : "mse" | "prod"
        Algorithm 1 or 2 from Zandieh et al., arXiv:2504.19874.
    bits : int
        Total bit budget per coordinate. For ``prod``, one bit is
        reserved for the QJL residual so the Lloyd-Max codebook holds
        ``2**(bits-1)`` entries; for ``mse`` it holds ``2**bits``.
    head_dim : int
        Quantization dimension d. Any positive integer is supported
        since we use QR(Gaussian) instead of Hadamard.
    seed : int
        Layer-distinct seed. Different seeds -> different Pi and S
        across layers.

    Attributes
    ----------
    codebook : (K,) tensor
        Lloyd-Max centroids in [-1, 1], trained on the Lemma 1 Beta
        density f_X for dimension ``head_dim``.
    boundaries : (K-1,) tensor
        Midpoints of consecutive centroids; used by the store kernel
        as searchsorted bin edges.
    Pi : (d, d) tensor
        Random orthogonal rotation matrix, paper Section 3.1.
    Pi_T : (d, d) tensor
        Cached transpose of Pi (the inverse). Used by the attend
        kernel to un-rotate the accumulated V output. Stored
        contiguously so downstream matmuls see a fresh layout.
    S : (d, d) tensor, prod only
        Independent iid-N(0, 1) QJL projection matrix (Definition 1).
    pack_bits : int
        Storage bits per coordinate for the Lloyd-Max idx. Always a
        power of two >= main_bits so the kernel can read whole bytes:
        main_bits=1 -> pack=1, =2 -> 2, =3 -> 4 (wastes 1 bit),
        =4 -> 4, =5..8 -> 8. The wasted bits are unavoidable unless
        we cross byte boundaries in the kernel.
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
        self.pack_bits = _pow2_ceil(self.main_bits)
        if self.pack_bits not in (1, 2, 4, 8):
            raise ValueError(
                f"pack_bits must be 1, 2, 4, or 8; got {self.pack_bits} "
                f"from main_bits={self.main_bits}"
            )
        if (head_dim * self.pack_bits) % 8 != 0:
            raise ValueError(
                f"head_dim * pack_bits must be divisible by 8 for "
                f"byte-aligned storage; got head_dim={head_dim} "
                f"pack_bits={self.pack_bits}"
            )

        centroids = _lloyd_max_beta(self.main_bits, head_dim)
        self.codebook = torch.tensor(centroids, dtype=dtype, device=device)
        self.boundaries = 0.5 * (self.codebook[:-1] + self.codebook[1:])

        self.Pi = _random_orthogonal(head_dim, dtype, device, seed)
        self.Pi_T = self.Pi.t().contiguous()

        if algo == "prod":
            g_qjl = torch.Generator().manual_seed(seed ^ 0x51EE5B0)
            self.S: torch.Tensor | None = (
                torch.randn(head_dim, head_dim, generator=g_qjl)
                .to(dtype)
                .to(device)
            )
        else:
            self.S = None


def _pow2_ceil(n: int) -> int:
    """Smallest power of two >= max(1, n)."""
    if n <= 1:
        return 1
    return 1 << (n - 1).bit_length()
