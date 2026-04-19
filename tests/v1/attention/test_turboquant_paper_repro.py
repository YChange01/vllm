# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Paper-faithfulness tests for the turboquant-paper-repro branch.

These tests do not require a GPU; they validate the math of the
rotation, codebook, and Q_prod round-trip on pure PyTorch CPU tensors.
Kernel correctness and end-to-end attention parity live in the
existing B200 test suite (test/ directory).

Paper: Zandieh et al., arXiv:2504.19874.
"""

from __future__ import annotations

import math

import pytest
import torch

from vllm.turboquant.codebook import (
    QuantState,
    _lloyd_max_beta,
    _random_orthogonal,
    _sample_unit_sphere_coord,
)


torch.manual_seed(0)


# ---------------------------------------------------------------------------
# Random orthogonal rotation
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("d", [16, 32, 64, 96, 128])
def test_pi_is_orthogonal(d: int) -> None:
    Pi = _random_orthogonal(d, torch.float32, torch.device("cpu"), seed=42)
    assert Pi.shape == (d, d)
    I = Pi @ Pi.t()
    assert torch.allclose(I, torch.eye(d), atol=1e-5), (
        f"Pi not orthogonal at d={d}: max |Pi Pi^T - I| = "
        f"{(I - torch.eye(d)).abs().max().item():.2e}"
    )


def test_pi_is_not_hadamard() -> None:
    """Sanity check that we really switched to QR(Gaussian)."""
    Pi = _random_orthogonal(128, torch.float32, torch.device("cpu"), seed=1)
    # Hadamard entries are +/- 1/sqrt(d). Random orthogonal entries are
    # generally not.
    unique_abs = Pi.abs().unique().numel()
    assert unique_abs > 16, (
        f"Pi only has {unique_abs} unique |values|; this looks like a "
        f"structured matrix, not QR(Gaussian)."
    )


def test_pi_seed_reproducible() -> None:
    Pi_a = _random_orthogonal(64, torch.float32, torch.device("cpu"), seed=7)
    Pi_b = _random_orthogonal(64, torch.float32, torch.device("cpu"), seed=7)
    Pi_c = _random_orthogonal(64, torch.float32, torch.device("cpu"), seed=8)
    assert torch.equal(Pi_a, Pi_b)
    assert not torch.equal(Pi_a, Pi_c)


@pytest.mark.parametrize("d", [32, 96, 128])
def test_pi_supports_non_power_of_two(d: int) -> None:
    # Critical property for Stage 3 outlier split (96 is not 2^n).
    Pi = _random_orthogonal(d, torch.float32, torch.device("cpu"), seed=0)
    I = Pi @ Pi.t()
    assert torch.allclose(I, torch.eye(d), atol=1e-5)


# ---------------------------------------------------------------------------
# Beta-distribution Lloyd-Max codebook
# ---------------------------------------------------------------------------
def test_beta_coord_on_unit_sphere() -> None:
    """Sanity: first coord of unit-sphere samples has mean 0 and variance
    1/d (Lemma 1 limit for moderate d)."""
    rng = __import__("numpy").random.default_rng(0)
    d = 128
    samples = _sample_unit_sphere_coord(d, 100_000, rng)
    assert abs(samples.mean()) < 0.01
    # Variance of Beta-on-hypersphere first coord is 1/d.
    assert abs(samples.var() - 1.0 / d) < 0.001


@pytest.mark.parametrize("bits,d", [(1, 128), (2, 128), (3, 128), (3, 96), (3, 32)])
def test_codebook_sorted_and_in_unit_range(bits: int, d: int) -> None:
    c = _lloyd_max_beta(bits, d)
    assert len(c) == 2**bits
    assert (c[:-1] <= c[1:]).all(), "centroids must be sorted"
    # Paper support is [-1, 1] (Lemma 1 domain).
    assert c.min() > -1.01 and c.max() < 1.01
    # Paper: |c| ~ 1/sqrt(d) for the moderately-high-d regime.
    expected_magnitude = 1.0 / math.sqrt(d)
    median_abs = abs(c).mean()
    assert 0.3 * expected_magnitude < median_abs < 3.0 * expected_magnitude, (
        f"codebook magnitude {median_abs:.3f} far from paper's "
        f"1/sqrt({d}) = {expected_magnitude:.3f}"
    )


def test_codebook_b1_matches_paper_formula() -> None:
    """Paper §3.1 explicit: b=1 centroids approximate {+/- sqrt(2/pi)/sqrt(d)}."""
    d = 128
    c = _lloyd_max_beta(1, d)
    target = math.sqrt(2.0 / math.pi) / math.sqrt(d)
    assert len(c) == 2
    assert c[0] < 0 < c[1]
    # Allow 10% tolerance: Lloyd on Beta vs Normal approximation + MC noise.
    assert abs(c[1] - target) / target < 0.1, (
        f"b=1 centroid {c[1]:.4f} far from paper's sqrt(2/pi)/sqrt(d) = "
        f"{target:.4f}"
    )


# ---------------------------------------------------------------------------
# Q_prod round-trip (Algorithm 2 pure-PyTorch reference)
# ---------------------------------------------------------------------------
def _reference_q_prod(x: torch.Tensor, state: QuantState) -> torch.Tensor:
    """Pure-PyTorch implementation of Algorithm 2 for one batch of vectors.

    Input  : x (N, d)  original-scale.
    Output : x_approx (N, d) reconstruction.
    """
    x_norm = x.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    x_hat = x / x_norm                                   # unit norm
    rotated = x_hat @ state.Pi.to(x.dtype)               # unit sphere
    # Q_mse bit
    idx = torch.searchsorted(state.boundaries.to(x.dtype), rotated)
    idx = idx.clamp(max=state.codebook.shape[0] - 1)
    mse_approx = state.codebook.to(x.dtype)[idx]         # (N, d)
    if state.algo == "mse":
        y_approx = mse_approx
    else:
        # Q_qjl bit on the residual
        r = rotated - mse_approx
        r_norm = r.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        r_unit = r / r_norm
        qjl_raw = r_unit @ state.S.t().to(x.dtype)       # (N, d)
        qjl = qjl_raw.sign()                             # {-1, +1}
        qjl[qjl == 0] = 1.0
        d = state.head_dim
        qjl_coef = math.sqrt(math.pi / 2.0) / d
        y_approx = mse_approx + qjl_coef * r_norm * (qjl @ state.S.to(x.dtype))
    # Un-rotate + rescale.
    return (y_approx @ state.Pi_T.to(x.dtype)) * x_norm


@pytest.mark.parametrize("d", [32, 64, 128])
def test_q_prod_inner_product_is_unbiased(d: int) -> None:
    """Paper Theorem 2: E[<y, x_approx>] = <y, x>."""
    state = QuantState(
        algo="prod", bits=5, head_dim=d, seed=123,
        dtype=torch.float32, device=torch.device("cpu"),
    )
    # Average over many independent qjl draws -> bias -> 0.
    n_trials = 200
    x = torch.randn(1, d) * 3.0
    y = torch.randn(d)
    true_inner = torch.dot(x[0], y)

    # The only random source at dequant time is which QJL sign each
    # coordinate lands on given r_unit; to average over this we would
    # need to resample S, but Algorithm 2's S is fixed per layer. The
    # unbiasedness in the paper is over the randomness of S.
    # So we average over independent QuantState draws (different S).
    approxes = []
    for trial in range(n_trials):
        st = QuantState(
            algo="prod", bits=5, head_dim=d, seed=1000 + trial,
            dtype=torch.float32, device=torch.device("cpu"),
        )
        x_apx = _reference_q_prod(x, st)
        approxes.append(torch.dot(x_apx[0], y).item())

    mean_inner = sum(approxes) / n_trials
    std_err = (
        (sum((a - mean_inner) ** 2 for a in approxes) / n_trials) ** 0.5
        / n_trials ** 0.5
    )
    bias = abs(mean_inner - true_inner.item())
    # Bias should be within ~3 standard errors of zero (99% CI).
    assert bias < 3 * std_err + 1e-3, (
        f"Q_prod inner-product bias {bias:.4f} exceeds 3 * SE "
        f"{3 * std_err:.4f}; true={true_inner.item():.4f} "
        f"mean_approx={mean_inner:.4f}"
    )


@pytest.mark.parametrize("d", [64, 128])
def test_q_mse_reconstruction_bounded(d: int) -> None:
    """Paper Theorem 1: D_mse <= sqrt(3)*pi/2 * 1/4^b for unit-norm x."""
    bits = 4
    state = QuantState(
        algo="mse", bits=bits, head_dim=d, seed=42,
        dtype=torch.float32, device=torch.device("cpu"),
    )
    x = torch.randn(100, d)
    x = x / x.norm(dim=-1, keepdim=True)  # unit sphere
    x_approx = _reference_q_prod(x, state)
    mse = (x - x_approx).pow(2).sum(dim=-1).mean().item()

    # Paper bound for b=4: D_mse ~ 0.009 for unit-norm x.
    # Our Monte-Carlo average should be under the looser Thm 1 bound.
    thm1_bound = math.sqrt(3) * math.pi / 2 * (1.0 / 4.0 ** bits)
    paper_value_b4 = 0.009
    assert mse < thm1_bound, (
        f"Q_mse distortion {mse:.4f} exceeds Thm 1 bound {thm1_bound:.4f}"
    )
    # And within ~3x of the paper's Fig 3 numerical estimate.
    assert mse < 3 * paper_value_b4, (
        f"Q_mse distortion {mse:.4f} is >3x paper's ~0.009 at b=4"
    )
