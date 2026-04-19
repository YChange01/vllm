# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Outlier channel splitting for TurboQuant (paper arXiv:2504.19874 §4.3).

The paper's 2.5-bit / 3.5-bit LongBench results split each head's
channels into an outlier set and a regular set, applying an
independent TurboQuant instance to each. For example 2.5-bit:
    32 outlier channels at b=3  + 96 regular channels at b=2
    -> (32 * 3 + 96 * 2) / 128 = 2.5 bits/coord effective.

This module provides:

* ``OutlierMask`` : loader for the calibration script's .pt output.
  Aggregates the per-(layer, kv_head) stats into one outlier index
  vector per layer (averaging across kv_heads, top-N by mean-abs).
* ``SplitQuantState`` : two-slice container holding independent
  ``QuantState`` instances for the outlier and regular slices, plus
  the channel index tensors used by the store/attend paths to gather
  and scatter.

Per-layer (rather than per-(layer, kv_head)) outlier indices keep the
kernel surface area small and match how QJL [63] selects outliers.
Per-head splitting is a future extension.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from vllm.turboquant.codebook import QuantState


@dataclass(frozen=True)
class OutlierMask:
    """Loaded artifact from ``scripts/calibrate_outliers.py``.

    Per-layer outlier indices for K and V, derived by averaging the
    per-kv_head mean-abs stats across heads then taking top-N.
    """

    model_name: str
    head_dim: int
    num_layers: int
    num_kv_heads: int
    num_outliers: int
    # Shape (num_layers, num_outliers), int64 (torch.long) for gather.
    k_outlier_idx: torch.Tensor
    v_outlier_idx: torch.Tensor

    @classmethod
    def load(cls, path: str | Path) -> "OutlierMask":
        payload = torch.load(path, map_location="cpu", weights_only=False)
        required = {
            "model_name", "head_dim", "num_layers", "num_kv_heads",
            "num_outliers", "k_stats", "v_stats",
        }
        missing = required - set(payload)
        if missing:
            raise ValueError(
                f"Outlier .pt at {path} is missing keys: {sorted(missing)}"
            )
        num_outliers = int(payload["num_outliers"])
        head_dim = int(payload["head_dim"])
        if num_outliers <= 0 or num_outliers >= head_dim:
            raise ValueError(
                f"num_outliers={num_outliers} must be in (0, head_dim="
                f"{head_dim}); check calibration run."
            )

        # (L, H_kv, d) -> (L, d) by mean over heads.
        k_layer_stats = payload["k_stats"].mean(dim=1)
        v_layer_stats = payload["v_stats"].mean(dim=1)
        k_idx = torch.topk(k_layer_stats, num_outliers, dim=-1).indices
        v_idx = torch.topk(v_layer_stats, num_outliers, dim=-1).indices
        # Sorting makes the regular slice a deterministic complement.
        k_idx, _ = k_idx.sort(dim=-1)
        v_idx, _ = v_idx.sort(dim=-1)

        return cls(
            model_name=str(payload["model_name"]),
            head_dim=head_dim,
            num_layers=int(payload["num_layers"]),
            num_kv_heads=int(payload["num_kv_heads"]),
            num_outliers=num_outliers,
            k_outlier_idx=k_idx.to(torch.int64).contiguous(),
            v_outlier_idx=v_idx.to(torch.int64).contiguous(),
        )

    def for_layer(
        self, layer_idx: int, kind: str
    ) -> torch.Tensor:
        if kind == "k":
            return self.k_outlier_idx[layer_idx]
        elif kind == "v":
            return self.v_outlier_idx[layer_idx]
        else:
            raise ValueError(f"kind must be 'k' or 'v', got {kind!r}")


def _regular_from_outlier(
    outlier_idx: torch.Tensor, head_dim: int
) -> torch.Tensor:
    """Channels in [0, head_dim) not in outlier_idx, sorted ascending."""
    mask = torch.ones(head_dim, dtype=torch.bool, device=outlier_idx.device)
    mask[outlier_idx] = False
    return torch.nonzero(mask, as_tuple=False).squeeze(-1).to(torch.int64)


class SplitQuantState:
    """Two independent TurboQuant instances for one layer (one for
    each of K's outlier / regular slices; V is paired separately --
    this class stores only one K/V's worth of state).

    Parameters
    ----------
    algo : "mse" | "prod"
    bits_outlier : int
        Total bit budget per coord for the outlier slice.
    bits_regular : int
        Total bit budget per coord for the regular slice.
    head_dim : int
    outlier_idx : (d_out,) long tensor
        Channel positions in [0, head_dim) to route to the outlier
        slice. Sorted ascending.
    seed : int
    dtype : torch.dtype
    device : torch.device
    """

    def __init__(
        self,
        algo: str,
        bits_outlier: int,
        bits_regular: int,
        head_dim: int,
        outlier_idx: torch.Tensor,
        seed: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        if outlier_idx.dim() != 1:
            raise ValueError("outlier_idx must be 1-D")
        outlier_idx = outlier_idx.to(dtype=torch.int64, device=device)
        if outlier_idx.numel() == 0 or outlier_idx.numel() >= head_dim:
            raise ValueError(
                f"outlier_idx size must be in (0, head_dim={head_dim}); "
                f"got {outlier_idx.numel()}"
            )
        if (outlier_idx < 0).any() or (outlier_idx >= head_dim).any():
            raise ValueError("outlier_idx values must be in [0, head_dim)")
        if torch.unique(outlier_idx).numel() != outlier_idx.numel():
            raise ValueError("outlier_idx must be unique")

        self.algo = algo
        self.head_dim = head_dim
        self.bits_outlier = bits_outlier
        self.bits_regular = bits_regular

        self.outlier_idx = outlier_idx.contiguous()
        self.regular_idx = _regular_from_outlier(
            outlier_idx, head_dim
        ).contiguous()

        d_out = int(self.outlier_idx.numel())
        d_reg = int(self.regular_idx.numel())
        assert d_out + d_reg == head_dim

        self.state_out = QuantState(
            algo=algo, bits=bits_outlier, head_dim=d_out,
            seed=seed ^ 0xA1A1A1A1,
            dtype=dtype, device=device,
        )
        self.state_reg = QuantState(
            algo=algo, bits=bits_regular, head_dim=d_reg,
            seed=seed ^ 0xB2B2B2B2,
            dtype=dtype, device=device,
        )

    @property
    def d_outlier(self) -> int:
        return int(self.outlier_idx.numel())

    @property
    def d_regular(self) -> int:
        return int(self.regular_idx.numel())

    def effective_bits(self) -> float:
        """Effective bits/coord = (d_out * b_out + d_reg * b_reg) / d.

        Paper's §4.3 gives the example "32 outliers at 3 bits + 96
        regular at 2 bits = 2.5 bits effective". The arithmetic
        (32*3 + 96*2)/128 = 2.25, not 2.5 -- the paper has a typo.
        True 2.5-bit is (32 outliers at 4 + 96 regular at 2) = 2.5, or
        (64 outliers at 3 + 64 regular at 2) = 2.5. Pick bit budgets
        to match the advertised effective rate you want.
        """
        return (
            self.d_outlier * self.bits_outlier
            + self.d_regular * self.bits_regular
        ) / self.head_dim
