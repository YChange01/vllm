# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""TurboQuant configuration dataclass.

Single source of truth for the 4 axes of TurboQuant configuration:

    A. Algorithm:        algo (mse | prod) + bits (homog) or
                         bits_outlier + bits_regular (split)
    B. Mode:             outlier_mask_path = "" -> homog; else split
    C. Storage layout:   tight_pack, norm_dtype, rnorm_dtype
    D. Kernel:           use_cuda

Old code reads 9 individual env vars (TURBOQUANT_ALGO, _BITS, ...);
new code constructs a TurboQuantConfig from env or from a stage
registry. Both paths produce the same downstream state.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
from typing import Literal


@dataclass(frozen=True)
class TurboQuantConfig:
    """Frozen, hashable config describing one TurboQuant configuration."""

    # Axis A: algorithm + bit budget
    algo: Literal["mse", "prod"] = "prod"
    bits: int = 4                              # homog total bit budget

    # Axis B: split mode (homog when outlier_mask_path is empty)
    outlier_mask_path: str = ""
    bits_outlier: int = 0                      # used in split mode only
    bits_regular: int = 0

    # Axis C: storage layout flags (orthogonal to A and B)
    tight_pack: bool = False                   # idx + qjl in one nibble
    norm_dtype: Literal["fp32", "fp16"] = "fp32"
    rnorm_dtype: Literal["fp32", "fp16", "uint8"] = "fp32"

    # Axis D: kernel implementation
    use_cuda: bool = False                     # CUDA WMMA vs Triton TC

    # ------------------------------------------------------------------
    # Derived properties
    # ------------------------------------------------------------------
    @property
    def is_split(self) -> bool:
        return bool(self.outlier_mask_path)

    @property
    def is_homog(self) -> bool:
        return not self.is_split

    @property
    def main_bits(self) -> int:
        """Lloyd-Max main codebook bits for homog mode."""
        return self.bits - 1 if self.algo == "prod" else self.bits

    # ------------------------------------------------------------------
    # Env-var round-trip
    # ------------------------------------------------------------------
    @classmethod
    def from_env(
        cls, env: dict[str, str] | None = None
    ) -> "TurboQuantConfig":
        e = env if env is not None else os.environ

        algo = e.get("TURBOQUANT_ALGO", "prod").lower()
        bits = int(e.get("TURBOQUANT_BITS", "4"))
        outlier_mask = e.get("TURBOQUANT_OUTLIER_MASK", "")
        bits_outlier = int(e.get("TURBOQUANT_BITS_OUTLIER", str(bits)))
        bits_regular = int(e.get("TURBOQUANT_BITS_REGULAR", str(bits)))
        tight_pack = e.get("TURBOQUANT_TIGHT_PACK", "0") == "1"
        norm_dtype = (
            "fp16" if e.get("TURBOQUANT_FP16_NORMS", "0") == "1" else "fp32"
        )
        # uint8 rnorm independent of norm dtype; default tracks norm dtype
        # so the legacy "FP16_NORMS=1 alone" path stays fp16/fp16.
        if e.get("TURBOQUANT_UINT8_RNORM", "0") == "1":
            rnorm_dtype: Literal["fp32", "fp16", "uint8"] = "uint8"
        else:
            rnorm_dtype = norm_dtype  # type: ignore[assignment]
        use_cuda = e.get("TURBOQUANT_USE_CUDA", "0") == "1"

        return cls(
            algo=algo, bits=bits,
            outlier_mask_path=outlier_mask,
            bits_outlier=bits_outlier, bits_regular=bits_regular,
            tight_pack=tight_pack,
            norm_dtype=norm_dtype, rnorm_dtype=rnorm_dtype,
            use_cuda=use_cuda,
        )

    def to_env_dict(self) -> dict[str, str]:
        """Inverse of from_env; only sets non-default keys for compactness."""
        d: dict[str, str] = {
            "TURBOQUANT_ALGO": self.algo,
            "TURBOQUANT_BITS": str(self.bits),
        }
        if self.is_split:
            d["TURBOQUANT_OUTLIER_MASK"] = self.outlier_mask_path
            d["TURBOQUANT_BITS_OUTLIER"] = str(self.bits_outlier)
            d["TURBOQUANT_BITS_REGULAR"] = str(self.bits_regular)
        if self.tight_pack:
            d["TURBOQUANT_TIGHT_PACK"] = "1"
        if self.norm_dtype == "fp16":
            d["TURBOQUANT_FP16_NORMS"] = "1"
        if self.rnorm_dtype == "uint8":
            d["TURBOQUANT_UINT8_RNORM"] = "1"
        if self.use_cuda:
            d["TURBOQUANT_USE_CUDA"] = "1"
        return d

    def with_outlier_mask(self, path: str) -> "TurboQuantConfig":
        """Return a new config with outlier_mask_path overridden."""
        return replace(self, outlier_mask_path=path)

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------
    def validate(self) -> None:
        if self.algo not in ("mse", "prod"):
            raise ValueError(f"algo must be 'mse' or 'prod', got {self.algo!r}")
        if self.algo == "prod" and self.bits < 2:
            raise ValueError(
                f"prod requires bits >= 2 (1 bit reserved for QJL); "
                f"got {self.bits}"
            )
        # Tight pack validation: only homog prod b=4 makes sense.
        if self.tight_pack:
            if self.algo != "prod":
                raise ValueError("tight_pack requires algo='prod'")
            if self.bits != 4:
                raise ValueError(
                    f"tight_pack only valid for b=4 prod (main_bits=3); "
                    f"got bits={self.bits}"
                )
            if self.is_split:
                raise ValueError(
                    "tight_pack not supported in split mode (split's b=5+b=3 "
                    "are already clean-packed)"
                )
        # Split mode requires both bits set.
        if self.is_split:
            if self.bits_outlier <= 0 or self.bits_regular <= 0:
                raise ValueError(
                    f"split mode requires bits_outlier > 0 and "
                    f"bits_regular > 0; got out={self.bits_outlier} "
                    f"reg={self.bits_regular}"
                )
            for b, name in [
                (self.bits_outlier, "bits_outlier"),
                (self.bits_regular, "bits_regular"),
            ]:
                main_b = b - 1 if self.algo == "prod" else b
                if main_b < 1 or main_b > 4:
                    raise ValueError(
                        f"{name}={b} -> main_bits={main_b} outside "
                        f"supported range {{1..4}} on this branch."
                    )
        # Norm dtype consistency.
        if self.rnorm_dtype not in ("fp32", "fp16", "uint8"):
            raise ValueError(
                f"rnorm_dtype must be one of fp32/fp16/uint8, got "
                f"{self.rnorm_dtype!r}"
            )
        if self.norm_dtype not in ("fp32", "fp16"):
            raise ValueError(
                f"norm_dtype must be fp32 or fp16, got {self.norm_dtype!r}"
            )

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------
    def summary(self) -> str:
        if self.is_split:
            mode = (
                f"split (b_out={self.bits_outlier}, "
                f"b_reg={self.bits_regular}, mask={self.outlier_mask_path})"
            )
        else:
            mode = f"homog (b={self.bits})"
        flags = []
        if self.tight_pack:
            flags.append("tight")
        if self.norm_dtype == "fp16":
            flags.append("fp16-norm")
        if self.rnorm_dtype == "uint8":
            flags.append("uint8-rnorm")
        if self.use_cuda:
            flags.append("cuda")
        flag_str = (", " + ", ".join(flags)) if flags else ""
        return f"algo={self.algo}, {mode}{flag_str}"
