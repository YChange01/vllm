# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Registry of named TurboQuant stages.

A *stage* is a string label that maps to a fully-specified
``TurboQuantConfig``. Stages are the unit eval / throughput / Figure-4
scripts iterate over via the ``STAGES`` env var, e.g.

    STAGES="TURBOQUANT_b4 TURBOQUANT_split_3_5bit_f"

Naming convention:
  TURBOQUANT_<bit_spec>[_<flags>]

  <bit_spec>:
    b<N>            homog mode, b in {2, 3, 4, 5}
    split_<eff>bit  split mode (<eff> = paper effective bit, e.g. 3_5)

  <flags> (combinable, alphabetical):
    t  tight nibble pack (idx + qjl into one byte; b=4 prod homog only)
    f  fp16 norm + rnorm (saves 50% norm storage)
    u  uint8 rnorm (line ar [0, 2.0]; saves another 50% on rnorm)

Algo defaults to prod; use mse_ prefix for Algorithm 1.

Add a new stage in one place: append to STAGES below.
"""

from __future__ import annotations

import sys

from vllm.turboquant.config import TurboQuantConfig

# Default outlier mask path produced by scripts/calibrate.sh.
DEFAULT_OUTLIER_MASK = "/tmp/outliers_llama-3_1-8b_32.pt"


# Build configs once; reuse references where two names share a config
# (e.g., TURBOQUANT_b4 and TURBOQUANT_prod_b4 are the same thing).
_PROD_B4 = TurboQuantConfig(algo="prod", bits=4)
_PROD_B4_T = TurboQuantConfig(algo="prod", bits=4, tight_pack=True)


STAGES: dict[str, TurboQuantConfig] = {
    # ----- Homog (single QuantState) -----
    "TURBOQUANT_b2":      TurboQuantConfig(algo="prod", bits=2),
    "TURBOQUANT_b3":      TurboQuantConfig(algo="prod", bits=3),
    "TURBOQUANT_b4":      _PROD_B4,
    "TURBOQUANT_b4_t":    _PROD_B4_T,
    "TURBOQUANT_b4_tf":   TurboQuantConfig(
        algo="prod", bits=4, tight_pack=True, norm_dtype="fp16",
    ),
    "TURBOQUANT_b4_tfu":  TurboQuantConfig(
        algo="prod", bits=4, tight_pack=True,
        norm_dtype="fp16", rnorm_dtype="uint8",
    ),
    "TURBOQUANT_b5":      TurboQuantConfig(algo="prod", bits=5),
    "TURBOQUANT_mse_b4":  TurboQuantConfig(algo="mse", bits=4),

    # ----- Split (paper §4.3) -----
    "TURBOQUANT_split_2_25bit":     TurboQuantConfig(
        algo="prod", bits_outlier=3, bits_regular=2,
        outlier_mask_path=DEFAULT_OUTLIER_MASK,
    ),
    "TURBOQUANT_split_3_5bit":      TurboQuantConfig(
        algo="prod", bits_outlier=5, bits_regular=3,
        outlier_mask_path=DEFAULT_OUTLIER_MASK,
    ),
    "TURBOQUANT_split_3_5bit_f":    TurboQuantConfig(
        algo="prod", bits_outlier=5, bits_regular=3,
        outlier_mask_path=DEFAULT_OUTLIER_MASK,
        norm_dtype="fp16",
    ),
    "TURBOQUANT_split_3_5bit_fu":   TurboQuantConfig(
        algo="prod", bits_outlier=5, bits_regular=3,
        outlier_mask_path=DEFAULT_OUTLIER_MASK,
        norm_dtype="fp16", rnorm_dtype="uint8",
    ),

    # ----- CUDA path (legacy; rejected on this branch but registered
    #       so worktree-on-old-branch usage still resolves the name) -----
    "TURBOQUANT_mse_b4_cuda":       TurboQuantConfig(
        algo="mse", bits=4, use_cuda=True,
    ),
    "TURBOQUANT_prod_b4_cuda":      TurboQuantConfig(
        algo="prod", bits=4, use_cuda=True,
    ),
}


# Backward-compat alias map.
#
# Two flavors:
# * SILENT_ALIASES: alternative spellings considered equally valid; no
#   warning. E.g. TURBOQUANT_prod_b4 is the long form of TURBOQUANT_b4.
# * DEPRECATED_ALIASES: old names from the _r / _rr suffix scheme;
#   emit a stderr warning so callers update their scripts.
SILENT_ALIASES: dict[str, str] = {
    "TURBOQUANT_prod_b4":   "TURBOQUANT_b4",
    "TURBOQUANT_prod_b4_t": "TURBOQUANT_b4_t",
}

DEPRECATED_ALIASES: dict[str, str] = {
    "TURBOQUANT_b4_r":              "TURBOQUANT_b4_t",
    "TURBOQUANT_prod_b4_r":         "TURBOQUANT_b4_t",
    "TURBOQUANT_split_3_5bit_r":    "TURBOQUANT_split_3_5bit_f",
    "TURBOQUANT_split_3_5bit_rr":   "TURBOQUANT_split_3_5bit_fu",
}


class UnknownStageError(KeyError):
    pass


def resolve_stage(name: str) -> tuple[str, TurboQuantConfig]:
    """Return (canonical_name, config) for a possibly-aliased stage name.

    Raises UnknownStageError on miss.
    """
    if name in DEPRECATED_ALIASES:
        new = DEPRECATED_ALIASES[name]
        sys.stderr.write(f"[stages] DEPRECATED: {name} -> {new}\n")
        name = new
    elif name in SILENT_ALIASES:
        name = SILENT_ALIASES[name]
    if name not in STAGES:
        known = sorted(STAGES.keys()) + sorted(SILENT_ALIASES.keys()) \
                + sorted(DEPRECATED_ALIASES.keys())
        raise UnknownStageError(
            f"Unknown stage: {name!r}\nKnown stages:\n  "
            + "\n  ".join(known)
        )
    return name, STAGES[name]


def list_stage_names() -> list[str]:
    """All canonical stage names, sorted."""
    return sorted(STAGES.keys())


def all_stage_names_including_aliases() -> list[str]:
    """All names accepted by resolve_stage(), sorted."""
    return sorted(
        list(STAGES.keys())
        + list(SILENT_ALIASES.keys())
        + list(DEPRECATED_ALIASES.keys())
    )
