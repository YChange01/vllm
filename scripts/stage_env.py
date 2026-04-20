#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bash bridge: emit env-var assignments for a TurboQuant stage name.

Usage from bash:
    extra_env=$(python3 scripts/stage_env.py TURBOQUANT_b4_t \
                ${OUTLIER_MASK:+--outlier-mask "$OUTLIER_MASK"})
    env $extra_env vllm serve ...

Output:
    Single line of "KEY=VAL KEY=VAL ..." (or empty for FLASH_ATTN).
    Backend name on stderr line "BACKEND=TURBOQUANT" (or FLASH_ATTN).

Use --backend to print just the backend name (FLASH_ATTN or
TURBOQUANT) for the bash --attention-backend argument.
"""

from __future__ import annotations

import argparse
import sys


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", help="Stage name; see scripts/stages.py")
    ap.add_argument(
        "--backend", action="store_true",
        help="Print just the backend name (FLASH_ATTN or TURBOQUANT)",
    )
    ap.add_argument(
        "--outlier-mask", default=None,
        help="Override the OUTLIER_MASK path embedded in split stages",
    )
    args = ap.parse_args()

    if args.stage == "FLASH_ATTN":
        if args.backend:
            print("FLASH_ATTN")
        # No env vars for FLASH; print empty line.
        else:
            print("")
        return 0

    # Defer the heavy import until we know we need it.
    from vllm.turboquant.stages import (  # noqa: E402
        UnknownStageError, resolve_stage,
    )

    try:
        canonical, cfg = resolve_stage(args.stage)
    except UnknownStageError as e:
        sys.stderr.write(str(e) + "\n")
        return 1

    if args.outlier_mask is not None and cfg.is_split:
        cfg = cfg.with_outlier_mask(args.outlier_mask)

    cfg.validate()

    if args.backend:
        print("TURBOQUANT")
        return 0

    env = cfg.to_env_dict()
    print(" ".join(f"{k}={v}" for k, v in env.items()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
