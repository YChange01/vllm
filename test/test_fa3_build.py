"""Probe FA3 build toolchain.

Verifies:
  1) turboquant_cuda extension JIT-compiles with CUTLASS headers on the
     current arch (sm_90a / sm_100a).
  2) The FA3 translation unit linked successfully and exposes
     cutlass_version_probe().

Run:
    VLLM_CUTLASS_SRC_DIR=/path/to/cutlass python3 test/test_fa3_build.py

Exit codes: 0 = all green, 1 = build or probe failed.
"""

from __future__ import annotations

import os
import sys
import traceback


def main() -> int:
    try:
        from vllm.turboquant.attend_cuda import _ext
    except Exception:
        traceback.print_exc()
        print("[fa3_build] import failed", file=sys.stderr)
        return 1

    try:
        ext = _ext()
    except Exception:
        traceback.print_exc()
        print("[fa3_build] JIT compile failed", file=sys.stderr)
        return 1

    if not hasattr(ext, "cutlass_version_probe"):
        print("[fa3_build] extension loaded but probe symbol missing "
              "(binding.cpp out of sync?)", file=sys.stderr)
        return 1

    version = ext.cutlass_version_probe()
    if version < 0:
        print("[fa3_build] FA3 path not built (TURBOQUANT_BUILD_FA3=0 ?)",
              file=sys.stderr)
        return 1

    major = version // 10000
    minor = (version // 100) % 100
    patch = version % 100
    print(f"[fa3_build] OK  cutlass={major}.{minor}.{patch}  "
          f"sm_arch={os.environ.get('TURBOQUANT_CUDA_ARCH', '(default)')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
