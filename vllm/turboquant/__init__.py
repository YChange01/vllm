# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""TurboQuant KV cache quantization for vLLM (paper arXiv:2504.19874).

Layout
------
``codebook``    : QuantState -- Lloyd-Max codebook, Hadamard, signs, and
                  (for prod) QJL projection S. One per attention layer.
``store``       : K/V quantize + paged scatter (Triton kernel).
``attend_tc``   : Triton tensor-core attend kernel (default). tl.dot
                  over BLOCK_N KV tiles, flash-attention style.
``attend_cuda`` : Raw-CUDA / WMMA attend kernel. JIT-compiled on first
                  use via torch.utils.cpp_extension. Selected by
                  ``TURBOQUANT_USE_CUDA=1``.
``csrc/``       : .cu / .cpp / .h sources for the CUDA extension.

Algorithms
----------
- ``Q_mse``  (Algorithm 1): b-bit Lloyd-Max on Hadamard-rotated vectors.
- ``Q_prod`` (Algorithm 2): (b-1)-bit Q_mse + 1-bit QJL on the residual,
                            giving an unbiased inner-product estimator.

V is quantized with Q_mse; the attend kernel reconstructs V per slot
in the rotated space and the Python wrapper does the final post-rotate.

Configuration via environment variables (read at backend module load):
  TURBOQUANT_ALGO      mse | prod    (default prod)
  TURBOQUANT_BITS      4             (b=4 only on this branch)
  TURBOQUANT_USE_CUDA  0 | 1         (default 0 -> Triton TC kernel)
"""

from vllm.turboquant.codebook import QuantState

__all__ = ["QuantState"]
