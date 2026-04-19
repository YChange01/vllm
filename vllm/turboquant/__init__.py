# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""TurboQuant KV cache quantization for vLLM (paper arXiv:2504.19874).

Paper-faithful implementation:
  * Random orthogonal rotation via QR(Gaussian) (Section 3.1).
  * Input normalized to the unit sphere before rotation (Lemma 1).
  * Lloyd-Max codebook trained on the exact Beta distribution f_X.
  * Both K and V are quantized with the same Q_mse or Q_prod instance.

Layout
------
``codebook``    : QuantState -- Lloyd-Max codebook, Pi (rotation), and
                  (prod) QJL projection S. One per attention layer.
``store``       : K/V quantize + paged scatter (Triton kernel).
``attend_tc``   : Triton tensor-core attend kernel. Default path;
                  supports the V-QJL accumulator split needed for
                  paper-faithful Q_prod on V.
``attend_cuda`` : Raw-CUDA / WMMA attend kernel. Not updated for the
                  V-QJL split yet; rejected when ALGO=prod on this
                  branch.
``csrc/``       : .cu / .cpp / .h sources for the CUDA extension.

Algorithms
----------
- ``Q_mse``  (Algorithm 1): b-bit Lloyd-Max on ``Pi @ (x/||x||)``.
- ``Q_prod`` (Algorithm 2): (b-1)-bit Q_mse + 1-bit QJL on the residual,
                            giving an unbiased inner-product estimator.

Configuration via environment variables (read at backend module load):
  TURBOQUANT_ALGO      mse | prod    (default prod)
  TURBOQUANT_BITS      4             (b=4 only on this branch)
  TURBOQUANT_USE_CUDA  0 | 1         (default 0 -> Triton TC kernel;
                                     rejected when ALGO=prod)
"""

from vllm.turboquant.codebook import QuantState

__all__ = ["QuantState"]
