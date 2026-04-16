# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""TurboQuant KV cache quantization (paper arXiv:2504.19874).

Algorithms
----------
- ``Q_mse`` (Algorithm 1): b-bit Lloyd-Max on Hadamard-rotated K.
- ``Q_prod`` (Algorithm 2): (b-1)-bit Q_mse + 1-bit QJL on residual.

V is stored as per-(slot, head) symmetric int8.

Selection is via environment variables read by the attention backend:
``TURBOQUANT_ALGO`` ∈ {mse, prod}, ``TURBOQUANT_BITS`` ∈ {2, 3, ...}.
"""

from vllm.turboquant.codebook import QuantState

__all__ = ["QuantState"]
