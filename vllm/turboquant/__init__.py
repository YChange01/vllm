# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""TurboQuant: real 4-bit K cache quantization for vLLM.

Pipeline:
  1. Apply a fixed Hadamard rotation Pi to make each coordinate of K roughly
     N(0, 1).
  2. Quantize each coordinate with a Lloyd-Max codebook (K = 2^bits entries).
  3. Pack 4-bit indices into uint8 inside the paged KV cache (4x smaller).
  4. A Triton kernel fuses dequantize + attention on the GPU.

V is kept in standard fp16 / bf16 paged layout (softmax smooths V errors).

MVP scope: mse mode, 4-bit, fp16 inference, GQA supported, V not quantized,
no FP8, no sliding window, no ALiBi.
"""
from vllm.turboquant.codebook import (
    GaussianCodebook,
    RandomRotation,
    build_codebook,
    build_rotation,
)

__all__ = [
    "GaussianCodebook",
    "RandomRotation",
    "build_codebook",
    "build_rotation",
]
