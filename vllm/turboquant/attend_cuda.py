# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Raw-CUDA turboquant attend kernel with inline dequant.

JIT-compiled at import time via ``torch.utils.cpp_extension.load``. First
import takes ~30 seconds on B200 while nvcc runs; subsequent imports
hit the torch extension cache.

This is Stage 1 of the CUDA-native attend kernel -- scalar fp32 matmul
inside, so performance is NOT yet competitive with the Triton LUT
kernel. Subsequent stages will add WMMA / wgmma tensor-core matmuls,
warp specialization, and TMA loads.

Usage (same signature as the other turboquant_paged_attention_* wrappers):

    from vllm.turboquant.attend_cuda import turboquant_paged_attention_cuda
    out = turboquant_paged_attention_cuda(
        q, cache_k_idx, cache_k_norm, cache_v_idx, cache_v_norm,
        block_table, seq_lens, query_start_loc, state,
    )

Switch on via ``TURBOQUANT_USE_CUDA=1`` in the backend.
"""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from vllm.turboquant.codebook import QuantState


_CSRC_DIR = Path(__file__).parent / "csrc"
_BUILD_DIR = Path(
    os.environ.get(
        "TURBOQUANT_CUDA_BUILD_DIR",
        str(Path.home() / ".cache" / "torch_extensions" / "turboquant_cuda"),
    )
)


def _load_extension():
    """JIT-compile and load the CUDA extension. Only run once per process."""
    from torch.utils.cpp_extension import load
    _BUILD_DIR.mkdir(parents=True, exist_ok=True)

    extra_cuda_cflags = [
        "-O3",
        "--use_fast_math",
        "-std=c++17",
        "--expt-relaxed-constexpr",
        "--expt-extended-lambda",
    ]
    # Target current and recent GPU arches. If nvcc doesn't know sm_100,
    # drop it. sm_80 (A100) and sm_90 (H100) cover most deployments.
    arch_flags = os.environ.get("TURBOQUANT_CUDA_ARCH",
                                "80;86;89;90;100").split(";")
    for arch in arch_flags:
        arch = arch.strip()
        if not arch:
            continue
        extra_cuda_cflags += [
            f"-gencode=arch=compute_{arch},code=sm_{arch}",
        ]

    return load(
        name="turboquant_cuda",
        sources=[
            str(_CSRC_DIR / "binding.cpp"),
            str(_CSRC_DIR / "attend_cuda.cu"),
        ],
        build_directory=str(_BUILD_DIR),
        extra_cflags=["-O3", "-std=c++17"],
        extra_cuda_cflags=extra_cuda_cflags,
        verbose=os.environ.get("TURBOQUANT_CUDA_VERBOSE", "0") == "1",
    )


_EXT = None


def _ext():
    global _EXT
    if _EXT is None:
        _EXT = _load_extension()
    return _EXT


def turboquant_paged_attention_cuda(
    q: torch.Tensor,
    cache_k_idx: torch.Tensor,
    cache_k_norm: torch.Tensor,
    cache_v_idx: torch.Tensor,
    cache_v_norm: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    query_start_loc: torch.Tensor,
    state: "QuantState",
    cache_k_qjl_sign: torch.Tensor | None = None,
    cache_k_rnorm: torch.Tensor | None = None,
) -> torch.Tensor:
    """CUDA-backed paged attention. Stage 1: mse only, scalar fp32 matmul."""
    use_qjl = state.algo == "prod"
    if use_qjl:
        raise NotImplementedError(
            "turboquant_paged_attention_cuda only supports mse path for now "
            "(Stage 1). prod/QJL will come in a follow-up PR."
        )

    num_query_tokens, num_heads_q, head_size = q.shape
    _, block_size, num_heads_kv, idx_dim = cache_k_idx.shape
    num_seqs = int(seq_lens.shape[0])

    dev = q.device
    qsl = query_start_loc.to(device=dev, dtype=torch.int64)
    query_lens = qsl[1:] - qsl[:-1]
    seq_ids = torch.arange(num_seqs, dtype=torch.int64, device=dev)
    seq_id_per_query_i64 = torch.repeat_interleave(seq_ids, query_lens)
    q_pos_per_query = (
        torch.arange(num_query_tokens, dtype=torch.int64, device=dev)
        - qsl[:-1][seq_id_per_query_i64]
    )
    prefix_len = seq_lens.to(device=dev, dtype=torch.int64) - query_lens
    kv_end_per_query_i64 = (
        prefix_len[seq_id_per_query_i64] + q_pos_per_query + 1
    )
    seq_id_per_query = seq_id_per_query_i64.to(torch.int32).contiguous()
    kv_end_per_query = kv_end_per_query_i64.to(torch.int32).contiguous()

    # Pre-rotate Q in Python (bf16 tensor core via cuBLAS). Same as the
    # LUT path -- fusing into the kernel is deferred.
    q_signed = q * state.signs
    q_rotated = (
        q_signed.reshape(-1, head_size) @ state.H
    ).view(num_query_tokens, num_heads_q, head_size).contiguous()

    out = torch.empty_like(q)
    gqa_group = num_heads_q // num_heads_kv

    _ext().attend_mse(
        q_rotated,
        cache_k_idx,
        cache_k_norm,
        cache_v_idx,
        cache_v_norm,
        block_table,
        seq_id_per_query,
        kv_end_per_query,
        state.codebook.to(q.dtype).contiguous(),
        out,
        int(block_size),
        int(gqa_group),
    )

    # Post-rotate V back to original space.
    inv_sqrt_d = 1.0 / math.sqrt(float(head_size))
    output = (
        out.reshape(-1, head_size) @ state.H
    ).view(num_query_tokens, num_heads_q, head_size)
    output = output * state.signs * inv_sqrt_d
    return output.contiguous()
