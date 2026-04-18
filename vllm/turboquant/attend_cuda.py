# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Raw-CUDA turboquant attend kernel with inline dequant.

Uses ``nvcuda::wmma`` (mma.m16n8k16 family) tensor cores for both
Q @ K.T and P @ V matmuls; Ampere (sm_80) and newer. Same signature
as ``turboquant_paged_attention_tc`` so the two are drop-in swappable
from the backend.

JIT-compiled on first call via ``torch.utils.cpp_extension.load``
(~30-60 s while nvcc runs; cached thereafter). Compilation is lazy so
importing this module is cheap.

Usage:
    from vllm.turboquant.attend_cuda import turboquant_paged_attention_cuda

Selected by setting ``TURBOQUANT_USE_CUDA=1`` in the attention backend.

Scope: mse path only. prod / QJL is a follow-up. Future work will
replace WMMA with wgmma + TMA + warp specialization (true FA3 shape).
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


def _discover_cutlass_include() -> str | None:
    """Locate CUTLASS headers. Order: env var → vllm build tree → None."""
    for var in ("VLLM_CUTLASS_SRC_DIR", "CUTLASS_DIR", "CUTLASS_SRC_DIR"):
        root = os.environ.get(var)
        if root and (Path(root) / "include" / "cutlass" / "cutlass.h").exists():
            return str(Path(root) / "include")

    try:
        import vllm
        vllm_root = Path(vllm.__file__).resolve().parent.parent
    except ImportError:
        return None

    candidates = [
        vllm_root / "build" / "_deps" / "cutlass-src" / "include",
        vllm_root / ".." / "build" / "_deps" / "cutlass-src" / "include",
    ]
    candidates.extend(
        vllm_root.glob("build/cp*/_deps/cutlass-src/include")
    )
    for c in candidates:
        if (c / "cutlass" / "cutlass.h").exists():
            return str(c.resolve())
    return None


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
    # Hopper (sm_90a) and Blackwell (sm_100a) — the `a` suffix enables
    # wgmma. Older arches drop to WMMA kernel only (no wgmma codegen).
    arch_flags = os.environ.get(
        "TURBOQUANT_CUDA_ARCH", "80;86;89;90a;100a"
    ).split(";")
    # wgmma requires the architecture-conditional suffix on BOTH the
    # virtual (compute_XXa) and real (sm_XXa) arch — see
    # cmake/utils.cmake's "compute_90a,code=sm_90a" pattern.
    for arch in arch_flags:
        arch = arch.strip()
        if not arch:
            continue
        extra_cuda_cflags += [
            f"-gencode=arch=compute_{arch},code=sm_{arch}",
        ]

    sources = [
        str(_CSRC_DIR / "binding.cpp"),
        str(_CSRC_DIR / "attend_cuda.cu"),
    ]

    build_fa3 = os.environ.get("TURBOQUANT_BUILD_FA3", "1") == "1"
    extra_include_paths: list[str] = []
    if build_fa3:
        cutlass_inc = _discover_cutlass_include()
        if cutlass_inc is None:
            raise RuntimeError(
                "CUTLASS headers not found. Set VLLM_CUTLASS_SRC_DIR to the "
                "CUTLASS checkout, or set TURBOQUANT_BUILD_FA3=0 to skip the "
                "FA3 path (WMMA only)."
            )
        extra_include_paths.append(cutlass_inc)
        extra_cuda_cflags += ["-DTURBOQUANT_HAS_FA3=1"]
        sources.append(str(_CSRC_DIR / "attend_fa3.cu"))

    return load(
        name="turboquant_cuda",
        sources=sources,
        build_directory=str(_BUILD_DIR),
        extra_cflags=["-O3", "-std=c++17"],
        extra_cuda_cflags=extra_cuda_cflags,
        extra_include_paths=extra_include_paths,
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
    """CUDA-backed paged attention (WMMA tensor cores). mse only for now."""
    use_qjl = state.algo == "prod"
    if use_qjl:
        raise NotImplementedError(
            "turboquant_paged_attention_cuda only supports mse path for now "
            "prod/QJL will come in a follow-up PR."
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
    # TC path -- fusing into the kernel is deferred.
    q_signed = q * state.signs
    q_rotated = (
        q_signed.reshape(-1, head_size) @ state.H
    ).view(num_query_tokens, num_heads_q, head_size).contiguous()

    out = torch.empty_like(q)
    gqa_group = num_heads_q // num_heads_kv

    # Flash-decoding split-KV: divide each query's KV into chunks of
    # split_len tokens. A grid.z = num_splits dimension fans compute out
    # across more SMs when the batch alone can't fill them; the reduce
    # kernel merges partials via log-sum-exp. split_len must be a multiple
    # of BLOCK_N=32 (kernel-side assertion).
    split_len = int(os.environ.get("TURBOQUANT_KV_SPLIT_LEN", "512"))
    max_kv_end = int(kv_end_per_query.max().item())
    num_splits = max(1, (max_kv_end + split_len - 1) // split_len)
    num_splits = min(num_splits, 64)  # kernel MAX_SPLITS

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
        int(split_len),
        int(num_splits),
    )

    # Post-rotate V back to original space.
    inv_sqrt_d = 1.0 / math.sqrt(float(head_size))
    output = (
        out.reshape(-1, head_size) @ state.H
    ).view(num_query_tokens, num_heads_q, head_size)
    output = output * state.signs * inv_sqrt_d
    return output.contiguous()
