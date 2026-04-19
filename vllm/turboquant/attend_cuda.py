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

Supports both mse (Algorithm 1) and prod (Algorithm 2 with QJL residual)
paths; selected via `state.algo`. Future work will replace WMMA with
wgmma + TMA + warp specialization (true FA3 shape).
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
    cache_v_qjl_sign: torch.Tensor | None = None,
    cache_v_rnorm: torch.Tensor | None = None,
) -> torch.Tensor:
    """CUDA-backed paged attention (WMMA + cp.async). Supports mse and prod.

    NOT UPDATED for the turboquant-paper-repro branch. The in-kernel
    logit scaling and the Python wrapper both still expect the old
    Hadamard + signs rotation and ``1/d`` attention scale; they also
    lack the V-QJL accumulator split needed for paper-faithful Q_prod
    on V. The backend rejects ``TURBOQUANT_USE_CUDA=1`` at module load
    so this function is not reachable from vLLM on this branch.
    """
    raise NotImplementedError(
        "turboquant_paged_attention_cuda is not supported on the "
        "paper-repro branch (Pi/unit-norm/1/sqrt(d) scale and V-QJL "
        "split not implemented). Use the Triton TC kernel instead."
    )
    use_qjl = state.algo == "prod"
    if use_qjl:
        assert cache_k_qjl_sign is not None and cache_k_rnorm is not None, (
            "prod path requires cache_k_qjl_sign and cache_k_rnorm"
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

    # Pre-rotate Q in Python (bf16 tensor core via cuBLAS). H is symmetric
    # so H.T == H. Fusing into the kernel is deferred (see
    # `TURBOQUANT.md`'s weight-fold plan).
    q_signed = q * state.signs
    q_rotated = (
        q_signed.reshape(-1, head_size) @ state.H
    ).view(num_query_tokens, num_heads_q, head_size).contiguous()

    if use_qjl:
        # prod path: also need Sq = Q_rot @ S^T for the QJL correction
        # Sq @ qjl_sign^T inside the kernel. Matches attend_tc.py.
        Sq = (
            q_rotated.reshape(-1, head_size) @ state.S.T
        ).view(num_query_tokens, num_heads_q, head_size).contiguous()
    else:
        Sq = None

    out = torch.empty_like(q)
    gqa_group = num_heads_q // num_heads_kv
    algo_id = 1 if use_qjl else 0

    _ext().attend(
        q_rotated,
        Sq,
        cache_k_idx,
        cache_k_norm,
        cache_v_idx,
        cache_v_norm,
        cache_k_qjl_sign if use_qjl else None,
        cache_k_rnorm if use_qjl else None,
        block_table,
        seq_id_per_query,
        kv_end_per_query,
        state.codebook.to(q.dtype).contiguous(),
        out,
        int(block_size),
        int(gqa_group),
        int(algo_id),
    )

    # Post-rotate V back to original space.
    inv_sqrt_d = 1.0 / math.sqrt(float(head_size))
    output = (
        out.reshape(-1, head_size) @ state.H
    ).view(num_query_tokens, num_heads_q, head_size)
    output = output * state.signs * inv_sqrt_d
    return output.contiguous()
