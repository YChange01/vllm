# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Preflight parity check: Triton kernels vs pure-PyTorch reference.

Runs on B200 against a small synthetic KV cache. Reports max-abs and
cosine-similarity diffs between each Triton kernel output (bf16) and
the ``vllm.turboquant.reference`` implementation (fp32, the ideal
algorithm). If these numbers are outside the empirical bf16 noise
floor, don't bother running NIAH -- a kernel bug needs fixing first.

Usage:
    python3 test/test_kernel_vs_reference.py

Thresholds (see HOMOG_* / SPLIT_* constants for rationale):
    homog paths   : max_abs < 0.25, cos_sim > 0.985
    split paths   : max_abs < 0.40, cos_sim > 0.975

Tests exercised:
    homog b=4                 : prod Q2, 3-bit main + 1-bit QJL
    homog b=2                 : prod Q2, 1-bit main + 1-bit QJL
    homog b=5                 : prod Q2, 4-bit main + 1-bit QJL
    split 3.5-bit             : 64@b=4 + 64@b=3 (no outliers)
    split 'paper 2.5-bit'     : 32@b=3 + 96@b=2 (paper §4.3 literal,
                                arithmetically 2.25), first 32 chans *= 10
    split high-precision      : 32@b=5 + 96@b=4 (no outliers)
"""

from __future__ import annotations

import math
import sys

import torch

from vllm.turboquant.attend_split_tc import (
    turboquant_paged_attention_split_tc,
)
from vllm.turboquant.attend_tc import turboquant_paged_attention_tc
from vllm.turboquant.codebook import QuantState
from vllm.turboquant.outlier import SplitQuantState
from vllm.turboquant.reference import (
    turboquant_attend_reference,
    turboquant_attend_split_reference,
)
from vllm.turboquant.store import (
    turboquant_store_kv,
    turboquant_store_split,
    turboquant_store_v,
)


DEVICE = torch.device("cuda:0")
DTYPE = torch.bfloat16
# Realistic bf16-kernel vs fp32-reference noise floor for this pipeline.
# The bf16 rotation (q @ Pi, store) introduces ~3e-3 per-coord error, and
# Lloyd-Max + QJL both have discrete decision boundaries: ~2-3% of idx/
# sign bits flip between the bf16 kernel and the fp32 reference. Each
# flip contributes ~0.25 * probs (idx) or ~0.02-0.05 * probs (sign) to
# the output, and summed over d=128 coords at T_kv=32 gives ~0.1-0.2
# max-abs diff. Direction stays tight (cos > 0.995) because flips are
# locally small relative to the total inner product magnitude.
# 0.99015 observed empirically for b=4; 0.985 leaves a 0.005 margin
# and absorbs b-dependent variation (coarser codebooks -> more flips).
HOMOG_MAX_ABS = 0.25
HOMOG_COS = 0.985
# Split path runs one @ S and one @ Pi_T per slice -> 4 bf16 matmuls
# total vs 2 for homog; plus the slice gather/scatter adds more
# numerical drift.
SPLIT_MAX_ABS = 0.40
SPLIT_COS = 0.975


def _paged_layout(
    k: torch.Tensor, v: torch.Tensor, block_size: int
) -> tuple[int, int, torch.Tensor, torch.Tensor]:
    """Lay out (T_kv, H_kv, d) K/V into a single-sequence paged cache.

    Returns: (num_blocks, block_size, block_table, slot_mapping).
    """
    T_kv, H_kv, d = k.shape
    num_blocks = (T_kv + block_size - 1) // block_size
    block_table = torch.arange(
        num_blocks, dtype=torch.int32, device=DEVICE
    ).unsqueeze(0)
    slot_mapping = torch.arange(T_kv, dtype=torch.int64, device=DEVICE)
    return num_blocks, block_size, block_table, slot_mapping


def _max_abs_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    return (a.float() - b.float()).abs().max().item()


def _cos_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    flat_a = a.float().flatten(0, -2)
    flat_b = b.float().flatten(0, -2)
    return torch.nn.functional.cosine_similarity(
        flat_a, flat_b, dim=-1
    ).mean().item()


def check_homogeneous(bits: int) -> bool:
    torch.manual_seed(0)
    T_q, T_kv, H_q, H_kv, d = 4, 32, 8, 2, 128
    block_size = 16
    assert T_kv % block_size == 0

    state = QuantState(
        algo="prod", bits=bits, head_dim=d, seed=0,
        dtype=DTYPE, device=DEVICE,
    )

    k = torch.randn(T_kv, H_kv, d, dtype=DTYPE, device=DEVICE)
    v = torch.randn(T_kv, H_kv, d, dtype=DTYPE, device=DEVICE)
    q = torch.randn(T_q, H_q, d, dtype=DTYPE, device=DEVICE)

    # Allocate paged buffers matching ._ensure_buffers layout.
    num_blocks, _, block_table, slot_mapping = _paged_layout(k, v, block_size)
    pack_bits = state.pack_bits
    idx_dim = d * pack_bits // 8
    qjl_dim = d // 8

    cache_k_idx = torch.zeros(num_blocks, block_size, H_kv, idx_dim,
                              dtype=torch.uint8, device=DEVICE)
    cache_k_norm = torch.zeros(num_blocks, block_size, H_kv,
                               dtype=torch.float32, device=DEVICE)
    cache_v_idx = torch.zeros_like(cache_k_idx)
    cache_v_norm = torch.zeros_like(cache_k_norm)
    cache_k_qjl = torch.zeros(num_blocks, block_size, H_kv, qjl_dim,
                              dtype=torch.uint8, device=DEVICE)
    cache_k_rn = torch.zeros_like(cache_k_norm)
    cache_v_qjl = torch.zeros_like(cache_k_qjl)
    cache_v_rn = torch.zeros_like(cache_k_norm)

    turboquant_store_kv(
        new_k=k, cache_k_idx=cache_k_idx, cache_k_norm=cache_k_norm,
        slot_mapping=slot_mapping, state=state, block_size=block_size,
        cache_k_qjl_sign=cache_k_qjl, cache_k_rnorm=cache_k_rn,
    )
    turboquant_store_v(
        new_v=v, cache_v_idx=cache_v_idx, cache_v_norm=cache_v_norm,
        slot_mapping=slot_mapping, state=state, block_size=block_size,
        cache_v_qjl_sign=cache_v_qjl, cache_v_rnorm=cache_v_rn,
    )

    # Single sequence of length T_kv; the last T_q positions are the
    # "new" queries in this step (prefix_len = T_kv - T_q preceding them
    # are the cached prefix, visible under causal mask).
    seq_lens = torch.tensor([T_kv], dtype=torch.int32, device=DEVICE)
    query_start_loc = torch.tensor([0, T_q], dtype=torch.int32, device=DEVICE)

    # Triton path
    out_triton = turboquant_paged_attention_tc(
        q=q, cache_k_idx=cache_k_idx, cache_k_norm=cache_k_norm,
        cache_v_idx=cache_v_idx, cache_v_norm=cache_v_norm,
        block_table=block_table, seq_lens=seq_lens,
        query_start_loc=query_start_loc, state=state,
        cache_k_qjl_sign=cache_k_qjl, cache_k_rnorm=cache_k_rn,
        cache_v_qjl_sign=cache_v_qjl, cache_v_rnorm=cache_v_rn,
    )

    # Reference path in fp32 (the ideal algorithm baseline). The kernel
    # runs bf16 and will legitimately differ due to (i) bf16 rotation
    # noise flipping Lloyd-Max idx near bucket boundaries, (ii) bf16
    # residual noise flipping QJL sign near zero. Thresholds are sized
    # to absorb these discrete flips at d=128 + T_kv=32; see module-
    # level HOMOG_MAX_ABS / SPLIT_MAX_ABS comments.
    # Match the kernel's causal semantics: the kernel treats the T_q
    # queries as the last T_q new tokens in a T_kv-long sequence, so
    # kv_end[i] = (T_kv - T_q) + i + 1.
    prefix_len = T_kv - T_q
    kv_end = (
        torch.arange(T_q, dtype=torch.int64, device=DEVICE)
        + prefix_len + 1
    )
    out_ref = turboquant_attend_reference(
        q.float(), k.float(), v.float(), state, kv_end,
    ).to(DTYPE)

    diff = _max_abs_diff(out_triton, out_ref)
    cos = _cos_sim(out_triton, out_ref)
    ok = diff < HOMOG_MAX_ABS and cos > HOMOG_COS
    status = "PASS" if ok else "FAIL"
    print(
        f"  [{status}] homog b={bits}: max_abs_diff={diff:.4f}  "
        f"cos_sim={cos:.5f}"
    )
    return ok


def check_split(bits_out: int, bits_reg: int, d_out: int,
                with_outliers: bool, label: str) -> bool:
    torch.manual_seed(0)
    T_q, T_kv, H_q, H_kv, d = 4, 32, 8, 2, 128
    block_size = 16
    assert d_out % 8 == 0 and (d - d_out) % 8 == 0
    d_reg = d - d_out

    k = torch.randn(T_kv, H_kv, d, dtype=DTYPE, device=DEVICE)
    v = torch.randn(T_kv, H_kv, d, dtype=DTYPE, device=DEVICE)
    q = torch.randn(T_q, H_q, d, dtype=DTYPE, device=DEVICE)
    if with_outliers:
        k[..., :d_out] *= 10.0
        v[..., :d_out] *= 10.0

    outlier_idx = torch.arange(d_out, dtype=torch.int64, device=DEVICE)
    state_k = SplitQuantState(
        algo="prod", bits_outlier=bits_out, bits_regular=bits_reg,
        head_dim=d, outlier_idx=outlier_idx, seed=0,
        dtype=DTYPE, device=DEVICE,
    )
    state_v = SplitQuantState(
        algo="prod", bits_outlier=bits_out, bits_regular=bits_reg,
        head_dim=d, outlier_idx=outlier_idx, seed=1,
        dtype=DTYPE, device=DEVICE,
    )

    # Paged buffers for each slice.
    num_blocks, _, block_table, slot_mapping = _paged_layout(k, v, block_size)
    pack_out = state_k.state_out.pack_bits
    pack_reg = state_k.state_reg.pack_bits

    def _mk_bufs(d_slice: int, pack_bits: int):
        idx_d = d_slice * pack_bits // 8
        qjl_d = d_slice // 8
        return (
            torch.zeros(num_blocks, block_size, H_kv, idx_d,
                        dtype=torch.uint8, device=DEVICE),
            torch.zeros(num_blocks, block_size, H_kv,
                        dtype=torch.float32, device=DEVICE),
            torch.zeros(num_blocks, block_size, H_kv, qjl_d,
                        dtype=torch.uint8, device=DEVICE),
            torch.zeros(num_blocks, block_size, H_kv,
                        dtype=torch.float32, device=DEVICE),
        )

    k_idx_out, k_norm_out, k_qjl_out, k_rn_out = _mk_bufs(d_out, pack_out)
    k_idx_reg, k_norm_reg, k_qjl_reg, k_rn_reg = _mk_bufs(d_reg, pack_reg)
    v_idx_out, v_norm_out, v_qjl_out, v_rn_out = _mk_bufs(d_out, pack_out)
    v_idx_reg, v_norm_reg, v_qjl_reg, v_rn_reg = _mk_bufs(d_reg, pack_reg)

    turboquant_store_split(
        new_x=k, state_split=state_k,
        cache_idx_out=k_idx_out, cache_norm_out=k_norm_out,
        cache_qjl_sign_out=k_qjl_out, cache_rnorm_out=k_rn_out,
        cache_idx_reg=k_idx_reg, cache_norm_reg=k_norm_reg,
        cache_qjl_sign_reg=k_qjl_reg, cache_rnorm_reg=k_rn_reg,
        slot_mapping=slot_mapping, block_size=block_size,
    )
    turboquant_store_split(
        new_x=v, state_split=state_v,
        cache_idx_out=v_idx_out, cache_norm_out=v_norm_out,
        cache_qjl_sign_out=v_qjl_out, cache_rnorm_out=v_rn_out,
        cache_idx_reg=v_idx_reg, cache_norm_reg=v_norm_reg,
        cache_qjl_sign_reg=v_qjl_reg, cache_rnorm_reg=v_rn_reg,
        slot_mapping=slot_mapping, block_size=block_size,
    )

    seq_lens = torch.tensor([T_kv], dtype=torch.int32, device=DEVICE)
    query_start_loc = torch.tensor([0, T_q], dtype=torch.int32, device=DEVICE)

    out_triton = turboquant_paged_attention_split_tc(
        q=q,
        cache_k_idx_out=k_idx_out, cache_k_norm_out=k_norm_out,
        cache_v_idx_out=v_idx_out, cache_v_norm_out=v_norm_out,
        cache_k_qjl_sign_out=k_qjl_out, cache_k_rnorm_out=k_rn_out,
        cache_v_qjl_sign_out=v_qjl_out, cache_v_rnorm_out=v_rn_out,
        cache_k_idx_reg=k_idx_reg, cache_k_norm_reg=k_norm_reg,
        cache_v_idx_reg=v_idx_reg, cache_v_norm_reg=v_norm_reg,
        cache_k_qjl_sign_reg=k_qjl_reg, cache_k_rnorm_reg=k_rn_reg,
        cache_v_qjl_sign_reg=v_qjl_reg, cache_v_rnorm_reg=v_rn_reg,
        block_table=block_table, seq_lens=seq_lens,
        query_start_loc=query_start_loc,
        state_k=state_k, state_v=state_v,
    )

    prefix_len = T_kv - T_q
    kv_end = (
        torch.arange(T_q, dtype=torch.int64, device=DEVICE)
        + prefix_len + 1
    )
    out_ref = turboquant_attend_split_reference(
        q.float(), k.float(), v.float(), state_k, state_v, kv_end,
    ).to(DTYPE)

    diff = _max_abs_diff(out_triton, out_ref)
    cos = _cos_sim(out_triton, out_ref)
    ok = diff < SPLIT_MAX_ABS and cos > SPLIT_COS
    status = "PASS" if ok else "FAIL"
    print(
        f"  [{status}] {label}: max_abs_diff={diff:.4f}  "
        f"cos_sim={cos:.5f}"
    )
    return ok


def main() -> int:
    print(f"== turboquant kernel vs reference parity on {DEVICE} ==")
    print(
        f"dtype={DTYPE}, "
        f"homog thresholds: max_abs<{HOMOG_MAX_ABS}, cos>{HOMOG_COS}; "
        f"split thresholds: max_abs<{SPLIT_MAX_ABS}, cos>{SPLIT_COS}\n"
    )

    # Run every case and collect pass/fail so we see all 6 numbers
    # in one invocation, even when early cases fail.
    results: list[bool] = [
        check_homogeneous(bits=4),
        check_homogeneous(bits=2),
        check_homogeneous(bits=5),
        check_split(bits_out=4, bits_reg=3, d_out=64, with_outliers=False,
                    label="split 3.5-bit (64@4 + 64@3) no outliers"),
        check_split(bits_out=3, bits_reg=2, d_out=32, with_outliers=True,
                    label="split paper '2.5-bit' (32@3 + 96@2) w/ outliers"),
        check_split(bits_out=5, bits_reg=4, d_out=32, with_outliers=False,
                    label="split high-precision (32@5 + 96@4)"),
    ]

    passed = sum(results)
    total = len(results)
    print(f"\n{passed}/{total} parity checks PASSED.")
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
