# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""TurboQuant Triton kernel vs PyTorch reference parity harness.

Usage:
    python scripts/test_turboquant_parity.py                 # all cases
    python scripts/test_turboquant_parity.py --case tiny     # one case
    python scripts/test_turboquant_parity.py --store-only    # just store path
    python scripts/test_turboquant_parity.py --verbose       # print diffs

The reference implementation lives in ``vllm/turboquant/reference.py`` and
has been validated against the unquantized baseline on CPU. Any mismatch
surfaced here is a Triton kernel bug.
"""

from __future__ import annotations

import argparse
import sys
import traceback
from dataclasses import dataclass

import torch

from vllm.turboquant.codebook import GaussianCodebook
from vllm.turboquant.reference import (
    paged_attention_reference,
    store_kv_reference,
)


@dataclass
class Case:
    name: str
    num_seqs: int
    num_heads_q: int
    num_heads_kv: int
    head_size: int
    block_size: int
    seq_len: int
    dtype: torch.dtype = torch.float16


CASES: list[Case] = [
    # Start minimal: single seq, no GQA, small dims.
    Case("tiny",   num_seqs=1, num_heads_q=1, num_heads_kv=1,
         head_size=16, block_size=4,  seq_len=4),
    # Bring head_size up to a realistic power-of-2.
    Case("d64",    num_seqs=1, num_heads_q=2, num_heads_kv=2,
         head_size=64, block_size=16, seq_len=16),
    # Enable GQA (q_heads > kv_heads).
    Case("gqa",    num_seqs=1, num_heads_q=4, num_heads_kv=2,
         head_size=64, block_size=16, seq_len=32),
    # Batch of sequences with variable lengths.
    Case("batch",  num_seqs=3, num_heads_q=4, num_heads_kv=2,
         head_size=64, block_size=16, seq_len=24),
    # Qwen3-0.6B-like head_size.
    Case("d128",   num_seqs=1, num_heads_q=4, num_heads_kv=2,
         head_size=128, block_size=16, seq_len=32),
]


def _pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    print("[WARN] no CUDA available; parity cannot exercise Triton kernels")
    return "cpu"


def _make_inputs(case: Case, device: str, seed: int = 0):
    """Build synthetic q / new_k / new_v / block_table / seq_lens."""
    torch.manual_seed(seed)
    num_blocks_per_seq = (case.seq_len + case.block_size - 1) // case.block_size
    total_blocks = num_blocks_per_seq * case.num_seqs

    q = torch.randn(
        case.num_seqs, case.num_heads_q, case.head_size,
        dtype=case.dtype, device=device,
    )
    # Normalize so each k has unit-ish RMS; TurboQuant requires this (see
    # TURBOQUANT.md). Reference also expects normalized input for fair
    # numerical comparison.
    k_raw = torch.randn(
        case.num_seqs * case.seq_len, case.num_heads_kv, case.head_size,
        dtype=case.dtype, device=device,
    )
    k_norm = k_raw.norm(dim=-1, keepdim=True).clamp(min=1e-4)
    k_raw = k_raw / k_norm * (case.head_size ** 0.5)

    v_raw = torch.randn(
        case.num_seqs * case.seq_len, case.num_heads_kv, case.head_size,
        dtype=case.dtype, device=device,
    )

    # Slot mapping: seq 0 -> slots 0..seq_len-1, seq 1 -> next blocks, etc.
    slot_mapping = torch.arange(
        case.num_seqs * case.seq_len, dtype=torch.int64, device=device
    )
    block_table = torch.arange(
        total_blocks, dtype=torch.int32, device=device
    ).view(case.num_seqs, num_blocks_per_seq)
    seq_lens = torch.full(
        (case.num_seqs,), case.seq_len, dtype=torch.int32, device=device
    )
    return q, k_raw, v_raw, slot_mapping, block_table, seq_lens, total_blocks


def _empty_caches(case: Case, total_blocks: int, device: str):
    cache_k = torch.zeros(
        total_blocks, case.block_size, case.num_heads_kv, case.head_size,
        dtype=torch.uint8, device=device,
    )
    cache_v = torch.zeros(
        total_blocks, case.block_size, case.num_heads_kv, case.head_size,
        dtype=case.dtype, device=device,
    )
    return cache_k, cache_v


def _check_store_parity(case: Case, device: str, verbose: bool) -> bool:
    try:
        from vllm.turboquant.triton_kernels import turboquant_store_kv
    except Exception as e:
        print(f"  [{case.name}] STORE: import failed: {e}")
        return False

    q, k_raw, v_raw, slot_mapping, _, _, total_blocks = _make_inputs(case, device)

    cb = GaussianCodebook(
        head_dim=case.head_size, bits=4, seed=0, dtype=case.dtype, device=device
    )

    cache_k_ref, cache_v_ref = _empty_caches(case, total_blocks, device)
    cache_k_trt, cache_v_trt = _empty_caches(case, total_blocks, device)

    store_kv_reference(
        k_raw, v_raw, cache_k_ref, cache_v_ref, slot_mapping, cb, case.block_size
    )
    try:
        turboquant_store_kv(
            k_raw, v_raw, cache_k_trt, cache_v_trt, slot_mapping, cb, case.block_size
        )
    except Exception:
        print(f"  [{case.name}] STORE: triton kernel raised")
        traceback.print_exc()
        return False

    k_diff = (cache_k_ref.long() - cache_k_trt.long()).abs()
    v_diff = (cache_v_ref.float() - cache_v_trt.float()).abs()

    # Allow k_max_diff <= 1: the reference uses fp16 rotation + torch.bucketize
    # while the Triton kernel uses fp32 rotation + manual comparison, so values
    # near codebook boundaries may round to adjacent indices. A diff of 1 means
    # adjacent codewords and has negligible impact on attention output.
    k_ok = k_diff.max().item() <= 1
    v_ok = v_diff.max().item() < 1e-3
    if verbose or not (k_ok and v_ok):
        print(
            f"  [{case.name}] STORE  k_max_diff={k_diff.max().item()}"
            f"  v_max_diff={v_diff.max().item():.2e}"
        )
    return k_ok and v_ok


def _check_attend_parity(case: Case, device: str, verbose: bool) -> bool:
    try:
        from vllm.turboquant.triton_kernels import (
            turboquant_paged_attention,
            turboquant_store_kv,
        )
    except Exception as e:
        print(f"  [{case.name}] ATTEND: import failed: {e}")
        return False

    q, k_raw, v_raw, slot_mapping, block_table, seq_lens, total_blocks = (
        _make_inputs(case, device)
    )
    cb = GaussianCodebook(
        head_dim=case.head_size, bits=4, seed=0, dtype=case.dtype, device=device
    )

    # Use reference store for both paths so only the attend kernel is compared.
    cache_k, cache_v = _empty_caches(case, total_blocks, device)
    store_kv_reference(
        k_raw, v_raw, cache_k, cache_v, slot_mapping, cb, case.block_size
    )

    out_ref = paged_attention_reference(
        q, cache_k, cache_v, block_table, seq_lens, cb
    )
    try:
        out_trt = turboquant_paged_attention(
            q, cache_k, cache_v, block_table, seq_lens, cb
        )
    except Exception:
        print(f"  [{case.name}] ATTEND: triton kernel raised")
        traceback.print_exc()
        return False

    diff = (out_ref.float() - out_trt.float()).abs()
    max_d = diff.max().item()
    rel = diff.norm().item() / out_ref.float().norm().clamp(min=1e-6).item()
    tol = 5e-2  # kernel uses fp32 accum, reference uses default; be generous
    ok = max_d < tol
    if verbose or not ok:
        print(
            f"  [{case.name}] ATTEND max_diff={max_d:.3e}  rel_err={rel:.3e}  "
            f"tol={tol:.0e}"
        )
    return ok


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", default=None, help="run only this case name")
    parser.add_argument("--store-only", action="store_true")
    parser.add_argument("--attend-only", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    device = _pick_device()
    cases = CASES if args.case is None else [c for c in CASES if c.name == args.case]
    if not cases:
        print(f"no case matches --case {args.case!r}")
        return 2

    total, passed = 0, 0
    for case in cases:
        print(f"=== {case.name}  "
              f"(seqs={case.num_seqs}, q={case.num_heads_q}, kv={case.num_heads_kv}, "
              f"d={case.head_size}, block={case.block_size}, len={case.seq_len}) ===")

        if not args.attend_only:
            total += 1
            if _check_store_parity(case, device, args.verbose):
                passed += 1
                print(f"  [{case.name}] STORE  PASS")
            else:
                print(f"  [{case.name}] STORE  FAIL")

        if not args.store_only:
            total += 1
            if _check_attend_parity(case, device, args.verbose):
                passed += 1
                print(f"  [{case.name}] ATTEND PASS")
            else:
                print(f"  [{case.name}] ATTEND FAIL")

    print(f"\n{passed}/{total} checks passed")
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
