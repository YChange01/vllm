#!/usr/bin/env python3
"""Numerical equivalence smoke test: turboquant_paged_attention vs _lut.

Drives both kernels on the same quantized KV cache and the same query,
compares outputs element-wise. Expected:

  * ``max_abs_diff`` <= 1e-3 (fp32 accumulation order differs between
    base and LUT-flash-decoding variants; online softmax is
    mathematically associative but fp rounding differs when the KV
    dimension is split across programs).
  * ``mean_rel_diff`` <= 1%.

If diffs blow up beyond these, the LUT kernel drifted -- fix before
benchmarking throughput.

Usage::
    python3 test/test_lut_vs_base.py
    python3 test/test_lut_vs_base.py --algo prod --bits 4 --num-tokens 128
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from vllm.turboquant.attend import turboquant_paged_attention
from vllm.turboquant.attend_lut import turboquant_paged_attention_lut
from vllm.turboquant.codebook import QuantState
from vllm.turboquant.store import turboquant_store_kv, turboquant_store_v


def _alloc_cache(num_blocks, block_size, num_heads_kv, head_size, device, algo, bits):
    K_CB = 1 << (bits - 1 if algo == "prod" else bits)
    idx_last = head_size // 2 if K_CB <= 16 else head_size
    c_k_idx = torch.zeros(num_blocks, block_size, num_heads_kv, idx_last,
                          dtype=torch.uint8, device=device)
    c_k_norm = torch.zeros(num_blocks, block_size, num_heads_kv,
                           dtype=torch.float32, device=device)
    c_v_idx = torch.zeros(num_blocks, block_size, num_heads_kv, idx_last,
                          dtype=torch.uint8, device=device)
    c_v_norm = torch.zeros(num_blocks, block_size, num_heads_kv,
                           dtype=torch.float32, device=device)
    if algo == "prod":
        c_k_qjl = torch.zeros(num_blocks, block_size, num_heads_kv, head_size // 8,
                              dtype=torch.uint8, device=device)
        c_k_rnorm = torch.zeros(num_blocks, block_size, num_heads_kv,
                                dtype=torch.float32, device=device)
    else:
        c_k_qjl = None
        c_k_rnorm = None
    return c_k_idx, c_k_norm, c_v_idx, c_v_norm, c_k_qjl, c_k_rnorm


def run_one(algo: str, bits: int, num_tokens: int, num_heads_q: int,
            num_heads_kv: int, head_size: int, block_size: int,
            seed: int) -> dict:
    device = "cuda"
    torch.manual_seed(seed)
    dtype = torch.bfloat16

    q = torch.randn(num_tokens, num_heads_q, head_size, dtype=dtype, device=device)
    k = torch.randn(num_tokens, num_heads_kv, head_size, dtype=dtype, device=device)
    v = torch.randn(num_tokens, num_heads_kv, head_size, dtype=dtype, device=device)

    state = QuantState(algo=algo, bits=bits, head_dim=head_size,
                       seed=42, dtype=dtype, device=device)
    num_blocks = max(1, (num_tokens + block_size - 1) // block_size) + 1
    caches = _alloc_cache(num_blocks, block_size, num_heads_kv, head_size,
                          device, algo, bits)
    c_k_idx, c_k_norm, c_v_idx, c_v_norm, c_k_qjl, c_k_rnorm = caches

    first_slot = block_size
    slot_mapping = torch.arange(first_slot, first_slot + num_tokens,
                                dtype=torch.int64, device=device)
    block_table = torch.zeros(1, num_blocks, dtype=torch.int32, device=device)
    for i in range(num_blocks - 1):
        block_table[0, i] = i + 1
    seq_lens = torch.tensor([num_tokens], dtype=torch.int32, device=device)
    query_start_loc = torch.tensor([0, num_tokens], dtype=torch.int32, device=device)

    turboquant_store_kv(
        new_k=k, cache_k_idx=c_k_idx, cache_k_norm=c_k_norm,
        slot_mapping=slot_mapping, state=state, block_size=block_size,
        cache_k_qjl_sign=c_k_qjl, cache_k_rnorm=c_k_rnorm,
    )
    turboquant_store_v(
        new_v=v, cache_v_idx=c_v_idx, cache_v_norm=c_v_norm,
        slot_mapping=slot_mapping, state=state, block_size=block_size,
    )
    torch.cuda.synchronize()

    common_kwargs = dict(
        q=q, cache_k_idx=c_k_idx, cache_k_norm=c_k_norm,
        cache_v_idx=c_v_idx, cache_v_norm=c_v_norm,
        block_table=block_table, seq_lens=seq_lens,
        query_start_loc=query_start_loc, state=state,
        cache_k_qjl_sign=c_k_qjl, cache_k_rnorm=c_k_rnorm,
    )
    out_base = turboquant_paged_attention(**common_kwargs)
    out_lut = turboquant_paged_attention_lut(**common_kwargs)
    torch.cuda.synchronize()

    diff = (out_base.float() - out_lut.float()).abs()
    base_abs = out_base.float().abs().mean().clamp(min=1e-9)
    return {
        "max_abs_diff": float(diff.max().item()),
        "mean_abs_diff": float(diff.mean().item()),
        "mean_rel_diff": float((diff.mean() / base_abs).item()),
        "out_base_mean": float(out_base.float().abs().mean().item()),
        "out_lut_mean": float(out_lut.float().abs().mean().item()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--algo", default="both", choices=["mse", "prod", "both"])
    ap.add_argument("--bits", type=int, default=4)
    ap.add_argument("--num-tokens", type=int, default=None)
    ap.add_argument("--num-heads-q", type=int, default=16)
    ap.add_argument("--num-heads-kv", type=int, default=8)
    ap.add_argument("--head-size", type=int, default=128)
    ap.add_argument("--block-size", type=int, default=16)
    args = ap.parse_args()

    cases = [args.num_tokens] if args.num_tokens else [1, 16, 64, 256]
    algos = ["mse", "prod"] if args.algo == "both" else [args.algo]

    print(f"# head_q={args.num_heads_q} head_kv={args.num_heads_kv} "
          f"d={args.head_size} block_size={args.block_size} "
          f"bits={args.bits}")
    print(f"{'algo':<5} {'tokens':>7} {'max_abs':>12} {'mean_abs':>12} "
          f"{'mean_rel':>10} {'base_mean':>11} {'lut_mean':>11}")
    for algo in algos:
        for n in cases:
            r = run_one(
                algo=algo, bits=args.bits, num_tokens=n,
                num_heads_q=args.num_heads_q,
                num_heads_kv=args.num_heads_kv,
                head_size=args.head_size, block_size=args.block_size,
                seed=0,
            )
            print(f"{algo:<5} {n:>7} "
                  f"{r['max_abs_diff']:>12.2e} {r['mean_abs_diff']:>12.2e} "
                  f"{r['mean_rel_diff']:>10.4%} "
                  f"{r['out_base_mean']:>11.4f} {r['out_lut_mean']:>11.4f}")


if __name__ == "__main__":
    main()
