#!/usr/bin/env python3
"""Numerical equivalence smoke test: CUDA attend vs Triton TC attend.

Drives both kernels on the same quantized KV cache and query, compares
outputs element-wise. TC is our current performance milestone; CUDA
(WMMA) should produce matching values modulo bf16 rounding (~1.6e-2
max_abs, <1% mean_rel).

First run triggers the CUDA extension JIT compile (~30-60 s on B200).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from vllm.turboquant.attend_tc import turboquant_paged_attention_tc
from vllm.turboquant.attend_cuda import turboquant_paged_attention_cuda
from vllm.turboquant.codebook import QuantState
from vllm.turboquant.store import turboquant_store_kv, turboquant_store_v


def _alloc_cache(num_blocks, block_size, num_heads_kv, head_size, device,
                 algo, bits):
    K_CB = 1 << (bits - 1 if algo == "prod" else bits)
    idx_last = head_size // 2 if K_CB <= 16 else head_size
    c_k_idx = torch.zeros(num_blocks, block_size, num_heads_kv, idx_last,
                          dtype=torch.uint8, device=device)
    c_k_norm = torch.zeros(num_blocks, block_size, num_heads_kv,
                           dtype=torch.float32, device=device)
    c_v_idx = torch.zeros_like(c_k_idx)
    c_v_norm = torch.zeros_like(c_k_norm)
    return c_k_idx, c_k_norm, c_v_idx, c_v_norm


def run_one(num_tokens: int, bits: int = 4, num_heads_q: int = 16,
            num_heads_kv: int = 8, head_size: int = 128,
            block_size: int = 16, seed: int = 0) -> dict:
    device = "cuda"
    torch.manual_seed(seed)
    dtype = torch.bfloat16

    q = torch.randn(num_tokens, num_heads_q, head_size, dtype=dtype, device=device)
    k = torch.randn(num_tokens, num_heads_kv, head_size, dtype=dtype, device=device)
    v = torch.randn(num_tokens, num_heads_kv, head_size, dtype=dtype, device=device)

    state = QuantState(algo="mse", bits=bits, head_dim=head_size,
                       seed=42, dtype=dtype, device=device)
    num_blocks = max(1, (num_tokens + block_size - 1) // block_size) + 1
    c_k_idx, c_k_norm, c_v_idx, c_v_norm = _alloc_cache(
        num_blocks, block_size, num_heads_kv, head_size, device, "mse", bits
    )

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
    )

    out_tc = turboquant_paged_attention_tc(**common_kwargs)
    out_cuda = turboquant_paged_attention_cuda(**common_kwargs)
    torch.cuda.synchronize()

    diff = (out_tc.float() - out_cuda.float()).abs()
    base_abs = out_tc.float().abs().mean().clamp(min=1e-9)
    return {
        "num_tokens": num_tokens,
        "max_abs_diff": float(diff.max().item()),
        "mean_abs_diff": float(diff.mean().item()),
        "mean_rel_diff": float((diff.mean() / base_abs).item()),
        "tc_mean": float(out_tc.float().abs().mean().item()),
        "cuda_mean": float(out_cuda.float().abs().mean().item()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bits", type=int, default=4)
    ap.add_argument("--num-tokens", type=int, default=None)
    ap.add_argument("--head-size", type=int, default=128)
    args = ap.parse_args()

    cases = [args.num_tokens] if args.num_tokens else [1, 16, 64, 256]

    print(f"# bits={args.bits} head_size={args.head_size} algo=mse")
    print(f"{'tokens':>7} {'max_abs':>12} {'mean_abs':>12} "
          f"{'mean_rel':>10} {'tc':>10} {'cuda':>10}")
    for n in cases:
        r = run_one(num_tokens=n, bits=args.bits, head_size=args.head_size)
        print(f"{r['num_tokens']:>7} "
              f"{r['max_abs_diff']:>12.2e} {r['mean_abs_diff']:>12.2e} "
              f"{r['mean_rel_diff']:>10.4%} "
              f"{r['tc_mean']:>10.4f} {r['cuda_mean']:>10.4f}")


if __name__ == "__main__":
    main()
