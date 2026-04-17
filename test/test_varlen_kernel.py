#!/usr/bin/env python3
"""Standalone smoke test for TurboQuant kernels (no vLLM runtime).

Drives ``turboquant_store_kv`` + ``turboquant_paged_attention`` on random
Gaussian K, V, Q and compares against a pure-PyTorch softmax attention on
the UNQUANTIZED vectors. Reports separate numbers for Algorithm 1 (Q_mse)
and Algorithm 2 (Q_prod).

Usage::
    python3 test/test_varlen_kernel.py                 # default bits=4
    python3 test/test_varlen_kernel.py --num-tokens 61 # single size
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from vllm.turboquant.attend import turboquant_paged_attention
from vllm.turboquant.codebook import QuantState
from vllm.turboquant.store import turboquant_store_kv, turboquant_store_v


def reference_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    kv_end_per_query: torch.Tensor,
) -> torch.Tensor:
    """Causal softmax attention with GQA, pure PyTorch fp32."""
    num_q, num_heads_q, d = q.shape
    num_heads_kv = k.shape[1]
    gqa = num_heads_q // num_heads_kv
    scale = 1.0 / (d**0.5)
    out = torch.zeros_like(q)
    for qi in range(num_q):
        kv_end = int(kv_end_per_query[qi].item())
        for h in range(num_heads_q):
            kh = h // gqa
            scores = (q[qi, h] @ k[:kv_end, kh].T) * scale
            weights = torch.softmax(scores, dim=-1)
            out[qi, h] = weights @ v[:kv_end, kh]
    return out


def _alloc_cache(num_blocks: int, block_size: int, num_heads_kv: int,
                 head_size: int, device, algo: str, bits: int):
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
        assert head_size % 8 == 0, "head_size must be divisible by 8 for QJL bit-pack"
        c_k_qjl = torch.zeros(num_blocks, block_size, num_heads_kv, head_size // 8,
                              dtype=torch.uint8, device=device)
        c_k_rnorm = torch.zeros(num_blocks, block_size, num_heads_kv,
                                dtype=torch.float32, device=device)
    else:
        c_k_qjl = None
        c_k_rnorm = None
    return c_k_idx, c_k_norm, c_v_idx, c_v_norm, c_k_qjl, c_k_rnorm


def _run_algo(algo: str, num_tokens: int, bits: int, q, k, v, slot_mapping,
              block_table, seq_lens, query_start_loc, num_blocks, block_size,
              num_heads_kv, head_size, device, dtype) -> torch.Tensor:
    state = QuantState(algo=algo, bits=bits, head_dim=head_size,
                       seed=42, dtype=dtype, device=device)
    (c_k_idx, c_k_norm, c_v_idx, c_v_norm,
     c_k_qjl, c_k_rnorm) = _alloc_cache(
        num_blocks, block_size, num_heads_kv, head_size, device, algo, bits,
    )
    turboquant_store_kv(
        new_k=k,
        cache_k_idx=c_k_idx, cache_k_norm=c_k_norm,
        slot_mapping=slot_mapping,
        state=state, block_size=block_size,
        cache_k_qjl_sign=c_k_qjl, cache_k_rnorm=c_k_rnorm,
    )
    turboquant_store_v(
        new_v=v,
        cache_v_idx=c_v_idx, cache_v_norm=c_v_norm,
        slot_mapping=slot_mapping,
        state=state, block_size=block_size,
    )
    torch.cuda.synchronize()
    out = turboquant_paged_attention(
        q=q,
        cache_k_idx=c_k_idx, cache_k_norm=c_k_norm,
        cache_v_idx=c_v_idx, cache_v_norm=c_v_norm,
        block_table=block_table,
        seq_lens=seq_lens, query_start_loc=query_start_loc,
        state=state,
        cache_k_qjl_sign=c_k_qjl, cache_k_rnorm=c_k_rnorm,
    )
    torch.cuda.synchronize()
    return out


def run_one_case(num_tokens: int, bits: int = 8, num_heads_q: int = 16,
                 num_heads_kv: int = 8, head_size: int = 128,
                 block_size: int = 16,
                 dtype: torch.dtype = torch.bfloat16,
                 seed: int = 0) -> dict:
    assert num_heads_q % num_heads_kv == 0, (
        f"num_heads_q ({num_heads_q}) must be divisible by "
        f"num_heads_kv ({num_heads_kv})"
    )
    device = "cuda"
    torch.manual_seed(seed)

    q = torch.randn(num_tokens, num_heads_q, head_size, dtype=dtype, device=device)
    k = torch.randn(num_tokens, num_heads_kv, head_size, dtype=dtype, device=device)
    v = torch.randn(num_tokens, num_heads_kv, head_size, dtype=dtype, device=device)

    num_blocks = max(1, (num_tokens + block_size - 1) // block_size) + 1

    first_slot = block_size  # mimic vLLM reserving physical block 0
    slot_mapping = torch.arange(first_slot, first_slot + num_tokens,
                                dtype=torch.int64, device=device)

    block_table = torch.zeros(1, num_blocks, dtype=torch.int32, device=device)
    for i in range(num_blocks - 1):
        block_table[0, i] = i + 1

    seq_lens = torch.tensor([num_tokens], dtype=torch.int32, device=device)
    query_start_loc = torch.tensor([0, num_tokens], dtype=torch.int32, device=device)

    kv_end = torch.arange(1, num_tokens + 1, dtype=torch.int32, device=device)
    fp_ref = reference_attention(q.float(), k.float(), v.float(), kv_end)

    args = dict(
        num_tokens=num_tokens, bits=bits, q=q, k=k, v=v,
        slot_mapping=slot_mapping, block_table=block_table,
        seq_lens=seq_lens, query_start_loc=query_start_loc,
        num_blocks=num_blocks, block_size=block_size,
        num_heads_kv=num_heads_kv, head_size=head_size,
        device=device, dtype=dtype,
    )
    out_mse = _run_algo("mse", **args)
    out_prod = _run_algo("prod", **args)

    ref_abs_mean = fp_ref.abs().mean().clamp(min=1e-6)
    mse_diff = (out_mse.float() - fp_ref).abs()
    prod_diff = (out_prod.float() - fp_ref).abs()
    return {
        "num_tokens": num_tokens,
        "bits": bits,
        "mse_rel": float((mse_diff.mean() / ref_abs_mean).item()),
        "prod_rel": float((prod_diff.mean() / ref_abs_mean).item()),
        "mse_max": float(mse_diff.max().item()),
        "prod_max": float(prod_diff.max().item()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-tokens", type=int, default=None)
    ap.add_argument("--bits", type=int, default=4)
    ap.add_argument("--num-heads-q", type=int, default=16)
    ap.add_argument("--num-heads-kv", type=int, default=8)
    ap.add_argument("--head-size", type=int, default=128)
    args = ap.parse_args()

    cases = [args.num_tokens] if args.num_tokens else [1, 4, 16, 32, 61, 128]

    gqa = args.num_heads_q // args.num_heads_kv
    print(f"# config: heads_q={args.num_heads_q} heads_kv={args.num_heads_kv} "
          f"gqa={gqa} head_size={args.head_size} bits={args.bits}")
    print(f"{'num_tokens':>10} {'mse_rel':>10} {'prod_rel':>10} "
          f"{'reduction':>10} {'mse_max':>8} {'prod_max':>8}")
    for n in cases:
        r = run_one_case(
            num_tokens=n,
            bits=args.bits,
            num_heads_q=args.num_heads_q,
            num_heads_kv=args.num_heads_kv,
            head_size=args.head_size,
        )
        reduction = 1.0 - r["prod_rel"] / max(r["mse_rel"], 1e-12)
        print(f"{r['num_tokens']:>10} "
              f"{r['mse_rel']:>10.4%} {r['prod_rel']:>10.4%} "
              f"{reduction:>10.2%} "
              f"{r['mse_max']:>8.4f} {r['prod_max']:>8.4f}")


if __name__ == "__main__":
    main()
