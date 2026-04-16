#!/usr/bin/env python3
"""Standalone smoke test for TurboQuant store+attend kernels, no vLLM runtime.

Why: baseline.sh proves the single-query code path (prefill 1 token + decode)
is bit-exact with FP. NIAH (61-token prefill) produces garbage. The gap
must live somewhere in the multi-query dispatch, and that path is hard to
instrument inside vLLM. This script drives the kernels directly with
controlled inputs and compares against a pure-PyTorch causal attention
reference on the SAME quantised cache, so any discrepancy is a kernel bug.

Usage (on the B200 box, inside the gyc_vllm env):
    cd /mnt/nvme3n1/g00872988/turboquant/vllm
    python3 test/test_varlen_kernel.py            # all sizes
    python3 test/test_varlen_kernel.py --num-tokens 61
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Make `vllm` importable without `pip install -e .` in case that slipped.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from vllm.turboquant.codebook import GaussianCodebook
from vllm.turboquant.triton_kernels import (
    turboquant_paged_attention,
    turboquant_store_kv,
)


def dequant_kv_reference(
    cache_k: torch.Tensor,
    cache_k_norm: torch.Tensor,
    cache_v: torch.Tensor,
    cache_v_scale: torch.Tensor,
    codebook: GaussianCodebook,
    block_table: torch.Tensor,
    seq_len: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Dequantise the K cache for sequence 0 up to seq_len, using the SAME
    formula the Triton attend kernel does. Returns k of shape
    (seq_len, num_kv_heads, head_size), dtype fp32."""
    num_blocks, block_size, num_kv_heads, head_size = cache_k.shape
    inv_sqrt_d = 1.0 / (head_size ** 0.5)

    cb = codebook.codebook.to(torch.float32)
    H = codebook.H.to(torch.float32)
    signs = codebook.signs.to(torch.float32)

    k_full = torch.zeros(seq_len, num_kv_heads, head_size,
                         dtype=torch.float32, device=cache_k.device)
    v_full = torch.zeros_like(k_full)
    for p in range(seq_len):
        block_i = p // block_size
        tok_in_block = p % block_size
        phys_block = int(block_table[0, block_i].item())
        for h in range(num_kv_heads):
            idx = cache_k[phys_block, tok_in_block, h].long()  # (d,)
            rk = cb[idx]                                       # (d,)
            k_unrot = rk @ H                                   # (d,) (H symmetric)
            k_unit = k_unrot * signs
            k_norm_head = float(cache_k_norm[phys_block, tok_in_block, h].item())
            k_full[p, h] = k_unit * (k_norm_head * inv_sqrt_d)
            v_scale_head = float(cache_v_scale[phys_block, tok_in_block, h].item())
            v_full[p, h] = cache_v[phys_block, tok_in_block, h].float() * v_scale_head
    return k_full, v_full


def reference_attention(
    q: torch.Tensor,              # (num_q, num_heads_q, d) fp32
    k: torch.Tensor,              # (seq_len, num_heads_kv, d) fp32
    v: torch.Tensor,              # (seq_len, num_heads_kv, d) fp32
    kv_end_per_query: torch.Tensor,  # (num_q,) int -- causal upper bound
) -> torch.Tensor:
    """Pure-PyTorch attention matching what the Triton kernel is SUPPOSED
    to compute: for each query token i, attend to k/v[:kv_end[i]] with GQA.
    Output shape: (num_q, num_heads_q, d) fp32.
    """
    num_q, num_heads_q, d = q.shape
    num_heads_kv = k.shape[1]
    gqa = num_heads_q // num_heads_kv
    scale = 1.0 / (d ** 0.5)
    out = torch.zeros_like(q)
    for qi in range(num_q):
        kv_end = int(kv_end_per_query[qi].item())
        for h in range(num_heads_q):
            kh = h // gqa
            q_vec = q[qi, h]                    # (d,)
            k_mat = k[:kv_end, kh]              # (kv_end, d)
            v_mat = v[:kv_end, kh]              # (kv_end, d)
            scores = (q_vec @ k_mat.T) * scale  # (kv_end,)
            weights = torch.softmax(scores, dim=-1)
            out[qi, h] = weights @ v_mat
    return out


def run_one_case(
    num_tokens: int,
    num_heads_q: int = 16,
    num_heads_kv: int = 8,
    head_size: int = 128,
    block_size: int = 16,
    bits: int = 8,
    dtype: torch.dtype = torch.bfloat16,
    seed: int = 0,
) -> dict:
    device = "cuda"
    torch.manual_seed(seed)

    # Random Q K V
    q = torch.randn(num_tokens, num_heads_q, head_size, dtype=dtype, device=device)
    k = torch.randn(num_tokens, num_heads_kv, head_size, dtype=dtype, device=device)
    v = torch.randn(num_tokens, num_heads_kv, head_size, dtype=dtype, device=device)

    # Paged cache with enough blocks to hold num_tokens
    num_blocks = max(1, (num_tokens + block_size - 1) // block_size) + 1
    cache_k = torch.zeros(num_blocks, block_size, num_heads_kv, head_size,
                          dtype=torch.uint8, device=device)
    cache_v = torch.zeros(num_blocks, block_size, num_heads_kv, head_size,
                          dtype=torch.int8, device=device)  # Step 1: int8 V
    cache_k_norm = torch.zeros(num_blocks, block_size, num_heads_kv,
                               dtype=torch.float32, device=device)
    cache_v_scale = torch.zeros(num_blocks, block_size, num_heads_kv,
                                dtype=torch.float32, device=device)

    # Put this seq's tokens starting at block 1 slot 0 (skip block 0 to
    # mimic vLLM behaviour where block 0 is often reserved / warmup).
    first_slot = block_size  # global slot 16 for block_size=16
    slot_mapping = torch.arange(
        first_slot, first_slot + num_tokens, dtype=torch.int64, device=device,
    )

    # block_table: one seq, stride = num_blocks (use all allocated entries)
    block_table = torch.zeros(1, num_blocks, dtype=torch.int32, device=device)
    # Logical block i -> physical block i+1 (matching vLLM's block 0 reserve)
    for i in range(num_blocks - 1):
        block_table[0, i] = i + 1

    seq_lens = torch.tensor([num_tokens], dtype=torch.int32, device=device)
    query_start_loc = torch.tensor([0, num_tokens], dtype=torch.int32, device=device)

    codebook = GaussianCodebook(
        head_dim=head_size, bits=bits, seed=42, dtype=dtype, device=device,
    )

    # --- 1) Store ---------------------------------------------------------
    turboquant_store_kv(
        new_k=k,
        new_v=v,
        cache_k=cache_k,
        cache_v=cache_v,
        cache_k_norm=cache_k_norm,
        cache_v_scale=cache_v_scale,
        slot_mapping=slot_mapping,
        codebook=codebook,
        block_size=block_size,
    )
    torch.cuda.synchronize()

    # --- 2) Triton attend over the whole prefill -------------------------
    attn_out = turboquant_paged_attention(
        q=q,
        cache_k=cache_k,
        cache_v=cache_v,
        cache_k_norm=cache_k_norm,
        cache_v_scale=cache_v_scale,
        block_table=block_table,
        seq_lens=seq_lens,
        query_start_loc=query_start_loc,
        codebook=codebook,
        scale=None,
    )
    torch.cuda.synchronize()

    # --- 3) Reference attention using the SAME dequantised K/V -----------
    k_dq, v_dq = dequant_kv_reference(
        cache_k=cache_k,
        cache_k_norm=cache_k_norm,
        cache_v=cache_v,
        cache_v_scale=cache_v_scale,
        codebook=codebook,
        block_table=block_table,
        seq_len=num_tokens,
    )  # (num_tokens, num_heads_kv, d) fp32 each

    kv_end_per_query = torch.arange(1, num_tokens + 1, dtype=torch.int32, device=device)
    ref_out = reference_attention(q.float(), k_dq, v_dq, kv_end_per_query)

    # --- 4) Compare -------------------------------------------------------
    diff = (attn_out.float() - ref_out).abs()
    return {
        "num_tokens": num_tokens,
        "bits": bits,
        "out_max_abs_err": float(diff.max().item()),
        "out_mean_abs_err": float(diff.mean().item()),
        "ref_out_std": float(ref_out.std().item()),
        "attn_out_std": float(attn_out.float().std().item()),
        "rel_err": float((diff.mean() / ref_out.abs().mean().clamp(min=1e-6)).item()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-tokens", type=int, default=None,
                    help="if given, run only this size; else run a sweep")
    ap.add_argument("--bits", type=int, default=8)
    args = ap.parse_args()

    cases = [args.num_tokens] if args.num_tokens else [1, 4, 16, 32, 61, 128]

    print(f"{'num_tokens':>12} {'max_err':>10} {'mean_err':>10} "
          f"{'rel_err':>10} {'ref_std':>10} {'attn_std':>10}")
    for n in cases:
        r = run_one_case(num_tokens=n, bits=args.bits)
        print(f"{r['num_tokens']:>12} "
              f"{r['out_max_abs_err']:>10.4f} "
              f"{r['out_mean_abs_err']:>10.4f} "
              f"{r['rel_err']:>10.4%} "
              f"{r['ref_out_std']:>10.4f} "
              f"{r['attn_out_std']:>10.4f}")


if __name__ == "__main__":
    main()
