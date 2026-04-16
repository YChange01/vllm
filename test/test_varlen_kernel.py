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
    cache_k_resid_sign: torch.Tensor,
    cache_k_resid_scale: torch.Tensor,
    codebook: GaussianCodebook,
    block_table: torch.Tensor,
    seq_len: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Dequantise the K cache for sequence 0 up to seq_len, using the SAME
    formula the Triton attend kernel does.

    Step 2 adds the residual correction: in rotated space we reconstruct
    rk ≈ codebook[idx] + sign * scale before un-rotation.
    """
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
            # Step 2: 1-bit residual correction in rotated space
            rk_sign = cache_k_resid_sign[phys_block, tok_in_block, h].float()
            rk_scale = float(cache_k_resid_scale[phys_block, tok_in_block, h].item())
            rk = rk + rk_sign * rk_scale
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
    # Step 2: 1-bit K residual sign + per-(slot,head) scale
    cache_k_resid_sign = torch.zeros(num_blocks, block_size, num_heads_kv, head_size,
                                     dtype=torch.int8, device=device)
    cache_k_resid_scale = torch.zeros(num_blocks, block_size, num_heads_kv,
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
        cache_k_resid_sign=cache_k_resid_sign,
        cache_k_resid_scale=cache_k_resid_scale,
        slot_mapping=slot_mapping,
        codebook=codebook,
        block_size=block_size,
    )
    torch.cuda.synchronize()

    # --- 2) Triton attend: first WITHOUT residual (Step 1 baseline) ------
    # Zero the scale tensor to cancel sign*scale in the kernel -- keeps the
    # dequant path bit-identical except the residual contribution.
    scale_backup = cache_k_resid_scale.clone()
    cache_k_resid_scale.zero_()
    attn_out_step1 = turboquant_paged_attention(
        q=q,
        cache_k=cache_k,
        cache_v=cache_v,
        cache_k_norm=cache_k_norm,
        cache_v_scale=cache_v_scale,
        cache_k_resid_sign=cache_k_resid_sign,
        cache_k_resid_scale=cache_k_resid_scale,
        block_table=block_table,
        seq_lens=seq_lens,
        query_start_loc=query_start_loc,
        codebook=codebook,
        scale=None,
    )
    torch.cuda.synchronize()
    cache_k_resid_scale.copy_(scale_backup)

    # --- 2b) Triton attend WITH residual (Step 2) ------------------------
    attn_out = turboquant_paged_attention(
        q=q,
        cache_k=cache_k,
        cache_v=cache_v,
        cache_k_norm=cache_k_norm,
        cache_v_scale=cache_v_scale,
        cache_k_resid_sign=cache_k_resid_sign,
        cache_k_resid_scale=cache_k_resid_scale,
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
        cache_k_resid_sign=cache_k_resid_sign,
        cache_k_resid_scale=cache_k_resid_scale,
        codebook=codebook,
        block_table=block_table,
        seq_len=num_tokens,
    )  # (num_tokens, num_heads_kv, d) fp32 each

    kv_end_per_query = torch.arange(1, num_tokens + 1, dtype=torch.int32, device=device)
    ref_out = reference_attention(q.float(), k_dq, v_dq, kv_end_per_query)

    # --- 3b) FP ground-truth attention on the ORIGINAL unquantized K/V ----
    # This is what Step 2 should actually move. The ref_out above uses the
    # same quantized+dequantized K/V as the Triton kernel, so it measures
    # kernel arithmetic fidelity (~0.14% bf16 floor) but NOT quantization
    # quality. Comparing attn_out against this fp_ref_out shows the true
    # end-to-end quantization error that the residual is supposed to shrink.
    fp_ref_out = reference_attention(
        q.float(), k.float(), v.float(), kv_end_per_query,
    )

    # --- 4) Compare -------------------------------------------------------
    fp_ref_abs_mean = fp_ref_out.abs().mean().clamp(min=1e-6)
    step1_diff = (attn_out_step1.float() - fp_ref_out).abs()
    step2_diff = (attn_out.float() - fp_ref_out).abs()
    kern_diff = (attn_out.float() - ref_out).abs()
    return {
        "num_tokens": num_tokens,
        "bits": bits,
        "kern_rel_err": float(
            (kern_diff.mean() / ref_out.abs().mean().clamp(min=1e-6)).item()
        ),
        "step1_rel_err": float((step1_diff.mean() / fp_ref_abs_mean).item()),
        "step2_rel_err": float((step2_diff.mean() / fp_ref_abs_mean).item()),
        "step1_max_err": float(step1_diff.max().item()),
        "step2_max_err": float(step2_diff.max().item()),
        "fp_ref_std": float(fp_ref_out.std().item()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-tokens", type=int, default=None,
                    help="if given, run only this size; else run a sweep")
    ap.add_argument("--bits", type=int, default=8)
    args = ap.parse_args()

    cases = [args.num_tokens] if args.num_tokens else [1, 4, 16, 32, 61, 128]

    # Four error columns:
    #   kern_rel    : Triton vs PyTorch on SAME quantized K/V
    #                 (arithmetic fidelity, ~0.14% bf16 floor)
    #   step1_rel   : Triton vs UNQUANTIZED K/V with residual scale zeroed
    #                 (Step 1 equivalent -- Lloyd-Max + V int8, no residual)
    #   step2_rel   : Triton vs UNQUANTIZED K/V with residual active
    #                 (Step 2 -- should be smaller than step1_rel)
    #   reduction   : 1 - step2_rel/step1_rel (how much residual helped)
    print(f"{'num_tokens':>10} {'kern_rel':>10} {'step1_rel':>10} "
          f"{'step2_rel':>10} {'reduction':>10} "
          f"{'s1_max':>8} {'s2_max':>8}")
    for n in cases:
        r = run_one_case(num_tokens=n, bits=args.bits)
        reduction = 1.0 - r["step2_rel_err"] / max(r["step1_rel_err"], 1e-12)
        print(f"{r['num_tokens']:>10} "
              f"{r['kern_rel_err']:>10.4%} "
              f"{r['step1_rel_err']:>10.4%} "
              f"{r['step2_rel_err']:>10.4%} "
              f"{reduction:>10.2%} "
              f"{r['step1_max_err']:>8.4f} "
              f"{r['step2_max_err']:>8.4f}")


if __name__ == "__main__":
    main()
