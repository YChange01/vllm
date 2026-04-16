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
    cache_k_qjl_sign: torch.Tensor,
    cache_k_rnorm: torch.Tensor,
    codebook: GaussianCodebook,
    block_table: torch.Tensor,
    seq_len: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Dequantise the K cache using the TurboQuant Algorithm 2 formula.

    Per (slot, head):
        r_unit_approx  = sqrt(pi/2)/d * S^T @ qjl_sign
        rotated_approx = codebook[idx] + r_norm * r_unit_approx
        k              = ((rotated_approx @ H) * signs) * (k_norm / sqrt(d))
    """
    import math
    num_blocks, block_size, num_kv_heads, head_size = cache_k.shape
    inv_sqrt_d = 1.0 / (head_size ** 0.5)
    qjl_scale_const = math.sqrt(math.pi / 2.0) / float(head_size)

    cb = codebook.codebook.to(torch.float32)
    H = codebook.H.to(torch.float32)
    S = codebook.S.to(torch.float32)
    signs = codebook.signs.to(torch.float32)

    k_full = torch.zeros(seq_len, num_kv_heads, head_size,
                         dtype=torch.float32, device=cache_k.device)
    v_full = torch.zeros_like(k_full)
    for p in range(seq_len):
        block_i = p // block_size
        tok_in_block = p % block_size
        phys_block = int(block_table[0, block_i].item())
        for h in range(num_kv_heads):
            idx = cache_k[phys_block, tok_in_block, h].long()
            rk_main = cb[idx]
            qjl_sign = cache_k_qjl_sign[phys_block, tok_in_block, h].float()
            r_norm_h = float(cache_k_rnorm[phys_block, tok_in_block, h].item())
            r_unit_approx = qjl_scale_const * (S.T @ qjl_sign)
            rotated_approx = rk_main + r_norm_h * r_unit_approx
            k_unrot = rotated_approx @ H
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
    # Algorithm 2: QJL sign bits + residual L2 norm
    cache_k_qjl_sign = torch.zeros(num_blocks, block_size, num_heads_kv, head_size,
                                   dtype=torch.int8, device=device)
    cache_k_rnorm = torch.zeros(num_blocks, block_size, num_heads_kv,
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
        cache_k_qjl_sign=cache_k_qjl_sign,
        cache_k_rnorm=cache_k_rnorm,
        slot_mapping=slot_mapping,
        codebook=codebook,
        block_size=block_size,
    )
    torch.cuda.synchronize()

    # --- 2) Triton attend: Algorithm 1 baseline (no QJL residual) --------
    # Zero the residual-norm tensor to cancel the QJL term -- effectively
    # reverts to pure MSE TurboQuant (paper Algorithm 1).
    rnorm_backup = cache_k_rnorm.clone()
    cache_k_rnorm.zero_()
    attn_out_mse = turboquant_paged_attention(
        q=q,
        cache_k=cache_k,
        cache_v=cache_v,
        cache_k_norm=cache_k_norm,
        cache_v_scale=cache_v_scale,
        cache_k_qjl_sign=cache_k_qjl_sign,
        cache_k_rnorm=cache_k_rnorm,
        block_table=block_table,
        seq_lens=seq_lens,
        query_start_loc=query_start_loc,
        codebook=codebook,
        scale=None,
    )
    torch.cuda.synchronize()
    cache_k_rnorm.copy_(rnorm_backup)

    # --- 2b) Triton attend WITH QJL residual (Algorithm 2) --------------
    attn_out = turboquant_paged_attention(
        q=q,
        cache_k=cache_k,
        cache_v=cache_v,
        cache_k_norm=cache_k_norm,
        cache_v_scale=cache_v_scale,
        cache_k_qjl_sign=cache_k_qjl_sign,
        cache_k_rnorm=cache_k_rnorm,
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
        cache_k_qjl_sign=cache_k_qjl_sign,
        cache_k_rnorm=cache_k_rnorm,
        codebook=codebook,
        block_table=block_table,
        seq_len=num_tokens,
    )  # (num_tokens, num_heads_kv, d) fp32 each

    kv_end_per_query = torch.arange(1, num_tokens + 1, dtype=torch.int32, device=device)
    ref_out = reference_attention(q.float(), k_dq, v_dq, kv_end_per_query)

    # --- 3b) FP ground-truth attention on the ORIGINAL unquantized K/V ----
    # ref_out uses the quantized+dequantized K/V that the kernel sees, so
    # it mostly measures Triton-vs-Python arithmetic. fp_ref_out uses the
    # UNQUANTIZED K/V; comparing attn_out against fp_ref_out is the real
    # end-to-end quantization error number.
    fp_ref_out = reference_attention(
        q.float(), k.float(), v.float(), kv_end_per_query,
    )

    # --- 4) Compare -------------------------------------------------------
    fp_ref_abs_mean = fp_ref_out.abs().mean().clamp(min=1e-6)
    mse_diff = (attn_out_mse.float() - fp_ref_out).abs()      # Algorithm 1
    a2_diff = (attn_out.float() - fp_ref_out).abs()           # Algorithm 2
    kern_diff = (attn_out.float() - ref_out).abs()
    return {
        "num_tokens": num_tokens,
        "bits": bits,
        "kern_rel_err": float(
            (kern_diff.mean() / ref_out.abs().mean().clamp(min=1e-6)).item()
        ),
        "mse_rel_err": float((mse_diff.mean() / fp_ref_abs_mean).item()),
        "a2_rel_err": float((a2_diff.mean() / fp_ref_abs_mean).item()),
        "mse_max_err": float(mse_diff.max().item()),
        "a2_max_err": float(a2_diff.max().item()),
        "fp_ref_std": float(fp_ref_out.std().item()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-tokens", type=int, default=None,
                    help="if given, run only this size; else run a sweep")
    ap.add_argument("--bits", type=int, default=8)
    args = ap.parse_args()

    cases = [args.num_tokens] if args.num_tokens else [1, 4, 16, 32, 61, 128]

    # Columns:
    #   kern_rel   : Triton vs PyTorch on SAME quantized K/V (bf16 floor)
    #   mse_rel    : Triton vs UNQUANTIZED K/V with r_norm=0
    #                 (TurboQuant Algorithm 1, pure MSE)
    #   a2_rel     : Triton vs UNQUANTIZED K/V with QJL residual active
    #                 (TurboQuant Algorithm 2, paper's recipe)
    #   reduction  : 1 - a2_rel / mse_rel (gain from QJL)
    print(f"{'num_tokens':>10} {'kern_rel':>10} {'mse_rel':>10} "
          f"{'a2_rel':>10} {'reduction':>10} "
          f"{'mse_max':>8} {'a2_max':>8}")
    for n in cases:
        r = run_one_case(num_tokens=n, bits=args.bits)
        reduction = 1.0 - r["a2_rel_err"] / max(r["mse_rel_err"], 1e-12)
        print(f"{r['num_tokens']:>10} "
              f"{r['kern_rel_err']:>10.4%} "
              f"{r['mse_rel_err']:>10.4%} "
              f"{r['a2_rel_err']:>10.4%} "
              f"{reduction:>10.2%} "
              f"{r['mse_max_err']:>8.4f} "
              f"{r['a2_max_err']:>8.4f}")


if __name__ == "__main__":
    main()
