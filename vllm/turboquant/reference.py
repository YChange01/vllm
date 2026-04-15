# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pure-PyTorch reference for TurboQuant paged attention.

Purpose:
  1. Ground-truth algorithm that we compare the Triton kernel against.
  2. Runs anywhere (CPU, MPS, CUDA) for algorithmic debugging.

Performance: naive Python loops, ~100x slower than the Triton kernel. Not
intended for serving.
"""

from __future__ import annotations

import math

import torch

from vllm.turboquant.codebook import GaussianCodebook


def store_kv_reference(
    new_k: torch.Tensor,
    new_v: torch.Tensor,
    cache_k: torch.Tensor,
    cache_v: torch.Tensor,
    slot_mapping: torch.Tensor,
    codebook: GaussianCodebook,
    block_size: int,
) -> None:
    """Quantize ``new_k`` into ``cache_k`` (uint8 idx) and copy ``new_v`` into
    ``cache_v``, following ``slot_mapping``."""
    num_tokens = new_k.shape[0]
    idx = codebook.quantize_to_idx(new_k)  # (num_tokens, num_kv_heads, head_dim) uint8

    for i in range(num_tokens):
        slot = int(slot_mapping[i].item())
        if slot < 0:
            continue
        b = slot // block_size
        off = slot % block_size
        cache_k[b, off] = idx[i]
        cache_v[b, off] = new_v[i]


def paged_attention_reference(
    q: torch.Tensor,
    cache_k: torch.Tensor,
    cache_v: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    codebook: GaussianCodebook,
    scale: float | None = None,
) -> torch.Tensor:
    """Reference: gather all cached K/V per seq, dequantize, standard attention."""
    num_seqs, num_q_heads, head_dim = q.shape
    _, block_size, num_kv_heads, _ = cache_k.shape
    gqa_group = num_q_heads // num_kv_heads
    if scale is None:
        scale = 1.0 / math.sqrt(head_dim)

    # 码本 lookup + 反旋转
    cb = codebook.codebook  # (K_CB,)
    H = codebook.H
    signs = codebook.signs

    out = torch.empty_like(q)
    for s in range(num_seqs):
        seq_len = int(seq_lens[s].item())
        num_blocks = (seq_len + block_size - 1) // block_size

        k_list = []
        v_list = []
        for bi in range(num_blocks):
            phys = int(block_table[s, bi].item())
            take = min(block_size, seq_len - bi * block_size)
            k_idx = cache_k[phys, :take].long()  # (take, num_kv_heads, head_dim)
            rk = cb[k_idx]                        # (take, num_kv_heads, head_dim)
            k = (rk @ H) * signs                  # 反旋转
            k_list.append(k)
            v_list.append(cache_v[phys, :take])

        k_full = torch.cat(k_list, dim=0)   # (seq_len, num_kv_heads, head_dim)
        v_full = torch.cat(v_list, dim=0)

        for qh in range(num_q_heads):
            kvh = qh // gqa_group
            q_vec = q[s, qh]                                # (head_dim,)
            k_mat = k_full[:, kvh]                          # (seq_len, head_dim)
            v_mat = v_full[:, kvh]                          # (seq_len, head_dim)
            logits = (k_mat @ q_vec) * scale                # (seq_len,)
            weights = torch.softmax(logits.float(), dim=-1).to(v_mat.dtype)
            out[s, qh] = weights @ v_mat

    return out
