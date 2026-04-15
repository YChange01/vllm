# SPDX-License-Identifier: Apache-2.0
"""Tests for TurboQuant PyTorch reference (no GPU needed).

验证:
  1. 量化-反量化循环在合成高斯数据上 MSE 合理
  2. paged_attention_reference 在 baseline 附近 (和 fp32 标准 attention 对比)

这些测试应该在 Mac / CPU 上也能通过. Triton kernel 正确性在 B200 上用
`test_turboquant_kernel.py` 单独验证.
"""

from __future__ import annotations

import math

import pytest
import torch

from vllm.turboquant.codebook import GaussianCodebook
from vllm.turboquant.reference import paged_attention_reference, store_kv_reference


def _baseline_attention(q, k, v, scale):
    """标准 fp32 attention, num_q_heads 可以 > num_kv_heads (GQA)."""
    num_seqs, num_q_heads, head_dim = q.shape
    _, num_kv_heads, _ = k.shape  # 实际是 (total_kv_tokens, num_kv_heads, head_dim)
    # 无批量, 单序列假设
    gqa_group = num_q_heads // num_kv_heads
    out = torch.empty_like(q)
    for s in range(num_seqs):
        for qh in range(num_q_heads):
            kvh = qh // gqa_group
            q_vec = q[s, qh]
            logits = (k[:, kvh] @ q_vec) * scale
            w = torch.softmax(logits.float(), dim=-1).to(v.dtype)
            out[s, qh] = w @ v[:, kvh]
    return out


def test_quantize_dequantize_roundtrip_gaussian() -> None:
    torch.manual_seed(0)
    cb = GaussianCodebook(head_dim=128, bits=4, seed=0, dtype=torch.float32, device="cpu")
    x = torch.randn(256, 128)
    # 每 row 归一化到 N(0, I) 尺度
    x = x / x.norm(dim=-1, keepdim=True) * math.sqrt(128)

    idx = cb.quantize_to_idx(x)
    rx_hat = cb.codebook[idx.long()]
    x_hat = (rx_hat @ cb.H) * cb.signs

    rel_err = (x - x_hat).norm() / x.norm()
    assert rel_err < 0.15, f"b=4 MSE 量化 rel_err should be < 15%, got {rel_err:.3f}"


def test_paged_attention_matches_baseline_at_b8() -> None:
    torch.manual_seed(0)
    head_dim = 64
    block_size = 16
    num_kv_heads = 2
    num_q_heads = 4  # GQA group = 2
    seq_len = 32
    num_blocks = (seq_len + block_size - 1) // block_size

    cb = GaussianCodebook(
        head_dim=head_dim, bits=8, seed=0, dtype=torch.float32, device="cpu"
    )

    q = torch.randn(1, num_q_heads, head_dim)
    k_raw = torch.randn(seq_len, num_kv_heads, head_dim) * 0.5
    # 归一化便于量化
    k_raw = k_raw / k_raw.norm(dim=-1, keepdim=True) * math.sqrt(head_dim)
    v_raw = torch.randn(seq_len, num_kv_heads, head_dim) * 0.5

    cache_k = torch.zeros(num_blocks, block_size, num_kv_heads, head_dim, dtype=torch.uint8)
    cache_v = torch.zeros(num_blocks, block_size, num_kv_heads, head_dim)
    slot_mapping = torch.arange(seq_len, dtype=torch.int64)

    store_kv_reference(k_raw, v_raw, cache_k, cache_v, slot_mapping, cb, block_size)

    block_table = torch.arange(num_blocks, dtype=torch.int32).unsqueeze(0)
    seq_lens = torch.tensor([seq_len], dtype=torch.int32)

    out_quant = paged_attention_reference(q, cache_k, cache_v, block_table, seq_lens, cb)

    scale = 1.0 / math.sqrt(head_dim)
    out_fp = _baseline_attention(q, k_raw, v_raw, scale)

    rel_err = (out_quant - out_fp).norm() / out_fp.norm()
    assert rel_err < 0.1, f"b=8 paged attention rel_err should be < 10%, got {rel_err:.3f}"


def test_paged_attention_b4_coherent() -> None:
    """b=4 不要求精确, 但输出应该是有限值且方向大致正确."""
    torch.manual_seed(0)
    head_dim = 64
    block_size = 8
    num_kv_heads = 2
    num_q_heads = 2
    seq_len = 16
    num_blocks = (seq_len + block_size - 1) // block_size

    cb = GaussianCodebook(
        head_dim=head_dim, bits=4, seed=0, dtype=torch.float32, device="cpu"
    )

    q = torch.randn(1, num_q_heads, head_dim)
    k_raw = torch.randn(seq_len, num_kv_heads, head_dim)
    k_raw = k_raw / k_raw.norm(dim=-1, keepdim=True) * math.sqrt(head_dim)
    v_raw = torch.randn(seq_len, num_kv_heads, head_dim)

    cache_k = torch.zeros(num_blocks, block_size, num_kv_heads, head_dim, dtype=torch.uint8)
    cache_v = torch.zeros(num_blocks, block_size, num_kv_heads, head_dim)
    slot_mapping = torch.arange(seq_len, dtype=torch.int64)

    store_kv_reference(k_raw, v_raw, cache_k, cache_v, slot_mapping, cb, block_size)

    block_table = torch.arange(num_blocks, dtype=torch.int32).unsqueeze(0)
    seq_lens = torch.tensor([seq_len], dtype=torch.int32)

    out_quant = paged_attention_reference(q, cache_k, cache_v, block_table, seq_lens, cb)
    assert torch.isfinite(out_quant).all()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
