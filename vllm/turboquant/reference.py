# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pure-PyTorch reference implementations of the TurboQuant pipeline.

These mirror the Triton kernels in ``store.py`` / ``attend_tc.py`` /
``attend_split_tc.py`` but use plain tensor ops, so they run on CPU
without Triton installed. They are slow and unoptimized -- do not use
at inference time -- but they serve three purposes:

1. **CI on Mac / CPU boxes** for algorithm correctness.
2. **Numerical baseline on GPU**: on B200 we can compare the Triton
   kernel output against this reference element-wise (expected diff
   ~1e-2 in bf16, ~1e-5 in fp32).
3. **Debugging**: if a kernel regresses, bisect by replacing one
   stage with its reference and re-running.

Paper: Zandieh et al., arXiv:2504.19874, Algorithm 1 / Algorithm 2,
applied uniformly to K and V.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from vllm.turboquant.codebook import QuantState
    from vllm.turboquant.outlier import SplitQuantState


# ---------------------------------------------------------------------------
# Quantization primitives (Algorithm 1 / 2, vector form)
# ---------------------------------------------------------------------------
def quantize_prod(
    x: torch.Tensor, state: "QuantState"
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Algorithm 2 quantize. Returns (idx, qjl, x_norm, r_norm).

    x shape (..., d); returns are (..., d), (..., d), (...,), (...,).
    QJL sign stored as {-1, +1} int tensor -- the Triton kernel
    bit-packs this but the algorithm is the same.
    """
    d = x.shape[-1]
    x_norm = x.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    rotated = (x / x_norm) @ state.Pi

    boundaries = state.boundaries.to(x.dtype)
    idx = torch.searchsorted(boundaries, rotated)
    idx = idx.clamp(max=state.codebook.shape[0] - 1).to(torch.int32)

    r = rotated - state.codebook.to(x.dtype)[idx]
    r_norm = r.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    qjl = ((r / r_norm) @ state.S.t()).sign()
    qjl = torch.where(qjl == 0, torch.ones_like(qjl), qjl).to(torch.int8)
    return idx, qjl, x_norm.squeeze(-1), r_norm.squeeze(-1)


def quantize_mse(
    x: torch.Tensor, state: "QuantState"
) -> tuple[torch.Tensor, torch.Tensor]:
    """Algorithm 1 quantize. Returns (idx, x_norm)."""
    x_norm = x.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    rotated = (x / x_norm) @ state.Pi
    boundaries = state.boundaries.to(x.dtype)
    idx = torch.searchsorted(boundaries, rotated)
    idx = idx.clamp(max=state.codebook.shape[0] - 1).to(torch.int32)
    return idx, x_norm.squeeze(-1)


def dequantize_prod(
    idx: torch.Tensor,
    qjl: torch.Tensor,
    x_norm: torch.Tensor,
    r_norm: torch.Tensor,
    state: "QuantState",
) -> torch.Tensor:
    """Reconstruct x_approx in original (unrotated, ||x||-scaled) space.

    Returns shape (..., d). idx/qjl are (..., d) integers; x_norm /
    r_norm are (...,).
    """
    d = state.head_dim
    qjl_coef = math.sqrt(math.pi / 2.0) / d
    dtype = x_norm.dtype
    rotated_approx = state.codebook.to(dtype)[idx] + (
        qjl_coef * r_norm.unsqueeze(-1) * (qjl.to(dtype) @ state.S.to(dtype))
    )
    return (rotated_approx @ state.Pi_T.to(dtype)) * x_norm.unsqueeze(-1)


def dequantize_mse(
    idx: torch.Tensor, x_norm: torch.Tensor, state: "QuantState",
) -> torch.Tensor:
    dtype = x_norm.dtype
    rotated_approx = state.codebook.to(dtype)[idx]
    return (rotated_approx @ state.Pi_T.to(dtype)) * x_norm.unsqueeze(-1)


# ---------------------------------------------------------------------------
# Homogeneous attend reference (one QuantState, Q_prod on both K and V)
# ---------------------------------------------------------------------------
def turboquant_attend_reference(
    q: torch.Tensor,          # (T_q, H_q, d)
    k: torch.Tensor,          # (T_kv, H_kv, d)  -- full cache for one sequence
    v: torch.Tensor,          # (T_kv, H_kv, d)
    state: "QuantState",
    kv_end_per_query: torch.Tensor,  # (T_q,) causal cutoff per query
) -> torch.Tensor:
    """Naive single-sequence attention with Q_prod quantization.

    No paged addressing -- all K/V for one sequence laid out as
    contiguous (T_kv, H_kv, d). ``kv_end_per_query[i]`` bounds the
    KV slots visible to query i (causal mask).

    Returns (T_q, H_q, d).
    """
    T_q, H_q, d = q.shape
    T_kv, H_kv, _ = k.shape
    gqa_group = H_q // H_kv
    inv_sqrt_d = 1.0 / math.sqrt(float(d))

    # Quantize + dequantize K and V (this models what the paged store
    # + attend kernel does end-to-end).
    if state.algo == "prod":
        k_idx, k_qjl, k_norm, k_rnorm = quantize_prod(k, state)
        v_idx, v_qjl, v_norm, v_rnorm = quantize_prod(v, state)
        k_approx = dequantize_prod(k_idx, k_qjl, k_norm, k_rnorm, state)
        v_approx = dequantize_prod(v_idx, v_qjl, v_norm, v_rnorm, state)
    else:
        k_idx, k_norm = quantize_mse(k, state)
        v_idx, v_norm = quantize_mse(v, state)
        k_approx = dequantize_mse(k_idx, k_norm, state)
        v_approx = dequantize_mse(v_idx, v_norm, state)

    # Repeat K/V for GQA.
    if gqa_group > 1:
        k_approx = k_approx.repeat_interleave(gqa_group, dim=1)
        v_approx = v_approx.repeat_interleave(gqa_group, dim=1)

    # Attention per (query, head). q: (T_q, H_q, d), k: (T_kv, H_q, d).
    # logits[i, h, j] = <q[i, h], k[j, h]> / sqrt(d)
    logits = torch.einsum("ihd,jhd->ihj", q, k_approx) * inv_sqrt_d
    # Causal mask per query.
    kv_idx = torch.arange(T_kv, device=q.device)
    mask = kv_idx[None, None, :] >= kv_end_per_query[:, None, None]
    logits = logits.masked_fill(mask, float("-inf"))
    probs = torch.softmax(logits, dim=-1)
    out = torch.einsum("ihj,jhd->ihd", probs, v_approx)
    return out


# ---------------------------------------------------------------------------
# Split attend reference (two QuantState per side, outlier+regular)
# ---------------------------------------------------------------------------
def turboquant_attend_split_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    state_k: "SplitQuantState",
    state_v: "SplitQuantState",
    kv_end_per_query: torch.Tensor,
) -> torch.Tensor:
    """Naive single-sequence split attention with Q_prod on each slice.

    For paper §4.3's outlier channel split. K and V must share the
    same outlier index set (matches the kernel constraint).
    """
    assert state_k.outlier_idx.equal(state_v.outlier_idx)
    T_q, H_q, d = q.shape
    T_kv, H_kv, _ = k.shape
    gqa_group = H_q // H_kv
    inv_sqrt_d = 1.0 / math.sqrt(float(d))

    # Gather slices for K and V.
    outlier_idx = state_k.outlier_idx.to(k.device)
    regular_idx = state_k.regular_idx.to(k.device)
    k_out = k.index_select(-1, outlier_idx)
    k_reg = k.index_select(-1, regular_idx)
    v_out = v.index_select(-1, outlier_idx)
    v_reg = v.index_select(-1, regular_idx)

    # Quantize + dequantize per slice.
    if state_k.algo == "prod":
        k_out_idx, k_out_qjl, k_out_norm, k_out_rn = quantize_prod(
            k_out, state_k.state_out
        )
        k_reg_idx, k_reg_qjl, k_reg_norm, k_reg_rn = quantize_prod(
            k_reg, state_k.state_reg
        )
        v_out_idx, v_out_qjl, v_out_norm, v_out_rn = quantize_prod(
            v_out, state_v.state_out
        )
        v_reg_idx, v_reg_qjl, v_reg_norm, v_reg_rn = quantize_prod(
            v_reg, state_v.state_reg
        )
        k_out_approx = dequantize_prod(
            k_out_idx, k_out_qjl, k_out_norm, k_out_rn, state_k.state_out,
        )
        k_reg_approx = dequantize_prod(
            k_reg_idx, k_reg_qjl, k_reg_norm, k_reg_rn, state_k.state_reg,
        )
        v_out_approx = dequantize_prod(
            v_out_idx, v_out_qjl, v_out_norm, v_out_rn, state_v.state_out,
        )
        v_reg_approx = dequantize_prod(
            v_reg_idx, v_reg_qjl, v_reg_norm, v_reg_rn, state_v.state_reg,
        )
    else:
        k_out_idx, k_out_norm = quantize_mse(k_out, state_k.state_out)
        k_reg_idx, k_reg_norm = quantize_mse(k_reg, state_k.state_reg)
        v_out_idx, v_out_norm = quantize_mse(v_out, state_v.state_out)
        v_reg_idx, v_reg_norm = quantize_mse(v_reg, state_v.state_reg)
        k_out_approx = dequantize_mse(k_out_idx, k_out_norm, state_k.state_out)
        k_reg_approx = dequantize_mse(k_reg_idx, k_reg_norm, state_k.state_reg)
        v_out_approx = dequantize_mse(v_out_idx, v_out_norm, state_v.state_out)
        v_reg_approx = dequantize_mse(v_reg_idx, v_reg_norm, state_v.state_reg)

    # Scatter back to full head_dim.
    k_approx = torch.zeros_like(k)
    k_approx.index_copy_(-1, outlier_idx, k_out_approx)
    k_approx.index_copy_(-1, regular_idx, k_reg_approx)
    v_approx = torch.zeros_like(v)
    v_approx.index_copy_(-1, outlier_idx, v_out_approx)
    v_approx.index_copy_(-1, regular_idx, v_reg_approx)

    # Repeat for GQA.
    if gqa_group > 1:
        k_approx = k_approx.repeat_interleave(gqa_group, dim=1)
        v_approx = v_approx.repeat_interleave(gqa_group, dim=1)

    logits = torch.einsum("ihd,jhd->ihj", q, k_approx) * inv_sqrt_d
    kv_idx = torch.arange(T_kv, device=q.device)
    mask = kv_idx[None, None, :] >= kv_end_per_query[:, None, None]
    logits = logits.masked_fill(mask, float("-inf"))
    probs = torch.softmax(logits, dim=-1)
    out = torch.einsum("ihj,jhd->ihd", probs, v_approx)
    return out


# ---------------------------------------------------------------------------
# Naive full-precision attention (for quality-loss baselines)
# ---------------------------------------------------------------------------
def naive_attend(
    q: torch.Tensor,           # (T_q, H_q, d)
    k: torch.Tensor,           # (T_kv, H_kv, d)
    v: torch.Tensor,           # (T_kv, H_kv, d)
    kv_end_per_query: torch.Tensor,
) -> torch.Tensor:
    """Exact attention on unquantized K/V -- the ground truth for
    measuring quantization-induced quality loss."""
    T_q, H_q, d = q.shape
    T_kv, H_kv, _ = k.shape
    gqa_group = H_q // H_kv
    inv_sqrt_d = 1.0 / math.sqrt(float(d))

    if gqa_group > 1:
        k = k.repeat_interleave(gqa_group, dim=1)
        v = v.repeat_interleave(gqa_group, dim=1)

    logits = torch.einsum("ihd,jhd->ihj", q, k) * inv_sqrt_d
    kv_idx = torch.arange(T_kv, device=q.device)
    mask = kv_idx[None, None, :] >= kv_end_per_query[:, None, None]
    logits = logits.masked_fill(mask, float("-inf"))
    probs = torch.softmax(logits, dim=-1)
    return torch.einsum("ihj,jhd->ihd", probs, v)
