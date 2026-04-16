#!/usr/bin/env python3
"""Python-only simulator of the turboquant vLLM flow (no Triton).

Runs on CPU. Exercises the EXACT same calling pattern as vLLM does:

  step 1  (prefill): store_kv(K_0..K_{P-1})  -> paged_attention(q for all P)
  step 2  (decode):  store_kv(K_P)           -> paged_attention(q for token P)
  step 3  (decode):  store_kv(K_{P+1})       -> paged_attention(q for token P+1)
  ...

Uses a pure-PyTorch port of the store and attend kernels (same math as the
Triton version). If this passes locally, the bug is either in Triton-
specific kernel code or in a vLLM interaction we can't simulate. If this
FAILS, the algorithmic flow itself is broken under multi-call pattern.

The simulator also exercises multi-layer (each layer with its own seed),
which smoke test does not.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Inline copy of vllm/turboquant/codebook.py (Mac can't import vllm package).
# Keep in sync with the real file.
# ---------------------------------------------------------------------------
def _lloyd_max_gaussian(bits: int, n_iter: int = 100,
                        n_samples: int = 200_000, seed: int = 0) -> np.ndarray:
    K = 2 ** bits
    rng = np.random.default_rng(seed)
    samples = rng.standard_normal(n_samples)
    edges = np.quantile(samples, np.linspace(0, 1, K + 1))
    centroids = 0.5 * (edges[:-1] + edges[1:])
    for _ in range(n_iter):
        boundaries = 0.5 * (centroids[:-1] + centroids[1:])
        idx = np.digitize(samples, boundaries)
        for k in range(K):
            mask = idx == k
            if mask.any():
                centroids[k] = samples[mask].mean()
    return np.sort(centroids)


def _hadamard_matrix(d: int, dtype: torch.dtype) -> torch.Tensor:
    assert d & (d - 1) == 0, f"head_dim must be pow2, got {d}"
    H = torch.tensor([[1.0]], dtype=dtype)
    while H.shape[0] < d:
        H = torch.cat(
            [torch.cat([H, H], dim=1), torch.cat([H, -H], dim=1)], dim=0
        )
    return H / math.sqrt(d)


@dataclass
class QuantState:
    algo: str
    bits: int
    head_dim: int
    main_bits: int
    codebook: torch.Tensor
    boundaries: torch.Tensor
    H: torch.Tensor
    signs: torch.Tensor
    S: torch.Tensor | None

    @classmethod
    def make(cls, algo: str, bits: int, head_dim: int, seed: int,
             dtype: torch.dtype, device: torch.device) -> "QuantState":
        main_bits = bits - 1 if algo == "prod" else bits
        centroids = _lloyd_max_gaussian(main_bits)
        codebook = torch.tensor(centroids, dtype=dtype, device=device)
        boundaries = 0.5 * (codebook[:-1] + codebook[1:])
        H = _hadamard_matrix(head_dim, dtype).to(device)
        g = torch.Generator().manual_seed(seed)
        signs = (
            (torch.randint(0, 2, (head_dim,), generator=g) * 2 - 1)
            .to(dtype).to(device)
        )
        if algo == "prod":
            g_qjl = torch.Generator().manual_seed(seed ^ 0x51EE5B0)
            S = (torch.randn(head_dim, head_dim, generator=g_qjl)
                 .to(dtype).to(device))
        else:
            S = None
        return cls(algo=algo, bits=bits, head_dim=head_dim, main_bits=main_bits,
                   codebook=codebook, boundaries=boundaries, H=H, signs=signs, S=S)


# ---------------------------------------------------------------------------
# Pure-torch reference ports of the two Triton kernels
# ---------------------------------------------------------------------------
def _store_kv_ref(
    new_k: torch.Tensor,          # (T, H_kv, d) fp
    new_v: torch.Tensor,          # (T, H_kv, d) fp
    cache_k_idx: torch.Tensor,    # (num_blocks, bs, H_kv, d) uint8
    cache_k_norm: torch.Tensor,   # (num_blocks, bs, H_kv) fp32
    cache_v_idx: torch.Tensor,    # (num_blocks, bs, H_kv, d) int8
    cache_v_scale: torch.Tensor,  # (num_blocks, bs, H_kv) fp32
    cache_k_qjl_sign: torch.Tensor | None,
    cache_k_rnorm: torch.Tensor | None,
    slot_mapping: torch.Tensor,   # (T,) int64
    state: QuantState,
    block_size: int,
) -> None:
    T, H_kv, d = new_k.shape
    H = state.H.float()
    signs = state.signs.float()
    codebook = state.codebook.float()
    boundaries = state.boundaries.float()
    S = state.S.float() if state.S is not None else None

    for tok in range(T):
        slot = int(slot_mapping[tok].item())
        if slot < 0:
            continue
        b_i = slot // block_size
        off = slot % block_size
        for h in range(H_kv):
            k_vec = new_k[tok, h].float()
            v_vec = new_v[tok, h].float()

            k_norm = math.sqrt(max(float((k_vec * k_vec).sum()), 1e-12))
            cache_k_norm[b_i, off, h] = k_norm
            k_normed = k_vec * (math.sqrt(d) / k_norm)
            rotated = H @ (k_normed * signs)

            idx = torch.zeros(d, dtype=torch.int32)
            for i in range(codebook.numel() - 1):
                idx += (rotated > boundaries[i]).int()
            cache_k_idx[b_i, off, h] = idx.to(torch.uint8)

            if state.algo == "prod":
                rk_dq = codebook[idx]
                r = rotated - rk_dq
                r_norm = math.sqrt(max(float((r * r).sum()), 1e-12))
                r_unit = r / r_norm
                qjl_sign = torch.where(
                    S @ r_unit >= 0,
                    torch.ones(d),
                    -torch.ones(d),
                )
                cache_k_rnorm[b_i, off, h] = r_norm
                cache_k_qjl_sign[b_i, off, h] = qjl_sign.to(torch.int8)

            v_abs = v_vec.abs()
            v_scale = max(float(v_abs.max()) / 127.0, 1e-12)
            cache_v_scale[b_i, off, h] = v_scale
            cache_v_idx[b_i, off, h] = (v_vec / v_scale).to(torch.int8)


def _attend_ref(
    q: torch.Tensor,                  # (T_q, H_q, d)
    cache_k_idx: torch.Tensor,
    cache_k_norm: torch.Tensor,
    cache_v_idx: torch.Tensor,
    cache_v_scale: torch.Tensor,
    cache_k_qjl_sign: torch.Tensor | None,
    cache_k_rnorm: torch.Tensor | None,
    block_table: torch.Tensor,        # (num_seqs, num_blocks) int
    seq_id_per_query: torch.Tensor,   # (T_q,)
    kv_end_per_query: torch.Tensor,   # (T_q,)
    state: QuantState,
    block_size: int,
) -> torch.Tensor:
    T_q, H_q, d = q.shape
    H_kv = cache_k_idx.shape[2]
    gqa = H_q // H_kv

    H = state.H.float()
    signs = state.signs.float()
    codebook = state.codebook.float()
    S = state.S.float() if state.S is not None else None
    inv_d = 1.0 / d
    qjl_coef = math.sqrt(math.pi / 2.0) / d

    # Pre-rotate Q (matches turboquant_paged_attention Python-side rotate).
    q_f = q.float()
    q_rot = (q_f * signs) @ H.T  # (T_q, H_q, d)
    if state.algo == "prod":
        Sq = q_rot @ S.T
    else:
        Sq = None

    out = torch.zeros_like(q)
    for qi in range(T_q):
        seq = int(seq_id_per_query[qi].item())
        kv_end = int(kv_end_per_query[qi].item())
        for h in range(H_q):
            kvh = h // gqa
            q_rot_th = q_rot[qi, h]
            sq_th = Sq[qi, h] if Sq is not None else None

            m_i = float("-inf")
            l_i = 0.0
            acc = torch.zeros(d)
            num_blocks_q = (kv_end + block_size - 1) // block_size
            for bi in range(num_blocks_q):
                phys = int(block_table[seq, bi].item())
                for off in range(block_size):
                    pos = bi * block_size + off
                    if pos >= kv_end:
                        break
                    k_idx = cache_k_idx[phys, off, kvh].to(torch.long)
                    rk = codebook[k_idx]
                    main_dot = float((q_rot_th * rk).sum())
                    k_norm = float(cache_k_norm[phys, off, kvh])

                    if state.algo == "prod":
                        qjl_sign = cache_k_qjl_sign[phys, off, kvh].float()
                        qjl_dot = float((sq_th * qjl_sign).sum())
                        r_norm = float(cache_k_rnorm[phys, off, kvh])
                        logit = (main_dot + qjl_coef * r_norm * qjl_dot) * k_norm * inv_d
                    else:
                        logit = main_dot * k_norm * inv_d

                    v_vec = cache_v_idx[phys, off, kvh].float() * float(
                        cache_v_scale[phys, off, kvh]
                    )

                    m_new = max(m_i, logit)
                    alpha = math.exp(m_i - m_new) if m_i != float("-inf") else 0.0
                    beta = math.exp(logit - m_new)
                    l_i = l_i * alpha + beta
                    acc = acc * alpha + beta * v_vec
                    m_i = m_new
            out[qi, h] = (acc / max(l_i, 1e-12)).to(q.dtype)
    return out


# ---------------------------------------------------------------------------
# FP ground truth
# ---------------------------------------------------------------------------
def fp_attention(
    q: torch.Tensor,
    k_all: torch.Tensor,
    v_all: torch.Tensor,
    kv_end: torch.Tensor,
) -> torch.Tensor:
    T_q, H_q, d = q.shape
    H_kv = k_all.shape[1]
    gqa = H_q // H_kv
    scale = 1.0 / math.sqrt(d)
    out = torch.zeros_like(q)
    for qi in range(T_q):
        end = int(kv_end[qi].item())
        for h in range(H_q):
            kh = h // gqa
            scores = (q[qi, h].float() @ k_all[:end, kh].float().T) * scale
            w = torch.softmax(scores, dim=-1)
            out[qi, h] = (w @ v_all[:end, kh].float()).to(q.dtype)
    return out


# ---------------------------------------------------------------------------
# Simulator
# ---------------------------------------------------------------------------
def simulate_one_layer(
    algo: str,
    bits: int,
    layer_seed: int,
    prefill_len: int,
    decode_steps: int,
    num_heads_q: int,
    num_heads_kv: int,
    head_size: int,
    block_size: int,
    dtype: torch.dtype,
) -> dict:
    torch.manual_seed(123)
    total = prefill_len + decode_steps

    # All K/V/Q that will flow through this layer across steps.
    k_all = torch.randn(total, num_heads_kv, head_size, dtype=dtype)
    v_all = torch.randn(total, num_heads_kv, head_size, dtype=dtype)
    q_all = torch.randn(total, num_heads_q, head_size, dtype=dtype)

    state = QuantState.make(
        algo=algo, bits=bits, head_dim=head_size,
        seed=layer_seed, dtype=dtype, device=torch.device("cpu"),
    )

    # Paged cache sized for total. Reserve block 0 as padding (mimics vLLM).
    num_blocks = (total + block_size - 1) // block_size + 1
    cache_k_idx = torch.zeros(num_blocks, block_size, num_heads_kv, head_size, dtype=torch.uint8)
    cache_k_norm = torch.zeros(num_blocks, block_size, num_heads_kv, dtype=torch.float32)
    cache_v_idx = torch.zeros(num_blocks, block_size, num_heads_kv, head_size, dtype=torch.int8)
    cache_v_scale = torch.zeros(num_blocks, block_size, num_heads_kv, dtype=torch.float32)
    if algo == "prod":
        cache_k_qjl_sign = torch.zeros(num_blocks, block_size, num_heads_kv, head_size, dtype=torch.int8)
        cache_k_rnorm = torch.zeros(num_blocks, block_size, num_heads_kv, dtype=torch.float32)
    else:
        cache_k_qjl_sign = None
        cache_k_rnorm = None

    # Slot assignment: contiguous slots starting at first_slot = block_size.
    first_slot = block_size
    num_blocks_used = (total + block_size - 1) // block_size
    block_table = torch.zeros(1, num_blocks, dtype=torch.int32)
    for i in range(num_blocks_used):
        block_table[0, i] = i + 1

    all_outputs = []
    # Step 1: prefill.
    slot_prefill = torch.arange(first_slot, first_slot + prefill_len, dtype=torch.int64)
    _store_kv_ref(
        k_all[:prefill_len], v_all[:prefill_len],
        cache_k_idx, cache_k_norm, cache_v_idx, cache_v_scale,
        cache_k_qjl_sign, cache_k_rnorm,
        slot_prefill, state, block_size,
    )
    seq_id = torch.zeros(prefill_len, dtype=torch.int64)
    kv_end = torch.arange(1, prefill_len + 1, dtype=torch.int64)
    out_pref = _attend_ref(
        q_all[:prefill_len],
        cache_k_idx, cache_k_norm, cache_v_idx, cache_v_scale,
        cache_k_qjl_sign, cache_k_rnorm,
        block_table, seq_id, kv_end, state, block_size,
    )
    all_outputs.append(("prefill", out_pref, fp_attention(q_all[:prefill_len], k_all, v_all, kv_end)))

    # Step 2..N: decode one token at a time.
    for step in range(decode_steps):
        pos = prefill_len + step
        slot_dec = torch.tensor([first_slot + pos], dtype=torch.int64)
        _store_kv_ref(
            k_all[pos:pos + 1], v_all[pos:pos + 1],
            cache_k_idx, cache_k_norm, cache_v_idx, cache_v_scale,
            cache_k_qjl_sign, cache_k_rnorm,
            slot_dec, state, block_size,
        )
        seq_id = torch.zeros(1, dtype=torch.int64)
        kv_end = torch.tensor([pos + 1], dtype=torch.int64)
        out_dec = _attend_ref(
            q_all[pos:pos + 1],
            cache_k_idx, cache_k_norm, cache_v_idx, cache_v_scale,
            cache_k_qjl_sign, cache_k_rnorm,
            block_table, seq_id, kv_end, state, block_size,
        )
        fp = fp_attention(q_all[pos:pos + 1], k_all, v_all, kv_end)
        all_outputs.append((f"decode[{step}]", out_dec, fp))

    results = []
    for tag, got, ref in all_outputs:
        diff = (got.float() - ref.float()).abs()
        rel = float(diff.mean() / ref.float().abs().mean().clamp(min=1e-6))
        results.append((tag, rel, float(diff.max())))
    return {"algo": algo, "seed": layer_seed, "results": results}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bits", type=int, default=8)
    ap.add_argument("--prefill", type=int, default=2, help="mimics 'Hello' -> 2 tokens")
    ap.add_argument("--decode", type=int, default=4, help="max_tokens=4")
    ap.add_argument("--heads-q", type=int, default=32)
    ap.add_argument("--heads-kv", type=int, default=8)
    ap.add_argument("--head-size", type=int, default=128)
    ap.add_argument("--block-size", type=int, default=16)
    ap.add_argument("--layers", type=int, default=4, help="how many distinct-seed layers to test")
    ap.add_argument("--algo", default="both")
    args = ap.parse_args()

    print(
        f"# bits={args.bits} prefill={args.prefill} decode={args.decode} "
        f"heads_q={args.heads_q}/{args.heads_kv} d={args.head_size} "
        f"block={args.block_size} layers={args.layers}"
    )

    algos = ("mse", "prod") if args.algo == "both" else (args.algo,)
    for algo in algos:
        print(f"\n=== algo={algo} ===")
        print(f"{'layer_seed':>10} {'step':>12} {'rel_err':>10} {'max_err':>10}")
        for layer_seed in range(args.layers):
            r = simulate_one_layer(
                algo=algo, bits=args.bits, layer_seed=layer_seed,
                prefill_len=args.prefill, decode_steps=args.decode,
                num_heads_q=args.heads_q, num_heads_kv=args.heads_kv,
                head_size=args.head_size, block_size=args.block_size,
                dtype=torch.bfloat16,
            )
            for tag, rel, mx in r["results"]:
                print(f"{layer_seed:>10} {tag:>12} {rel:>10.4%} {mx:>10.4f}")


if __name__ == "__main__":
    main()
