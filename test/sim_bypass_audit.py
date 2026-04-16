#!/usr/bin/env python3
"""Comprehensive Mac-side audit of the BYPASS code path.

BYPASS on B200 produces ://24 even though the Triton kernel, the
multi-layer/multi-call flow, and GQA=4 all check out. This script
exhaustively exercises _fp_paged_attention (copied from turboquant_attn)
and the output.copy_ handoff, against torch.nn.functional.scaled_dot_product_attention
as ground truth. If any sub-test fails here, that's the bug.

Covers:
  A) _fp_paged_attention output vs SDPA (pure math sanity)
  B) output = torch.empty(2D).view(3D); copy_(attn_out); read back 2D
  C) Different seq_lens semantics (= total incl. current vs prefix only)
  D) Different query/key layouts (2D view to 3D and back)
  E) Prefill then N decode steps, cumulative correctness
  F) slot_mapping with sparse / non-contiguous physical blocks
  G) block_table[0, i] = i+1 vs block_table[0, i] = some permutation
"""

from __future__ import annotations

import math
import sys
import traceback

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Ported from vllm/v1/attention/backends/turboquant_attn.py (verbatim logic)
# ---------------------------------------------------------------------------
def _fp_paged_attention(
    q: torch.Tensor,
    k_fp: torch.Tensor,
    v_fp: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    query_start_loc: torch.Tensor,
    scale: float,
    num_heads_q: int,
    num_heads_kv: int,
) -> torch.Tensor:
    T_q, H_q, d = q.shape
    num_blocks, block_size, H_kv, _ = k_fp.shape
    num_seqs = int(seq_lens.shape[0])
    dev = q.device
    gqa = H_q // H_kv

    qsl = query_start_loc.to(device=dev, dtype=torch.int64)
    query_lens = qsl[1:] - qsl[:-1]
    seq_ids = torch.arange(num_seqs, dtype=torch.int64, device=dev)
    seq_id_per_q = torch.repeat_interleave(seq_ids, query_lens)
    q_pos = (
        torch.arange(T_q, dtype=torch.int64, device=dev)
        - qsl[:-1][seq_id_per_q]
    )
    prefix_len = seq_lens.to(device=dev, dtype=torch.int64) - query_lens
    kv_end_per_q = prefix_len[seq_id_per_q] + q_pos + 1

    out = torch.empty_like(q)
    for qi in range(T_q):
        seq = int(seq_id_per_q[qi].item())
        end = int(kv_end_per_q[qi].item())
        num_blocks_q = (end + block_size - 1) // block_size
        k_list, v_list = [], []
        taken = 0
        for bi in range(num_blocks_q):
            phys = int(block_table[seq, bi].item())
            use = min(block_size, end - taken)
            k_list.append(k_fp[phys, :use])
            v_list.append(v_fp[phys, :use])
            taken += use
        k_seq = torch.cat(k_list, dim=0)
        v_seq = torch.cat(v_list, dim=0)
        for h in range(H_q):
            kh = h // gqa
            scores = (q[qi, h].float() @ k_seq[:, kh].float().T) * scale
            w = torch.softmax(scores, dim=-1)
            out[qi, h] = (w @ v_seq[:, kh].float()).to(q.dtype)
    return out


# ---------------------------------------------------------------------------
# Reference: torch SDPA with causal mask per query position
# ---------------------------------------------------------------------------
def sdpa_ref(
    q: torch.Tensor,   # (T, H_q, d)
    k: torch.Tensor,   # (T, H_kv, d)
    v: torch.Tensor,   # (T, H_kv, d)
    kv_end: torch.Tensor,  # (T,) int, number of KV this query attends to
) -> torch.Tensor:
    T, H_q, d = q.shape
    H_kv = k.shape[1]
    gqa = H_q // H_kv
    scale = 1.0 / math.sqrt(d)
    out = torch.zeros_like(q)
    for qi in range(T):
        end = int(kv_end[qi].item())
        for h in range(H_q):
            kh = h // gqa
            scores = (q[qi, h].float() @ k[:end, kh].float().T) * scale
            w = torch.softmax(scores, dim=-1)
            out[qi, h] = (w @ v[:end, kh].float()).to(q.dtype)
    return out


# ---------------------------------------------------------------------------
# Testing harness
# ---------------------------------------------------------------------------
def _rel(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(
        (a.float() - b.float()).abs().mean()
        / b.float().abs().mean().clamp(min=1e-6)
    )


def _alloc_paged(T: int, H_kv: int, d: int, block_size: int):
    # Reserve block 0 as padding, fill blocks 1, 2, ...
    num_blocks = (T + block_size - 1) // block_size + 1
    k_fp = torch.zeros(num_blocks, block_size, H_kv, d, dtype=torch.bfloat16)
    v_fp = torch.zeros(num_blocks, block_size, H_kv, d, dtype=torch.bfloat16)
    block_table = torch.zeros(1, num_blocks, dtype=torch.int32)
    num_blocks_used = (T + block_size - 1) // block_size
    for i in range(num_blocks_used):
        block_table[0, i] = i + 1
    return k_fp, v_fp, block_table


def _store_seq(k_fp, v_fp, k, v, block_table, seq, offset, block_size):
    """Store k, v into paged cache starting at logical position `offset` of seq."""
    n = k.shape[0]
    for i in range(n):
        abs_pos = offset + i
        bi = abs_pos // block_size
        off = abs_pos % block_size
        phys = int(block_table[seq, bi].item())
        k_fp[phys, off] = k[i]
        v_fp[phys, off] = v[i]


def _slot_mapping_for(block_table, seq, offset, n, block_size):
    out = torch.zeros(n, dtype=torch.int64)
    for i in range(n):
        abs_pos = offset + i
        bi = abs_pos // block_size
        off = abs_pos % block_size
        phys = int(block_table[seq, bi].item())
        out[i] = phys * block_size + off
    return out


def test_A_core_math_vs_sdpa():
    """A single-call prefill: _fp_paged vs SDPA on same data."""
    torch.manual_seed(0)
    T, H_q, H_kv, d, bs = 8, 32, 8, 128, 16
    k = torch.randn(T, H_kv, d, dtype=torch.bfloat16)
    v = torch.randn(T, H_kv, d, dtype=torch.bfloat16)
    q = torch.randn(T, H_q, d, dtype=torch.bfloat16)
    k_fp, v_fp, bt = _alloc_paged(T, H_kv, d, bs)
    _store_seq(k_fp, v_fp, k, v, bt, 0, 0, bs)
    seq_lens = torch.tensor([T], dtype=torch.int32)
    qsl = torch.tensor([0, T], dtype=torch.int32)
    scale = 1.0 / math.sqrt(d)
    got = _fp_paged_attention(q, k_fp, v_fp, bt, seq_lens, qsl, scale, H_q, H_kv)
    kv_end = torch.arange(1, T + 1, dtype=torch.int32)
    ref = sdpa_ref(q, k, v, kv_end)
    return ("A core math", _rel(got, ref))


def test_B_output_copy_roundtrip():
    """Simulate vLLM wrapper: output = empty(2D).view(3D) -> impl copy_ -> view back 2D."""
    T, H, d = 8, 32, 128
    attn_out = torch.randn(T, H, d, dtype=torch.bfloat16)
    # Wrapper allocates 2D:
    output_2d = torch.empty(T, H * d, dtype=torch.bfloat16)
    # Wrapper views 3D before passing to impl:
    output_3d = output_2d.view(-1, H, d)
    # Impl does:
    output_3d.copy_(attn_out.reshape_as(output_3d))
    # Wrapper returns 2D view:
    returned = output_2d.view(-1, H * d)
    # It should contain the elements of attn_out:
    rel = _rel(returned, attn_out.reshape(T, H * d))
    return ("B output.copy_ roundtrip", rel)


def test_C_seq_lens_semantics():
    """Try both interpretations of seq_lens: INCLUDING current vs prefix only."""
    torch.manual_seed(1)
    H_q, H_kv, d, bs = 32, 8, 128, 16
    prefill = 2
    decode = 2
    T = prefill + decode
    k = torch.randn(T, H_kv, d, dtype=torch.bfloat16)
    v = torch.randn(T, H_kv, d, dtype=torch.bfloat16)
    q = torch.randn(T, H_q, d, dtype=torch.bfloat16)
    k_fp, v_fp, bt = _alloc_paged(T, H_kv, d, bs)
    # Fill all tokens into cache.
    _store_seq(k_fp, v_fp, k, v, bt, 0, 0, bs)
    scale = 1.0 / math.sqrt(d)

    # SDPA reference for a single decode token at position prefill+0.
    q_dec = q[prefill: prefill + 1]
    kv_end = torch.tensor([prefill + 1], dtype=torch.int32)
    ref = sdpa_ref(q_dec, k, v, kv_end)

    # Interpretation 1 (ours):  seq_lens = total after this call (prefill+1)
    seq_lens_incl = torch.tensor([prefill + 1], dtype=torch.int32)
    qsl_dec = torch.tensor([0, 1], dtype=torch.int32)
    got_incl = _fp_paged_attention(q_dec, k_fp, v_fp, bt, seq_lens_incl, qsl_dec,
                                   scale, H_q, H_kv)
    rel_incl = _rel(got_incl, ref)

    # Interpretation 2: seq_lens = prefix only (current not yet counted)
    #   -> prefix = seq_lens = prefill; kv_end = prefill + 0 + 1 = prefill + 1
    # But our code does prefix = seq_lens - query_lens = prefill - 1, then
    # kv_end = prefix + q_pos + 1 = prefill. That's WRONG by 1.
    # Simulate by injecting seq_lens = prefix = prefill.
    seq_lens_prefix_only = torch.tensor([prefill], dtype=torch.int32)
    got_pref = _fp_paged_attention(q_dec, k_fp, v_fp, bt, seq_lens_prefix_only, qsl_dec,
                                   scale, H_q, H_kv)
    rel_pref = _rel(got_pref, ref)

    return [
        ("C1 seq_lens=total (ours)", rel_incl),
        ("C2 seq_lens=prefix only", rel_pref),
    ]


def test_D_query_layout_2d_to_3d():
    """If query arrives 2D (T, H*d), does .view(-1, H, d) give right layout?"""
    T, H, d = 4, 32, 128
    # Ground-truth 3D:
    q3d = torch.randn(T, H, d, dtype=torch.bfloat16)
    # Simulate a 2D container whose rows are [h0_d0...h0_d127, h1_d0...]:
    q2d = q3d.reshape(T, H * d).contiguous()
    viewed = q2d.view(-1, H, d)
    rel = _rel(viewed, q3d)
    return ("D .view(T, H, d) preserves layout", rel)


def test_E_prefill_then_multiple_decodes():
    """Full vLLM-style flow: one prefill store+attend, then N decode store+attend."""
    torch.manual_seed(2)
    H_q, H_kv, d, bs = 32, 8, 128, 16
    prefill = 2
    decodes = 4
    total = prefill + decodes
    k = torch.randn(total, H_kv, d, dtype=torch.bfloat16)
    v = torch.randn(total, H_kv, d, dtype=torch.bfloat16)
    q = torch.randn(total, H_q, d, dtype=torch.bfloat16)
    k_fp, v_fp, bt = _alloc_paged(total, H_kv, d, bs)
    scale = 1.0 / math.sqrt(d)

    results = []

    # Prefill
    _store_seq(k_fp, v_fp, k[:prefill], v[:prefill], bt, 0, 0, bs)
    qsl = torch.tensor([0, prefill], dtype=torch.int32)
    seq_lens = torch.tensor([prefill], dtype=torch.int32)
    got = _fp_paged_attention(q[:prefill], k_fp, v_fp, bt, seq_lens, qsl,
                              scale, H_q, H_kv)
    kv_end = torch.arange(1, prefill + 1, dtype=torch.int32)
    ref = sdpa_ref(q[:prefill], k, v, kv_end)
    results.append(("E prefill", _rel(got, ref)))

    # Decodes
    for step in range(decodes):
        pos = prefill + step
        _store_seq(k_fp, v_fp, k[pos:pos + 1], v[pos:pos + 1], bt, 0, pos, bs)
        qsl_d = torch.tensor([0, 1], dtype=torch.int32)
        sl_d = torch.tensor([pos + 1], dtype=torch.int32)
        got_d = _fp_paged_attention(q[pos:pos + 1], k_fp, v_fp, bt, sl_d, qsl_d,
                                    scale, H_q, H_kv)
        kv_end_d = torch.tensor([pos + 1], dtype=torch.int32)
        ref_d = sdpa_ref(q[pos:pos + 1], k, v, kv_end_d)
        results.append((f"E decode[{step}]", _rel(got_d, ref_d)))
    return results


def test_F_non_contiguous_block_table():
    """Simulate vLLM allocating non-contiguous physical blocks (e.g., 7, 3, 9)."""
    torch.manual_seed(3)
    H_q, H_kv, d, bs = 32, 8, 128, 16
    T = 40
    k = torch.randn(T, H_kv, d, dtype=torch.bfloat16)
    v = torch.randn(T, H_kv, d, dtype=torch.bfloat16)
    q = torch.randn(T, H_q, d, dtype=torch.bfloat16)

    num_blocks = 12
    k_fp = torch.zeros(num_blocks, bs, H_kv, d, dtype=torch.bfloat16)
    v_fp = torch.zeros(num_blocks, bs, H_kv, d, dtype=torch.bfloat16)
    bt = torch.zeros(1, num_blocks, dtype=torch.int32)
    # Non-contiguous allocation:
    perm = [7, 3, 9]
    for i, p in enumerate(perm):
        bt[0, i] = p

    _store_seq(k_fp, v_fp, k, v, bt, 0, 0, bs)
    qsl = torch.tensor([0, T], dtype=torch.int32)
    sl = torch.tensor([T], dtype=torch.int32)
    scale = 1.0 / math.sqrt(d)
    got = _fp_paged_attention(q, k_fp, v_fp, bt, sl, qsl, scale, H_q, H_kv)
    kv_end = torch.arange(1, T + 1, dtype=torch.int32)
    ref = sdpa_ref(q, k, v, kv_end)
    return ("F non-contig block_table", _rel(got, ref))


def test_G_output_via_custom_op_pattern():
    """Full wrapper-style pattern: empty output allocated outside, filled inside."""
    torch.manual_seed(4)
    H_q, H_kv, d, bs = 32, 8, 128, 16
    T = 8
    k = torch.randn(T, H_kv, d, dtype=torch.bfloat16)
    v = torch.randn(T, H_kv, d, dtype=torch.bfloat16)
    q = torch.randn(T, H_q, d, dtype=torch.bfloat16)
    k_fp, v_fp, bt = _alloc_paged(T, H_kv, d, bs)
    _store_seq(k_fp, v_fp, k, v, bt, 0, 0, bs)
    qsl = torch.tensor([0, T], dtype=torch.int32)
    sl = torch.tensor([T], dtype=torch.int32)
    scale = 1.0 / math.sqrt(d)

    output_2d = torch.empty(T, H_q * d, dtype=torch.bfloat16)
    output_3d = output_2d.view(-1, H_q, d)

    attn_out = _fp_paged_attention(q, k_fp, v_fp, bt, sl, qsl, scale, H_q, H_kv)
    output_3d.copy_(attn_out.reshape_as(output_3d))

    returned = output_2d.view(-1, H_q * d)
    kv_end = torch.arange(1, T + 1, dtype=torch.int32)
    ref = sdpa_ref(q, k, v, kv_end).reshape(T, H_q * d)
    return ("G wrapper output pattern", _rel(returned, ref))


def test_H_multi_block_span():
    """Sequence crossing block_size=16 boundary (17+ tokens, 2 blocks)."""
    torch.manual_seed(5)
    H_q, H_kv, d, bs = 32, 8, 128, 16
    T = 17  # 1 block + 1 more token
    k = torch.randn(T, H_kv, d, dtype=torch.bfloat16)
    v = torch.randn(T, H_kv, d, dtype=torch.bfloat16)
    q = torch.randn(T, H_q, d, dtype=torch.bfloat16)
    k_fp, v_fp, bt = _alloc_paged(T, H_kv, d, bs)
    _store_seq(k_fp, v_fp, k, v, bt, 0, 0, bs)
    qsl = torch.tensor([0, T], dtype=torch.int32)
    sl = torch.tensor([T], dtype=torch.int32)
    scale = 1.0 / math.sqrt(d)
    got = _fp_paged_attention(q, k_fp, v_fp, bt, sl, qsl, scale, H_q, H_kv)
    kv_end = torch.arange(1, T + 1, dtype=torch.int32)
    ref = sdpa_ref(q, k, v, kv_end)
    return ("H multi-block span (17 tok, 2 blocks)", _rel(got, ref))


def test_I_scale_wrong_values():
    """What if self.scale passed in is NOT 1/sqrt(d)?

    Llama-3.1-8B uses 1/sqrt(128). But if vLLM passes a different scale
    to the impl, we'd produce garbage. Quantify the damage.
    """
    torch.manual_seed(6)
    H_q, H_kv, d, bs = 32, 8, 128, 16
    T = 8
    k = torch.randn(T, H_kv, d, dtype=torch.bfloat16)
    v = torch.randn(T, H_kv, d, dtype=torch.bfloat16)
    q = torch.randn(T, H_q, d, dtype=torch.bfloat16)
    k_fp, v_fp, bt = _alloc_paged(T, H_kv, d, bs)
    _store_seq(k_fp, v_fp, k, v, bt, 0, 0, bs)
    qsl = torch.tensor([0, T], dtype=torch.int32)
    sl = torch.tensor([T], dtype=torch.int32)
    kv_end = torch.arange(1, T + 1, dtype=torch.int32)
    ref = sdpa_ref(q, k, v, kv_end)  # uses 1/sqrt(d) internally

    results = []
    for label, scale in [
        ("correct 1/sqrt(d)", 1.0 / math.sqrt(d)),
        ("wrong 1/d",          1.0 / d),
        ("wrong 1.0",          1.0),
        ("wrong sqrt(d)",      math.sqrt(d)),
    ]:
        got = _fp_paged_attention(q, k_fp, v_fp, bt, sl, qsl, scale, H_q, H_kv)
        results.append((f"I scale={label}", _rel(got, ref)))
    return results


def test_J_slot_mapping_sparse():
    """Simulate vLLM-style slot_mapping with -1 padding mixed in."""
    torch.manual_seed(7)
    H_q, H_kv, d, bs = 32, 8, 128, 16
    T_real = 4
    T_padded = 8  # 4 real + 4 padding
    k = torch.randn(T_real, H_kv, d, dtype=torch.bfloat16)
    v = torch.randn(T_real, H_kv, d, dtype=torch.bfloat16)
    q = torch.randn(T_real, H_q, d, dtype=torch.bfloat16)
    k_fp, v_fp, bt = _alloc_paged(T_real, H_kv, d, bs)

    # Store via slot_mapping semantics used in BYPASS do_kv_cache_update:
    slot_mapping = _slot_mapping_for(bt, 0, 0, T_real, bs)
    # Simulate: we're given T_padded-length key/value + slot_mapping with -1 for padding.
    key_padded = torch.zeros(T_padded, H_kv, d, dtype=torch.bfloat16)
    val_padded = torch.zeros(T_padded, H_kv, d, dtype=torch.bfloat16)
    key_padded[:T_real] = k
    val_padded[:T_real] = v
    slot_padded = torch.full((T_padded,), -1, dtype=torch.int64)
    slot_padded[:T_real] = slot_mapping

    # Replicate BYPASS logic:
    valid = slot_padded >= 0
    slots = slot_padded[valid].to(torch.int64)
    b_idx = slots // bs
    off = slots % bs
    k_fp[b_idx, off] = key_padded[valid]
    v_fp[b_idx, off] = val_padded[valid]

    qsl = torch.tensor([0, T_real], dtype=torch.int32)
    sl = torch.tensor([T_real], dtype=torch.int32)
    scale = 1.0 / math.sqrt(d)
    got = _fp_paged_attention(q, k_fp, v_fp, bt, sl, qsl, scale, H_q, H_kv)
    kv_end = torch.arange(1, T_real + 1, dtype=torch.int32)
    ref = sdpa_ref(q, k, v, kv_end)
    return ("J slot_mapping with -1 padding", _rel(got, ref))


def test_K_kv_cache_shape_variants():
    """What if our code misreads kv_cache.shape[1] / [2]?

    Our _ensure_buffers uses kv_cache.shape[1] as num_blocks and
    kv_cache.shape[2] as block_size. Verify this against the declared
    shape (2, num_blocks, block_size, H_kv, head_size).
    """
    num_blocks, block_size, H_kv, d = 16, 16, 8, 128
    kv_cache = torch.zeros(2, num_blocks, block_size, H_kv, d, dtype=torch.bfloat16)
    got_blocks = kv_cache.shape[1]
    got_bs = kv_cache.shape[2]
    ok = got_blocks == num_blocks and got_bs == block_size
    return ("K kv_cache.shape[1]=num_blocks, [2]=block_size",
            0.0 if ok else 1.0)



    """Full wrapper-style pattern: empty output allocated outside, filled inside."""
    torch.manual_seed(4)
    H_q, H_kv, d, bs = 32, 8, 128, 16
    T = 8
    k = torch.randn(T, H_kv, d, dtype=torch.bfloat16)
    v = torch.randn(T, H_kv, d, dtype=torch.bfloat16)
    q = torch.randn(T, H_q, d, dtype=torch.bfloat16)
    k_fp, v_fp, bt = _alloc_paged(T, H_kv, d, bs)
    _store_seq(k_fp, v_fp, k, v, bt, 0, 0, bs)
    qsl = torch.tensor([0, T], dtype=torch.int32)
    sl = torch.tensor([T], dtype=torch.int32)
    scale = 1.0 / math.sqrt(d)

    # Wrapper allocates 2D output:
    output_2d = torch.empty(T, H_q * d, dtype=torch.bfloat16)
    output_3d = output_2d.view(-1, H_q, d)

    # Impl logic (mirrors turboquant_attn.forward):
    attn_out = _fp_paged_attention(q, k_fp, v_fp, bt, sl, qsl, scale, H_q, H_kv)
    output_3d.copy_(attn_out.reshape_as(output_3d))

    # Caller reads via 2D view:
    returned = output_2d.view(-1, H_q * d)
    # Compare to SDPA (reshaped 2D):
    kv_end = torch.arange(1, T + 1, dtype=torch.int32)
    ref = sdpa_ref(q, k, v, kv_end).reshape(T, H_q * d)
    return ("G wrapper output pattern", _rel(returned, ref))


def main():
    cases = []
    tests = [
        test_A_core_math_vs_sdpa,
        test_B_output_copy_roundtrip,
        test_C_seq_lens_semantics,
        test_D_query_layout_2d_to_3d,
        test_E_prefill_then_multiple_decodes,
        test_F_non_contiguous_block_table,
        test_G_output_via_custom_op_pattern,
        test_H_multi_block_span,
        test_I_scale_wrong_values,
        test_J_slot_mapping_sparse,
        test_K_kv_cache_shape_variants,
    ]
    for t in tests:
        try:
            r = t()
        except Exception as e:
            traceback.print_exc()
            cases.append((t.__name__, float("nan"), f"EXCEPTION {e}"))
            continue
        if isinstance(r, list):
            for tag, rel in r:
                cases.append((t.__name__, rel, tag))
        else:
            tag, rel = r
            cases.append((t.__name__, rel, tag))

    print(f"{'test':>40} {'rel':>10}  {'tag'}")
    print("-" * 80)
    any_fail = False
    for name, rel, tag in cases:
        status = "OK" if rel < 0.05 else "FAIL"
        if status == "FAIL":
            any_fail = True
        print(f"{name:>40} {rel:>10.4%}  {tag}  [{status}]")
    print()
    if any_fail:
        print(">>> At least one sub-test FAILED -- that is the bug.")
        sys.exit(1)
    print(">>> All sub-tests passed within 5% rel error.")


if __name__ == "__main__":
    main()
