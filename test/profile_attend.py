#!/usr/bin/env python3
"""Profile turboquant store + attend directly (no vLLM).

The vLLM LLM class runs its EngineCore in a subprocess, so a torch.profiler
in the main process cannot see the decode-step GPU work. Instead we drive
``turboquant_store_kv / _v`` and ``turboquant_paged_attention[_lut]`` in
the same process with synthetic tensors, simulating a 32-layer decode.
This isolates exactly the path we can optimize (store + attend + the
Python rotation matmuls), gives clean CUDA timings, and keeps the
profiler in-process.

The simulation:
  - NUM_LAYERS  decode-style forwards per decode step
  - PROMPT_LEN  tokens of pre-filled KV cache (via store_kv/v once)
  - DECODE_STEPS decode iterations (each step: NUM_LAYERS * (store + attend))

Usage
-----
    GPU=3 python3 test/profile_attend.py
    GPU=3 TURBOQUANT_USE_LUT=1 python3 test/profile_attend.py
    GPU=3 TURBOQUANT_ALGO=prod TURBOQUANT_USE_LUT=1 python3 test/profile_attend.py

Env overrides:
    PROMPT_LEN     : pre-fill KV length (default 1024)
    DECODE_STEPS   : decode iterations to profile (default 16)
    NUM_LAYERS     : layers per forward (default 32, matches Llama-3 8B)
    BATCH          : concurrent query tokens per step (default 16, matches
                     throughput.sh concurrency=16)
    HEAD_Q         : num q heads (default 32)
    HEAD_KV        : num kv heads (default 8)
    HEAD_SIZE      : head dim (default 128)
    BLOCK_SIZE     : paged cache block size (default 16)
    WARMUP_STEPS   : warmup decode steps before profiling (default 4)
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

# Env defaults before importing the backend module (which reads env at load).
os.environ.setdefault("TURBOQUANT_ALGO", "mse")
os.environ.setdefault("TURBOQUANT_BITS", "4")
os.environ.setdefault("TURBOQUANT_USE_LUT", "0")
os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("GPU", "0")

import torch
from torch.profiler import profile, ProfilerActivity

from vllm.turboquant.attend import turboquant_paged_attention
from vllm.turboquant.attend_lut import turboquant_paged_attention_lut
from vllm.turboquant.codebook import QuantState
from vllm.turboquant.store import turboquant_store_kv, turboquant_store_v


ALGO = os.environ["TURBOQUANT_ALGO"]
BITS = int(os.environ["TURBOQUANT_BITS"])
USE_LUT = os.environ["TURBOQUANT_USE_LUT"] == "1"

PROMPT_LEN = int(os.environ.get("PROMPT_LEN", "1024"))
DECODE_STEPS = int(os.environ.get("DECODE_STEPS", "16"))
NUM_LAYERS = int(os.environ.get("NUM_LAYERS", "32"))
BATCH = int(os.environ.get("BATCH", "16"))
HEAD_Q = int(os.environ.get("HEAD_Q", "32"))
HEAD_KV = int(os.environ.get("HEAD_KV", "8"))
HEAD_SIZE = int(os.environ.get("HEAD_SIZE", "128"))
BLOCK_SIZE = int(os.environ.get("BLOCK_SIZE", "16"))
WARMUP_STEPS = int(os.environ.get("WARMUP_STEPS", "4"))


def _tag() -> str:
    return f"{ALGO}_b{BITS}_{'lut' if USE_LUT else 'base'}"


def _categorize(name: str) -> str:
    n = name.lower()
    if "tc_attend" in n or "_attend_kernel" in n or "combine_kernel" in n:
        return "tq_attend"
    if "_store_quant_kernel" in n or "_store_kernel" in n:
        return "tq_store"
    if "scatter" in n or "index_put" in n or "searchsorted" in n:
        return "tq_store"
    if "gemm" in n or "cutlass" in n or "nt_kernel" in n \
            or "sm90" in n or "sm80" in n or "sm100" in n or "bmm" in n \
            or "nvjet" in n or n == "aten::mm" or n == "aten::matmul":
        return "matmul"  # hadamard / QJL rotation cuBLAS/cutlass calls
    if "copy" in n or "memset" in n or "to_copy" in n or "memcpy" in n \
            or "contiguous" in n:
        return "mem"
    if "local_scalar_dense" in n or "item" in n or "nonzero" in n \
            or "cub::detail::select" in n:
        return "sync"
    if "sum" in n or "reduce" in n or n == "aten::norm":
        return "reduce"
    # Elementwise arithmetic ops -- catch aten-level names too.
    if ("elementwise" in n or "pointwise" in n
            or n in ("aten::mul", "aten::add", "aten::sub", "aten::div",
                     "aten::pow", "aten::sqrt", "aten::clamp_min",
                     "aten::clamp", "aten::where", "aten::any",
                     "aten::ge", "aten::le", "aten::lt", "aten::gt",
                     "aten::remainder", "aten::floor_divide",
                     "aten::__lshift__", "aten::__rshift__",
                     "aten::mul_", "aten::add_", "aten::normal_")):
        return "elemwise"
    if n == "aten::index_select" or "index_elementwise" in n \
            or "vectorized_gather" in n or "index_put_impl" in n \
            or "index" in n:
        return "index"
    return "other"


def _alloc_caches(num_blocks: int, device: torch.device, dtype: torch.dtype):
    K_CB = 1 << (BITS - 1 if ALGO == "prod" else BITS)
    idx_last = HEAD_SIZE // 2 if K_CB <= 16 else HEAD_SIZE

    c_k_idx = torch.zeros(
        num_blocks, BLOCK_SIZE, HEAD_KV, idx_last,
        dtype=torch.uint8, device=device,
    )
    c_k_norm = torch.zeros(
        num_blocks, BLOCK_SIZE, HEAD_KV, dtype=torch.float32, device=device,
    )
    c_v_idx = torch.zeros_like(c_k_idx)
    c_v_norm = torch.zeros_like(c_k_norm)

    if ALGO == "prod":
        c_k_qjl = torch.zeros(
            num_blocks, BLOCK_SIZE, HEAD_KV, HEAD_SIZE // 8,
            dtype=torch.uint8, device=device,
        )
        c_k_rnorm = torch.zeros_like(c_k_norm)
    else:
        c_k_qjl = None
        c_k_rnorm = None

    return c_k_idx, c_k_norm, c_v_idx, c_v_norm, c_k_qjl, c_k_rnorm


def _one_decode_step(layers, caches, block_table, seq_lens,
                     query_start_loc, slot_mapping, q_dec,
                     k_per_layer, v_per_layer):
    """One decode step: for each layer, store 1 new K/V token + attend.

    ``k_per_layer`` and ``v_per_layer`` are pre-generated so the hot loop
    has no ``torch.randn`` overhead (randn per layer is harness noise --
    in real vLLM K/V come from linear projections of hidden_states).
    """
    attend_fn = (turboquant_paged_attention_lut
                 if USE_LUT else turboquant_paged_attention)
    for layer_state, cache, k, v in zip(layers, caches,
                                         k_per_layer, v_per_layer):
        c_k_idx, c_k_norm, c_v_idx, c_v_norm, c_k_qjl, c_k_rnorm = cache

        turboquant_store_kv(
            new_k=k, cache_k_idx=c_k_idx, cache_k_norm=c_k_norm,
            slot_mapping=slot_mapping,
            state=layer_state, block_size=BLOCK_SIZE,
            cache_k_qjl_sign=c_k_qjl, cache_k_rnorm=c_k_rnorm,
        )
        turboquant_store_v(
            new_v=v, cache_v_idx=c_v_idx, cache_v_norm=c_v_norm,
            slot_mapping=slot_mapping,
            state=layer_state, block_size=BLOCK_SIZE,
        )

        attend_fn(
            q=q_dec,
            cache_k_idx=c_k_idx, cache_k_norm=c_k_norm,
            cache_v_idx=c_v_idx, cache_v_norm=c_v_norm,
            block_table=block_table,
            seq_lens=seq_lens, query_start_loc=query_start_loc,
            state=layer_state,
            cache_k_qjl_sign=c_k_qjl, cache_k_rnorm=c_k_rnorm,
        )


def run():
    tag = _tag()
    print(f"# tag={tag} algo={ALGO} bits={BITS} lut={USE_LUT}")
    print(f"# batch={BATCH} prompt_len={PROMPT_LEN} decode_steps={DECODE_STEPS}")
    print(f"# num_layers={NUM_LAYERS} head_q={HEAD_Q} head_kv={HEAD_KV} "
          f"d={HEAD_SIZE} block={BLOCK_SIZE}")

    dtype = torch.bfloat16
    device = torch.device("cuda")

    # Build per-layer QuantState.
    layers = [
        QuantState(algo=ALGO, bits=BITS, head_dim=HEAD_SIZE,
                   seed=i, dtype=dtype, device=device)
        for i in range(NUM_LAYERS)
    ]

    # Paged cache sizing: one seq of length PROMPT_LEN + DECODE_STEPS.
    # Multiple "sequences" -- we fake them as one long seq per batch row.
    total_kv_per_seq = PROMPT_LEN + DECODE_STEPS + 4
    num_blocks = (total_kv_per_seq + BLOCK_SIZE - 1) // BLOCK_SIZE + 2
    num_blocks *= BATCH
    num_blocks += 2  # reserve block 0

    caches = [_alloc_caches(num_blocks, device, dtype) for _ in range(NUM_LAYERS)]

    # Block table: each batch row gets its own contiguous set of blocks.
    blocks_per_seq = (total_kv_per_seq + BLOCK_SIZE - 1) // BLOCK_SIZE + 1
    block_table = torch.zeros(BATCH, blocks_per_seq,
                              dtype=torch.int32, device=device)
    phys = 1
    for b in range(BATCH):
        for i in range(blocks_per_seq):
            block_table[b, i] = phys
            phys += 1

    def _compute_slot_mapping_gpu(pos_per_batch: torch.Tensor) -> torch.Tensor:
        """Compute slot_mapping on GPU via vectorized block_table gather --
        no .item() sync."""
        blk_idx = pos_per_batch // BLOCK_SIZE
        off = pos_per_batch % BLOCK_SIZE
        batch_ar = torch.arange(BATCH, device=device)
        phys_blks = block_table[batch_ar, blk_idx].to(torch.int64)
        return phys_blks * BLOCK_SIZE + off

    # Initial prefill: one big store so the cache is populated.
    prompt_k = torch.randn(BATCH * PROMPT_LEN, HEAD_KV, HEAD_SIZE,
                           dtype=dtype, device=device)
    prompt_v = torch.randn_like(prompt_k)
    # Vectorized prompt slot mapping: pos = b*PROMPT_LEN_IGNORED ... actually
    # each row b has its own block range, so compute per-(b, i).
    pos_grid = torch.arange(PROMPT_LEN, device=device).view(1, -1).expand(BATCH, -1)
    batch_grid = torch.arange(BATCH, device=device).view(-1, 1).expand(-1, PROMPT_LEN)
    blk_grid = pos_grid // BLOCK_SIZE
    off_grid = pos_grid % BLOCK_SIZE
    phys_grid = block_table[batch_grid, blk_grid].to(torch.int64)
    prompt_slot = (phys_grid * BLOCK_SIZE + off_grid).reshape(-1).contiguous()

    for layer_state, cache in zip(layers, caches):
        c_k_idx, c_k_norm, c_v_idx, c_v_norm, c_k_qjl, c_k_rnorm = cache
        turboquant_store_kv(
            new_k=prompt_k, cache_k_idx=c_k_idx, cache_k_norm=c_k_norm,
            slot_mapping=prompt_slot,
            state=layer_state, block_size=BLOCK_SIZE,
            cache_k_qjl_sign=c_k_qjl, cache_k_rnorm=c_k_rnorm,
        )
        turboquant_store_v(
            new_v=prompt_v, cache_v_idx=c_v_idx, cache_v_norm=c_v_norm,
            slot_mapping=prompt_slot,
            state=layer_state, block_size=BLOCK_SIZE,
        )
    torch.cuda.synchronize()

    # Set up the tensors reused across decode steps.
    seq_lens = torch.full((BATCH,), PROMPT_LEN, dtype=torch.int32, device=device)
    query_start_loc = torch.arange(BATCH + 1, dtype=torch.int32, device=device)

    # Pre-build slot_mapping, Q, and per-layer K/V for every warmup+profile
    # step so the profile loop itself has no Python-side allocation or
    # sync. These are harness artifacts -- in real vLLM they come from
    # the scheduler and linear projections.
    all_slot_mappings = []
    all_q_decs = []
    all_ks_per_step = []   # list of list of tensors: [step][layer] -> K
    all_vs_per_step = []
    for step in range(WARMUP_STEPS + DECODE_STEPS):
        pos_per_batch = torch.full((BATCH,), PROMPT_LEN + step,
                                   dtype=torch.int64, device=device)
        all_slot_mappings.append(_compute_slot_mapping_gpu(pos_per_batch))
        all_q_decs.append(torch.randn(BATCH, HEAD_Q, HEAD_SIZE,
                                      dtype=dtype, device=device))
        ks = [torch.randn(BATCH, HEAD_KV, HEAD_SIZE, dtype=dtype, device=device)
              for _ in range(NUM_LAYERS)]
        vs = [torch.randn(BATCH, HEAD_KV, HEAD_SIZE, dtype=dtype, device=device)
              for _ in range(NUM_LAYERS)]
        all_ks_per_step.append(ks)
        all_vs_per_step.append(vs)
    torch.cuda.synchronize()

    # Warmup.
    print(f"Warmup ({WARMUP_STEPS} decode steps)...")
    for step in range(WARMUP_STEPS):
        _one_decode_step(
            layers, caches, block_table, seq_lens + step,
            query_start_loc,
            all_slot_mappings[step], all_q_decs[step],
            all_ks_per_step[step], all_vs_per_step[step],
        )
    torch.cuda.synchronize()

    # Profile.
    print(f"Profiling {DECODE_STEPS} decode steps...")
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=False,
        profile_memory=False,
        with_stack=False,
    ) as prof:
        for step in range(DECODE_STEPS):
            global_step = WARMUP_STEPS + step
            _one_decode_step(
                layers, caches, block_table,
                seq_lens + global_step,
                query_start_loc,
                all_slot_mappings[global_step],
                all_q_decs[global_step],
                all_ks_per_step[global_step],
                all_vs_per_step[global_step],
            )
        torch.cuda.synchronize()

    averages = prof.key_averages()

    # Detect which device-time attribute this torch version uses.
    if averages and hasattr(averages[0], "self_device_time_total"):
        time_attr = "self_device_time_total"
    else:
        time_attr = "self_cuda_time_total"

    # ---------------- Top N kernels ----------------
    print()
    print("=" * 70)
    print(f"TOP 30 kernels by {time_attr}")
    print("=" * 70)
    print(averages.table(sort_by=time_attr, row_limit=30))

    # ---------------- Category summary ----------------
    totals = {}
    total_all = 0
    for evt in averages:
        t = getattr(evt, time_attr, 0)
        if t <= 0:
            continue
        cat = _categorize(evt.key)
        totals[cat] = totals.get(cat, 0) + t
        total_all += t

    print()
    print("=" * 70)
    print(f"Category breakdown  (total device time = {total_all/1000:.1f} ms "
          f"across {DECODE_STEPS} decode steps)")
    print(f"Per-step avg: {total_all/1000/DECODE_STEPS:.2f} ms")
    print("=" * 70)
    print(f"{'category':<20} {'device_ms':>12} {'per_step_ms':>12} {'pct':>8}")
    print("-" * 60)
    for cat in sorted(totals, key=lambda c: -totals[c]):
        ms = totals[cat] / 1000.0
        per = ms / DECODE_STEPS
        pct = totals[cat] / max(total_all, 1) * 100
        print(f"{cat:<20} {ms:>12.2f} {per:>12.3f} {pct:>7.1f}%")

    # ---------------- Chrome trace ----------------
    out_dir = _ROOT / "logs" / "profile_latest"
    out_dir.mkdir(parents=True, exist_ok=True)
    trace_path = out_dir / f"trace_{tag}.json"
    prof.export_chrome_trace(str(trace_path))
    print()
    print(f"Chrome trace : {trace_path}")


if __name__ == "__main__":
    run()
