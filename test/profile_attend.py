#!/usr/bin/env python3
"""Profile turboquant backend latency breakdown during decode.

Wraps vLLM's LLM class with ``torch.profiler`` and captures every CUDA
kernel / CPU op during a few generate calls. Prints:

  1. Top N CUDA kernels by self-time (sorted).
  2. A coarse category breakdown (attend / store / matmul / mem / other).
  3. A Chrome trace path -- open in chrome://tracing or
     https://ui.perfetto.dev/ for the full timeline view.

Usage
-----
    # Base turboquant attend kernel (per-slot CUDA-core)
    GPU=2 python3 test/profile_attend.py

    # LUT tensor-core attend kernel
    GPU=2 TURBOQUANT_USE_LUT=1 python3 test/profile_attend.py

    # Switch algorithm
    GPU=2 TURBOQUANT_ALGO=prod TURBOQUANT_USE_LUT=1 python3 test/profile_attend.py

    # Compare against FLASH_ATTN (no turboquant)
    GPU=2 ATTN_BACKEND=FLASH_ATTN python3 test/profile_attend.py

Required env:
    GPU                 : CUDA device id (default 0)
    MODEL               : model path
    ATTN_BACKEND        : FLASH_ATTN | TURBOQUANT  (default TURBOQUANT)
    TURBOQUANT_ALGO     : mse | prod (default mse)
    TURBOQUANT_BITS     : 4 (only 4 supported on this branch)
    TURBOQUANT_USE_LUT  : 0 | 1 (default 0)
    PROMPT_TOKENS       : target prompt token count (default 1024)
    OUTPUT_TOKENS       : decode steps to profile (default 64)
    WARMUP_CALLS        : warmup generate calls before profiling (default 2)
    PROFILE_CALLS       : generate calls to profile (default 3)
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

# Env defaults before importing vLLM (backend reads env at module load).
os.environ.setdefault("TURBOQUANT_ALGO", "mse")
os.environ.setdefault("TURBOQUANT_BITS", "4")
os.environ.setdefault("TURBOQUANT_USE_LUT", "0")

GPU = os.environ.get("GPU", "0")
os.environ["CUDA_VISIBLE_DEVICES"] = GPU

import torch
from torch.profiler import profile, ProfilerActivity

from vllm import LLM, SamplingParams


MODEL = os.environ.get(
    "MODEL", "/mnt/nvme3n1/g00872988/models/Llama-3.1-8B-Instruct"
)
BACKEND = os.environ.get("ATTN_BACKEND", "TURBOQUANT").upper()
PROMPT_TOKENS = int(os.environ.get("PROMPT_TOKENS", "1024"))
OUTPUT_TOKENS = int(os.environ.get("OUTPUT_TOKENS", "64"))
WARMUP_CALLS = int(os.environ.get("WARMUP_CALLS", "2"))
PROFILE_CALLS = int(os.environ.get("PROFILE_CALLS", "3"))


def _make_prompt(target_tokens: int) -> str:
    """Repeat a stock sentence to approximately target_tokens tokens."""
    snippet = "Machine learning is transforming many fields. "
    # ~8 tokens per snippet for Llama-3 tokenizer.
    n = max(1, target_tokens // 8)
    return snippet * n


def _tag() -> str:
    if BACKEND == "FLASH_ATTN":
        return "flash_attn"
    algo = os.environ.get("TURBOQUANT_ALGO", "mse")
    kernel = "lut" if os.environ.get("TURBOQUANT_USE_LUT") == "1" else "base"
    return f"{algo}_b4_{kernel}"


def _categorize(name: str) -> str:
    n = name.lower()
    # Our Triton kernels
    if "tc_attend" in n or "flash_lut_attend" in n or "_attend_kernel" in n \
            or "combine_kernel" in n:
        return "tq_attend"
    # FlashAttention family kernels (upstream)
    if "flash_attn" in n or "flashinfer" in n or "fa2" in n or "fa3" in n \
            or "_attn_fwd" in n:
        return "fa_attend"
    # Turboquant store path (pure PyTorch ops -- shows up as elemwise, matmul,
    # searchsorted, scatter, etc., but called from turboquant_store_*).
    # We can't easily tag per-op; fold store-adjacent ops into "tq_other".
    # Hadamard and Sq matmul are cuBLAS gemm from turboquant wrapper.
    if "searchsorted" in n or "scatter" in n or "scatter_" in n:
        return "tq_store"
    # Matmul / linear (model fwd + hadamard rotations)
    if "gemm" in n or "cutlass" in n or "matmul" in n or "mm_" in n \
            or "linear" in n or "nt_kernel" in n or "sm90" in n or "sm80" in n:
        return "matmul"
    # Memory ops
    if "copy" in n or "memset" in n or "to_copy" in n or "memcpy" in n:
        return "mem"
    # Reductions / normalization
    if "norm" in n or "rmsnorm" in n or "reduce" in n:
        return "norm"
    # Activations / elementwise
    if "silu" in n or "mul" in n or "add" in n or "elementwise" in n \
            or "pointwise" in n or "rotary" in n:
        return "elemwise"
    return "other"


def run():
    tag = _tag()
    print(f"# tag={tag} backend={BACKEND} gpu={GPU}")
    print(f"# prompt_tokens≈{PROMPT_TOKENS} output_tokens={OUTPUT_TOKENS}")

    llm_kwargs = dict(
        model=MODEL,
        enforce_eager=True,
        gpu_memory_utilization=0.3,
        max_model_len=max(4096, PROMPT_TOKENS + OUTPUT_TOKENS + 128),
        tensor_parallel_size=1,
    )
    if BACKEND != "TURBOQUANT":
        # vLLM selects backend via env var for FLASH_ATTN at import time.
        # For a flat comparison, users should set ATTN_BACKEND and restart.
        os.environ["VLLM_ATTENTION_BACKEND"] = BACKEND

    print("Loading model...")
    llm = LLM(**llm_kwargs)

    prompt = _make_prompt(PROMPT_TOKENS)
    sampling = SamplingParams(
        max_tokens=OUTPUT_TOKENS,
        temperature=0.0,
        ignore_eos=True,
    )

    # Warmup so Triton JIT / autotune land before profile starts.
    print(f"Warmup ({WARMUP_CALLS} calls)...")
    for _ in range(WARMUP_CALLS):
        llm.generate([prompt], sampling, use_tqdm=False)
    torch.cuda.synchronize()

    print(f"Profiling {PROFILE_CALLS} generate calls...")
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=False,
        profile_memory=False,
        with_stack=False,
    ) as prof:
        for _ in range(PROFILE_CALLS):
            llm.generate([prompt], sampling, use_tqdm=False)
        torch.cuda.synchronize()

    averages = prof.key_averages()

    # ---------------- Top N kernels ----------------
    print()
    print("=" * 70)
    print("TOP 30 kernels by self CUDA time")
    print("=" * 70)
    print(averages.table(sort_by="self_cuda_time_total", row_limit=30))

    # ---------------- Category summary ----------------
    totals = {}
    total_all = 0
    for evt in averages:
        t = evt.self_cuda_time_total
        if t <= 0:
            continue
        cat = _categorize(evt.key)
        totals[cat] = totals.get(cat, 0) + t
        total_all += t

    print()
    print("=" * 70)
    print(f"Category breakdown  (total self CUDA time = {total_all/1000:.1f} ms)")
    print("=" * 70)
    print(f"{'category':<15} {'self_cuda_ms':>14} {'pct':>8}")
    print("-" * 40)
    for cat in sorted(totals, key=lambda c: -totals[c]):
        ms = totals[cat] / 1000.0
        pct = totals[cat] / max(total_all, 1) * 100
        print(f"{cat:<15} {ms:>14.2f} {pct:>7.1f}%")

    # ---------------- Chrome trace ----------------
    out_dir = _ROOT / "logs" / "profile_latest"
    out_dir.mkdir(parents=True, exist_ok=True)
    trace_path = out_dir / f"trace_{tag}.json"
    prof.export_chrome_trace(str(trace_path))
    print()
    print(f"Chrome trace : {trace_path}")
    print(f"   Open in:    chrome://tracing/   OR   https://ui.perfetto.dev/")


if __name__ == "__main__":
    run()
