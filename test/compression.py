#!/usr/bin/env python3
"""Measure TurboQuant K+V cache compression vs bf16 baseline.

Two ways to inspect:
  (1) Static calculation -- pure arithmetic on the cache dtype/shape.
      No GPU needed. Run with --num-blocks / --num-kv-heads etc to model
      a specific deployment.

  (2) Runtime allocation -- actually allocates the cache tensors via
      ``TurboQuantAttentionImpl._ensure_buffers`` and reports their
      ``element_size() * numel()``. Requires CUDA. Useful as a sanity
      check that the static formula matches reality.

Both report bytes-per-coord and the K+V compression ratio vs the bf16
baseline (vLLM's standard ``kv_cache`` is 2 bytes K + 2 bytes V per
coord).

Usage
-----
    python3 test/compression.py                       # static, default config
    python3 test/compression.py --runtime --gpu 3     # runtime check on GPU 3
    python3 test/compression.py \\
        --num-blocks 23000 --num-kv-heads 8 \\
        --head-size 128 --num-layers 32              # custom config
"""

from __future__ import annotations

import argparse
import math


def static_layer_bytes(
    num_blocks: int, block_size: int, num_kv_heads: int, head_size: int,
    algo: str, bits: int,
) -> dict[str, int]:
    """Bytes per layer for our K + V cache buffers."""
    K_CB = 1 << (bits - 1 if algo == "prod" else bits)
    pack_4bit = K_CB <= 16
    idx_last = head_size // 2 if pack_4bit else head_size
    dim_elems = num_blocks * block_size * num_kv_heads * head_size
    idx_elems = num_blocks * block_size * num_kv_heads * idx_last
    meta_elems = num_blocks * block_size * num_kv_heads
    out = {
        "k_idx (uint8)": idx_elems * 1,
        "k_norm (fp32)": meta_elems * 4,
        "v_idx (uint8)": idx_elems * 1,
        "v_norm (fp32)": meta_elems * 4,
    }
    if algo == "prod":
        # QJL sign stays unpacked int8 for now.
        out["k_qjl_sign (int8)"] = dim_elems * 1
        out["k_rnorm (fp32)"] = meta_elems * 4
    return out


def bf16_baseline_layer_bytes(
    num_blocks: int, block_size: int, num_kv_heads: int, head_size: int,
) -> int:
    """Bytes per layer for vLLM's bf16 K + V cache (the baseline)."""
    dim_elems = num_blocks * block_size * num_kv_heads * head_size
    return 2 * dim_elems * 2  # 2 (K + V) * 2 bytes (bf16)


def fmt_bytes(n: int) -> str:
    if n >= 1 << 30: return f"{n / (1 << 30):>7.2f} GB"
    if n >= 1 << 20: return f"{n / (1 << 20):>7.1f} MB"
    if n >= 1 << 10: return f"{n / (1 << 10):>7.1f} KB"
    return f"{n:>7d}  B"


def report(num_blocks, block_size, num_kv_heads, head_size, num_layers):
    print(f"Config: num_blocks={num_blocks} block_size={block_size} "
          f"num_kv_heads={num_kv_heads} head_size={head_size} "
          f"num_layers={num_layers}")
    print()
    print(f"{'algo':<6} {'bits':>5} {'per-layer K+V':>14} "
          f"{'all layers':>14} {'bf16 all layers':>16} "
          f"{'B/coord':>10} {'ratio':>8}")
    print("-" * 87)
    bf16_layer = bf16_baseline_layer_bytes(
        num_blocks, block_size, num_kv_heads, head_size)
    bf16_total = bf16_layer * num_layers
    coords_per_layer = num_blocks * block_size * num_kv_heads * head_size

    for algo in ("mse", "prod"):
        for bits in (4, 8):
            sizes = static_layer_bytes(
                num_blocks, block_size, num_kv_heads, head_size, algo, bits)
            layer_total = sum(sizes.values())
            full_total = layer_total * num_layers
            b_per_coord = layer_total / (2 * coords_per_layer)
            ratio = bf16_total / full_total
            print(f"{algo:<6} {bits:>5} {fmt_bytes(layer_total):>14} "
                  f"{fmt_bytes(full_total):>14} "
                  f"{fmt_bytes(bf16_total):>16} "
                  f"{b_per_coord:>9.3f}  {ratio:>7.2f}x")

    print()
    print("Per-buffer breakdown:")
    for algo in ("mse", "prod"):
        for bits in (4, 8):
            print(f"  algo={algo}  bits={bits}:")
            for name, b in static_layer_bytes(
                    num_blocks, block_size, num_kv_heads, head_size,
                    algo, bits).items():
                print(f"    {name:<20} {fmt_bytes(b)}")
    print()
    print("NOTES:")
    print("  - bits=4 packs two 4-bit indices per byte (head_size//2 "
          "storage).")
    print("  - bits=8 stores one index per byte (full head_size).")
    print("  - QJL sign for prod is still int8 unpacked (could become a "
          "1-bit bitfield for further savings).")
    print("  - vLLM still allocates its own bf16 kv_cache that we ignore;"
          " that's separate wasted memory.")


def runtime_check(gpu: int):
    import os
    import sys
    import torch

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    # Import after setting env, since the backend reads TURBOQUANT_ALGO at
    # module load.
    print("Runtime allocation check on CUDA device:")
    for algo in ("mse", "prod"):
        os.environ["TURBOQUANT_ALGO"] = algo
        os.environ["TURBOQUANT_BITS"] = "8"

        # Force re-import so the module-level ALGO constant is re-read.
        for m in list(sys.modules):
            if m.startswith("vllm.v1.attention.backends.turboquant_attn"):
                del sys.modules[m]
        from vllm.v1.attention.backends.turboquant_attn import (
            TurboQuantAttentionImpl,
        )
        TurboQuantAttentionImpl._layer_counter = 0

        impl = TurboQuantAttentionImpl(
            num_heads=32, head_size=128, scale=1.0 / math.sqrt(128),
            num_kv_heads=8, attn_type="decoder",
        )
        # Fake kv_cache to size buffers. Use small num_blocks to avoid OOM.
        kv_cache = torch.zeros(
            (2, 1024, 16, 8, 128), dtype=torch.bfloat16, device="cuda",
        )
        impl._ensure_buffers(kv_cache)
        total = 0
        bufs = [
            ("k_idx", impl._k_idx),
            ("k_norm", impl._k_norm),
            ("v_idx", impl._v_idx),
            ("v_norm", impl._v_norm),
        ]
        if algo == "prod":
            bufs.append(("k_qjl_sign", impl._k_qjl_sign))
            bufs.append(("k_rnorm", impl._k_rnorm))
        for name, t in bufs:
            nb = t.numel() * t.element_size()
            total += nb
            print(f"  algo={algo} {name:<12} shape={tuple(t.shape)} "
                  f"dtype={t.dtype} bytes={fmt_bytes(nb)}")
        print(f"  algo={algo}  per-layer total = {fmt_bytes(total)}")
        print()


def main():
    ap = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter,
                                 description=__doc__)
    ap.add_argument("--num-blocks", type=int, default=23000,
                    help="paged blocks (vLLM determines this from gpu mem)")
    ap.add_argument("--block-size", type=int, default=16)
    ap.add_argument("--num-kv-heads", type=int, default=8,
                    help="GQA: 8 for Llama-3 8B, 4 for Llama-3.2-3B")
    ap.add_argument("--head-size", type=int, default=128)
    ap.add_argument("--num-layers", type=int, default=32,
                    help="32 for Llama-3 8B")
    ap.add_argument("--runtime", action="store_true",
                    help="also do runtime allocation check on a GPU")
    ap.add_argument("--gpu", type=int, default=0,
                    help="CUDA device for --runtime")
    args = ap.parse_args()

    report(args.num_blocks, args.block_size, args.num_kv_heads,
           args.head_size, args.num_layers)
    if args.runtime:
        print()
        print("=" * 75)
        runtime_check(args.gpu)


if __name__ == "__main__":
    main()
