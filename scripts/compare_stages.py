#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Aggregate storage + throughput/latency into one comparison table.

Storage numbers come from ``vllm.turboquant.compression`` (single source
of truth). Throughput / latency numbers come from ``test/throughput.sh``
output -- per-stage ``<tag>_bench.log`` files are parsed with regexes
matching the ``vllm bench serve`` summary block.

Usage:
    # After running test/throughput.sh (or test/compare.sh):
    python3 scripts/compare_stages.py
    python3 scripts/compare_stages.py --log-dir logs/throughput_20260420_180000
    python3 scripts/compare_stages.py \
        --stages "FLASH_ATTN TURBOQUANT_b4 TURBOQUANT_split_3_5bit"
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path


# ``vllm bench serve`` summary block regexes. Labels are stable across
# current vLLM versions; keep the match loose (any whitespace + colon).
_BENCH_PATTERNS: dict[str, re.Pattern[str]] = {
    "req_per_s":  re.compile(r"Request throughput[^:\n]*:\s*([\d.]+)"),
    "out_tps":    re.compile(r"Output token throughput[^:\n]*:\s*([\d.]+)"),
    "total_tps":  re.compile(r"Total Token throughput[^:\n]*:\s*([\d.]+)"),
    "mean_ttft":  re.compile(r"Mean TTFT[^:\n]*:\s*([\d.]+)"),
    "mean_itl":   re.compile(r"Mean ITL[^:\n]*:\s*([\d.]+)"),
}


def _storage_row(
    tag: str, head_dim: int, num_outliers: int
) -> dict[str, float]:
    """Per-stage storage breakdown.

    Returns a dict with:
      B_per_tok  -- K+V bytes per (token, kv_head)
      B_per_128  -- K+V bytes per 128 tokens per kv_head
      ratio      -- bf16_baseline / kv_total
      eff_bpc    -- effective bits per coord including metadata
    """
    if tag == "FLASH_ATTN":
        bf16_kv = head_dim * 2 * 2   # K + V, 2 bytes per coord
        return {
            "B_per_tok": float(bf16_kv),
            "B_per_128": float(bf16_kv * 128),
            "ratio":     1.0,
            "eff_bpc":   16.0,
        }
    # Lazy import: compression.py pulls in vllm, which isn't importable
    # on the dev Mac (torch mismatch). Importing here lets --help work
    # anywhere.
    from vllm.turboquant.compression import report_for_config
    from vllm.turboquant.stages import resolve_stage

    _, cfg = resolve_stage(tag)
    rep = report_for_config(
        cfg, head_dim=head_dim, num_outliers=num_outliers,
    )
    return {
        "B_per_tok": float(rep.kv_total),
        "B_per_128": float(rep.kv_total * 128),
        "ratio":     rep.compression_ratio,
        "eff_bpc":   rep.effective_bits_per_coord,
    }


def _parse_bench_log(log: Path) -> dict[str, float | None]:
    if not log.exists():
        return {k: None for k in _BENCH_PATTERNS}
    text = log.read_text(encoding="utf-8", errors="replace")
    out: dict[str, float | None] = {}
    for key, pat in _BENCH_PATTERNS.items():
        m = pat.search(text)
        out[key] = float(m.group(1)) if m else None
    return out


def _fmt(v: float | None, decimals: int, width: int) -> str:
    if v is None:
        return f"{'?':>{width}s}"
    return f"{v:>{width}.{decimals}f}"


def main() -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Merge storage (from vllm.turboquant.compression) with "
            "throughput/latency (from throughput.sh bench logs) into "
            "one comparison table."
        )
    )
    ap.add_argument(
        "--log-dir", default="logs/throughput_latest",
        help="Directory from test/throughput.sh containing <tag>_bench.log",
    )
    ap.add_argument(
        "--stages", default=None,
        help=(
            "Space/comma separated stage list. Default: every _bench.log "
            "file found in --log-dir, plus FLASH_ATTN if missing."
        ),
    )
    ap.add_argument("--head-dim", type=int, default=128)
    ap.add_argument("--num-outliers", type=int, default=32)
    ap.add_argument(
        "--csv", default=None,
        help="Also write a CSV copy of the table to this path.",
    )
    args = ap.parse_args()

    log_dir = Path(args.log_dir)
    if args.stages:
        stages = [s for s in re.split(r"[,\s]+", args.stages.strip()) if s]
    else:
        if not log_dir.exists():
            print(f"[compare] log dir not found: {log_dir}")
            return 1
        stages = sorted(
            p.name.removesuffix("_bench.log")
            for p in log_dir.glob("*_bench.log")
        )
        if not stages:
            print(f"[compare] no *_bench.log in {log_dir}")
            return 1

    rows = []
    for tag in stages:
        try:
            sto = _storage_row(tag, args.head_dim, args.num_outliers)
        except Exception as e:
            print(f"[compare] skip {tag}: {e}")
            continue
        bench = _parse_bench_log(log_dir / f"{tag}_bench.log")
        rows.append((tag, sto, bench))

    print()
    print(
        f"head_dim={args.head_dim}, "
        f"num_outliers(split)={args.num_outliers}, "
        f"log_dir={log_dir}"
    )
    print()
    header = (
        f"{'stage':<28s} "
        f"{'B/tok':>6s} {'B/128':>7s} {'ratio':>7s} {'eff_bpc':>8s}  "
        f"{'req/s':>7s} {'out_tps':>9s} {'tot_tps':>9s} "
        f"{'ttft_ms':>8s} {'itl_ms':>8s}"
    )
    print(header)
    print("-" * len(header))

    for tag, sto, b in rows:
        line = (
            f"{tag:<28s} "
            f"{int(sto['B_per_tok']):>6d} "
            f"{int(sto['B_per_128']):>7d} "
            f"{sto['ratio']:>6.2f}x "
            f"{sto['eff_bpc']:>8.2f}  "
            f"{_fmt(b['req_per_s'],  2, 7)} "
            f"{_fmt(b['out_tps'],    1, 9)} "
            f"{_fmt(b['total_tps'],  1, 9)} "
            f"{_fmt(b['mean_ttft'],  1, 8)} "
            f"{_fmt(b['mean_itl'],   2, 8)}"
        )
        print(line)
    print()

    if args.csv:
        import csv
        with open(args.csv, "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow([
                "stage", "B_per_tok", "B_per_128", "compression_ratio",
                "effective_bits_per_coord", "req_per_s", "output_tps",
                "total_tps", "mean_ttft_ms", "mean_itl_ms",
            ])
            for tag, sto, b in rows:
                w.writerow([
                    tag,
                    int(sto["B_per_tok"]), int(sto["B_per_128"]),
                    round(sto["ratio"], 4),
                    round(sto["eff_bpc"], 4),
                    b["req_per_s"], b["out_tps"], b["total_tps"],
                    b["mean_ttft"], b["mean_itl"],
                ])
        print(f"[compare] wrote CSV -> {args.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
