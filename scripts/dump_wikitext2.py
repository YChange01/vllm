#!/usr/bin/env python3
"""One-shot download of wikitext-2 calibration texts on an
internet-connected machine. Writes a JSONL file suitable for offline
consumption by scripts/calibrate_outliers.py on B200.

Usage:
    python3 scripts/dump_wikitext2.py --output ~/Desktop/wikitext2_calib.jsonl

JSONL format: one JSON object per line, with a single ``text`` field.
calibrate_outliers.py --texts-file <jsonl> filters on len(text) > 200
and takes the first N entries matching --num-samples.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--output",
        default=str(Path.home() / "Desktop" / "wikitext2_calib.jsonl"),
    )
    ap.add_argument("--min-chars", type=int, default=200)
    ap.add_argument(
        "--max-samples", type=int, default=512,
        help="Cap on number of rows written. Defaults to 512 so "
             "calibrate_outliers.py --num-samples up to 512 works.",
    )
    args = ap.parse_args()

    # Import late so --help is fast.
    from datasets import load_dataset

    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    texts = [t for t in ds["text"] if len(t.strip()) > args.min_chars]
    texts = texts[: args.max_samples]

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for t in texts:
            f.write(json.dumps({"text": t}, ensure_ascii=False) + "\n")
    size_mb = out.stat().st_size / 1e6
    print(f"wrote {len(texts)} samples -> {out} ({size_mb:.2f} MB)")


if __name__ == "__main__":
    main()
