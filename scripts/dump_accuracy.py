#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Dump accuracy-style eval datasets (gsm8k / boolq / mmlu / gpqa).

Produces ``calib_data/accuracy/<task>.jsonl`` + ``config.json`` with
prompt template and metric name per task. Mirrors the layout of
``scripts/dump_longbench.py`` so ``test/eval_accuracy.py`` can treat
both harnesses the same way.

Usage:
    python3 scripts/dump_accuracy.py
    python3 scripts/dump_accuracy.py --tasks gsm8k,boolq
    python3 scripts/dump_accuracy.py --mirror hfmirror
    python3 scripts/dump_accuracy.py --offline   # validate existing files
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import zipfile
from pathlib import Path


# ---------------------------------------------------------------------------
# HF mirror registry (shared with dump_longbench.py layout)
# ---------------------------------------------------------------------------
_MIRRORS: dict[str, tuple[str, bool, bool]] = {
    "huawei":   ("http://mirrors.tools.huawei.com/huggingface", False, False),
    "hfmirror": ("https://hf-mirror.com",                       True,  True),
}


def _apply_hf_mirror(name: str) -> None:
    import requests
    from huggingface_hub import configure_http_backend

    endpoint, verify, trust_env = _MIRRORS[name]
    os.environ["HF_ENDPOINT"] = endpoint

    def _factory() -> requests.Session:
        s = requests.Session()
        s.verify = verify
        s.trust_env = trust_env
        return s

    configure_http_backend(backend_factory=_factory)


# ---------------------------------------------------------------------------
# Per-task config (prompt templates + metrics + max_output).
# Kept in code so config.json can be regenerated deterministically.
# ---------------------------------------------------------------------------
_CONFIG: dict[str, dict] = {
    "gsm8k": {
        "metric": "exact_match_number",
        "max_output": 256,
        "prompt_template": (
            "Question: {question}\n\n"
            "Answer: Let's think step by step."
        ),
    },
    "boolq": {
        "metric": "yes_no",
        "max_output": 4,
        "prompt_template": (
            "Passage: {passage}\n\n"
            "Question: {question}?\n\n"
            "Answer (yes or no):"
        ),
    },
    "mmlu": {
        "metric": "multiple_choice",
        "max_output": 8,
        "prompt_template": (
            "The following is a multiple choice question about "
            "{subject}.\n\n"
            "{question}\nA) {c0}\nB) {c1}\nC) {c2}\nD) {c3}\n\n"
            "Answer:"
        ),
    },
    "gpqa": {
        "metric": "multiple_choice",
        "max_output": 8,
        "prompt_template": (
            "{question}\nA) {c0}\nB) {c1}\nC) {c2}\nD) {c3}\n\n"
            "Answer:"
        ),
    },
}


_KNOWN_TASKS = list(_CONFIG.keys())


# ---------------------------------------------------------------------------
# Per-task dumpers
# ---------------------------------------------------------------------------
def _dump_gsm8k(out_dir: Path) -> int:
    from datasets import load_dataset

    ds = load_dataset("openai/gsm8k", "main", split="test")
    p = out_dir / "gsm8k.jsonl"
    with p.open("w", encoding="utf-8") as f:
        for i, row in enumerate(ds):
            ans = row["answer"]
            gold = ans.split("####")[-1].strip().replace(",", "")
            f.write(json.dumps({
                "id": i,
                "question": row["question"],
                "answer_full": ans,
                "gold": gold,
            }, ensure_ascii=False) + "\n")
    return len(ds)


def _dump_boolq(out_dir: Path) -> int:
    from datasets import load_dataset

    ds = load_dataset("google/boolq", split="validation")
    p = out_dir / "boolq.jsonl"
    with p.open("w", encoding="utf-8") as f:
        for i, row in enumerate(ds):
            f.write(json.dumps({
                "id": i,
                "question": row["question"],
                "passage": row["passage"],
                "gold": "yes" if row["answer"] else "no",
            }, ensure_ascii=False) + "\n")
    return len(ds)


def _dump_mmlu(out_dir: Path) -> int:
    from datasets import load_dataset

    ds = load_dataset("cais/mmlu", "all", split="test")
    p = out_dir / "mmlu.jsonl"
    with p.open("w", encoding="utf-8") as f:
        for i, row in enumerate(ds):
            f.write(json.dumps({
                "id": i,
                "subject": row["subject"],
                "question": row["question"],
                "choices": row["choices"],
                "answer_idx": row["answer"],
                "gold": "ABCD"[row["answer"]],
            }, ensure_ascii=False) + "\n")
    return len(ds)


def _dump_gpqa(out_dir: Path) -> int:
    """GPQA main subset. Source is github.com/idavidrein/gpqa's
    password-protected dataset.zip (password in the repo README).
    Choices get shuffled with a deterministic per-example seed so
    gold letters distribute evenly."""
    import urllib.request

    zip_url = (
        "https://raw.githubusercontent.com/idavidrein/gpqa/main/dataset.zip"
    )
    zip_path = Path("/tmp/gpqa_dataset.zip")
    csv_path = Path("/tmp/dataset/gpqa_main.csv")
    password = b"deserted-untie-orchid"

    if not csv_path.exists():
        if not zip_path.exists():
            print("  fetching gpqa dataset.zip ...")
            urllib.request.urlretrieve(zip_url, zip_path)
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall("/tmp", pwd=password)

    with csv_path.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    out_path = out_dir / "gpqa.jsonl"
    with out_path.open("w", encoding="utf-8") as f:
        for i, row in enumerate(rows):
            correct = row["Correct Answer"].strip()
            wrongs = [
                row["Incorrect Answer 1"].strip(),
                row["Incorrect Answer 2"].strip(),
                row["Incorrect Answer 3"].strip(),
            ]
            order = list(range(4))
            random.Random(42 + i).shuffle(order)
            choices_src = [correct, *wrongs]   # 0 == correct
            choices = [choices_src[j] for j in order]
            gold_letter = "ABCD"[order.index(0)]
            f.write(json.dumps({
                "id": i,
                "subject": row.get("High-level domain", ""),
                "question": row["Question"].strip(),
                "choices": choices,
                "gold": gold_letter,
            }, ensure_ascii=False) + "\n")
    return len(rows)


_DUMPERS = {
    "gsm8k": _dump_gsm8k,
    "boolq": _dump_boolq,
    "mmlu":  _dump_mmlu,
    "gpqa":  _dump_gpqa,
}


def _write_config(out_dir: Path, tasks: list[str]) -> None:
    """Write the subset of _CONFIG matching dumped tasks."""
    existing = {}
    cfg_path = out_dir / "config.json"
    if cfg_path.exists():
        existing = json.loads(cfg_path.read_text(encoding="utf-8"))
    for t in tasks:
        existing[t] = _CONFIG[t]
    cfg_path.write_text(
        json.dumps(existing, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--output", default="calib_data/accuracy",
        help="Target directory for <task>.jsonl + config.json",
    )
    ap.add_argument(
        "--tasks", default=None,
        help="Comma-separated subset of {gsm8k,boolq,mmlu,gpqa}. "
             "Default: all four.",
    )
    ap.add_argument(
        "--mirror", choices=sorted(_MIRRORS.keys()), default=None,
        help="HF mirror. 'huawei' for intranet, 'hfmirror' for public.",
    )
    ap.add_argument(
        "--offline", action="store_true",
        help="Validate that <task>.jsonl files already exist; "
             "re-emit config.json but don't touch HF / network.",
    )
    args = ap.parse_args()

    if args.mirror:
        _apply_hf_mirror(args.mirror)
        print(f"[mirror] HF_ENDPOINT={os.environ['HF_ENDPOINT']}")

    if args.tasks:
        tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
        for t in tasks:
            if t not in _KNOWN_TASKS:
                raise ValueError(
                    f"Unknown task {t!r}; known: {_KNOWN_TASKS}"
                )
    else:
        tasks = _KNOWN_TASKS

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    total = 0
    for t in tasks:
        p = out_dir / f"{t}.jsonl"
        if args.offline:
            if not p.exists():
                raise FileNotFoundError(
                    f"{p} missing -- drop the file in or re-run without "
                    "--offline to download."
                )
            with p.open(encoding="utf-8") as f:
                n = sum(1 for _ in f)
            size_mb = p.stat().st_size / 1e6
            print(f"  {t:6s} {n:5d} rows, {size_mb:5.2f} MB [present]")
        else:
            print(f"  {t} -> {p} ...")
            n = _DUMPERS[t](out_dir)
            size_mb = p.stat().st_size / 1e6
            print(f"  {t:6s} {n:5d} rows, {size_mb:5.2f} MB")
        total += n

    _write_config(out_dir, tasks)
    print(f"\nwrote config.json ({len(tasks)} tasks)")
    print(f"total rows: {total}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
