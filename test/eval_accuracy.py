#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Accuracy-style evals (gsm8k / boolq / mmlu / gpqa) against a running
vLLM OpenAI-compatible server. Single harness with per-metric scorers:

    exact_match_number  -- extract last number from output (GSM8K)
    yes_no              -- check yes/no tokens (BoolQ)
    multiple_choice     -- match first A/B/C/D (MMLU, GPQA)

Layout and flow mirror test/eval_longbench.py so stage sweeps and
output parsing stay consistent.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path


# ---------------------------------------------------------------------------
# Metric functions
# ---------------------------------------------------------------------------
_NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")
_CHOICE_RE = re.compile(r"\b([ABCD])\b")


def _normalize_number(s: str) -> str:
    return s.replace(",", "").replace("$", "").strip()


def _extract_last_number(text: str) -> str | None:
    """GSM8K convention: the last number in the model's output is the
    predicted final answer. Works for both CoT ('...the answer is 42')
    and bare numeric outputs."""
    if not text:
        return None
    cleaned = _normalize_number(text)
    m = list(_NUM_RE.finditer(cleaned))
    if not m:
        return None
    return m[-1].group(0)


def _score_exact_match_number(pred: str, gold: str) -> float:
    p = _extract_last_number(pred or "")
    if p is None:
        return 0.0
    try:
        return 1.0 if float(p) == float(gold) else 0.0
    except ValueError:
        return 0.0


def _score_yes_no(pred: str, gold: str) -> float:
    if not pred:
        return 0.0
    low = pred.strip().lower()
    # First occurrence decides (handles "yes, because..." answers).
    yes_at = low.find("yes")
    no_at = low.find("no")
    if yes_at == -1 and no_at == -1:
        return 0.0
    if yes_at == -1:
        pred_label = "no"
    elif no_at == -1:
        pred_label = "yes"
    else:
        pred_label = "yes" if yes_at < no_at else "no"
    return 1.0 if pred_label == gold else 0.0


def _score_multiple_choice(pred: str, gold: str) -> float:
    if not pred:
        return 0.0
    m = _CHOICE_RE.search(pred.upper())
    if not m:
        return 0.0
    return 1.0 if m.group(1) == gold else 0.0


_METRIC_FNS = {
    "exact_match_number": _score_exact_match_number,
    "yes_no":              _score_yes_no,
    "multiple_choice":     _score_multiple_choice,
}


# ---------------------------------------------------------------------------
# vLLM HTTP helpers
# ---------------------------------------------------------------------------
def _post_completion(
    endpoint: str, model: str, prompt: str,
    max_tokens: int, timeout: float,
) -> str:
    data = json.dumps({
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "top_p": 1.0,
    }).encode("utf-8")
    req = urllib.request.Request(
        f"{endpoint}/v1/completions",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        obj = json.loads(resp.read().decode("utf-8"))
    return obj["choices"][0]["text"]


def _tokenize(
    endpoint: str, model: str, prompt: str, timeout: float,
) -> tuple[int, list[int]]:
    data = json.dumps({"model": model, "prompt": prompt}).encode("utf-8")
    req = urllib.request.Request(
        f"{endpoint}/tokenize",
        data=data, headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        obj = json.loads(resp.read().decode("utf-8"))
    return obj["count"], obj["tokens"]


def _detokenize(
    endpoint: str, model: str, tokens: list[int], timeout: float,
) -> str:
    data = json.dumps({"model": model, "tokens": tokens}).encode("utf-8")
    req = urllib.request.Request(
        f"{endpoint}/detokenize",
        data=data, headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        obj = json.loads(resp.read().decode("utf-8"))
    return obj["prompt"]


def _truncate_middle(
    endpoint: str, model: str, prompt: str,
    max_prompt_tokens: int, timeout: float,
) -> tuple[str, int]:
    """LongBench-style middle truncation. Only BoolQ passages come close
    to the budget; GSM8K / MMLU / GPQA prompts are under 2K tokens."""
    count, tokens = _tokenize(endpoint, model, prompt, timeout)
    if count <= max_prompt_tokens:
        return prompt, count
    half = max_prompt_tokens // 2
    keep = tokens[:half] + tokens[-half:]
    return _detokenize(endpoint, model, keep, timeout), len(keep)


# ---------------------------------------------------------------------------
# Prompt building
# ---------------------------------------------------------------------------
def _build_prompt(template: str, row: dict) -> str:
    if "{c0}" in template:
        choices = row.get("choices", [])
        mapping = {
            "question": row.get("question", ""),
            "subject":  row.get("subject", ""),
            "c0": choices[0] if len(choices) > 0 else "",
            "c1": choices[1] if len(choices) > 1 else "",
            "c2": choices[2] if len(choices) > 2 else "",
            "c3": choices[3] if len(choices) > 3 else "",
        }
        return template.format(**mapping)
    return template.format(
        question=row.get("question", ""),
        passage=row.get("passage", ""),
        subject=row.get("subject", ""),
    )


# ---------------------------------------------------------------------------
# Per-task runner
# ---------------------------------------------------------------------------
def _run_task(
    task: str, meta: dict, rows: list[dict],
    endpoint: str, model: str, timeout: float,
    max_samples: int | None,
    max_prompt_tokens: int | None,
) -> tuple[list[float], list[dict]]:
    scores: list[float] = []
    details: list[dict] = []
    n = len(rows) if max_samples is None else min(len(rows), max_samples)
    metric = meta["metric"]
    max_output = meta["max_output"]
    template = meta["prompt_template"]
    fn = _METRIC_FNS[metric]
    n_truncated = 0

    t_start = time.time()
    for i in range(n):
        row = rows[i]
        prompt = _build_prompt(template, row)
        if max_prompt_tokens is not None:
            try:
                prompt, final = _truncate_middle(
                    endpoint, model, prompt, max_prompt_tokens, timeout,
                )
                if final == max_prompt_tokens:
                    n_truncated += 1
            except Exception as e:
                print(
                    f"  [{task}] {i+1}/{n} tokenize failed "
                    f"({type(e).__name__}); sending untruncated prompt",
                    file=sys.stderr,
                )
        try:
            pred = _post_completion(
                endpoint, model, prompt, max_output, timeout,
            )
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", errors="ignore")[:300]
            print(
                f"  [{task}] {i+1}/{n} HTTP {e.code}: {detail}",
                file=sys.stderr,
            )
            pred = ""
        except Exception as e:
            print(f"  [{task}] {i+1}/{n} error: {e}", file=sys.stderr)
            pred = ""

        gold = row.get("gold", "")
        s = fn(pred, gold)
        scores.append(s)
        details.append({
            "idx": i, "pred": pred[:400], "gold": gold, "score": s,
        })
        if (i + 1) % 25 == 0 or (i + 1) == n:
            elapsed = time.time() - t_start
            trunc = f" trunc={n_truncated}" if n_truncated else ""
            print(
                f"  [{task}] {i+1:5d}/{n} "
                f"acc={sum(scores)/len(scores):.4f}  "
                f"({elapsed:.0f}s){trunc}",
                flush=True,
            )

    return scores, details


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _load_task_rows(data_dir: Path, task: str) -> list[dict]:
    p = data_dir / f"{task}.jsonl"
    if not p.exists():
        raise FileNotFoundError(
            f"Missing {p}. Run scripts/dump_accuracy.py first."
        )
    rows = []
    with p.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoint", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--data-dir", default="calib_data/accuracy")
    ap.add_argument(
        "--tasks", default=None,
        help="Comma-separated task list. Default: all in config.json.",
    )
    ap.add_argument("--tag", default="run")
    ap.add_argument(
        "--max-samples", type=int, default=None,
        help="Cap per-task sample count (smoke testing).",
    )
    ap.add_argument(
        "--request-timeout", type=float, default=600.0,
        help="Per-request HTTP timeout in seconds.",
    )
    ap.add_argument(
        "--max-context", type=int, default=32768,
        help="Model max_model_len; prompts above "
             "(max_context - max_output - margin) get middle-truncated.",
    )
    ap.add_argument(
        "--truncation-margin", type=int, default=32,
    )
    ap.add_argument(
        "--no-truncate", action="store_true",
        help="Disable middle truncation.",
    )
    ap.add_argument("--save-details", default=None)
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    cfg_path = data_dir / "config.json"
    if not cfg_path.exists():
        print(f"Missing {cfg_path}; run scripts/dump_accuracy.py",
              file=sys.stderr)
        return 1
    config = json.loads(cfg_path.read_text(encoding="utf-8"))

    if args.tasks:
        tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
        for t in tasks:
            if t not in config:
                print(
                    f"Task {t!r} not in config.json; known: "
                    f"{sorted(config.keys())}",
                    file=sys.stderr,
                )
                return 1
    else:
        tasks = sorted(config.keys())

    print(f"=== Accuracy eval [{args.tag}] ===")
    print(f"  endpoint: {args.endpoint}")
    print(f"  model:    {args.model}")
    print(f"  tasks:    {len(tasks)} -> {', '.join(tasks)}")
    if args.max_samples:
        print(f"  max_samples per task: {args.max_samples}")
    print()

    per_task: dict[str, float] = {}
    all_details: dict[str, list[dict]] = {}
    t0 = time.time()

    for task in tasks:
        meta = config[task]
        rows = _load_task_rows(data_dir, task)
        if args.no_truncate:
            budget = None
        else:
            budget = (
                args.max_context - int(meta["max_output"])
                - args.truncation_margin
            )
        print(
            f"[{task}] {len(rows)} examples, metric={meta['metric']}"
            f"{f', prompt_budget={budget}' if budget else ''}"
        )
        scores, details = _run_task(
            task, meta, rows,
            endpoint=args.endpoint,
            model=args.model,
            timeout=args.request_timeout,
            max_samples=args.max_samples,
            max_prompt_tokens=budget,
        )
        avg = sum(scores) / len(scores) if scores else 0.0
        per_task[task] = avg
        all_details[task] = details
        print(f"  -> {task}: acc={avg:.4f}\n")

    total = time.time() - t0

    print("=" * 72)
    print(f"Accuracy results [{args.tag}]  (total {total:.0f}s)")
    print(f"  {'task':10s} {'n':>6s} {'metric':>20s} {'accuracy':>10s}")
    print("  " + "-" * 54)
    overall = 0.0
    for t in tasks:
        cfg = config[t]
        n = min(len(all_details[t]), args.max_samples or 10**9)
        print(
            f"  {t:10s} {n:6d} {cfg['metric']:>20s}  {per_task[t]:8.4f}"
        )
        overall += per_task[t]
    avg_all = overall / len(tasks) if tasks else 0.0
    print("  " + "-" * 54)
    print(f"  {'OVERALL (avg)':10s} {'':6s} {'':20s}  {avg_all:8.4f}")
    print("=" * 72)

    if args.save_details:
        with open(args.save_details, "w", encoding="utf-8") as f:
            json.dump(
                {"tag": args.tag, "per_task": per_task,
                 "details": all_details},
                f, ensure_ascii=False, indent=2,
            )
        print(f"Wrote detail log: {args.save_details}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
