#!/usr/bin/env python3
"""GSM8K accuracy eval against a running vLLM OpenAI-compatible server.

Targets a chat model (the standard TurboQuant eval runs Llama-3.1-8B-
Instruct). Reads the gsm8k test split from HuggingFace datasets, sends
each question to ``/v1/chat/completions``, extracts the final numeric
answer (``#### N`` format preferred, else last number in response),
and compares to ground truth.

Parallelism via a thread pool so vLLM can batch concurrent requests --
keeps wall time sane for N >= 100.

Usage:
    python3 test/eval_gsm8k.py --port 8009 --n 100
    python3 test/eval_gsm8k.py --port 8009 --n 200 --concurrency 32
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests


_FINAL_ANS_RE = re.compile(r"####\s*(-?\d[\d,]*\.?\d*)")
_NUM_RE = re.compile(r"-?\d[\d,]*\.?\d*")

_SYSTEM = (
    "You are a careful math tutor. Reason step by step. After the "
    "reasoning, output the final numeric answer on its own line in "
    "the form '#### <answer>'."
)
_USER_TMPL = (
    "Problem: {q}\n\n"
    "Solve step by step, then output the final numeric answer after "
    "'####'."
)


def load_gsm8k(n: int) -> list[dict]:
    try:
        from datasets import load_dataset
    except ImportError:
        sys.exit(
            "[gsm8k] `datasets` not installed. "
            "Run: pip install datasets"
        )
    ds = load_dataset("gsm8k", "main", split="test")
    if n > 0:
        ds = ds.select(range(min(n, len(ds))))
    return [{"question": x["question"], "answer": x["answer"]} for x in ds]


def _norm_num(s: str) -> str:
    return s.replace(",", "").rstrip(".")


def extract_answer(text: str) -> str | None:
    m = _FINAL_ANS_RE.search(text)
    if m:
        return _norm_num(m.group(1))
    nums = _NUM_RE.findall(text)
    if nums:
        return _norm_num(nums[-1])
    return None


def ground_truth(answer_field: str) -> str:
    m = _FINAL_ANS_RE.search(answer_field)
    if not m:
        raise ValueError(f"no #### in ground truth: {answer_field!r}")
    return _norm_num(m.group(1))


def query_one(port: int, model: str, user_msg: str,
              max_tokens: int, timeout: int) -> str:
    r = requests.post(
        f"http://localhost:{port}/v1/chat/completions",
        json={
            "model": model,
            "messages": [
                {"role": "system", "content": _SYSTEM},
                {"role": "user", "content": user_msg},
            ],
            "max_tokens": max_tokens,
            "temperature": 0.0,
        },
        timeout=timeout,
    )
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


def run_eval(port: int, model: str, problems: list[dict],
             max_tokens: int, timeout: int, concurrency: int,
             show_wrong: int) -> dict:
    results: list[dict | None] = [None] * len(problems)

    def work(i: int) -> dict:
        p = problems[i]
        user_msg = _USER_TMPL.format(q=p["question"])
        try:
            resp = query_one(port, model, user_msg, max_tokens, timeout)
            err = None
        except Exception as e:  # noqa: BLE001
            resp, err = "", str(e)
        pred = extract_answer(resp) if resp else None
        gt = ground_truth(p["answer"])
        return {
            "i": i, "resp": resp, "err": err, "pred": pred, "gt": gt,
            "correct": (pred is not None and pred == gt),
        }

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futs = {pool.submit(work, i): i for i in range(len(problems))}
        done_count = 0
        for fut in as_completed(futs):
            r = fut.result()
            results[r["i"]] = r
            done_count += 1
            if done_count % 20 == 0 or done_count == len(problems):
                so_far = [x for x in results if x is not None]
                n_ok = sum(1 for x in so_far if x["correct"])
                print(f"[gsm8k] {done_count}/{len(problems)}  "
                      f"acc_so_far={n_ok}/{len(so_far)} "
                      f"= {n_ok/max(1, len(so_far)):.1%}",
                      flush=True)
    dt = time.time() - t0

    assert all(r is not None for r in results)
    n_correct = sum(1 for r in results if r["correct"])
    n_parsed = sum(1 for r in results if r["pred"] is not None)
    n_err = sum(1 for r in results if r["err"] is not None)
    n_total = len(results)

    wrong = [r for r in results if not r["correct"]][:show_wrong]
    return {
        "n_total": n_total,
        "n_correct": n_correct,
        "n_parsed": n_parsed,
        "n_err": n_err,
        "elapsed": dt,
        "wrong_samples": wrong,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument(
        "--model",
        default="/mnt/nvme3n1/g00872988/models/Llama-3.1-8B-Instruct",
    )
    ap.add_argument("--n", type=int, default=100,
                    help="number of test problems (<=0 => all 1319)")
    ap.add_argument("--max_tokens", type=int, default=512)
    ap.add_argument("--timeout", type=int, default=120)
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--show_wrong", type=int, default=3)
    args = ap.parse_args()

    problems = load_gsm8k(args.n)
    print(f"[gsm8k] loaded {len(problems)} problems, "
          f"port={args.port}, concurrency={args.concurrency}", flush=True)

    summary = run_eval(
        port=args.port,
        model=args.model,
        problems=problems,
        max_tokens=args.max_tokens,
        timeout=args.timeout,
        concurrency=args.concurrency,
        show_wrong=args.show_wrong,
    )

    n = summary["n_total"]
    print()
    print("======================================================")
    print(f"GSM8K  n={n}  port={args.port}")
    print(f"  accuracy      : {summary['n_correct']}/{n} = "
          f"{summary['n_correct']/n:.2%}")
    print(f"  parsed rate   : {summary['n_parsed']}/{n} = "
          f"{summary['n_parsed']/n:.2%}")
    print(f"  request errors: {summary['n_err']}/{n}")
    print(f"  elapsed       : {summary['elapsed']:.1f}s  "
          f"({summary['elapsed']/n:.2f}s/q avg, with concurrency="
          f"{args.concurrency})")
    print("======================================================")

    if summary["wrong_samples"]:
        print(f"\nFirst {len(summary['wrong_samples'])} wrong "
              "(trimmed, for inspection):")
        for w in summary["wrong_samples"]:
            q = w["i"]
            err_str = f" err={w['err']!r}" if w["err"] else ""
            print(f"  q{q}: gt={w['gt']}  pred={w['pred']}{err_str}")
            tail = (w["resp"] or "")[-200:].replace("\n", " ")
            print(f"    resp tail: ...{tail}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
