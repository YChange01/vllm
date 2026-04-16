#!/usr/bin/env python3
"""Needle-in-a-Haystack accuracy eval against an OpenAI-compatible vLLM endpoint.

Synthesizes long-context prompts with a single "magic number" sentence buried
somewhere inside a sea of filler, asks the model to recall the number, and
scores by substring match on the completion. Produces a ctx x position grid
so you can spot backend-specific long-context degradation.

Matches the style of the paper's NIAH experiments (arXiv:2504.19874) but uses
deterministic filler + digit match for a fast, reproducible CI-friendly signal.

Usage:
    python test/eval_niah.py \
        --endpoint http://localhost:8009 \
        --model /mnt/nvme3n1/g00872988/models/Qwen3-0.6B \
        --ctx-lens 512,2048,4096 \
        --positions 0.1,0.5,0.9 \
        --trials 3 \
        --tag TQ_b8
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from typing import Any

import urllib.request

FILLER_SENTENCE = "The cat sat on the mat and watched the rain. "


_REQUEST_TIMEOUT = 1800.0  # set from CLI; default 30 min for first Triton JIT


def _post(url: str, payload: dict[str, Any], timeout: float | None = None) -> dict[str, Any]:
    if timeout is None:
        timeout = _REQUEST_TIMEOUT
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def count_tokens(text: str, endpoint: str, model: str) -> int:
    """Ask the server to tokenize and return the token count."""
    # /tokenize is cheap; cap at 60s so a dead server fails fast.
    out = _post(
        f"{endpoint}/tokenize",
        {"model": model, "prompt": text},
        timeout=60.0,
    )
    # vLLM returns {"tokens": [...], "count": int, "max_model_len": int}
    return int(out.get("count", len(out.get("tokens", []))))


def build_haystack(
    target_tokens: int,
    needle_pos: float,
    needle_num: int,
    endpoint: str,
    model: str,
) -> tuple[str, int]:
    """Build a prompt whose token length is close to target_tokens, with the
    needle sentence inserted at the given fractional position. Returns
    (prompt, actual_token_count)."""
    needle = f"The magic number is {needle_num}. Remember this well. "
    suffix = "\n\nQuestion: What is the magic number mentioned above?\nAnswer:"

    # Reserve tokens for needle + suffix.
    overhead = count_tokens(needle + suffix, endpoint, model)
    budget = max(1, target_tokens - overhead)

    per_sent = max(1, count_tokens(FILLER_SENTENCE, endpoint, model))
    n_sents = max(1, budget // per_sent)
    filler = FILLER_SENTENCE * n_sents

    # Snap the insertion point to a sentence boundary so the needle is not
    # spliced into the middle of a token.
    char_pos = int(len(filler) * max(0.0, min(1.0, needle_pos)))
    while 0 < char_pos < len(filler) and filler[char_pos - 1] != " ":
        char_pos -= 1

    prompt = filler[:char_pos] + needle + filler[char_pos:] + suffix
    actual = count_tokens(prompt, endpoint, model)
    return prompt, actual


def query(
    prompt: str,
    endpoint: str,
    model: str,
    max_tokens: int = 16,
) -> str:
    out = _post(
        f"{endpoint}/v1/completions",
        {
            "model": model,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": 0,
        },
    )
    try:
        return out["choices"][0]["text"]
    except (KeyError, IndexError):
        return f"<bad response: {out}>"


def grade(completion: str, needle_num: int) -> bool:
    # Pull every integer-ish run out of the answer; pass if the exact needle
    # digits appear. Tolerates extra tokens around the number.
    nums = re.findall(r"\d+", completion)
    return str(needle_num) in nums


def run_grid(args: argparse.Namespace) -> None:
    random.seed(args.seed)

    ctx_lens = [int(x) for x in args.ctx_lens.split(",") if x.strip()]
    positions = [float(x) for x in args.positions.split(",") if x.strip()]

    results: dict[tuple[int, float], tuple[int, int]] = {}
    t0 = time.time()

    total = len(ctx_lens) * len(positions) * args.trials
    done = 0

    for ctx in ctx_lens:
        for pos in positions:
            correct = 0
            for trial in range(args.trials):
                needle = random.randint(10_000, 99_999)
                prompt, actual_tokens = build_haystack(
                    ctx, pos, needle, args.endpoint, args.model
                )
                req_t0 = time.time()
                try:
                    completion = query(
                        prompt, args.endpoint, args.model, args.max_tokens
                    )
                    ok = grade(completion, needle)
                    req_elapsed = time.time() - req_t0
                    status = "OK" if ok else "FAIL"
                except Exception as e:
                    completion = f"<error: {type(e).__name__}: {e}>"
                    ok = False
                    req_elapsed = time.time() - req_t0
                    status = "ERR "
                correct += int(ok)
                done += 1
                shown = completion.replace("\n", " ").strip()[:60]
                print(
                    f"[{args.tag}] {done}/{total} "
                    f"ctx~{ctx}(={actual_tokens}) pos={pos:.2f} "
                    f"needle={needle} {status} ({req_elapsed:.1f}s) "
                    f"| {shown!r}",
                    flush=True,
                )
            results[(ctx, pos)] = (correct, args.trials)

    elapsed = time.time() - t0

    # Grid summary
    print(f"\n=== NIAH grid [{args.tag}] (elapsed {elapsed:.0f}s) ===")
    header = f"{'ctx':>7}"
    for p in positions:
        header += f"{f'pos={p:.2f}':>12}"
    header += f"{'row_avg':>10}"
    print(header)
    print("-" * len(header))

    grand_correct = 0
    grand_total = 0
    for ctx in ctx_lens:
        row = f"{ctx:>7}"
        row_correct = 0
        for p in positions:
            c, t = results[(ctx, p)]
            row += f"{f'{c}/{t}':>12}"
            row_correct += c
        row_total = args.trials * len(positions)
        row += f"{f'{row_correct / row_total:.1%}':>10}"
        grand_correct += row_correct
        grand_total += row_total
        print(row)
    print("-" * len(header))
    print(f"overall [{args.tag}]: {grand_correct}/{grand_total}"
          f" = {grand_correct / grand_total:.1%}\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoint", required=True,
                    help="e.g. http://localhost:8009")
    ap.add_argument("--model", required=True,
                    help="model path or name as served by vLLM")
    ap.add_argument("--ctx-lens", default="512,2048,4096",
                    help="comma-separated target token counts")
    ap.add_argument("--positions", default="0.1,0.5,0.9",
                    help="comma-separated needle positions (fraction of ctx)")
    ap.add_argument("--trials", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-tokens", type=int, default=16)
    ap.add_argument("--tag", default="run",
                    help="label for this run in logs / summary")
    ap.add_argument("--request-timeout", type=float, default=1800.0,
                    help="per-request timeout in seconds (default 1800, "
                         "generous for first Triton JIT compile)")
    args = ap.parse_args()

    global _REQUEST_TIMEOUT
    _REQUEST_TIMEOUT = float(args.request_timeout)

    try:
        run_grid(args)
    except urllib.error.URLError as e:
        print(f"[{args.tag}] endpoint unreachable: {e}", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
