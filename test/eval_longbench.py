#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""LongBench evaluation against a running vLLM OpenAI-compatible server.

For each task:
  1. Load per-task JSONL (produced by scripts/dump_longbench.py).
  2. Apply the official LongBench prompt template.
  3. Send each example to the server; capture the completion.
  4. Score against gold answers using the task's metric.

The LongBench metric implementations are vendored here (trimmed from
https://github.com/THUDM/LongBench/blob/main/metrics.py). Keep them in
sync with the LongBench repo if metric definitions change.

Usage:
    python3 test/eval_longbench.py \\
        --endpoint http://localhost:8009 \\
        --model /path/to/Llama-3.1-8B-Instruct \\
        --data-dir calib_data/longbench_v1 \\
        --tasks narrativeqa,qasper \\
        --tag TURBOQUANT_b4 \\
        --max-samples 50
"""

from __future__ import annotations

import argparse
import json
import re
import string
import sys
import time
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path


# ---------------------------------------------------------------------------
# Vendored LongBench metrics (from THUDM/LongBench/metrics.py)
# ---------------------------------------------------------------------------
def _normalize_answer(s: str) -> str:
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def _normalize_zh_answer(s: str) -> str:
    # Remove CJK punctuation + English punctuation + spaces.
    # Smart-quote characters escaped via \u to avoid delimiter conflicts.
    cn_punctuation = (
        "！？｡。＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠"
        "［＼］＾＿｀｛｜｝～｟｠｢｣､、〃》「」『』"
        "【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—"
        "\u2018\u2019\u201a\u201b"  # single-quote variants
        "\u201c\u201d\u201e\u201f"  # double-quote variants
        "…‧﹏."
    )
    all_punctuation = set(string.punctuation + cn_punctuation)
    return "".join(
        ch for ch in s.lower() if ch not in all_punctuation and not ch.isspace()
    )


def _qa_f1_score(prediction: str, ground_truth: str) -> float:
    norm_pred = _normalize_answer(prediction).split()
    norm_gold = _normalize_answer(ground_truth).split()
    common = Counter(norm_pred) & Counter(norm_gold)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(norm_pred) if norm_pred else 0
    recall = num_same / len(norm_gold) if norm_gold else 0
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _qa_f1_zh_score(prediction: str, ground_truth: str) -> float:
    # Character-level F1 for Chinese.
    norm_pred = list(_normalize_zh_answer(prediction))
    norm_gold = list(_normalize_zh_answer(ground_truth))
    common = Counter(norm_pred) & Counter(norm_gold)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(norm_pred) if norm_pred else 0
    recall = num_same / len(norm_gold) if norm_gold else 0
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _rouge_score(prediction: str, ground_truth: str) -> float:
    """Rouge-L (F-score) using a simple longest common subsequence impl.

    LongBench officially uses rouge-chinese / rouge libs, but pulling
    those in is heavy. LCS-based rouge-L is close enough for relative
    ranking across KV-quant configs.
    """
    p = _normalize_answer(prediction).split()
    g = _normalize_answer(ground_truth).split()
    if not p or not g:
        return 0.0
    # LCS length
    n, m = len(p), len(g)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if p[i - 1] == g[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    lcs = dp[n][m]
    if lcs == 0:
        return 0.0
    prec = lcs / n
    rec = lcs / m
    return 2 * prec * rec / (prec + rec)


def _rouge_zh_score(prediction: str, ground_truth: str) -> float:
    # Character-level LCS for Chinese.
    p = list(_normalize_zh_answer(prediction))
    g = list(_normalize_zh_answer(ground_truth))
    if not p or not g:
        return 0.0
    n, m = len(p), len(g)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if p[i - 1] == g[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    lcs = dp[n][m]
    if lcs == 0:
        return 0.0
    prec = lcs / n
    rec = lcs / m
    return 2 * prec * rec / (prec + rec)


def _classification_score(
    prediction: str, ground_truth: str, all_classes: list[str] | None = None
) -> float:
    em_match_list = []
    all_classes_norm = (
        [c.lower() for c in all_classes] if all_classes else []
    )
    pred_lower = prediction.lower()
    for c in all_classes_norm:
        if c in pred_lower:
            em_match_list.append(c)
    for c in all_classes_norm:
        if c in ground_truth.lower():
            if c in em_match_list:
                return 1.0 / len(em_match_list) if em_match_list else 0.0
    return 0.0


def _retrieval_score(prediction: str, ground_truth: str) -> float:
    pattern = r"Paragraph (\d+)"
    matches = re.findall(pattern, ground_truth)
    ground_truth_id = matches[0] if matches else ""
    numbers = re.findall(r"\d+", prediction)
    return 1.0 if ground_truth_id in numbers else 0.0


def _retrieval_zh_score(prediction: str, ground_truth: str) -> float:
    pattern = r"段落(\d+)"
    matches = re.findall(pattern, ground_truth)
    ground_truth_id = matches[0] if matches else ""
    numbers = re.findall(r"\d+", prediction)
    return 1.0 if ground_truth_id in numbers else 0.0


def _count_score(prediction: str, ground_truth: str) -> float:
    numbers = re.findall(r"\d+", prediction)
    pred = numbers[0] if numbers else ""
    return 1.0 if pred == ground_truth else 0.0


def _code_sim_score(prediction: str, ground_truth: str) -> float:
    from difflib import SequenceMatcher
    # First-line of prediction vs gold; LongBench strips at first newline.
    pred = prediction.lstrip().split("\n")[0]
    gold = ground_truth.lstrip().split("\n")[0]
    return SequenceMatcher(None, pred, gold).ratio()


_METRIC_FNS = {
    "qa_f1": _qa_f1_score,
    "qa_f1_zh": _qa_f1_zh_score,
    "rouge": _rouge_score,
    "rouge_zh": _rouge_zh_score,
    "classification": _classification_score,
    "retrieval_en": _retrieval_score,
    "retrieval_zh": _retrieval_zh_score,
    "count": _count_score,
    "code_sim": _code_sim_score,
}


def score_example(
    metric: str, prediction: str, gold_answers: list[str],
    all_classes: list[str] | None = None,
) -> float:
    """Best score over all gold answers (LongBench convention)."""
    fn = _METRIC_FNS[metric]
    best = 0.0
    for gold in gold_answers:
        if metric == "classification":
            s = fn(prediction, gold, all_classes)
        else:
            s = fn(prediction, gold)
        if s > best:
            best = s
    return best


# ---------------------------------------------------------------------------
# Inference against vLLM
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
    """Call vLLM's /tokenize endpoint. Returns (count, token_ids)."""
    data = json.dumps({"model": model, "prompt": prompt}).encode("utf-8")
    req = urllib.request.Request(
        f"{endpoint}/tokenize",
        data=data,
        headers={"Content-Type": "application/json"},
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
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        obj = json.loads(resp.read().decode("utf-8"))
    return obj["prompt"]


def _truncate_middle(
    endpoint: str, model: str, prompt: str,
    max_prompt_tokens: int, timeout: float,
) -> tuple[str, int]:
    """LongBench-style middle truncation: keep first half + last half.

    Returns the (possibly-truncated) prompt and its final token count.
    Mirrors THUDM/LongBench/pred.py so overlong samples don't hit the
    server's max_model_len check.
    """
    count, tokens = _tokenize(endpoint, model, prompt, timeout)
    if count <= max_prompt_tokens:
        return prompt, count
    half = max_prompt_tokens // 2
    keep = tokens[:half] + tokens[-half:]
    return _detokenize(endpoint, model, keep, timeout), len(keep)


def _build_prompt(template: str, row: dict) -> str:
    # LongBench rows have `input` (question/query) and `context` (long doc).
    return template.format(
        input=row.get("input", ""),
        context=row.get("context", ""),
    )


def _run_task(
    task: str, meta: dict, rows: list[dict],
    endpoint: str, model: str, timeout: float,
    max_samples: int | None,
    max_prompt_tokens: int | None,
) -> tuple[list[float], list[dict]]:
    """Run inference on each row; return per-row scores + detail log."""
    scores: list[float] = []
    details: list[dict] = []
    n = len(rows) if max_samples is None else min(len(rows), max_samples)
    metric = meta["metric"]
    max_output = meta["max_output"]
    prompt_tmpl = meta["prompt_template"]
    n_truncated = 0

    t_start = time.time()
    for i in range(n):
        row = rows[i]
        prompt = _build_prompt(prompt_tmpl, row)
        if max_prompt_tokens is not None:
            try:
                prompt, final_tokens = _truncate_middle(
                    endpoint, model, prompt, max_prompt_tokens, timeout,
                )
                if final_tokens == max_prompt_tokens:
                    # was truncated to exact budget (LongBench: head+tail halves)
                    n_truncated += 1
            except Exception as e:
                print(
                    f"  [{task}] {i+1}/{n} tokenize failed "
                    f"({type(e).__name__}); sending untruncated prompt",
                    file=sys.stderr,
                )
        try:
            pred = _post_completion(
                endpoint, model, prompt, max_output, timeout
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

        gold = row.get("answers", [])
        if isinstance(gold, str):
            gold = [gold]
        all_classes = row.get("all_classes", None)
        s = score_example(metric, pred, gold, all_classes)
        scores.append(s)
        details.append({
            "idx": i,
            "pred": pred[:400],
            "gold": gold[:3] if isinstance(gold, list) else gold,
            "score": s,
        })
        if (i + 1) % 10 == 0 or (i + 1) == n:
            elapsed = time.time() - t_start
            trunc_note = (
                f" trunc={n_truncated}" if n_truncated else ""
            )
            print(
                f"  [{task}] {i+1:4d}/{n} "
                f"avg={sum(scores)/len(scores):.4f}  "
                f"({elapsed:.0f}s){trunc_note}",
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
            f"Missing {p}. Run scripts/dump_longbench.py first."
        )
    rows = []
    with p.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


# Paper-like small English subset (LongBench Table 1 highlights).
# Kept in sync with scripts/dump_longbench.py's _SUBSETS['mini'].
_MINI_TASKS = [
    "narrativeqa", "qasper", "hotpotqa",
    "gov_report", "qmsum", "multi_news",
]


def _resolve_subset(subset: str, config: dict) -> list[str]:
    """Map a subset name -> list of task names, derived from config.json."""
    all_tasks = sorted(config.keys())
    if subset == "full":
        return all_tasks
    if subset == "english":
        return sorted(t for t, m in config.items() if m.get("lang") == "en")
    if subset == "chinese":
        return sorted(t for t, m in config.items() if m.get("lang") == "zh")
    if subset == "mini":
        return [t for t in _MINI_TASKS if t in config]
    raise ValueError(
        f"Unknown subset {subset!r}; known: full, english, chinese, mini"
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoint", required=True)
    ap.add_argument("--model", required=True,
                    help="Model path used by vllm serve")
    ap.add_argument("--data-dir", default="calib_data/longbench_v1")
    ap.add_argument(
        "--tasks", default=None,
        help="Comma-separated task names. Overrides --subset.",
    )
    ap.add_argument(
        "--subset", default=None,
        choices=["full", "english", "chinese", "mini"],
        help="Curated task subset. Ignored if --tasks is set.",
    )
    ap.add_argument("--tag", default="run",
                    help="Label shown in output table / log filename.")
    ap.add_argument(
        "--max-samples", type=int, default=None,
        help="Cap per-task examples (for smoke testing).",
    )
    ap.add_argument(
        "--request-timeout", type=float, default=1800.0,
        help="Per-request HTTP timeout (seconds).",
    )
    ap.add_argument(
        "--max-context", type=int, default=32768,
        help=(
            "Model's max_model_len. Prompts exceeding "
            "max_context - task.max_output - margin are truncated "
            "from the middle (LongBench convention)."
        ),
    )
    ap.add_argument(
        "--truncation-margin", type=int, default=32,
        help="Safety margin in tokens; prompt_budget = max_context "
             "- task.max_output - margin.",
    )
    ap.add_argument(
        "--no-truncate", action="store_true",
        help="Disable middle truncation (send prompts as-is).",
    )
    ap.add_argument(
        "--save-details", default=None,
        help="Optional JSON file to dump per-example pred/gold/score.",
    )
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    config_path = data_dir / "config.json"
    if not config_path.exists():
        print(f"Missing {config_path}; run scripts/dump_longbench.py",
              file=sys.stderr)
        return 1
    config = json.loads(config_path.read_text(encoding="utf-8"))

    if args.tasks:
        tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
        for t in tasks:
            if t not in config:
                print(f"Task {t!r} not in config; available: "
                      f"{sorted(config.keys())}", file=sys.stderr)
                return 1
    elif args.subset:
        tasks = _resolve_subset(args.subset, config)
        if not tasks:
            print(f"Subset {args.subset!r} resolved to empty list",
                  file=sys.stderr)
            return 1
    else:
        tasks = sorted(config.keys())

    print(f"=== LongBench eval [{args.tag}] ===")
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
            if budget <= 0:
                raise ValueError(
                    f"[{task}] prompt budget <= 0 "
                    f"(max_context={args.max_context}, "
                    f"max_output={meta['max_output']}, "
                    f"margin={args.truncation_margin})"
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
        print(f"  -> {task}: avg={avg:.4f}\n")

    total = time.time() - t0

    # Summary table
    print("=" * 72)
    print(f"LongBench results [{args.tag}]  (total {total:.0f}s)")
    print(f"  {'task':28s} {'n':>6s} {'metric':>14s} {'score':>8s}")
    print("  " + "-" * 64)
    overall_sum = 0.0
    for t in tasks:
        cfg = config[t]
        n = min(len(all_details[t]), args.max_samples or 10**9)
        print(f"  {t:28s} {n:6d} {cfg['metric']:>14s}  {per_task[t]:6.4f}")
        overall_sum += per_task[t]
    avg_all = overall_sum / len(tasks) if tasks else 0.0
    print("  " + "-" * 64)
    print(f"  {'OVERALL (avg)':28s} {'':6s} {'':14s}  {avg_all:6.4f}")
    print("=" * 72)

    if args.save_details:
        with open(args.save_details, "w", encoding="utf-8") as f:
            json.dump(
                {"tag": args.tag, "per_task": per_task, "details": all_details},
                f, ensure_ascii=False, indent=2,
            )
        print(f"Wrote detail log: {args.save_details}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
