#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Download LongBench v1 tasks and pack into per-task JSONL files.

Can run on a connected machine (Mac) and the resulting directory
scp'd to B200, or directly on B200 if it has outbound HTTPS to HF.

Usage:
    python3 scripts/dump_longbench.py --output calib_data/longbench_v1
    python3 scripts/dump_longbench.py --output calib_data/longbench_v1 \
            --subset english
    python3 scripts/dump_longbench.py --output calib_data/longbench_v1 \
            --subset mini
    python3 scripts/dump_longbench.py --output calib_data/longbench_v1 \
            --tasks narrativeqa,qasper
    # Inside Huawei network (B200), route through the internal mirror:
    python3 scripts/dump_longbench.py --output calib_data/longbench_v1 \
            --tasks narrativeqa --mirror huawei

Each task emits <output>/<task>.jsonl, one row per test example. The
script also writes <output>/config.json with per-task metadata
(prompt template, max output tokens, metric name, language) vendored
from https://github.com/THUDM/LongBench. The eval harness reads this
config at runtime.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


# Internal HF mirror endpoints (no auth, no SSL verify).
# Extend here if more corporate mirrors are needed.
_MIRRORS: dict[str, tuple[str, bool, bool]] = {
    # name: (endpoint, verify_ssl, trust_env_proxy)
    "huawei":   ("http://mirrors.tools.huawei.com/huggingface", False, False),
    "hfmirror": ("https://hf-mirror.com",                       True,  True),
}


def _apply_hf_mirror(name: str) -> None:
    """Route huggingface_hub + datasets through an alternative endpoint.

    Must be called before `from datasets import load_dataset` runs.
    `huawei` targets the internal intranet mirror (no SSL, no proxy);
    `hfmirror` targets the public hf-mirror.com (SSL on, honors env
    proxy so HTTP_PROXY / HTTPS_PROXY still work).
    """
    import requests
    from huggingface_hub import configure_http_backend

    endpoint, verify_ssl, trust_env = _MIRRORS[name]
    os.environ["HF_ENDPOINT"] = endpoint

    def _factory() -> requests.Session:
        session = requests.Session()
        session.verify = verify_ssl
        session.trust_env = trust_env
        return session

    configure_http_backend(backend_factory=_factory)


# Full set of 21 LongBench v1 tasks (HF config names).
#   en: English, zh: Chinese
#   metric: one of {qa_f1, qa_f1_zh, rouge, rouge_zh, classification,
#                    retrieval_en, retrieval_zh, count, code_sim}
_TASK_META: dict[str, dict] = {
    # Single-doc QA
    "narrativeqa":          {"lang": "en", "metric": "qa_f1",          "max_output": 128},
    "qasper":               {"lang": "en", "metric": "qa_f1",          "max_output": 128},
    "multifieldqa_en":      {"lang": "en", "metric": "qa_f1",          "max_output": 64},
    "multifieldqa_zh":      {"lang": "zh", "metric": "qa_f1_zh",       "max_output": 64},
    # Multi-doc QA
    "hotpotqa":             {"lang": "en", "metric": "qa_f1",          "max_output": 32},
    "2wikimqa":             {"lang": "en", "metric": "qa_f1",          "max_output": 32},
    "musique":              {"lang": "en", "metric": "qa_f1",          "max_output": 32},
    "dureader":             {"lang": "zh", "metric": "rouge_zh",       "max_output": 128},
    # Summarization
    "gov_report":           {"lang": "en", "metric": "rouge",          "max_output": 512},
    "qmsum":                {"lang": "en", "metric": "rouge",          "max_output": 512},
    "multi_news":           {"lang": "en", "metric": "rouge",          "max_output": 512},
    "vcsum":                {"lang": "zh", "metric": "rouge_zh",       "max_output": 512},
    # Few-shot
    "trec":                 {"lang": "en", "metric": "classification", "max_output": 64},
    "triviaqa":             {"lang": "en", "metric": "qa_f1",          "max_output": 32},
    "samsum":               {"lang": "en", "metric": "rouge",          "max_output": 128},
    "lsht":                 {"lang": "zh", "metric": "classification", "max_output": 64},
    # Synthetic
    "passage_count":        {"lang": "en", "metric": "count",          "max_output": 32},
    "passage_retrieval_en": {"lang": "en", "metric": "retrieval_en",   "max_output": 32},
    "passage_retrieval_zh": {"lang": "zh", "metric": "retrieval_zh",   "max_output": 32},
    # Code
    "lcc":                  {"lang": "en", "metric": "code_sim",       "max_output": 64},
    "repobench-p":          {"lang": "en", "metric": "code_sim",       "max_output": 64},
}


# Prompt templates, vendored from LongBench repo's
# config/dataset2prompt.json (commit on 2024-02). These are stable
# across versions; kept here so eval doesn't depend on network access.
_PROMPTS: dict[str, str] = {
    "narrativeqa": (
        "You are given a story, which can be either a novel or a movie script, "
        "and a question. Answer the question as concisely as you can, using a "
        "single phrase if possible. Do not provide any explanation.\n\n"
        "Story: {context}\n\n"
        "Now, answer the question based on the story as concisely as you can, "
        "using a single phrase if possible. Do not provide any explanation.\n\n"
        "Question: {input}\n\nAnswer:"
    ),
    "qasper": (
        "You are given a scientific article and a question. Answer the question "
        "as concisely as you can, using a single phrase or sentence if possible. "
        'If the question cannot be answered based on the information in the '
        'article, write "unanswerable". If the question is a yes/no question, '
        'answer "yes", "no", or "unanswerable". Do not provide any explanation.\n\n'
        "Article: {context}\n\n"
        "Answer the question based on the above article as concisely as you can, "
        "using a single phrase or sentence if possible. If the question cannot be "
        'answered based on the information in the article, write "unanswerable". '
        'If the question is a yes/no question, answer "yes", "no", or "unanswerable". '
        "Do not provide any explanation.\n\n"
        "Question: {input}\n\nAnswer:"
    ),
    "multifieldqa_en": (
        "Read the following text and answer briefly.\n\n{context}\n\n"
        "Now, answer the following question based on the above text, only give "
        "me the answer and do not output any other words.\n\n"
        "Question: {input}\nAnswer:"
    ),
    "multifieldqa_zh": (
        "阅读以下文字并用中文简短回答：\n\n{context}\n\n"
        "现在请基于上面的文章回答下面的问题，只告诉我答案，不要输出任何其他字词。"
        "\n\n问题：{input}\n回答："
    ),
    "hotpotqa": (
        "Answer the question based on the given passages. Only give me the answer "
        "and do not output any other words.\n\n"
        "The following are given passages.\n{context}\n\n"
        "Answer the question based on the given passages. Only give me the answer "
        "and do not output any other words.\n\nQuestion: {input}\nAnswer:"
    ),
    "2wikimqa": (
        "Answer the question based on the given passages. Only give me the answer "
        "and do not output any other words.\n\n"
        "The following are given passages.\n{context}\n\n"
        "Answer the question based on the given passages. Only give me the answer "
        "and do not output any other words.\n\nQuestion: {input}\nAnswer:"
    ),
    "musique": (
        "Answer the question based on the given passages. Only give me the answer "
        "and do not output any other words.\n\n"
        "The following are given passages.\n{context}\n\n"
        "Answer the question based on the given passages. Only give me the answer "
        "and do not output any other words.\n\nQuestion: {input}\nAnswer:"
    ),
    "dureader": (
        "请基于给定的文章回答下述问题。\n\n文章：{context}\n\n"
        "请基于上述文章回答下面的问题。\n\n问题：{input}\n回答："
    ),
    "gov_report": (
        "You are given a report by a government agency. Write a one-page "
        "summary of the report.\n\nReport:\n{context}\n\n"
        "Now, write a one-page summary of the report.\n\nSummary:"
    ),
    "qmsum": (
        "You are given a meeting transcript and a query containing a question "
        "or instruction. Answer the query in one or more sentences.\n\n"
        "Transcript:\n{context}\n\nNow, answer the query based on the above "
        "meeting transcript in one or more sentences.\n\nQuery: {input}\nAnswer:"
    ),
    "multi_news": (
        "You are given several news passages. Write a one-page summary of all "
        "news. \n\nNews:\n{context}\n\nNow, write a one-page summary of all "
        "the news.\n\nSummary:"
    ),
    "vcsum": (
        "下面有一段会议记录，请你阅读后，写一段总结，总结会议的内容。\n"
        "会议记录：\n{context}\n\n会议总结："
    ),
    "trec": (
        "Please determine the type of the question below. Here are some examples "
        "of questions.\n\n{context}\n{input}"
    ),
    "triviaqa": (
        "Answer the question based on the given passage. Only give me the answer "
        "and do not output any other words. The following are some examples.\n\n"
        "{context}\n\n{input}"
    ),
    "samsum": (
        "Summarize the dialogue into a few short sentences. The following are some "
        "examples.\n\n{context}\n\n{input}"
    ),
    "lsht": (
        "请判断给定新闻的类别，下面是一些例子。\n\n{context}\n{input}"
    ),
    "passage_count": (
        "There are some paragraphs below sourced from Wikipedia. Some of them may "
        "be duplicates. Please carefully read these paragraphs and determine how "
        "many unique paragraphs there are after removing duplicates. In other words, "
        "how many non-repeating paragraphs are there in total?\n\n{context}\n\n"
        "Please enter the final count of unique paragraphs after removing duplicates. "
        'The output format should only contain the number, such as 1, 2, 3, and so on.\n\n'
        "The final answer is: "
    ),
    "passage_retrieval_en": (
        "Here are 30 paragraphs from Wikipedia, along with an abstract. Please "
        "determine which paragraph the abstract is from.\n\n{context}\n\n"
        "The following is an abstract.\n\n{input}\n\n"
        "Please enter the number of the paragraph that the abstract is from. "
        "The answer format must be like \"Paragraph 1\", \"Paragraph 2\", etc.\n\n"
        "The answer is: "
    ),
    "passage_retrieval_zh": (
        "以下是若干段落文字，以及其中一个段落的摘要。请确定给定的摘要出自哪一段。\n\n"
        "{context}\n\n下面是一个摘要\n\n{input}\n\n"
        "请输入摘要所属段落的编号。答案格式必须是\"段落1\"，\"段落2\"等格式\n\n答案是："
    ),
    "lcc": (
        "Please complete the code given below. \n{context}Next line of code:\n"
    ),
    "repobench-p": (
        "Please complete the code given below. \n{context}{input}Next line of code:\n"
    ),
}


# Curated subsets for partial runs.
_SUBSETS: dict[str, list[str]] = {
    "full": sorted(_TASK_META.keys()),
    "english": sorted(
        t for t, m in _TASK_META.items() if m["lang"] == "en"
    ),
    "chinese": sorted(
        t for t, m in _TASK_META.items() if m["lang"] == "zh"
    ),
    # Paper Table-1-like small subset: highest-signal English tasks.
    "mini": [
        "narrativeqa", "qasper", "hotpotqa",
        "gov_report", "qmsum", "multi_news",
    ],
}


def _dump_one(task: str, out_dir: Path) -> int:
    """Download one LongBench task from HF and write as JSONL.

    Returns number of rows written.
    """
    from datasets import load_dataset

    ds = load_dataset("THUDM/LongBench", task, split="test")
    out = out_dir / f"{task}.jsonl"
    n = 0
    with out.open("w", encoding="utf-8") as f:
        for row in ds:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    size_mb = out.stat().st_size / 1e6
    print(f"  {task:28s} {n:4d} rows, {size_mb:5.2f} MB -> {out}")
    return n


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--output", default="calib_data/longbench_v1",
        help="Output directory for per-task JSONL files.",
    )
    ap.add_argument(
        "--subset", choices=sorted(_SUBSETS.keys()), default="english",
        help="Curated subset to download (default: english = 16 tasks).",
    )
    ap.add_argument(
        "--tasks", default=None,
        help="Comma-separated task names (overrides --subset).",
    )
    ap.add_argument(
        "--mirror", choices=sorted(_MIRRORS.keys()), default=None,
        help="Route HF through internal mirror (e.g. 'huawei' for B200).",
    )
    ap.add_argument(
        "--offline", action="store_true",
        help=(
            "Skip HF download; only write config.json. Expects .jsonl "
            "files to be present already (copied from a machine with "
            "HF access)."
        ),
    )
    args = ap.parse_args()

    if args.mirror:
        _apply_hf_mirror(args.mirror)
        print(f"[mirror] HF_ENDPOINT={os.environ['HF_ENDPOINT']} (verify=off)")

    if args.tasks:
        tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
        for t in tasks:
            if t not in _TASK_META:
                known = ", ".join(sorted(_TASK_META.keys()))
                raise ValueError(
                    f"Unknown task {t!r}. Known: {known}"
                )
    else:
        tasks = _SUBSETS[args.subset]

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    total_rows = 0
    if args.offline:
        print(f"[offline] skipping HF; checking .jsonl files in {out_dir}")
        missing = [t for t in tasks if not (out_dir / f"{t}.jsonl").exists()]
        if missing:
            raise FileNotFoundError(
                f"Missing {len(missing)} .jsonl file(s): {missing}. "
                "Download them on a machine with HF access and scp into "
                f"{out_dir} before running --offline."
            )
        for t in tasks:
            jl = out_dir / f"{t}.jsonl"
            with jl.open("r", encoding="utf-8") as f:
                n = sum(1 for _ in f)
            size_mb = jl.stat().st_size / 1e6
            print(f"  {t:28s} {n:4d} rows, {size_mb:5.2f} MB [present]")
            total_rows += n
    else:
        print(f"Downloading {len(tasks)} task(s) to {out_dir} ...")
        for t in tasks:
            total_rows += _dump_one(t, out_dir)

    # Write task config (prompt template + max output + metric + lang).
    # The eval harness reads this to reconstruct prompts.
    config = {
        t: {
            **_TASK_META[t],
            "prompt_template": _PROMPTS[t],
        }
        for t in tasks
    }
    (out_dir / "config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\nWrote config.json for {len(tasks)} tasks")
    print(f"Total rows: {total_rows}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
