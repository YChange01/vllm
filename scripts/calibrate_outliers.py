# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compute per-(layer, kv_head) outlier channel indices.

Runs an HF transformers model on a calibration dataset, hooks each
layer's ``k_proj`` / ``v_proj`` outputs, accumulates per-channel
magnitude statistics, and emits the top-N outlier channels per kv_head
per layer. vLLM is not needed for this step.

Consumed by ``TurboQuantAttentionImpl`` (Stage 3b) to split each K and
V vector into an outlier slice and a regular slice, each quantized by
its own TurboQuant instance (paper arXiv:2504.19874, Section 4.3).

Usage
-----
    python3 scripts/calibrate_outliers.py \\
        --model meta-llama/Llama-3.1-8B-Instruct \\
        --dataset wikitext2 \\
        --num-samples 128 \\
        --seq-len 2048 \\
        --num-outliers 32 \\
        --device cuda:0 \\
        --output /tmp/outliers_llama3_8b_32.pt

Output
------
A torch.save()'d dict with the following keys:

  model_name       : str
  head_dim         : int
  num_layers       : int
  num_kv_heads     : int
  num_outliers     : int
  metric           : str ("mean_abs" currently)
  k_outlier_idx    : int32 (num_layers, num_kv_heads, num_outliers)
  v_outlier_idx    : int32 (num_layers, num_kv_heads, num_outliers)
  k_stats          : fp32 (num_layers, num_kv_heads, head_dim)  raw magnitudes
  v_stats          : fp32 (num_layers, num_kv_heads, head_dim)

The runtime attention backend selects the top-``num_outliers`` columns
per row of ``*_outlier_idx`` and quantizes them separately at a higher
bit-width than the remaining channels. Averaging the magnitude over
calibration tokens is consistent with SmoothQuant / QuaRot-style
offline outlier selection; TurboQuant paper references [63, 51] but
does not specify the selection rule.
"""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path

import torch


def _load_calibration_texts(name: str, num_samples: int) -> list[str]:
    """Load text samples from a small, well-known calibration set."""
    from datasets import load_dataset

    if name == "wikitext2":
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
        texts = [t for t in ds["text"] if len(t.strip()) > 200][:num_samples]
    elif name == "c4":
        ds = load_dataset(
            "allenai/c4", "en", split="train", streaming=True
        )
        texts = []
        for row in ds:
            if len(row["text"].strip()) > 200:
                texts.append(row["text"])
            if len(texts) >= num_samples:
                break
    else:
        raise ValueError(f"Unknown dataset {name!r}; use 'wikitext2' or 'c4'.")
    if len(texts) < num_samples:
        raise RuntimeError(
            f"Only found {len(texts)} calibration samples, wanted "
            f"{num_samples}. Pick a different dataset or lower the count."
        )
    return texts


class _ChannelMagnitudeCollector:
    """Online mean-abs accumulator per (layer, kv_head, channel).

    Stats shape: (num_layers, num_kv_heads, head_dim), fp64 for
    numerical stability across many tokens.
    """

    def __init__(
        self,
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
        device: torch.device,
    ) -> None:
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.k_sum = torch.zeros(
            num_layers, num_kv_heads, head_dim,
            dtype=torch.float64, device=device,
        )
        self.v_sum = torch.zeros_like(self.k_sum)
        self.count = torch.zeros(num_layers, dtype=torch.int64, device=device)

    def update(
        self, layer_idx: int, k: torch.Tensor, v: torch.Tensor
    ) -> None:
        """k, v shape: (..., num_kv_heads, head_dim) after reshape."""
        k2 = k.reshape(-1, self.num_kv_heads, self.head_dim).to(torch.float64)
        v2 = v.reshape(-1, self.num_kv_heads, self.head_dim).to(torch.float64)
        self.k_sum[layer_idx] += k2.abs().sum(dim=0)
        self.v_sum[layer_idx] += v2.abs().sum(dim=0)
        self.count[layer_idx] += k2.shape[0]

    def finalize(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (k_mean_abs, v_mean_abs) as fp32, shape (L, H_kv, d)."""
        count = self.count.view(-1, 1, 1).clamp_min(1)
        k_mean = (self.k_sum / count).to(torch.float32)
        v_mean = (self.v_sum / count).to(torch.float32)
        return k_mean.cpu(), v_mean.cpu()


def _register_kv_hooks(
    model,
    num_kv_heads: int,
    head_dim: int,
    collector: _ChannelMagnitudeCollector,
) -> list:
    """Install forward hooks on every self_attn.k_proj / v_proj."""
    handles = []
    layers = model.model.layers

    def _make_hook(layer_idx: int, which: str):
        def hook(module, inputs, output):
            # output shape: (..., num_kv_heads * head_dim).
            shape = output.shape
            assert shape[-1] == num_kv_heads * head_dim, (
                f"Unexpected {which}_proj output dim {shape[-1]} != "
                f"{num_kv_heads} * {head_dim}"
            )
            reshaped = output.reshape(
                -1, num_kv_heads, head_dim
            ).detach()
            if which == "k":
                collector.update(layer_idx, reshaped, reshaped.new_zeros(reshaped.shape))
                # update stored v stat in separate pass -- hook for v_proj
                # updates v_sum too. Avoid double-counting by splitting:
                collector.v_sum[layer_idx] -= reshaped.new_zeros(reshaped.shape[1:], dtype=torch.float64)
            return output
        return hook

    # Separate K and V hooks so we don't conflate the two tensors.
    def _k_hook(layer_idx: int):
        def fn(module, inputs, output):
            reshaped = output.reshape(-1, num_kv_heads, head_dim).detach()
            collector.k_sum[layer_idx] += reshaped.abs().to(torch.float64).sum(dim=0)
            collector.count[layer_idx] += reshaped.shape[0]
        return fn

    def _v_hook(layer_idx: int):
        def fn(module, inputs, output):
            reshaped = output.reshape(-1, num_kv_heads, head_dim).detach()
            collector.v_sum[layer_idx] += reshaped.abs().to(torch.float64).sum(dim=0)
            # count is incremented by the k hook for this layer.
        return fn

    for i, layer in enumerate(layers):
        k_proj = layer.self_attn.k_proj
        v_proj = layer.self_attn.v_proj
        handles.append(k_proj.register_forward_hook(_k_hook(i)))
        handles.append(v_proj.register_forward_hook(_v_hook(i)))
    return handles


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", default="wikitext2",
                        choices=["wikitext2", "c4"])
    parser.add_argument("--num-samples", type=int, default=128)
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--num-outliers", type=int, default=32,
                        help="Number of outlier channels per kv_head.")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--output", required=True)
    parser.add_argument("--hf-cache", default=None,
                        help="Override HF_HOME / TRANSFORMERS_CACHE.")
    args = parser.parse_args()

    if args.hf_cache:
        os.environ["HF_HOME"] = args.hf_cache
        os.environ["TRANSFORMERS_CACHE"] = args.hf_cache

    # Import late so --help is fast.
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16,
             "float32": torch.float32}[args.dtype]

    print(f"Loading model {args.model} ({args.dtype}) on {args.device} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=dtype, device_map=args.device,
    )
    model.eval()

    cfg = model.config
    num_kv_heads = getattr(cfg, "num_key_value_heads", cfg.num_attention_heads)
    head_dim = cfg.hidden_size // cfg.num_attention_heads
    num_layers = cfg.num_hidden_layers
    print(f"  num_layers={num_layers}, num_kv_heads={num_kv_heads}, "
          f"head_dim={head_dim}")

    if args.num_outliers <= 0 or args.num_outliers >= head_dim:
        raise ValueError(
            f"num_outliers={args.num_outliers} must satisfy "
            f"0 < num_outliers < head_dim={head_dim}."
        )

    collector = _ChannelMagnitudeCollector(
        num_layers, num_kv_heads, head_dim,
        device=torch.device(args.device),
    )
    handles = _register_kv_hooks(model, num_kv_heads, head_dim, collector)
    try:
        texts = _load_calibration_texts(args.dataset, args.num_samples)
        print(f"Running calibration on {len(texts)} samples "
              f"(seq_len <= {args.seq_len}) ...")
        with torch.inference_mode():
            for i, text in enumerate(texts):
                inp = tokenizer(
                    text, return_tensors="pt", truncation=True,
                    max_length=args.seq_len,
                ).to(args.device)
                if inp.input_ids.shape[1] < 32:
                    continue  # skip tiny snippets
                model(**inp)
                if (i + 1) % 16 == 0 or (i + 1) == len(texts):
                    print(f"  [{i + 1}/{len(texts)}] tokens accumulated "
                          f"per layer ~= {int(collector.count[0].item()):,}")
    finally:
        for h in handles:
            h.remove()

    k_stats, v_stats = collector.finalize()  # fp32 (L, H_kv, d)

    # Top-N outlier channels per (layer, kv_head) by magnitude.
    N = args.num_outliers
    k_outlier_idx = torch.topk(k_stats, N, dim=-1).indices.to(torch.int32)
    v_outlier_idx = torch.topk(v_stats, N, dim=-1).indices.to(torch.int32)

    payload = {
        "model_name": args.model,
        "head_dim": int(head_dim),
        "num_layers": int(num_layers),
        "num_kv_heads": int(num_kv_heads),
        "num_outliers": int(N),
        "metric": "mean_abs",
        "k_outlier_idx": k_outlier_idx,
        "v_outlier_idx": v_outlier_idx,
        "k_stats": k_stats,
        "v_stats": v_stats,
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, out_path)
    print(f"\nWrote outlier indices to {out_path}")

    # Quick summary for sanity: average concentration ratio across layers.
    mean_top_k = k_stats.gather(-1, k_outlier_idx.long()).mean(dim=-1)
    mean_all = k_stats.mean(dim=-1)
    k_ratio = (mean_top_k / mean_all.clamp_min(1e-9)).mean().item()
    mean_top_v = v_stats.gather(-1, v_outlier_idx.long()).mean(dim=-1)
    mean_all_v = v_stats.mean(dim=-1)
    v_ratio = (mean_top_v / mean_all_v.clamp_min(1e-9)).mean().item()
    print(f"Average top-{N} vs mean channel magnitude: "
          f"K = {k_ratio:.2f}x, V = {v_ratio:.2f}x")
    print("(Ratios >>1 indicate strong outlier concentration -- good.)")


if __name__ == "__main__":
    main()
