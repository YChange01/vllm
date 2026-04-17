# TurboQuant backend for vLLM

K-side KV cache quantization for vLLM, based on
[TurboQuant (Zandieh et al., arXiv:2504.19874)](https://arxiv.org/abs/2504.19874).
Algorithm prototype lives in [tqlite](https://github.com/YChange01/tqlite).

## What's implemented

- **Algorithm 1 (`Q_mse`)**: per-coord b-bit Lloyd-Max on
  `H @ diag(signs) @ k_normed` (Hadamard rotation of the unit-sphere-
  scaled K).
- **Algorithm 2 (`Q_prod`)**: Q_mse + 1-bit QJL on the residual,
  giving an unbiased inner-product estimate (paper Lemma 4).
- vLLM v1 attention backend (`TurboQuantAttentionBackend`,
  `TurboQuantAttentionImpl`).
- Triton attend kernel (per-query, on-the-fly K dequant, varlen +
  causal flash softmax). Pre-rotation of Q on-device via `torch.matmul`.
- Pure-PyTorch K store (Lloyd-Max bucket via `torch.searchsorted`,
  Hadamard rotation via cuBLAS matmul, scatter into paged cache via
  advanced indexing).

V is stored **raw** in the model dtype (no quantization). The paper
also quantizes V; an int8 V quant was attempted here but produced
`|attn|.max > max|v|` on Llama prefill -- mathematically impossible
for a correct softmax-weighted sum -- which traced to a Triton multi-
program write corruption on B200. K-only quant gives ~8x compression
on K (b=8) without that failure mode; V quant can be re-added later
with a different store path.

## Files

```
vllm/turboquant/
├── __init__.py
├── codebook.py            QuantState: Lloyd-Max + Hadamard + QJL S
├── store.py               turboquant_store_kv (pure PyTorch on GPU)
└── attend.py              Triton attend kernel + Python wrapper

vllm/v1/attention/backends/
├── turboquant_attn.py     Backend + Impl + MetadataBuilder
└── registry.py            TURBOQUANT enum entry
```

## Environment variables

| var | default | meaning |
|---|---|---|
| `TURBOQUANT_ALGO` | `prod` | `mse` (Algorithm 1) or `prod` (Algorithm 2) |
| `TURBOQUANT_BITS` | `8`    | total bit budget per coord; `prod` requires `>=2` |

## Test scripts

```
test/baseline.sh           Three-way correctness: FLASH_ATTN vs
                           TURBOQUANT b=8 vs TURBOQUANT b=4 on a
                           single completion request.

test/serve.sh              Manually start a TurboQuant server.
test/query.sh              Send one completion to the running server.

test/test_varlen_kernel.py Synthetic Gaussian K/V/Q kernel sanity
                           (mse_rel and prod_rel side by side).

test/eval.sh + eval_niah.py
                           NIAH (Needle in a Haystack) end-to-end
                           accuracy harness, three-way side by side.

test/temp.sh               Scratchpad for the current debugging probe.
```

All scripts write logs into `logs/<script>_<timestamp>/` with a
`logs/<script>_latest` symlink for convenience.

## Quickstart on B200

```bash
cd /mnt/nvme3n1/g00872988/turboquant/vllm
git pull --ff-only

# Three-way text comparison on Llama-3.1-8B-Instruct:
TURBOQUANT_ALGO=mse  TURBOQUANT_BITS=8 \
    bash test/baseline.sh "Hello" 4 0
TURBOQUANT_ALGO=prod TURBOQUANT_BITS=8 \
    bash test/baseline.sh "Hello" 4 0
```

For a longer prompt or different model, edit the `MODEL` env var or
positional args of `test/baseline.sh`.

## Known limitations

- V is not quantized (see "What's implemented" above).
- The Triton store kernel was removed; K store is pure PyTorch on the
  GPU. Slower than a fused Triton store would be, but only runs during
  prefill and during single-token storage on each decode step.
- ALiBi, sliding window, and `logits_soft_cap` are not supported.
- `cache_dtype` is fixed to `auto` (matches model dtype).
