# TurboQuant backend for vLLM

KV cache quantization for vLLM, based on
[TurboQuant (Zandieh et al., arXiv:2504.19874)](https://arxiv.org/abs/2504.19874).
Algorithm prototype lives in [tqlite](https://github.com/YChange01/tqlite).

## What's implemented (turboquant-paper-repro branch)

Paper-faithful Algorithm 1 / Algorithm 2 on both K and V:

- **Random orthogonal rotation Π** via QR decomposition of a Gaussian
  matrix (paper §3.1). Works for any `head_dim`, including non-powers
  of two -- required for Stage 3 outlier channel splitting.
- **Unit-sphere input**: ``x̂ = x / ||x||`` before rotation, matching
  the paper's Lemma 1 support of the Beta coordinate distribution.
- **Beta-trained Lloyd-Max codebook**: centroids in ``[-1, 1]``, trained
  on the exact distribution of a single coordinate of a uniformly
  random unit vector (first coord of normalized Gaussian draws). No
  Gaussian approximation -- b=1 centroids match paper's
  ``sqrt(2/pi)/sqrt(d)`` formula within 1%.
- **Algorithm 1 (`Q_mse`)**: b-bit Lloyd-Max on ``Pi @ x_hat``.
- **Algorithm 2 (`Q_prod`)**: (b-1)-bit Q_mse + 1-bit QJL on the
  residual, giving an unbiased inner-product estimator (Theorem 2 /
  Lemma 4). Applied uniformly to K and V.
- **Triton tensor-core attend kernel**: accumulates two V outputs --
  ``acc_main`` from the codebook gather and ``acc_qjl`` from the 1-bit
  QJL signs -- and resolves the QJL residual post-kernel in Python
  via a single bf16 matmul against ``S`` followed by ``Pi_T``.
- vLLM v1 attention backend (`TurboQuantAttentionBackend`,
  `TurboQuantAttentionImpl`) with per-layer distinct seeds so every
  layer uses an independent rotation and QJL projection.

Scope of this branch: b=4 only, Triton TC kernel only. The CUDA WMMA
kernel has not been updated for the paper-faithful scaling or the
V-QJL accumulator split; ``TURBOQUANT_USE_CUDA=1`` is rejected at
backend load.

## Files

```
vllm/turboquant/
├── __init__.py
├── codebook.py            QuantState: QR(Gaussian) Pi + Beta Lloyd-Max + QJL S
├── store.py               turboquant_store_{kv,v}: unit-norm rotate + Triton store
├── attend_tc.py           Triton tensor-core attend (supports V-QJL split)
├── attend_cuda.py         CUDA WMMA attend (disabled on this branch)
└── csrc/                  CUDA sources (not built on this branch)

vllm/v1/attention/backends/
├── turboquant_attn.py     Backend + Impl + MetadataBuilder
└── registry.py            TURBOQUANT enum entry
```

## Environment variables

| var | default | meaning |
|---|---|---|
| `TURBOQUANT_ALGO` | `prod` | `mse` (Algorithm 1) or `prod` (Algorithm 2) |
| `TURBOQUANT_BITS` | `4`    | total bit budget per coord; `prod` in {2..5}, `mse` in {1..4} on this branch |
| `TURBOQUANT_USE_CUDA` | `0` | rejected on this branch (pending CUDA kernel rewrite) |

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

## Known limitations (paper-repro branch)

- Homogeneous bit width across channels. Paper's 2.5-bit / 3.5-bit
  modes (Table 1) require per-head, per-layer outlier channel
  splitting (Stage 3).
- b in {2, 3, 4, 5} for prod; {1, 2, 3, 4} for mse. main_bits=5..8
  (pack_bits=8) is not tuned on this branch.
- CUDA WMMA kernel is disabled; only the Triton TC path is updated
  for paper-faithful scaling and the V-QJL accumulator split.
- ALiBi, sliding window, and `logits_soft_cap` are not supported.
- `cache_dtype` is fixed to `auto` (matches model dtype).
- QJL projection ``S`` is shared between K and V per layer. Using
  independent ``S_k`` / ``S_v`` is not required by the paper and
  would double S storage for no measured benefit.
