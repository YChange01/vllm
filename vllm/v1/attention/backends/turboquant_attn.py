# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""TurboQuant attention backend (paper arXiv:2504.19874).

Environment variables
---------------------
TURBOQUANT_ALGO : "mse" | "prod"  (default "prod")
    "mse"  -> Algorithm 1: b-bit Lloyd-Max on Hadamard-rotated K.
    "prod" -> Algorithm 2: (b-1)-bit Lloyd-Max + 1-bit QJL residual.
TURBOQUANT_BITS : int  (default 8)
    Total bit budget per coordinate. For ``prod`` the main codebook has
    2^(bits-1) entries (1 bit reserved for QJL).

Enable with::

    VLLM_ATTENTION_BACKEND=TURBOQUANT vllm serve ...
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import ClassVar

import torch

from vllm.config import VllmConfig
from vllm.config.cache import CacheDType
from vllm.logger import init_logger
from vllm.turboquant.codebook import QuantState
from vllm.turboquant.triton_kernels import (
    python_paged_attention,
    turboquant_paged_attention,
    turboquant_store_kv,
)
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionImpl,
    AttentionLayer,
    AttentionMetadataBuilder,
    AttentionType,
    CommonAttentionMetadata,
    MultipleOf,
)
from vllm.v1.attention.backends.utils import set_kv_cache_layout
from vllm.v1.kv_cache_interface import AttentionSpec

# Force NHD so cache offset math matches the paged allocation.
set_kv_cache_layout("NHD")

logger = init_logger(__name__)


TURBOQUANT_ALGO = os.environ.get("TURBOQUANT_ALGO", "prod").lower()
TURBOQUANT_BITS = int(os.environ.get("TURBOQUANT_BITS", "8"))
# Diagnostic bypass: skip quantization entirely and run FP attention over
# the original bf16 K/V stored in a parallel buffer. If output is still
# broken with BYPASS=1, the bug is NOT in the quant kernel but in how the
# backend plugs into vLLM (output tensor, slot semantics, etc).
TURBOQUANT_BYPASS = os.environ.get("TURBOQUANT_BYPASS", "0") == "1"
# Diagnostic: bypass the Triton attend kernel and use a pure PyTorch fp32
# reference that consumes the SAME quantized cache. Localizes whether a
# wrong output comes from the Triton kernel implementation (PYREF correct,
# kernel wrong) or from the algorithm itself (both wrong identically).
TURBOQUANT_PYREF = os.environ.get("TURBOQUANT_PYREF", "0") == "1"
# ALWAYS-ON diagnostic: log the FIRST do_kv_cache_update and FIRST forward()
# payload for every distinct layer, the first time each is seen. No env
# gate -- vllm serve spawns subprocesses and env vars don't propagate
# reliably, so we instrument unconditionally and use a file instead of
# stdout (which gets swallowed by vllm's logger).
import sys as _sys

# Prefer the repo-local logs/ dir when the backend can find it (imported
# from an editable install). Fall back to a few well-known paths so debug
# output is never lost.
_REPO_LOGS_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "logs")
)
_DEBUG_CANDIDATE_PATHS = [
    os.environ.get("TURBOQUANT_DEBUG_LOG", ""),
    os.path.join(_REPO_LOGS_DIR, "turboquant_debug.log"),
    "./logs/turboquant_debug.log",
    "/tmp/turboquant_debug.log",
    "/mnt/nvme3n1/g00872988/turboquant/turboquant_debug.log",
    "./turboquant_debug.log",
]
# Counter of how many times each layer's store/fwd has been logged.
# We log at most _DEBUG_MAX_PER_LAYER calls per layer (so prefill + a
# couple of decodes appear, but not thousands).
_DEBUG_STORE_COUNT: dict[int, int] = {}
_DEBUG_FWD_COUNT: dict[int, int] = {}
_DEBUG_MAX_PER_LAYER = 3


def _dbg_write(msg: str) -> None:
    """Write to the first path that succeeds; also echo to stderr so vllm's
    logger captures it even if no filesystem path is writable."""
    line = msg + "\n"
    _sys.stderr.write("[TURBOQUANT_DBG] " + line)
    try:
        _sys.stderr.flush()
    except Exception:
        pass
    for p in _DEBUG_CANDIDATE_PATHS:
        if not p:
            continue
        try:
            with open(p, "a") as f:
                f.write(line)
            return
        except Exception:
            continue


def _dbg(msg: str) -> None:
    _dbg_write(msg)


# Module-load marker so we can confirm the module was imported and by which pid.
_dbg_write(
    f"# module load pid={os.getpid()} algo={TURBOQUANT_ALGO} "
    f"bits={TURBOQUANT_BITS} bypass={TURBOQUANT_BYPASS} "
    f"pyref={TURBOQUANT_PYREF}"
)

if TURBOQUANT_ALGO not in ("mse", "prod"):
    raise ValueError(
        f"TURBOQUANT_ALGO must be 'mse' or 'prod', got {TURBOQUANT_ALGO!r}"
    )


@dataclass
class TurboQuantAttentionMetadata:
    num_actual_tokens: int
    max_query_len: int
    query_start_loc: torch.Tensor
    max_seq_len: int
    seq_lens: torch.Tensor
    block_table: torch.Tensor
    slot_mapping: torch.Tensor


class TurboQuantAttentionMetadataBuilder(
    AttentionMetadataBuilder[TurboQuantAttentionMetadata]
):
    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.NEVER

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        self.block_size = kv_cache_spec.block_size

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> TurboQuantAttentionMetadata:
        return TurboQuantAttentionMetadata(
            num_actual_tokens=common_attn_metadata.num_actual_tokens,
            max_query_len=common_attn_metadata.max_query_len,
            query_start_loc=common_attn_metadata.query_start_loc,
            max_seq_len=common_attn_metadata.max_seq_len,
            seq_lens=common_attn_metadata.seq_lens,
            block_table=common_attn_metadata.block_table_tensor,
            slot_mapping=common_attn_metadata.slot_mapping,
        )


class TurboQuantAttentionBackend(AttentionBackend):
    supported_dtypes: ClassVar[list[torch.dtype]] = [torch.float16, torch.bfloat16]
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = ["auto"]

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        return [MultipleOf(16)]

    @classmethod
    def supports_block_size(cls, block_size: int | None) -> bool:
        return block_size is None or block_size % 16 == 0

    # IMPORTANT: must be True.
    # We own a SEPARATE `_k_fp`/`_k_idx` buffer (not vllm's kv_cache), so we
    # need `forward` to receive `key`/`value` every call and store them
    # ourselves. With False, vllm uses `unified_kv_cache_update` -> ideal for
    # backends that write into vllm's kv_cache, but NOT called on every
    # decode step when async scheduling is on -- observed: decode step 2 was
    # missing a `do_kv_cache_update` call for slot 19, leaving our buffer
    # zero at that slot, which produced `://24`.
    forward_includes_kv_cache_update: bool = True

    @staticmethod
    def get_name() -> str:
        return "TURBOQUANT"

    @staticmethod
    def get_impl_cls() -> type["TurboQuantAttentionImpl"]:
        return TurboQuantAttentionImpl

    @staticmethod
    def get_metadata_cls() -> type[TurboQuantAttentionMetadata]:
        return TurboQuantAttentionMetadata

    @staticmethod
    def get_builder_cls() -> type[TurboQuantAttentionMetadataBuilder]:
        return TurboQuantAttentionMetadataBuilder

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        # TurboQuant owns its per-layer buffers. vLLM's native 2-slab
        # allocation still happens but stays unused; shrinking this to a
        # 1-slab shape is a follow-up cleanup.
        return (2, num_blocks, block_size, num_kv_heads, head_size)

    @staticmethod
    def get_kv_cache_stride_order() -> tuple[int, ...]:
        return (0, 1, 2, 3, 4)


class TurboQuantAttentionImpl(AttentionImpl):
    """Per-layer TurboQuant attention.

    Each instance owns its own ``QuantState`` so every attention layer has
    a distinct Hadamard rotation and QJL matrix.
    """

    _layer_counter: ClassVar[int] = 0

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int | None = None,
        alibi_slopes: list[float] | None = None,
        sliding_window: int | None = None,
        kv_cache_dtype: str = "auto",
        logits_soft_cap: float | None = None,
        attn_type: str = AttentionType.DECODER,
        kv_sharing_target_layer_name: str | None = None,
    ) -> None:
        if alibi_slopes is not None:
            raise NotImplementedError("TurboQuant: ALiBi unsupported.")
        if sliding_window is not None:
            raise NotImplementedError("TurboQuant: sliding window unsupported.")
        if logits_soft_cap is not None:
            raise NotImplementedError("TurboQuant: logits_soft_cap unsupported.")

        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads
        self.kv_cache_dtype = kv_cache_dtype
        self.attn_type = attn_type
        self.kv_sharing_target_layer_name = kv_sharing_target_layer_name

        self.num_queries_per_kv = self.num_heads // self.num_kv_heads

        self._layer_seed = TurboQuantAttentionImpl._layer_counter
        TurboQuantAttentionImpl._layer_counter += 1

        self._state: QuantState | None = None
        # Paged-cache buffers (lazily allocated in _ensure_buffers).
        #   K quantized path (non-BYPASS): _k_idx + _k_norm
        #   V path (both BYPASS and quant): _v_fp (bf16/fp16, no quant)
        #       Paper only compresses K; int8 V was found to break long-
        #       prefill reconstruction (|attn|.max exceeded max|v|).
        self._k_idx: torch.Tensor | None = None
        self._k_norm: torch.Tensor | None = None
        self._v_fp: torch.Tensor | None = None
        # prod-only buffers
        self._k_qjl_sign: torch.Tensor | None = None
        self._k_rnorm: torch.Tensor | None = None
        # BYPASS mode only.
        self._k_fp: torch.Tensor | None = None

    # -------------------------------------------------------------------
    # Lazy state / buffer allocation
    # -------------------------------------------------------------------
    def _ensure_state(self, dtype: torch.dtype, device: torch.device) -> QuantState:
        if self._state is None:
            self._state = QuantState(
                algo=TURBOQUANT_ALGO,
                bits=TURBOQUANT_BITS,
                head_dim=self.head_size,
                seed=self._layer_seed,
                dtype=dtype,
                device=device,
            )
        return self._state

    def _ensure_buffers(self, kv_cache: torch.Tensor):
        if self._k_idx is not None or self._k_fp is not None:
            return  # already allocated

        num_blocks = kv_cache.shape[1]
        block_size = kv_cache.shape[2]
        device = kv_cache.device

        shape_dim = (num_blocks, block_size, self.num_kv_heads, self.head_size)
        shape_meta = (num_blocks, block_size, self.num_kv_heads)

        # V is always stored raw in input dtype -- used by both BYPASS and
        # quant paths.
        self._v_fp = torch.zeros(shape_dim, dtype=kv_cache.dtype, device=device)

        if TURBOQUANT_BYPASS:
            # Parallel FP bf16 K buffer. Isolates vLLM integration from quant.
            self._k_fp = torch.zeros(shape_dim, dtype=kv_cache.dtype,
                                     device=device)
            return

        self._k_idx = torch.zeros(shape_dim, dtype=torch.uint8, device=device)
        self._k_norm = torch.zeros(shape_meta, dtype=torch.float32, device=device)

        if TURBOQUANT_ALGO == "prod":
            self._k_qjl_sign = torch.zeros(
                shape_dim, dtype=torch.int8, device=device
            )
            self._k_rnorm = torch.zeros(
                shape_meta, dtype=torch.float32, device=device
            )

        # ONE-TIME ptr / overlap dump per layer. Catches the case where
        # PyTorch caching allocator returns aliased / overlapping memory
        # for our independent torch.zeros() calls (which would explain
        # cache_v_fp containing K-derived values).
        if self._layer_seed < 2:
            def _bp(name, t):
                if t is None:
                    return None
                p = t.data_ptr()
                nb = t.numel() * t.element_size()
                return (name, p, nb, p + nb)
            entries = [
                _bp("k_idx",   self._k_idx),
                _bp("k_norm",  self._k_norm),
                _bp("v_fp",    self._v_fp),
                _bp("k_qjl_sign", self._k_qjl_sign),
                _bp("k_rnorm", self._k_rnorm),
            ]
            entries = [e for e in entries if e is not None]
            for n, p, nb, pe in entries:
                _dbg(
                    f"[ptr L{self._layer_seed}] {n} ptr=0x{p:x} "
                    f"nbytes={nb} end=0x{pe:x}"
                )
            # pairwise overlap check
            import itertools
            for (a, b) in itertools.combinations(entries, 2):
                an, ap, anb, ape = a
                bn, bp, bnb, bpe = b
                overlap = max(ap, bp) < min(ape, bpe)
                if overlap:
                    _dbg(f"[ptr L{self._layer_seed}] OVERLAP {an} vs {bn} !!!")

    # -------------------------------------------------------------------
    # vLLM hooks
    # -------------------------------------------------------------------
    def do_kv_cache_update(
        self,
        layer: AttentionLayer,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> None:
        if self.attn_type in (AttentionType.ENCODER_ONLY, AttentionType.ENCODER):
            return

        num_tokens = key.shape[0]
        k = key.view(num_tokens, self.num_kv_heads, self.head_size)
        v = value.view(num_tokens, self.num_kv_heads, self.head_size)

        self._ensure_buffers(kv_cache)

        _cnt = _DEBUG_STORE_COUNT.get(self._layer_seed, 0)
        # Always log real calls (skip the num_tokens=8192 profile warmup).
        if _cnt < _DEBUG_MAX_PER_LAYER and num_tokens != 8192:
            _DEBUG_STORE_COUNT[self._layer_seed] = _cnt + 1
            try:
                sm_full = slot_mapping.detach().cpu()
                sm_head = sm_full[:16].tolist()
                sm_nvalid = int((sm_full >= 0).sum().item())
                k_f = k.detach().float()
                v_f = v.detach().float()
                k_norms = k_f.norm(dim=-1)  # (T, H_kv)
                v_norms = v_f.norm(dim=-1)
                k_abs_max = float(k_f.abs().max().item())
                v_abs_max = float(v_f.abs().max().item())
                k0 = k_f.cpu()[:2, 0, :4].tolist()
                v0 = v_f.cpu()[:2, 0, :4].tolist()
                _dbg(
                    f"[store L{self._layer_seed}] num_tokens={num_tokens} "
                    f"slot_mapping.len={sm_full.numel()} nvalid={sm_nvalid} "
                    f"slot_mapping[:16]={sm_head} bypass={TURBOQUANT_BYPASS} "
                    f"||k||=(min={float(k_norms.min()):.3f},"
                    f"max={float(k_norms.max()):.3f},"
                    f"mean={float(k_norms.mean()):.3f}) "
                    f"||v||=(min={float(v_norms.min()):.3f},"
                    f"max={float(v_norms.max()):.3f},"
                    f"mean={float(v_norms.mean()):.3f}) "
                    f"|k|max={k_abs_max:.3f} |v|max={v_abs_max:.3f} "
                    f"k[:2,0,:4]={k0} v[:2,0,:4]={v0}"
                )
            except Exception as e:
                _dbg(f"[store L{self._layer_seed}] dbg err: {e}")

        block_size = kv_cache.shape[2]

        if TURBOQUANT_BYPASS:
            valid = slot_mapping >= 0
            slots = slot_mapping[valid].to(torch.int64)
            if slots.numel() > 0:
                b_idx = slots // block_size
                off = slots % block_size
                self._k_fp[b_idx, off] = k[valid]
                self._v_fp[b_idx, off] = v[valid]
            return

        state = self._ensure_state(key.dtype, key.device)
        turboquant_store_kv(
            new_k=k,
            new_v=v,
            cache_k_idx=self._k_idx,
            cache_k_norm=self._k_norm,
            cache_v_fp=self._v_fp,
            slot_mapping=slot_mapping,
            state=state,
            block_size=kv_cache.shape[2],
            cache_k_qjl_sign=self._k_qjl_sign,
            cache_k_rnorm=self._k_rnorm,
        )

        # POST-STORE VERIFICATION: read back V from cache_v_fp at the slots
        # we just wrote and confirm it matches the input. If they differ,
        # the Triton store kernel is corrupting V (which is the only way
        # PYREF can produce |attn|.max > max|v|).
        if _cnt < _DEBUG_MAX_PER_LAYER and num_tokens != 8192:
            try:
                valid = slot_mapping >= 0
                slots = slot_mapping[valid].to(torch.int64)
                if slots.numel() > 0:
                    b_idx = slots // block_size
                    off = slots % block_size
                    stored_v = self._v_fp[b_idx, off]              # (n, H_kv, d)
                    stored_v_f = stored_v.detach().float()
                    in_v_f = v[valid].detach().float()
                    diff = (stored_v_f - in_v_f).abs()
                    _dbg(
                        f"[verify L{self._layer_seed}] "
                        f"kv_cache.dtype={kv_cache.dtype} "
                        f"v_fp.dtype={self._v_fp.dtype} "
                        f"k_idx.dtype={self._k_idx.dtype} "
                        f"k_norm.dtype={self._k_norm.dtype} "
                        f"input_v|.|max={float(in_v_f.abs().max()):.4f} "
                        f"stored_v|.|max={float(stored_v_f.abs().max()):.4f} "
                        f"max_abs_diff={float(diff.max()):.6f} "
                        f"mean_abs_diff={float(diff.mean()):.6f} "
                        f"first_input_slot_v[0,0,:4]={in_v_f.cpu()[0,0,:4].tolist()} "
                        f"first_stored_slot_v[0,0,:4]={stored_v_f.cpu()[0,0,:4].tolist()}"
                    )
                    # Also check K cache: read back k_norm at first slot
                    stored_k_idx = self._k_idx[b_idx[0], off[0]]
                    stored_k_norm = self._k_norm[b_idx[0], off[0]]
                    in_k_norm = in_v_f.new_tensor([
                        float(k[valid][0, h].float().norm())
                        for h in range(self.num_kv_heads)
                    ])
                    _dbg(
                        f"[verifyK L{self._layer_seed}] "
                        f"first_slot k_idx.dtype={stored_k_idx.dtype} "
                        f"k_idx min={int(stored_k_idx.min())} max={int(stored_k_idx.max())} "
                        f"stored k_norm={stored_k_norm.cpu().tolist()} "
                        f"input ||k||={in_k_norm.cpu().tolist()}"
                    )
            except Exception as e:
                _dbg(f"[verify L{self._layer_seed}] err: {e}")

    def forward(
        self,
        layer: AttentionLayer,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: TurboQuantAttentionMetadata,
        output: torch.Tensor,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if output_scale is not None or output_block_scale is not None:
            raise NotImplementedError(
                "TurboQuant: fused output quantization unsupported."
            )
        if attn_metadata is None:
            output.zero_()
            return output

        # forward_includes_kv_cache_update is True: the wrapper does NOT
        # invoke unified_kv_cache_update, so we store K/V ourselves here,
        # every forward call (including every decode step). Using our own
        # parallel buffer is what forces this -- see class-level comment.
        if key is not None and value is not None:
            self.do_kv_cache_update(
                layer, key, value, kv_cache, attn_metadata.slot_mapping
            )

        num_tokens = query.shape[0]
        q = query.view(num_tokens, self.num_heads, self.head_size)

        self._ensure_buffers(kv_cache)

        _cnt_f = _DEBUG_FWD_COUNT.get(self._layer_seed, 0)
        if _cnt_f < _DEBUG_MAX_PER_LAYER and num_tokens != 8192:
            _DEBUG_FWD_COUNT[self._layer_seed] = _cnt_f + 1
            try:
                qsl = attn_metadata.query_start_loc.detach().cpu().tolist()
                sl = attn_metadata.seq_lens.detach().cpu().tolist()
                bt_head = (
                    attn_metadata.block_table[: max(1, len(sl)), :4]
                    .detach().cpu().tolist()
                )
                sm_full = attn_metadata.slot_mapping.detach().cpu()
                sm_head = sm_full[:16].tolist()
                sm_nvalid = int((sm_full >= 0).sum().item())
                q_f = q.detach().float()
                q_norms = q_f.norm(dim=-1)  # (T, H_q)
                q_abs_max = float(q_f.abs().max().item())
                q0 = q_f.cpu()[:1, 0, :4].tolist()
                out_shape = tuple(output.shape)
                _dbg(
                    f"[fwd   L{self._layer_seed}] num_tokens={num_tokens} "
                    f"q.shape={tuple(q.shape)} out.shape={out_shape} "
                    f"qsl={qsl} seq_lens={sl} bt[:4]={bt_head} "
                    f"slot_mapping[:16]={sm_head} nvalid={sm_nvalid} "
                    f"||q||=(min={float(q_norms.min()):.3f},"
                    f"max={float(q_norms.max()):.3f},"
                    f"mean={float(q_norms.mean()):.3f}) "
                    f"|q|max={q_abs_max:.3f} bypass={TURBOQUANT_BYPASS} "
                    f"q[0,0,:4]={q0}"
                )
            except Exception as e:
                _dbg(f"[fwd L{self._layer_seed}] dbg err: {e}")

        if TURBOQUANT_BYPASS:
            attn_out = _fp_paged_attention(
                q=q,
                k_fp=self._k_fp,
                v_fp=self._v_fp,
                block_table=attn_metadata.block_table,
                seq_lens=attn_metadata.seq_lens,
                query_start_loc=attn_metadata.query_start_loc,
                scale=self.scale,
                num_heads_q=self.num_heads,
                num_heads_kv=self.num_kv_heads,
            )
        else:
            state = self._ensure_state(query.dtype, query.device)
            attend_fn = (
                python_paged_attention if TURBOQUANT_PYREF
                else turboquant_paged_attention
            )
            attn_out = attend_fn(
                q=q,
                cache_k_idx=self._k_idx,
                cache_k_norm=self._k_norm,
                cache_v_fp=self._v_fp,
                block_table=attn_metadata.block_table,
                seq_lens=attn_metadata.seq_lens,
                query_start_loc=attn_metadata.query_start_loc,
                state=state,
                cache_k_qjl_sign=self._k_qjl_sign,
                cache_k_rnorm=self._k_rnorm,
            )
        output.copy_(attn_out.reshape_as(output))
        if _cnt_f < _DEBUG_MAX_PER_LAYER and num_tokens != 8192:
            try:
                a_f = attn_out.detach().float()
                a0 = a_f.cpu().reshape(-1)[:4].tolist()
                o0 = output.detach().float().cpu().reshape(-1)[:4].tolist()
                a_mean = float(a_f.abs().mean().item())
                a_max = float(a_f.abs().max().item())
                o_mean = float(output.detach().float().abs().mean().item())
                a_nan = bool(a_f.isnan().any().item())
                a_inf = bool(a_f.isinf().any().item())
                _dbg(
                    f"[out   L{self._layer_seed}] call={_cnt_f} "
                    f"attn_out[:4]={a0} output[:4]={o0} "
                    f"|attn|.mean={a_mean:.4f} |attn|.max={a_max:.4f} "
                    f"|out|.mean={o_mean:.4f} "
                    f"nan={a_nan} inf={a_inf}"
                )
            except Exception as e:
                _dbg(f"[out L{self._layer_seed}] dbg err: {e}")
        return output


def _fp_paged_attention(
    q: torch.Tensor,                  # (T_q, H_q, d) bf16/fp16
    k_fp: torch.Tensor,               # (num_blocks, bs, H_kv, d)
    v_fp: torch.Tensor,               # (num_blocks, bs, H_kv, d)
    block_table: torch.Tensor,        # (num_seqs, max_blocks)
    seq_lens: torch.Tensor,           # (num_seqs,)
    query_start_loc: torch.Tensor,    # (num_seqs + 1,)
    scale: float,
    num_heads_q: int,
    num_heads_kv: int,
) -> torch.Tensor:
    """BYPASS-only: vectorized FP attention over paged bf16 K/V.

    Recomputes the same per-query metadata the Triton path uses, then does
    full softmax(q @ k^T * scale) @ v with no quantization. If this produces
    coherent output on Llama, the vLLM backend plumbing is correct and any
    `://24` under the real path must come from the quant kernel.
    """
    T_q, H_q, d = q.shape
    num_blocks, block_size, H_kv, _ = k_fp.shape
    num_seqs = int(seq_lens.shape[0])
    dev = q.device
    gqa = H_q // H_kv

    qsl = query_start_loc.to(device=dev, dtype=torch.int64)
    query_lens = qsl[1:] - qsl[:-1]
    seq_ids = torch.arange(num_seqs, dtype=torch.int64, device=dev)
    seq_id_per_q = torch.repeat_interleave(seq_ids, query_lens)
    q_pos = (
        torch.arange(T_q, dtype=torch.int64, device=dev)
        - qsl[:-1][seq_id_per_q]
    )
    prefix_len = seq_lens.to(device=dev, dtype=torch.int64) - query_lens
    kv_end_per_q = prefix_len[seq_id_per_q] + q_pos + 1

    out = torch.empty_like(q)
    # Simple python loop (this is diagnostic code; perf irrelevant).
    for qi in range(T_q):
        seq = int(seq_id_per_q[qi].item())
        end = int(kv_end_per_q[qi].item())
        # Gather contiguous K, V for seq[0 : end] via block_table.
        num_blocks_q = (end + block_size - 1) // block_size
        k_list = []
        v_list = []
        taken = 0
        for bi in range(num_blocks_q):
            phys = int(block_table[seq, bi].item())
            use = min(block_size, end - taken)
            k_list.append(k_fp[phys, :use])
            v_list.append(v_fp[phys, :use])
            taken += use
        k_seq = torch.cat(k_list, dim=0)  # (end, H_kv, d)
        v_seq = torch.cat(v_list, dim=0)
        for h in range(H_q):
            kh = h // gqa
            scores = (q[qi, h].float() @ k_seq[:, kh].float().T) * scale
            w = torch.softmax(scores, dim=-1)
            out[qi, h] = (w @ v_seq[:, kh].float()).to(q.dtype)
    return out
