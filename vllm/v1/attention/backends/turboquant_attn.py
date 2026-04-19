# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""TurboQuant attention backend (paper arXiv:2504.19874, paper-faithful).

Quantizes both K and V with the same Q_mse or Q_prod algorithm. The
paper applies TurboQuant uniformly to the KV cache (Section 4.2 / 4.3)
and does not separate K and V treatment, so each layer owns one
``QuantState`` (shared Pi, codebook, and QJL projection S for both K
and V).

Environment variables
---------------------
``TURBOQUANT_ALGO`` : ``"mse" | "prod"``  (default ``"prod"``)
    ``mse`` -> Algorithm 1 (b-bit Lloyd-Max).
    ``prod`` -> Algorithm 2 (b-1 bits Lloyd-Max + 1-bit QJL residual).
``TURBOQUANT_BITS`` : int  (default 4; ``prod`` requires >=2)

Storage layout per layer (allocated lazily on first forward):

    _k_idx       : (num_blocks, bs, H_kv, idx_d)  uint8  K Lloyd-Max bucket
                   idx_d = d // 2 when K_CB <= 16 (4-bit nibble pack)
    _k_norm      : (num_blocks, bs, H_kv)         fp32   ||k||
    _v_idx       : (num_blocks, bs, H_kv, idx_d)  uint8  V Lloyd-Max bucket
    _v_norm      : (num_blocks, bs, H_kv)         fp32   ||v||
    _k_qjl_sign  : (num_blocks, bs, H_kv, d/8)    uint8  K QJL 1-bit sign  (prod)
    _k_rnorm     : (num_blocks, bs, H_kv)         fp32   ||r_k||           (prod)
    _v_qjl_sign  : (num_blocks, bs, H_kv, d/8)    uint8  V QJL 1-bit sign  (prod)
    _v_rnorm     : (num_blocks, bs, H_kv)         fp32   ||r_v||           (prod)

vLLM integration
----------------
``forward_includes_kv_cache_update = True`` because we own private K/V
buffers (not vLLM's native ``kv_cache``). The async scheduler's fast
path writes to ``kv_cache`` but skips ``unified_kv_cache_update`` on
some decode steps; with the flag set, ``forward()`` always receives
``key`` / ``value`` and we store them ourselves -- no missed writes.

Attend kernel selection
-----------------------
This branch ships the Triton tensor-core attend only. The CUDA WMMA
path (``attend_cuda.cu``) does not yet implement the V-QJL accumulator
split needed for paper-faithful Q_prod on V, so ``TURBOQUANT_USE_CUDA=1``
is rejected when ``TURBOQUANT_ALGO=prod``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import ClassVar

import torch

from vllm.config import VllmConfig
from vllm.config.cache import CacheDType
from vllm.logger import init_logger
from vllm.turboquant.attend_split_tc import (
    turboquant_paged_attention_split_tc,
)
from vllm.turboquant.attend_tc import turboquant_paged_attention_tc
from vllm.turboquant.codebook import QuantState
from vllm.turboquant.outlier import OutlierMask, SplitQuantState
from vllm.turboquant.store import (
    turboquant_store_kv,
    turboquant_store_split,
    turboquant_store_v,
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

# NHD so cache offset math matches the paged allocation.
set_kv_cache_layout("NHD")

logger = init_logger(__name__)

TURBOQUANT_ALGO = os.environ.get("TURBOQUANT_ALGO", "prod").lower()
TURBOQUANT_BITS = int(os.environ.get("TURBOQUANT_BITS", "4"))
TURBOQUANT_USE_CUDA = os.environ.get("TURBOQUANT_USE_CUDA", "0") == "1"
# Outlier channel splitting (paper §4.3). When set, the mask file is
# loaded once at module init and the backend switches from homogeneous
# quantization to two-slice quantization. BITS_OUTLIER / BITS_REGULAR
# control the per-slice bit budgets (e.g., 3+2 for 2.5-bit effective on
# Llama's 128-dim heads with 32 outliers).
TURBOQUANT_OUTLIER_MASK = os.environ.get("TURBOQUANT_OUTLIER_MASK", "")
TURBOQUANT_BITS_OUTLIER = int(
    os.environ.get("TURBOQUANT_BITS_OUTLIER", str(TURBOQUANT_BITS))
)
TURBOQUANT_BITS_REGULAR = int(
    os.environ.get("TURBOQUANT_BITS_REGULAR", str(TURBOQUANT_BITS))
)

if TURBOQUANT_ALGO not in ("mse", "prod"):
    raise ValueError(
        f"TURBOQUANT_ALGO must be 'mse' or 'prod', got {TURBOQUANT_ALGO!r}"
    )
_MAIN_BITS = (
    TURBOQUANT_BITS - 1 if TURBOQUANT_ALGO == "prod" else TURBOQUANT_BITS
)
# Triton kernel supports PACK_BITS in {1, 2, 4, 8}. main_bits in {1..4}
# keep pack_bits<=4 on this branch -- main_bits>=5 would need byte-wide
# storage (pack_bits=8) plus corresponding autotune re-tuning.
if _MAIN_BITS < 1 or _MAIN_BITS > 4:
    raise ValueError(
        f"turboquant-paper-repro branch supports bits in {{1..5}} "
        f"(main_bits {{1..4}}); got algo={TURBOQUANT_ALGO} "
        f"bits={TURBOQUANT_BITS} -> main_bits={_MAIN_BITS}."
    )
if TURBOQUANT_USE_CUDA:
    raise ValueError(
        "TURBOQUANT_USE_CUDA=1 is not supported on the paper-repro "
        "branch. The CUDA WMMA kernel has not been updated for the "
        "paper-faithful rotation (QR Gaussian Pi, unit-norm input, "
        "1/sqrt(d) scale) or the V-QJL accumulator split. Use the "
        "Triton TC kernel (unset the env var)."
    )

_OUTLIER_MASK: OutlierMask | None = None
if TURBOQUANT_OUTLIER_MASK:
    _OUTLIER_MASK = OutlierMask.load(TURBOQUANT_OUTLIER_MASK)
    for b, name in [
        (TURBOQUANT_BITS_OUTLIER, "TURBOQUANT_BITS_OUTLIER"),
        (TURBOQUANT_BITS_REGULAR, "TURBOQUANT_BITS_REGULAR"),
    ]:
        main_b = b - 1 if TURBOQUANT_ALGO == "prod" else b
        if main_b < 1 or main_b > 4:
            raise ValueError(
                f"{name}={b} -> main_bits={main_b} outside supported "
                f"range {{1..4}} on this branch."
            )
    logger.info(
        "TurboQuant split-mode: mask=%s, bits_outlier=%d, bits_regular=%d, "
        "d_out=%d, d_reg=%d",
        TURBOQUANT_OUTLIER_MASK,
        TURBOQUANT_BITS_OUTLIER,
        TURBOQUANT_BITS_REGULAR,
        _OUTLIER_MASK.num_outliers,
        _OUTLIER_MASK.head_dim - _OUTLIER_MASK.num_outliers,
    )

logger.info(
    "TurboQuant backend (paper-repro): algo=%s bits=%d split=%s use_cuda=%s",
    TURBOQUANT_ALGO, TURBOQUANT_BITS,
    bool(_OUTLIER_MASK), TURBOQUANT_USE_CUDA,
)


# CUDA extension is imported lazily only if the user explicitly opts in.
_cuda_attend_fn = None


def _get_cuda_attend():
    global _cuda_attend_fn
    if _cuda_attend_fn is None:
        from vllm.turboquant.attend_cuda import turboquant_paged_attention_cuda
        _cuda_attend_fn = turboquant_paged_attention_cuda
    return _cuda_attend_fn


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

    # Critical: this must be True. See module docstring for why.
    forward_includes_kv_cache_update: bool = True

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        return [MultipleOf(16)]

    @classmethod
    def supports_block_size(cls, block_size: int | None) -> bool:
        return block_size is None or block_size % 16 == 0

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
        # vLLM allocates this shape; we ignore the contents and use it
        # only to learn num_blocks and block_size for our own buffers.
        return (2, num_blocks, block_size, num_kv_heads, head_size)

    @staticmethod
    def get_kv_cache_stride_order() -> tuple[int, ...]:
        return (0, 1, 2, 3, 4)


class TurboQuantAttentionImpl(AttentionImpl):
    """Per-attention-layer TurboQuant impl.

    Two modes:

    * Homogeneous (``_OUTLIER_MASK is None``): one ``QuantState`` per
      layer, four buffers for K (idx, norm, qjl_sign, rnorm) mirrored
      for V in prod mode.
    * Split (``_OUTLIER_MASK is not None``): two ``SplitQuantState``
      instances per layer (one for K, one for V -- same outlier
      channel indices but independent Pi/S/codebooks), eight buffers
      each for K and V.
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
        self._split = _OUTLIER_MASK is not None
        if self._split:
            if _OUTLIER_MASK.head_dim != head_size:
                raise ValueError(
                    f"Outlier mask head_dim {_OUTLIER_MASK.head_dim} != "
                    f"layer head_size {head_size}. Re-run calibration "
                    f"against the current model."
                )

        # Homogeneous state/buffers
        self._state: QuantState | None = None
        self._k_idx: torch.Tensor | None = None
        self._k_norm: torch.Tensor | None = None
        self._v_idx: torch.Tensor | None = None
        self._v_norm: torch.Tensor | None = None
        self._k_qjl_sign: torch.Tensor | None = None
        self._k_rnorm: torch.Tensor | None = None
        self._v_qjl_sign: torch.Tensor | None = None
        self._v_rnorm: torch.Tensor | None = None

        # Split state/buffers
        self._k_split: SplitQuantState | None = None
        self._v_split: SplitQuantState | None = None
        # K: outlier + regular each with (idx, norm, qjl_sign, rnorm)
        self._k_idx_out: torch.Tensor | None = None
        self._k_norm_out: torch.Tensor | None = None
        self._k_qjl_sign_out: torch.Tensor | None = None
        self._k_rnorm_out: torch.Tensor | None = None
        self._k_idx_reg: torch.Tensor | None = None
        self._k_norm_reg: torch.Tensor | None = None
        self._k_qjl_sign_reg: torch.Tensor | None = None
        self._k_rnorm_reg: torch.Tensor | None = None
        # V same shape as K
        self._v_idx_out: torch.Tensor | None = None
        self._v_norm_out: torch.Tensor | None = None
        self._v_qjl_sign_out: torch.Tensor | None = None
        self._v_rnorm_out: torch.Tensor | None = None
        self._v_idx_reg: torch.Tensor | None = None
        self._v_norm_reg: torch.Tensor | None = None
        self._v_qjl_sign_reg: torch.Tensor | None = None
        self._v_rnorm_reg: torch.Tensor | None = None

    # ------------------------------------------------------------------
    # Lazy state / buffer allocation
    # ------------------------------------------------------------------
    def _ensure_state(self, dtype: torch.dtype, device: torch.device):
        if self._split:
            if self._k_split is None:
                # K and V share the outlier channel positions (same
                # calibration used for both sides within a layer) but
                # each side has an independent QuantState so its Pi/S
                # are distinct.
                assert _OUTLIER_MASK is not None
                k_outlier_idx = _OUTLIER_MASK.for_layer(
                    self._layer_seed % _OUTLIER_MASK.num_layers, "k"
                ).to(device)
                v_outlier_idx = _OUTLIER_MASK.for_layer(
                    self._layer_seed % _OUTLIER_MASK.num_layers, "v"
                ).to(device)
                # For the current split kernel, K and V must use the
                # same outlier index set (see assertion in attend
                # wrapper). Union of K and V top-N is the safe choice
                # when they disagree, but that increases d_out; instead
                # we default to the K mask for both sides, as the K
                # side drives the kernel's layout. V calibration is
                # still useful for future per-side splits.
                v_outlier_idx = k_outlier_idx
                self._k_split = SplitQuantState(
                    algo=TURBOQUANT_ALGO,
                    bits_outlier=TURBOQUANT_BITS_OUTLIER,
                    bits_regular=TURBOQUANT_BITS_REGULAR,
                    head_dim=self.head_size,
                    outlier_idx=k_outlier_idx,
                    seed=self._layer_seed,
                    dtype=dtype,
                    device=device,
                )
                self._v_split = SplitQuantState(
                    algo=TURBOQUANT_ALGO,
                    bits_outlier=TURBOQUANT_BITS_OUTLIER,
                    bits_regular=TURBOQUANT_BITS_REGULAR,
                    head_dim=self.head_size,
                    outlier_idx=v_outlier_idx,
                    seed=self._layer_seed ^ 0x7FFFFFFF,
                    dtype=dtype,
                    device=device,
                )
            return self._k_split  # not actually used by callers
        else:
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

    def _ensure_buffers(self, kv_cache: torch.Tensor) -> None:
        if self._split:
            if self._k_idx_out is not None:
                return
            self._allocate_split_buffers(kv_cache)
        else:
            if self._k_idx is not None:
                return
            self._allocate_homogeneous_buffers(kv_cache)

    def _allocate_homogeneous_buffers(self, kv_cache: torch.Tensor) -> None:
        num_blocks = kv_cache.shape[1]
        block_size = kv_cache.shape[2]
        device = kv_cache.device

        main_bits = _MAIN_BITS
        pack_bits = _pow2_ceil(main_bits)
        idx_last_dim = self.head_size * pack_bits // 8

        shape_idx = (num_blocks, block_size, self.num_kv_heads, idx_last_dim)
        shape_meta = (num_blocks, block_size, self.num_kv_heads)

        self._k_idx = torch.zeros(shape_idx, dtype=torch.uint8, device=device)
        self._k_norm = torch.zeros(shape_meta, dtype=torch.float32, device=device)
        self._v_idx = torch.zeros(shape_idx, dtype=torch.uint8, device=device)
        self._v_norm = torch.zeros(shape_meta, dtype=torch.float32, device=device)

        if TURBOQUANT_ALGO == "prod":
            assert self.head_size % 8 == 0
            shape_qjl = (num_blocks, block_size, self.num_kv_heads,
                         self.head_size // 8)
            self._k_qjl_sign = torch.zeros(
                shape_qjl, dtype=torch.uint8, device=device
            )
            self._k_rnorm = torch.zeros(
                shape_meta, dtype=torch.float32, device=device
            )
            self._v_qjl_sign = torch.zeros(
                shape_qjl, dtype=torch.uint8, device=device
            )
            self._v_rnorm = torch.zeros(
                shape_meta, dtype=torch.float32, device=device
            )

    def _allocate_split_buffers(self, kv_cache: torch.Tensor) -> None:
        num_blocks = kv_cache.shape[1]
        block_size = kv_cache.shape[2]
        device = kv_cache.device

        assert _OUTLIER_MASK is not None
        d_out = _OUTLIER_MASK.num_outliers
        d_reg = self.head_size - d_out

        main_out = TURBOQUANT_BITS_OUTLIER - 1 if TURBOQUANT_ALGO == "prod" \
            else TURBOQUANT_BITS_OUTLIER
        main_reg = TURBOQUANT_BITS_REGULAR - 1 if TURBOQUANT_ALGO == "prod" \
            else TURBOQUANT_BITS_REGULAR
        pack_out = _pow2_ceil(main_out)
        pack_reg = _pow2_ceil(main_reg)

        if (d_out * pack_out) % 8 != 0:
            raise ValueError(
                f"d_out={d_out} * pack_bits={pack_out} must be byte-aligned"
            )
        if (d_reg * pack_reg) % 8 != 0:
            raise ValueError(
                f"d_reg={d_reg} * pack_bits={pack_reg} must be byte-aligned"
            )
        if TURBOQUANT_ALGO == "prod":
            if d_out % 8 != 0:
                raise ValueError(
                    f"d_out={d_out} must be divisible by 8 for QJL bit-packing"
                )
            if d_reg % 8 != 0:
                raise ValueError(
                    f"d_reg={d_reg} must be divisible by 8 for QJL bit-packing"
                )

        idx_last_out = d_out * pack_out // 8
        idx_last_reg = d_reg * pack_reg // 8
        shape_idx_out = (num_blocks, block_size, self.num_kv_heads, idx_last_out)
        shape_idx_reg = (num_blocks, block_size, self.num_kv_heads, idx_last_reg)
        shape_meta = (num_blocks, block_size, self.num_kv_heads)

        def _uzeros(shape):
            return torch.zeros(shape, dtype=torch.uint8, device=device)

        def _fzeros(shape):
            return torch.zeros(shape, dtype=torch.float32, device=device)

        self._k_idx_out = _uzeros(shape_idx_out)
        self._k_norm_out = _fzeros(shape_meta)
        self._k_idx_reg = _uzeros(shape_idx_reg)
        self._k_norm_reg = _fzeros(shape_meta)
        self._v_idx_out = _uzeros(shape_idx_out)
        self._v_norm_out = _fzeros(shape_meta)
        self._v_idx_reg = _uzeros(shape_idx_reg)
        self._v_norm_reg = _fzeros(shape_meta)

        if TURBOQUANT_ALGO == "prod":
            shape_qjl_out = (num_blocks, block_size, self.num_kv_heads,
                             d_out // 8)
            shape_qjl_reg = (num_blocks, block_size, self.num_kv_heads,
                             d_reg // 8)
            self._k_qjl_sign_out = _uzeros(shape_qjl_out)
            self._k_rnorm_out = _fzeros(shape_meta)
            self._k_qjl_sign_reg = _uzeros(shape_qjl_reg)
            self._k_rnorm_reg = _fzeros(shape_meta)
            self._v_qjl_sign_out = _uzeros(shape_qjl_out)
            self._v_rnorm_out = _fzeros(shape_meta)
            self._v_qjl_sign_reg = _uzeros(shape_qjl_reg)
            self._v_rnorm_reg = _fzeros(shape_meta)

    # ------------------------------------------------------------------
    # vLLM hooks
    # ------------------------------------------------------------------
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
        self._ensure_state(key.dtype, key.device)
        block_size = kv_cache.shape[2]

        if self._split:
            turboquant_store_split(
                new_x=k, state_split=self._k_split,
                cache_idx_out=self._k_idx_out,
                cache_norm_out=self._k_norm_out,
                cache_qjl_sign_out=self._k_qjl_sign_out,
                cache_rnorm_out=self._k_rnorm_out,
                cache_idx_reg=self._k_idx_reg,
                cache_norm_reg=self._k_norm_reg,
                cache_qjl_sign_reg=self._k_qjl_sign_reg,
                cache_rnorm_reg=self._k_rnorm_reg,
                slot_mapping=slot_mapping,
                block_size=block_size,
            )
            turboquant_store_split(
                new_x=v, state_split=self._v_split,
                cache_idx_out=self._v_idx_out,
                cache_norm_out=self._v_norm_out,
                cache_qjl_sign_out=self._v_qjl_sign_out,
                cache_rnorm_out=self._v_rnorm_out,
                cache_idx_reg=self._v_idx_reg,
                cache_norm_reg=self._v_norm_reg,
                cache_qjl_sign_reg=self._v_qjl_sign_reg,
                cache_rnorm_reg=self._v_rnorm_reg,
                slot_mapping=slot_mapping,
                block_size=block_size,
            )
        else:
            turboquant_store_kv(
                new_k=k,
                cache_k_idx=self._k_idx,
                cache_k_norm=self._k_norm,
                slot_mapping=slot_mapping,
                state=self._state,
                block_size=block_size,
                cache_k_qjl_sign=self._k_qjl_sign,
                cache_k_rnorm=self._k_rnorm,
            )
            turboquant_store_v(
                new_v=v,
                cache_v_idx=self._v_idx,
                cache_v_norm=self._v_norm,
                slot_mapping=slot_mapping,
                state=self._state,
                block_size=block_size,
                cache_v_qjl_sign=self._v_qjl_sign,
                cache_v_rnorm=self._v_rnorm,
            )

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

        if key is not None and value is not None:
            self.do_kv_cache_update(
                layer, key, value, kv_cache, attn_metadata.slot_mapping
            )

        num_tokens = query.shape[0]
        q = query.view(num_tokens, self.num_heads, self.head_size)
        self._ensure_buffers(kv_cache)
        self._ensure_state(query.dtype, query.device)

        if self._split:
            attn_out = turboquant_paged_attention_split_tc(
                q=q,
                cache_k_idx_out=self._k_idx_out,
                cache_k_norm_out=self._k_norm_out,
                cache_v_idx_out=self._v_idx_out,
                cache_v_norm_out=self._v_norm_out,
                cache_k_qjl_sign_out=self._k_qjl_sign_out,
                cache_k_rnorm_out=self._k_rnorm_out,
                cache_v_qjl_sign_out=self._v_qjl_sign_out,
                cache_v_rnorm_out=self._v_rnorm_out,
                cache_k_idx_reg=self._k_idx_reg,
                cache_k_norm_reg=self._k_norm_reg,
                cache_v_idx_reg=self._v_idx_reg,
                cache_v_norm_reg=self._v_norm_reg,
                cache_k_qjl_sign_reg=self._k_qjl_sign_reg,
                cache_k_rnorm_reg=self._k_rnorm_reg,
                cache_v_qjl_sign_reg=self._v_qjl_sign_reg,
                cache_v_rnorm_reg=self._v_rnorm_reg,
                block_table=attn_metadata.block_table,
                seq_lens=attn_metadata.seq_lens,
                query_start_loc=attn_metadata.query_start_loc,
                state_k=self._k_split,
                state_v=self._v_split,
            )
        else:
            attn_out = turboquant_paged_attention_tc(
                q=q,
                cache_k_idx=self._k_idx,
                cache_k_norm=self._k_norm,
                cache_v_idx=self._v_idx,
                cache_v_norm=self._v_norm,
                block_table=attn_metadata.block_table,
                seq_lens=attn_metadata.seq_lens,
                query_start_loc=attn_metadata.query_start_loc,
                state=self._state,
                cache_k_qjl_sign=self._k_qjl_sign,
                cache_k_rnorm=self._k_rnorm,
                cache_v_qjl_sign=self._v_qjl_sign,
                cache_v_rnorm=self._v_rnorm,
            )
        output.copy_(attn_out.reshape_as(output))
        return output


def _pow2_ceil(n: int) -> int:
    """Smallest power of two >= max(1, n)."""
    if n <= 1:
        return 1
    return 1 << (n - 1).bit_length()
