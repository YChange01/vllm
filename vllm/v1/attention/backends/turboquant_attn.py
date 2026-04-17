# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""TurboQuant attention backend (paper arXiv:2504.19874).

Quantizes K (Lloyd-Max + Hadamard, optionally with 1-bit QJL on the
residual) and stores V raw. Each attention layer owns its own
``QuantState`` so every layer has a distinct random rotation.

Environment variables
---------------------
``TURBOQUANT_ALGO`` : ``"mse" | "prod"``  (default ``"prod"``)
    ``mse`` -> Algorithm 1 (b-bit Lloyd-Max).
    ``prod`` -> Algorithm 2 (b-1 bits Lloyd-Max + 1-bit QJL residual).
``TURBOQUANT_BITS`` : int  (default 8; ``prod`` requires >=2)

Storage layout per layer (allocated lazily on first forward):

    _k_idx       : (num_blocks, bs, H_kv, idx_d)  uint8  K Lloyd-Max bucket
                   idx_d = d // 2 when K_CB <= 16 (4-bit nibble pack), else d
    _k_norm      : (num_blocks, bs, H_kv)         fp32   ||k||
    _v_idx       : (num_blocks, bs, H_kv, idx_d)  uint8  V Lloyd-Max bucket
    _v_norm      : (num_blocks, bs, H_kv)         fp32   ||v||
    _k_qjl_sign  : (num_blocks, bs, H_kv, d/8)    uint8  K QJL 1-bit sign
                                                         (prod only, 8 signs/byte)
    _k_rnorm     : (num_blocks, bs, H_kv)         fp32   K residual ||r||
                                                         (prod only)

V is quantized via Q_mse only (no QJL on V). K and V share the same
``QuantState`` (H, signs, codebook). V reconstruction in the attend
kernel is done in rotated/sqrt(d)-normalized space; the inverse
Hadamard + signs + 1/sqrt(d) is applied once per (query, head) in the
attend wrapper, not per attended KV slot.

vLLM integration
----------------
``forward_includes_kv_cache_update = True`` because we own private K/V
buffers (not vLLM's native ``kv_cache``). The async scheduler's fast
path writes to ``kv_cache`` but skips ``unified_kv_cache_update`` on
some decode steps; with the flag set, ``forward()`` always receives
``key`` / ``value`` and we store them ourselves -- no missed writes.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import ClassVar

import torch

from vllm.config import VllmConfig
from vllm.config.cache import CacheDType
from vllm.logger import init_logger
from vllm.turboquant.attend import turboquant_paged_attention
from vllm.turboquant.codebook import QuantState
from vllm.turboquant.store import turboquant_store_kv, turboquant_store_v
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

if TURBOQUANT_ALGO not in ("mse", "prod"):
    raise ValueError(
        f"TURBOQUANT_ALGO must be 'mse' or 'prod', got {TURBOQUANT_ALGO!r}"
    )

logger.info(
    "TurboQuant backend: algo=%s bits=%d", TURBOQUANT_ALGO, TURBOQUANT_BITS,
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

    Each instance owns a ``QuantState`` (so every layer's H, signs, S
    are independent) and lazily-allocated paged K/V buffers sized to
    match vLLM's ``kv_cache`` shape.
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

        # Class-level counter -- gives every layer a distinct seed so
        # H/signs/S differ across layers. Profile-pass instantiation may
        # bump the counter beyond `num_layers`; that's harmless because
        # store and attend always use the same instance's QuantState.
        self._layer_seed = TurboQuantAttentionImpl._layer_counter
        TurboQuantAttentionImpl._layer_counter += 1

        self._state: QuantState | None = None
        # Lazily allocated in _ensure_buffers.
        self._k_idx: torch.Tensor | None = None
        self._k_norm: torch.Tensor | None = None
        self._v_idx: torch.Tensor | None = None
        self._v_norm: torch.Tensor | None = None
        self._k_qjl_sign: torch.Tensor | None = None  # prod only
        self._k_rnorm: torch.Tensor | None = None     # prod only

    # ------------------------------------------------------------------
    # Lazy state / buffer allocation
    # ------------------------------------------------------------------
    def _ensure_state(
        self, dtype: torch.dtype, device: torch.device
    ) -> QuantState:
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
        if self._k_idx is not None:
            return
        num_blocks = kv_cache.shape[1]
        block_size = kv_cache.shape[2]
        device = kv_cache.device

        # Pack two 4-bit indices per byte when the codebook fits in 4
        # bits. ``QuantState.codebook`` size is 2^bits for mse and
        # 2^(bits-1) for prod.
        K_CB = 1 << (TURBOQUANT_BITS - 1 if TURBOQUANT_ALGO == "prod"
                     else TURBOQUANT_BITS)
        idx_last_dim = (
            self.head_size // 2 if K_CB <= 16 else self.head_size
        )

        shape_idx = (num_blocks, block_size, self.num_kv_heads, idx_last_dim)
        shape_meta = (num_blocks, block_size, self.num_kv_heads)

        self._k_idx = torch.zeros(shape_idx, dtype=torch.uint8, device=device)
        self._k_norm = torch.zeros(shape_meta, dtype=torch.float32, device=device)
        self._v_idx = torch.zeros(shape_idx, dtype=torch.uint8, device=device)
        self._v_norm = torch.zeros(shape_meta, dtype=torch.float32, device=device)

        if TURBOQUANT_ALGO == "prod":
            # QJL sign is bit-packed: 8 +/-1 signs per byte, bit_j = (sign_j < 0).
            assert self.head_size % 8 == 0, (
                f"head_size {self.head_size} must be divisible by 8 for "
                f"QJL bit-packing"
            )
            shape_qjl = (num_blocks, block_size, self.num_kv_heads,
                         self.head_size // 8)
            self._k_qjl_sign = torch.zeros(
                shape_qjl, dtype=torch.uint8, device=device
            )
            self._k_rnorm = torch.zeros(
                shape_meta, dtype=torch.float32, device=device
            )

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
        block_size = kv_cache.shape[2]
        state = self._ensure_state(key.dtype, key.device)

        # K quant: Lloyd-Max + Hadamard (+ QJL residual if prod).
        turboquant_store_kv(
            new_k=k,
            cache_k_idx=self._k_idx,
            cache_k_norm=self._k_norm,
            slot_mapping=slot_mapping,
            state=state,
            block_size=block_size,
            cache_k_qjl_sign=self._k_qjl_sign,
            cache_k_rnorm=self._k_rnorm,
        )
        # V quant: Q_mse only (no QJL).
        turboquant_store_v(
            new_v=v,
            cache_v_idx=self._v_idx,
            cache_v_norm=self._v_norm,
            slot_mapping=slot_mapping,
            state=state,
            block_size=block_size,
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

        # forward_includes_kv_cache_update is True, so the wrapper does
        # NOT call unified_kv_cache_update. We must store K/V here on
        # every forward (including every decode step).
        if key is not None and value is not None:
            self.do_kv_cache_update(
                layer, key, value, kv_cache, attn_metadata.slot_mapping
            )

        num_tokens = query.shape[0]
        q = query.view(num_tokens, self.num_heads, self.head_size)
        self._ensure_buffers(kv_cache)
        state = self._ensure_state(query.dtype, query.device)

        attn_out = turboquant_paged_attention(
            q=q,
            cache_k_idx=self._k_idx,
            cache_k_norm=self._k_norm,
            cache_v_idx=self._v_idx,
            cache_v_norm=self._v_norm,
            block_table=attn_metadata.block_table,
            seq_lens=attn_metadata.seq_lens,
            query_start_loc=attn_metadata.query_start_loc,
            state=state,
            cache_k_qjl_sign=self._k_qjl_sign,
            cache_k_rnorm=self._k_rnorm,
        )
        output.copy_(attn_out.reshape_as(output))
        return output
