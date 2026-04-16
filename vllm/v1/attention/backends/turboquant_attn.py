# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""TurboQuant attention backend with 4-bit quantized K cache.

This backend stores K cache entries as 4-bit indices into a learned Lloyd-Max
codebook after a fixed Hadamard rotation. V cache is kept in standard fp16 /
bf16 paged layout. A Triton kernel fuses dequantize + attend; a pure-PyTorch
reference lives in ``vllm/turboquant/reference.py`` for correctness checks.

Enable with::

    VLLM_ATTENTION_BACKEND=TURBOQUANT vllm serve ...

Status: MVP / skeleton. Needs validation on B200 (see TURBOQUANT.md).
"""

from dataclasses import dataclass
from typing import ClassVar

import torch

from vllm.config import VllmConfig
from vllm.config.cache import CacheDType
from vllm.logger import init_logger
from vllm.turboquant.codebook import GaussianCodebook
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
from vllm.v1.kv_cache_interface import AttentionSpec

logger = init_logger(__name__)


# Default codebook parameters. Configurable via env var.
import os as _os
TURBOQUANT_BITS = int(_os.environ.get("TURBOQUANT_BITS", "8"))


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

        model_config = vllm_config.model_config
        self.num_heads_q = model_config.get_num_attention_heads(
            vllm_config.parallel_config
        )
        self.num_heads_kv = model_config.get_num_kv_heads(vllm_config.parallel_config)
        self.headdim = model_config.get_head_size()

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
        if block_size is None:
            return True
        return block_size % 16 == 0

    forward_includes_kv_cache_update: bool = False

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
        # NOTE(turboquant): we currently request a fp16/bf16-sized block so K idx
        # bytes fit inside the same allocation. That wastes ~50% compared to a
        # dedicated int8 shape; a follow-up should split K/V allocations.
        return (2, num_blocks, block_size, num_kv_heads, head_size)

    @staticmethod
    def get_kv_cache_stride_order() -> tuple[int, ...]:
        return (0, 1, 2, 3, 4)


class TurboQuantAttentionImpl(AttentionImpl):
    """Per-layer TurboQuant attention.

    Each instance owns its own ``GaussianCodebook`` so that different layers
    use different random rotations. The seed is derived from an instance
    counter to stay deterministic without relying on layer-name introspection.
    """

    # Monotonic counter so each layer gets a distinct rotation seed.
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
            raise NotImplementedError("TurboQuant does not support ALiBi yet.")
        if sliding_window is not None:
            raise NotImplementedError(
                "TurboQuant does not support sliding window yet."
            )
        if logits_soft_cap is not None:
            raise NotImplementedError(
                "TurboQuant does not support logits_soft_cap yet."
            )

        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads
        self.alibi_slopes = alibi_slopes
        self.sliding_window = sliding_window
        self.kv_cache_dtype = kv_cache_dtype
        self.logits_soft_cap = logits_soft_cap
        self.attn_type = attn_type
        self.kv_sharing_target_layer_name = kv_sharing_target_layer_name

        self.num_queries_per_kv = self.num_heads // self.num_kv_heads

        self._layer_seed = TurboQuantAttentionImpl._layer_counter
        TurboQuantAttentionImpl._layer_counter += 1

        self._codebook: GaussianCodebook | None = None
        self._k_norms: torch.Tensor | None = None

    def _ensure_codebook(
        self, dtype: torch.dtype, device: torch.device
    ) -> GaussianCodebook:
        if self._codebook is None:
            self._codebook = GaussianCodebook(
                head_dim=self.head_size,
                bits=TURBOQUANT_BITS,
                seed=self._layer_seed,
                dtype=dtype,
                device=device,
            )
        return self._codebook

    def _get_cache_views(self, kv_cache: torch.Tensor):
        """Split kv_cache into K (uint8 idx) and V (fp16/bf16) views."""
        cache_k_idx = kv_cache[0].view(torch.uint8)[
            ..., : self.head_size
        ].contiguous()
        cache_v = kv_cache[1]
        return cache_k_idx, cache_v

    def _ensure_k_norms(self, kv_cache: torch.Tensor) -> torch.Tensor:
        """Lazily allocate per-key norm buffer: (num_blocks, block_size, num_kv_heads) fp32."""
        if self._k_norms is None:
            num_blocks = kv_cache.shape[1]
            block_size = kv_cache.shape[2]
            self._k_norms = torch.zeros(
                num_blocks, block_size, self.num_kv_heads,
                dtype=torch.float32, device=kv_cache.device,
            )
        return self._k_norms

    def do_kv_cache_update(
        self,
        layer: AttentionLayer,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> None:
        from vllm.turboquant.triton_kernels import turboquant_store_kv

        if self.attn_type in (AttentionType.ENCODER_ONLY, AttentionType.ENCODER):
            return

        num_tokens = key.shape[0]
        k = key.view(num_tokens, self.num_kv_heads, self.head_size)
        v = value.view(num_tokens, self.num_kv_heads, self.head_size)

        codebook = self._ensure_codebook(key.dtype, key.device)
        cache_k_idx, cache_v = self._get_cache_views(kv_cache)
        k_norms = self._ensure_k_norms(kv_cache)

        turboquant_store_kv(
            new_k=k,
            new_v=v,
            cache_k=cache_k_idx,
            cache_v=cache_v,
            cache_k_norm=k_norms,
            slot_mapping=slot_mapping,
            codebook=codebook,
            block_size=kv_cache.shape[2],
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
                "TurboQuant does not support fused output quantization."
            )

        if attn_metadata is None:
            output.zero_()
            return output

        from vllm.turboquant.triton_kernels import turboquant_paged_attention

        num_tokens = query.shape[0]
        q = query.view(num_tokens, self.num_heads, self.head_size)

        codebook = self._ensure_codebook(query.dtype, query.device)
        cache_k_idx, cache_v = self._get_cache_views(kv_cache)
        k_norms = self._ensure_k_norms(kv_cache)

        attn_out = turboquant_paged_attention(
            q=q,
            cache_k=cache_k_idx,
            cache_v=cache_v,
            cache_k_norm=k_norms,
            block_table=attn_metadata.block_table,
            seq_lens=attn_metadata.seq_lens,
            codebook=codebook,
            scale=self.scale,
        )
        output.copy_(attn_out.reshape_as(output))
        return output
