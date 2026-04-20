# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for TurboQuantConfig + stages registry + compression report.

CPU-only; no GPU / Triton / vLLM imports. Designed to run on Mac dev
boxes as a fast pre-flight before pushing changes to B200.
"""

from __future__ import annotations

import pytest

from vllm.turboquant.compression import report_for_config
from vllm.turboquant.config import TurboQuantConfig
from vllm.turboquant.stages import (
    DEPRECATED_ALIASES,
    SILENT_ALIASES,
    STAGES,
    UnknownStageError,
    resolve_stage,
)


# ---------------------------------------------------------------------------
# TurboQuantConfig
# ---------------------------------------------------------------------------
class TestConfigFromEnv:
    def test_default(self):
        cfg = TurboQuantConfig.from_env({})
        assert cfg.algo == "prod"
        assert cfg.bits == 4
        assert cfg.is_homog
        assert not cfg.tight_pack
        assert cfg.norm_dtype == "fp32"
        assert cfg.rnorm_dtype == "fp32"

    def test_full(self):
        env = {
            "TURBOQUANT_ALGO": "prod",
            "TURBOQUANT_BITS": "4",
            "TURBOQUANT_OUTLIER_MASK": "/tmp/foo.pt",
            "TURBOQUANT_BITS_OUTLIER": "5",
            "TURBOQUANT_BITS_REGULAR": "3",
            "TURBOQUANT_TIGHT_PACK": "0",
            "TURBOQUANT_FP16_NORMS": "1",
            "TURBOQUANT_UINT8_RNORM": "1",
        }
        cfg = TurboQuantConfig.from_env(env)
        assert cfg.is_split
        assert cfg.bits_outlier == 5
        assert cfg.bits_regular == 3
        assert cfg.norm_dtype == "fp16"
        assert cfg.rnorm_dtype == "uint8"

    def test_uint8_independent_of_norm(self):
        # uint8_rnorm=1 alone keeps norm at fp32
        cfg = TurboQuantConfig.from_env(
            {"TURBOQUANT_UINT8_RNORM": "1"}
        )
        assert cfg.norm_dtype == "fp32"
        assert cfg.rnorm_dtype == "uint8"

    def test_fp16_norms_propagates_to_rnorm(self):
        # Legacy: FP16_NORMS=1 alone makes both fp16 (rnorm tracks norm).
        cfg = TurboQuantConfig.from_env(
            {"TURBOQUANT_FP16_NORMS": "1"}
        )
        assert cfg.norm_dtype == "fp16"
        assert cfg.rnorm_dtype == "fp16"


class TestConfigRoundTrip:
    @pytest.mark.parametrize("name", sorted(STAGES.keys()))
    def test_to_env_round_trip(self, name):
        original = STAGES[name]
        env = original.to_env_dict()
        recovered = TurboQuantConfig.from_env(env)
        assert recovered == original, (
            f"Round-trip mismatch for {name}:\n"
            f"  original:  {original}\n"
            f"  recovered: {recovered}"
        )


class TestConfigValidate:
    def test_valid_default(self):
        TurboQuantConfig.from_env({}).validate()

    def test_invalid_algo(self):
        with pytest.raises(ValueError, match="algo must be"):
            TurboQuantConfig(algo="foo").validate()  # type: ignore[arg-type]

    def test_prod_bits_too_low(self):
        with pytest.raises(ValueError, match="bits >= 2"):
            TurboQuantConfig(algo="prod", bits=1).validate()

    def test_tight_pack_only_b4(self):
        with pytest.raises(ValueError, match="b=4 prod"):
            TurboQuantConfig(algo="prod", bits=5, tight_pack=True).validate()

    def test_tight_pack_only_homog(self):
        with pytest.raises(ValueError, match="not supported in split"):
            TurboQuantConfig(
                algo="prod", bits=4,
                outlier_mask_path="/tmp/x.pt",
                bits_outlier=5, bits_regular=3,
                tight_pack=True,
            ).validate()

    def test_tight_pack_only_prod(self):
        with pytest.raises(ValueError, match="algo='prod'"):
            TurboQuantConfig(algo="mse", bits=4, tight_pack=True).validate()


# ---------------------------------------------------------------------------
# Stage registry
# ---------------------------------------------------------------------------
class TestStageRegistry:
    @pytest.mark.parametrize("name", sorted(STAGES.keys()))
    def test_all_stages_validate(self, name):
        STAGES[name].validate()

    @pytest.mark.parametrize("alias,canonical", sorted(SILENT_ALIASES.items()))
    def test_silent_alias_resolves(self, alias, canonical):
        resolved_name, cfg = resolve_stage(alias)
        assert resolved_name == canonical
        assert cfg is STAGES[canonical]

    @pytest.mark.parametrize(
        "alias,canonical", sorted(DEPRECATED_ALIASES.items())
    )
    def test_deprecated_alias_resolves(self, alias, canonical, capsys):
        resolved_name, _ = resolve_stage(alias)
        assert resolved_name == canonical
        captured = capsys.readouterr()
        assert "DEPRECATED" in captured.err

    def test_unknown_stage_raises(self):
        with pytest.raises(UnknownStageError):
            resolve_stage("TURBOQUANT_does_not_exist")


# ---------------------------------------------------------------------------
# Compression report
# ---------------------------------------------------------------------------
class TestCompressionReport:
    @pytest.mark.parametrize("name", sorted(STAGES.keys()))
    def test_renders(self, name):
        cfg = STAGES[name]
        if cfg.use_cuda:
            pytest.skip("CUDA stages are paper-repro-rejected; report still renders")
        rep = report_for_config(cfg)
        assert rep.kv_total > 0
        assert rep.compression_ratio > 1.0
        rendered = rep.render()
        assert "Compression ratio" in rendered
        assert "Effective bits/coord" in rendered

    def test_b4_baseline(self):
        rep = report_for_config(STAGES["TURBOQUANT_b4"])
        # 4-bit pack idx (64) + 1-bit qjl (16) + fp32 norm (4) + fp32 rnorm (4) = 88 K
        assert rep.k_total == 88
        assert rep.kv_total == 176
        assert abs(rep.compression_ratio - 512 / 176) < 1e-6

    def test_b4_tight(self):
        rep = report_for_config(STAGES["TURBOQUANT_b4_t"])
        # tight: 4-bit nibble (idx+qjl, 64) + 4 + 4 = 72 K
        assert rep.k_total == 72
        assert rep.kv_total == 144

    def test_split_3_5bit_baseline(self):
        rep = report_for_config(STAGES["TURBOQUANT_split_3_5bit"])
        # 32@b=5: idx 16 + qjl 4 + norm 4 + rnorm 4 = 28
        # 96@b=3: idx 24 + qjl 12 + norm 4 + rnorm 4 = 44
        # K total = 72; K+V = 144; compression = 512/144 ≈ 3.56
        assert rep.k_total == 72
        assert rep.kv_total == 144
        assert abs(rep.compression_ratio - 512 / 144) < 1e-6

    def test_split_3_5bit_fu_matches_4_13x(self):
        rep = report_for_config(STAGES["TURBOQUANT_split_3_5bit_fu"])
        # fp16 norm + uint8 rnorm: K-side metadata 16 -> 6
        # K data unchanged 56; K total = 62; K+V = 124
        assert rep.k_total == 62
        assert rep.kv_total == 124
        # 512/124 = 4.129... -- this is the corrected number after the
        # earlier 4.4x manual-arithmetic mistake.
        assert abs(rep.compression_ratio - 512 / 124) < 1e-6
        assert 4.10 < rep.compression_ratio < 4.20

    def test_data_bits_match_paper_density(self):
        # split_3_5bit nominal density is 3.5 bit/coord (data only).
        rep = report_for_config(STAGES["TURBOQUANT_split_3_5bit"])
        assert abs(rep.data_bits_per_coord - 3.5) < 1e-6
        # split_2_25bit (paper §4.3 literal) is 2.25.
        rep2 = report_for_config(STAGES["TURBOQUANT_split_2_25bit"])
        assert abs(rep2.data_bits_per_coord - 2.25) < 1e-6
