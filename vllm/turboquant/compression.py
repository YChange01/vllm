# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Per-config storage breakdown and compression report.

Given a ``TurboQuantConfig`` and a head_dim, compute the exact
per-(token, kv_head) byte cost of every cache buffer the backend will
allocate, the K+V total, and the compression ratio vs bf16. Used both
as a library (eval scripts) and a CLI (sanity checking before running).

CLI:
    python3 -m vllm.turboquant.compression TURBOQUANT_split_3_5bit_fu
    python3 -m vllm.turboquant.compression --all   # report all stages

Single source of truth for compression numbers -- avoids the kind of
arithmetic mistake that produced an erroneous "4.4x" estimate during
manual review.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

from vllm.turboquant.config import TurboQuantConfig

# Default outlier count assumed for split mode reports. Real value
# comes from the calibration .pt file at runtime; this is just for
# the CLI breakdown.
_DEFAULT_NUM_OUTLIERS = 32

_NORM_BYTES = {"fp32": 4, "fp16": 2}
_RNORM_BYTES = {"fp32": 4, "fp16": 2, "uint8": 1}


def _pow2_ceil(n: int) -> int:
    if n <= 1:
        return 1
    return 1 << (n - 1).bit_length()


@dataclass(frozen=True)
class FieldInfo:
    name: str
    bytes_: int
    desc: str


@dataclass(frozen=True)
class StorageReport:
    cfg: TurboQuantConfig
    head_dim: int
    num_outliers: int
    k_fields: list[FieldInfo]
    bf16_baseline_kv: int   # per (token, kv_head)

    @property
    def k_total(self) -> int:
        return sum(f.bytes_ for f in self.k_fields)

    @property
    def kv_total(self) -> int:
        return self.k_total * 2  # V mirrors K

    @property
    def compression_ratio(self) -> float:
        return self.bf16_baseline_kv / self.kv_total

    @property
    def data_bits_per_coord(self) -> float:
        """Idx + qjl bits per coord, ignoring metadata."""
        if self.cfg.is_split:
            d_out = self.num_outliers
            d_reg = self.head_dim - d_out
            b_out_data = self.cfg.bits_outlier
            b_reg_data = self.cfg.bits_regular
            return (d_out * b_out_data + d_reg * b_reg_data) / self.head_dim
        # Homog: pack_bits + 1 (qjl) for prod, pack_bits for mse.
        # tight_pack: data per coord is exactly bits (qjl in nibble).
        if self.cfg.tight_pack:
            return float(self.cfg.bits)
        main_bits = self.cfg.main_bits
        pack_bits = _pow2_ceil(main_bits)
        if self.cfg.algo == "prod":
            return float(pack_bits + 1)  # +1 for separate qjl_sign
        return float(pack_bits)

    @property
    def effective_bits_per_coord(self) -> float:
        """Including all metadata. Counts both K and V coords (256 for d=128)."""
        return self.kv_total * 8 / (self.head_dim * 2)

    def render(self) -> str:
        lines = []
        lines.append(
            f"Storage breakdown for {self.cfg.summary()} "
            f"(head_dim={self.head_dim}):"
        )
        lines.append("")
        lines.append("  K side per (token, kv_head):")
        for f in self.k_fields:
            lines.append(f"    {f.name:30s} {f.bytes_:4d} B  ({f.desc})")
        lines.append(f"    {'K total':30s} {self.k_total:4d} B")
        lines.append("")
        lines.append(f"  V side: same as K -> {self.k_total:4d} B")
        lines.append(f"  K+V total per slot:        {self.kv_total:4d} B")
        lines.append(f"  bf16 baseline:             {self.bf16_baseline_kv:4d} B")
        lines.append(
            f"  Compression ratio:        {self.compression_ratio:5.2f}x"
        )
        lines.append("")
        lines.append("  Effective bits/coord:")
        lines.append(
            f"    data only:             {self.data_bits_per_coord:5.2f}"
        )
        lines.append(
            f"    incl. metadata:        {self.effective_bits_per_coord:5.2f}"
        )
        return "\n".join(lines)


def _homog_k_fields(cfg: TurboQuantConfig, head_dim: int) -> list[FieldInfo]:
    main_bits = cfg.main_bits
    pack_bits = _pow2_ceil(main_bits)
    idx_bytes = head_dim * pack_bits // 8

    fields: list[FieldInfo] = []
    if cfg.tight_pack:
        fields.append(FieldInfo(
            "cache_k_idx", idx_bytes,
            f"{pack_bits}-bit nibble (idx + qjl merged)",
        ))
    else:
        fields.append(FieldInfo(
            "cache_k_idx", idx_bytes, f"{pack_bits}-bit pack",
        ))
        if cfg.algo == "prod":
            fields.append(FieldInfo(
                "cache_k_qjl_sign", head_dim // 8, "1-bit pack",
            ))

    fields.append(FieldInfo(
        "cache_k_norm", _NORM_BYTES[cfg.norm_dtype], cfg.norm_dtype,
    ))
    if cfg.algo == "prod":
        rdesc = (
            "uint8 [0, 2.0]" if cfg.rnorm_dtype == "uint8"
            else cfg.rnorm_dtype
        )
        fields.append(FieldInfo(
            "cache_k_rnorm", _RNORM_BYTES[cfg.rnorm_dtype], rdesc,
        ))
    return fields


def _split_k_fields(
    cfg: TurboQuantConfig, head_dim: int, num_outliers: int
) -> list[FieldInfo]:
    fields: list[FieldInfo] = []
    d_out = num_outliers
    d_reg = head_dim - d_out
    for slice_name, d_slice, b in [
        ("out", d_out, cfg.bits_outlier),
        ("reg", d_reg, cfg.bits_regular),
    ]:
        main_bits = b - 1 if cfg.algo == "prod" else b
        pack_bits = _pow2_ceil(main_bits)
        idx_bytes = d_slice * pack_bits // 8
        fields.append(FieldInfo(
            f"cache_k_idx_{slice_name}", idx_bytes,
            f"{pack_bits}-bit pack ({d_slice} ch)",
        ))
        if cfg.algo == "prod":
            fields.append(FieldInfo(
                f"cache_k_qjl_{slice_name}", d_slice // 8, "1-bit pack",
            ))
        fields.append(FieldInfo(
            f"cache_k_norm_{slice_name}",
            _NORM_BYTES[cfg.norm_dtype], cfg.norm_dtype,
        ))
        if cfg.algo == "prod":
            rdesc = (
                "uint8 [0, 2.0]" if cfg.rnorm_dtype == "uint8"
                else cfg.rnorm_dtype
            )
            fields.append(FieldInfo(
                f"cache_k_rnorm_{slice_name}",
                _RNORM_BYTES[cfg.rnorm_dtype], rdesc,
            ))
    return fields


def report_for_config(
    cfg: TurboQuantConfig,
    head_dim: int = 128,
    num_outliers: int = _DEFAULT_NUM_OUTLIERS,
) -> StorageReport:
    cfg.validate()
    if cfg.is_split:
        if num_outliers <= 0 or num_outliers >= head_dim:
            raise ValueError(
                f"num_outliers={num_outliers} must satisfy "
                f"0 < num_outliers < head_dim={head_dim}"
            )
        k_fields = _split_k_fields(cfg, head_dim, num_outliers)
    else:
        k_fields = _homog_k_fields(cfg, head_dim)
    bf16_baseline = head_dim * 2 * 2  # bf16 K + V
    return StorageReport(
        cfg=cfg, head_dim=head_dim,
        num_outliers=num_outliers if cfg.is_split else 0,
        k_fields=k_fields,
        bf16_baseline_kv=bf16_baseline,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> int:
    from vllm.turboquant.stages import (
        STAGES,
        UnknownStageError,
        list_stage_names,
        resolve_stage,
    )

    ap = argparse.ArgumentParser(
        description="Storage / compression report for TurboQuant stages.",
    )
    ap.add_argument("stage", nargs="?", help="Stage name (omit if --all)")
    ap.add_argument(
        "--all", action="store_true", help="Report every stage in the registry"
    )
    ap.add_argument(
        "--head-dim", type=int, default=128,
        help="Head dim used for the breakdown (default 128 = Llama 3.x)",
    )
    ap.add_argument(
        "--num-outliers", type=int, default=_DEFAULT_NUM_OUTLIERS,
        help="Outlier channel count for split-mode breakdown",
    )
    args = ap.parse_args()

    if args.all:
        names = list_stage_names()
    elif args.stage:
        names = [args.stage]
    else:
        ap.error("either provide a stage name or pass --all")

    sep = "\n" + "=" * 60 + "\n"
    for n in names:
        try:
            canonical, cfg = resolve_stage(n)
        except UnknownStageError as e:
            print(str(e))
            return 1
        rep = report_for_config(
            cfg, head_dim=args.head_dim, num_outliers=args.num_outliers
        )
        print(f"\n[{canonical}]")
        print(rep.render())
        if args.all:
            print(sep)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
