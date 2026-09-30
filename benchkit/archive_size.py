"""Offline correction of historical payload-only FZGM size measurements.

This transformation is deliberately in-memory. Callers should write results to a
new derived artifact and retain the untouched source session as provenance.
"""
from __future__ import annotations

import copy
from pathlib import Path

import yaml

from .metrics import compute_size


class ArchiveSizeCorrectionError(ValueError):
    """A historical FZGM row cannot be corrected from the audited mapping."""


DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "configs/analysis/fzgm_archive_size_v1.yaml"


def load_archive_size_config(path: str | Path = DEFAULT_CONFIG) -> dict:
    with Path(path).open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict) or config.get("schema_version") != 1:
        raise ArchiveSizeCorrectionError("archive-size config must have schema_version: 1")
    selectors = config.get("selectors")
    if not isinstance(selectors, dict) or not selectors:
        raise ArchiveSizeCorrectionError("archive-size config has no selectors")
    return config


def correct_fzgm_archive_sizes(rows: list[dict], config_path: str | Path = DEFAULT_CONFIG) -> list[dict]:
    """Return copied rows with historical FZGM ``compressed_bytes`` archive-inclusive.

    Each selector is the exact ``(variant, pipeline)`` pair audited for FZM v3.1.
    A missing or unknown selector is an error, including rows that already have
    the new diagnostic fields, so a changed pipeline cannot silently inherit a
    neighboring topology's header size. Non-FZGM rows are copied unchanged.
    """
    selectors = load_archive_size_config(config_path)["selectors"]
    corrected = []
    for source in rows:
        row = copy.deepcopy(source)
        if row.get("compressor") != "fzgm":
            corrected.append(row)
            continue
        variant, pipeline = row.get("variant"), row.get("pipeline")
        key = f"{variant}|{pipeline}"
        if key not in selectors:
            raise ArchiveSizeCorrectionError(
                f"no audited FZM archive-size selector for variant={variant!r}, pipeline={pipeline!r}"
            )
        overhead = selectors[key].get("header_bytes")
        if not isinstance(overhead, int) or overhead <= 0:
            raise ArchiveSizeCorrectionError(f"invalid header_bytes for selector {key!r}")

        already = row.get("compressed_archive_overhead_bytes")
        if already is not None:
            payload = row.get("compressed_payload_bytes")
            if already != overhead or not isinstance(payload, int) or row.get("compressed_bytes") != payload + overhead:
                raise ArchiveSizeCorrectionError(f"inconsistent previously corrected size for selector {key!r}")
            corrected.append(row)
            continue

        payload = row.get("compressed_bytes")
        if not isinstance(payload, int) or payload <= 0:
            # Failed/non-measurement rows have no size to transform.
            if row.get("status") != "ok":
                corrected.append(row)
                continue
            raise ArchiveSizeCorrectionError(f"FZGM measurement lacks positive compressed_bytes: {key!r}")
        row["compressed_payload_bytes"] = payload
        row["compressed_archive_overhead_bytes"] = overhead
        row["compressed_bytes"] = payload + overhead
        if row.get("original_bytes") and row.get("num_elements"):
            size = compute_size(int(row["original_bytes"]), row["compressed_bytes"], int(row["num_elements"]))
            row["cr"] = size.cr
            row["bitrate_bits_per_elem"] = size.bitrate_bits_per_elem
        corrected.append(row)
    return corrected
