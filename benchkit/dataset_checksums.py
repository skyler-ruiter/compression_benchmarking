"""Versioned checksum lock files for dataset manifests."""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import yaml


CHECKSUM_SCHEMA_VERSION = 1


def load_checksum_lock(path: Path) -> tuple[dict[str, dict[str, str]], str | None]:
    if not path.exists():
        return {}, None
    text = path.read_text()
    raw = yaml.safe_load(text) or {}
    if raw.get("checksum_schema_version") != CHECKSUM_SCHEMA_VERSION:
        raise ValueError(f"{path}: unsupported checksum_schema_version")
    if raw.get("algorithm") != "sha256" or raw.get("byte_scope") != "declared_prefix":
        raise ValueError(f"{path}: expected sha256 over the declared byte prefix")
    datasets = raw.get("datasets", {})
    if not isinstance(datasets, dict):
        raise ValueError(f"{path}: datasets must be a mapping")
    for dataset, fields in datasets.items():
        if not isinstance(fields, dict):
            raise ValueError(f"{path}: datasets.{dataset} must be a mapping")
        for field, digest in fields.items():
            if not valid_sha256(digest):
                raise ValueError(f"{path}: invalid digest for {dataset}/{field}")
    return datasets, text


def valid_sha256(value: Any) -> bool:
    return (isinstance(value, str) and len(value) == 64 and
            all(c in "0123456789abcdef" for c in value))


def checksum_prefix(path: Path, size: int, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    remaining = size
    with path.open("rb") as handle:
        while remaining:
            block = handle.read(min(chunk, remaining))
            if not block:
                raise ValueError(f"{path}: ended before declared {size} byte prefix")
            digest.update(block)
            remaining -= len(block)
    return digest.hexdigest()


def dump_checksum_lock(datasets: dict[str, dict[str, str]]) -> str:
    document = {
        "checksum_schema_version": CHECKSUM_SCHEMA_VERSION,
        "algorithm": "sha256",
        "byte_scope": "declared_prefix",
        "provenance_note": ("Locally established from integrity-checked SDRBench "
                            "extractions; identifies exact bytes, not a vendor signature."),
        "datasets": {dataset: dict(sorted(fields.items()))
                     for dataset, fields in sorted(datasets.items()) if fields},
    }
    return yaml.safe_dump(document, sort_keys=False)
