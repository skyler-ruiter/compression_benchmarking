"""MANS reference adapter — NOT YET FULLY SUPPORTED.

MANS (Multi-dimensional Adaptive Non-uniform Superposition) is a lossless
integer compressor. Its CLI (`nv_mans_compress`) takes quantized integer data
(u16 or u32), not floating-point data with an error bound. This makes it
fundamentally different from the other error-bounded compressors.

Current status: stub. AdapterError is raised on all interface methods.
To use MANS, a quantization pre-step that converts float → integers at a
target error bound is required, followed by dequantization post-decompress.
This round-trip does not map cleanly to the single-adapter model used by
the other compressors.

See docs/adapters/mans.md for the integration design.
"""
from __future__ import annotations

import os
import shutil
from pathlib import Path

from .base import (Adapter, AdapterError, BenchmarkResult, CompressResult,
                   DecompressResult, Prepared, RunSpec)


def resolve_compress_cli(explicit: str | None = None) -> str:
    for cand in (explicit, os.environ.get("MANS_CLI")):
        if cand:
            return cand
    found = shutil.which("nv_mans_compress")
    if found:
        return found
    raise AdapterError(
        "nv_mans_compress not found: set MANS_CLI or cli_path in the run entry. "
        "See docs/adapters/mans.md for the required wrapper design.")


class MansAdapter(Adapter):
    """MANS adapter — currently unsupported (lossless integer compressor).

    Raises AdapterError on all methods. Registered to reserve the key
    and surface a clear error message with guidance.
    """

    name = "mans"

    def __init__(self, variant: str = "mans", cli_path: str | None = None):
        self.variant = variant
        self.cli = resolve_compress_cli(cli_path)

    def is_available(self) -> bool:
        try:
            return (Path(self.cli).exists() or shutil.which(self.cli) is not None)
        except Exception:
            return False

    def provenance(self) -> dict:
        return {
            "cli_path": self.cli,
            "name": "mans",
            "status": "stub — lossless integer compressor; float quantization wrapper required",
        }

    def prepare(self, spec: RunSpec, workdir: Path) -> Prepared:
        raise AdapterError(
            "MANS is a lossless integer compressor (u16/u32 input). "
            "It does not accept float data or an error bound directly. "
            "A quantization wrapper (float → int at target eb, then decompress → dequantize) "
            "is needed before this adapter can work. "
            "See docs/adapters/mans.md for the required integration design.")

    def compress(self, spec: RunSpec, prep: Prepared, workdir: Path) -> CompressResult:
        raise AdapterError("MANS adapter not implemented. See docs/adapters/mans.md.")

    def decompress(self, spec: RunSpec, compressed: Path, workdir: Path) -> DecompressResult:
        raise AdapterError("MANS adapter not implemented. See docs/adapters/mans.md.")

    def benchmark(self, spec: RunSpec, prep: Prepared, n_runs: int, workdir: Path) -> BenchmarkResult:
        raise AdapterError("MANS adapter not implemented. See docs/adapters/mans.md.")
