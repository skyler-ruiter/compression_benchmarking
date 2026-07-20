"""lsCOMP reference adapter — NOT YET SUPPORTED.

lsCOMP is a GPU compressor for unsigned integers (`uint32`/`uint16`), not
floating-point data with an error bound. Its lossy control knobs are
per-level quantization bins (`-b x y z w`) and a pooling threshold (`-p`),
neither of which is an error bound in the abs/rel_range/rel_maxabs sense used
here.

Current status: stub. AdapterError is raised on all interface methods, same
pattern as the MANS adapter. To use lsCOMP on the SDRBench float fields this
harness benchmarks, a quantization pre-step (float -> uint32/16 at a target
error bound, then dequantize post-decompress) is required, and the harness
would need to attribute the quantization error into the eb_ok check itself
(lsCOMP's own bins/pooling don't map onto abs/rel_range/rel_maxabs directly).
This round-trip does not fit the single-adapter model used by the other
compressors any better than MANS's did — see docs/adapters/mans.md and
docs/adapters/lscomp.md for the integration design this would require.
"""
from __future__ import annotations

import os
import shutil
from pathlib import Path

from .base import (Adapter, AdapterError, BenchmarkResult, CompressResult,
                   DecompressResult, Prepared, RunSpec)


def resolve_cli(explicit: str | None = None) -> str:
    for cand in (explicit, os.environ.get("LSCOMP_CLI")):
        if cand:
            return cand
    found = shutil.which("lsCOMP_uint32")
    if found:
        return found
    raise AdapterError(
        "lsCOMP_uint32 not found: set LSCOMP_CLI or cli_path in the run entry. "
        "See docs/adapters/lscomp.md for the required wrapper design.")


class LscompAdapter(Adapter):
    """lsCOMP adapter — currently unsupported (quantized-integer compressor).

    Raises AdapterError on all methods. Registered to reserve the key and
    surface a clear error message with guidance, same pattern as MansAdapter.
    """

    name = "lscomp"

    def __init__(self, variant: str = "lscomp", cli_path: str | None = None):
        self.variant = variant
        self.cli = resolve_cli(cli_path)

    def is_available(self) -> bool:
        try:
            return Path(self.cli).exists() or shutil.which(self.cli) is not None
        except Exception:
            return False

    def provenance(self) -> dict:
        return {
            "cli_path": self.cli,
            "name": "lscomp",
            "status": "stub — quantized-integer compressor; float error-bound wrapper required",
        }

    def prepare(self, spec: RunSpec, workdir: Path) -> Prepared:
        raise AdapterError(
            "lsCOMP compresses unsigned integers (uint32/uint16) via quantization "
            "bins + a pooling threshold, not float data with an error bound. "
            "A quantization wrapper (float -> int at target eb, then decompress -> "
            "dequantize, with eb_ok attributed through the quantization step) is "
            "needed before this adapter can work. "
            "See docs/adapters/lscomp.md for the required integration design.")

    def compress(self, spec: RunSpec, prep: Prepared, workdir: Path) -> CompressResult:
        raise AdapterError("lsCOMP adapter not implemented. See docs/adapters/lscomp.md.")

    def decompress(self, spec: RunSpec, compressed: Path, workdir: Path) -> DecompressResult:
        raise AdapterError("lsCOMP adapter not implemented. See docs/adapters/lscomp.md.")

    def benchmark(self, spec: RunSpec, prep: Prepared, n_runs: int, workdir: Path) -> BenchmarkResult:
        raise AdapterError("lsCOMP adapter not implemented. See docs/adapters/lscomp.md.")
