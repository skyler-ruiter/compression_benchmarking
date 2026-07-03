"""Adapter interface + shared subprocess/JSON helpers.

An adapter wraps one compressor. Its only jobs: build the command line, run the
subprocess, and surface (a) the artifacts harness-owned metrics need and (b) the one
number the harness cannot observe from outside — device kernel time. Everything else
(CR, PSNR, throughput) is computed downstream.

Interface split (see docs/adapters/fzgm.md for why):
  - compress()/decompress() produce *artifacts* (compressed blob, decompressed array).
    Their timing is unreliable across fresh processes (cold GPU clocks) and is ignored.
  - benchmark() produces *timing* via the tool's native in-process repeat mode.
"""
from __future__ import annotations

import json
import subprocess
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path

from ..config import FieldSpec


@dataclass
class RunSpec:
    """One atomic measurement target."""
    field: FieldSpec
    error_mode: str        # canonical: abs | rel_range | rel_maxabs | from_toml
    error_bound: float | None   # None when error_mode == from_toml
    pipeline: str          # .toml path or --stages chain
    variant: str           # "reference" | "fzgm"
    graph: bool = False    # request CUDA Graph capture (fzgm only; see fzgm.md)


@dataclass
class Prepared:
    """The resolved, ready-to-run config for one cell.

    Produced by Adapter.prepare(): translates the canonical mode into native flags,
    renders/archives the effective pipeline config, and reports the eb basis the harness
    must use to check bound satisfaction.
    """
    config_args: list[str]      # pipeline+bound flags, e.g. ["-c", "<rendered.toml>"]
    eb: float                   # absolute-or-relative bound actually applied (for row)
    native_mode: str            # "NOA" / "REL" / "ABS"
    basis: str                  # "abs" | "range" | "maxabs"
    pipeline_ref: str           # source template path or stages string
    pipeline_path: Path | None  # archived rendered config (None for --stages)
    pipeline_sha256: str | None


@dataclass
class CompressResult:
    compressed_path: Path
    compressed_bytes: int
    original_bytes: int
    raw_json: dict
    log_path: Path


@dataclass
class DecompressResult:
    decompressed_path: Path
    raw_json: dict
    log_path: Path


@dataclass
class BenchmarkResult:
    compress_device_ms_all: list[float]
    decompress_device_ms_all: list[float]
    compressed_bytes: int
    stages: list[dict] = field(default_factory=list)
    native_quality: dict | None = None     # tool's self-reported quality, for cross-check
    raw_json: dict | None = None
    log_path: Path | None = None
    # CUDA Graph capture (fzgm only). graph_requested mirrors what we asked for;
    # graph_active/graph_reason come from the tool (None if it predates --graph support).
    # See docs/adapters/fzgm.md "Graph mode".
    graph_requested: bool = False
    graph_active: bool | None = None
    graph_reason: str | None = None


class AdapterError(RuntimeError):
    pass


class Adapter(ABC):
    name: str
    variant: str

    @abstractmethod
    def is_available(self) -> bool: ...

    @abstractmethod
    def provenance(self) -> dict:
        """version / commit / build flags for the provenance manifest."""

    @abstractmethod
    def prepare(self, spec: RunSpec, workdir: Path) -> Prepared:
        """Resolve canonical mode + bound into native config; render/archive it."""

    @abstractmethod
    def compress(self, spec: RunSpec, prep: Prepared, workdir: Path) -> CompressResult: ...

    @abstractmethod
    def decompress(self, spec: RunSpec, compressed: Path, workdir: Path) -> DecompressResult: ...

    @abstractmethod
    def benchmark(self, spec: RunSpec, prep: Prepared, n_runs: int, workdir: Path) -> BenchmarkResult: ...


# ---- shared helpers ---------------------------------------------------------


def run_cli(argv: list[str], log_path: Path) -> subprocess.CompletedProcess:
    """Run a subprocess, teeing combined stdout/stderr to log_path."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(argv, capture_output=True, text=True)
    with open(log_path, "w") as fh:
        fh.write("$ " + " ".join(argv) + "\n\n")
        fh.write("---- stdout ----\n" + proc.stdout + "\n")
        fh.write("---- stderr ----\n" + proc.stderr + "\n")
        fh.write(f"\n[exit {proc.returncode}]\n")
    return proc


def load_report_json(path: Path) -> dict:
    """Load a --report-json file and assert tool success."""
    if not path.exists():
        raise AdapterError(f"no JSON report written to {path} "
                           f"(stale binary without --report-json support?)")
    with open(path) as fh:
        report = json.load(fh)
    if report.get("status") != "ok":
        raise AdapterError(f"tool reported failure: {report.get('error_message')}")
    return report
