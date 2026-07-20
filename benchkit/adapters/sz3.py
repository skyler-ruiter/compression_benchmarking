"""SZ3 reference adapter — wraps the `sz3` CLI (CPU-only).

SZ3 differences from the GPU adapters that shape this adapter:

  - CPU-only, no CUDA. Timing is the tool's own self-reported wall-clock time
    (not a device/CUDA-event measurement) — comparable across SZ3 runs, but
    NOT comparable to the GPU adapters' device_ms. Treat SZ3 numbers as a
    CPU-baseline reference, not a throughput-ranking peer.

  - No in-process repeat flag: benchmark() makes N separate subprocess calls
    per phase. Process-launch overhead is negligible next to SZ3's own
    compute time (unlike the CUDA-context-init cost that made cold
    subprocess loops unusable for the GPU tools — see fzgm.md / cusz.md).

  - Native modes: `-M ABS <eb>` and `-M REL <eb>` (SZ3 calls the latter
    "VR_REL" — value-range-based, i.e. eb x (max-min), identical semantics to
    canonical rel_range / cuSZ r2r / FZGM NOA). rel_maxabs has no native
    equivalent (SZ3 also offers PSNR/NORM/ABS_AND_REL/ABS_OR_REL modes, none
    of which are a maxabs-relative bound) and raises AdapterError.

  - Separate compress/decompress invocations (not a combined round-trip).
    Dimensions are fastest-first (`-1`/-2/-3/-4 nx [ny [nz]]`), matching
    FieldSpec.dims directly — no reordering needed.

See docs/adapters/sz3.md for the full contract.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path

from .base import (Adapter, AdapterError, BenchmarkResult, CompressResult,
                   DecompressResult, Prepared, RunSpec, run_cli)

_MODE_MAP = {
    "abs":       ("ABS", "abs"),
    "rel_range": ("REL", "range"),
}

_DTYPE_FLAG = {"f32": "-f", "f64": "-d"}

_TIME_RE = re.compile(
    r"^(compression|decompression) time\s*=\s*([0-9]+\.?[0-9]*(?:[eE][+-]?[0-9]+)?)",
)


def resolve_cli(explicit: str | None = None) -> str:
    for cand in (explicit, os.environ.get("SZ3_CLI")):
        if cand:
            return cand
    found = shutil.which("sz3")
    if found:
        return found
    raise AdapterError(
        "sz3 not found: set SZ3_CLI or cli_path in the run entry. "
        "See scripts/env-jetstream2.sh for the build path.")


def _dim_args(dims: list[int]) -> list[str]:
    flag = {1: "-1", 2: "-2", 3: "-3", 4: "-4"}.get(len(dims))
    if flag is None:
        raise AdapterError(f"SZ3 supports 1-4D data; got {len(dims)}D: {dims}")
    return [flag, *[str(d) for d in dims]]


def _parse_time_ms(stdout: str, phase: str) -> float:
    for line in stdout.splitlines():
        m = _TIME_RE.match(line.strip())
        if m and m.group(1) == phase:
            return float(m.group(2)) * 1000.0
    raise AdapterError(
        f"No '{phase} time = ...' line found in sz3 output. "
        "Is this the sz3 binary? Check the log.")


class Sz3Adapter(Adapter):
    """SZ3 adapter — CPU-only, ABS and REL(range) error modes."""

    name = "sz3"

    def __init__(self, variant: str = "sz3", cli_path: str | None = None):
        self.variant = variant
        self.cli = resolve_cli(cli_path)

    def is_available(self) -> bool:
        return Path(self.cli).exists() or shutil.which(self.cli) is not None

    def provenance(self) -> dict:
        return {
            "cli_path": self.cli,
            "name": "sz3",
            "timing_method": "cpu_wall_clock_self_reported",
            "timing_note": (
                "SZ3's own internal timer around the compress/decompress call, "
                "CPU-only. Not comparable to GPU adapters' device_ms."
            ),
        }

    def prepare(self, spec: RunSpec, workdir: Path) -> Prepared:
        if spec.error_mode not in _MODE_MAP:
            raise AdapterError(
                f"SZ3 does not support error mode '{spec.error_mode}' "
                f"(supported: {sorted(_MODE_MAP)}). "
                f"rel_maxabs has no equivalent in the sz3 CLI (only ABS/REL/PSNR/"
                f"NORM/ABS_AND_REL/ABS_OR_REL).")
        native_mode, basis = _MODE_MAP[spec.error_mode]
        eb = float(spec.error_bound)

        workdir.mkdir(parents=True, exist_ok=True)
        config_args = [
            _DTYPE_FLAG[spec.field.dtype],
            *_dim_args(spec.field.dims),
            "-M", native_mode, repr(eb),
        ]
        return Prepared(
            config_args=config_args,
            eb=eb,
            native_mode=native_mode,
            basis=basis,
            pipeline_ref="sz3:default",
            pipeline_path=None,
            pipeline_sha256=None,
        )

    def compress(self, spec: RunSpec, prep: Prepared, workdir: Path) -> CompressResult:
        compressed = workdir / "c.sz3"
        log = workdir / "compress.log"
        dtype_flag = _DTYPE_FLAG[spec.field.dtype]

        argv = [self.cli, dtype_flag, "-i", str(spec.field.path), "-z", str(compressed),
                *_dim_args(spec.field.dims), "-M", prep.native_mode, repr(prep.eb)]
        proc = run_cli(argv, log)
        if proc.returncode != 0:
            raise AdapterError(f"compress failed (exit {proc.returncode}); see {log}")
        if not compressed.exists():
            raise AdapterError(f"compress produced no output at {compressed}; see {log}")

        return CompressResult(
            compressed_path=compressed,
            compressed_bytes=compressed.stat().st_size,
            original_bytes=spec.field.original_bytes,
            raw_json={},
            log_path=log,
        )

    def decompress(self, spec: RunSpec, compressed: Path, workdir: Path) -> DecompressResult:
        decompressed = workdir / "d.bin"
        log = workdir / "decompress.log"
        dtype_flag = _DTYPE_FLAG[spec.field.dtype]

        argv = [self.cli, dtype_flag, "-z", str(compressed), "-o", str(decompressed),
                *_dim_args(spec.field.dims)]
        proc = run_cli(argv, log)
        if proc.returncode != 0:
            raise AdapterError(f"decompress failed (exit {proc.returncode}); see {log}")
        if not decompressed.exists():
            raise AdapterError(f"decompress produced no output at {decompressed}; see {log}")

        return DecompressResult(decompressed_path=decompressed, raw_json={}, log_path=log)

    def benchmark(self, spec: RunSpec, prep: Prepared, n_runs: int, workdir: Path) -> BenchmarkResult:
        """N separate subprocess calls per phase; SZ3's own self-reported time is used
        (CPU wall-clock around the algorithm, not process startup)."""
        dtype_flag = _DTYPE_FLAG[spec.field.dtype]
        dim_args = _dim_args(spec.field.dims)
        compressed = workdir / "c.sz3"
        decompressed = workdir / "d_bench.bin"
        log = workdir / "benchmark.log"

        c_argv = [self.cli, dtype_flag, "-i", str(spec.field.path), "-z", str(compressed),
                  *dim_args, "-M", prep.native_mode, repr(prep.eb)]
        d_argv = [self.cli, dtype_flag, "-z", str(compressed), "-o", str(decompressed),
                  *dim_args]

        compress_ms: list[float] = []
        decompress_ms: list[float] = []

        with open(log, "w") as fh:
            for i in range(n_runs):
                fh.write(f"\n# --- compress run {i} ---\n$ {' '.join(c_argv)}\n")
                proc = subprocess.run(c_argv, capture_output=True, text=True)
                fh.write(proc.stdout + proc.stderr)
                if proc.returncode != 0:
                    raise AdapterError(
                        f"benchmark compress run {i} failed (exit {proc.returncode}); see {log}")
                compress_ms.append(_parse_time_ms(proc.stdout, "compression"))

                fh.write(f"\n# --- decompress run {i} ---\n$ {' '.join(d_argv)}\n")
                proc = subprocess.run(d_argv, capture_output=True, text=True)
                fh.write(proc.stdout + proc.stderr)
                if proc.returncode != 0:
                    raise AdapterError(
                        f"benchmark decompress run {i} failed (exit {proc.returncode}); see {log}")
                decompress_ms.append(_parse_time_ms(proc.stdout, "decompression"))

        return BenchmarkResult(
            compress_device_ms_all=compress_ms,
            decompress_device_ms_all=decompress_ms,
            compressed_bytes=compressed.stat().st_size if compressed.exists() else 0,
            stages=[],
            native_quality=None,
            log_path=log,
        )
