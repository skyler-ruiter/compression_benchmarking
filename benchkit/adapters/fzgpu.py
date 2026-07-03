"""FZ-GPU reference adapter.

FZ-GPU (HPDC'23) is a float32, range-relative Lorenzo+bitshuffle compressor.
Its single binary does a full compress+decompress round-trip in GPU memory and
prints wall-clock timing (std::chrono::system_clock) for both phases.

The binary has been patched (see docs/adapters/fzgpu.md) to write compressed and
decompressed files to disk when optional 6th and 7th positional args are given:

    fz-gpu <input> <x> <y> <z> <eb> [compressed_out] [decompressed_out]

If those args are omitted the binary runs the original in-memory round-trip —
benchmark() uses this mode to avoid file-I/O overhead during timing.

Adapter model (same as cuSZp — single round-trip binary):
  - compress(): full round-trip, writes both files; timing from stdout.
  - decompress(): returns the decompressed file already written by compress().
  - benchmark(): N subprocess calls without output paths, parses timing.

Limitations vs other adapters:
  - float32 only (no f64 support)
  - rel_range error mode only (FZ-GPU uses eb × range internally)
  - wall-clock timing (std::chrono), not CUDA events — expect more variance
  - VERIFICATION block always runs (hardcoded in source); timing is unaffected
    since both timers are captured before the verification block.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path

from .base import (Adapter, AdapterError, BenchmarkResult, CompressResult,
                   DecompressResult, Prepared, RunSpec, run_cli)


def _resolve_cli(explicit: str | None = None) -> str:
    for cand in (explicit, os.environ.get("FZGPU_CLI")):
        if cand:
            return cand
    found = shutil.which("fz-gpu")
    if found:
        return found
    raise AdapterError(
        "fz-gpu not found: set FZGPU_CLI or cli_path in the run entry. "
        "See scripts/env-bigred200.sh. Rebuild with 'make main' after patching src/fz.cu.")


def _parse_time_s(stdout: str, phase: str) -> float:
    """Extract wall-clock seconds from 'compression/decompression e2e time: X s'."""
    key = f"{phase} e2e time:"
    for line in stdout.splitlines():
        stripped = line.strip()
        if stripped.startswith(key):
            m = re.search(r"([0-9]+\.?[0-9]*(?:e[+-]?[0-9]+)?)\s*s", stripped)
            if m:
                return float(m.group(1))
    raise AdapterError(
        f"No '{key}' line found in fz-gpu output. "
        "Is this the patched binary? Check the log.")


def _parse_psnr(stdout: str) -> float | None:
    for line in stdout.splitlines():
        if line.strip().startswith("PSNR:"):
            m = re.search(r"PSNR:\s*([0-9]+\.?[0-9]*)", line)
            if m:
                return float(m.group(1))
    return None


class FzgpuAdapter(Adapter):
    """FZ-GPU adapter — float32, rel_range only, wall-clock timing."""

    name = "fzgpu"

    def __init__(self, variant: str = "fzgpu", cli_path: str | None = None):
        self.variant = variant
        self.cli = _resolve_cli(cli_path)

    def is_available(self) -> bool:
        return Path(self.cli).exists() or shutil.which(self.cli) is not None

    def provenance(self) -> dict:
        return {
            "cli_path": self.cli,
            "name": "fz-gpu",
            "timing_method": "wall_clock_chrono_e2e",
            "timing_note": (
                "std::chrono::system_clock around kernel launches + cudaDeviceSynchronize. "
                "More variance than CUDA-event adapters; cv threshold still applies."
            ),
            "dtype": "f32 only",
            "error_mode": "rel_range only (NOA: eb × range)",
        }

    def prepare(self, spec: RunSpec, workdir: Path) -> Prepared:
        if spec.error_mode != "rel_range":
            raise AdapterError(
                f"FZ-GPU only supports 'rel_range' (eb × data range). "
                f"Got '{spec.error_mode}'. "
                f"Note: FZ-GPU REL = eb × max(|data|) is NOT rel_range; this adapter "
                f"uses NOA mode (eb × range) which is the cross-tool comparable.")
        if spec.field.dtype != "f32":
            raise AdapterError(
                f"FZ-GPU only supports float32. Got dtype='{spec.field.dtype}'.")

        eb = float(spec.error_bound)
        workdir.mkdir(parents=True, exist_ok=True)

        dims = spec.field.dims
        if len(dims) == 1:
            x, y, z = dims[0], 1, 1
        elif len(dims) == 2:
            x, y, z = dims[0], dims[1], 1
        else:
            x, y, z = dims[0], dims[1], dims[2]

        return Prepared(
            config_args=[str(spec.field.path), str(x), str(y), str(z), repr(eb)],
            eb=eb,
            native_mode="NOA",
            basis="range",
            pipeline_ref="fzgpu:lorenzo+bitshuffle",
            pipeline_path=None,
            pipeline_sha256=None,
        )

    def compress(self, spec: RunSpec, prep: Prepared, workdir: Path) -> CompressResult:
        """Full round-trip; writes both compressed bitstream and decompressed floats."""
        compressed = workdir / "c.fzg"
        decompressed = workdir / "d.bin"
        log = workdir / "compress.log"

        argv = [self.cli, *prep.config_args, str(compressed), str(decompressed)]
        proc = run_cli(argv, log)
        if proc.returncode != 0:
            raise AdapterError(f"fz-gpu compress failed (exit {proc.returncode}); see {log}")
        if not compressed.exists():
            raise AdapterError(
                f"fz-gpu wrote no compressed file at {compressed}. "
                "Is this the patched binary? Rebuild with 'make main' in the FZ-GPU dir.")
        if not decompressed.exists():
            raise AdapterError(
                f"fz-gpu wrote no decompressed file at {decompressed}; see {log}")

        return CompressResult(
            compressed_path=compressed,
            compressed_bytes=compressed.stat().st_size,
            original_bytes=spec.field.original_bytes,
            raw_json={},
            log_path=log,
        )

    def decompress(self, spec: RunSpec, compressed: Path, workdir: Path) -> DecompressResult:
        """Return decompressed file written by compress(). FZ-GPU has no decompress-only mode."""
        decompressed = workdir / "d.bin"
        if not decompressed.exists():
            raise AdapterError(
                f"Decompressed file not found at {decompressed}. "
                "compress() must be called first — fz-gpu does both phases in one run.")
        return DecompressResult(
            decompressed_path=decompressed,
            raw_json={},
            log_path=workdir / "compress.log",
        )

    def benchmark(self, spec: RunSpec, prep: Prepared, n_runs: int, workdir: Path) -> BenchmarkResult:
        """N subprocess calls without output paths (avoids file I/O overhead during timing).

        Each call does the full in-memory compress+decompress round-trip and prints
        wall-clock timing for both phases. Timing is in seconds; the adapter converts to ms.

        The VERIFICATION block in fz-gpu runs after the timing window, so it does not
        affect the reported times — but it does print verification messages to stdout.
        """
        compressed_ref = workdir / "c.fzg"
        log = workdir / "benchmark.log"

        # No output path args → pure in-memory run, no file I/O overhead
        argv = [self.cli, *prep.config_args]

        compress_ms: list[float] = []
        decompress_ms: list[float] = []

        with open(log, "w") as fh:
            for i in range(n_runs):
                fh.write(f"\n# --- run {i} ---\n$ {' '.join(argv)}\n")
                proc = subprocess.run(argv, capture_output=True, text=True)
                fh.write(proc.stdout + proc.stderr)
                if proc.returncode != 0:
                    raise AdapterError(
                        f"benchmark run {i} failed (exit {proc.returncode}); see {log}")
                try:
                    c_s = _parse_time_s(proc.stdout, "compression")
                    d_s = _parse_time_s(proc.stdout, "decompression")
                except AdapterError as e:
                    raise AdapterError(f"run {i}: {e}") from e
                compress_ms.append(c_s * 1000.0)
                decompress_ms.append(d_s * 1000.0)

        psnr = _parse_psnr(open(log).read()) if log.exists() else None
        native_quality = {"psnr_db": psnr} if psnr is not None else None

        return BenchmarkResult(
            compress_device_ms_all=compress_ms,
            decompress_device_ms_all=decompress_ms,
            compressed_bytes=compressed_ref.stat().st_size if compressed_ref.exists() else 0,
            stages=[],
            native_quality=native_quality,
            log_path=log,
        )
