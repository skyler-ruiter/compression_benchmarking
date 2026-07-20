"""SPERR reference adapter — wraps the `sperr2d`/`sperr3d` CLIs (CPU, OpenMP).

SPERR quirks that shape this adapter:

  - **CPU-only**, no CUDA build in this project's usage. Timing is measured
    externally (wall-clock around the subprocess, like zfp) since neither
    binary prints its own elapsed time — `--print_stats` reports range/PSNR/
    bitrate, not timing.

  - **Separate 2D/3D binaries, no 1D or 4D.** `sperr2d` takes `--dims nx ny`;
    `sperr3d` takes `--dims nx ny nz` (both fastest-first, matching
    FieldSpec.dims directly). The adapter picks the binary from
    `len(field.dims)` and raises AdapterError for 1D/4D fields (HACC, EXAALT,
    QMCPACK) — exclude those with `skip_datasets`/`only_datasets` in the
    experiment config, the same pattern used for cuszhi vs. HACC.

  - **No native relative mode.** Only `--pwe <tolerance>` (absolute
    point-wise error). rel_range / rel_maxabs are emulated exactly as in the
    zfp/MGARD adapters: read the input file (`read_range_stats`), compute
    range or maxabs, and pass `--pwe (eb * range_or_maxabs)`.

  - Positional argument is the data volume (compress) or bitstream
    (decompress), given last on the command line.

See docs/adapters/sperr.md for the full contract.
"""
from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path

from .base import (Adapter, AdapterError, BenchmarkResult, CompressResult,
                   DecompressResult, Prepared, RunSpec, run_cli, read_range_stats)

_FTYPE_FLAG = {"f32": "32", "f64": "64"}
_DECOMP_FLAG = {"f32": "--decomp_f", "f64": "--decomp_d"}


def resolve_bin_dir(explicit: str | None = None) -> Path:
    for cand in (explicit, os.environ.get("SPERR_BIN_DIR")):
        if cand:
            return Path(cand)
    raise AdapterError(
        "SPERR bin dir not found: set SPERR_BIN_DIR or cli_path in the run entry. "
        "Expected layout: <bin_dir>/sperr2d, <bin_dir>/sperr3d. "
        "See scripts/env-jetstream2.sh.")


def _binary_and_dims(bin_dir: Path, dims: list[int]) -> tuple[Path, list[str]]:
    if len(dims) == 2:
        return bin_dir / "sperr2d", [str(d) for d in dims]
    if len(dims) == 3:
        return bin_dir / "sperr3d", [str(d) for d in dims]
    raise AdapterError(
        f"SPERR only supports 2D or 3D data; got {len(dims)}D: {dims}. "
        f"Exclude this dataset with skip_datasets/only_datasets.")


class SperrAdapter(Adapter):
    """SPERR adapter — CPU, 2D/3D only, pwe (abs) native + emulated rel."""

    name = "sperr"

    def __init__(self, variant: str = "sperr", cli_path: str | None = None):
        self.variant = variant
        self.bin_dir = resolve_bin_dir(cli_path)

    def is_available(self) -> bool:
        try:
            return (self.bin_dir / "sperr3d").exists()
        except Exception:
            return False

    def provenance(self) -> dict:
        return {
            "cli_path": str(self.bin_dir),
            "name": "sperr",
            "timing_method": "cpu_wall_clock_external",
            "timing_note": (
                "Neither sperr2d nor sperr3d report elapsed time; timing is "
                "measured externally around the subprocess (includes process "
                "startup) — not comparable to the GPU adapters' device_ms."
            ),
            "dims_supported": "2D, 3D only (no 1D/4D)",
        }

    def prepare(self, spec: RunSpec, workdir: Path) -> Prepared:
        eb = float(spec.error_bound)
        if spec.error_mode == "abs":
            abs_eb, basis = eb, "abs"
        elif spec.error_mode == "rel_range":
            vrange, _ = read_range_stats(spec.field)
            abs_eb, basis = eb * vrange, "range"
        elif spec.error_mode == "rel_maxabs":
            _, vmaxabs = read_range_stats(spec.field)
            abs_eb, basis = eb * vmaxabs, "maxabs"
        else:
            raise AdapterError(f"SPERR: unsupported error mode '{spec.error_mode}'.")

        exe, dim_args = _binary_and_dims(self.bin_dir, spec.field.dims)
        if not exe.exists():
            raise AdapterError(f"SPERR executable not found: {exe}")

        workdir.mkdir(parents=True, exist_ok=True)
        return Prepared(
            config_args=["--ftype", _FTYPE_FLAG[spec.field.dtype], "--dims", *dim_args,
                         "--pwe", repr(abs_eb)],
            eb=eb,
            native_mode="pwe" if spec.error_mode == "abs" else f"pwe-emulated({spec.error_mode})",
            basis=basis,
            pipeline_ref=f"sperr:{exe.name}",
            pipeline_path=None,
            pipeline_sha256=None,
        )

    def compress(self, spec: RunSpec, prep: Prepared, workdir: Path) -> CompressResult:
        exe, _ = _binary_and_dims(self.bin_dir, spec.field.dims)
        compressed = workdir / "c.sperr"
        log = workdir / "compress.log"

        argv = [str(exe), "-c", *prep.config_args, "--bitstream", str(compressed),
                str(spec.field.path)]
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
        exe, _ = _binary_and_dims(self.bin_dir, spec.field.dims)
        decompressed = workdir / "d.bin"
        log = workdir / "decompress.log"
        decomp_flag = _DECOMP_FLAG[spec.field.dtype]

        argv = [str(exe), "-d", decomp_flag, str(decompressed), str(compressed)]
        proc = run_cli(argv, log)
        if proc.returncode != 0:
            raise AdapterError(f"decompress failed (exit {proc.returncode}); see {log}")
        if not decompressed.exists():
            raise AdapterError(f"decompress produced no output at {decompressed}; see {log}")

        return DecompressResult(decompressed_path=decompressed, raw_json={}, log_path=log)

    def benchmark(self, spec: RunSpec, prep: Prepared, n_runs: int, workdir: Path) -> BenchmarkResult:
        """N subprocess calls per phase, timed externally with perf_counter()
        (neither binary reports its own elapsed time)."""
        exe, _ = _binary_and_dims(self.bin_dir, spec.field.dims)
        compressed = workdir / "c.sperr"
        decompressed = workdir / "d_bench.bin"
        decomp_flag = _DECOMP_FLAG[spec.field.dtype]
        log = workdir / "benchmark.log"

        c_argv = [str(exe), "-c", *prep.config_args, "--bitstream", str(compressed),
                  str(spec.field.path)]
        d_argv = [str(exe), "-d", decomp_flag, str(decompressed), str(compressed)]

        compress_ms: list[float] = []
        decompress_ms: list[float] = []

        with open(log, "w") as fh:
            for i in range(n_runs):
                fh.write(f"\n# --- compress run {i} ---\n$ {' '.join(c_argv)}\n")
                t0 = time.perf_counter()
                proc = subprocess.run(c_argv, capture_output=True, text=True)
                elapsed = time.perf_counter() - t0
                fh.write(proc.stdout + proc.stderr)
                if proc.returncode != 0:
                    raise AdapterError(
                        f"benchmark compress run {i} failed (exit {proc.returncode}); see {log}")
                compress_ms.append(elapsed * 1000.0)

                fh.write(f"\n# --- decompress run {i} ---\n$ {' '.join(d_argv)}\n")
                t0 = time.perf_counter()
                proc = subprocess.run(d_argv, capture_output=True, text=True)
                elapsed = time.perf_counter() - t0
                fh.write(proc.stdout + proc.stderr)
                if proc.returncode != 0:
                    raise AdapterError(
                        f"benchmark decompress run {i} failed (exit {proc.returncode}); see {log}")
                decompress_ms.append(elapsed * 1000.0)

        return BenchmarkResult(
            compress_device_ms_all=compress_ms,
            decompress_device_ms_all=decompress_ms,
            compressed_bytes=compressed.stat().st_size if compressed.exists() else 0,
            stages=[],
            native_quality=None,
            log_path=log,
        )
