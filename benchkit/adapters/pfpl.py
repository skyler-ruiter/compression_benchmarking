"""PFPL reference adapter — wraps the LC-framework GPU executables.

PFPL uses separate executables per mode and dtype, located at:
  <bin_dir>/<dtype>/gpu/<dtype>_<mode>_compress_cuda
  <bin_dir>/<dtype>/gpu/<dtype>_<mode>_decompress_cuda

Supported modes:
  abs       -> f32_abs_compress_cuda   basis=abs
  rel_range -> f32_noa_compress_cuda   basis=range  (= canonical rel_range)
  rel_maxabs -> f32_rel_compress_cuda  basis=maxabs (approximate per-element)

CLI (compress):
  f32_abs_compress_cuda input_file compressed_file error_bound [threshold]

CLI (decompress):
  f32_abs_decompress_cuda compressed_file decompressed_file

Timing (CUDA events):
  Each executable runs NUM_RUNS=9 iterations internally.
  Each iteration prints: "lc comp ecltime,  X.XXXXXXXXX" (seconds).
  The adapter parses all 9 values and returns them as ms.
  The runner's warmup_reps controls how many leading values are discarded
  before computing median/CV; PFPL's own runs are all warm after the first.
  n_runs passed to benchmark() is advisory — PFPL always produces 9 values.

Output files:
  The compressed and decompressed files are written AFTER all 9 timing runs.
  compress() and decompress() each do one subprocess call (9 timed reps
  implicit). benchmark() does the same, overwriting those files.

See docs/adapters/pfpl.md for the full contract.
"""
from __future__ import annotations

import os
import shutil
from pathlib import Path

from .base import (Adapter, AdapterError, BenchmarkResult, CompressResult,
                   DecompressResult, Prepared, RunSpec, run_cli)

# canonical → (mode_str_in_exe, native_mode_label, eb_basis)
_MODE_MAP = {
    "abs":        ("abs", "ABS",  "abs"),
    "rel_range":  ("noa", "NOA",  "range"),
    "rel_maxabs": ("rel", "REL",  "maxabs"),
}

_PFPL_RUNS = 9   # hardcoded NUM_RUNS in all PFPL GPU executables


def resolve_bin_dir(explicit: str | None = None) -> Path:
    for cand in (explicit, os.environ.get("PFPL_BIN_DIR")):
        if cand:
            return Path(cand)
    raise AdapterError(
        "PFPL bin dir not found: set PFPL_BIN_DIR or cli_path in the run entry. "
        "Expected layout: <bin_dir>/f32/gpu/f32_abs_compress_cuda. "
        "See scripts/env-bigred200.sh.")


def _exe(bin_dir: Path, dtype: str, mode_str: str, direction: str) -> Path:
    return bin_dir / dtype / "gpu" / f"{dtype}_{mode_str}_{direction}_cuda"


def _parse_timings_s(stdout: str, phase: str) -> list[float]:
    """Parse lines like 'lc comp ecltime,  0.001234567' → list of seconds."""
    prefix = f"lc {phase} ecltime,"
    vals = []
    for line in stdout.splitlines():
        line = line.strip()
        if line.startswith(prefix):
            vals.append(float(line[len(prefix):].strip()))
    return vals


class PfplAdapter(Adapter):
    """PFPL LC-framework GPU adapter (f32 only)."""

    name = "pfpl"

    def __init__(self, variant: str = "pfpl", cli_path: str | None = None):
        self.variant = variant
        self.bin_dir = resolve_bin_dir(cli_path)

    def _check_exe(self, dtype: str, mode_str: str, direction: str) -> Path:
        p = _exe(self.bin_dir, dtype, mode_str, direction)
        if not p.exists():
            raise AdapterError(
                f"PFPL executable not found: {p}\n"
                f"Check that PFPL_BIN_DIR={self.bin_dir} is correct and "
                f"PFPL was built with 'make all'.")
        return p

    def is_available(self) -> bool:
        try:
            return _exe(self.bin_dir, "f32", "abs", "compress").exists()
        except Exception:
            return False

    def provenance(self) -> dict:
        return {
            "cli_path": str(self.bin_dir),
            "name": "pfpl",
            "timing_method": "cuda_events_seconds",
            "timing_reps_per_call": _PFPL_RUNS,
        }

    def prepare(self, spec: RunSpec, workdir: Path) -> Prepared:
        if spec.error_mode not in _MODE_MAP:
            raise AdapterError(
                f"PFPL does not support error mode '{spec.error_mode}' "
                f"(supported: {sorted(_MODE_MAP)}).")
        mode_str, native_mode, basis = _MODE_MAP[spec.error_mode]
        dtype = spec.field.dtype
        if dtype not in ("f32", "f64"):
            raise AdapterError(f"PFPL: unsupported dtype '{dtype}'.")

        self._check_exe(dtype, mode_str, "compress")
        self._check_exe(dtype, mode_str, "decompress")

        eb = float(spec.error_bound)
        workdir.mkdir(parents=True, exist_ok=True)

        return Prepared(
            config_args=[repr(eb)],  # error_bound positional arg
            eb=eb,
            native_mode=native_mode,
            basis=basis,
            pipeline_ref=f"pfpl:{native_mode}",
            pipeline_path=None,
            pipeline_sha256=None,
        )

    def _mode_str(self, spec: RunSpec) -> str:
        return _MODE_MAP[spec.error_mode][0]

    def compress(self, spec: RunSpec, prep: Prepared, workdir: Path) -> CompressResult:
        dtype = spec.field.dtype
        mode_str = self._mode_str(spec)
        exe = self._check_exe(dtype, mode_str, "compress")
        compressed = workdir / "c.pfpl"
        log = workdir / "compress.log"

        argv = [str(exe), str(spec.field.path), str(compressed), *prep.config_args]
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
        dtype = spec.field.dtype
        mode_str = self._mode_str(spec)
        exe = self._check_exe(dtype, mode_str, "decompress")
        decompressed = workdir / "d.bin"
        log = workdir / "decompress.log"

        argv = [str(exe), str(compressed), str(decompressed)]
        proc = run_cli(argv, log)
        if proc.returncode != 0:
            raise AdapterError(f"decompress failed (exit {proc.returncode}); see {log}")
        if not decompressed.exists():
            raise AdapterError(f"decompress produced no output at {decompressed}; see {log}")

        return DecompressResult(decompressed_path=decompressed, raw_json={}, log_path=log)

    def benchmark(self, spec: RunSpec, prep: Prepared, n_runs: int, workdir: Path) -> BenchmarkResult:
        """Run compress once (9 internal reps) then decompress once (9 internal reps).

        PFPL runs 9 timed iterations per invocation (NUM_RUNS=9 hardcoded).
        n_runs is advisory — we always return exactly 9 values per phase.
        Set warmup_reps ≤ 8 in the experiment config; the runner discards
        those leading values before computing statistics.
        """
        dtype = spec.field.dtype
        mode_str = self._mode_str(spec)
        c_exe = self._check_exe(dtype, mode_str, "compress")
        d_exe = self._check_exe(dtype, mode_str, "decompress")
        compressed = workdir / "c.pfpl"
        decompressed_tmp = workdir / "d_bench.bin"
        log = workdir / "benchmark.log"

        with open(log, "w") as fh:
            # --- compress (produces 9 timing values + writes compressed file) ---
            c_argv = [str(c_exe), str(spec.field.path), str(compressed), *prep.config_args]
            fh.write(f"# --- compress ---\n$ {' '.join(c_argv)}\n")
            import subprocess
            proc = subprocess.run(c_argv, capture_output=True, text=True)
            fh.write(proc.stdout + proc.stderr)
            if proc.returncode != 0:
                raise AdapterError(
                    f"benchmark compress failed (exit {proc.returncode}); see {log}")
            compress_s = _parse_timings_s(proc.stdout, "comp")
            if not compress_s:
                raise AdapterError(
                    f"benchmark compress: no 'lc comp ecltime' lines found; "
                    f"check binary; see {log}")

            if not compressed.exists():
                raise AdapterError(
                    f"benchmark compress wrote no file at {compressed}; see {log}")

            # --- decompress (produces 9 timing values) ---
            d_argv = [str(d_exe), str(compressed), str(decompressed_tmp)]
            fh.write(f"\n# --- decompress ---\n$ {' '.join(d_argv)}\n")
            proc = subprocess.run(d_argv, capture_output=True, text=True)
            fh.write(proc.stdout + proc.stderr)
            if proc.returncode != 0:
                raise AdapterError(
                    f"benchmark decompress failed (exit {proc.returncode}); see {log}")
            decompress_s = _parse_timings_s(proc.stdout, "decomp")
            if not decompress_s:
                raise AdapterError(
                    f"benchmark decompress: no 'lc decomp ecltime' lines found; "
                    f"check binary; see {log}")

        compress_ms = [s * 1000.0 for s in compress_s]
        decompress_ms = [s * 1000.0 for s in decompress_s]

        return BenchmarkResult(
            compress_device_ms_all=compress_ms,
            decompress_device_ms_all=decompress_ms,
            compressed_bytes=compressed.stat().st_size,
            stages=[],
            native_quality=None,
            log_path=log,
        )
