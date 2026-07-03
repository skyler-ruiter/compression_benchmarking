"""cuSZ-Hi reference adapter — wraps the `cuszhi` binary (psz/cuSZ-Hi project).

cuSZ-Hi differences from the patched cuSZ adapter:

  - Timing via -R time text table: cuSZ-Hi has not been patched to emit
    JSON. benchmark() passes -R time to both -z and -x invocations and
    parses the "(total)" row from the human-readable table. The row format
    is "  (total)   <ms>   <GiB/s>" printed via printf %'12f (locale-aware
    thousand separators). The adapter strips commas before parsing.

  - No -S write2disk: cuSZ-Hi always writes output files. This is fine
    for timing accuracy — file I/O occurs outside the CUDA event window.
    The -R time (total) row measures kernel time only, comparable to fzgm.

  - Same output path convention as cuSZ: compressed output is
    <input_file>.cusza (same directory as input). We symlink the input
    into the workdir. Decompressed output is <stem>.cuszx in the same
    directory as the compressed file.

  - Supported modes: abs and rel_range (cuszhi calls the latter "r2r").
    rel_maxabs is not supported.

  - Dim order: cuszhi uses -l x,y,z (fastest-to-slowest for 3D).
    The adapter passes the dim_arg from FieldSpec (x notation).

See docs/adapters/cuszhi.md for the full contract.
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
    "abs":       ("abs", "abs"),
    "rel_range": ("r2r", "range"),
}

_DTYPE_FLAG = {"f32": "f32", "f64": "f64"}


def resolve_cli(explicit: str | None = None) -> str:
    for cand in (explicit, os.environ.get("CUSZHI_CLI")):
        if cand:
            return cand
    found = shutil.which("cuszhi")
    if found:
        return found
    raise AdapterError(
        "cuszhi not found: set CUSZHI_CLI or cli_path in the run entry. "
        "See scripts/env-bigred200.sh for build path.")


def _parse_total_ms(stdout: str) -> float:
    """Parse the '(total)' timing row from cuSZ-Hi -R time output.

    Row format (printf "  %-12s %'12f %'10.2f"):
      "  (total)        1234.567890    12.34"
    The %' modifier may add locale thousand-separators (commas); strip them.
    """
    for line in stdout.splitlines():
        if "(total)" in line:
            # Extract numeric tokens — first is ms, second is GiB/s
            cleaned = line.replace(",", "")
            nums = re.findall(r"\d+\.\d+", cleaned)
            if nums:
                return float(nums[0])
    raise AdapterError(
        "No '(total)' timing row found in cuszhi -R time output. "
        "Was -R time passed and is this a cuszhi build with reporting?")


class CuszhiAdapter(Adapter):
    name = "cuszhi"

    def __init__(self, variant: str = "cuszhi", cli_path: str | None = None):
        self.variant = variant
        self.cli = resolve_cli(cli_path)

    def is_available(self) -> bool:
        return Path(self.cli).exists() or shutil.which(self.cli) is not None

    def provenance(self) -> dict:
        return {"cli_path": self.cli, "name": "cuszhi",
                "timing_method": "cuda_events_total_ms_from_report_table"}

    def prepare(self, spec: RunSpec, workdir: Path) -> Prepared:
        if spec.error_mode not in _MODE_MAP:
            raise AdapterError(
                f"cuSZ-Hi does not support error mode '{spec.error_mode}' "
                f"(supported: {sorted(_MODE_MAP)}). "
                f"rel_maxabs has no equivalent in the cuszhi CLI.")
        native_mode, basis = _MODE_MAP[spec.error_mode]
        eb = float(spec.error_bound)

        # pipeline selects the -s lossless-pipeline mode: "cr" (default, high-ratio,
        # slow) or "tp" (fast, lower CR). "default" is accepted as an alias for "cr"
        # (the tool's own default) so existing configs keep working.
        lossless_mode = spec.pipeline.strip().lower()
        if lossless_mode == "default":
            lossless_mode = "cr"
        if lossless_mode not in ("cr", "tp"):
            raise AdapterError(
                f"cuSZ-Hi: unknown pipeline '{spec.pipeline}' "
                f"(supported: 'cr', 'tp', 'default')")

        ext = spec.field.dtype
        link = workdir / f"input.{ext}"
        workdir.mkdir(parents=True, exist_ok=True)
        if not link.exists():
            link.symlink_to(spec.field.path)

        config_args = [
            "-t", _DTYPE_FLAG[spec.field.dtype],
            "-l", spec.field.dim_arg,
            "-m", native_mode,
            "-e", repr(eb),
            "-s", lossless_mode,
        ]
        return Prepared(
            config_args=config_args,
            eb=eb,
            native_mode=native_mode,
            basis=basis,
            pipeline_ref=f"cuszhi:spline+{lossless_mode}",
            pipeline_path=None,
            pipeline_sha256=None,
        )

    def _link(self, spec: RunSpec, workdir: Path) -> Path:
        return workdir / f"input.{spec.field.dtype}"

    def _compressed_path(self, spec: RunSpec, workdir: Path) -> Path:
        return Path(str(self._link(spec, workdir)) + ".cusza")

    def _decompressed_raw(self, compressed: Path) -> Path:
        stem = compressed.name.removesuffix(".cusza")
        return compressed.parent / (stem + ".cuszx")

    def compress(self, spec: RunSpec, prep: Prepared, workdir: Path) -> CompressResult:
        link = self._link(spec, workdir)
        compressed = self._compressed_path(spec, workdir)
        log = workdir / "compress.log"

        argv = [self.cli, "-z", "-i", str(link), *prep.config_args]
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
        raw_out = self._decompressed_raw(compressed)
        final_out = workdir / "d.bin"
        log = workdir / "decompress.log"

        argv = [self.cli, "-x", "-i", str(compressed)]
        proc = run_cli(argv, log)
        if proc.returncode != 0:
            raise AdapterError(f"decompress failed (exit {proc.returncode}); see {log}")
        if not raw_out.exists():
            raise AdapterError(f"decompress produced no output at {raw_out}; see {log}")
        raw_out.rename(final_out)

        return DecompressResult(decompressed_path=final_out, raw_json={}, log_path=log)

    def benchmark(self, spec: RunSpec, prep: Prepared, n_runs: int, workdir: Path) -> BenchmarkResult:
        """cuSZ-Hi timing via -R time text table, N subprocess invocations each.

        The "(total)" row from -R time is CUDA-event-measured kernel time
        (sum of all pipeline stage events). File I/O (disk reads/writes) happens
        outside the CUDA event window so it does NOT contaminate the reported ms.
        cuszhi has no -S write2disk flag, but none is needed for timing accuracy.
        Comparable to fzgm and cusz CUDA-event timing.
        """
        link = self._link(spec, workdir)
        compressed = self._compressed_path(spec, workdir)
        log = workdir / "benchmark.log"

        compress_ms: list[float] = []
        decompress_ms: list[float] = []

        c_argv = [self.cli, "-z", "-i", str(link), *prep.config_args, "-R", "time"]
        d_argv = [self.cli, "-x", "-i", str(compressed), "-R", "time"]

        with open(log, "w") as fh:
            for i in range(n_runs):
                fh.write(f"\n# --- run {i} compress ---\n$ {' '.join(c_argv)}\n")
                proc = subprocess.run(c_argv, capture_output=True, text=True)
                fh.write(proc.stdout + proc.stderr)
                if proc.returncode != 0:
                    raise AdapterError(
                        f"benchmark compress run {i} failed "
                        f"(exit {proc.returncode}); see {log}")
                try:
                    compress_ms.append(_parse_total_ms(proc.stdout + proc.stderr))
                except AdapterError as e:
                    raise AdapterError(f"run {i}: {e}") from e

                # Decompress from the compressed file just written.
                fh.write(f"\n# --- run {i} decompress ---\n$ {' '.join(d_argv)}\n")
                proc = subprocess.run(d_argv, capture_output=True, text=True)
                fh.write(proc.stdout + proc.stderr)
                if proc.returncode != 0:
                    raise AdapterError(
                        f"benchmark decompress run {i} failed "
                        f"(exit {proc.returncode}); see {log}")
                try:
                    decompress_ms.append(_parse_total_ms(proc.stdout + proc.stderr))
                except AdapterError as e:
                    raise AdapterError(f"run {i}: {e}") from e

                # Clean up .cuszx left by decompress to avoid stale reads.
                raw = self._decompressed_raw(compressed)
                raw.unlink(missing_ok=True)

        return BenchmarkResult(
            compress_device_ms_all=compress_ms,
            decompress_device_ms_all=decompress_ms,
            compressed_bytes=compressed.stat().st_size,
            stages=[],
            native_quality=None,
            log_path=log,
        )
