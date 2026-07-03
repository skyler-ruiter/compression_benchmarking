"""cuSZp v2 and v3 reference adapter.

cuSZp differences from cuSZ that shape this adapter:

  - Single-run compress+decompress: one invocation does both phases, with
    10 GPU warmup iterations and 1 timed iteration each. benchmark() runs
    N subprocesses to collect N timing values per phase.

  - Timing via CUDA events (TimingGPU / cudaEventElapsedTime → ms):
    Printed as "cuSZp compression   end-to-end speed: X GB/s" where X is
    actually MiB/ms (mislabeled). The adapter recovers device_ms as:
      device_ms = (original_bytes / 1024 / 1024) / X

  - rel mode = range-relative (= canonical rel_range): cuSZp internally
    computes max−min and multiplies the error bound. The adapter passes
    the fractional bound with basis="range".

  - Output paths are optional (-x compressed -o decompressed). compress()
    writes both; decompress() returns the file already written by compress().
    benchmark() omits -x and -o to avoid file I/O overhead per timing run.

  - Version differences:
    v2 CLI: -i input -t f32|f64 -m plain|outlier -eb abs|rel bound
            [-x compressed] [-o decompressed]
    v3 CLI: same + -d 1|2|3 [dz dy dx] for multi-dimensional processing.
    The pipeline string encodes the encoding mode:
      "plain"        → -m plain  (v2: always 1D; v3: 1D by default)
      "outlier"      → -m outlier
      "fixed"        → -m fixed  (v3 only)
      "plain:2d"     → -m plain -d 2 dz dy dx  (v3 only, dims from FieldSpec)
      "plain:3d"     → -m plain -d 3 dz dy dx  (v3 only, 3D data required)

  See docs/adapters/cuszp.md for the full contract.
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
    "rel_range": ("rel", "range"),
}

_DTYPE_FLAG = {"f32": "f32", "f64": "f64"}

# Valid encoding modes per version
_MODES_V2 = {"plain", "outlier"}
_MODES_V3 = {"plain", "outlier", "fixed"}


def _resolve_cli(env_var: str, explicit: str | None) -> str:
    for cand in (explicit, os.environ.get(env_var)):
        if cand:
            return cand
    found = shutil.which("cuSZp")
    if found:
        return found
    raise AdapterError(
        f"cuSZp not found: set {env_var} or cli_path in the run entry. "
        "See scripts/env-bigred200.sh for build paths.")


def _parse_speed(stdout: str, phase: str) -> float:
    """Extract the mislabeled 'GB/s' (actually MiB/ms) from cuSZp output.

    cuSZp pads "compression" with extra spaces so its own printed table
    aligns with "decompression" — the whitespace between the phase name and
    "end-to-end" is not constant, so match on it loosely.
    """
    pattern = re.compile(
        rf"cuSZp\s+{re.escape(phase)}\s+end-to-end speed:\s*"
        rf"([0-9]+\.?[0-9]*(?:e[+-]?[0-9]+)?)\s*GB/s",
        re.IGNORECASE)
    for line in stdout.splitlines():
        m = pattern.search(line)
        if m:
            return float(m.group(1))
    raise AdapterError(
        f"No 'cuSZp {phase} end-to-end speed:' line found in cuSZp output. "
        "Is this the cuSZp binary? Check the log.")


def _speed_to_ms(speed_mib_per_ms: float, original_bytes: int) -> float:
    return (original_bytes / (1024.0 * 1024.0)) / speed_mib_per_ms


class CuszpAdapter(Adapter):
    """cuSZp adapter — version 2 or 3, selected by the version parameter."""

    name = "cuszp"

    def __init__(self, version: int = 2, variant: str = "cuszp",
                 cli_path: str | None = None):
        if version not in (2, 3):
            raise AdapterError(f"CuszpAdapter: version must be 2 or 3, got {version}")
        self.version = version
        self.variant = variant
        env_var = f"CUSZP{version}_CLI"
        self.cli = _resolve_cli(env_var, cli_path)
        self._valid_modes = _MODES_V2 if version == 2 else _MODES_V3

    def is_available(self) -> bool:
        return Path(self.cli).exists() or shutil.which(self.cli) is not None

    def provenance(self) -> dict:
        return {
            "cli_path": self.cli,
            "name": f"cuszp{self.version}",
            "timing_method": "cuda_events_device_only",
            "timing_note": (
                "1 timed run after 10 GPU warmup iterations per subprocess call. "
                "Printed as 'GB/s' but actually MiB/ms; adapter recovers ms."
            ),
        }

    def _parse_pipeline(self, pipeline: str) -> tuple[str, list[str]]:
        """Return (encoding_mode, extra_dim_args).

        Pipeline syntax: "<mode>[:<proc_dim>]" where proc_dim in {1d, 2d, 3d}.
        For v2, proc_dim is ignored (v2 has no -d flag).
        """
        parts = pipeline.split(":", 1)
        enc_mode = parts[0].strip()
        if enc_mode not in self._valid_modes:
            raise AdapterError(
                f"cuSZp v{self.version}: unknown encoding mode '{enc_mode}' in "
                f"pipeline '{pipeline}'. Valid: {sorted(self._valid_modes)}")
        dim_args: list[str] = []
        if self.version == 3:
            proc_dim_str = parts[1].lower() if len(parts) > 1 else "1d"
            if proc_dim_str == "1d":
                dim_args = ["-d", "1"]
            elif proc_dim_str in ("2d", "3d"):
                dim_args = []   # filled in prepare() once we have field dims
                # Store the requested proc_dim for use in prepare()
                self._pending_proc_dim = int(proc_dim_str[0])
            else:
                raise AdapterError(
                    f"cuSZp v3: unknown proc_dim '{proc_dim_str}' in pipeline "
                    f"'{pipeline}'. Use 1d, 2d, or 3d.")
        return enc_mode, dim_args

    def prepare(self, spec: RunSpec, workdir: Path) -> Prepared:
        if spec.error_mode not in _MODE_MAP:
            raise AdapterError(
                f"cuSZp does not support error mode '{spec.error_mode}' "
                f"(supported: {sorted(_MODE_MAP)}). "
                f"rel_maxabs has no equivalent in the cuSZp CLI.")
        native_mode, basis = _MODE_MAP[spec.error_mode]
        eb = float(spec.error_bound)

        enc_mode, dim_args = self._parse_pipeline(spec.pipeline)

        # For v3 multi-dim, build -d N dz dy dx from FieldSpec dims.
        if self.version == 3 and not dim_args:
            pd = getattr(self, "_pending_proc_dim", 1)
            dims = spec.field.dims  # fast-to-slow: [x] or [x, y] or [x, y, z]
            if len(dims) < pd:
                raise AdapterError(
                    f"cuSZp v3 {pd}D processing requested but field has "
                    f"{len(dims)} dims: {dims}")
            if pd == 2:
                # -d 2 dz dy dx: for 2D data, set dz=1, dy=dims[1], dx=dims[0]
                dx, dy = dims[0], dims[1] if len(dims) > 1 else 1
                dim_args = ["-d", "2", "1", str(dy), str(dx)]
            elif pd == 3:
                dx, dy, dz = dims[0], dims[1], dims[2]
                dim_args = ["-d", "3", str(dz), str(dy), str(dx)]

        workdir.mkdir(parents=True, exist_ok=True)
        config_args = [
            "-i", str(spec.field.path),
            "-t", _DTYPE_FLAG[spec.field.dtype],
            "-m", enc_mode,
            *dim_args,
            "-eb", native_mode, repr(eb),
        ]
        pipeline_label = f"cuszp{self.version}:{enc_mode}"
        return Prepared(
            config_args=config_args,
            eb=eb,
            native_mode=native_mode,
            basis=basis,
            pipeline_ref=pipeline_label,
            pipeline_path=None,
            pipeline_sha256=None,
        )

    def compress(self, spec: RunSpec, prep: Prepared, workdir: Path) -> CompressResult:
        """Run cuSZp compress+decompress; save both files for quality metrics."""
        compressed = workdir / "c.cuszp"
        decompressed = workdir / "d.bin"
        log = workdir / "compress.log"

        argv = [self.cli, *prep.config_args, "-x", str(compressed), "-o", str(decompressed)]
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
        """Return decompressed file written by compress(). cuSZp has no decompress-only mode."""
        decompressed = workdir / "d.bin"
        if not decompressed.exists():
            raise AdapterError(
                f"Decompressed file not found at {decompressed}. "
                "compress() must be called first — cuSZp does both phases in one run.")
        return DecompressResult(decompressed_path=decompressed, raw_json={}, log_path=workdir / "compress.log")

    def benchmark(self, spec: RunSpec, prep: Prepared, n_runs: int, workdir: Path) -> BenchmarkResult:
        """N subprocess calls, each: 10 GPU warmup + 1 timed compress + 1 timed decompress.

        No -x/-o: cuSZp skips file writes when paths are omitted, so there
        is no file I/O overhead during timing. The compressed file from
        compress() is still present in workdir for reference.

        The printed speed is (MiB / ms) mislabeled 'GB/s'; the adapter
        recovers device_ms = (original_MiB) / speed.
        """
        compressed_ref = workdir / "c.cuszp"
        log = workdir / "benchmark.log"
        original_bytes = spec.field.original_bytes

        # Strip -x and -o from config_args for benchmark (no file I/O needed)
        bench_args = []
        skip_next = False
        for tok in prep.config_args:
            if skip_next:
                skip_next = False
                continue
            if tok in ("-x", "-o"):
                skip_next = True
                continue
            bench_args.append(tok)

        argv = [self.cli, *bench_args]

        compress_ms: list[float] = []
        decompress_ms: list[float] = []

        with open(log, "w") as fh:
            for i in range(n_runs):
                fh.write(f"\n# --- run {i} ---\n$ {' '.join(argv)}\n")
                proc = subprocess.run(argv, capture_output=True, text=True)
                fh.write(proc.stdout + proc.stderr)
                if proc.returncode != 0:
                    raise AdapterError(
                        f"benchmark run {i} failed "
                        f"(exit {proc.returncode}); see {log}")
                try:
                    c_spd = _parse_speed(proc.stdout, "compression")
                    d_spd = _parse_speed(proc.stdout, "decompression")
                except AdapterError as e:
                    raise AdapterError(f"run {i}: {e}") from e
                compress_ms.append(_speed_to_ms(c_spd, original_bytes))
                decompress_ms.append(_speed_to_ms(d_spd, original_bytes))

        return BenchmarkResult(
            compress_device_ms_all=compress_ms,
            decompress_device_ms_all=decompress_ms,
            compressed_bytes=compressed_ref.stat().st_size if compressed_ref.exists() else 0,
            stages=[],
            native_quality=None,
            log_path=log,
        )
