"""zfp reference adapter — wraps the `zfp` CLI.

zfp quirks that shape this adapter:

  - **CPU-only for error-bounded modes.** zfp's CUDA execution policy
    (`-x cuda`) only implements *fixed-rate* compression (confirmed
    empirically: `-x cuda -a <tol>` fails with "compression failed"; `-x cuda
    -r <rate>` works). Fixed-rate has no error-bound semantics compatible with
    this harness's abs/rel_range/rel_maxabs model, so this adapter always uses
    `-x serial` (CPU). zfp is included as a CPU quality/ratio baseline, not a
    GPU-throughput peer — see docs/adapters/zfp.md.

  - **No native relative mode.** zfp only offers `-a <tolerance>` (absolute).
    rel_range / rel_maxabs are emulated: the adapter reads the input file
    itself (`read_range_stats`), computes range or max|x|, and passes
    `eb * range` (or `eb * maxabs`) as the absolute tolerance. The harness's
    own independent quality check reads the same file and computes the same
    statistic, so the emulated bound and eb_ok agree.

  - **No self-reported timing.** zfp prints a one-line size/rate summary, no
    elapsed time. Timing is wall-clock, measured externally around the
    subprocess call (`time.perf_counter()`) — includes process startup, not
    just the compression kernel. Treat as informational, not a precise
    device-time comparison.

  - **Self-describing compressed files** (`-h`): the adapter always passes
    `-h` so decompress needs only `-z <compressed> -o <output>` — type, dims,
    and mode are read back from the embedded header.

See docs/adapters/zfp.md for the full contract.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import time
from pathlib import Path

from .base import (Adapter, AdapterError, BenchmarkResult, CompressResult,
                   DecompressResult, Prepared, RunSpec, run_cli, read_range_stats)

_DTYPE_FLAG = {"f32": "-f", "f64": "-d"}


def resolve_cli(explicit: str | None = None) -> str:
    for cand in (explicit, os.environ.get("ZFP_CLI")):
        if cand:
            return cand
    found = shutil.which("zfp")
    if found:
        return found
    raise AdapterError(
        "zfp not found: set ZFP_CLI or cli_path in the run entry. "
        "See scripts/env-jetstream2.sh for the build path.")


def _dim_args(dims: list[int]) -> list[str]:
    flag = {1: "-1", 2: "-2", 3: "-3", 4: "-4"}.get(len(dims))
    if flag is None:
        raise AdapterError(f"zfp supports 1-4D data; got {len(dims)}D: {dims}")
    return [flag, *[str(d) for d in dims]]


class ZfpAdapter(Adapter):
    """zfp adapter — CPU (serial), abs native + emulated rel_range/rel_maxabs."""

    name = "zfp"

    def __init__(self, variant: str = "zfp", cli_path: str | None = None):
        self.variant = variant
        self.cli = resolve_cli(cli_path)

    def is_available(self) -> bool:
        return Path(self.cli).exists() or shutil.which(self.cli) is not None

    def provenance(self) -> dict:
        return {
            "cli_path": self.cli,
            "name": "zfp",
            "timing_method": "cpu_wall_clock_external",
            "timing_note": (
                "zfp's CUDA backend only supports fixed-rate mode (no error-bound "
                "support), so this adapter runs '-x serial' (CPU) for all error-"
                "bounded modes. Timing is measured externally around the "
                "subprocess (includes process startup) — not comparable to the "
                "GPU adapters' device_ms."
            ),
            "execution": "serial (CPU)",
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
            raise AdapterError(f"zfp: unsupported error mode '{spec.error_mode}'.")

        workdir.mkdir(parents=True, exist_ok=True)
        config_args = [
            _DTYPE_FLAG[spec.field.dtype],
            *_dim_args(spec.field.dims),
            "-a", repr(abs_eb),
            "-x", "serial",
            "-h",
        ]
        return Prepared(
            config_args=config_args,
            eb=eb,
            native_mode="abs" if spec.error_mode == "abs" else f"abs-emulated({spec.error_mode})",
            basis=basis,
            pipeline_ref="zfp:default",
            pipeline_path=None,
            pipeline_sha256=None,
        )

    def compress(self, spec: RunSpec, prep: Prepared, workdir: Path) -> CompressResult:
        compressed = workdir / "c.zfp"
        log = workdir / "compress.log"

        argv = [self.cli, "-i", str(spec.field.path), "-z", str(compressed), *prep.config_args]
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

        argv = [self.cli, "-z", str(compressed), "-o", str(decompressed), "-h", "-x", "serial"]
        proc = run_cli(argv, log)
        if proc.returncode != 0:
            raise AdapterError(f"decompress failed (exit {proc.returncode}); see {log}")
        if not decompressed.exists():
            raise AdapterError(f"decompress produced no output at {decompressed}; see {log}")

        return DecompressResult(decompressed_path=decompressed, raw_json={}, log_path=log)

    def benchmark(self, spec: RunSpec, prep: Prepared, n_runs: int, workdir: Path) -> BenchmarkResult:
        """N subprocess calls per phase, timed externally with perf_counter().

        No in-process repeat support in the zfp CLI; process-launch overhead is
        included in every measurement (see provenance() timing_note).
        """
        compressed = workdir / "c.zfp"
        decompressed = workdir / "d_bench.bin"
        log = workdir / "benchmark.log"

        c_argv = [self.cli, "-i", str(spec.field.path), "-z", str(compressed), *prep.config_args]
        d_argv = [self.cli, "-z", str(compressed), "-o", str(decompressed), "-h", "-x", "serial"]

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
