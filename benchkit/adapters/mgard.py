"""MGARD reference adapter — wraps the `mgard-x` CLI (GPU, CUDA device).

MGARD quirks that shape this adapter:

  - **Dims are slowest-first** (`-dim N d_slow ... d_fast`), the opposite of
    FieldSpec.dims (fastest-first) — the adapter reverses them.

  - **Native "rel" is not range- or maxabs-relative.** MGARD's `-em rel` scales
    the tolerance by an s-norm of the data (`-s <smoothness>` selects which
    norm; see `CalculateNorm` in Compressor.hpp), not `max-min` or `max|x|`.
    That has no equivalent to this harness's canonical rel_range/rel_maxabs.
    Both are instead emulated the same way as the zfp adapter: read the input
    file (`read_range_stats`), compute range or maxabs, and pass
    `-em abs -e (eb * range_or_maxabs)`. `abs` mode is passed straight through
    natively.

  - **`-s inf`, not `-s 0`.** The smoothness parameter also controls how the
    ABS tolerance is distributed across decomposition levels (see
    `CalcQuantizers` in Quantization/LinearQuantization.hpp), independent of
    `-em`. `-s 0` allocates levels for a smoothness-weighted L2-ish criterion
    and does NOT bound the pointwise max error — confirmed empirically: with
    `-s 0` the realized max|error| ran ~1.7x over the nominal abs bound on
    CESM/HURR test fields (MGARD's own "Absolute L_2 error" line was
    satisfied, but that's a different norm than this harness's max-error
    eb_ok check). `-s inf` selects the branch that allocates
    `abs_tol / ((l_target+1) * (1+3^D))` per level, which does bound the
    pointwise max error — MGARD's own report then prints "Absolute L_inf
    error" and satisfies it with margin. This adapter always passes `-s inf`.

  - **No in-process repeat.** Every subprocess call pays a fresh CUDA context
    init ("Prepare device environment", tens to hundreds of ms here) — there
    is no `--repeat` equivalent to hide it (contrast fzgm.md / cusz.md, where
    that cold-start problem was fixed with a source patch; not attempted here).
    The adapter reports "Aggregated low-level compression/decompression time"
    (kernel + memory work, excludes context init and I/O framing) rather than
    "High-level" time (which includes both) — the least-contaminated number
    mgard-x prints, but still expect more run-to-run variance than the
    in-process-repeat adapters (cuSZ, FZGM, FZ-GPU).

  - **compress() always runs an internal decompress-for-verification** (to
    print its own L2/PSNR line), which prints a second, later set of
    decompression timing lines in the same stdout. The adapter takes the
    *first* "Aggregated low-level compression time" match, which precedes
    that verification pass, so this does not contaminate the compress timing.

  - `-l huffman|huffman-lz4|huffman-zstd` selects the lossless back end; the
    `pipeline:` field selects it (`default` -> `huffman`).

See docs/adapters/mgard.md for the full contract.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path

from .base import (Adapter, AdapterError, BenchmarkResult, CompressResult,
                   DecompressResult, Prepared, RunSpec, run_cli, read_range_stats)

_DTYPE_FLAG = {"f32": "s", "f64": "d"}

_LOSSLESS_MODES = {"huffman", "huffman-lz4", "huffman-zstd"}

_TIME_RE = re.compile(
    r"Aggregated low-level (compression|decompression) time:\s*"
    r"([0-9]+\.?[0-9]*(?:[eE][+-]?[0-9]+)?)\s*s")


def resolve_cli(explicit: str | None = None) -> str:
    for cand in (explicit, os.environ.get("MGARD_CLI")):
        if cand:
            return cand
    found = shutil.which("mgard-x")
    if found:
        return found
    raise AdapterError(
        "mgard-x not found: set MGARD_CLI or cli_path in the run entry. "
        "Also ensure MGARD's install lib/ dir is on LD_LIBRARY_PATH "
        "(libmgard.so etc.) — see scripts/env-jetstream2.sh.")


def _dim_args(dims: list[int]) -> list[str]:
    slow_to_fast = list(reversed(dims))
    return [str(len(dims)), *[str(d) for d in slow_to_fast]]


def _parse_time_ms(stdout: str, phase: str) -> float:
    for m in _TIME_RE.finditer(stdout):
        if m.group(1) == phase:
            return float(m.group(2)) * 1000.0
    raise AdapterError(
        f"No 'Aggregated low-level {phase} time' line found in mgard-x output "
        f"(need -v 2 or higher). Check the log.")


class MgardAdapter(Adapter):
    """MGARD adapter — GPU (cuda device), abs native + emulated rel_range/rel_maxabs."""

    name = "mgard"

    def __init__(self, variant: str = "mgard", cli_path: str | None = None):
        self.variant = variant
        self.cli = resolve_cli(cli_path)

    def is_available(self) -> bool:
        return Path(self.cli).exists() or shutil.which(self.cli) is not None

    def provenance(self) -> dict:
        return {
            "cli_path": self.cli,
            "name": "mgard-x",
            "timing_method": "wall_clock_self_reported_aggregated_low_level",
            "timing_note": (
                "'Aggregated low-level compression/decompression time' from mgard-x "
                "-v 2 (excludes CUDA context init + serialization framing, but each "
                "subprocess call still pays that init cost since there is no "
                "in-process repeat — expect more variance than fzgm/cusz)."
            ),
            "device": "cuda",
        }

    def _parse_pipeline(self, pipeline: str) -> str:
        if pipeline in ("default", ""):
            return "huffman"
        if pipeline not in _LOSSLESS_MODES:
            raise AdapterError(
                f"MGARD: unknown pipeline '{pipeline}'. Valid: default, "
                f"{sorted(_LOSSLESS_MODES)}.")
        return pipeline

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
            raise AdapterError(f"MGARD: unsupported error mode '{spec.error_mode}'.")

        lossless = self._parse_pipeline(spec.pipeline)
        workdir.mkdir(parents=True, exist_ok=True)

        config_args = [
            "-dt", _DTYPE_FLAG[spec.field.dtype],
            "-dim", *_dim_args(spec.field.dims),
            "-em", "abs",
            "-e", repr(abs_eb),
            "-s", "inf",
            "-l", lossless,
            "-d", "cuda",
            "-v", "2",
        ]
        return Prepared(
            config_args=config_args,
            eb=eb,
            native_mode="abs" if spec.error_mode == "abs" else f"abs-emulated({spec.error_mode})",
            basis=basis,
            pipeline_ref=f"mgard:{lossless}",
            pipeline_path=None,
            pipeline_sha256=None,
        )

    def compress(self, spec: RunSpec, prep: Prepared, workdir: Path) -> CompressResult:
        compressed = workdir / "c.mgard"
        log = workdir / "compress.log"

        argv = [self.cli, "-z", "-i", str(spec.field.path), "-o", str(compressed), *prep.config_args]
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

        argv = [self.cli, "-x", "-i", str(compressed), "-o", str(decompressed), "-d", "cuda", "-v", "2"]
        proc = run_cli(argv, log)
        if proc.returncode != 0:
            raise AdapterError(f"decompress failed (exit {proc.returncode}); see {log}")
        if not decompressed.exists():
            raise AdapterError(f"decompress produced no output at {decompressed}; see {log}")

        return DecompressResult(decompressed_path=decompressed, raw_json={}, log_path=log)

    def benchmark(self, spec: RunSpec, prep: Prepared, n_runs: int, workdir: Path) -> BenchmarkResult:
        """N separate subprocess calls per phase (no in-process repeat available).

        Each call pays a fresh CUDA context init; "Aggregated low-level" timing
        excludes that init but the wall-clock cost is still incurred per rep —
        see module docstring.
        """
        compressed = workdir / "c.mgard"
        decompressed = workdir / "d_bench.bin"
        log = workdir / "benchmark.log"

        c_argv = [self.cli, "-z", "-i", str(spec.field.path), "-o", str(compressed), *prep.config_args]
        d_argv = [self.cli, "-x", "-i", str(compressed), "-o", str(decompressed), "-d", "cuda", "-v", "2"]

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

                if not compressed.exists():
                    raise AdapterError(
                        f"benchmark compress run {i} produced no output; see {log}")

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
