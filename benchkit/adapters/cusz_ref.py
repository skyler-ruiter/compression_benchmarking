"""cuSZ reference adapter — wraps the `cusz` binary (psz project).

cuSZ differences from fzgm that shape this adapter:

  - Timing via -R time + CUDA events: executor.cc was patched to wrap
    psz_compress/decompress_float with cudaEventRecord pairs and emit a JSON
    line to stdout. benchmark() passes -R time, parses that JSON, and returns
    cuda_event_device_only ms — directly comparable to fzgm's timing.

  - In-process repeat (--repeat N): this build of cusz has been patched
    (local, unmerged — see ~/research/compressors/cuSZ git log) to loop the
    compress/decompress task N times inside one process, sharing one CUDA
    stream/context across reps, before exiting. benchmark() passes
    `--repeat n_runs` and makes ONE subprocess call per phase instead of
    looping n_runs subprocess calls itself — each rep prints its own JSON
    line, so N reps show up as N lines in one process's stdout. This
    replaced the old N-cold-subprocess loop, which measured a cold
    CUDA-context/clock-ramp penalty on every single rep — the same failure
    mode documented in docs/adapters/fzgm.md for FZGM itself.
    `warmup_reps` (dropped by `metrics.summarize_timing`) now discards the
    first (cold) rep, matching how FZGM's own single internal warmup rep is
    handled. See docs/adapters/cusz.md "Fairness caveat" for how the fix
    was root-caused: the naive first attempt (just hoisting the CUDA
    stream, matching cuSZ-Hi's fix) still segfaulted — the real bug was
    `psz_release_resource()` deleting `args->cli` out from under the
    caller on every rep (an aliased, not owned, pointer), fixed in
    executor.cc.

  - Output path not configurable: cusz always writes <input>.cusza in the same
    directory as the input. We symlink the input into the workdir so that the
    compressed file lands there. Decompressed output is <stem>.cuszx; we rename
    it to d.bin after decompression.

  - Supported modes: abs and rel_range (cusz calls the latter "r2r" — range-
    relative, same semantic as NOA). rel_maxabs is not supported by cusz and
    raises AdapterError.

See docs/adapters/cusz.md for the full contract.
"""
from __future__ import annotations

import json
import os
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
    for cand in (explicit, os.environ.get("CUSZ_CLI")):
        if cand:
            return cand
    found = shutil.which("cusz")
    if found:
        return found
    raise AdapterError(
        "cusz not found: set CUSZ_CLI or cli_path in the run entry. "
        "See scripts/env-bigred200.sh for the build path and how to uncomment it.")


class CuszAdapter(Adapter):
    name = "cusz"

    def __init__(self, variant: str = "cusz", cli_path: str | None = None):
        self.variant = variant
        self.cli = resolve_cli(cli_path)

    def is_available(self) -> bool:
        return Path(self.cli).exists() or shutil.which(self.cli) is not None

    def provenance(self) -> dict:
        return {"cli_path": self.cli, "name": "cusz",
                "timing_method": "cuda_event_device_only"}

    def prepare(self, spec: RunSpec, workdir: Path) -> Prepared:
        if spec.error_mode not in _MODE_MAP:
            raise AdapterError(
                f"cuSZ does not support error mode '{spec.error_mode}' "
                f"(supported: {sorted(_MODE_MAP)}). "
                f"rel_maxabs has no equivalent in the cusz CLI.")
        native_mode, basis = _MODE_MAP[spec.error_mode]
        eb = float(spec.error_bound)

        # cusz writes <input>.cusza next to the input file. Symlinking the data
        # file into the workdir makes the compressed output land in the workdir.
        ext = spec.field.dtype  # "f32" or "f64"
        link = workdir / f"input.{ext}"
        workdir.mkdir(parents=True, exist_ok=True)
        if not link.exists():
            link.symlink_to(spec.field.path)

        config_args = [
            "-t", _DTYPE_FLAG[spec.field.dtype],
            "-l", spec.field.dim_arg,
            "-m", native_mode,
            "-e", repr(eb),
        ]
        return Prepared(
            config_args=config_args,
            eb=eb,
            native_mode=native_mode,
            basis=basis,
            pipeline_ref="cusz:Lorenzo+Huffman",
            pipeline_path=None,
            pipeline_sha256=None,
        )

    # -- internal helpers ------------------------------------------------------

    def _link(self, spec: RunSpec, workdir: Path) -> Path:
        return workdir / f"input.{spec.field.dtype}"

    def _compressed_path(self, spec: RunSpec, workdir: Path) -> Path:
        return Path(str(self._link(spec, workdir)) + ".cusza")

    def _decompressed_raw(self, compressed: Path) -> Path:
        # cusz strips the last suffix (.cusza) then appends .cuszx
        stem = compressed.name.removesuffix(".cusza")
        return compressed.parent / (stem + ".cuszx")

    # -- interface -------------------------------------------------------------

    def compress(self, spec: RunSpec, prep: Prepared, workdir: Path) -> CompressResult:
        link = self._link(spec, workdir)
        compressed = self._compressed_path(spec, workdir)
        log = workdir / "compress.log"

        argv = [self.cli, "-z", "-i", str(link), *prep.config_args, "-R", "cr"]
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

        # --compare triggers a quality printout to stdout (logged); the harness
        # ignores it and computes quality independently from the output file.
        argv = [self.cli, "-x", "-i", str(compressed), "--compare", str(spec.field.path)]
        proc = run_cli(argv, log)
        if proc.returncode != 0:
            raise AdapterError(f"decompress failed (exit {proc.returncode}); see {log}")
        if not raw_out.exists():
            raise AdapterError(f"decompress produced no output at {raw_out}; see {log}")
        raw_out.rename(final_out)

        return DecompressResult(decompressed_path=final_out, raw_json={}, log_path=log)

    def _parse_all_device_ms(
        self, stdout: str, key: str, expected_n: int, phase: str, log: Path
    ) -> list[float]:
        """Extract every device_ms value from JSON lines in cusz stdout.

        With `--repeat N`, one process prints N such JSON lines (one per rep).
        """
        out = []
        for line in stdout.splitlines():
            line = line.strip()
            if line.startswith("{") and line.endswith("}"):
                try:
                    data = json.loads(line)
                    if key in data:
                        out.append(float(data[key]))
                except json.JSONDecodeError:
                    pass
        if len(out) != expected_n:
            raise AdapterError(
                f"benchmark {phase}: expected {expected_n} '{key}' lines in cusz "
                f"stdout, got {len(out)} — binary may not be built with device "
                f"timing (executor.cc patch) or --repeat support; see {log}")
        return out

    def benchmark(self, spec: RunSpec, prep: Prepared, n_runs: int, workdir: Path) -> BenchmarkResult:
        """CUDA-event device-only timing via one in-process --repeat n_runs call each.

        One subprocess per phase, each looping n_runs reps inside the binary
        (--repeat, local patch — see module docstring) sharing one CUDA
        stream/context across reps. The reported time spans only the GPU
        kernels (cudaEventRecord placed immediately before and after
        psz_compress/decompress_float), excluding PCIe H2D/D2H and file I/O —
        comparable to fzgm's cuda_event_device_only timing.

        -S write2disk skips disk writes during timing runs; the .cusza file from
        compress() is reused for all decompress timing runs.
        """
        link = self._link(spec, workdir)
        compressed = self._compressed_path(spec, workdir)
        log = workdir / "benchmark.log"

        c_argv = [self.cli, "-z", "-i", str(link), *prep.config_args,
                  "-S", "write2disk", "-R", "time", "--repeat", str(n_runs)]
        d_argv = [self.cli, "-x", "-i", str(compressed),
                  "-S", "write2disk", "-R", "time", "--repeat", str(n_runs)]

        with open(log, "w") as fh:
            fh.write(f"\n# --- compress (--repeat {n_runs}) ---\n$ {' '.join(c_argv)}\n")
            proc = subprocess.run(c_argv, capture_output=True, text=True)
            fh.write(proc.stdout + proc.stderr)
            if proc.returncode != 0:
                raise AdapterError(
                    f"benchmark compress failed (exit {proc.returncode}); see {log}")
            compress_ms = self._parse_all_device_ms(
                proc.stdout, "compress_device_ms", n_runs, "compress", log)

            fh.write(f"\n# --- decompress (--repeat {n_runs}) ---\n$ {' '.join(d_argv)}\n")
            proc = subprocess.run(d_argv, capture_output=True, text=True)
            fh.write(proc.stdout + proc.stderr)
            if proc.returncode != 0:
                raise AdapterError(
                    f"benchmark decompress failed (exit {proc.returncode}); see {log}")
            decompress_ms = self._parse_all_device_ms(
                proc.stdout, "decompress_device_ms", n_runs, "decompress", log)

        return BenchmarkResult(
            compress_device_ms_all=compress_ms,
            decompress_device_ms_all=decompress_ms,
            compressed_bytes=compressed.stat().st_size,
            stages=[],
            native_quality=None,
            log_path=log,
        )
