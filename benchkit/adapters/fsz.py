"""FSZ reference adapter (SC'26, Jiajun Huang).

FSZ is the compressor whose prediction stage FZGM reimplements as
`AdaptiveLorenzoStage` (written from the paper before this source existed), so
this adapter's main job is to make the `fsz.toml` preset directly comparable
against the thing it was reconstructed from. Notes that shape it:

  - **Single-run compress+decompress.** With neither -z nor -x, `fsz -i data`
    does a full in-memory round trip and reports compressed size, both device
    times, and its own quality check, writing nothing. benchmark() uses that
    form; compress()/decompress() use -z/-x to produce artifacts.

  - **Timing is 3 warmup iterations, then ONE timed launch** per phase
    (WARMUP_ITERS in tools/fsz.cu), via a cudaEvent pair around the kernel.
    Unlike cuSZp's 100-launch internal mean, each subprocess yields a genuine
    single-shot measurement, so the cv across the harness's N subprocess calls
    is a real variance estimate rather than a deflated one.

  - **Device-only, and there is very little host work to miss.** FSZ compresses
    with a single fused kernel into one contiguous device buffer, so unlike
    FZGM's split-mode pipelines there is no host-side archive assembly outside
    the event bracket (see D33). Nothing to add, so no host_ms is reported.

  - **`-eb rel` is range-relative** (the CLI states "a fraction of (max - min)"),
    which is the canonical `rel_range` — the same basis as FZGM `NOA` and cuSZ
    `REL`. `rel_maxabs` has no equivalent and is rejected.

  - **CR basis: the adapter reports the CONTAINER size.** `-z -o` writes a
    self-describing `.fsz` file whose header is a fixed 56 bytes
    (fsz_file_format.hpp), whereas the `--csv` line reports the bare bitstream.
    The harness computes CR from the file on disk, so a cell's CR includes the
    56 bytes exactly as an FZGM cell's includes the .fzm header. At corpus
    sizes the difference is under 1e-5 relative, but keeping the file as the
    source of truth avoids a silent per-tool CR convention mismatch.

  - **No pipeline knobs.** FSZ is deliberately untunable ("nothing to tune" —
    no modes, no flags to sweep), so `pipeline` must be the single value
    `default`. The field exists only because RunSpec requires it.

  See docs/adapters/fsz.md for the full contract.
"""
from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

from .base import (Adapter, AdapterError, BenchmarkResult, CompressResult,
                   DecompressResult, Prepared, RunSpec, run_cli)

# canonical mode -> (native -eb keyword, harness eb basis)
_MODE_MAP = {
    "abs":       ("abs", "abs"),
    "rel_range": ("rel", "range"),
}

_DTYPE_FLAG = {"f32": "f32", "f64": "f64"}

# Fixed container header, fsz_file_format.hpp (static_assert'd there at 56).
_FSZ_HEADER_BYTES = 56

_CSV_FIELDS = ("tag", "path", "nele", "dims", "eb_mode", "eb_value", "abs_eb",
               "cmp_bytes", "cr", "compress_ms", "compress_gbs",
               "decompress_ms", "decompress_gbs", "max_err", "status")

# tools/fsz_hosttime — the host-wall-clock harness (see docs/adapters/fsz.md).
# The stock CLI reports device time only; without this, an FSZ row carries no
# host figure and the D33 host-over-device audit cannot be done on it.
_HOSTCSV_FIELDS = ("tag", "path", "nele", "cmp_bytes", "abs_eb", "max_err",
                   "compress_dev_ms", "compress_host_ms",
                   "decompress_dev_ms", "decompress_host_ms",
                   "compress_dev_gbs", "compress_host_gbs",
                   "decompress_dev_gbs", "decompress_host_gbs",
                   "compress_h_over_d", "decompress_h_over_d", "status")


def _resolve_cli(explicit: str | None) -> str:
    for cand in (explicit, os.environ.get("FSZ_CLI")):
        if cand:
            return cand
    found = shutil.which("fsz")
    if found:
        return found
    raise AdapterError(
        "FSZ not found: set FSZ_CLI or cli_path in the run entry. "
        "Build with: cmake -S . -B build -DCMAKE_CUDA_ARCHITECTURES=90 "
        "&& cmake --build build -j")


def _resolve_hosttime_cli(explicit: str | None) -> str | None:
    """Locate the optional host-timing harness; None means fall back to the CLI."""
    for cand in (explicit, os.environ.get("FSZ_HOSTTIME_CLI")):
        if cand and Path(cand).exists():
            return cand
    found = shutil.which("fsz_hosttime")
    if found:
        return found
    # Alongside the stock binary, where build-fsz-hosttime.sh puts it.
    if explicit is None:
        sibling = Path(__file__).resolve().parents[2] / "tools" / "fsz_hosttime" / "fsz_hosttime"
        if sibling.exists():
            return str(sibling)
    return None


def _parse_hostcsv(stdout: str) -> dict:
    for line in stdout.splitlines():
        if not line.startswith("hostcsv,"):
            continue
        parts = line.rstrip("\n").split(",")
        if len(parts) != len(_HOSTCSV_FIELDS):
            raise AdapterError(
                f"fsz_hosttime line has {len(parts)} fields, expected "
                f"{len(_HOSTCSV_FIELDS)}: {line!r}")
        return dict(zip(_HOSTCSV_FIELDS, parts))
    raise AdapterError(
        "no 'hostcsv,' line in fsz_hosttime output — stale binary? Check the log.")


def _parse_csv(stdout: str) -> dict:
    """Parse the single `csv,...` line emitted by --csv.

    Emitted only on the round-trip path, after FSZ's own bound check; the
    trailing `status` field is that check's verdict and is surfaced so a
    reference tool failing its own contract cannot be silently averaged in.
    """
    for line in stdout.splitlines():
        if not line.startswith("csv,"):
            continue
        parts = line.rstrip("\n").split(",")
        if len(parts) != len(_CSV_FIELDS):
            raise AdapterError(
                f"FSZ csv line has {len(parts)} fields, expected "
                f"{len(_CSV_FIELDS)}: {line!r}")
        return dict(zip(_CSV_FIELDS, parts))
    raise AdapterError(
        "no 'csv,' line in FSZ output — was --csv passed, and is this the "
        "fsz binary? Check the log.")


class FszAdapter(Adapter):
    """FSZ reference adapter."""

    name = "fsz"

    def __init__(self, variant: str = "fsz", cli_path: str | None = None,
                 hosttime_cli: str | None = None):
        self.variant = variant
        self.cli = _resolve_cli(cli_path)
        self.hosttime_cli = _resolve_hosttime_cli(hosttime_cli)

    def is_available(self) -> bool:
        return Path(self.cli).exists() or shutil.which(self.cli) is not None

    def provenance(self) -> dict:
        version = "unknown"
        try:
            proc = subprocess.run([self.cli, "--help"], capture_output=True,
                                  text=True, timeout=30)
            first = proc.stdout.splitlines()[0] if proc.stdout else ""
            if first.startswith("FSZ "):
                version = first.split()[1]
        except Exception:
            pass
        return {
            "cli_path": self.cli,
            "name": "fsz",
            "version": version,
            "hosttime_cli_path": self.hosttime_cli or "",
            "timing_method": (
                "cuda_events_plus_host_wall"
                if self.hosttime_cli else "cuda_events_device_only"),
            "timing_note": (
                "One timed launch per phase after 3 GPU warmup iterations, per "
                "subprocess call. Single fused kernel writing one contiguous "
                "device buffer, so no host-side assembly falls outside the "
                "event bracket." +
                (" Host wall time is measured by tools/fsz_hosttime around "
                 "fsz::compress + cudaStreamSynchronize, the same bracket "
                 "FZGM's 'Host elapsed' uses; device events recorded inside "
                 "the same iteration for both f32 and f64."
                 if self.hosttime_cli else
                 " NO HOST TIME: tools/fsz_hosttime not built, so the stock "
                 "CLI (device-only) was used — see docs/adapters/fsz.md.")
            ),
        }

    def _dim_args(self, spec: RunSpec) -> list[str]:
        """FSZ takes -d in C order, D1 slowest-varying; FieldSpec is fast-to-slow."""
        dims = list(spec.field.dims)
        if not dims:
            return []
        return ["-d", *[str(d) for d in reversed(dims)]]

    def prepare(self, spec: RunSpec, workdir: Path) -> Prepared:
        if spec.error_mode not in _MODE_MAP:
            raise AdapterError(
                f"FSZ does not support error mode '{spec.error_mode}' "
                f"(supported: {sorted(_MODE_MAP)}). FSZ's -eb rel is "
                f"range-relative; rel_maxabs has no equivalent in its CLI.")
        pipeline = (spec.pipeline or "default").strip()
        if pipeline != "default":
            raise AdapterError(
                f"FSZ takes no pipeline options (got '{pipeline}'); it exposes "
                f"no modes or tuning flags by design. Use pipeline: default.")

        native_mode, basis = _MODE_MAP[spec.error_mode]
        eb = float(spec.error_bound)

        workdir.mkdir(parents=True, exist_ok=True)
        config_args = [
            "-i", str(spec.field.path),
            "-t", _DTYPE_FLAG[spec.field.dtype],
            *self._dim_args(spec),
            "-eb", native_mode, repr(eb),
        ]
        return Prepared(
            config_args=config_args,
            eb=eb,
            native_mode=native_mode,
            basis=basis,
            pipeline_ref="fsz:default",
            pipeline_path=None,
            pipeline_sha256=None,
        )

    def compress(self, spec: RunSpec, prep: Prepared, workdir: Path) -> CompressResult:
        compressed = workdir / "c.fsz"
        log = workdir / "compress.log"

        argv = [self.cli, "-z", *prep.config_args, "-o", str(compressed)]
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
        """A .fsz container is self-describing: -x needs no type, dims or bound."""
        decompressed = workdir / "d.bin"
        log = workdir / "decompress.log"

        argv = [self.cli, "-x", "-i", str(compressed), "-o", str(decompressed)]
        proc = run_cli(argv, log)
        if proc.returncode != 0:
            raise AdapterError(f"decompress failed (exit {proc.returncode}); see {log}")
        if not decompressed.exists():
            raise AdapterError(f"decompress produced no output at {decompressed}; see {log}")

        return DecompressResult(decompressed_path=decompressed, raw_json={}, log_path=log)

    def benchmark(self, spec: RunSpec, prep: Prepared, n_runs: int,
                  workdir: Path) -> BenchmarkResult:
        """N subprocess calls of the write-nothing round trip.

        No -z/-x and no -o: FSZ compresses and decompresses in memory, so no
        file I/O is inside the measurement.

        Uses `tools/fsz_hosttime` when it is built, which reports host
        wall time beside the device events (`-r 1`, so each subprocess still
        contributes ONE launch and the cv across n_runs stays a genuine
        cross-process variance estimate — same shape as the stock-CLI path).
        Falls back to the stock CLI, device-only, when the harness is absent.
        """
        log = workdir / "benchmark.log"
        use_host = self.hosttime_cli is not None
        if use_host:
            argv = [self.hosttime_cli, *prep.config_args, "-r", "1"]
        else:
            argv = [self.cli, *prep.config_args, "--csv"]

        compress_ms: list[float] = []
        decompress_ms: list[float] = []
        compress_host_ms: list[float] = []
        decompress_host_ms: list[float] = []
        cmp_bytes = 0
        native_quality: dict | None = None

        with open(log, "w") as fh:
            for i in range(n_runs):
                fh.write(f"\n# --- run {i} ---\n$ {' '.join(argv)}\n")
                proc = subprocess.run(argv, capture_output=True, text=True)
                fh.write(proc.stdout + proc.stderr)
                # Exit 1 means FSZ's own bound check failed; that is a result to
                # record, not a harness error, so only exit >= 2 (usage/file) raises.
                if proc.returncode >= 2:
                    raise AdapterError(
                        f"benchmark run {i} failed "
                        f"(exit {proc.returncode}); see {log}")
                if use_host:
                    row = _parse_hostcsv(proc.stdout)
                    compress_ms.append(float(row["compress_dev_ms"]))
                    decompress_ms.append(float(row["decompress_dev_ms"]))
                    compress_host_ms.append(float(row["compress_host_ms"]))
                    decompress_host_ms.append(float(row["decompress_host_ms"]))
                    cmp_bytes = int(row["cmp_bytes"])
                    native_quality = {
                        "max_abs_err": float(row["max_err"]),
                        "abs_eb": float(row["abs_eb"]),
                        "native_status": row["status"],
                    }
                else:
                    row = _parse_csv(proc.stdout)
                    compress_ms.append(float(row["compress_ms"]))
                    decompress_ms.append(float(row["decompress_ms"]))
                    cmp_bytes = int(row["cmp_bytes"])
                    native_quality = {
                        "cr": float(row["cr"]),
                        "max_abs_err": float(row["max_err"]),
                        "abs_eb": float(row["abs_eb"]),
                        "native_status": row["status"],
                    }

        # The csv reports the bare bitstream; a stored cell is the container.
        # Prefer the artifact on disk when compress() already wrote one.
        compressed_ref = workdir / "c.fsz"
        if compressed_ref.exists():
            cmp_bytes = compressed_ref.stat().st_size
        else:
            cmp_bytes += _FSZ_HEADER_BYTES

        return BenchmarkResult(
            compress_device_ms_all=compress_ms,
            decompress_device_ms_all=decompress_ms,
            compress_host_ms_all=compress_host_ms,
            decompress_host_ms_all=decompress_host_ms,
            compressed_bytes=cmp_bytes,
            stages=[],
            native_quality=native_quality,
            log_path=log,
        )
