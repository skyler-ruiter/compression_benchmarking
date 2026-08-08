"""nvCOMP reference adapter — wraps `nvcomp_cli` (tools/nvcomp_cli).

nvCOMP is NVIDIA's GPU lossless compression SDK. It ships as a library with no
vendor CLI that reports device time, so this adapter drives `nvcomp_cli`, a small
tool built in this repo against the nvCOMP manager API — see
tools/nvcomp_cli/nvcomp_cli.cu and docs/adapters/nvcomp.md.

What shapes this adapter:

  - **Lossless only, no error bound.** Every nvCOMP algorithm compresses a byte
    stream; none takes a tolerance. So this adapter accepts exactly one canonical
    error mode, `lossless`, and rejects the lossy ones with a message that points
    at the right experiment rather than silently ignoring the bound. `prepare()`
    reports eb=0 and basis="lossless"; the harness then checks bit-exactness
    instead of a bound (metrics.compute_quality).

    That is also why the comparison this adapter exists for is set up the way it
    is: FZGM's `gpu_zstd` preset is *lossy* (LorenzoQuant -> GPULZ -> Huffman/ANS),
    so putting its 21x next to nvCOMP Zstd's 1.14x compares two different jobs.
    The head-to-head runs FZGM's coder stages with the predictor removed
    (configs/pipelines/gpu_zstd_lossless.toml) on the same bytes.

  - **Algorithm selected by the pipeline string**, since nvCOMP has no config
    file: `nvcomp:<algo>[:<key>=<val>...]`, e.g.

        nvcomp:zstd
        nvcomp:zstd:chunk=131072
        nvcomp:gdeflate:level=5
        nvcomp:lz4

    Keys are `chunk` (uncompressed chunk size in bytes, default 65536 — nvCOMP's
    documented sweet spot for Zstd) and `level` (deflate/gdeflate only, 0-5). A
    `level` on an algorithm that has none is an error, not a no-op, so a config
    that thinks it is sweeping levels cannot quietly emit identical rows.

  - **Timing is device-only and in-process.** `nvcomp_cli --benchmark` holds the
    input on the device and brackets only the manager's compress()/decompress()
    call with CUDA events, reusing one manager across reps so nvCOMP's lazy
    scratch allocation is not counted. One subprocess yields all N reps, so
    process startup is outside every measurement — unlike the zfp/SPERR CPU
    adapters, these numbers ARE comparable to the FZGM and cuSZp device_ms.

  - **CR includes nvCOMP's own header.** The manager runs in NVCOMP_NATIVE
    bitstream mode, whose header carries the chunking metadata needed to
    decompress. That is the real artifact, so its bytes count — same rule applied
    to every other tool here.
"""
from __future__ import annotations

import os
import re
import shutil
from pathlib import Path

from .base import (Adapter, AdapterError, BenchmarkResult, CompressResult,
                   DecompressResult, Prepared, RunSpec, load_report_json, run_cli)

# Algorithms nvcomp_cli exposes. `level` is nvCOMP's `algorithm` field on the
# Deflate/Gdeflate opts structs (0 = entropy-only .. 5 = highest ratio); Zstd, LZ4
# and ANS expose no equivalent knob in nvCOMP 5.2.
_ALGOS = {"zstd", "lz4", "deflate", "gdeflate", "ans",
          "snappy", "gzip", "bitcomp", "cascaded"}
_LEVEL_ALGOS = {"deflate", "gdeflate", "bitcomp"}

# Bitcomp and Cascaded model the input as an array of a given element type rather
# than as bytes; the other seven are byte coders with nothing to be told. nvCOMP's
# own default for that type is `uchar`, which on this corpus measures CR ~1.0 for
# both (vs 1.44 / 1.52 at the correct 4-byte width on CESM-2D/CLDHGH) — the D34
# "vendor default is wrong for your data" trap, in its most extreme form yet. So
# `dtype` is REQUIRED for these two and rejected for the rest.
_TYPED_ALGOS = {"bitcomp", "cascaded"}

# nvCOMP 5.3 removed NVCOMP_TYPE_FLOAT (enum value 8); there is no 32-bit float
# element type. f32 data can only be described by 4-byte integer width, so an f32
# field is run as `dtype=uint` and that limitation belongs in the writeup, not
# hidden behind an alias here. Deliberately no "float"/"f32" spelling.
_DTYPES = {"char", "uchar", "short", "ushort", "int", "uint",
           "longlong", "ulonglong"}

# Cascaded is a configurable scheme, not a fixed algorithm: N run-length passes,
# M delta passes, and an optional final bitpack. Every one of those primitives has
# an FZGM counterpart (rle/rre/rze/rare/raze, Lorenzo, bitpack/adaptive_bitpack),
# which is why Cascaded is mirrored by a DAG rather than paired with one coder.
_CASCADED_KEYS = {"rles", "deltas", "bp"}

_DEFAULT_CHUNK = 65536


def _sdk_version(root: str | None) -> str | None:
    """nvCOMP SDK version from the install's CMake package file.

    Recorded in provenance because the adapter's own numbers are only meaningful
    against a named SDK build, and an nvCOMP install is a hand-unpacked tarball
    with nothing forcing it to be current. 5.2.0.10 -> 5.3.0.16 left Zstd's
    bitstream byte-identical but moved ANS compress by +34%, so a session that
    does not record which one it ran cannot be compared against another.
    """
    if not root:
        return None
    cfg = Path(root) / "lib" / "cmake" / "nvcomp" / "nvcomp-config-version.cmake"
    try:
        for line in cfg.read_text().splitlines():
            m = re.match(r'\s*set\(PACKAGE_VERSION\s+"([^"]+)"\s*\)', line)
            if m:
                return m.group(1)
    except OSError:
        pass
    return None


def resolve_cli(explicit: str | None = None) -> str:
    for cand in (explicit, os.environ.get("NVCOMP_CLI")):
        if cand:
            return cand
    found = shutil.which("nvcomp_cli")
    if found:
        return found
    raise AdapterError(
        "nvcomp_cli not found: set NVCOMP_CLI or cli_path in the run entry. "
        "It is built from this repo — see scripts/build-nvcomp-cli.sh and "
        "docs/adapters/nvcomp.md.")


def _parse_pipeline(pipeline: str) -> tuple[str, int, int, str | None, dict]:
    """'nvcomp:<algo>[:key=val...]' -> (algo, chunk, level, dtype, cascaded).

    level == -1 means "the algorithm's own default" (nvcomp_cli omits the flag).
    dtype is None for the byte coders. `cascaded` is empty unless algo is cascaded.

    Keys: chunk, level, dtype, rles, deltas, bp. An option that does not apply to
    the chosen algorithm is an error rather than a no-op, so a sweep cannot emit
    N identical rows under N distinct variant names.
    """
    parts = [p.strip() for p in pipeline.split(":") if p.strip()]
    if parts and parts[0] == "nvcomp":
        parts = parts[1:]
    if not parts:
        raise AdapterError(
            f"nvcomp: cannot read an algorithm out of pipeline '{pipeline}'. "
            f"Expected 'nvcomp:<algo>[:key=val...]', e.g. 'nvcomp:zstd'.")

    algo = parts[0].lower()
    if algo not in _ALGOS:
        raise AdapterError(
            f"nvcomp: unknown algorithm '{algo}' (have: {sorted(_ALGOS)})")

    chunk, level = _DEFAULT_CHUNK, -1
    dtype: str | None = None
    cascaded: dict[str, int] = {}
    for opt in parts[1:]:
        if "=" not in opt:
            raise AdapterError(
                f"nvcomp: malformed pipeline option '{opt}' in '{pipeline}' "
                f"(expected key=value; keys: chunk, level, dtype, rles, deltas, bp)")
        key, _, val = opt.partition("=")
        key = key.strip().lower()

        # dtype is the one string-valued option; handle it before the int parse.
        if key == "dtype":
            if algo not in _TYPED_ALGOS:
                raise AdapterError(
                    f"nvcomp: '{algo}' compresses an undifferentiated byte stream and "
                    f"has no element type — only {sorted(_TYPED_ALGOS)} do. Drop "
                    f"'dtype=' from '{pipeline}'.")
            dtype = val.strip().lower()
            if dtype not in _DTYPES:
                raise AdapterError(
                    f"nvcomp: unknown dtype '{dtype}' (have: {sorted(_DTYPES)}). "
                    f"nvCOMP 5.3 has no 32-bit float type — use 'uint' for f32 data "
                    f"and 'ulonglong' for f64, and record that in the result.")
            continue

        try:
            ival = int(val)
        except ValueError:
            raise AdapterError(
                f"nvcomp: option '{key}' needs an integer, got '{val}'") from None

        if key in _CASCADED_KEYS:
            if algo != "cascaded":
                raise AdapterError(
                    f"nvcomp: '{key}' is a Cascaded scheme setting; '{algo}' has no "
                    f"such scheme. Drop it from '{pipeline}'.")
            if ival < 0 or (key == "bp" and ival not in (0, 1)):
                raise AdapterError(f"nvcomp: bad value for '{key}': {ival}")
            cascaded[key] = ival
        elif key == "chunk":
            if ival <= 0:
                raise AdapterError(f"nvcomp: chunk must be > 0, got {ival}")
            chunk = ival
        elif key == "level":
            if algo not in _LEVEL_ALGOS:
                raise AdapterError(
                    f"nvcomp: '{algo}' has no compression level in nvCOMP 5.3 — "
                    f"only {sorted(_LEVEL_ALGOS)} do. Drop 'level=' from "
                    f"'{pipeline}' rather than letting it be ignored.")
            hi = 1 if algo == "bitcomp" else 5   # bitcomp: 0=default, 1=sparse
            if not 0 <= ival <= hi:
                raise AdapterError(
                    f"nvcomp: level must be 0-{hi} for {algo}, got {ival}")
            level = ival
        else:
            raise AdapterError(
                f"nvcomp: unknown pipeline option '{key}' in '{pipeline}' "
                f"(have: chunk, level, dtype, rles, deltas, bp)")

    if algo in _TYPED_ALGOS and dtype is None:
        raise AdapterError(
            f"nvcomp: '{algo}' models its input as a typed array and nvCOMP's own "
            f"default (uchar) measures CR ~1.0 on this corpus. Pass it explicitly, "
            f"e.g. '{pipeline}:dtype=uint' for f32 or ':dtype=ushort' for uint16 "
            f"quant codes. See docs/adapters/nvcomp.md.")
    return algo, chunk, level, dtype, cascaded


class NvcompAdapter(Adapter):
    """nvCOMP adapter — GPU lossless (zstd / lz4 / deflate / gdeflate / ans)."""

    name = "nvcomp"

    def __init__(self, variant: str = "nvcomp", cli_path: str | None = None):
        self.variant = variant
        self.cli = resolve_cli(cli_path)

    def is_available(self) -> bool:
        return Path(self.cli).exists() or shutil.which(self.cli) is not None

    def provenance(self) -> dict:
        return {
            "cli_path": self.cli,
            "name": "nvcomp",
            "nvcomp_root": os.environ.get("NVCOMP_ROOT"),
            "nvcomp_version": _sdk_version(os.environ.get("NVCOMP_ROOT")),
            "timing_method": "cuda_events_device_only_in_process",
            "timing_note": (
                "One subprocess runs all reps in process with the input resident on "
                "the device; CUDA events bracket only the nvCOMP manager's "
                "compress()/decompress() call. One manager is reused across reps so "
                "nvCOMP's lazy scratch allocation is not timed. No H2D/D2H, no file "
                "I/O and no process startup inside the timed region — directly "
                "comparable to the FZGM and cuSZp device_ms."
            ),
            "lossless": True,
            "lossless_note": (
                "Every nvCOMP algorithm is lossless and takes no error bound. Rows "
                "carry error_mode=lossless, eb=0 and max_abs_err=0; the harness "
                "checks bit-exactness instead of bound satisfaction."
            ),
            "bitstream_kind": "NVCOMP_NATIVE (metadata header included in compressed_bytes)",
            "checksum_policy": "NoComputeNoVerify",
        }

    def prepare(self, spec: RunSpec, workdir: Path) -> Prepared:
        if spec.error_mode != "lossless":
            raise AdapterError(
                f"nvcomp: error mode '{spec.error_mode}' is not applicable — every "
                f"nvCOMP algorithm is lossless and takes no error bound. Set "
                f"error.mode: lossless in the experiment (and drop error.bounds). "
                f"To compare against FZGM on equal terms, run FZGM's coder-only "
                f"pipeline configs/pipelines/gpu_zstd_lossless.toml in the same "
                f"experiment — see docs/adapters/nvcomp.md.")

        algo, chunk, level, dtype, cascaded = _parse_pipeline(spec.pipeline)
        workdir.mkdir(parents=True, exist_ok=True)

        config_args = ["-a", algo, "--chunk-size", str(chunk)]
        if level >= 0:
            config_args += ["--level", str(level)]
        if dtype is not None:
            config_args += ["--dtype", dtype]
        for key in ("rles", "deltas", "bp"):
            if key in cascaded:
                config_args += [f"--{key}", str(cascaded[key])]

        # The label is what lands in pipeline_ref and distinguishes rows, so every
        # setting that changes the bitstream has to appear in it.
        label = f"nvcomp:{algo}:chunk={chunk}"
        if level >= 0:
            label += f":level={level}"
        if dtype is not None:
            label += f":dtype={dtype}"
        for key in ("rles", "deltas", "bp"):
            if key in cascaded:
                label += f":{key}={cascaded[key]}"

        return Prepared(
            config_args=config_args,
            eb=0.0,
            native_mode="lossless",
            basis="lossless",
            pipeline_ref=label,
            pipeline_path=None,
            pipeline_sha256=None,
        )

    def compress(self, spec: RunSpec, prep: Prepared, workdir: Path) -> CompressResult:
        compressed = workdir / "c.nvc"
        log = workdir / "compress.log"
        report = workdir / "compress_report.json"

        argv = [self.cli, "--compress", "-i", str(spec.field.path),
                "-o", str(compressed), *prep.config_args,
                "--report-json", str(report)]
        proc = run_cli(argv, log)
        if proc.returncode != 0:
            raise AdapterError(f"compress failed (exit {proc.returncode}); see {log}")
        raw = load_report_json(report)
        if not compressed.exists():
            raise AdapterError(f"compress produced no output at {compressed}; see {log}")

        return CompressResult(
            compressed_path=compressed,
            compressed_bytes=compressed.stat().st_size,
            original_bytes=spec.field.original_bytes,
            raw_json=raw,
            log_path=log,
        )

    def decompress(self, spec: RunSpec, compressed: Path, workdir: Path) -> DecompressResult:
        decompressed = workdir / "d.bin"
        log = workdir / "decompress.log"
        report = workdir / "decompress_report.json"

        # The NVCOMP_NATIVE header carries chunking and sizes, so -a only has to
        # name the same algorithm family the bitstream was written with.
        algo_args = prep_algo_args(workdir, spec)
        argv = [self.cli, "--decompress", "-i", str(compressed),
                "-o", str(decompressed), *algo_args,
                "--report-json", str(report)]
        proc = run_cli(argv, log)
        if proc.returncode != 0:
            raise AdapterError(f"decompress failed (exit {proc.returncode}); see {log}")
        raw = load_report_json(report)
        if not decompressed.exists():
            raise AdapterError(f"decompress produced no output at {decompressed}; see {log}")

        return DecompressResult(decompressed_path=decompressed, raw_json=raw, log_path=log)

    def benchmark(self, spec: RunSpec, prep: Prepared, n_runs: int, workdir: Path) -> BenchmarkResult:
        """One subprocess, n_runs in-process timed reps per phase.

        `--warmup 0` on purpose: benchkit already runs `warmup_reps + repetitions`
        and drops the first `warmup_reps` in metrics.summarize_timing. Doing it in
        the tool as well would hide the ramp from that machinery instead of
        letting it be measured — and the first rep is where nvCOMP's lazy scratch
        allocation would land if the tool were not already absorbing it with an
        untimed compress before the loop.
        """
        log = workdir / "benchmark.log"
        report = workdir / "benchmark_report.json"

        argv = [self.cli, "--benchmark", "-i", str(spec.field.path),
                *prep.config_args, "--reps", str(n_runs), "--warmup", "0",
                "--quiet", "--report-json", str(report)]
        proc = run_cli(argv, log)
        if proc.returncode != 0:
            raise AdapterError(f"benchmark failed (exit {proc.returncode}); see {log}")
        raw = load_report_json(report)

        comp_ms = list(raw.get("compress_device_ms") or [])
        dec_ms = list(raw.get("decompress_device_ms") or [])
        if len(comp_ms) != n_runs or len(dec_ms) != n_runs:
            raise AdapterError(
                f"benchmark returned {len(comp_ms)} compress / {len(dec_ms)} "
                f"decompress timings, expected {n_runs} of each; see {log}")

        return BenchmarkResult(
            compress_device_ms_all=comp_ms,
            decompress_device_ms_all=dec_ms,
            # ~0.01 ms above device_ms in practice: nvCOMP's manager writes one
            # contiguous device buffer, so almost nothing sits outside the event
            # bracket. Recorded so the FZGM comparison can be audited — see the
            # note on BenchmarkResult.compress_host_ms_all.
            compress_host_ms_all=[float(x) for x in (raw.get("compress_host_ms") or [])],
            decompress_host_ms_all=[float(x) for x in (raw.get("decompress_host_ms") or [])],
            compressed_bytes=int(raw.get("compressed_bytes", 0)),
            stages=[],
            native_quality=None,
            raw_json=raw,
            log_path=log,
        )


def prep_algo_args(workdir: Path, spec: RunSpec) -> list[str]:
    """Re-derive the flags decompress() needs from the run's pipeline string.

    decompress() is handed a path, not a Prepared — see Adapter.decompress in
    base.py — so these are read back out of the pipeline string rather than
    carried across.

    Chunk size and the Cascaded scheme ARE recovered from the NVCOMP_NATIVE
    header and are not repeated. Two things are not, and both are silent if
    omitted:

    - `--dtype`, because nvcomp_cli refuses to guess an element type.
    - `--level` **for bitcomp only**. Deflate/Gdeflate's `algorithm` is an
      encoder-side choice the header describes, but Bitcomp's is not: a stream
      written with algorithm=1 (sparse) and decoded by a manager constructed with
      the default algorithm=0 decodes to WRONG BYTES at exit 0 — measured on
      EXAALT-qcodes/xx as PSNR 20.85 dB on a codec that is lossless by
      construction. Only the harness's own bit-exactness check catches it, the
      same way it caught E23. Do not "simplify" this back to `-a` alone.
    """
    algo, _, level, dtype, _ = _parse_pipeline(spec.pipeline)
    args = ["-a", algo]
    if dtype is not None:
        args += ["--dtype", dtype]
    if algo == "bitcomp" and level >= 0:
        args += ["--level", str(level)]
    return args
