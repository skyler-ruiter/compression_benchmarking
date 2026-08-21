"""MANS as an error-bounded float baseline via a uniform quantizer.

MANS itself is lossless on u16/u32. This adapter owns the lossy transform and
stores its parameters beside the native bitstream in a self-describing
benchkit container. Timings are full external wall clock for quantization +
native CLI, and native CLI + dequantization; they are useful implementation
timings but are not native GPU-kernel figures.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import time
from pathlib import Path

from .base import (Adapter, AdapterError, BenchmarkResult, CompressResult,
                   DecompressResult, Prepared, RunSpec, run_cli)
from .quantized_integer import (dequantize, make_metadata, metadata_sha256,
                                pack, quantize, read_metadata, unpack,
                                write_metadata)


def _resolve_compress(explicit: str | None = None) -> str:
    for cand in (explicit, os.environ.get("MANS_CLI")):
        if cand:
            return cand
    found = shutil.which("nv_mans_compress")
    if found:
        return found
    raise AdapterError("nv_mans_compress not found: set MANS_CLI or cli_path")


def _resolve_decompress(compress_cli: str) -> str:
    for cand in (os.environ.get("MANS_DECOMPRESS_CLI"),
                 str(Path(compress_cli).with_name("nv_mans_decompress"))):
        if cand and Path(cand).exists():
            return cand
    found = shutil.which("nv_mans_decompress")
    if found:
        return found
    raise AdapterError("nv_mans_decompress not found: set MANS_DECOMPRESS_CLI")


class MansAdapter(Adapter):
    name = "mans"

    def __init__(self, variant: str = "mans", cli_path: str | None = None):
        self.variant = variant
        self.cli = _resolve_compress(cli_path)
        self.decompress_cli = _resolve_decompress(self.cli)

    def is_available(self) -> bool:
        return Path(self.cli).exists() and Path(self.decompress_cli).exists()

    def provenance(self) -> dict:
        return {
            "name": "mans+benchkit_uniform_quantizer",
            "cli_path": self.cli,
            "decompress_cli_path": self.decompress_cli,
            "timing_method": "external_wall_end_to_end_wrapper",
            "timing_note": (
                "compress = CPU uniform quantization including q-file write + "
                "MANS compress subprocess; decompress = MANS decompress "
                "subprocess + CPU dequantization including output write. Includes "
                "process startup and intermediate I/O; not comparable to CUDA-event "
                "device_ms from native float GPU compressors. This machine uses the "
                "CPU MANS CLI because the installed NVIDIA backend crashes on u32 "
                "quantization codes required by tight bounds."
            ),
            "quantizer": "q=round((x-min)/(abs_eb/4)); exact u16/u32 MANS coding",
            "mapping": "1d_flattened",
            "backend": "cpu" if "/cpu/" in self.cli else "configured_cli",
        }

    @staticmethod
    def _mode(spec: RunSpec) -> str:
        pipeline = (spec.pipeline or "default").strip().lower()
        if pipeline == "default":
            return "p"
        if pipeline not in {"p", "r"}:
            raise AdapterError("MANS pipeline must be default, p, or r")
        return pipeline

    @staticmethod
    def _flag(meta: dict) -> str:
        return "-u2" if meta["integer_dtype"] == "u16" else "-u4"

    def _compress_argv(self, spec: RunSpec, meta: dict, qpath: Path,
                       payload: Path) -> list[str]:
        # Flatten deliberately. The installed MANS multidimensional mapping
        # segfaults on valid odd-shaped fields (e.g. 69x69x115), while the 1-D
        # path is bit-exact for the same data. Preserve this choice in provenance.
        return [self.cli, self._flag(meta), str(qpath), str(payload),
                "--mode", meta["pipeline"]]

    def prepare(self, spec: RunSpec, workdir: Path) -> Prepared:
        workdir.mkdir(parents=True, exist_ok=True)
        mode = self._mode(spec)
        meta = make_metadata(spec, "mans", mode)
        mpath = workdir / "quantization.json"
        write_metadata(meta, mpath)
        quantize(spec.field.path, workdir / "q.bin", meta)
        return Prepared([], float(spec.error_bound),
                        f"uniform-quantized-{meta['integer_dtype']}",
                        meta["error_basis"], f"mans:{mode}", mpath,
                        metadata_sha256(meta))

    def compress(self, spec: RunSpec, prep: Prepared, workdir: Path) -> CompressResult:
        meta = read_metadata(workdir / "quantization.json")
        payload, artifact = workdir / "c.mans", workdir / "c.bkq"
        log = workdir / "compress.log"
        proc = run_cli(self._compress_argv(spec, meta, workdir / "q.bin", payload), log)
        if proc.returncode != 0 or not payload.exists():
            raise AdapterError(f"compress failed (exit {proc.returncode}); see {log}")
        pack(payload, artifact, meta)
        return CompressResult(artifact, artifact.stat().st_size,
                              spec.field.original_bytes, {}, log)

    def decompress(self, spec: RunSpec, compressed: Path, workdir: Path) -> DecompressResult:
        payload, qout = workdir / "decode.mans", workdir / "q.dec.bin"
        meta = unpack(compressed, payload)
        log = workdir / "decompress.log"
        proc = run_cli([self.decompress_cli, self._flag(meta), str(payload), str(qout)], log)
        if proc.returncode != 0 or not qout.exists():
            raise AdapterError(f"decompress failed (exit {proc.returncode}); see {log}")
        out = workdir / "d.bin"
        dequantize(qout, out, meta)
        return DecompressResult(out, {}, log)

    def benchmark(self, spec: RunSpec, prep: Prepared, n_runs: int,
                  workdir: Path) -> BenchmarkResult:
        meta = read_metadata(workdir / "quantization.json")
        qpath, payload = workdir / "q.bench.bin", workdir / "bench.mans"
        qout, out = workdir / "q.bench.dec.bin", workdir / "d.bench.bin"
        log = workdir / "benchmark.log"
        cms: list[float] = []
        dms: list[float] = []
        with open(log, "w") as fh:
            for i in range(n_runs):
                qms = quantize(spec.field.path, qpath, meta)
                argv = self._compress_argv(spec, meta, qpath, payload)
                t0 = time.perf_counter()
                proc = subprocess.run(argv, capture_output=True, text=True)
                codec_cms = (time.perf_counter() - t0) * 1000.0
                fh.write(f"\n# run {i} compress\n$ {' '.join(argv)}\n{proc.stdout}{proc.stderr}")
                if proc.returncode != 0:
                    raise AdapterError(f"benchmark compress failed (exit {proc.returncode}); see {log}")
                cms.append(qms + codec_cms)
                argv = [self.decompress_cli, self._flag(meta), str(payload), str(qout)]
                t0 = time.perf_counter()
                proc = subprocess.run(argv, capture_output=True, text=True)
                codec_dms = (time.perf_counter() - t0) * 1000.0
                fh.write(f"\n# run {i} decompress\n$ {' '.join(argv)}\n{proc.stdout}{proc.stderr}")
                if proc.returncode != 0:
                    raise AdapterError(f"benchmark decompress failed (exit {proc.returncode}); see {log}")
                dms.append(codec_dms + dequantize(qout, out, meta))
        artifact = workdir / "c.bkq"
        return BenchmarkResult(cms, dms, artifact.stat().st_size,
                               compress_host_ms_all=list(cms),
                               decompress_host_ms_all=list(dms), log_path=log)
