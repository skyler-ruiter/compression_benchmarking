"""lsCOMP float adapter: uniform quantizer + lossless lsCOMP integer coding."""
from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path

from .base import (Adapter, AdapterError, BenchmarkResult, CompressResult,
                   DecompressResult, Prepared, RunSpec, run_cli)
from .quantized_integer import (dequantize, make_metadata, metadata_sha256,
                                pack, quantize, read_metadata, unpack,
                                write_metadata)

_CT = re.compile(r"GPU compression time:\s+([0-9.eE+-]+) s")
_DT = re.compile(r"GPU decompression time:\s+([0-9.eE+-]+) s")


def _resolve_cli(explicit: str | None = None) -> str:
    for cand in (explicit, os.environ.get("LSCOMP_CLI")):
        if cand:
            return cand
    found = shutil.which("lsCOMP_uint32")
    if found:
        return found
    raise AdapterError("lsCOMP_uint32 not found: set LSCOMP_CLI or cli_path")


def _resolve_decoder() -> str:
    for cand in (os.environ.get("LSCOMP_DECODE_CLI"),
                 str(Path(__file__).resolve().parents[2] / "tools" /
                     "lscomp_decode" / "lscomp_decode")):
        if cand and Path(cand).exists():
            return cand
    raise AdapterError("lscomp_decode not found; run scripts/build-lscomp-decode.sh")


class LscompAdapter(Adapter):
    name = "lscomp"

    def __init__(self, variant: str = "lscomp", cli_path: str | None = None):
        self.variant = variant
        self.cli32 = _resolve_cli(cli_path)
        self.cli16 = os.environ.get("LSCOMP_UINT16_CLI") or str(
            Path(self.cli32).with_name("lsCOMP_uint16"))
        self.decoder = _resolve_decoder()

    def is_available(self) -> bool:
        return all(Path(p).exists() for p in (self.cli32, self.cli16, self.decoder))

    def provenance(self) -> dict:
        return {
            "name": "lscomp+benchkit_uniform_quantizer",
            "uint32_cli_path": self.cli32,
            "uint16_cli_path": self.cli16,
            "decoder_cli_path": self.decoder,
            "timing_method": "mixed_cpu_wall_quantizer_plus_native_cuda_events",
            "timing_note": (
                "Reported compression time is CPU quantization including q-file "
                "write plus lsCOMP's CUDA-event compression time. Decompression "
                "is lsCOMP CUDA-event time plus CPU dequantization/output write. "
                "This complete implemented transform is not directly comparable "
                "to a pure native GPU device_ms figure."
            ),
            "lscomp_settings": "-b 1 1 1 1 -p 1 (lossless integer payload)",
        }

    @staticmethod
    def _dims(spec: RunSpec) -> list[str]:
        slow = list(reversed(spec.field.dims))
        return [str(x) for x in ([1] * (3 - len(slow)) + slow)]

    def _cli(self, meta: dict) -> str:
        return self.cli16 if meta["integer_dtype"] == "u16" else self.cli32

    def _argv(self, spec: RunSpec, meta: dict, qpath: Path,
              payload: Path, qout: Path) -> list[str]:
        return [self._cli(meta), "-i", str(qpath), "-d", *self._dims(spec),
                "-b", "1", "1", "1", "1", "-p", "1",
                "-x", str(payload), "-o", str(qout)]

    @staticmethod
    def _native_ms(stdout: str) -> tuple[float, float]:
        cm, dm = _CT.search(stdout), _DT.search(stdout)
        if not cm or not dm:
            raise AdapterError("could not parse lsCOMP GPU phase timings")
        # Upstream labels these seconds after dividing its millisecond timer by
        # 1024. Recover the native CUDA-event milliseconds as implemented.
        return float(cm.group(1)) * 1024.0, float(dm.group(1)) * 1024.0

    def prepare(self, spec: RunSpec, workdir: Path) -> Prepared:
        pipeline = (spec.pipeline or "default").strip().lower()
        if pipeline != "default":
            raise AdapterError("lsCOMP wrapper has one pipeline: default")
        if len(spec.field.dims) > 3:
            raise AdapterError("lsCOMP supports 1-3D fields")
        workdir.mkdir(parents=True, exist_ok=True)
        meta = make_metadata(spec, "lscomp", "lossless-bins1-pool1")
        mpath = workdir / "quantization.json"
        write_metadata(meta, mpath)
        quantize(spec.field.path, workdir / "q.bin", meta)
        return Prepared([], float(spec.error_bound),
                        f"uniform-quantized-{meta['integer_dtype']}",
                        meta["error_basis"], "lscomp:lossless-bins1-pool1",
                        mpath, metadata_sha256(meta))

    def compress(self, spec: RunSpec, prep: Prepared, workdir: Path) -> CompressResult:
        meta = read_metadata(workdir / "quantization.json")
        payload, qout = workdir / "c.lscomp", workdir / "q.cli.dec.bin"
        artifact, log = workdir / "c.bkq", workdir / "compress.log"
        proc = run_cli(self._argv(spec, meta, workdir / "q.bin", payload, qout), log)
        if proc.returncode != 0 or not payload.exists():
            raise AdapterError(f"compress failed (exit {proc.returncode}); see {log}")
        pack(payload, artifact, meta)
        return CompressResult(artifact, artifact.stat().st_size,
                              spec.field.original_bytes, {}, log)

    def decompress(self, spec: RunSpec, compressed: Path, workdir: Path) -> DecompressResult:
        payload, qout = workdir / "decode.lscomp", workdir / "q.dec.bin"
        meta = unpack(compressed, payload)
        dims = [str(x) for x in ([1] * (3 - len(spec.field.dims)) +
                                list(reversed(spec.field.dims)))]
        log = workdir / "decompress.log"
        argv = [self.decoder, "-i", str(payload), "-o", str(qout),
                "-t", meta["integer_dtype"], "-d", *dims]
        proc = run_cli(argv, log)
        if proc.returncode != 0 or not qout.exists():
            raise AdapterError(f"decompress failed (exit {proc.returncode}); see {log}")
        out = workdir / "d.bin"
        dequantize(qout, out, meta)
        return DecompressResult(out, {}, log)

    def benchmark(self, spec: RunSpec, prep: Prepared, n_runs: int,
                  workdir: Path) -> BenchmarkResult:
        meta = read_metadata(workdir / "quantization.json")
        qpath, payload = workdir / "q.bench.bin", workdir / "bench.lscomp"
        qout, out = workdir / "q.bench.dec.bin", workdir / "d.bench.bin"
        log = workdir / "benchmark.log"
        cms: list[float] = []
        dms: list[float] = []
        with open(log, "w") as fh:
            for i in range(n_runs):
                qms = quantize(spec.field.path, qpath, meta)
                argv = self._argv(spec, meta, qpath, payload, qout)
                proc = subprocess.run(argv, capture_output=True, text=True)
                fh.write(f"\n# run {i}\n$ {' '.join(argv)}\n{proc.stdout}{proc.stderr}")
                if proc.returncode != 0:
                    raise AdapterError(f"benchmark failed (exit {proc.returncode}); see {log}")
                c_native, d_native = self._native_ms(proc.stdout)
                cms.append(qms + c_native)
                dms.append(d_native + dequantize(qout, out, meta))
        artifact = workdir / "c.bkq"
        return BenchmarkResult(cms, dms, artifact.stat().st_size, log_path=log)
