"""FZGPUModules adapter — wraps `fzgmod-cli`, TOML-first.

Pipelines are driven via TOML configs (`-c config.toml`), which expose the full DAG
(branches, fused stages) that the CLI --stages text path cannot, and which we archive
per run for provenance. A `--stages` chain is still accepted for quick linear tests.
See docs/adapters/fzgm.md for the contract and confirmed gotchas.
"""
from __future__ import annotations

import os
import shutil
from pathlib import Path

from ..config import FieldSpec
from ..pipelines import PipelineToml, sha256
from .base import (Adapter, AdapterError, BenchmarkResult, CompressResult,
                   DecompressResult, Prepared, RunSpec, load_report_json, run_cli)

# canonical mode -> (native FZGM mode string, eb basis for the satisfaction check).
# rel_range -> NOA (FZGM's range-relative); rel_maxabs -> REL (Lorenzo max|data| basis).
_MODE_MAP = {
    "abs":        ("ABS", "abs"),
    "rel_range":  ("NOA", "range"),
    "rel_maxabs": ("REL", "maxabs"),
}
# reverse: native TOML mode -> basis, for `from_toml` (read the file's declared mode).
_NATIVE_BASIS = {"ABS": "abs", "NOA": "range", "REL": "maxabs"}
# CLI lowercase form for the --stages path.
_CLI_MODE = {"ABS": "abs", "NOA": "noa", "REL": "rel"}


def resolve_cli(explicit: str | None = None) -> str:
    # Precedence: per-run cli_path > $FZGMOD_CLI (set by Site/site.local.yaml) > PATH.
    # No hardcoded path — portable across desktop and HPC (set FZGMOD_CLI via module/Spack
    # or configs/site.local.yaml). Note: a system fzgmod-cli may be stale; prefer an
    # explicit FZGMOD_CLI pointing at the intended build.
    for cand in (explicit, os.environ.get("FZGMOD_CLI")):
        if cand:
            return cand
    found = shutil.which("fzgmod-cli")
    if found:
        return found
    raise AdapterError("fzgmod-cli not found: set FZGMOD_CLI, cli_path, or "
                       "configs/site.local.yaml (see configs/site.example.yaml)")


class FzgmAdapter(Adapter):
    name = "fzgm"

    def __init__(self, variant: str = "fzgm", cli_path: str | None = None):
        self.variant = variant
        self.cli = resolve_cli(cli_path)

    def is_available(self) -> bool:
        return Path(self.cli).exists() or shutil.which(self.cli) is not None

    def provenance(self) -> dict:
        return {"cli_path": self.cli, "name": "fzgmod-cli"}

    # -- prepare: translate mode + render/archive config ----------------------
    def prepare(self, spec: RunSpec, workdir: Path) -> Prepared:
        if spec.pipeline.strip().endswith(".toml"):
            return self._prepare_toml(spec, workdir)
        return self._prepare_stages(spec, workdir)

    def _prepare_toml(self, spec: RunSpec, workdir: Path) -> Prepared:
        tpl = PipelineToml.load(spec.pipeline)
        out = workdir / "pipeline.toml"
        if spec.error_mode == "from_toml":
            eb, native_mode = tpl.declared_eb_mode()
            text = tpl.text                      # ship the config verbatim
            basis = _NATIVE_BASIS.get(native_mode, "abs")
        else:
            native_mode, basis = _MODE_MAP[spec.error_mode]
            eb = float(spec.error_bound)
            text = tpl.render(eb, native_mode)   # override every lossy stage's bound+mode
        out.write_text(text)
        return Prepared(
            config_args=["-c", str(out)], eb=eb, native_mode=native_mode, basis=basis,
            pipeline_ref=str(tpl.path), pipeline_path=out, pipeline_sha256=sha256(text),
        )

    def _prepare_stages(self, spec: RunSpec, workdir: Path) -> Prepared:
        if spec.error_mode == "from_toml":
            raise AdapterError("from_toml mode requires a .toml pipeline, not a --stages chain")
        native_mode, basis = _MODE_MAP[spec.error_mode]
        eb = float(spec.error_bound)
        args = ["--stages", spec.pipeline, "-m", _CLI_MODE[native_mode], "-e", repr(eb)]
        return Prepared(config_args=args, eb=eb, native_mode=native_mode, basis=basis,
                        pipeline_ref=spec.pipeline, pipeline_path=None, pipeline_sha256=None)

    # -- run ------------------------------------------------------------------
    def _io(self, f: FieldSpec) -> list[str]:
        return ["-l", f.dim_arg, "-t", f.dtype]

    def compress(self, spec: RunSpec, prep: Prepared, workdir: Path) -> CompressResult:
        f = spec.field
        out = workdir / "c.fzm"
        report = workdir / "z.json"
        log = workdir / "compress.log"
        argv = [self.cli, "-z", "-i", str(f.path), "-o", str(out),
                *self._io(f), *prep.config_args, "--report-json", str(report)]
        proc = run_cli(argv, log)
        if proc.returncode != 0:
            raise AdapterError(f"compress failed (exit {proc.returncode}); see {log}")
        rep = load_report_json(report)
        size = rep["size"]
        return CompressResult(compressed_path=out,
                              compressed_bytes=int(size["compressed_bytes"]),
                              original_bytes=int(size["original_bytes"]),
                              raw_json=rep, log_path=log)

    def decompress(self, spec: RunSpec, compressed: Path, workdir: Path) -> DecompressResult:
        f = spec.field
        out = workdir / "d.bin"
        report = workdir / "x.json"
        log = workdir / "decompress.log"
        # .fzm is self-describing; --compare truncates output to the original length.
        argv = [self.cli, "-x", "-i", str(compressed), "-o", str(out),
                "--compare", str(f.path), "--report-json", str(report)]
        proc = run_cli(argv, log)
        if proc.returncode != 0:
            raise AdapterError(f"decompress failed (exit {proc.returncode}); see {log}")
        load_report_json(report)
        return DecompressResult(decompressed_path=out, raw_json={}, log_path=log)

    def benchmark(self, spec: RunSpec, prep: Prepared, n_runs: int, workdir: Path) -> BenchmarkResult:
        f = spec.field
        report = workdir / "b.json"
        log = workdir / "benchmark.log"
        argv = [self.cli, "-b", "-i", str(f.path), *self._io(f), *prep.config_args,
                "--runs", str(n_runs), "--compare", str(f.path),
                "--report-json", str(report)]
        proc = run_cli(argv, log)
        if proc.returncode != 0:
            raise AdapterError(f"benchmark failed (exit {proc.returncode}); see {log}")
        rep = load_report_json(report)
        t = rep["timing"]
        comp = t.get("compress", {}).get("device_ms", {}).get("all", [])
        dec = t.get("decompress", {}).get("device_ms", {}).get("all", [])
        return BenchmarkResult(
            compress_device_ms_all=[float(x) for x in comp],
            decompress_device_ms_all=[float(x) for x in dec],
            compressed_bytes=int(rep["size"]["compressed_bytes"]),
            stages=rep.get("stages", []), native_quality=rep.get("quality"),
            raw_json=rep, log_path=log,
        )
