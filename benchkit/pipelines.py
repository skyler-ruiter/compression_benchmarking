"""FZGM TOML pipeline handling.

TOML is the primary pipeline interface (not the CLI --stages text path): it exposes the
full DAG — branches, fused stages, per-stage params — that the CLI parser cannot, and the
exact rendered TOML is archived into the results bundle so a run ships together with its
config (DESIGN.md D9).

We read with tomllib and render swept error bounds by targeted text substitution rather
than re-serializing — there is no stdlib TOML writer, and editing the text preserves the
template's comments (themselves useful provenance).
"""
from __future__ import annotations

import hashlib
import re
import tomllib
from dataclasses import dataclass
from pathlib import Path

# Match the value side of `error_bound = ...` but NOT `error_bound_mode = ...`
# (after "error_bound" the mode line has "_mode", so \s*= cannot match it).
_EB_RE = re.compile(r"(?m)^(?P<pre>\s*error_bound\s*=\s*).*$")
_MODE_RE = re.compile(r"(?m)^(?P<pre>\s*error_bound_mode\s*=\s*).*$")
# [pipeline]-table sizing hints (config_file.md): some presets (e.g. cuSZ-Hi's
# GInterp, which needs real x/y/z for its interpolation pyramid) declare these
# with a placeholder value baked in for the preset's example data. PREALLOCATE
# uses input_size as a hard buffer-size hint at finalize() — a stale value
# fails outright on any dataset bigger than the placeholder, so it must be
# re-rendered per field just like error_bound.
_INPUT_SIZE_RE = re.compile(r"(?m)^(?P<pre>\s*input_size\s*=\s*).*$")
_DIMS_RE = re.compile(r"(?m)^(?P<pre>\s*dims\s*=\s*).*$")
# Only ever matches a FLOAT input_type, which in every preset here belongs to the
# stage that consumes the raw field (LorenzoQuant / Quantizer / GInterp) — the
# intermediate stages declare int32/uint16/uint32 and must not be touched. Audited
# across configs/pipelines/*.toml: all 21 float-typed keys are that first stage.
_INPUT_DTYPE_RE = re.compile(r'(?m)^(?P<pre>\s*input_type\s*=\s*)"float(?:32|64)"\s*$')
_FZGM_DTYPE = {"f32": "float32", "f64": "float64"}


def _fmt_float(x: float) -> str:
    return repr(float(x))   # '0.0001', '1e-05', '0.01' — all valid TOML floats


@dataclass
class PipelineToml:
    path: Path
    text: str
    doc: dict

    @classmethod
    def load(cls, path: str | Path) -> "PipelineToml":
        p = Path(path).resolve()
        text = p.read_text()
        return cls(path=p, text=text, doc=tomllib.loads(text))

    def lossy_stages(self) -> list[dict]:
        """Stages that declare an error_bound (the bound-carrying / lossy stages)."""
        return [s for s in self.doc.get("stage", []) if "error_bound" in s]

    def declared_eb_mode(self) -> tuple[float, str]:
        """The (error_bound, mode) baked into the first lossy stage (for `from_toml`)."""
        stages = self.lossy_stages()
        if not stages:
            raise ValueError(f"{self.path}: no stage declares error_bound")
        s = stages[0]
        return float(s["error_bound"]), str(s.get("error_bound_mode", "ABS"))

    def check_dtype(self, dtype: str) -> None:
        """Raise if the raw-consuming stage's float input_type contradicts `dtype`.

        For the `from_toml` path, where the config ships verbatim and so cannot be
        retargeted. A contradiction here is the silent-corruption case described in
        `render`, so it is worth failing the cell loudly instead of recording a row
        with a nan PSNR that looks like a compressor result.
        """
        want = _FZGM_DTYPE.get(dtype)
        if want is None:
            return
        declared = [s["input_type"] for s in self.doc.get("stage", [])
                    if str(s.get("input_type", "")).startswith("float")]
        if declared and want not in declared:
            raise ValueError(
                f"{self.path}: pipeline declares input_type={declared[0]!r} but the "
                f"field is {dtype} ({want}). With error_mode='from_toml' the config is "
                f"shipped verbatim and cannot be retargeted — point this run at a "
                f"{want} preset, or use a swept error mode so the dtype is rendered.")

    def render(self, eb: float, toml_mode: str,
               dims: list[int] | None = None, input_size: int | None = None,
               dtype: str | None = None) -> str:
        """Return the template text with every lossy stage's bound+mode overridden.

        If the template's [pipeline] table declares `dims`/`input_size` (sizing
        hints for PREALLOCATE), and the caller passes the real field's values,
        those are overridden too — otherwise a placeholder baked into the preset
        (sized for its own example data) silently mismatches any other dataset.
        Templates that don't declare these keys are left untouched (no-op).

        `dtype` ("f32"/"f64") rewrites the float `input_type` of the raw-consuming
        stage to match the field. Presets are written float32 because every dataset
        benchmarked before 2026-07-28 was f32; without this an f64 field is fed to a
        float32 Quantizer, which reads the 8-byte values as 4-byte ones. That is not
        a clean failure: TiledLorenzo/GInterp pipelines abort with "Benchmark size
        mismatch" (the reconstruction comes back half-size), but the plain-Lorenzo
        ones SILENTLY return garbage — PSNR nan, eb_satisfied false, and a CR of
        exactly 64.00/128.00 from degenerate all-zero codes. FZGM itself has always
        supported f64 (QuantizerStage<double, uint32_t> etc.); only these presets
        and this renderer were f32-only. See DESIGN.md D27.
        """
        text, n_eb = _EB_RE.subn(lambda m: m.group("pre") + _fmt_float(eb), self.text)
        text, n_mode = _MODE_RE.subn(lambda m: m.group("pre") + f'"{toml_mode}"', text)
        if n_eb == 0:
            raise ValueError(f"{self.path}: no error_bound line to render")
        if dims is not None:
            dims3 = list(dims) + [1] * (3 - len(dims))
            dims_str = "[" + ", ".join(str(d) for d in dims3[:3]) + "]"
            text, _ = _DIMS_RE.subn(lambda m: m.group("pre") + dims_str, text)
        if input_size is not None:
            text, _ = _INPUT_SIZE_RE.subn(lambda m: m.group("pre") + str(int(input_size)), text)
        if dtype is not None:
            fz = _FZGM_DTYPE.get(dtype)
            if fz is None:
                raise ValueError(f"{self.path}: cannot render unknown dtype {dtype!r} "
                                 f"(known: {sorted(_FZGM_DTYPE)})")
            text, n_dt = _INPUT_DTYPE_RE.subn(lambda m: m.group("pre") + f'"{fz}"', text)
            if n_dt == 0:
                # Every FZGM preset starts with a stage that consumes the raw field and
                # declares a float input_type. None means the template is not shaped the
                # way this renderer assumes, and silently continuing would reintroduce
                # exactly the f64 corruption this argument exists to prevent.
                raise ValueError(
                    f"{self.path}: no float input_type line to render for dtype={dtype}. "
                    f"A preset consuming raw field data must declare "
                    f'input_type = "float32"|"float64" on its first stage.')
        tomllib.loads(text)  # validate the result re-parses
        return text


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()
