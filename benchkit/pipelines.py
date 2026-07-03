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

    def render(self, eb: float, toml_mode: str,
               dims: list[int] | None = None, input_size: int | None = None) -> str:
        """Return the template text with every lossy stage's bound+mode overridden.

        If the template's [pipeline] table declares `dims`/`input_size` (sizing
        hints for PREALLOCATE), and the caller passes the real field's values,
        those are overridden too — otherwise a placeholder baked into the preset
        (sized for its own example data) silently mismatches any other dataset.
        Templates that don't declare these keys are left untouched (no-op).
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
        tomllib.loads(text)  # validate the result re-parses
        return text


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()
