#!/usr/bin/env python3
"""Probe every pipeline preset to record which stages it runs, per phase.

Why this exists
---------------
Result rows carry `stages[]`, but every baseline captured before 2026-07-29 has
decompress entries only (FZGM's `setMemoryStrategy()` silently disabled profiling
on the compress DAG — fixed, see FZGPUModules CHANGELOG). Stage-level result
invalidation needs to know which stages a cell depended on, and re-running a
9,816-cell sweep just to recover a list of *names* would be absurd.

It is also unnecessary. Which stages a preset runs is a property of the preset,
not of the field: across 4,860 FZGM rows every preset TOML maps to exactly one
stage set. So probe each preset once on a few KB of synthetic data (seconds, not
hours) and record the mapping. `benchkit stale --stage X` then joins on
`pipeline_ref`, and no existing row has to be rewritten or invented.

The probe runs the real binary rather than parsing the TOML statically, because
the runtime node name is `Stage::getName()` and need not equal the TOML `type`
string — and because a static parse cannot see stage selection that depends on
anything but the file.

Output: configs/pipeline_stages.json, checked in, regenerate when presets change.

Usage:
    python scripts/probe_pipeline_stages.py                # write the index
    python scripts/probe_pipeline_stages.py --check        # verify, don't write
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchkit.pipelines import PipelineToml  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
PRESET_DIR = REPO / "configs" / "pipelines"
OUT = REPO / "configs" / "pipeline_stages.json"

# A preset only has to run; the data need not be meaningful. 1-D presets get 1-D
# dims because a dimension-specialised preset (cuszp3_1d vs cuszp3_3d) will refuse
# or mis-size otherwise.
DIMS_3D = [32, 32, 32]
DIMS_1D = [32768, 1, 1]


def dims_for(name: str) -> list[int]:
    return DIMS_1D if "_1d" in name else DIMS_3D


def _literal_bklen(tpl: PipelineToml) -> int | None:
    """Codebook length of a uint16 Huffman stage, if the preset has one.

    Only consulted for coder-only presets, to size their synthetic input's symbol
    alphabet (see probe()). None means "no uint16 Huffman", i.e. a byte-alphabet
    preset that the f32 ramp already satisfies.
    """
    for s in tpl.doc.get("stage", []):
        if s.get("type") == "Huffman" and s.get("input_type") == "uint16":
            return int(s.get("bklen", 0)) or None
    return None


def probe(cli: str, preset: Path, workdir: Path) -> dict:
    dims = dims_for(preset.stem)
    n = dims[0] * dims[1] * dims[2]
    import struct

    tpl = PipelineToml.load(preset)
    raw = workdir / f"{preset.stem}.f32"

    # A smooth ramp: compresses, and avoids the all-constant path that some stages
    # short-circuit (which could hide a stage from the trace).
    #
    # A *coder-only* preset (no predictor) reads its input as symbols, not floats,
    # so an f32 ramp is not merely meaningless to it — it is invalid. gpu_zstd_codes
    # feeds a Huffman<uint16> with a 4096-entry book, and f32 bytes reinterpreted as
    # uint16 overflow it ("out-of-range symbol(s) detected"), failing the probe and
    # dropping the preset out of the stage map that `benchkit stale` depends on. So
    # match the synthetic data to what the preset's literals coder declares.
    bklen = _literal_bklen(tpl) if not tpl.lossy_stages() else None
    with open(raw, "wb") as fh:
        if bklen:
            fh.write(struct.pack(f"<{2 * n}H", *[(i % 997) % bklen for i in range(2 * n)]))
        else:
            fh.write(struct.pack(f"<{n}f", *[(i % 997) * 0.01 for i in range(n)]))
    cfg = workdir / f"{preset.stem}.toml"
    # Coder-only (lossless) presets declare no error_bound, so there is nothing to
    # render — ship them verbatim. They still need to appear in the stage map:
    # `benchkit stale` uses it to answer "which cells does a change to GPULZ /
    # Huffman / ANS invalidate?", and those are exactly the stages these presets
    # are made of. Skipping them would silently under-report the stale set.
    # See configs/pipelines/gpu_zstd_lossless.toml and DESIGN.md D31/D32.
    if tpl.lossy_stages():
        cfg.write_text(tpl.render(1e-3, "NOA", dims=dims, input_size=n * 4, dtype="f32"))
    else:
        cfg.write_text(tpl.text)

    report = workdir / f"{preset.stem}.json"
    argv = [cli, "-b", "-i", str(raw), "-l", f"{dims[0]}x{dims[1]}x{dims[2]}",
            "-t", "f32", "-c", str(cfg), "--runs", "2", "--compare", str(raw),
            "--report-json", str(report)]
    proc = subprocess.run(argv, capture_output=True, text=True)
    if proc.returncode != 0 or not report.exists():
        return {"error": (proc.stderr or proc.stdout or "no report").strip()[:400]}

    rep = json.loads(report.read_text())
    phases: dict[str, list[str]] = {}
    for s in rep.get("stages", []):
        phases.setdefault(s["phase"], []).append(s["name"])
    return {
        "compress": phases.get("compress", []),
        "decompress": phases.get("decompress", []),
        "stages": sorted({s["name"] for s in rep.get("stages", [])}),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cli", default=os.environ.get("FZGMOD_CLI"),
                    help="path to fzgmod-cli (default: $FZGMOD_CLI)")
    ap.add_argument("--check", action="store_true",
                    help="compare against the committed index and exit non-zero on drift")
    args = ap.parse_args()
    if not args.cli:
        print("error: set FZGMOD_CLI or pass --cli", file=sys.stderr)
        return 2

    presets = sorted(PRESET_DIR.glob("*.toml"))
    if not presets:
        print(f"error: no presets under {PRESET_DIR}", file=sys.stderr)
        return 2

    index, failed = {}, []
    with tempfile.TemporaryDirectory() as td:
        wd = Path(td)
        for p in presets:
            r = probe(args.cli, p, wd)
            if "error" in r:
                failed.append((p.name, r["error"]))
                print(f"  {p.name:28s} FAILED: {r['error'].splitlines()[0][:80]}")
                continue
            index[p.name] = r
            same = set(r["compress"]) == set(r["decompress"])
            flag = "" if same else "   << compress/decompress stage sets DIFFER"
            print(f"  {p.name:28s} {len(r['stages'])} stages: "
                  f"{', '.join(r['stages'])}{flag}")

    if failed:
        print(f"\n{len(failed)} preset(s) failed to probe; they are omitted from the "
              f"index rather than guessed.", file=sys.stderr)

    payload = {
        "_comment": (
            "Which stages each pipeline preset runs, probed from the real binary "
            "(scripts/probe_pipeline_stages.py). Used by `benchkit stale --stage` to "
            "find cells affected by a stage change without re-running them. "
            "Regenerate whenever a preset changes."
        ),
        "pipelines": index,
    }

    if args.check:
        if not OUT.exists():
            print(f"\n{OUT} does not exist yet", file=sys.stderr)
            return 1
        old = json.loads(OUT.read_text()).get("pipelines", {})
        drift = [k for k in set(old) | set(index)
                 if old.get(k, {}).get("stages") != index.get(k, {}).get("stages")]
        if drift:
            print(f"\nDRIFT in {len(drift)} preset(s): {', '.join(sorted(drift))}",
                  file=sys.stderr)
            return 1
        print(f"\n{OUT.name} is up to date ({len(index)} presets)")
        return 0

    OUT.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"\nwrote {OUT} ({len(index)} presets)")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
