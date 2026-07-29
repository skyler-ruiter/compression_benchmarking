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


def probe(cli: str, preset: Path, workdir: Path) -> dict:
    dims = dims_for(preset.stem)
    n = dims[0] * dims[1] * dims[2]
    raw = workdir / f"{preset.stem}.f32"
    # A smooth ramp: compresses, and avoids the all-constant path that some stages
    # short-circuit (which could hide a stage from the trace).
    import struct
    with open(raw, "wb") as fh:
        fh.write(struct.pack(f"<{n}f", *[(i % 997) * 0.01 for i in range(n)]))

    tpl = PipelineToml.load(preset)
    cfg = workdir / f"{preset.stem}.toml"
    cfg.write_text(tpl.render(1e-3, "NOA", dims=dims, input_size=n * 4, dtype="f32"))

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
