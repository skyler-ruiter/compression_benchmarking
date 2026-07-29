"""Stage-level result invalidation — which cells does a changed stage affect?

A full-corpus sweep is ~20 h per machine; most FZGM changes touch one stage. This
maps stage -> affected cells so only those need re-running.

How it knows
------------
Result rows carry `stages[]`, but baselines captured before 2026-07-29 have
decompress entries only (FZGM's `setMemoryStrategy()` silently disabled profiling
on the compress DAG). Rather than re-run 9,816 cells to recover a list of *names*,
the mapping comes from `configs/pipeline_stages.json`, produced by
`scripts/probe_pipeline_stages.py` running each preset once on a few KB of
synthetic data.

That substitution is sound because which stages a preset runs is a property of the
preset, not of the field: across the 4,860 FZGM rows of the 2026-07-29 sweep, every
preset TOML maps to exactly one stage set, and the probed index reproduces all 13
of them exactly. It is also strictly better than reading `stages[]` off the rows,
which would report only what that build happened to emit.

Compress vs decompress does not matter for this question: the probe found the two
phases run the identical stage *set* for all 20 presets (they are the same DAG
inverted). Order differs on branching DAGs — `cusz_hi_tp` compresses
GInterp->Zigzag->Bitshuffle->RRE->Merge->Bitshuffle->RRE->RZE and does not simply
reverse that on the way back — but membership is what invalidation turns on.

Deliberately conservative
-------------------------
A cell is stale if its pipeline *contains* the stage, whether or not the change
affected that cell's numbers. The alternative (proving a change is inert for some
inputs) is not something this can know. Two consequences worth stating:

- A stage change can perturb pipelines that do NOT contain it, via shared buffers
  or memory-pool sizing. Rare, but it means this answers "what probably changed",
  not "what provably changed" — periodic full sweeps stay necessary.
- Native rows have no stage decomposition and never will (opaque third-party
  binaries). They are never reported stale; rebuilding a reference invalidates
  them wholesale, which is coarse and rare.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

_DEFAULT_INDEX = Path(__file__).resolve().parent.parent / "configs" / "pipeline_stages.json"


def load_index(path: str | Path | None = None) -> dict[str, list[str]]:
    """preset filename -> sorted stage names."""
    p = Path(path) if path else _DEFAULT_INDEX
    if not p.exists():
        raise FileNotFoundError(
            f"{p} not found — generate it with scripts/probe_pipeline_stages.py")
    data = json.loads(p.read_text())
    return {k: v.get("stages", []) for k, v in data.get("pipelines", {}).items()}


def row_stages(row: dict, index: dict[str, list[str]]) -> list[str]:
    """Stages a row's pipeline runs, or [] for rows with no stage decomposition.

    Prefers the index over the row's own `stages[]`: the index covers both phases
    even for rows recorded before compress-phase timings worked, and it describes
    the preset rather than one build's emissions. Falls back to the row when the
    preset is absent from the index (e.g. a one-off pipeline outside configs/).
    """
    if row.get("compressor") != "fzgm":
        return []
    ref = row.get("pipeline_ref") or ""
    stages = index.get(os.path.basename(ref))
    if stages:
        return list(stages)
    return sorted({s.get("name") for s in (row.get("stages") or []) if s.get("name")})


def stage_usage(rows: list[dict], index: dict[str, list[str]]) -> dict[str, int]:
    """stage name -> number of cells that run it."""
    counts: dict[str, int] = {}
    for r in rows:
        for s in row_stages(r, index):
            counts[s] = counts.get(s, 0) + 1
    return counts


def stale_cells(rows: list[dict], stages: list[str],
                index: dict[str, list[str]]) -> list[dict]:
    """Rows whose pipeline contains ANY of `stages` (case-insensitive)."""
    want = {s.lower() for s in stages}
    return [r for r in rows if want & {s.lower() for s in row_stages(r, index)}]


def print_stage_usage(rows: list[dict], index: dict[str, list[str]]) -> None:
    counts = stage_usage(rows, index)
    fzgm = sum(1 for r in rows if r.get("compressor") == "fzgm")
    other = len(rows) - fzgm
    if not counts:
        print("(no FZGM rows with a known pipeline)")
        return
    print(f"{'stage':20s} {'cells':>7s}  {'% of FZGM':>9s}")
    print("-" * 40)
    for name, n in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])):
        print(f"{name:20s} {n:7d}  {100.0 * n / fzgm:8.1f}%")
    print(f"\n{fzgm} FZGM cells; {other} native cells have no stage decomposition "
          f"and are never reported stale.")


def build_fingerprints(cli: str) -> dict[str, str]:
    """{stage: fingerprint} for a given fzgmod-cli build, via --list-stages=json."""
    import subprocess
    proc = subprocess.run([cli, "--list-stages=json"], capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"{cli} --list-stages=json failed (exit {proc.returncode}): "
            f"{(proc.stderr or proc.stdout).strip()[:200]}\n"
            "A build predating stage fingerprints cannot answer this — rebuild FZGM.")
    data = json.loads(proc.stdout)
    return {s["name"]: s.get("fingerprint", "") for s in data.get("stages", [])}


def drifted_stages(rows: list[dict], current: dict[str, str]) -> dict[str, set[str]]:
    """stage -> the recorded fingerprints that disagree with `current`.

    Rows carrying no `stage_versions` are skipped rather than assumed stale: a
    missing fingerprint means "this build did not report one", which is not
    evidence of change. They are counted separately by the caller so the gap is
    visible instead of silently treated as clean.
    """
    drift: dict[str, set[str]] = {}
    for r in rows:
        for stage, fp in (r.get("stage_versions") or {}).items():
            now = current.get(stage)
            if now and fp and fp != now:
                drift.setdefault(stage, set()).add(fp)
    return drift


def print_against_build(rows: list[dict], cli: str,
                        index: dict[str, list[str]], show: int = 20) -> None:
    current = build_fingerprints(cli)
    fzgm = [r for r in rows if r.get("compressor") == "fzgm"]
    with_fp = [r for r in fzgm if r.get("stage_versions")]

    print(f"Comparing {len(rows)} rows against {cli}\n")
    if not with_fp:
        print(f"None of the {len(fzgm)} FZGM rows carry stage fingerprints, so no "
              f"comparison is possible.\nThey predate the feature; re-run them, or "
              f"invalidate by name with --stage.")
        return
    if len(with_fp) < len(fzgm):
        print(f"NOTE: {len(fzgm) - len(with_fp)} of {len(fzgm)} FZGM rows carry no "
              f"fingerprints and are\n      excluded from this comparison — absent "
              f"data is not evidence of no change.\n")

    drift = drifted_stages(with_fp, current)
    if not drift:
        print(f"No drift: every fingerprint in {len(with_fp)} rows matches the build.")
        return

    print(f"{len(drift)} stage(s) changed since these rows were recorded:")
    for stage, olds in sorted(drift.items()):
        old = ", ".join(sorted(olds))
        print(f"  {stage:20s} recorded {old}  ->  now {current.get(stage)}")
    print()
    print_stale(rows, sorted(drift), index, show=show)


def print_stale(rows: list[dict], stages: list[str], index: dict[str, list[str]],
                show: int = 20) -> None:
    hits = stale_cells(rows, stages, index)
    fzgm = sum(1 for r in rows if r.get("compressor") == "fzgm")
    label = ", ".join(stages)
    print(f"Stage(s) changed: {label}")
    print(f"Stale cells: {len(hits)} of {len(rows)} total "
          f"({fzgm} FZGM) — {100.0 * len(hits) / len(rows) if rows else 0:.1f}% of the sweep\n")
    if not hits:
        known = sorted(stage_usage(rows, index))
        print(f"No pipeline in this session runs that stage. Known: {', '.join(known)}")
        return

    by_pipe: dict[str, int] = {}
    for r in hits:
        by_pipe[os.path.basename(r.get("pipeline_ref") or "?")] = \
            by_pipe.get(os.path.basename(r.get("pipeline_ref") or "?"), 0) + 1
    print("affected pipelines:")
    for p, n in sorted(by_pipe.items(), key=lambda kv: -kv[1]):
        print(f"  {p:28s} {n:6d} cells")

    print(f"\nfirst {min(show, len(hits))} stale cell keys:")
    for r in hits[:show]:
        print(f"  {r.get('cell_key')}")
    if len(hits) > show:
        print(f"  ... and {len(hits) - show} more")
    print("\nRe-run them by re-submitting the same experiment with the SAME "
          "--session-id after deleting these rows,\nor keep both and let `merge` "
          "prefer the newest per cell_key.")
