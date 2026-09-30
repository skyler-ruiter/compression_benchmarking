#!/usr/bin/env python3
"""Targeted diagnostic cells for explaining differences in the peak-memory study.

Each diagnostic re-measures one planned cell with exactly one factor changed and
appends a row to <session>/diagnostics.jsonl, so every explanation in the artifact
is backed by a measurement rather than an assertion:

  binary     : a different fzgmod-cli (e.g. the September build_paper tree)
  from_toml  : the pipeline TOML's own literal bound (ABS 1e-3), as the September
               standalone script ran it, instead of benchkit's rel_range rendering
  pool_off   : FZ_FORCE_MEMPOOL_FALLBACK=1 (plain cudaMalloc/cudaFree, no pool retention)
  runs_sweep : --runs 2/3/11 on PFPL HACC Auto (does the growth plateau?)
  runs6      : fzgmod-cli -b --runs 6 instead of 1 (benchkit's 1 warmup + 5 timed shape)

  python tools/peak_memory/diagnose.py <session> --fzgm-cli <pinned> [--alt-cli <other>]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_peak_memory as rpm  # noqa: E402

CASES = [
    # (family, arm, field)
    ("pfpl", "fzgm:planned:auto", "HACC/vx"),
    ("cuszp2_outlier", "fzgm:planned:auto", "HACC/vx"),
    ("pfpl", "fzgm:planned:auto", "CESMATM-3D/CLDICE"),
    ("cuszp2_outlier", "fzgm:planned:off", "HACC/vx"),
]
POOL_CASES = [
    ("cuszp2_outlier", "fzgm:planned_split:off", "MIRANDA/density"),
    ("cuszp2_outlier", "fzgm:planned:off", "MIRANDA/density"),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("session")
    ap.add_argument("--fzgm-cli", required=True)
    ap.add_argument("--alt-cli")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--only", choices=["runs6", "runs_sweep"])
    a = ap.parse_args()
    session = Path(a.session).resolve()
    prov = json.loads((session / "session.json").read_text())
    cfg = yaml.safe_load(Path(prov["config_path"]).read_text())
    if rpm.sha256_file(Path(prov["config_path"])) != prov["config_sha256"]:
        raise SystemExit("config changed since the session ran")
    catalog = rpm.DatasetCatalog.load(rpm.REPO / "configs" / "datasets.yaml")
    pinned = rpm.snapshot_pipelines(cfg, session)
    cells = {c["cell_id"]: c for c in rpm.plan_cells(cfg, catalog, pinned=pinned)}
    nvml = rpm.Nvml(0)
    out = session / "diagnostics.jsonl"
    work = session / "diag_work"

    def run(tag, fam, arm, field, cli, cfg_over=None, env=None, n_runs=1):
        cell = dict(cells[f"{fam}|{arm}|{field}"], n_runs=n_runs)
        c2 = dict(cfg, **(cfg_over or {}))
        ns = argparse.Namespace(fzgm_cli=cli, no_probe=False, keep_work=False)
        for rep in range(a.reps):
            rpm.wait_for_idle_gpu(nvml, 21600)
            with rpm.env_override(**(env or {})):
                m = rpm.measure(cell, c2, ns, nvml, work)
            row = {"diagnostic": tag, "cell_id": cell["cell_id"], "rep": rep, "fzgm_cli": cli,
                   "error": c2["error"], "env": env or {}, "utc": rpm.now(),
                   **{k: m.get(k) for k in ("status", "nvml_peak_bytes", "probe_peak_bytes",
                                            "fzgm_self_peak_bytes", "max_stack_limit_bytes",
                                            "error")},
                   "per_process": [{"argv1": p["argv"][1], "nvml": p["nvml_peak_bytes"],
                                    "probe": (p.get("probe") or {}).get("peak_device_bytes")}
                                   for p in m["processes"]],
                   "fzgm_reports": m.get("fzgm_reports")}
            with open(out, "a") as fh:
                fh.write(json.dumps(row, default=str) + "\n")
            print(f"{tag:<10} {cell['cell_id']:<52} rep{rep} {m['status']} "
                  f"nvml={(m['nvml_peak_bytes'] or 0) / 1e6:9.1f} self={(m.get('fzgm_self_peak_bytes') or 0) / 1e6:8.1f}",
                  flush=True)

    if a.only == "runs_sweep":
        for n in (2, 3, 11):
            run(f"runs{n}", "pfpl", "fzgm:planned:auto", "HACC/vx", a.fzgm_cli, n_runs=n)
        return 0
    if a.only == "runs6":
        for fam, arm, field in CASES:
            run("runs6", fam, arm, field, a.fzgm_cli, n_runs=6)
        return 0
    for fam, arm, field in CASES:
        if a.alt_cli:
            run("binary", fam, arm, field, a.alt_cli)
        run("from_toml", fam, arm, field, a.fzgm_cli,
            cfg_over={"error": {"mode": "from_toml", "bound": None}})
    for fam, arm, field in POOL_CASES:
        run("pool_off", fam, arm, field, a.fzgm_cli, env={"FZ_FORCE_MEMPOOL_FALLBACK": "1"})
    return 0


if __name__ == "__main__":
    sys.exit(main())
