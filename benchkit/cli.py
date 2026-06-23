"""benchkit CLI.

    python -m benchkit run    configs/experiments/smoke.yaml
        [--datasets configs/datasets.yaml] [--results-root DIR]
        [--session-id ID] [--shard k/N]
    python -m benchkit merge  results/<session>/        # combine shard files -> runs.jsonl
    python -m benchkit report results/<session>/        # print the table

HPC: a SLURM job array runs `--shard $SLURM_ARRAY_TASK_ID/$N --session-id $SLURM_JOB_ID`;
each task writes its own shard file, then one `merge` combines them. Re-running the same
--session-id resumes (completed cells are skipped). See scripts/submit.slurm.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .analysis import print_table
from .config import DatasetCatalog, ExperimentConfig
from .runner import run_experiment
from .site import Site
from .store import ResultStore

REPO_ROOT = Path(__file__).resolve().parent.parent


def _parse_shard(s: str | None) -> tuple[int, int] | None:
    if not s:
        return None
    try:
        k, n = (int(x) for x in s.split("/"))
    except ValueError:
        raise SystemExit(f"--shard must be 'k/N' (0-based k < N), got {s!r}")
    if not (0 <= k < n):
        raise SystemExit(f"--shard k/N requires 0 <= k < N, got {s!r}")
    return k, n


def cmd_run(args: argparse.Namespace) -> int:
    site = Site.load(args.results_root)
    site.export_env()                          # publish FZGMOD_CLI for the adapter
    cfg = ExperimentConfig.load(args.experiment)
    catalog = DatasetCatalog.load(args.datasets)
    store = run_experiment(cfg, catalog, site.results_root, REPO_ROOT,
                           session_id=args.session_id, shard=_parse_shard(args.shard))
    print()
    print_table(store.load_rows())
    print(f"\n[done] rows -> {store.runs_path}")
    return 0


def cmd_merge(args: argparse.Namespace) -> int:
    session = Path(args.session_dir)
    rows, seen = [], set()
    for f in sorted(session.glob("runs.shard-*.jsonl")) + sorted(session.glob("runs.jsonl")):
        for line in f.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            k = row.get("cell_key") or row.get("run_id")
            if k in seen:                      # dedupe (e.g. a retried cell)
                continue
            seen.add(k)
            rows.append(row)
    out = session / "runs.jsonl"
    with open(out, "w") as fh:
        for row in rows:
            fh.write(json.dumps(row, default=str) + "\n")
    print(f"[merge] {len(rows)} unique rows -> {out}")
    print_table(rows)
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    target = Path(args.target)
    if target.is_dir():
        rows = ResultStore(target.parent, target.name).load_rows()
    else:
        rows = [json.loads(l) for l in target.read_text().splitlines() if l.strip()]
    print_table(rows)
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="benchkit")
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="run an experiment matrix")
    r.add_argument("experiment", help="path to experiment YAML")
    r.add_argument("--datasets", default=str(REPO_ROOT / "configs" / "datasets.yaml"))
    r.add_argument("--results-root", default=None,
                   help="override results root (else env BENCHKIT_RESULTS_ROOT / site config)")
    r.add_argument("--session-id", default=None,
                   help="reuse/resume a session (e.g. $SLURM_JOB_ID for array jobs)")
    r.add_argument("--shard", default=None, help="k/N — run only cells where index %% N == k")
    r.set_defaults(func=cmd_run)

    m = sub.add_parser("merge", help="combine shard files into runs.jsonl")
    m.add_argument("session_dir")
    m.set_defaults(func=cmd_merge)

    rep = sub.add_parser("report", help="print a table from a session dir or runs.jsonl")
    rep.add_argument("target")
    rep.set_defaults(func=cmd_report)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
