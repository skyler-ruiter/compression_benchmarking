"""benchkit CLI.

    python -m benchkit run    configs/experiments/smoke.yaml [--datasets configs/datasets.yaml]
    python -m benchkit report results/<session>/runs.jsonl
"""
from __future__ import annotations

import argparse
from pathlib import Path

from .analysis import print_table
from .config import DatasetCatalog, ExperimentConfig
from .runner import run_experiment

REPO_ROOT = Path(__file__).resolve().parent.parent


def cmd_run(args: argparse.Namespace) -> int:
    cfg = ExperimentConfig.load(args.experiment)
    catalog = DatasetCatalog.load(args.datasets)
    results_root = Path(args.results_root)
    store = run_experiment(cfg, catalog, results_root, REPO_ROOT)
    print()
    print_table(store.load_rows())
    print(f"\n[done] rows -> {store.runs_path}")
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    import json
    rows = [json.loads(l) for l in open(args.runs) if l.strip()]
    print_table(rows)
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="benchkit")
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="run an experiment matrix")
    r.add_argument("experiment", help="path to experiment YAML")
    r.add_argument("--datasets", default=str(REPO_ROOT / "configs" / "datasets.yaml"))
    r.add_argument("--results-root", default=str(REPO_ROOT / "results"))
    r.set_defaults(func=cmd_run)

    rep = sub.add_parser("report", help="print a table from a runs.jsonl")
    rep.add_argument("runs", help="path to runs.jsonl")
    rep.set_defaults(func=cmd_report)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
