"""benchkit CLI.

    python -m benchkit run      configs/experiments/smoke.yaml
        [--datasets configs/datasets.yaml] [--results-root DIR]
        [--session-id ID] [--shard k/N]
    python -m benchkit merge    results/<session>/        # combine shard files -> runs.jsonl
    python -m benchkit report   results/<session>/        # print the table
        [--aggregate]                                     # roll fields up into one CR per pipeline
    python -m benchkit download [DATA_DIR]                # fetch SDRBench datasets

HPC: a SLURM job array runs `--shard $SLURM_ARRAY_TASK_ID/$N --session-id $SLURM_JOB_ID`;
each task writes its own shard file, then one `merge` combines them. Re-running the same
--session-id resumes (completed cells are skipped). See scripts/submit.slurm.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from . import invalidate, validity
from .analysis import _AGG_GROUP_KEYS, print_aggregate_table, print_table
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


def _stale_keys(args: argparse.Namespace, site: Site) -> set[str]:
    """cell_keys to re-measure, from an existing session's rows."""
    if not args.session_id:
        raise SystemExit("--only-stale needs --session-id: it re-measures cells in an "
                         "existing session, so there must be one to read rows from.")
    if not (args.stage or args.against_build):
        raise SystemExit("--only-stale needs --stage NAME or --against-build FZGMOD_CLI "
                         "to say WHICH cells are stale.")
    session = Path(site.results_root) / args.session_id
    if not session.is_dir():
        raise SystemExit(f"--only-stale: no such session {session}")
    rows = ResultStore(session.parent, session.name).load_rows()
    index = invalidate.load_index(None)

    stages = list(args.stage or [])
    if args.against_build:
        current = invalidate.build_fingerprints(args.against_build)
        drift = invalidate.drifted_stages(rows, current)
        if not drift:
            print("[stale] no stage fingerprints differ from that build — nothing to "
                  "re-measure.")
        stages += sorted(drift)
    if not stages:
        return set()

    hits = invalidate.stale_cells(rows, stages, index)
    keys = {r["cell_key"] for r in hits if r.get("cell_key")}
    print(f"[stale] stage(s) {', '.join(sorted(set(stages)))} -> {len(keys)} cells "
          f"to re-measure")
    return keys


def cmd_run(args: argparse.Namespace) -> int:
    site = Site.load(args.results_root)
    site.export_env()                          # publish FZGMOD_CLI for the adapter
    cfg = ExperimentConfig.load(args.experiment)
    catalog = DatasetCatalog.load(args.datasets)
    only_stale = _stale_keys(args, site) if args.only_stale else None
    if only_stale is not None and not only_stale:
        print("[stale] nothing to re-measure; exiting without touching the session.")
        return 0
    store = run_experiment(cfg, catalog, site.results_root, REPO_ROOT,
                           session_id=args.session_id, shard=_parse_shard(args.shard),
                           only_stale=only_stale)
    print()
    print_table(store.load_rows())
    print(f"\n[done] rows -> {store.runs_path}")
    return 0


def cmd_merge(args: argparse.Namespace) -> int:
    """Combine shard files into runs.jsonl, one row per cell_key.

    Three precedence rules, in order. They were arrived at by two different
    investigations and all three are load-bearing:

    1. **An `ok` row always beats a non-ok row** (D29), whatever their positions.
       Resume *appends*, so a cell that failed and was later retried has its stale
       `fail` row ahead of the good retry in the same shard file. First-wins
       resurrected the failure and discarded the recovered cell — on
       fullcorpus-delta-mi100 that would have reported 701 failures instead of 108,
       throwing away 593 recovered cells at a correct-looking row count and exit 0.
    2. **Among rows of equal standing, later in the SAME file wins** (D30).
       `run --only-stale` appends a fresh measurement rather than editing in place,
       so the newest row is the one to keep, and the superseded row stays in the raw
       file — which is what keeps "this stage got 1.8x faster" answerable afterwards.
       Ordering is positional because rows carry no timestamp.
    3. **Across files, shard files beat runs.jsonl.** runs.jsonl is the *output* of a
       previous merge, so on a re-merge it holds older data than the shards; letting a
       later file win would resurrect exactly what rule 2 just superseded.

    Rule 1 deliberately outranks rules 2 and 3: recovering a cell matters more than
    positional recency, and a failure is never evidence that an earlier success was
    wrong.
    """
    session = Path(args.session_dir)
    files = sorted(session.glob("runs.shard-*.jsonl")) + sorted(session.glob("runs.jsonl"))
    best: dict[str, tuple[int, dict]] = {}      # cell_key -> (file rank, row)
    order: list[str] = []
    superseded = recovered = 0
    for rank, f in enumerate(files):
        for line in f.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            k = row.get("cell_key") or row.get("run_id")
            cur = best.get(k)
            if cur is None:
                best[k] = (rank, row)
                order.append(k)
                continue
            cur_rank, cur_row = cur
            cur_ok = cur_row.get("status") == "ok"
            new_ok = row.get("status") == "ok"
            if cur_ok != new_ok:                # rule 1
                if new_ok:
                    best[k] = (rank, row)
                    recovered += 1
                continue
            if rank == cur_rank:                # rule 2
                best[k] = (rank, row)
                superseded += 1
            # else: rule 3 — keep the earlier file's row

    rows = [best[k][1] for k in order]
    out = session / "runs.jsonl"
    with open(out, "w") as fh:
        for row in rows:
            fh.write(json.dumps(row, default=str) + "\n")
    notes = []
    if recovered:
        notes.append(f"{recovered} failed row(s) replaced by a successful retry")
    if superseded:
        notes.append(f"{superseded} superseded by a newer re-run")
    note = f" ({'; '.join(notes)})" if notes else ""
    print(f"[merge] {len(rows)} unique rows -> {out}{note}")
    print_table(rows)
    return 0


def _load_rows(target_str: str) -> list[dict]:
    target = Path(target_str)
    if target.is_dir():
        return ResultStore(target.parent, target.name).load_rows()
    return [json.loads(l) for l in target.read_text().splitlines() if l.strip()]


def cmd_stale(args: argparse.Namespace) -> int:
    rows = _load_rows(args.target)
    index = invalidate.load_index(args.index)
    if args.against_build:
        invalidate.print_against_build(rows, args.against_build, index, show=args.show)
    elif args.stage:
        invalidate.print_stale(rows, args.stage, index, show=args.show)
    else:
        invalidate.print_stage_usage(rows, index)
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    rows = _load_rows(args.target)
    if args.aggregate:
        # --by-dataset inserts `dataset` as the leading group key, so each dataset gets
        # its own aggregate CR per pipeline instead of every dataset being pooled into
        # one number. Pooling across datasets is almost never what you want in a paper
        # table: ratio-of-sums is size-weighted, so a 21 GB CESMATM would simply
        # determine the "overall" figure and the other seven datasets would not show up
        # in it. Without the flag the original pooled behaviour is unchanged.
        keys = (("dataset",) + _AGG_GROUP_KEYS) if args.by_dataset else _AGG_GROUP_KEYS
        print_aggregate_table(rows, keys, gate=not args.no_gate)
    elif args.exclusions:
        print(validity.exclusion_report(rows))
    else:
        print_table(rows)
    return 0


def cmd_download(args: argparse.Namespace) -> int:
    from .download import ALL_DATASETS, DATASET_KEYS, download_dataset

    if args.list:
        for ds in ALL_DATASETS:
            print(f"  {ds.key:<10}  {ds.name:<20}  {ds.dims:<20}  {ds.dtype}  {ds.num_fields} fields")
        return 0

    data_dir = Path(args.data_dir)
    tarball_dir = (
        Path(args.tarball_dir) if args.tarball_dir
        else data_dir.parent / "sdrbench_tarballs"
    )
    keys = args.datasets or DATASET_KEYS

    unknown = [k for k in keys if k not in DATASET_KEYS]
    if unknown:
        raise SystemExit(f"Unknown dataset key(s): {', '.join(unknown)}. "
                         f"Valid keys: {', '.join(DATASET_KEYS)}")

    data_dir.mkdir(parents=True, exist_ok=True)
    tarball_dir.mkdir(parents=True, exist_ok=True)

    print(f"[config] data_dir    = {data_dir}")
    print(f"[config] tarball_dir = {tarball_dir}")
    print(f"[config] datasets    = {' '.join(keys)}")
    print()

    for key in keys:
        download_dataset(key, data_dir, tarball_dir)

    print()
    print(f"[summary] {data_dir}:")
    for d in sorted(data_dir.iterdir()):
        if not d.is_dir():
            continue
        n = sum(1 for f in d.rglob("*") if f.is_file() and f.name != "metadata.yaml")
        meta = ""
        mf = d / "metadata.yaml"
        if mf.exists():
            for line in mf.read_text().splitlines():
                if line.startswith("dataset:"):
                    meta = f"  ({line.split(':', 1)[1].strip()})"
                    break
        print(f"  {d.name:<35} {n} file(s){meta}")
    print()
    print(f"Tarballs cached in: {tarball_dir}")
    print(f"Delete once verified: rm -rf '{tarball_dir}'")
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
    r.add_argument("--only-stale", action="store_true",
                   help="re-measure ONLY the cells invalidated by a changed FZGM stage, "
                        "instead of the whole matrix. Requires --session-id and either "
                        "--stage or --against-build. New rows are appended; `merge` "
                        "then keeps the newest per cell")
    r.add_argument("--stage", action="append", metavar="NAME",
                   help="with --only-stale: the stage that changed (repeatable)")
    r.add_argument("--against-build", metavar="FZGMOD_CLI",
                   help="with --only-stale: detect changed stages by comparing recorded "
                        "fingerprints against this build, instead of naming them")
    r.set_defaults(func=cmd_run)

    m = sub.add_parser("merge", help="combine shard files into runs.jsonl")
    m.add_argument("session_dir")
    m.set_defaults(func=cmd_merge)

    rep = sub.add_parser("report", help="print a table from a session dir or runs.jsonl")
    rep.add_argument("target")
    rep.add_argument("--aggregate", action="store_true",
                     help="roll multi-field cells up into one aggregate CR per "
                          "compressor/variant/pipeline/error_bound "
                          "(ratio-of-sums and geometric mean, see docs/DESIGN.md)")
    rep.add_argument("--by-dataset", action="store_true",
                     help="with --aggregate, group by dataset as well, so each dataset "
                          "gets its own CR per pipeline instead of one pooled number "
                          "(the pooled figure is size-weighted and would be decided by "
                          "the largest dataset alone)")
    rep.add_argument("--exclusions", action="store_true",
                     help="print only the validity-gate audit: how many rows are "
                          "excluded from reported means, under which reason code, and "
                          "why (see benchkit/validity.py)")
    rep.add_argument("--no-gate", action="store_true",
                     help="with --aggregate, DISABLE the validity gate and average every "
                          "ok row, including constant fields, expansions and severe "
                          "error-bound misses. For reproducing pre-gate numbers only — "
                          "the resulting means are not defensible")
    rep.set_defaults(func=cmd_report)

    st = sub.add_parser("stale",
                        help="which cells does a changed FZGM stage invalidate?")
    st.add_argument("target", help="session dir or runs.jsonl")
    st.add_argument("--stage", action="append", metavar="NAME",
                    help="stage that changed, e.g. AdaptiveBitpack (repeatable). "
                         "Omit to list every stage and how many cells use it")
    st.add_argument("--against-build", metavar="FZGMOD_CLI",
                    help="compare each row's recorded stage fingerprints against this "
                         "build and report which stages changed (needs an FZGM build "
                         "with --list-stages=json)")
    st.add_argument("--show", type=int, default=20,
                    help="how many stale cell keys to print (default 20)")
    st.add_argument("--index", default=None,
                    help="pipeline->stages index (default configs/pipeline_stages.json; "
                         "regenerate with scripts/probe_pipeline_stages.py)")
    st.set_defaults(func=cmd_stale)

    from .download import DATASET_KEYS
    _default_data_dir = os.environ.get("BENCHKIT_DATA_ROOT", str(Path.cwd() / "sdrbench_data"))
    dl = sub.add_parser("download", help="download SDRBench datasets from the Globus mirror")
    dl.add_argument("data_dir", nargs="?", default=_default_data_dir,
                    help="where extracted data goes (default: $BENCHKIT_DATA_ROOT or ./sdrbench_data)")
    dl.add_argument("--tarball-dir", default=None,
                    help="where .tar.gz files cache (default: DATA_DIR/../sdrbench_tarballs)")
    dl.add_argument("--datasets", nargs="+", metavar="KEY", default=None,
                    help=f"subset to download (default: all). Keys: {', '.join(DATASET_KEYS)}")
    dl.add_argument("--list", action="store_true",
                    help="list available dataset keys and exit")
    dl.set_defaults(func=cmd_download)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
