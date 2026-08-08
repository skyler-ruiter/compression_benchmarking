#!/usr/bin/env python3
"""B3 — framework-overhead / additivity fit for the FZGM DAG.

Question: does end-to-end DAG time equal the sum of its stages plus a fixed cost?

    e2e = a + b*n_stages + sum(stage_device_ms)

`a` is per-invocation DAG cost, `b` is per-stage dispatch cost. Both are fitted, not
assumed; the sum term is fixed at coefficient 1 by construction (we regress the
*residual* e2e - sum on n_stages), so a poor fit shows up as scatter rather than
being absorbed into a free slope.

SCOPE LIMIT — read before citing any number this prints:
  The stage-attribution corpus carries `device_ms` only. There is no host-inclusive
  field (the harness gained `compress_host_ms_median` later, D33). So this answers
  "is the GPU timeline accounted for by stage kernels?" It is NOT the attribution
  guard that separates "the framework is free" from "our kernels are fast" -- that
  needs host-inclusive timing and therefore a re-run.

Usage:
    python3 tools/b3_additivity.py <session_dir> [--phase compress|decompress]
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path


METRIC = ["device"]


def load(session: Path, phase: str):
    """Return usable rows, plus counts of what the validity gate dropped."""
    rows, dropped = [], defaultdict(int)
    with open(session / "runs.jsonl") as fh:
        for line in fh:
            r = json.loads(line)
            # Validity gate: `status: ok` alone is not a usable row.
            if r.get("status") != "ok":
                dropped["status != ok"] += 1
                continue
            if not r.get("timing_reliable", False):
                dropped["timing_reliable false"] += 1
                continue
            if not r.get(f"{phase}_stable", False):
                dropped[f"{phase}_stable false"] += 1
                continue
            e2e = r.get(f"{phase}_{METRIC[0]}_ms_median")
            stages = [s for s in (r.get("stages") or []) if s.get("phase") == phase]
            if e2e is None or not stages:
                dropped["missing e2e or stages"] += 1
                continue
            rows.append({
                "pipeline": Path(r["pipeline_ref"]).stem,
                "dataset": r["dataset"],
                "field": r.get("field"),
                "e2e_ms": float(e2e),
                "sum_ms": sum(float(s["device_ms"]) for s in stages),
                "n_stages": len(stages),
                "bytes": r.get("original_bytes"),
            })
    return rows, dropped


def ols(xs, ys):
    """Least squares y = a + b*x. Returns (a, b, r2)."""
    n = len(xs)
    if n < 2:
        return float("nan"), float("nan"), float("nan")
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx == 0:  # no variation in x -- slope undefined, report mean as intercept
        return my, float("nan"), float("nan")
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    a = my - b * mx
    sst = sum((y - my) ** 2 for y in ys)
    sse = sum((y - (a + b * x)) ** 2 for x, y in zip(xs, ys))
    r2 = 1 - sse / sst if sst > 0 else float("nan")
    return a, b, r2


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("session", type=Path)
    ap.add_argument("--phase", default="compress", choices=["compress", "decompress"])
    ap.add_argument("--metric", default="device", choices=["device", "host"],
                    help="device = dag->execute() event bracket; host = host_wall_ms, "
                         "which includes framework work outside that bracket. "
                         "host is the one that answers the attribution question.")
    args = ap.parse_args()
    METRIC[0] = args.metric

    rows, dropped = load(args.session, args.phase)
    print(f"B3 additivity — {args.session.name} — phase={args.phase} metric={args.metric}")
    print(f"usable rows: {len(rows)}" + (f"   dropped: {dict(dropped)}" if dropped else ""))
    if args.metric == "device":
        print("NOTE: device_ms. Framework work outside the event bracket is invisible here.\n")
    else:
        print("NOTE: host_wall_ms. Residual = stage kernels + ALL host-side framework cost.\n")
    if not rows:
        return

    for r in rows:
        r["resid_us"] = (r["e2e_ms"] - r["sum_ms"]) * 1000.0
        r["resid_pct"] = 100.0 * (r["e2e_ms"] - r["sum_ms"]) / r["e2e_ms"]

    # ---- per pipeline -------------------------------------------------------
    # Split by pipeline because additivity only holds for serial execution: on a
    # branching pipeline stages run concurrently on separate streams, so the sum
    # can exceed e2e and the residual goes negative. A negative residual is a
    # measurement of real overlap, not an error -- it is evidence for B4.
    print(f"{'pipeline':<26}{'n':>4}{'stg':>5}{'e2e ms':>10}{'sum ms':>10}"
          f"{'resid us':>11}{'resid %':>9}")
    print("-" * 75)
    by_pipe = defaultdict(list)
    for r in rows:
        by_pipe[r["pipeline"]].append(r)
    for pipe in sorted(by_pipe, key=lambda p: statistics.median(
            x["resid_us"] for x in by_pipe[p])):
        g = by_pipe[pipe]
        med = statistics.median(x["resid_us"] for x in g)
        print(f"{pipe:<26}{len(g):>4}{statistics.median(x['n_stages'] for x in g):>5}"
              f"{statistics.median(x['e2e_ms'] for x in g):>10.3f}"
              f"{statistics.median(x['sum_ms'] for x in g):>10.3f}"
              f"{med:>11.1f}"
              f"{statistics.median(x['resid_pct'] for x in g):>9.1f}")

    # ---- separate concurrent pipelines --------------------------------------
    # A pipeline whose *median* residual is negative ran stages concurrently, so
    # additivity does not apply to it by construction. Classify per pipeline
    # rather than per row: a single negative row is timing jitter around zero,
    # a negative median is real overlap.
    concurrent = {p for p, g in by_pipe.items()
                  if statistics.median(x["resid_us"] for x in g) < 0}
    serial = [r for r in rows if r["pipeline"] not in concurrent]
    if concurrent:
        print(f"\nconcurrent pipelines (median residual < 0, additivity N/A): "
              f"{', '.join(sorted(concurrent))}")
        for p in sorted(concurrent):
            g = by_pipe[p]
            print(f"  {p}: sum exceeds e2e by "
                  f"{-statistics.median(x['resid_us'] for x in g)/1000:.2f} ms median "
                  f"({-statistics.median(x['resid_pct'] for x in g):.0f}% of e2e) "
                  f"-- measured overlap, evidence for B4")

    # ---- does the fixed cost scale with stage count? ------------------------
    # This is the actual B3 question. If the residual is flat in n_stages, the DAG
    # charges a per-invocation cost and no per-stage cost.
    print(f"\nserial rows only (n={len(serial)}) -- residual vs stage count")
    print(f"{'n_stages':>9}{'rows':>6}{'median us':>11}{'p25':>8}{'p75':>8}  pipelines")
    by_n = defaultdict(list)
    for r in serial:
        by_n[r["n_stages"]].append(r)
    for n in sorted(by_n):
        g = sorted(by_n[n], key=lambda x: x["resid_us"])
        names = sorted({x["pipeline"] for x in g})
        print(f"{n:>9}{len(g):>6}"
              f"{statistics.median(x['resid_us'] for x in g):>11.1f}"
              f"{g[len(g)//4]['resid_us']:>8.1f}{g[3*len(g)//4]['resid_us']:>8.1f}"
              f"  {', '.join(names[:3])}{'...' if len(names) > 3 else ''}")

    a, b, r2 = ols([r["n_stages"] for r in serial], [r["resid_us"] for r in serial])
    print(f"\nfit  resid_us = a + b*n_stages   ->  a={a:.1f} us,  b={b:.1f} us/stage,"
          f"  R2={r2:.3f}")
    print("  (low R2 with near-flat medians = fixed per-invocation cost, no per-stage term)")

    allr = sorted(r["resid_us"] for r in serial)
    q = lambda p: allr[min(int(len(allr) * p), len(allr) - 1)]
    print(f"\nserial residual (us): min {allr[0]:.1f}  p25 {q(.25):.1f}  "
          f"median {q(.5):.1f}  p75 {q(.75):.1f}  max {allr[-1]:.1f}")
    pct = sorted(r["resid_pct"] for r in serial)
    qp = lambda p: pct[min(int(len(pct) * p), len(pct) - 1)]
    print(f"as % of e2e:          min {pct[0]:.1f}  p25 {qp(.25):.1f}  "
          f"median {qp(.5):.1f}  p75 {qp(.75):.1f}  max {pct[-1]:.1f}")


if __name__ == "__main__":
    main()
