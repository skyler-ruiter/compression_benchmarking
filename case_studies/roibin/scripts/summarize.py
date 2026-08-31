#!/usr/bin/env python3
"""
Aggregate the sweep CSVs into the study's two tables:

  1. Results table  — aggregate CR, ROI PSNR, background PSNR, throughput, per arm.
  2. Verification table — per-region error-bound satisfaction, per arm.

Aggregation rule: the *aggregate CR* is total original bytes / total compressed
bytes over all frames in the arm, not the mean of per-frame ratios. A mean of
ratios over-weights frames that compress well and is not the number that predicts
how much storage a run actually takes. Per-frame spread is reported separately as
the median and the 10th/90th percentiles.

Only rows with status == ok enter the aggregates, and the count of rows that failed
is reported alongside — an arm that crashed on half the corpus must not be presented
as if it merely had a lower ratio.
"""
import argparse
import csv
import sys
from collections import defaultdict

import numpy as np

ARM_LABEL = {
    "roibin_b1": "ROIBIN dual-eb (bin=1)",
    "roibin_b2": "ROIBIN binned  (bin=2)",
    "cuszhi": "cuSZ-Hi (tuned, global tight eb)",
    "pfpl": "PFPL (global tight eb)",
    "cuszp3": "cuSZp3 (global tight eb)",
}
ARM_ORDER = ["roibin_b1", "roibin_b2", "cuszhi", "pfpl", "cuszp3"]


def load(paths):
    rows = []
    for p in paths:
        with open(p) as f:
            for r in csv.DictReader(f):
                r["_src"] = p
                rows.append(r)
    return rows


def fnum(r, k):
    try:
        v = float(r[k])
        return v if np.isfinite(v) else np.nan
    except (KeyError, ValueError, TypeError):
        return np.nan


def summarize(rows, title):
    by = defaultdict(list)
    fails = defaultdict(int)
    for r in rows:
        if r.get("status") == "ok":
            by[r["arm"]].append(r)
        else:
            fails[r["arm"]] += 1

    if not by:
        return

    eb_roi = {r["eb_roi"] for rs in by.values() for r in rs}
    eb_bg = {r["eb_bg"] for rs in by.values() for r in rs}
    print(f"\n### {title}")
    print(f"eb_roi = {sorted(eb_roi)}   eb_bg (ROIBIN arms) = {sorted(eb_bg)}\n")

    print("| arm | frames | fail | aggregate CR | CR med [p10,p90] | ROI PSNR dB | "
          "bg PSNR dB | cmp GB/s | dec GB/s |")
    print("|---|---:|---:|---:|---|---:|---:|---:|---:|")
    for arm in ARM_ORDER:
        if arm not in by:
            continue
        rs = by[arm]
        comp = np.array([fnum(r, "compressed_bytes") for r in rs])
        # Recover each frame's original size from its own ratio rather than
        # assuming a fixed frame size, so this stays correct if a corpus with a
        # different geometry is added later.
        orig = comp * np.array([fnum(r, "cr") for r in rs])
        agg = float(np.nansum(orig) / np.nansum(comp))
        crs = np.array([fnum(r, "cr") for r in rs])
        roip = np.array([fnum(r, "roi_psnr_db") for r in rs])
        bgp = np.array([fnum(r, "bg_psnr_db") for r in rs])
        cg = np.array([fnum(r, "cmp_gbs") for r in rs])
        dg = np.array([fnum(r, "dec_gbs") for r in rs])
        print(f"| {ARM_LABEL[arm]} | {len(rs)} | {fails.get(arm,0)} | **{agg:.2f}×** | "
              f"{np.nanmedian(crs):.2f} [{np.nanpercentile(crs,10):.2f}, "
              f"{np.nanpercentile(crs,90):.2f}] | {np.nanmedian(roip):.2f} | "
              f"{np.nanmedian(bgp):.2f} | {np.nanmedian(cg):.1f} | {np.nanmedian(dg):.1f} |")

    print("\n**Per-region error-bound verification**\n")
    print("| arm | frames | ROI bound satisfied | ROI max abs err (worst frame) | "
          "bg bound satisfied | bg max abs err (worst frame) |")
    print("|---|---:|---:|---:|---:|---:|")
    for arm in ARM_ORDER:
        if arm not in by:
            continue
        rs = by[arm]
        roi_ok = sum(1 for r in rs if r.get("roi_bound_ok") == "1")
        roi_e = np.nanmax([fnum(r, "roi_max_abs_err") for r in rs])
        bg_e = np.nanmax([fnum(r, "bg_max_abs_err") for r in rs])
        bg_vals = [r.get("bg_bound_ok") for r in rs]
        if all(v == "n/a" for v in bg_vals):
            bg_txt = "n/a (binned)"
        else:
            bg_txt = f"{sum(1 for v in bg_vals if v == '1')}/{len(rs)}"
        print(f"| {ARM_LABEL[arm]} | {len(rs)} | {roi_ok}/{len(rs)} | {roi_e:.4f} | "
              f"{bg_txt} | {bg_e:.4f} |")


def paired(rows, title, ref="cuszhi"):
    """Per-frame paired comparison against the reference baseline.

    Aggregates can hide a mixed result -- an arm can win on total bytes while
    losing on many frames. Pairing by frame answers "on how many frames" and
    bounds the ROI-quality difference, which is the claim that has to hold
    frame by frame, not on average.
    """
    by = defaultdict(dict)
    for r in rows:
        if r.get("status") == "ok":
            by[r["arm"]][r["frame"]] = r
    if ref not in by:
        return
    print(f"\n**Paired per-frame comparison vs {ARM_LABEL[ref]}** — {title}\n")
    print("| arm | frames | CR ratio med [min, max] | frames with higher CR | "
          "ROI PSNR delta dB med [min, max] |")
    print("|---|---:|---|---:|---|")
    for arm in ARM_ORDER:
        if arm == ref or arm not in by:
            continue
        common = sorted(set(by[arm]) & set(by[ref]))
        if not common:
            continue
        cr = np.array([fnum(by[arm][f], "cr") / fnum(by[ref][f], "cr") for f in common])
        dp = np.array([fnum(by[arm][f], "roi_psnr_db") - fnum(by[ref][f], "roi_psnr_db")
                       for f in common])
        print(f"| {ARM_LABEL[arm]} | {len(common)} | "
              f"{np.nanmedian(cr):.2f} [{np.nanmin(cr):.2f}, {np.nanmax(cr):.2f}] | "
              f"{100*np.nanmean(cr > 1):.0f}% | "
              f"{np.nanmedian(dp):+.3f} [{np.nanmin(dp):+.3f}, {np.nanmax(dp):+.3f}] |")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csvs", nargs="+")
    ap.add_argument("--paired", action="store_true",
                    help="also emit the per-frame paired comparison vs cuSZ-Hi")
    ap.add_argument("--split-by-dataset", action="store_true")
    args = ap.parse_args()

    rows = load(args.csvs)
    if args.split_by_dataset:
        for ds in sorted({r["dataset"] for r in rows}):
            sub = [r for r in rows if r["dataset"] == ds]
            summarize(sub, ds)
            if args.paired:
                paired(sub, ds)
    else:
        summarize(rows, "all")
        if args.paired:
            paired(rows, "all")
    return 0


if __name__ == "__main__":
    sys.exit(main())
