#!/usr/bin/env python3
"""
Does the baseline's failure set bias the comparison?

Native cuSZ-Hi aborts on 84/130 EXAFEL frames at ABS eb=1.0 (clean at eb>=2.0).
Any comparison at a bound where a baseline fails is implicitly restricted to the
frames it survives, and those frames are not a random sample. The reflex reading
is that the survivor subset flatters the baseline. That is not always true, and
guessing the sign is not good enough — it has to be measured per study.

Method: take a bound where *every* arm ran on *every* frame (eb=10), split the
frames by whether the baseline survives at the tighter bound, and compare both
the per-arm ratios and — the number that actually matters — the paired CR ratio
against the baseline. If the paired ratio is unchanged, the failure set is not
biasing the claim.

Result on EXAFEL (2026-08-09): survivors are dimmer and more compressible, but
ROIBIN gains *more* on them, so restricting to survivors would flatter ROIBIN by
+1.8% (bin=1) / +10.1% (bin=2). Reporting the full corpus is the conservative
choice. Note this is the opposite sign from the HURR/SCALE-LETKF case, where the
fields cuSZ-Hi cannot process are the most compressible ones and the survivor
subset flatters the baseline — hence "measure it, don't reason about it".
"""
import argparse
import csv
import sys

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True,
                    help="sweep CSV at a bound where every arm ran on every frame")
    ap.add_argument("--failures", required=True,
                    help="CSV with columns eb,frame,rc from the failure-envelope probe")
    ap.add_argument("--fail-eb", default="1.0", help="the bound at which the baseline fails")
    ap.add_argument("--ref", default="cuszhi")
    ap.add_argument("--stats", default=None, help="optional frame_stats.csv for context")
    args = ap.parse_args()

    surv = {}
    for r in csv.DictReader(open(args.failures)):
        if r.get("frame") and r.get("eb") == args.fail_eb:
            surv[r["frame"]] = (r["rc"] == "0")
    if not surv:
        raise SystemExit(f"no failure rows at eb={args.fail_eb} in {args.failures}")

    rows = [r for r in csv.DictReader(open(args.results)) if r["status"] == "ok"]
    arms = sorted({r["arm"] for r in rows})
    cr = {a: {r["frame"]: float(r["cr"]) for r in rows if r["arm"] == a} for a in arms}
    if args.ref not in cr:
        raise SystemExit(f"reference arm {args.ref} not in results")

    n_ok = sum(1 for v in surv.values() if v)
    print(f"baseline {args.ref} at eb={args.fail_eb}: "
          f"{n_ok}/{len(surv)} frames survive\n")

    if args.stats:
        st = {}
        for r in csv.DictReader(open(args.stats)):
            key = r.get("frame")
            st["exafel_f%03d" % int(key)] = r if key.isdigit() else None
            if not key.isdigit():
                st[key] = r
        for lab, want in (("survive", True), ("abort", False)):
            g = [f for f, v in surv.items() if v is want and f in st and st[f]]
            if g:
                m = np.array([float(st[f]["mean"]) for f in g])
                print(f"  {lab:8s} n={len(g):3d}  frame mean median {np.median(m):8.1f}")
        print()

    print("| arm | CR med survivors | CR med aborters | ratio |")
    print("|---|---:|---:|---:|")
    for a in arms:
        s = np.array([v for k, v in cr[a].items() if surv.get(k) is True])
        b = np.array([v for k, v in cr[a].items() if surv.get(k) is False])
        if len(s) and len(b):
            print(f"| {a} | {np.median(s):.2f} | {np.median(b):.2f} | "
                  f"{np.median(s)/np.median(b):.2f} |")

    print("\n| arm | paired CR ratio vs "
          f"{args.ref}, all frames | restricted to survivors | shift |")
    print("|---|---:|---:|---:|")
    ref = cr[args.ref]
    for a in arms:
        if a == args.ref:
            continue
        common = [k for k in cr[a] if k in ref]
        if not common:
            continue
        allr = np.array([cr[a][k] / ref[k] for k in common])
        sr = np.array([cr[a][k] / ref[k] for k in common if surv.get(k) is True])
        if not len(sr):
            continue
        print(f"| {a} | {np.median(allr):.3f} | {np.median(sr):.3f} | "
              f"{100*(np.median(sr)/np.median(allr)-1):+.1f}% |")

    print("\nA shift near zero means the failure set is not biasing the claim. "
          "A positive shift means restricting to survivors would flatter the "
          "challenger, so reporting full coverage is the conservative choice.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
