#!/usr/bin/env python3
"""
Tune the cuSZ-Hi baseline once per (dataset, error bound).

The task requires the baseline to be tuned rather than default-configured, but a
per-frame oracle over the grid would flatter it beyond what any real deployment
does — you pick one configuration for the instrument and run it. So the grid is
swept over a sample of frames, and the setting with the best aggregate ratio
*that succeeds on every sampled frame* wins. Robustness is part of the choice: a
setting that gives a great ratio on the frames it survives and crashes on the
rest is not a usable configuration, and picking it would understate the baseline's
real-world behaviour in one direction while overstating its ratio in the other.

Output is a small table plus the winning "predictor/preference/lossless" string to
pass to run_sweep.py --cuszhi-tuning.
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

CUSZHI = os.path.expanduser("~/compressors/cuSZ-Hi/build/cuszhi")
GRID = [(p, a, s) for p in ("spline", "lorenzo")
        for a in ("cr-first", "rd-first")
        for s in ("cr", "tp")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--eb", type=float, required=True)
    ap.add_argument("--sample", type=int, default=12)
    ap.add_argument("--dims", default="1552x1480")
    args = ap.parse_args()

    frames = sorted(Path(args.root, "frames").glob("*.f32"))
    if not frames:
        raise SystemExit(f"no frames under {args.root}/frames")
    step = max(1, len(frames) // args.sample)
    sample = frames[::step][:args.sample]

    work = Path("/tmp/claude-1001/-home-exouser/"
                "72d5c768-8c2e-4cbe-9f21-11b391291811/scratchpad/tunecz")
    work.mkdir(parents=True, exist_ok=True)
    link = work / "input.f32"

    print(f"tuning cuSZ-Hi on {len(sample)} frames, ABS eb={args.eb}\n")
    print("| predictor/preference/lossless | ok/n | aggregate CR |")
    print("|---|---:|---:|")

    results = []
    for pred, pref, loss in GRID:
        orig = comp = 0
        ok = 0
        for f in sample:
            if link.is_symlink() or link.exists():
                link.unlink()
            link.symlink_to(f)
            out = Path(str(link) + ".cusza")
            if out.exists():
                out.unlink()
            r = subprocess.run(
                [CUSZHI, "-z", "-i", str(link), "-t", "f4", "-l", args.dims,
                 "-m", "abs", "-e", repr(args.eb), "--predictor", pred,
                 "-a", pref, "-s", loss],
                capture_output=True, text=True)
            if r.returncode == 0 and out.exists():
                ok += 1
                orig += f.stat().st_size
                comp += out.stat().st_size
        cr = (orig / comp) if comp else float("nan")
        tag = f"{pred}/{pref}/{loss}"
        results.append((tag, ok, len(sample), cr))
        print(f"| {tag} | {ok}/{len(sample)} | {cr:.3f} |")

    complete = [r for r in results if r[1] == r[2]]
    if not complete:
        best = max(results, key=lambda r: (r[1], r[3] if r[3] == r[3] else 0))
        print(f"\nNo setting succeeded on every sampled frame. "
              f"Best by coverage then ratio: {best[0]} ({best[1]}/{best[2]}).")
    else:
        best = max(complete, key=lambda r: r[3])
        print(f"\nWinner (best aggregate CR among settings that ran on all "
              f"{len(sample)} frames): {best[0]}  CR={best[3]:.3f}")
    print(f"\n--cuszhi-tuning {best[0]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
