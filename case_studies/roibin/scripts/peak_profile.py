#!/usr/bin/env python3
"""
Justify the ROI box size from the data instead of assuming it.

`roi_half_width` decides how much of each Bragg peak gets the tight bound;
everything outside the box falls to the background branch. Too small and real
peak signal is quietly demoted to the loose bound, which is precisely the failure
the study must not have. Too large and the ROI fraction (and the ratio) suffers
for nothing.

This measures the mean intensity on each square (Chebyshev) ring around the
published peak positions and prints it against the frame's median background. The
right half-width is where the profile goes flat.

Measured on EXAFEL (every 10th frame, first 80 peaks each), frame median 211.3 ADU:

    d= 0  (1x1)    3731.2   excess 3519.9
    d= 1  (3x3)    1242.0   excess 1030.7
    d= 2  (5x5)     507.4   excess  296.0
    d= 3  (7x7)     406.8   excess  195.5
    d= 4  (9x9)     394.2   excess  182.8
    d= 5  (11x11)   393.6   excess  182.2
    d>=6            ~389     excess ~178-188

The peak decays into a plateau by d=3-4; beyond that the profile is flat, so the
residual ~180 ADU excess is locally elevated background around peaks (they cluster
on bright rings), not peak signal. **half_width = 4 contains the peak** with a
ring of margin, which is the value the study uses.
"""
import argparse
import glob
import struct
import sys

import numpy as np


def read_roi(path):
    with open(path, "rb") as f:
        assert f.read(8) == b"FZROI1\0\0", path
        nx, ny, nz, n = struct.unpack("<IIII", f.read(16))
        rec = np.frombuffer(f.read(n * 8),
                            dtype=np.dtype([("z", "<u4"), ("x", "<u2"), ("y", "<u2")]))
    return nx, ny, rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--max-d", type=int, default=12)
    ap.add_argument("--frame-step", type=int, default=10)
    ap.add_argument("--peaks-per-frame", type=int, default=80)
    args = ap.parse_args()

    D = args.max_d + 1
    prof = np.zeros(D)
    cnt = np.zeros(D)
    meds = []

    for fn in sorted(glob.glob(f"{args.root}/frames/*.f32"))[::args.frame_step]:
        roi = fn.replace("/frames/", "/peaks/").replace(".f32", ".roi")
        nx, ny, rec = read_roi(roi)
        img = np.fromfile(fn, dtype=np.float32).reshape(ny, nx)
        meds.append(float(np.median(img)))
        for r in rec[:args.peaks_per_frame]:
            x, y = int(r["x"]), int(r["y"])
            if x < D or y < D or x >= nx - D or y >= ny - D:
                continue  # skip peaks whose rings would leave the frame
            for d in range(D):
                if d == 0:
                    v = np.array([img[y, x]])
                else:
                    sl = img[y - d:y + d + 1, x - d:x + d + 1]
                    v = np.concatenate([sl[0, :], sl[-1, :], sl[1:-1, 0], sl[1:-1, -1]])
                prof[d] += float(np.mean(v))
                cnt[d] += 1

    p = prof / np.maximum(cnt, 1)
    bg = float(np.mean(meds))
    print(f"frame median background ~ {bg:.1f} ADU  ({int(cnt[0])} peaks sampled)")
    print("radius d : box      : mean ring : excess over median")
    for d in range(D):
        print(f"  d={d:2d}   : {2*d+1:2d}x{2*d+1:<2d}  : {p[d]:9.1f} : {p[d]-bg:9.1f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
