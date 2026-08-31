#!/usr/bin/env python3
"""
Per-region validity gate for the ROIBIN study.

A global max-abs-error is useless here: it reports the looser of the two bounds
and says nothing about whether the Bragg peaks — the only part the science needs
— were actually protected. This tool rebuilds the ROI mask exactly as
ROIBinSplitStage does (same clamped boxes, same peak list) and reports error and
distortion separately for the ROI pixels and the background pixels.

Gate semantics (deliberately asymmetric, see notes):
  * ROI region: must satisfy the tight bound. This is a hard gate in every
    configuration; if it fails the run is not usable at any ratio.
  * Background, bin_factor == 1: must satisfy the loose bound pixel-wise.
  * Background, bin_factor > 1: NO bound is claimed. Binning is a resolution
    reduction, so the background error is binning error plus quantization error.
    Background fidelity is reported as PSNR/max-error only and `bg_bound_ok` is
    recorded as "n/a" rather than pass or fail.

PSNR uses the range of the *original full frame* as the peak reference for both
regions, so ROI and background PSNR are on one comparable scale. Using each
region's own range would make the background look better simply because it spans
less, which would flatter exactly the region we are degrading on purpose.
"""
import argparse
import json
import struct
import sys

import numpy as np

MAGIC = b"FZROI1\0\0"


def read_roi(path):
    with open(path, "rb") as f:
        if f.read(8) != MAGIC:
            raise ValueError(f"bad magic in {path}")
        nx, ny, nz, npk = struct.unpack("<IIII", f.read(16))
        rec = np.frombuffer(f.read(npk * 8),
                            dtype=np.dtype([("z", "<u4"), ("x", "<u2"), ("y", "<u2")]))
    return nx, ny, nz, rec


def roi_mask(nx, ny, nz, rec, hw):
    """Exactly mirrors the stage: fixed (2hw+1)^2 boxes, clamped at edges."""
    m = np.zeros((nz, ny, nx), dtype=bool)
    side = 2 * hw + 1
    dy, dx = np.mgrid[-hw:hw + 1, -hw:hw + 1]
    dy = dy.ravel()
    dx = dx.ravel()
    for z, x, y in zip(rec["z"], rec["x"], rec["y"]):
        yy = np.clip(int(y) + dy, 0, ny - 1)
        xx = np.clip(int(x) + dx, 0, nx - 1)
        m[int(z), yy, xx] = True
    return m


def psnr(orig, rec, peak_range):
    err = orig.astype(np.float64) - rec.astype(np.float64)
    mse = float(np.mean(err * err))
    if mse == 0.0:
        return float("inf")
    return 20.0 * np.log10(peak_range) - 10.0 * np.log10(mse)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--orig", required=True)
    ap.add_argument("--recon", required=True)
    ap.add_argument("--peaks", required=True)
    ap.add_argument("--half-width", type=int, default=4)
    ap.add_argument("--eb-roi", type=float, required=True)
    ap.add_argument("--eb-bg", type=float, required=True)
    ap.add_argument("--bin-factor", type=int, default=1)
    ap.add_argument("--dtype", default="f32")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    dt = np.float32 if args.dtype == "f32" else np.float64
    nx, ny, nz, rec = read_roi(args.peaks)
    o = np.fromfile(args.orig, dtype=dt)
    r = np.fromfile(args.recon, dtype=dt)
    n = nx * ny * nz
    if o.size < n or r.size < n:
        raise SystemExit(f"size mismatch: need {n}, orig {o.size}, recon {r.size}")
    # The pipeline pads to the coder's block size and the CLI truncates on write;
    # trim defensively so a padded tail can never enter the metrics.
    o = o[:n].reshape(nz, ny, nx)
    r = r[:n].reshape(nz, ny, nx)

    m = roi_mask(nx, ny, nz, rec, args.half_width)
    rng = float(o.max() - o.min())

    err = np.abs(o.astype(np.float64) - r.astype(np.float64))
    roi_err = float(err[m].max()) if m.any() else 0.0
    bg_err = float(err[~m].max())

    out = {
        "frame": args.orig.split("/")[-1],
        "npeaks": int(len(rec)),
        "roi_pixels": int(m.sum()),
        "roi_frac_pct": 100.0 * float(m.mean()),
        "bin_factor": args.bin_factor,
        "eb_roi": args.eb_roi,
        "eb_bg": args.eb_bg,
        "roi_max_abs_err": roi_err,
        "bg_max_abs_err": bg_err,
        "global_max_abs_err": float(err.max()),
        "roi_psnr_db": psnr(o[m], r[m], rng) if m.any() else float("nan"),
        "bg_psnr_db": psnr(o[~m], r[~m], rng),
        "global_psnr_db": psnr(o, r, rng),
        "value_range": rng,
        # Tolerance absorbs f32 round-off in the dequantized reconstruction; it is
        # relative to the bound, not absolute, so it does not scale into a free pass
        # at loose bounds.
        "roi_bound_ok": bool(roi_err <= args.eb_roi * (1 + 1e-5)),
        "bg_bound_ok": ("n/a" if args.bin_factor > 1
                        else bool(bg_err <= args.eb_bg * (1 + 1e-5))),
    }

    if args.json:
        with open(args.json, "w") as f:
            json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))
    # Only the ROI gate is fatal; the background gate is advisory when binning.
    return 0 if out["roi_bound_ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
