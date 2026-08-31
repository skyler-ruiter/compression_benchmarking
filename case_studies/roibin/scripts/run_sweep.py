#!/usr/bin/env python3
"""
ROIBIN study driver: run each arm over a frame corpus, validate per region, emit CSV.

Arms
  roibin_b1  ROIBinSplit, bin_factor=1 — dual error bound, background bound is real
  roibin_b2  ROIBinSplit, bin_factor=2 — the ROIBIN configuration, background binned
  cuszhi     native cuSZ-Hi at the ROI bound applied globally (both -s modes = tuned)
  pfpl       native PFPL (LC) at the ROI bound applied globally
  cuszp3     FZGM cuszp3 preset at the ROI bound applied globally (throughput reference)

Every arm is measured the same way and passed through the same per-region gate, so
the baselines' ROI and background PSNR are directly comparable to the ROIBIN arms'.
The baselines are run at the ROI (tight) bound because that is what a single-bound
compressor must do to protect the peaks — that is the comparison the study is about.

Standing measurement rules honoured here: PREALLOCATE (set in the configs),
`fzgmod-cli -b` for FZGM timing (never a cold single run), one bound mode (ABS)
across every arm, and repeated invocations rather than in-process repeats.
"""
import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from validate_regions import psnr, read_roi, roi_mask  # noqa: E402

CLI = os.path.expanduser("~/FZGPUModules/build_benchmarking/bin/fzgmod-cli")
CUSZHI = os.path.expanduser("~/compressors/cuSZ-Hi/build/cuszhi")
PFPL_DIR = os.path.expanduser("~/compressors/PFPL/bin/f32/gpu")
CFG_DIR = Path(__file__).parent.parent / "configs"

NX, NY = 1552, 1480


def sh(argv, **kw):
    return subprocess.run(argv, capture_output=True, text=True, **kw)


def region_metrics(orig_path, recon_path, mask, rng, eb_roi, eb_bg, bin_factor):
    o = np.fromfile(orig_path, dtype=np.float32)
    r = np.fromfile(recon_path, dtype=np.float32)
    n = mask.size
    if r.size < n:
        return None
    o = o[:n].reshape(mask.shape)
    r = r[:n].reshape(mask.shape)
    err = np.abs(o.astype(np.float64) - r.astype(np.float64))
    roi_e = float(err[mask].max()) if mask.any() else 0.0
    bg_e = float(err[~mask].max())
    return {
        "roi_max_abs_err": roi_e,
        "bg_max_abs_err": bg_e,
        "global_max_abs_err": float(err.max()),
        "roi_psnr_db": psnr(o[mask], r[mask], rng) if mask.any() else float("nan"),
        "bg_psnr_db": psnr(o[~mask], r[~mask], rng),
        "global_psnr_db": psnr(o, r, rng),
        "roi_bound_ok": int(roi_e <= eb_roi * (1 + 1e-5)),
        "bg_bound_ok": ("n/a" if bin_factor > 1 else int(bg_e <= eb_bg * (1 + 1e-5))),
    }


def run_fzgm(cfg_path, frame, peaks, work, eb_roi, eb_bg, runs):
    """Compress+decompress via a TOML config; returns (cr, cmp_gbs, dec_gbs, recon)."""
    text = Path(cfg_path).read_text()
    text = text.replace("PEAKS_FILE", str(peaks))
    text = text.replace("EB_ROI", repr(eb_roi)).replace("EB_BG", repr(eb_bg))
    cfg = work / "cfg.toml"
    cfg.write_text(text)

    fzm = work / "c.fzm"
    out = work / "d.f32"
    bj = work / "b.json"

    b = sh([CLI, "-b", "-i", str(frame), "-l", f"{NX}x{NY}", "-c", str(cfg),
            "--runs", str(runs), "--report-json", str(bj)])
    if b.returncode != 0 or not bj.exists():
        return None, b.stderr[-400:]
    j = json.loads(bj.read_text())
    if j.get("status") != "ok":
        return None, f"status={j.get('status')} {j.get('error_message')}"

    z = sh([CLI, "-z", "-i", str(frame), "-l", f"{NX}x{NY}", "-c", str(cfg), "-o", str(fzm)])
    if z.returncode != 0:
        return None, z.stderr[-400:]
    x = sh([CLI, "-x", "-i", str(fzm), "-o", str(out)])
    if x.returncode != 0:
        return None, x.stderr[-400:]

    return {
        "compressed_bytes": fzm.stat().st_size,
        "cr": frame.stat().st_size / fzm.stat().st_size,
        "cmp_gbs": j["throughput"]["compress_gbs"],
        "dec_gbs": j["throughput"]["decompress_gbs"],
        "recon": out,
    }, None


# cuSZ-Hi tuning grid. The task requires a tuned baseline, not a default-configured
# one, so every (predictor, preference, lossless) combination is tried and the best
# compression ratio that also decompresses is taken. Reported alongside the winning
# setting so the tuning is auditable.
CUSZHI_GRID = [(p, a, s) for p in ("spline", "lorenzo")
               for a in ("cr-first", "rd-first")
               for s in ("cr", "tp")]


def _cuszhi_once(frame, work, eb, pred, pref, lossless, runs):
    link = work / "input.f32"
    if link.is_symlink() or link.exists():
        link.unlink()
    link.symlink_to(frame)
    comp = Path(str(link) + ".cusza")
    recon = Path(str(link) + ".cuszx")
    for p in (comp, recon):
        if p.exists():
            p.unlink()

    z = sh([CUSZHI, "-z", "-i", str(link), "-t", "f4", "-l", f"{NX}x{NY}",
            "-m", "abs", "-e", repr(eb), "--predictor", pred, "-a", pref,
            "-s", lossless, "-R", "time", "--repeat", str(runs)])
    if z.returncode != 0 or not comp.exists():
        return None, (z.stderr or z.stdout)[-300:]
    x = sh([CUSZHI, "-x", "-i", str(comp), "-R", "time", "--repeat", str(runs)])
    if x.returncode != 0 or not recon.exists():
        return None, (x.stderr or x.stdout)[-300:]

    def gbs(txt):
        # -R time prints one row per rep:  "(total)   <time_ms>   <GiB/s>".
        # Take column 1 only (ms); the trailing GiB/s column must not be mixed in.
        # Drop the first rep: it carries one-time setup (2.8 ms vs 0.34 ms warm),
        # the same cold-run effect the standing rules warn about for fzgmod-cli.
        vals = []
        for line in txt.splitlines():
            if "(total)" not in line:
                continue
            toks = line.split()
            try:
                vals.append(float(toks[toks.index("(total)") + 1]))
            except (ValueError, IndexError):
                pass
        if len(vals) > 1:
            vals = vals[1:]
        if not vals:
            return float("nan")
        return frame.stat().st_size / (float(np.median(vals)) * 1e6)

    return {
        "compressed_bytes": comp.stat().st_size,
        "cr": frame.stat().st_size / comp.stat().st_size,
        "cmp_gbs": gbs(z.stdout),
        "dec_gbs": gbs(x.stdout),
        "recon": recon,
        "tuning": f"{pred}/{pref}/{lossless}",
    }, None


def run_cuszhi(frame, work, eb, runs, pinned=None):
    """Sweep the tuning grid, or use a pinned setting from an earlier tuning pass.

    Sweeping all 8 grid points per frame costs ~16 cuszhi process launches per
    frame, which does not scale to a 409-frame corpus. tune_cuszhi.py runs the
    grid once per (dataset, error bound) and the winner is pinned here; that is
    also the conventional meaning of "tuned baseline" -- one configuration chosen
    for the data, not a per-frame oracle, which would flatter the baseline.
    """
    grid = CUSZHI_GRID if pinned is None else [tuple(pinned.split("/"))]
    best, errs = None, []
    for pred, pref, lossless in grid:
        res, err = _cuszhi_once(frame, work, eb, pred, pref, lossless, runs)
        if res is None:
            errs.append(f"{pred}/{pref}/{lossless}: {err}")
            continue
        if best is None or res["cr"] > best["cr"]:
            # Keep a private copy: the next grid point overwrites .cuszx.
            keep = work / "cuszhi_best.f32"
            shutil.copyfile(res["recon"], keep)
            res["recon"] = keep
            best = res
    if best is None:
        return None, ("all cuszhi tunings failed: " + " | ".join(errs))[:300]
    return best, None


def run_pfpl(frame, work, eb):
    comp = work / "p.lc"
    recon = work / "p.f32"
    z = sh([f"{PFPL_DIR}/f32_abs_compress_cuda", str(frame), str(comp), repr(eb)])
    if z.returncode != 0 or not comp.exists():
        return None, z.stderr[-400:] or z.stdout[-400:]
    x = sh([f"{PFPL_DIR}/f32_abs_decompress_cuda", str(comp), str(recon)])
    if x.returncode != 0 or not recon.exists():
        return None, x.stderr[-400:] or x.stdout[-400:]

    def gbs(txt, key):
        # PFPL prints one "lc <phase> ecltime,  <seconds>" line per internal rep
        # (NUM_RUNS=9). Drop the first for the same cold-run reason as above.
        vals = [float(l.split(",")[1]) for l in txt.splitlines()
                if key in l and "," in l]
        if len(vals) > 1:
            vals = vals[1:]
        if not vals:
            return float("nan")
        return frame.stat().st_size / (float(np.median(vals)) * 1e9)

    return {
        "compressed_bytes": comp.stat().st_size,
        "cr": frame.stat().st_size / comp.stat().st_size,
        "cmp_gbs": gbs(z.stdout, "comp ecltime"),
        "dec_gbs": gbs(x.stdout, "decomp ecltime"),
        "recon": recon,
    }, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["EXAFEL", "CXIDB21"])
    ap.add_argument("--root", required=True)
    ap.add_argument("--limit", type=int, default=0, help="0 = all frames")
    ap.add_argument("--eb-roi", type=float, default=1.0)
    ap.add_argument("--eb-bg", type=float, default=10.0)
    ap.add_argument("--half-width", type=int, default=4)
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--arms", default="roibin_b1,roibin_b2,cuszhi,pfpl,cuszp3")
    ap.add_argument("--cuszhi-tuning", default=None,
                    help="pinned 'predictor/preference/lossless'; omit to sweep the "
                         "full grid per frame (slow)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    root = Path(args.root)
    frames = sorted((root / "frames").glob("*.f32"))
    if args.limit:
        frames = frames[:args.limit]
    arms = args.arms.split(",")

    rows = []
    work = Path(tempfile.mkdtemp(prefix="roibin_"))
    try:
        for i, frame in enumerate(frames):
            peaks = root / "peaks" / (frame.stem + ".roi")
            if not peaks.exists():
                continue
            nx, ny, nz, rec = read_roi(peaks)
            mask = roi_mask(nx, ny, nz, rec, args.half_width)
            o = np.fromfile(frame, dtype=np.float32)
            rng = float(o.max() - o.min())

            for arm in arms:
                if arm == "roibin_b1":
                    res, err = run_fzgm(CFG_DIR / "roibin_sweep_b1.toml", frame, peaks,
                                        work, args.eb_roi, args.eb_bg, args.runs)
                    bin_f = 1
                elif arm == "roibin_b2":
                    res, err = run_fzgm(CFG_DIR / "roibin_sweep_b2.toml", frame, peaks,
                                        work, args.eb_roi, args.eb_bg, args.runs)
                    bin_f = 2
                elif arm == "cuszp3":
                    res, err = run_fzgm(CFG_DIR / "baseline_cuszp3.toml", frame, peaks,
                                        work, args.eb_roi, args.eb_roi, args.runs)
                    bin_f = 1
                elif arm == "cuszhi":
                    res, err = run_cuszhi(frame, work, args.eb_roi, args.runs,
                                          args.cuszhi_tuning)
                    bin_f = 1
                elif arm == "pfpl":
                    res, err = run_pfpl(frame, work, args.eb_roi)
                    bin_f = 1
                else:
                    continue

                row = {"dataset": args.dataset, "frame": frame.stem, "arm": arm,
                       "npeaks": len(rec), "roi_pixels": int(mask.sum()),
                       "roi_frac_pct": 100.0 * float(mask.mean()),
                       "eb_roi": args.eb_roi, "eb_bg": args.eb_bg,
                       "half_width": args.half_width, "bin_factor": bin_f,
                       "value_range": rng}
                if res is None:
                    row["status"] = "fail"
                    row["error"] = (err or "")[:300].replace("\n", " ")
                else:
                    # The baselines apply the tight bound globally, so their
                    # "background bound" is the ROI bound; record it that way.
                    eb_bg_eff = args.eb_bg if arm.startswith("roibin") else args.eb_roi
                    m = region_metrics(frame, res["recon"], mask, rng,
                                       args.eb_roi, eb_bg_eff, bin_f)
                    if m is None:
                        row["status"] = "fail"
                        row["error"] = "recon shorter than field"
                    else:
                        row["status"] = "ok"
                        row.update({k: v for k, v in res.items() if k != "recon"})
                        row.update(m)
                rows.append(row)

            if (i + 1) % 10 == 0:
                print(f"[{args.dataset}] {i+1}/{len(frames)} frames", flush=True)
    finally:
        shutil.rmtree(work, ignore_errors=True)

    keys = sorted({k for r in rows for k in r})
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    ok = sum(1 for r in rows if r.get("status") == "ok")
    print(f"wrote {args.out}: {len(rows)} rows, {ok} ok")


if __name__ == "__main__":
    sys.exit(main())
