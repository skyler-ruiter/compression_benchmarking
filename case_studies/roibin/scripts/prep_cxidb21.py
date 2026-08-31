#!/usr/bin/env python3
"""
Prepare CXIDB ID 21 (5HT2B GPCR serial femtosecond crystallography) as the second,
independent light-source dataset for the ROIBIN study.

Provenance (see notes/data_provenance.md):
  CXIDB ID 21, DOI 10.11577/1169541, CC0.
  Liu et al., "Serial femtosecond crystallography of G protein-coupled receptors",
  Science 342:1521 (2013). LCLS, CXI instrument, CSPAD detector.
  Files are Cheetah per-event output: data/rawdata0 is the raw CSPAD frame
  (1480 x 1552 int16 ADU), processing/cheetah/peakinfo-raw is Cheetah's own
  peak-finder output for that event.

peakinfo-raw columns are (x, y, integrated_intensity, npix); the frame is indexed
frame[y, x]. Verified empirically over all 5,655 peaks: that reading gives median
intensity 1129 vs 149 for random pixels (55% above 1000 ADU vs 7%). The transposed
reading gives 148 -- indistinguishable from random. peakinfo-assembled is in
assembled-detector coordinates and does NOT index the raw frame (median 0).

Preprocessing applied (and why):
  * rawdata0 is int16; it is widened to float32 without rescaling. The compressors
    under test are float codecs, and the SDRBench EXAFEL build is likewise f32.
    No calibration/gain/dark correction is applied -- these are raw ADU, unlike
    the SDRBench build which is calibrated. The two datasets are therefore
    reported separately, never pooled.
  * Frames are written fast-axis-first (x contiguous), matching FZGM's
    -l 1552x1480 dimension order.
"""
import argparse
import glob
import os
import struct
import sys

import h5py
import numpy as np

NY, NX = 1480, 1552
MAGIC = b"FZROI1\0\0"


def write_roi(path, recs, nx, ny, nz):
    recs = recs[np.lexsort((recs[:, 1], recs[:, 2], recs[:, 0]))]
    with open(path, "wb") as f:
        f.write(MAGIC)
        f.write(struct.pack("<IIII", nx, ny, nz, len(recs)))
        buf = np.zeros(len(recs), dtype=np.dtype([("z", "<u4"), ("x", "<u2"), ("y", "<u2")]))
        buf["z"] = recs[:, 0]
        buf["x"] = recs[:, 1]
        buf["y"] = recs[:, 2]
        f.write(buf.tobytes())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="dir holding cxidb-21-run*/data1/*.h5")
    ap.add_argument("--out", default="/media/volume/Compression_Data/sdrbench_data/derived/CXIDB21_ROIBIN")
    args = ap.parse_args()

    os.makedirs(f"{args.out}/frames", exist_ok=True)
    os.makedirs(f"{args.out}/peaks", exist_ok=True)

    files = sorted(glob.glob(f"{args.src}/*/data1/*.h5"))
    kept, skipped = 0, 0
    rows = []
    for path in files:
        try:
            with h5py.File(path, "r") as h:
                img = h["data/rawdata0"][:]
                pk = h["processing/cheetah/peakinfo-raw"][:]
        except Exception:
            skipped += 1          # truncated tail of the partial fetch
            continue
        if img.shape != (NY, NX) or len(pk) == 0:
            skipped += 1
            continue

        run = os.path.basename(os.path.dirname(os.path.dirname(path)))
        evt = os.path.splitext(os.path.basename(path))[0]
        name = f"{run.replace('cxidb-21-', '')}_{evt.split('_')[-2]}_{evt.split('_')[-1]}"

        img32 = img.astype(np.float32)
        img32.tofile(f"{args.out}/frames/{name}.f32")

        x = np.clip(pk[:, 0].astype(np.int64), 0, NX - 1)
        y = np.clip(pk[:, 1].astype(np.int64), 0, NY - 1)
        recs = np.stack([np.zeros(len(x), dtype=np.int64), x, y], axis=1)
        write_roi(f"{args.out}/peaks/{name}.roi", recs, NX, NY, 1)

        rows.append((name, len(x), float(img32.min()), float(img32.max()),
                     float(img32.mean()), float(img32[y, x].mean())))
        kept += 1

    with open(f"{args.out}/frame_stats.csv", "w") as f:
        f.write("frame,npeaks,min,max,mean,peak_mean\n")
        for r in rows:
            f.write("%s,%d,%.4f,%.4f,%.6f,%.4f\n" % r)

    print(f"kept {kept} frames, skipped {skipped}; "
          f"{sum(r[1] for r in rows)} peaks total -> {args.out}")


if __name__ == "__main__":
    sys.exit(main())
