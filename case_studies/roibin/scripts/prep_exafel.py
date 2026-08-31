#!/usr/bin/env python3
"""
Prepare the SDRBench EXAFEL light-source corpus for the ROIBIN study.

Source (provenance, see notes/data_provenance.md):
  SDRBench "EXAFEL 3-D" build, 130 x 1480 x 1552 f32, LCLS CSPAD detector images.
  Ships alongside the *published peak-finder output* for the same 130 events:
      SDRBENCH-EXAFEL-nPeaks.i64              (130 x int64)   peaks per event
      SDRBENCH-EXAFEL-peakXPosRaw-130x2048.d64 (130 x 2048 f64) peak fast-axis coord
      SDRBENCH-EXAFEL-peakYPosRaw-130x2048.d64 (130 x 2048 f64) peak slow-axis coord

Coordinate convention was verified empirically, not assumed:
  frame[e][ y[e,i], x[e,i] ] is the Bragg peak. Intensities at those pixels run
  20-30x the frame median; the transposed reading is indistinguishable from
  random pixels. See notes/data_provenance.md for the check and its output.

Outputs:
  frames/exafel_f<NNN>.f32     130 single-event images, 1552x1480 f32 (9.19 MB each)
  peaks/exafel_f<NNN>.roi      per-frame peak list in the .roi format below
  peaks/exafel_full.roi        all 13,837 peaks, indexed by z, for the 3-D volume

.roi format (little-endian, consumed by ROIBinSplitStage):
  magic   char[8]  "FZROI1\0\0"
  nx      uint32   fast axis
  ny      uint32   slow axis
  nz      uint32   frames
  npeaks  uint32
  records npeaks x { uint32 z; uint16 x; uint16 y; }   (8 bytes each)
Records are sorted by (z, y, x) so the file is deterministic.
"""
import argparse
import os
import struct
import sys

import numpy as np

SRC = "/media/volume/Compression_Data/sdrbench_data/EXAFEL_130x1480x1552"
NZ, NY, NX = 130, 1480, 1552
MAGIC = b"FZROI1\0\0"


def load_peaks():
    n = np.fromfile(f"{SRC}/SDRBENCH-EXAFEL-nPeaks.i64", dtype=np.int64)
    x = np.fromfile(f"{SRC}/SDRBENCH-EXAFEL-peakXPosRaw-130x2048.d64",
                    dtype=np.float64).reshape(NZ, 2048)
    y = np.fromfile(f"{SRC}/SDRBENCH-EXAFEL-peakYPosRaw-130x2048.d64",
                    dtype=np.float64).reshape(NZ, 2048)
    assert n.shape == (NZ,), n.shape
    return n, x, y


def write_roi(path, recs, nx, ny, nz):
    """recs: (N,3) int array of (z, x, y), already validated in range."""
    recs = recs[np.lexsort((recs[:, 1], recs[:, 2], recs[:, 0]))]  # by z, then y, then x
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
    ap.add_argument("--out", default="/media/volume/Compression_Data/sdrbench_data/derived/EXAFEL_ROIBIN")
    ap.add_argument("--frames", action="store_true", help="also write the 130 per-frame images")
    args = ap.parse_args()

    os.makedirs(f"{args.out}/peaks", exist_ok=True)
    if args.frames:
        os.makedirs(f"{args.out}/frames", exist_ok=True)

    n, xs, ys = load_peaks()
    data = np.memmap(f"{SRC}/SDRBENCH-EXAFEL-data-130x1480x1552.f32",
                     dtype=np.float32, mode="r", shape=(NZ, NY, NX))

    all_recs = []
    stats = []
    for e in range(NZ):
        k = int(n[e])
        px = xs[e, :k]
        py = ys[e, :k]
        # Peak coords are stored as f64 but are integral pixel indices; assert that
        # so a silent fractional convention can never slip through as a rounding bug.
        assert np.all(px == np.floor(px)) and np.all(py == np.floor(py)), \
            f"non-integral peak coords in event {e}"
        px = px.astype(np.int64)
        py = py.astype(np.int64)
        assert px.min() >= 0 and px.max() < NX, (e, px.min(), px.max())
        assert py.min() >= 0 and py.max() < NY, (e, py.min(), py.max())

        recs = np.stack([np.full(k, e), px, py], axis=1)
        all_recs.append(recs)

        if args.frames:
            frame = np.array(data[e])
            frame.tofile(f"{args.out}/frames/exafel_f{e:03d}.f32")
            # per-frame peak file uses z=0 (single-frame field)
            fr = recs.copy()
            fr[:, 0] = 0
            write_roi(f"{args.out}/peaks/exafel_f{e:03d}.roi", fr, NX, NY, 1)
            stats.append((e, k, float(frame.min()), float(frame.max()),
                          float(frame.mean()), float(frame[py, px].mean())))

    allr = np.concatenate(all_recs, axis=0)
    write_roi(f"{args.out}/peaks/exafel_full.roi", allr, NX, NY, NZ)
    print(f"wrote {len(allr)} peak records for the 3-D volume")

    if stats:
        with open(f"{args.out}/frame_stats.csv", "w") as f:
            f.write("frame,npeaks,min,max,mean,peak_mean\n")
            for row in stats:
                f.write("%d,%d,%.4f,%.4f,%.6f,%.4f\n" % row)
        print(f"wrote {len(stats)} frames + per-frame peak lists")


if __name__ == "__main__":
    sys.exit(main())
