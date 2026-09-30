"""Native ROIBIN-SZ (LibPressio roibin + SZ3) over a whole ROIBIN corpus.

usage: run_native.py DATASET BIN MODE [WORKERS]
  DATASET  EXAFEL | CXIDB21
  BIN      1 (background error-bounded, SZ3 directly) | 2 (2x2 binning, ROIBIN-SZ)
  MODE     serial    one frame at a time, SZ3 single-threaded
           omp       one frame at a time, sz3:openmp=true with pressio:nthreads=WORKERS
           frames    WORKERS processes, each compressing whole frames (frame-level
                     parallelism); throughput = total bytes / wall time
Per-frame quality (CR, ROI/bg max error, PSNR) is identical across modes except omp,
which changes SZ3's blocking and is reported separately.
"""
import glob, json, os, sys, time
from multiprocessing import Pool
import numpy as np

ROOTS = {"EXAFEL": "/media/volume/Compression_Data/sdrbench_data/derived/EXAFEL_ROIBIN",
         "CXIDB21": "/media/volume/Compression_Data/sdrbench_data/derived/CXIDB21_ROIBIN"}

def frame_list(ds):
    root = ROOTS[ds]
    fs = sorted(glob.glob(f"{root}/frames/*.f32"))
    return [(f, f"{root}/peaks/{os.path.basename(f)[:-4]}.roi") for f in fs]

def one(args):
    import smoke
    f, roi, binf, omp, nthreads = args
    r = smoke.run(f, roi, reps=1, bin_factor=binf, nthreads=nthreads if omp else 1, sz3_openmp=omp)
    r["frame"] = os.path.basename(f)
    return r

def main():
    ds, binf, mode = sys.argv[1], int(sys.argv[2]), sys.argv[3]
    workers = int(sys.argv[4]) if len(sys.argv) > 4 else os.cpu_count()
    frames = frame_list(ds)
    omp = mode == "omp"
    jobs = [(f, roi, binf, omp, workers) for f, roi in frames]
    t0 = time.perf_counter()
    if mode == "frames":
        with Pool(workers) as p:
            rows = p.map(one, jobs, chunksize=1)
    else:
        rows = [one(j) for j in jobs]
    wall = time.perf_counter() - t0
    nb = sum(1552 * 1480 * 4 for _ in rows)
    comp = sum(1552 * 1480 * 4 / r["cr"] for r in rows)
    peaks = sum(r["npeaks"] for r in rows)
    summ = dict(dataset=ds, bin=binf, mode=mode, workers=workers, frames=len(rows),
                agg_cr=nb / comp, agg_cr_with_peak_table=nb / (comp + 8 * peaks),
                roi_max_err=max(r["roi_max_err"] for r in rows),
                bg_max_err=max(r["bg_max_err"] for r in rows),
                median_bg_psnr=float(np.median([r["bg_psnr"] for r in rows])),
                median_frame_cmp_gbs=float(np.median([r["cmp_gbs"] for r in rows])),
                median_frame_dec_gbs=float(np.median([r["dec_gbs"] for r in rows])),
                sum_cmp_s=sum(r["cmp_s"] for r in rows), sum_dec_s=sum(r["dec_s"] for r in rows),
                wall_s=wall)
    # serial/omp: throughput from summed codec time; frames: aggregate over wall time
    # (wall includes both encode and decode plus quality checks, so it is conservative).
    summ["agg_cmp_gbs"] = nb / summ["sum_cmp_s"] / 1e9
    summ["agg_dec_gbs"] = nb / summ["sum_dec_s"] / 1e9
    if mode == "frames":
        # per-worker codec time is serialized inside each process; with W workers in
        # parallel the achieved codec throughput is ~W x the per-frame rate, bounded by
        # wall time. Report the wall-time bound (conservative) and the ideal W x rate.
        summ["frames_wall_codec_gbs"] = nb / wall / 1e9
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       f"native_{ds}_bin{binf}_{mode}{workers if mode != 'serial' else ''}.json")
    json.dump(dict(summary=summ, frames=rows), open(out, "w"), indent=1, default=float)
    print(json.dumps(summ, default=float))

if __name__ == "__main__":
    main()
