"""Smoke test: native ROIBIN-SZ (LibPressio roibin + binning + SZ3) on one frame.

Mirrors FZGM's roibin_b2 arm: ROI abs eb 10 (9x9 box per peak), background
binned 2x2 then SZ3 abs eb 100. Reports CR, per-region max error, and wall time.
"""
import struct, sys, time
import numpy as np
import libpressio as lp

def read_roi(path):
    with open(path, "rb") as f:
        magic = f.read(8); assert magic.startswith(b"FZROI1"), magic
        nx, ny, nz, n = struct.unpack("<4I", f.read(16))
        rec = np.frombuffer(f.read(8 * n), dtype=[("z", "<u4"), ("x", "<u2"), ("y", "<u2")])
    return nx, ny, nz, rec

def roi_mask(shape_yx, rec, hw):
    m = np.zeros(shape_yx, bool)
    for r in rec:
        y, x = int(r["y"]), int(r["x"])
        m[max(0, y - hw):y + hw + 1, max(0, x - hw):x + hw + 1] = True
    return m

def run(frame_path, roi_path, eb_roi=10.0, eb_bg=100.0, hw=4, nthreads=20, reps=3, bin_factor=2, sz3_openmp=False):
    nx, ny, nz, rec = read_roi(roi_path)
    # Frames are passed as 3-D (x, y, 1). LibPressio's 1-D and 2-D binning inverses
    # (roibin_impl.h restore_omp<1>/<2>) loop over bins[2]/bins[3] out of bounds and
    # hang; the 3-D path is correct and is what upstream's test_roibin exercises.
    data = np.fromfile(frame_path, dtype=np.float32).reshape(1, ny, nx)   # C order -> dims (nx, ny, 1)
    centers = np.stack([rec["x"].astype(np.uint64), rec["y"].astype(np.uint64),
                        np.zeros(len(rec), np.uint64)])        # (3, npeaks), fastest axis first
    centers = np.ascontiguousarray(centers.T)   # libpressio reads centers[i*width + d]
    if bin_factor == 1:   # background error-bounded: SZ3 directly, no binning
        comp = lp.PressioCompressor("roibin", early_config={
            "roibin:roi": "sz3", "roibin:background": "sz3"}, name="rb")
        comp.set_options({
            "roibin:centers": centers, "roibin:roi_size": np.array([hw, hw, 0], dtype=np.uint64),
            "roibin:nthreads": nthreads,
            "/rb/roi:pressio:abs": eb_roi,
            "/rb/background:pressio:abs": eb_bg,
            "/rb/background:sz3:openmp": sz3_openmp,
            "/rb/background:pressio:nthreads": nthreads,
        })
    else:
        comp = lp.PressioCompressor("roibin", early_config={
            "roibin:roi": "sz3", "roibin:background": "binning", "binning:compressor": "sz3"}, name="rb")
        comp.set_options({
            "roibin:centers": centers, "roibin:roi_size": np.array([hw, hw, 0], dtype=np.uint64),
            "roibin:nthreads": nthreads, "binning:nthreads": nthreads,
            "binning:shape": np.array([bin_factor, bin_factor, 1], dtype=np.uint64),
            "/rb/roi:pressio:abs": eb_roi,
            "/rb/background/sz3:pressio:abs": eb_bg,
            "/rb/background/sz3:sz3:openmp": sz3_openmp,
            "/rb/background/sz3:pressio:nthreads": nthreads,
        })
    ct, dt = [], []
    for _ in range(reps):
        t = time.perf_counter(); cbuf = comp.encode(data); ct.append(time.perf_counter() - t)
        out = np.empty_like(data)
        t = time.perf_counter(); out = comp.decode(cbuf, out); dt.append(time.perf_counter() - t)
    err = np.abs(out.astype(np.float64) - data)
    m = roi_mask(data.shape[1:], rec, hw)[None]
    rng = float(data.max() - data.min())
    def psnr(e): return 20 * np.log10(rng) - 10 * np.log10(np.mean(e ** 2))
    return dict(npeaks=len(rec), cr=data.nbytes / len(cbuf),
                roi_max_err=float(err[m].max()), roi_psnr=psnr(err[m]),
                bg_max_err=float(err[~m].max()), bg_psnr=psnr(err[~m]),
                cmp_s=min(ct), dec_s=min(dt),
                cmp_gbs=data.nbytes / min(ct) / 1e9, dec_gbs=data.nbytes / min(dt) / 1e9)

if __name__ == "__main__":
    root = "/media/volume/Compression_Data/sdrbench_data/derived/EXAFEL_ROIBIN"
    for f in sys.argv[1:] or ["000"]:
        r = run(f"{root}/frames/exafel_f{f}.f32", f"{root}/peaks/exafel_f{f}.roi")
        print(f, {k: round(v, 4) if isinstance(v, float) else v for k, v in r.items()})
