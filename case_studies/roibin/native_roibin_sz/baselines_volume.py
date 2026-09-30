"""General-purpose GPU compressors on the ROIBIN volumes, one global (tight) bound.

Each compressor must apply the ROI bound (abs 10) to the whole volume to protect the
peaks. Volumes are presented as one tall 2-D image (1552 x 1480*N) so 2-D predictors
stay within frames (1480 % 8 == 0: cuSZp3's 8x8 tiles never straddle frames).
usage: baselines_volume.py DATASET   -> baselines_volume_<DATASET>.json

CXIDB 21 (2.56 GB) exceeds 2^31-byte limits in PFPL ("LC error"), cuSZ-Hi ("view
exceeds the legal length") and cuSZp3 (runs but violates the bound: max err 1280). It
is therefore compressed as two batches of 140 + 139 frames (~1.28 GB each) and
aggregated: CR = total bytes / total compressed bytes, GB/s = total bytes / total time.
"""
import json, os, re, subprocess, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
DATA = "/media/volume/Compression_Data/sdrbench_data"
VOLS = {"EXAFEL": (f"{DATA}/EXAFEL_130x1480x1552/SDRBENCH-EXAFEL-data-130x1480x1552.f32", 130),
        "CXIDB21": (f"{DATA}/derived/CXIDB21_ROIBIN/cxidb21_volume_279x1480x1552.f32", 279)}
CUSZHI = os.path.expanduser("~/compressors/cuSZ-Hi/build/cuszhi")
PFPL = os.path.expanduser("~/compressors/PFPL/bin/f32/gpu")
CUSZP3 = os.path.expanduser("~/compressors/cuSZp-V3.0.0/build/examples/bin/cuSZp")
CUSZ = os.path.expanduser("~/compressors/cuSZ/build/cusz")
EB = 10.0
RUNS = 7

def sh(a): return subprocess.run(a, capture_output=True, text=True)
def gpu_idle(): return sh(["nvidia-smi","--query-compute-apps=pid","--format=csv,noheader"]).stdout.strip() == ""

def err_stats(orig, recon_path):
    o = np.fromfile(recon_path, np.float32)
    pad = o.size - orig.size          # some tools (cuSZ) write a padded reconstruction
    assert pad >= 0, (recon_path, "reconstruction shorter than input")
    e = np.abs(o[:orig.size].astype(np.float64) - orig)
    rng = float(orig.max() - orig.min())
    return dict(max_err=float(e.max()), psnr=float(20*np.log10(rng) - 10*np.log10((e**2).mean())))

def cuszhi(vol, nz, work, orig):
    link = work / "in.f32"
    if link.exists() or link.is_symlink(): link.unlink()
    link.symlink_to(vol)
    comp, rec = Path(str(link)+".cusza"), Path(str(link)+".cuszx")
    z = sh([CUSZHI,"-z","-i",str(link),"-t","f4","-l",f"1552x{1480*nz}","-m","abs","-e",repr(EB),
            "--predictor","spline","-a","cr-first","-s","cr","-R","time","--repeat",str(RUNS)])
    x = sh([CUSZHI,"-x","-i",str(comp),"-R","time","--repeat",str(RUNS)])
    def ms(t):
        v=[float(l.split()[l.split().index("(total)")+1]) for l in t.splitlines() if "(total)" in l]
        return float(np.median(v[1:] if len(v)>1 else v))
    nb = Path(vol).stat().st_size
    r = dict(tool="cuSZ-Hi (spline/cr-first/cr)", cr=nb/comp.stat().st_size,
             cmp_gbs=nb/(ms(z.stdout)*1e6), dec_gbs=nb/(ms(x.stdout)*1e6))
    r.update(err_stats(orig, rec)); comp.unlink(); rec.unlink(); return r

def pfpl(vol, nz, work, orig):
    comp, rec = work/"p.lc", work/"p.f32"
    z = sh([f"{PFPL}/f32_abs_compress_cuda", vol, str(comp), repr(EB)])
    x = sh([f"{PFPL}/f32_abs_decompress_cuda", str(comp), str(rec)])
    def s(t,k):
        v=[float(l.split(",")[1]) for l in t.splitlines() if k in l and "," in l]
        return float(np.median(v[1:] if len(v)>1 else v))
    nb = Path(vol).stat().st_size
    r = dict(tool="PFPL", cr=nb/comp.stat().st_size,
             cmp_gbs=nb/(s(z.stdout,"comp ecltime")*1e9), dec_gbs=nb/(s(x.stdout,"decomp ecltime")*1e9))
    r.update(err_stats(orig, rec)); comp.unlink(); rec.unlink(); return r

def cuszp3(vol, nz, work, orig):
    # cuSZp prints "end-to-end speed: X GB/s" where X is MiB/ms (see benchkit adapter).
    out = sh([CUSZP3,"-i",vol,"-t","f32","-m","plain","-eb","abs",repr(EB),"-d","2","1",str(1480*nz),"1552",
              "-x",str(work/"c.cuszp"),"-o",str(work/"c.f32")])
    sp = {k: float(re.search(rf"{k}\s+end-to-end speed:\s*([0-9.eE+-]+)", out.stdout).group(1))
          for k in ("compression","decompression")}
    nb = Path(vol).stat().st_size
    to_gbs = lambda mib_per_ms: nb / ((nb/1048576.0)/mib_per_ms * 1e6)
    r = dict(tool="cuSZp3 plain 2-D", cr=nb/(work/"c.cuszp").stat().st_size,
             cmp_gbs=to_gbs(sp["compression"]), dec_gbs=to_gbs(sp["decompression"]))
    r.update(err_stats(orig, work/"c.f32")); (work/"c.cuszp").unlink(); (work/"c.f32").unlink(); return r

BATCH_FRAMES = 140   # keeps every batch under 2^31 bytes

def batches(vol, nz, work):
    if nz * 1480 * 1552 * 4 < 2**31:
        return [(vol, nz)]
    full = np.memmap(vol, np.float32, mode="r").reshape(nz, 1480, 1552)
    out = []
    for i, z0 in enumerate(range(0, nz, BATCH_FRAMES)):
        z1 = min(nz, z0 + BATCH_FRAMES); p = work / f"batch{i}.f32"
        if not p.exists() or p.stat().st_size != (z1 - z0) * 1480 * 1552 * 4:
            np.asarray(full[z0:z1]).tofile(p)
        out.append((str(p), z1 - z0))
    return out

def cusz(vol, nz, work, orig):
    # cusz writes <input>.cusza and reconstructs to <input>.cuszx; -R time prints one
    # JSON line per --repeat rep with compress_device_ms / decompress_device_ms.
    link = work / "cz.f32"
    if link.exists() or link.is_symlink(): link.unlink()
    link.symlink_to(vol)
    comp, rec = Path(str(link) + ".cusza"), Path(str(link) + ".cuszx")
    base = [CUSZ, "-z", "-i", str(link), "-t", "f32", "-l", f"1552x{1480*nz}", "-m", "abs", "-e", repr(EB)]
    # As benchkit's cusz adapter: --repeat timing runs do not write the archive, so
    # time with --repeat, then write the archive and reconstruction with plain calls.
    zt = sh(base + ["-S", "write2disk", "-R", "time", "--repeat", str(RUNS)])
    sh(base + ["-R", "cr"])
    xt = sh([CUSZ, "-x", "-i", str(comp), "-S", "write2disk", "-R", "time", "--repeat", str(RUNS)])
    sh([CUSZ, "-x", "-i", str(comp)])
    z, x = zt, xt
    def ms(t, k):
        v = [json.loads(l)[k] for l in t.splitlines() if l.strip().startswith("{") and k in l]
        return float(np.median(v[1:] if len(v) > 1 else v))
    nb = Path(vol).stat().st_size
    r = dict(tool="cuSZ", cr=nb / comp.stat().st_size,
             cmp_gbs=nb / (ms(z.stdout, "compress_device_ms") * 1e6),
             dec_gbs=nb / (ms(x.stdout, "decompress_device_ms") * 1e6))
    r.update(err_stats(orig, rec)); comp.unlink(); rec.unlink(); return r

def main():
    ds = sys.argv[1]; vol, nz = VOLS[ds]
    work = Path(os.environ.get("CT_WORK", "/tmp")) / f"bl_{ds}"; work.mkdir(parents=True, exist_ok=True)
    parts = batches(vol, nz, work)
    rows = []
    only = os.environ.get("BL_ONLY")
    fns = [f for f in (pfpl, cuszp3, cuszhi, cusz) if not only or f.__name__ in only.split(",")]
    for fn in fns:
        per = []
        try:
            for pv, pnz in parts:
                if not gpu_idle(): sys.exit("GPU busy")
                orig = np.fromfile(pv, np.float32)
                per.append((Path(pv).stat().st_size, fn(pv, pnz, work, orig)))
            nb = sum(b for b, _ in per)
            r = dict(tool=per[0][1]["tool"], batches=len(per),
                     cr=nb / sum(b / x["cr"] for b, x in per),
                     cmp_gbs=nb / sum(b / x["cmp_gbs"] for b, x in per),
                     dec_gbs=nb / sum(b / x["dec_gbs"] for b, x in per),
                     max_err=max(x["max_err"] for _, x in per),
                     psnr_per_batch=[x["psnr"] for _, x in per])
        except Exception as ex: r = dict(tool=fn.__name__, error=str(ex)[:300])
        r["dataset"] = ds; r["eb_abs"] = EB; rows.append(r); print(json.dumps(r), flush=True)
    out = HERE/f"baselines_volume_{ds}.json"
    prev = json.load(open(out)) if out.exists() and only else []
    keep = [r for r in prev if r.get("tool") not in {x.get("tool") for x in rows}]
    json.dump(keep + rows, open(out,"w"), indent=1)

if __name__ == "__main__":
    main()
