import json, os, subprocess, struct, sys
import numpy as np
S = os.path.dirname(os.path.abspath(__file__))
CLI = "/home/exouser/FZGPUModules-roibin-381a45e/build_rel/bin/fzgmod-cli"
V = "/media/volume/Compression_Data/sdrbench_data/EXAFEL_130x1480x1552/SDRBENCH-EXAFEL-data-130x1480x1552.f32"
R = "/media/volume/Compression_Data/sdrbench_data/derived/EXAFEL_ROIBIN/peaks/exafel_full.roi"
VARS = {  # name -> (bin1 config, bin2 config)
 "A_bitpack":   ("b1_A_bitpack.toml",   "var_A_bitpack.toml"),
 "B_golomb_tl": ("b1_B_golomb.toml",    "var_B_golomb.toml"),
 "C_tl_rze":    ("b1_C_rze.toml",       "var_C_rze.toml"),
 "D_tl_rze_ans":("b1_D_rze_ans.toml",   "var_D_rze_ans.toml"),
 "E_tl_huff16": ("b1_E_huff16.toml",    "var_E_huff16.toml"),
 "F_tl_rze_ans16":("b1_F_rze_ans16.toml","var_F_rze_ans16.toml"),
 "G_pfpl":      ("b1_G_pfpl.toml",      "b2_G_pfpl.toml"),
 "H_diff_golomb":("b1_H_golomb_chunk.toml","b2_H_golomb_chunk.toml"),
}
def gpu_busy():
    out = subprocess.run(["nvidia-smi","--query-compute-apps=pid","--format=csv,noheader"],capture_output=True,text=True).stdout.strip()
    return out != ""
def sh(args, env=None):
    e = dict(os.environ); e.update(env or {})
    return subprocess.run(args, capture_output=True, text=True, env=e)
x = np.fromfile(V, np.float32).reshape(130,1480,1552)
f = open(R,"rb"); f.read(8); nx,ny,nz,n = struct.unpack("<4I", f.read(16))
rec = np.frombuffer(f.read(8*n), dtype=[("z","<u4"),("x","<u2"),("y","<u2")])
m = np.zeros(x.shape, bool)
for r in rec:
    z,y,xx = int(r["z"]),int(r["y"]),int(r["x"]); m[z,max(0,y-4):y+5,max(0,xx-4):xx+5] = True
rng = float(x.max()-x.min())
rows = []
for name,(c1,c2) in VARS.items():
    for binf,cfg in ((1,c1),(2,c2)):
        cfgp = os.path.join(S,cfg); row = dict(variant=name, bin=binf)
        outs = {}
        for pol in ("off","auto"):
            if gpu_busy(): sys.exit("GPU busy - aborting so timing stays clean")
            fzm = os.path.join(S,f"ct_{pol}.fzm"); out = os.path.join(S,f"ct_{pol}.out")
            z = sh([CLI,"-z","-i",V,"-l","1552x1480x130","-c",cfgp,"-o",fzm],{"FZ_SPECIALIZE":pol})
            xx_ = sh([CLI,"-x","-i",fzm,"-o",out],{"FZ_SPECIALIZE":pol})
            row[f"rc_{pol}"] = (z.returncode, xx_.returncode); outs[pol]=out
            rj = os.path.join(S,f"ct_{name}_b{binf}_{pol}.json")
            b = sh([CLI,"-b","-i",V,"-l","1552x1480x130","-c",cfgp,"--runs","7","--report-json",rj],{"FZ_SPECIALIZE":pol})
            j = json.load(open(rj)); t=j["timing"]; sp=j["specialization"]
            row[f"cr_{pol}"]=j["size"]["ratio"]; row[f"C_{pol}"]=j["throughput"]["compress_gbs"]; row[f"D_{pol}"]=j["throughput"]["decompress_gbs"]
            c=np.array(t["compress"]["device_ms"]["all"][1:]); d=np.array(t["decompress"]["device_ms"]["all"][1:])
            row[f"cv_{pol}"]=round(float(max(c.std()/c.mean(), d.std()/d.mean())),3)
            if pol=="auto": row["inst"]=(sp["installed_group_count"],sp["inverse_installed_group_count"]); row["impl"]=[g["implementation"] for g in sp["groups"]]
        a=open(outs["off"],"rb").read(); b_=open(outs["auto"],"rb").read(); row["ident"]= a==b_
        o=np.frombuffer(b_,np.float32).reshape(x.shape); e=np.abs(o.astype(np.float64)-x)
        row["roi_max"]=round(float(e[m].max()),4); row["bg_max"]=round(float(e[~m].max()),4)
        row["bg_psnr"]=round(float(20*np.log10(rng)-10*np.log10((e[~m]**2).mean())),2)
        for p in outs.values(): os.remove(p)
        rows.append(row); print(json.dumps(row), flush=True)
json.dump(rows, open(os.path.join(S,"clean_timing_results.json"),"w"), indent=1)
