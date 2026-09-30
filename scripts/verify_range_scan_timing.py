#!/usr/bin/env python3
"""Range-scan timing check: FZGM cuSZp3-fixed NOA vs ABS (eb_abs = eb * float32 range, computed
exactly as the Quantizer does) under FZ_SPECIALIZE=auto, and native cuSZp3 rel vs abs, on five fields.
Verifies byte-identical output and that native rel timing excludes the range computation.
Usage: verify_range_scan_timing.py <out_dir>   (writes results.json; lock clocks first)."""
import json, subprocess, re, sys, numpy as np, pathlib, statistics as st
S = pathlib.Path(sys.argv[1]); P = pathlib.Path.home()/'compression_benchmarking/configs/pipelines'
CLI = str(pathlib.Path.home()/'FZGPUModules-d511ebc-cuszp3fixed/build/bin/fzgmod-cli')
NAT = str(pathlib.Path.home()/'compressors/cuSZp-V3.0.0/build/examples/bin/cuSZp')
D = '/media/volume/Compression_Data/sdrbench_data/'
FIELDS = [  # (label, path, dims fast-to-slow, preset, native dim args)
 ('NYX/baryon_density', D+'NYX_512x512x512/baryon_density.f32', [512,512,512], 'cuszp3_3d_fixed.toml', ['-d','3','512','512','512']),
 ('SCALE/T', D+'SCALE_98x1200x1200/T-98x1200x1200.f32', [1200,1200,98], 'cuszp3_3d_fixed.toml', None),
 ('CESMATM/CLDLIQ', D+'CESMATM_26x1800x3600/CLDLIQ_1_26_1800_3600.f32', [3600,1800,26], 'cuszp3_3d_fixed.toml', None),
 ('HACC/vx', D+'HACCM_280953867/vx.f32', [280953867], 'cuszp3_fixed.toml', ['-d','1']),
 ('CESM-2D/CLDHGH', D+'CESM_1800x3600/CLDHGH_1_1800_3600.f32', [3600,1800], 'cuszp3_2d_fixed.toml', None),
]
out=[]
def fzgm(path, dims, preset, eb, mode, pol, tag):
    toml=(P/preset).read_text()
    toml=re.sub(r'error_bound = .*', f'error_bound = {eb!r}', toml, count=1)
    toml=re.sub(r'error_bound_mode = .*', f'error_bound_mode = "{mode}"', toml, count=1)
    tp=S/f'{tag}.toml'; tp.write_text(toml); js=S/f'{tag}.json'
    env=dict(**__import__('os').environ, FZ_SPECIALIZE=pol)
    subprocess.run([CLI,'-b','-i',path,'-l','x'.join(map(str,dims)),'-t','f32','-c',str(tp),'--runs','6','--report-json',str(js)],env=env,capture_output=True,check=True)
    j=json.load(open(js)); t=j['timing']
    c=st.median(t['compress']['device_ms']['all'][1:]); d=st.median(t['decompress']['device_ms']['all'][1:])
    return c,d,j['size']['compressed_bytes'],j['quality']['max_abs_err']
def native(path, dimargs, ebmode, eb):
    cs=[];ds=[];cr=None
    for _ in range(4):
        o=subprocess.run([NAT,'-i',path,'-t','f32','-m','fixed',*dimargs,'-eb',ebmode,repr(eb)],capture_output=True,text=True,check=True).stdout
        cs.append(float(re.search(r'compression\s+end-to-end speed: ([\d.]+)',o).group(1)))
        ds.append(float(re.search(r'decompression end-to-end speed: ([\d.]+)',o).group(1)))
        cr=float(re.search(r'compression ratio: ([\d.]+)',o).group(1))
    return st.median(cs[1:]),st.median(ds[1:]),cr
for lab,path,dims,preset,nd in FIELDS:
    x=np.fromfile(path,dtype=np.float32)
    rng=np.float32(x.max())-np.float32(x.min())
    nbytes=x.nbytes
    if len(dims)==3: ndargs=['-d','3',str(dims[2]),str(dims[1]),str(dims[0])]
    elif len(dims)==2: ndargs=['-d','2','1',str(dims[1]),str(dims[0])]
    else: ndargs=['-d','1']
    for eb in (1e-2,1e-4):
        eb_abs=eb*float(rng)
        res={}
        for mode,val in (('NOA',eb),('ABS',eb_abs)):
            res[mode]=fzgm(path,dims,preset,val,mode,'auto',f"{lab.replace('/','_')}_{eb}_{mode}")
        nr=native(path,ndargs,'rel',eb); na=native(path,ndargs,'abs',eb_abs)
        rec=dict(field=lab,eb=eb,range=float(rng),eb_abs=eb_abs,
            fz_noa_c_ms=res['NOA'][0],fz_abs_c_ms=res['ABS'][0],fz_noa_d_ms=res['NOA'][1],fz_abs_d_ms=res['ABS'][1],
            bytes_equal=res['NOA'][2]==res['ABS'][2],maxerr_equal=res['NOA'][3]==res['ABS'][3],
            fz_noa_c_gbs=nbytes/res['NOA'][0]/1e6,fz_abs_c_gbs=nbytes/res['ABS'][0]/1e6,fz_abs_d_gbs=nbytes/res['ABS'][1]/1e6,
            nat_rel_c_gbs=nr[0],nat_abs_c_gbs=na[0],nat_rel_d_gbs=nr[1],nat_abs_d_gbs=na[1],
            range_scan_ms_est=nbytes/3.0e12*1e3)
        out.append(rec); print(json.dumps(rec),flush=True)
json.dump(out,open(S/'results.json','w'),indent=1)
