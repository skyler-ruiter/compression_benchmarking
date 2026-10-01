"""Split an ncu capture into compress/decompress kernels of ONE call and summarize."""
import csv, json, re, sys
def rows(tag):
    r = list(csv.reader(l for l in open(f"ncu_{tag}.csv") if l.startswith('"')))
    h = r[0]; return [dict(zip(h, x)) for x in r[2:] if len(x) == len(h)]
def g(r, k):
    v = r.get(k, "").replace(",", ""); return float(v) if v not in ("", "n/a") else 0.0
def stats(ks, nbytes):
    t = sum(g(r, "gpu__time_duration.sum") for r in ks)
    rd = sum(g(r, "dram__bytes_read.sum") for r in ks); wr = sum(g(r, "dram__bytes_write.sum") for r in ks)
    sec = sum(g(r, "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum") for r in ks)
    req = sum(g(r, "l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum") for r in ks)
    m = max(ks, key=lambda r: g(r, "gpu__time_duration.sum"))
    return dict(kernels=len(ks), time_us=t / 1e3, eff_GBps=nbytes / t, read_MB=rd / 1e6, write_MB=wr / 1e6,
                read_x=rd / nbytes, dram_GBps=(rd + wr) / t, sect_per_req=sec / req if req else 0,
                main=m["Kernel Name"][:48], main_share=g(m, "gpu__time_duration.sum") / t,
                main_sm=g(m, "sm__throughput.avg.pct_of_peak_sustained_elapsed"),
                main_dram=g(m, "dram__throughput.avg.pct_of_peak_sustained_elapsed"),
                main_occ=g(m, "sm__warps_active.avg.pct_of_peak_sustained_active"),
                main_regs=g(m, "launch__registers_per_thread"))
def fzgm_b(tag):
    ks = rows(tag); first = ks[0]["Kernel Name"]
    p = next(i for i in range(1, len(ks)) if ks[i]["Kernel Name"] == first)   # warm-up compress = ks[:p]
    return ks[p:2 * p], ks[2 * p:]
def by_name(tag, comp_re, dec_re):
    ks = rows(tag)
    c = [r for r in ks if re.search(comp_re, r["Kernel Name"]) and not re.search(dec_re, r["Kernel Name"])]
    d = [r for r in ks if re.search(dec_re, r["Kernel Name"])]
    return c[:1] if c else [], d[:1] if d else []
def first_run(tag):
    ks = rows(tag); first = ks[0]["Kernel Name"]
    p = next((i for i in range(1, len(ks)) if ks[i]["Kernel Name"] == first), len(ks))
    return ks[:p]
