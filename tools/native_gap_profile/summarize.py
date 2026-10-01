import csv, sys, glob, re, json
def load(tag, exclude=None):
    rows = list(csv.reader(l for l in open(f"ncu_{tag}.csv") if l.startswith('"')))
    hdr = rows[0]; data = [dict(zip(hdr, r)) for r in rows[2:] if len(r) == len(hdr)]
    if exclude: data = [r for r in data if not re.search(exclude, r["Kernel Name"])]
    f = lambda r, k: float(r[k].replace(",", "")) if r.get(k, "") not in ("", "n/a") else 0.0
    t = sum(f(r, "gpu__time_duration.sum") for r in data)            # ns
    rd = sum(f(r, "dram__bytes_read.sum") for r in data); wr = sum(f(r, "dram__bytes_write.sum") for r in data)
    sec = sum(f(r, "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum") for r in data)
    req = sum(f(r, "l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum") for r in data)
    wmax = max(data, key=lambda r: f(r, "gpu__time_duration.sum"))
    return dict(tag=tag, kernels=len(data), time_us=t/1e3, dram_read_MB=rd/1e6, dram_write_MB=wr/1e6,
                dram_GBps=(rd+wr)/t if t else 0, sectors_per_req=sec/req if req else 0,
                main_kernel=wmax["Kernel Name"][:60], main_share=f(wmax,"gpu__time_duration.sum")/t,
                main_sm_pct=f(wmax,"sm__throughput.avg.pct_of_peak_sustained_elapsed"),
                main_dram_pct=f(wmax,"dram__throughput.avg.pct_of_peak_sustained_elapsed"),
                main_occ_pct=f(wmax,"sm__warps_active.avg.pct_of_peak_sustained_active"),
                main_regs=f(wmax,"launch__registers_per_thread"))
if __name__ == "__main__":
    nb = float(sys.argv[1]); out = []
    for spec in sys.argv[2:]:
        tag, _, ex = spec.partition(":")
        s = load(tag, ex or None); s["input_MB"] = nb/1e6
        s["traffic_x"] = (s["dram_read_MB"] + s["dram_write_MB"]) / (nb/1e6); s["eff_GBps"] = nb / (s["time_us"]*1e3)
        out.append(s)
        print(f"{tag:22s} k={s['kernels']:2d} t={s['time_us']:8.0f}us eff={s['eff_GBps']:6.0f}GB/s DRAM={s['dram_GBps']:5.0f}GB/s traffic={s['traffic_x']:.2f}x sect/req={s['sectors_per_req']:.2f} | main {s['main_share']*100:3.0f}% sm={s['main_sm_pct']:.0f}% dram={s['main_dram_pct']:.0f}% occ={s['main_occ_pct']:.0f}% regs={s['main_regs']:.0f} {s['main_kernel']}")
    json.dump(out, open("summary_%s.json" % "_".join(a.partition(':')[0] for a in sys.argv[2:])[:80], "w"), indent=1)
