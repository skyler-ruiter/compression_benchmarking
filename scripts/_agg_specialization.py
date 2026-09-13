#!/usr/bin/env python3
"""Aggregate the 4-machine specialization vs native baselines into one JSON
for the comparison artifact. Not a permanent tool; scratch aggregator."""
import json, math, sys, statistics
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from benchkit import validity
from benchkit.identity import logical_cell_id_from_row

BASE = Path(__file__).resolve().parent.parent / "results" / "baselines"

MACHINES = {
    "H100": {"gpu": "NVIDIA H100 80GB HBM3", "site": "JetStream2 (single-tenant VM)",
             "clocks": True, "arch": "sm_90",
             "auto": "h100-jetstream2-20260905-spec-vsnative-full-auto",
             "staged": "h100-jetstream2-20260905-spec-vsnative-full-staged",
             "smoke_auto": "h100-jetstream2-20260901-spec-vsnative-auto",
             "smoke_staged": "h100-jetstream2-20260901-spec-vsnative-staged"},
    "A100": {"gpu": "NVIDIA A100", "site": "BigRed200 (SLURM)",
             "clocks": True, "arch": "sm_80",
             "auto": "a100-bigred200-20260903-spec-vsnative-auto",
             "staged": "a100-bigred200-20260903-spec-vsnative-staged"},
    "L40S": {"gpu": "NVIDIA L40S", "site": "LAIR (SLURM)",
             "clocks": True, "arch": "sm_89",
             "auto": "l40s-lair-20260903-spec-vsnative-full-auto",
             "staged": "l40s-lair-20260903-spec-vsnative-full-staged"},
    "H200": {"gpu": "NVIDIA H200", "site": "NCSA Delta (shared node, 1/8 GPU)",
             "clocks": False, "arch": "sm_90", "smoke_only": True,
             "auto": "h200-delta-20260903-spec-vsnative-auto",
             "staged": "h200-delta-20260903-spec-vsnative-staged"},
}

VARIANT_LABEL = {
    "cusz": "cuSZ", "cuszp2_outlier": "cuSZp2-outlier", "cuszp2_plain": "cuSZp2-plain",
    "cuszp2_outlier_sp": "cuSZp2-outlier", "cuszp2_plain_sp": "cuSZp2-plain",
    "cuszp3_outlier": "cuSZp3-outlier", "cuszp3_plain": "cuSZp3-plain",
    "cuszp3_outlier_sp": "cuSZp3-outlier", "cuszp3_plain_sp": "cuSZp3-plain",
    "cuszp3_fixed": "cuSZp3-fixed",
    "fzgpu": "FZ-GPU", "pfpl": "PFPL", "szp_composed": "SZp-composed",
    "cuszp3_outlier_hp": "cuSZp3-outlier (hi-prec)",
    "cuszp3_plain_hp": "cuSZp3-plain (hi-prec)",
}
# map fzgm variant -> canonical native variant it reimplements
FZGM_TO_NATIVE = {
    "cusz": "cusz",
    "cuszp2_outlier_sp": "cuszp2_outlier", "cuszp2_plain_sp": "cuszp2_plain",
    "cuszp3_outlier_sp": "cuszp3_outlier", "cuszp3_plain_sp": "cuszp3_plain",
    "cuszp3_fixed": "cuszp3_fixed",
    "fzgpu": "fzgpu", "pfpl": "pfpl",  # szp_composed & *_hp have no native pair -> keep own key
}

def load(dirn):
    p = BASE / dirn / "runs.jsonl"
    rows = [json.loads(l) for l in open(p)]
    return validity.annotate(rows)

def gmean(xs):
    xs = [x for x in xs if x and x > 0]
    if not xs: return None
    return math.exp(sum(math.log(x) for x in xs) / len(xs))

def index(rows):
    """key (side, canon_variant, dataset, field, eb) -> record"""
    idx = {}
    for r in rows:
        if not validity.is_valid(r): continue
        if r.get("status") != "ok": continue
        comp = r.get("compressor")
        var = r.get("variant")
        side = "fzgm" if comp == "fzgm" else "native"
        if side == "fzgm":
            canon = FZGM_TO_NATIVE.get(var, var)
        else:
            canon = var
        key = (side, canon, r["dataset"], r.get("field"), str(r.get("error_bound")))
        tok = r.get("timing_reliable") is not False
        idx[key] = {
            "lcid": logical_cell_id_from_row(r),
            "cr": r.get("cr"),
            "cgbs": r.get("compress_throughput_gbs") if tok else None,
            "dgbs": r.get("decompress_throughput_gbs") if tok else None,
            "raw_variant": var,
        }
    return idx

SMOKE_DATASETS = {"CESM-2D", "HURR", "NYX", "HACC"}

out = {"machines": {}, "generated": "specialization vs native, 4 machines"}

for mach, cfg in MACHINES.items():
    auto = load(cfg["auto"])
    staged = load(cfg["staged"])
    ia, ist = index(auto), index(staged)

    # -------- 1. FZGM vs native (auto arm) per variant --------
    fzgm_vs_native = {}
    for (side, canon, ds, fld, eb), rec in ia.items():
        if side != "fzgm": continue
        nat = ia.get(("native", canon, ds, fld, eb))
        if not nat: continue
        v = fzgm_vs_native.setdefault(canon, {"c_ratio": [], "d_ratio": [],
                                              "fzgm_c": [], "fzgm_d": [],
                                              "nat_c": [], "nat_d": [],
                                              "cr_fzgm": [], "cr_nat": []})
        if rec["cgbs"] and nat["cgbs"]:
            v["c_ratio"].append(rec["cgbs"]/nat["cgbs"]); v["fzgm_c"].append(rec["cgbs"]); v["nat_c"].append(nat["cgbs"])
        if rec["dgbs"] and nat["dgbs"]:
            v["d_ratio"].append(rec["dgbs"]/nat["dgbs"]); v["fzgm_d"].append(rec["dgbs"]); v["nat_d"].append(nat["dgbs"])
        if rec["cr"] and nat["cr"]:
            v["cr_fzgm"].append(rec["cr"]); v["cr_nat"].append(nat["cr"])

    fvn = {}
    for canon, v in fzgm_vs_native.items():
        if not v["c_ratio"] and not v["d_ratio"]: continue
        fvn[VARIANT_LABEL.get(canon, canon)] = {
            "n": len(v["c_ratio"]),
            "c_ratio": gmean(v["c_ratio"]), "d_ratio": gmean(v["d_ratio"]),
            "fzgm_c": gmean(v["fzgm_c"]), "nat_c": gmean(v["nat_c"]),
            "fzgm_d": gmean(v["fzgm_d"]), "nat_d": gmean(v["nat_d"]),
            "cr_fzgm": gmean(v["cr_fzgm"]), "cr_nat": gmean(v["cr_nat"]),
        }

    # -------- 2. specialization: auto vs staged (fzgm cells) --------
    spec = {}
    for key, rec in ia.items():
        side, canon = key[0], key[1]
        if side != "fzgm": continue
        st = ist.get(key)
        if not st: continue
        v = spec.setdefault(canon, {"c_ratio": [], "d_ratio": [], "cr_match": [], "cr_mismatch": 0})
        if rec["cgbs"] and st["cgbs"]:
            v["c_ratio"].append(rec["cgbs"]/st["cgbs"])
        if rec["dgbs"] and st["dgbs"]:
            v["d_ratio"].append(rec["dgbs"]/st["dgbs"])
        if rec["cr"] and st["cr"]:
            v["cr_match"].append(abs(rec["cr"]-st["cr"])/st["cr"])
    specout = {}
    for canon, v in spec.items():
        if not v["c_ratio"] and not v["d_ratio"]: continue
        specout[VARIANT_LABEL.get(canon, canon)] = {
            "n": max(len(v["c_ratio"]), len(v["d_ratio"])),
            "c_ratio": gmean(v["c_ratio"]), "d_ratio": gmean(v["d_ratio"]),
            "c_max": max(v["c_ratio"]) if v["c_ratio"] else None,
            "d_max": max(v["d_ratio"]) if v["d_ratio"] else None,
            "cr_max_reldiff": max(v["cr_match"]) if v["cr_match"] else 0.0,
        }

    # -------- 3. FZGM absolute throughput (auto), full + smoke matrix --------
    def fzgm_abs(idx, smoke=False):
        agg = {}
        for (side, canon, ds, fld, eb), rec in idx.items():
            if side != "fzgm": continue
            if smoke and ds not in SMOKE_DATASETS: continue
            v = agg.setdefault(canon, {"c": [], "d": []})
            if rec["cgbs"]: v["c"].append(rec["cgbs"])
            if rec["dgbs"]: v["d"].append(rec["dgbs"])
        return {VARIANT_LABEL.get(k,k): {"c": gmean(v["c"]), "d": gmean(v["d"]),
                                         "n": max(len(v["c"]),len(v["d"]))}
                for k,v in agg.items() if v["c"] or v["d"]}

    out["machines"][mach] = {
        "meta": {k: cfg[k] for k in ("gpu","site","clocks","arch")},
        "smoke_only": cfg.get("smoke_only", False),
        "n_rows_auto": len(auto),
        "fzgm_vs_native": fvn,
        "specialization": specout,
        "fzgm_abs_full": fzgm_abs(ia, smoke=False),
        "fzgm_abs_smoke": fzgm_abs(ia, smoke=True),
    }
    print(f"{mach}: fvn={len(fvn)} spec={len(specout)}", file=sys.stderr)

outp = Path(__file__).resolve().parent.parent / "results" / "comparisons" / "_spec4_data.json"
outp.write_text(json.dumps(out, indent=1))
print("wrote", outp)
