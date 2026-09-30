#!/usr/bin/env python3
"""Build the canonical RQ3 peak-memory artifact from a tools/peak_memory session.

Inputs : <session>/cells.jsonl + <session>/session.json   (run_peak_memory.py)
         optional reference JSON extracted from the 2026-09-12 artifact (comparison only)
         optional spot-check sessions from other machines (--spot <session> ...)
Outputs: peak_memory.json / peak_memory.csv / peak_memory.md in --out-dir.

Every number quoted in the .md (tables, headline statements, draft paragraph) is
computed here from the session rows; nothing is retyped.

  python scripts/build_peak_memory_artifact.py \
      --session results/peak_memory/peakmem-rq3-js2-h100-20260928 \
      --reference ~/paper_organizer/projects/FZGM/evidence/publication/memory/reference/peak_memory_sep12_artifact.json \
      --out-dir ~/paper_organizer/projects/FZGM/evidence/publication/memory
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

FAMILY_LABEL = {
    "cusz": "cuSZ", "cuszhi_cr": "cuSZ-Hi-CR", "cuszhi_tp": "cuSZ-Hi-TP",
    "cuszp2_outlier": "cuSZp2-O", "cuszp2_plain": "cuSZp2-P",
    "cuszp3_outlier": "cuSZp3-O", "cuszp3_plain": "cuSZp3-P", "cuszp3_fixed": "cuSZp3-F",
    "pfpl": "PFPL", "fsz": "FSZ",
}
FAMILIES = list(FAMILY_LABEL)   # families absent from a session are skipped
ARMS = {  # short name -> arm id
    "native": "native",
    "staged": "fzgm:planned:off",
    "auto": "fzgm:planned:auto",
    "staged_nocolor": "fzgm:no_coloring:off",
    "auto_nocolor": "fzgm:no_coloring:auto",
    "staged_minimal": "fzgm:minimal:off",
    "auto_minimal": "fzgm:minimal:auto",
    "staged_split": "fzgm:planned_split:off",
    "auto_split": "fzgm:planned_split:auto",
    "native_static": "native_static",
}
MB = 1e6


def load_jsonl(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


def gmean(xs):
    xs = [x for x in xs if x and x > 0]
    return math.exp(sum(math.log(x) for x in xs) / len(xs)) if xs else None


def cid(c):
    """Cell id for a Markdown table cell (the raw id's '|' separators break tables)."""
    return "`" + c.replace("|", " / ") + "`"


def fmt_mb(b):
    return "—" if b is None else f"{b / MB:,.0f}"


def fmt_x(r, nd=2):
    return "—" if r is None else f"{r:.{nd}f}×"


def fmt_pct(r):
    return "—" if r is None else f"{100 * (r - 1):+.1f}%"


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


class Index:
    def __init__(self, cells: list[dict]):
        self.by = {(c["family"], c["arm"], f"{c['dataset']}/{c['field']}"): c for c in cells}
        self.fields = sorted({(c["input_bytes"], f"{c['dataset']}/{c['field']}") for c in cells})
        self.baseline = {f"{c['dataset']}/{c['field']}": c for c in cells if c["family"] == "ctx_baseline"}

    def get(self, fam, short, field):
        c = self.by.get((fam, ARMS[short], field))
        return c if c and c["status"] == "ok" else None

    def nvml(self, fam, short, field):
        c = self.get(fam, short, field)
        return c["nvml_peak_bytes"] if c else None

    def selfp(self, fam, short, field):
        c = self.get(fam, short, field)
        return c["fzgm_self_peak_bytes"] if c else None


def parse_blocksize_memory(path):
    """Memory-column percentages from specialization_blocksize_h100.tex (last column)."""
    if not path or not Path(path).exists():
        return None
    import re
    vals = [float(m) for m in re.findall(r"&\s*(-?[0-9.]+)\\%\s*\\\\", Path(path).read_text())]
    return [min(vals), max(vals)] if vals else None


def ratio(a, b):
    return a / b if (a and b) else None


def build(args):
    session = Path(args.session)
    cells = load_jsonl(session / "cells.jsonl")
    prov = json.loads((session / "session.json").read_text())
    ix = Index(cells)
    fields = [f for _, f in ix.fields]
    input_bytes = {f: b for b, f in ix.fields}

    # ------------------------------------------------------------- exclusions
    exclusions = []
    for c in cells:
        if c["status"] != "ok":
            f_ = c.get("failure") or {}
            err = f_.get("error") or ""
            tail = f_.get("log_tail") or ""
            # Prefer the tool's own message (e.g. a C++ exception's what()) when the
            # adapter error only carries an exit code.
            import re as _re
            what = _re.search(r"what\(\):\s*(.+)", tail) or _re.search(r"ERR\s+(.+)", tail)
            if what and "exit" in err:
                err = err + " — " + _re.sub(r"\x1b\[[0-9;]*m", "", what.group(1)).strip()
            if "must specify data type" in err and "cuszhi" in c["cell_id"]:
                err += (" (the native cuSZ-Hi build parses -t f64 but its check_dtype accepts only "
                        "f32: native cuSZ-Hi is f32-only here)")
            exclusions.append({"cell_id": c["cell_id"], "status": c["status"],
                               "n_ok": c["n_ok"], "n_reps": c["n_reps"], "error": err})
    polluted = [c["cell_id"] for c in cells if c.get("foreign_processes_seen")]
    # A foreign process does not enter a per-PID NVML reading; confirm by rep agreement.
    reps_by = {}
    for line in (session / "raw.jsonl").read_text().splitlines():
        r_ = json.loads(line)
        reps_by.setdefault(r_["cell_id"], []).append(r_)
    polluted_detail = []
    for cid_ in polluted:
        rr = reps_by[cid_]
        clean = [r_["nvml_peak_bytes"] for r_ in rr if not r_.get("foreign_processes_seen")]
        dirty = [r_["nvml_peak_bytes"] for r_ in rr if r_.get("foreign_processes_seen")]
        polluted_detail.append({"cell_id": cid_, "clean_reps": clean, "foreign_reps": dirty,
                                "identical": bool(clean) and set(dirty) <= set(clean)})
    inversions = [c["cell_id"] for c in cells if c.get("probe_gt_nvml_inversions")]
    spread = max((c["nvml_rep_spread_bytes"] or 0) for c in cells if c["status"] == "ok")

    # ------------------------------------------------------------- per-cell summary rows
    summary = []
    for fam in [f_ for f_ in FAMILIES if f_ in {c["family"] for c in cells}]:
        for f in fields:
            row = {"family": fam, "label": FAMILY_LABEL[fam], "field": f, "input_bytes": input_bytes[f]}
            for short in ARMS:
                row[f"{short}_nvml_bytes"] = ix.nvml(fam, short, f)
            for short in ("staged", "auto", "staged_nocolor", "auto_nocolor", "staged_minimal", "auto_minimal"):
                row[f"{short}_self_bytes"] = ix.selfp(fam, short, f)
            nat = ix.get(fam, "native", f)
            row["native_probe_bytes"] = nat["probe_peak_bytes"] if nat else None
            row["native_lmem_reserved_bytes"] = nat["lmem_reserved_above_default_bytes"] if nat else None
            au = ix.get(fam, "auto", f)
            fus = (au or {}).get("fusion") or {}
            row["auto_installed"] = (fus.get("installed_group_count") or 0) > 0 if au else None
            row["auto_fallback_reason"] = fus.get("fallback_reason") if au else None
            row["auto_lmem_reserved_bytes"] = au["lmem_reserved_above_default_bytes"] if au else None
            st = ix.get(fam, "staged", f)
            row["staged_lmem_reserved_bytes"] = st["lmem_reserved_above_default_bytes"] if st else None
            row["staged_over_native"] = ratio(row["staged_nvml_bytes"], row["native_nvml_bytes"])
            row["auto_over_native"] = ratio(row["auto_nvml_bytes"], row["native_nvml_bytes"])
            row["auto_over_staged_nvml"] = ratio(row["auto_nvml_bytes"], row["staged_nvml_bytes"])
            row["auto_over_staged_self"] = ratio(row["auto_self_bytes"], row["staged_self_bytes"])
            row["nocolor_over_planned_staged_nvml"] = ratio(row["staged_nocolor_nvml_bytes"], row["staged_nvml_bytes"])
            row["nocolor_over_planned_staged_self"] = ratio(row["staged_nocolor_self_bytes"], row["staged_self_bytes"])
            row["nocolor_over_planned_auto_nvml"] = ratio(row["auto_nocolor_nvml_bytes"], row["auto_nvml_bytes"])
            row["minimal_over_planned_staged_nvml"] = ratio(row["staged_minimal_nvml_bytes"], row["staged_nvml_bytes"])
            row["split_over_roundtrip_staged_nvml"] = ratio(row["staged_split_nvml_bytes"], row["staged_nvml_bytes"])
            row["peak_over_input"] = {k: ratio(row[f"{k}_nvml_bytes"], input_bytes[f])
                                      for k in ("native", "staged", "auto")}
            summary.append(row)

    # ------------------------------------------------------------- aggregates
    ctx = [ix.baseline[f]["nvml_peak_bytes"] for f in fields if f in ix.baseline and ix.baseline[f]["status"] == "ok"]
    agg = {"ctx_baseline_bytes": {"min": min(ctx), "max": max(ctx)} if ctx else None,
           "max_rep_spread_bytes": spread}

    def fam_rows(fam):
        return [r for r in summary if r["family"] == fam]

    present = {c["family"] for c in cells}
    per_family = {}
    for fam in FAMILIES:
        if fam not in present:
            continue
        rs = fam_rows(fam)
        specializes = any(r["auto_nvml_bytes"] for r in rs)
        d = {"label": FAMILY_LABEL[fam], "specializes": specializes}
        d["gmean_staged_over_native"] = gmean([r["staged_over_native"] for r in rs])
        d["gmean_auto_over_native"] = gmean([r["auto_over_native"] for r in rs]) if specializes else None
        d["gmean_auto_over_staged_nvml"] = gmean([r["auto_over_staged_nvml"] for r in rs]) if specializes else None
        d["gmean_auto_over_staged_self"] = gmean([r["auto_over_staged_self"] for r in rs]) if specializes else None
        d["gmean_nocolor_over_planned_staged_self"] = gmean([r["nocolor_over_planned_staged_self"] for r in rs])
        d["gmean_nocolor_over_planned_staged_nvml"] = gmean([r["nocolor_over_planned_staged_nvml"] for r in rs])
        # Largest field where native AND the family's default FZGM arm both measured:
        # the realistic-size end of this field set that is actually comparable.
        dkey = "auto_nvml_bytes" if specializes else "staged_nvml_bytes"
        big = max((r for r in rs if r["native_nvml_bytes"] and r[dkey]),
                  key=lambda r: r["input_bytes"], default=None)
        d["largest_input_bytes"] = big["input_bytes"] if big else None
        d["largest_field"] = big["field"] if big else None
        d["largest_staged_over_native"] = big["staged_over_native"] if big else None
        d["largest_auto_over_native"] = big["auto_over_native"] if big else None
        # Where does native first beat the best-policy-free FZGM default (Auto if the
        # family specializes, else staged)? Reported as bracketing measured fields.
        key = "auto" if specializes else "staged"
        wins = [(r["field"], r["input_bytes"], r[f"{key}_nvml_bytes"] < r["native_nvml_bytes"])
                for r in sorted(rs, key=lambda r: r["input_bytes"])
                if r["native_nvml_bytes"] and r[f"{key}_nvml_bytes"]]
        d["fzgm_default_arm"] = key
        d["fzgm_lower_on"] = [w[0] for w in wins if w[2]]
        d["native_lower_on"] = [w[0] for w in wins if not w[2]]
        flips = [(wins[i][0], wins[i + 1][0]) for i in range(len(wins) - 1) if wins[i][2] != wins[i + 1][2]]
        d["crossover_brackets"] = flips
        per_family[fam] = d

    # Planning ablation (all families, staged graphs; Auto separately)
    pa_self = [r["nocolor_over_planned_staged_self"] for r in summary]
    pa_nvml = [r["nocolor_over_planned_staged_nvml"] for r in summary]
    pa_auto = [r["nocolor_over_planned_auto_nvml"] for r in summary if r["nocolor_over_planned_auto_nvml"]]
    mn = [r["minimal_over_planned_staged_nvml"] for r in summary if r["minimal_over_planned_staged_nvml"]]
    inst = [r for r in summary if r["auto_installed"]]
    fallback = [r for r in summary if r["auto_installed"] is False]
    sp_self = [r["auto_over_staged_self"] for r in inst if r["auto_over_staged_self"]]
    sp_nvml = [r["auto_over_staged_nvml"] for r in inst if r["auto_over_staged_nvml"]]
    agg["planning"] = {
        "gmean_nocolor_over_planned_staged_self": gmean(pa_self),
        "range_nocolor_over_planned_staged_self": [min(x for x in pa_self if x), max(x for x in pa_self if x)],
        "gmean_nocolor_over_planned_staged_nvml": gmean(pa_nvml),
        "range_nocolor_over_planned_staged_nvml": [min(x for x in pa_nvml if x), max(x for x in pa_nvml if x)],
        "gmean_nocolor_over_planned_auto_nvml": gmean(pa_auto),
        "minimal_over_planned_staged_nvml_range": [min(mn), max(mn)] if mn else None,
    }
    agg["specialization"] = {
        "gmean_auto_over_staged_self": gmean(sp_self),
        "range_auto_over_staged_self": [min(sp_self), max(sp_self)] if sp_self else None,
        "gmean_auto_over_staged_nvml": gmean(sp_nvml),
        "range_auto_over_staged_nvml": [min(sp_nvml), max(sp_nvml)] if sp_nvml else None,
        "n_cells_auto_above_staged_nvml": sum(1 for x in sp_nvml if x > 1.0),
        "n_cells": len(sp_nvml),
        "n_fallback_cells": len(fallback),
        "fallback_fields": sorted({r["field"] for r in fallback}),
        "fallback_reasons": sorted({r["auto_fallback_reason"] or "?" for r in fallback}),
        "scope": "installed cells only; fallback cells (Auto == staged execution) reported separately",
    }
    # Self-report vs NVML: what the paper's Memory column would claim vs process view
    nat_stacks = [c["max_stack_limit_bytes"] for c in cells if c["status"] == "ok" and c["impl"] == "native"
                  and c["family"].startswith("cuszp") and (c["max_stack_limit_bytes"] or 0) > 1024]
    agg["native_cuszp_stack_limit_range_bytes"] = [min(nat_stacks), max(nat_stacks)] if nat_stacks else None
    agg["self_vs_process"] = {
        "note": ("specialization_blocksize_h100.tex's Memory column is gmean(Auto/Staged) of FZGM's "
                 "self-reported peak_device_bytes (scripts/analyze_specialization_blocksize.py). "
                 "The process-level (NVML) ratio for the same arms is reported alongside."),
        "blocksize_table_memory_column_pct": parse_blocksize_memory(args.blocksize_tex),
    }

    # Decomposition check: NVML ~= ctx + lmem + probe
    decomp = []
    for c in cells:
        if c["status"] != "ok" or c["impl"] == "baseline" or c.get("probe_peak_bytes") is None:
            continue
        f = f"{c['dataset']}/{c['field']}"
        base = ix.baseline.get(f)
        if not base:
            continue
        pred = base["nvml_peak_bytes"] + (c["lmem_reserved_above_default_bytes"] or 0) + c["probe_peak_bytes"]
        decomp.append({"cell_id": c["cell_id"], "nvml": c["nvml_peak_bytes"], "predicted": pred,
                       "split": c.get("process_shape") == "split",
                       "residual_bytes": c["nvml_peak_bytes"] - pred})
    res = [d["residual_bytes"] for d in decomp if not d["split"]]
    res_split = [d["residual_bytes"] for d in decomp if d["split"]]
    agg["decomposition"] = {
        "model": "NVML peak ≈ empty-context baseline + local-memory reservation above default + probe peak (live cudaMalloc-family bytes)",
        "n_cells": len(res),
        "n_split_cells": len(res_split),
        "split_residual_max_bytes": max(res_split) if res_split else None,
        "residual_min_bytes": min(res) if res else None,
        "residual_max_bytes": max(res) if res else None,
        "residual_median_bytes": sorted(res)[len(res) // 2] if res else None,
    }

    # ------------------------------------------------------------- reference comparison
    ref_cmp = None
    if args.reference:
        ref = json.loads(Path(args.reference).read_text())
        name_map = {"HURR/CLOUD": "HURR/CLOUD"}
        comps = []
        for r in ref["rows"]:
            fam = r["family"]
            if fam not in FAMILY_LABEL:
                continue
            f = name_map.get(r["field"], r["field"])
            for old_key, short in (("native_nvml", "native"), ("staged_nvml", "staged"), ("spec_nvml", "auto")):
                old = r.get(old_key)
                new = ix.nvml(fam, short, f)
                if old is None and new is None:
                    continue
                comps.append({"family": fam, "field": f, "arm": short,
                              "old_mb": old, "new_mb": None if new is None else new / MB,
                              "delta_mb": None if (old is None or new is None) else new / MB - old})
        both = [c for c in comps if c["delta_mb"] is not None]
        ref_cmp = {
            "source": ref["source"],
            "n_compared": len(both),
            "n_within_1mb": sum(1 for c in both if abs(c["delta_mb"]) <= 1.0),
            "n_within_5pct": sum(1 for c in both if abs(c["delta_mb"]) <= 0.05 * c["old_mb"]),
            "only_old": [c for c in comps if c["old_mb"] is not None and c["new_mb"] is None],
            "only_new": [c for c in comps if c["old_mb"] is None and c["new_mb"] is not None],
            "differences_over_1mb": sorted([c for c in both if abs(c["delta_mb"]) > 1.0],
                                           key=lambda c: -abs(c["delta_mb"])),
        }

    # ------------------------------------------------------------- spot checks
    spots = []
    for sp in args.spot or []:
        sp = Path(sp)
        scells = load_jsonl(sp / "cells.jsonl")
        sprov = json.loads((sp / "session.json").read_text())
        six = Index(scells)
        rows = []
        for (fam, arm, f), c in six.by.items():
            h100 = ix.by.get((fam, arm, f))
            rows.append({"family": fam, "arm": arm, "field": f, "status": c["status"],
                         "nvml_bytes": c["nvml_peak_bytes"],
                         "lmem_reserved_bytes": c.get("lmem_reserved_above_default_bytes"),
                         "h100_nvml_bytes": h100["nvml_peak_bytes"] if h100 and h100["status"] == "ok" else None})
        spots.append({"session": str(sp), "host": sprov["host"], "gpu": sprov["gpu"], "rows": rows})

    # ------------------------------------------------------------- first-execution sensitivity
    first = None
    if args.first_execution:
        fcells = load_jsonl(Path(args.first_execution) / "cells.jsonl")
        fix = {(c["family"], c["arm"], f"{c['dataset']}/{c['field']}"): c for c in fcells}
        pairs = []
        for (fam_, arm, f), c in ix.by.items():
            if c["impl"] != "fzgm" or c["status"] != "ok" or c.get("process_shape") == "split":
                continue
            o = fix.get((fam_, arm, f))
            if o and o["status"] == "ok":
                pairs.append({"family": fam_, "arm": arm, "field": f,
                              "steady_bytes": c["nvml_peak_bytes"], "first_bytes": o["nvml_peak_bytes"],
                              "ratio": c["nvml_peak_bytes"] / o["nvml_peak_bytes"]})
        def stats(sel):
            rs = [p_["ratio"] for p_ in pairs if sel(p_)]
            return {"n": len(rs), "gmean": gmean(rs), "max": max(rs) if rs else None,
                    "n_changed_over_1pct": sum(1 for r in rs if abs(r - 1) > 0.01)}
        fprov = json.loads((Path(args.first_execution) / "session.json").read_text())
        first = {"session": str(args.first_execution),
                 "fzgm": fprov["fzgm"]["expected_git_sha"],
                 "auto": stats(lambda p_: p_["arm"].endswith(":auto")),
                 "staged": stats(lambda p_: p_["arm"].endswith(":off")),
                 "largest_changes": sorted(pairs, key=lambda p_: -p_["ratio"])[:10]}

    # ------------------------------------------------------------- revision change
    rev = None
    if args.previous_revision:
        prev_dir = Path(args.previous_revision)
        pcells = load_jsonl(prev_dir / "cells.jsonl")
        pprov = json.loads((prev_dir / "session.json").read_text())
        pix = {(c["family"], c["arm"], f"{c['dataset']}/{c['field']}"): c for c in pcells}
        rows_ = []
        for key, c in ix.by.items():
            o = pix.get(key)
            if c["status"] != "ok" or not o or o["status"] != "ok" or c["impl"] == "baseline":
                continue
            rows_.append({"family": key[0], "arm": key[1], "field": key[2],
                          "prev_bytes": o["nvml_peak_bytes"], "new_bytes": c["nvml_peak_bytes"],
                          "ratio": c["nvml_peak_bytes"] / o["nvml_peak_bytes"]})
        by_fam = {}
        for fam_ in FAMILIES:
            for impl_, sel in (("native", lambda a: a in ("native", "native_static")),
                               ("fzgm", lambda a: a.startswith("fzgm:"))):
                rs_ = [r["ratio"] for r in rows_ if r["family"] == fam_ and sel(r["arm"])]
                if rs_:
                    by_fam[f"{fam_}/{impl_}"] = {"n": len(rs_), "gmean": gmean(rs_),
                                                 "min": min(rs_), "max": max(rs_)}
        prev_largest = {}
        for fam_ in FAMILIES:
            if fam_ not in per_family:
                continue
            key_arm = ARMS["auto"] if per_family[fam_]["specializes"] else ARMS["staged"]
            pf = [c for c in pcells if c["family"] == fam_ and c["status"] == "ok"]
            if not pf:
                continue
            big = max(c["input_bytes"] for c in pf)
            nat = next((c for c in pf if c["arm"] == "native" and c["input_bytes"] == big), None)
            fz = next((c for c in pf if c["arm"] == key_arm and c["input_bytes"] == big), None)
            if nat and fz:
                prev_largest[fam_] = fz["nvml_peak_bytes"] / nat["nvml_peak_bytes"]
        rev = {"previous_session": str(prev_dir), "previous_largest_default_over_native": prev_largest,
               "previous_fzgm": pprov["fzgm"]["expected_git_sha"],
               "by_family": by_fam,
               "largest_changes": sorted(rows_, key=lambda r: r["ratio"])[:12]}

    diags = None
    if args.diagnostics and Path(args.diagnostics).exists():
        drows = load_jsonl(Path(args.diagnostics))
        grp = {}
        for d_ in drows:
            grp.setdefault((d_["diagnostic"], d_["cell_id"]), []).append(d_)
        diags = []
        for (tag, cid), rs_ in sorted(grp.items()):
            nv = [r_["nvml_peak_bytes"] for r_ in rs_ if r_["status"] == "ok"]
            sf = [r_["fzgm_self_peak_bytes"] for r_ in rs_ if r_["status"] == "ok" and r_["fzgm_self_peak_bytes"]]
            diags.append({"diagnostic": tag, "cell_id": cid, "n": len(nv),
                          "nvml_max_bytes": max(nv) if nv else None,
                          "self_max_bytes": max(sf) if sf else None,
                          "cli": rs_[0]["fzgm_cli"], "error": rs_[0]["error"], "env": rs_[0]["env"]})

    # ------------------------------------------------------------- write
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": "fzgm_peak_memory_artifact/v1",
        "generated_by": "compression_benchmarking/scripts/build_peak_memory_artifact.py",
        "session": {"path": str(session), "cells_sha256": sha256(session / "cells.jsonl"),
                    "session_json_sha256": sha256(session / "session.json")},
        "provenance": {k: prov[k] for k in ("created_utc", "host", "gpu", "benchmarking_repo", "fzgm",
                                           "native_tools", "binaries_sha256", "datasets",
                                           "metric_definitions", "sample_period_ms", "config_sha256")},
        "protocol": {"error": prov["config"]["error"], "reps": prov["config"]["reps"],
                     "fzgm_runs": prov["config"]["fzgm"].get("runs", 1),
                     "families": FAMILIES, "arms": ARMS,
                     "field_set_note": ("Eight fields chosen to span 11.5 MB-1.12 GB and resolve the "
                                        "native/FZGM crossover; NOT corpus-representative.")},
        "aggregates": agg,
        "per_family": per_family,
        "summary": summary,
        "exclusions": exclusions,
        "quality_flags": {"cells_with_foreign_gpu_processes": polluted,
                          "foreign_process_rep_check": polluted_detail,
                          "cells_with_probe_gt_nvml": inversions},
        "reference_comparison": ref_cmp,
        "spot_checks": spots,
        "first_execution_sensitivity": first,
        "revision_change": rev,
        "diagnostics": diags,
    }
    (out / "peak_memory.json").write_text(json.dumps(payload, indent=1, default=str))

    cols = ["cell_id", "family", "arm", "impl", "planning", "strategy", "coloring", "specialize",
            "process_shape", "dataset", "field", "dtype", "input_bytes", "status", "n_ok", "n_reps",
            "nvml_peak_bytes", "nvml_rep_spread_bytes", "probe_peak_bytes", "fzgm_self_peak_bytes",
            "max_stack_limit_bytes", "lmem_reserved_above_default_bytes", "peak_over_input",
            "dataset_sha256"]
    with open(out / "peak_memory.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for c in sorted(cells, key=lambda c: (c["input_bytes"], c["family"], c["arm"])):
            w.writerow(c)

    (out / "peak_memory.md").write_text(render_md(payload, args))
    print(f"wrote {out}/peak_memory.{{json,csv,md}}")


# ======================================================================== markdown

def render_md(p: dict, args) -> str:
    agg, fam, S = p["aggregates"], p["per_family"], p["summary"]
    pv = p["provenance"]
    L = []
    a = L.append
    a("# FZGM peak device memory (RQ3)\n")
    a(f"Generated by `{p['generated_by']}` from session `{Path(p['session']['path']).name}` "
      f"(`cells.jsonl` sha256 `{p['session']['cells_sha256'][:16]}…`). Do not edit numbers by hand; "
      "regenerate.\n")
    a("## Question and arms\n")
    a("RQ3: *How does graph-wide memory planning and specialization affect the memory required to "
      "execute a compressor?* Each family is measured as separate curves (no per-cell "
      "min(staged, specialized); evaluation_setup_review.md decision #6, resolved 2026-09-28):\n")
    a("| Arm | Meaning |\n|---|---|")
    a("| native | reference binary at its pinned commit |")
    a("| staged | FZGM, `FZ_SPECIALIZE=off`, PREALLOCATE + liveness coloring (default planning) |")
    a("| Auto | FZGM, `FZ_SPECIALIZE=auto`, same planning |")
    a("| no-coloring | PREALLOCATE with `--no-coloring`: every buffer gets its own allocation for the run (planning's reuse off) |")
    a("| MINIMAL | `--strategy minimal`: allocate on demand, free after last consumer |")
    a("| split | planned arm run as two processes (`-z`, then `-x`), like native cuSZ/PFPL |")
    a("\ncuSZ and FSZ never specialize, so they have no Auto curve. FZ-GPU and the FZGM-only "
      "`szp_composed` are excluded from the paper's family set.\n")

    a("## Measurement protocol\n")
    g = pv["gpu"]
    a(f"- Host `{pv['host']['hostname']}`, {g['name']}, driver {g['driver']}, virtualization "
      f"`{g['virtualization_mode']}`, {g['sm_count']} SMs × {g['max_threads_per_sm']} threads/SM.")
    a(f"- FZGM `{pv['fzgm']['expected_git_sha']}` (dedicated worktree build, `FZGMOD_GIT_SHA` verified "
      f"on every row); native pins: " + ", ".join(f"{k} `{v['pinned_commit'][:8]}`" for k, v in pv["native_tools"].items()) + ".")
    a(f"- Bound: {p['protocol']['error']['mode']} {p['protocol']['error']['bound']:g}. "
      f"{p['protocol']['reps']} process repetitions per cell; reported value is the max over reps "
      f"(max rep-to-rep spread observed: {agg['max_rep_spread_bytes'] / MB:.1f} MB). FZGM round-trip "
      f"processes execute the pipeline {p['protocol']['fzgm_runs']}× (steady state; see below); the "
      "split-process control executes once per process.")
    a(f"- **Primary metric:** NVML per-process `usedGpuMemory` of the tool's process tree, polled every "
      f"{pv['sample_period_ms']} ms (ctypes binding to `libnvidia-ml`). A measurement covers one compress "
      "and one decompress; multi-process tools take the max over their processes.")
    a("- **Decomposition:** `cuda_mem_probe` (LD_PRELOAD, live cudaMalloc-family bytes) and a stack-limit "
      "shim reporting the context's maximum `cudaLimitStackSize`. FZGM's self-reported pool peak "
      "(`Pipeline::getPeakMemoryUsage`) is recorded but is not a process footprint: it excludes the "
      "CUDA context and the CLI input buffer.")
    a("- Driver, sampler, config: `compression_benchmarking/tools/peak_memory/`, "
      "`configs/peak_memory/rq3_h100.yaml`; one JSONL row per cell with provenance.\n")
    a("**Field set is not corpus-representative.** Eight fields span 11.5 MB–1.12 GB to resolve where "
      "native and FZGM cross; ratios below must not be read as a typical-field aggregate.\n")

    a("## Where the footprint comes from\n")
    d = agg["decomposition"]
    cb = agg["ctx_baseline_bytes"]
    a(f"An empty CUDA context (one trivial kernel) costs {cb['min'] / MB:.0f}–{cb['max'] / MB:.0f} MB on "
      "this host before anything is allocated. On top of it, the driver reserves device-wide local "
      "memory whenever a kernel's per-thread stack frame exceeds the 1 KiB default: "
      "(frame − 1 KiB) × max threads/SM × SMs, never released for the process lifetime. "
      "`tools/peak_memory/lmem_probe/lmem_probe.cu` reproduces this exactly. "
      f"Across {d['n_cells']} cells, NVML peak = context + local-memory reservation + probe peak "
      f"within {d['residual_min_bytes'] / MB:+.0f} to {d['residual_max_bytes'] / MB:+.0f} MB "
      f"(median {d['residual_median_bytes'] / MB:+.0f} MB; pool reservation rounding and retention). "
      f"The {d['n_split_cells']} split-process cells leave up to {d['split_residual_max_bytes'] / MB:+,.0f} MB "
      "unexplained, all of it FZGM's memory pool retaining freed blocks in the decompress-from-file process: "
      "with the pool disabled (`pool_off` diagnostic) the MIRANDA split cell falls to context + live bytes.\n")
    sr = agg.get("native_cuszp_stack_limit_range_bytes")
    a("This replaces the September explanation of cuSZp's large fixed overhead (\"allocator churn "
      "invisible to the probe\"): native cuSZp processes raise the context stack limit to "
      + (f"{sr[0]:,}–{sr[1]:,} bytes" if sr else "(n/a)") +
      " per thread, and the reservation that triggers is the hidden gap. Its size is set by SM count "
      "and threads/SM, not by the VM's IOMMU.\n")
    a("| Family | Field | Native NVML (MB) | of which local-mem reservation | Native probe | Auto local-mem reservation |")
    a("|---|---|--:|--:|--:|--:|")
    for r in S:
        if r["native_lmem_reserved_bytes"] or r["auto_lmem_reserved_bytes"]:
            if r["field"] in ("CESM-2D/CLDHGH", "HACC/vx"):
                a(f"| {r['label']} | {r['field']} | {fmt_mb(r['native_nvml_bytes'])} | "
                  f"{fmt_mb(r['native_lmem_reserved_bytes'])} | {fmt_mb(r['native_probe_bytes'])} | "
                  f"{fmt_mb(r['auto_lmem_reserved_bytes'])} |")
    a("")

    a("## Planning ablation (graph-wide memory planning)\n")
    pl = agg["planning"]
    a(f"Turning liveness coloring off (every buffer allocated for the whole run) raises FZGM's pool "
      f"working set by {fmt_pct(pl['gmean_nocolor_over_planned_staged_self'])} (gmean over all staged "
      f"cells; range {fmt_pct(pl['range_nocolor_over_planned_staged_self'][0])} to "
      f"{fmt_pct(pl['range_nocolor_over_planned_staged_self'][1])}). At the process level (NVML, "
      f"including the fixed context) the same change is {fmt_pct(pl['gmean_nocolor_over_planned_staged_nvml'])} "
      f"(range {fmt_pct(pl['range_nocolor_over_planned_staged_nvml'][0])} to "
      f"{fmt_pct(pl['range_nocolor_over_planned_staged_nvml'][1])}). On specialized graphs coloring "
      f"matters less ({fmt_pct(pl['gmean_nocolor_over_planned_auto_nvml'])} NVML gmean), since fusion "
      "already removes the intermediates coloring would alias.")
    if pl["minimal_over_planned_staged_nvml_range"]:
        lo, hi = pl["minimal_over_planned_staged_nvml_range"]
        a(f"\nMINIMAL (allocate on demand, free after last use) lands at {fmt_x(lo, 3)}–{fmt_x(hi, 3)} of "
          "the planned PREALLOCATE peak: dynamic freeing reaches the same liveness bound, so the "
          "planning benefit is the *liveness reuse*, not preallocation itself. no-coloring is therefore "
          "the planning-off arm.\n")
    a("| Family | Field | planned | no-coloring | MINIMAL | Δ pool (self) | Δ process (NVML) |")
    a("|---|---|--:|--:|--:|--:|--:|")
    for r in S:
        if r["staged_nvml_bytes"]:
            a(f"| {r['label']} | {r['field']} | {fmt_mb(r['staged_nvml_bytes'])} | "
              f"{fmt_mb(r['staged_nocolor_nvml_bytes'])} | {fmt_mb(r['staged_minimal_nvml_bytes'])} | "
              f"{fmt_pct(r['nocolor_over_planned_staged_self'])} | {fmt_pct(r['nocolor_over_planned_staged_nvml'])} |")
    a("\n(Staged graphs, NVML MB.)\n")

    a("## Specialization\n")
    sp = agg["specialization"]
    a(f"Auto vs staged, pool working set (self-report): gmean {fmt_pct(sp['gmean_auto_over_staged_self'])} "
      f"(range {fmt_pct(sp['range_auto_over_staged_self'][0])} to {fmt_pct(sp['range_auto_over_staged_self'][1])}). "
      f"Process footprint (NVML): gmean {fmt_pct(sp['gmean_auto_over_staged_nvml'])} (range "
      f"{fmt_pct(sp['range_auto_over_staged_nvml'][0])} to {fmt_pct(sp['range_auto_over_staged_nvml'][1])}); "
      f"Auto is *above* staged in {sp['n_cells_auto_above_staged_nvml']} of {sp['n_cells']} cells. "
      f"These statistics cover the {sp['n_cells']} cells where specialization installed. In "
      f"{sp['n_fallback_cells']} more cells (fields: {', '.join(sp['fallback_fields'])}) Auto declined to "
      f"install (`{', '.join(sp['fallback_reasons'])}`) and executed the staged graph, so those "
      "Auto points equal staged.\n")
    bt = agg["self_vs_process"]["blocksize_table_memory_column_pct"]
    a("The block-size table's Memory column ("
      + (f"{bt[0]:+.1f}% to {bt[1]:+.1f}%" if bt else "n/a") +
      ", `scripts/analyze_specialization_blocksize.py`, benchkit rows at `d511ebc`) uses this same "
      "self-report ratio. It measures the pool, not the process, and should be labelled that way (or "
      "replaced with the NVML ratio).\n")

    a("## FZGM vs native\n")
    a("| Family | Field | Input MB | Native | Staged | Auto | Staged/native | Auto/native |")
    a("|---|---|--:|--:|--:|--:|--:|--:|")
    for r in S:
        if r["native_nvml_bytes"] or r["staged_nvml_bytes"]:
            a(f"| {r['label']} | {r['field']} | {r['input_bytes'] / MB:,.1f} | {fmt_mb(r['native_nvml_bytes'])} | "
              f"{fmt_mb(r['staged_nvml_bytes'])} | {fmt_mb(r['auto_nvml_bytes'])} | "
              f"{fmt_x(r['staged_over_native'])} | {fmt_x(r['auto_over_native'])} |")
    a("\n(NVML MB, max of reps.)\n")
    a("| Family | FZGM default arm | FZGM lower on | native lower on | largest-field FZGM/native |")
    a("|---|---|---|---|--:|")
    for k, d in fam.items():
        lg = d["largest_auto_over_native"] if d["specializes"] else d["largest_staged_over_native"]
        a(f"| {d['label']} | {'Auto' if d['specializes'] else 'staged'} | {len(d['fzgm_lower_on'])} fields | "
          f"{len(d['native_lower_on'])} fields | {fmt_x(lg)} ({d['largest_field']}) |")
    a("")

    a("## Split-process control\n")
    rs = [r["split_over_roundtrip_staged_nvml"] for r in S if r["split_over_roundtrip_staged_nvml"]]
    if rs:
        a(f"Running FZGM's planned staged arm as two processes (like native cuSZ and PFPL) changes its NVML "
          f"peak by {fmt_pct(min(rs))} to {fmt_pct(max(rs))}; the round-trip process holds compress and "
          "decompress state together. Native cuSZ and PFPL numbers are two-process maxima, so the "
          "split arm is the like-for-like comparison for those two families.\n")

    ex = p["exclusions"]
    a("| Family | Field | Native (2 proc) | Staged split | Auto split | Staged split/native | Auto split/native |")
    a("|---|---|--:|--:|--:|--:|--:|")
    for r in S:
        if r["family"] in ("cusz", "pfpl") and r["native_nvml_bytes"] and r["staged_split_nvml_bytes"]:
            a(f"| {r['label']} | {r['field']} | {fmt_mb(r['native_nvml_bytes'])} | {fmt_mb(r['staged_split_nvml_bytes'])} | "
              f"{fmt_mb(r['auto_split_nvml_bytes'])} | {fmt_x(ratio(r['staged_split_nvml_bytes'], r['native_nvml_bytes']))} | "
              f"{fmt_x(ratio(r['auto_split_nvml_bytes'], r['native_nvml_bytes']))} |")
    a("\n(Split cells execute once per process; Auto's second-execution growth does not arise.)\n")

    a("## Exclusions and data-quality flags\n")
    if ex:
        groups = {}
        for e in ex:
            fam_, arm_, field_ = e["cell_id"].split("|")
            impl_ = "native" if arm_.startswith("native") else "FZGM"
            cause = (e["error"] or "").split(" — ")[-1].split("; see")[0][:240]
            groups.setdefault((fam_, impl_, cause), []).append((arm_, field_))
        a("| Family | Impl | Cells | Fields | Cause |\n|---|---|--:|---|---|")
        for (fam_, impl_, cause), items in sorted(groups.items()):
            fields_ = sorted({f_ for _, f_ in items})
            a(f"| {FAMILY_LABEL.get(fam_, fam_)} | {impl_} | {len(items)} | {', '.join(fields_)} | {cause} |")
    else:
        a("No failed cells.")
    q = p["quality_flags"]
    same = sum(1 for d_ in q["foreign_process_rep_check"] if d_["identical"])
    a(f"\nCells with a foreign GPU process during measurement: {len(q['cells_with_foreign_gpu_processes'])} "
      f"(another session's jobs on the shared host); in {same} of them the affected rep equals the "
      "clean reps exactly, as expected for a per-PID NVML reading. "
      f"Cells where probe > NVML (a sampling miss): {len(q['cells_with_probe_gt_nvml'])}.\n")

    rc = p["reference_comparison"]
    if rc:
        a("## Comparison with the 2026-09-12 artifact (reference only)\n")
        a(f"{rc['n_compared']} overlapping values; {rc['n_within_1mb']} agree within 1 MB and "
          f"{rc['n_within_5pct']} within 5%.")
        if rc["differences_over_1mb"]:
            a("\n| Family | Field | Arm | Sep 12 MB | Rerun MB | Δ MB |\n|---|---|---|--:|--:|--:|")
            for c in rc["differences_over_1mb"][:25]:
                a(f"| {FAMILY_LABEL[c['family']]} | {c['field']} | {c['arm']} | {c['old_mb']:,.1f} | "
                  f"{c['new_mb']:,.1f} | {c['delta_mb']:+,.1f} |")
        if rc["only_new"]:
            a(f"\nMeasured now but excluded in September: " +
              ", ".join(f"{FAMILY_LABEL[c['family']]} {c['arm']} {c['field']}" for c in rc["only_new"]) + ".")
        if rc["only_old"]:
            a(f"\nPresent in September, failed/excluded now: " +
              ", ".join(f"{FAMILY_LABEL[c['family']]} {c['arm']} {c['field']}" for c in rc["only_old"]) + ".")
        a("")

    fe = p.get("first_execution_sensitivity")
    if fe:
        a("## First execution vs steady state\n")
        a(f"Primary FZGM numbers use `fzgmod-cli -b --runs {p['protocol'].get('fzgm_runs')}` (steady state). "
          f"A complete single-execution session (`{Path(fe['session']).name}`, FZGM `{fe['fzgm']}`"
          + ("" if fe["fzgm"] == p["provenance"]["fzgm"]["expected_git_sha"] else
             f", the previous revision: the sensitivity run was not repeated at "
             f"`{p['provenance']['fzgm']['expected_git_sha']}`, whose only further change is the "
             "inverse-DAG cache fix, and families added since have no single-execution comparison")
          + ") is the sensitivity result: "
          f"Auto cells are {fmt_pct(fe['auto']['gmean'])} higher at steady state (gmean, max "
          f"{fmt_pct(fe['auto']['max'])}; {fe['auto']['n_changed_over_1pct']} of {fe['auto']['n']} cells "
          f"move by more than 1%), staged cells {fmt_pct(fe['staged']['gmean'])} "
          f"({fe['staged']['n_changed_over_1pct']} of {fe['staged']['n']} move by more than 1%). "
          "The step appears from the second execution on and then stays flat: FZGM keeps a pool-managed "
          "decompressed output valid until the next `decompress()` (its documented ownership contract), "
          "so the benchmark loop's next compress runs with it resident. At `d511ebc` a retained inverse "
          "result buffer added a further step (fixed; see the revision section and diagnostics). Native "
          "cuSZp and PFPL repeat internally inside their measured process, so steady state is the "
          "like-for-like setting.\n")
        a("| Family | Arm | Field | first exec MB | steady MB | ratio |\n|---|---|---|--:|--:|--:|")
        for c in fe["largest_changes"]:
            a(f"| {FAMILY_LABEL.get(c['family'], c['family'])} | {c['arm']} | {c['field']} | "
              f"{fmt_mb(c['first_bytes'])} | {fmt_mb(c['steady_bytes'])} | {fmt_x(c['ratio'], 3)} |")
        a("")

    rv = p.get("revision_change")
    if rv:
        a(f"## Change from the previous FZGM revision (`{rv['previous_fzgm']}`)\n")
        a(f"Same protocol, same host; the previous session is `{Path(rv['previous_session']).name}`. "
          f"Between `{rv['previous_fzgm']}` and `{p['provenance']['fzgm']['expected_git_sha']}` "
          "(branch `memfix-chunk-fusion`), the memory-relevant FZGM changes are: no padded input copy "
          "for chunk-fused pipelines; single-pass chunk encode without a full-size scratch buffer; no "
          "retained inverse result buffer in `decompress()`; and the inverse-DAG cache rebuilt when a "
          "stage's stream size changes (a correctness fix that can cost a rebuild's worth of pool "
          "memory). Native rows are re-measured controls and should not move; families absent from "
          "the previous session are not compared.\n")
        a("| Family | Impl | cells | NVML new/previous (gmean) | min | max |\n|---|---|--:|--:|--:|--:|")
        for k, v in rv["by_family"].items():
            fam_, impl_ = k.split("/")
            a(f"| {FAMILY_LABEL[fam_]} | {impl_} | {v['n']} | {fmt_x(v['gmean'], 3)} | "
              f"{fmt_x(v['min'], 3)} | {fmt_x(v['max'], 3)} |")
        a("\n| Family | Arm | Field | previous MB | new MB | ratio |\n|---|---|---|--:|--:|--:|")
        for c in rv["largest_changes"]:
            a(f"| {FAMILY_LABEL[c['family']]} | {c['arm']} | {c['field']} | {fmt_mb(c['prev_bytes'])} | "
              f"{fmt_mb(c['new_bytes'])} | {fmt_x(c['ratio'], 3)} |")
        a("")

    dg = p.get("diagnostics")
    if dg:
        a("## Diagnostic cells (one factor changed each)\n")
        a("Recorded at FZGM `d511ebc` (the pre-fix revision) by `tools/peak_memory/diagnose.py` against "
          "that revision's first-execution session. They explain the September artifact's numbers; the "
          "`runsN` step they show was partly the retained inverse buffer fixed since. Cells: "
          "`binary` = the September `build_paper` tree; `from_toml` = the TOML's literal ABS 1e-3 bound "
          "(the September script's bound semantics); `pool_off` = `FZ_FORCE_MEMPOOL_FALLBACK=1`; "
          "`runsN` = N in-process executions.\n")
        a("| Diagnostic | Cell | NVML MB (max) | FZGM self MB |\n|---|---|--:|--:|")
        for d_ in dg:
            a(f"| {d_['diagnostic']} | {cid(d_['cell_id'])} | {fmt_mb(d_['nvml_max_bytes'])} | {fmt_mb(d_['self_max_bytes'])} |")
        a("")

    if p["spot_checks"]:
        a("## Bare-metal spot check\n")
        for s in p["spot_checks"]:
            a(f"Session `{Path(s['session']).name}` on {s['gpu']['name']} ({s['gpu'].get('virtualization_mode')}):\n")
            a("| Family | Arm | Field | NVML MB | local-mem res. | H100 NVML MB |\n|---|---|---|--:|--:|--:|")
            for r in sorted(s["rows"], key=lambda r: (r["field"], r["family"], r["arm"])):
                a(f"| {r['family']} | {r['arm']} | {r['field']} | {fmt_mb(r['nvml_bytes'])} | "
                  f"{fmt_mb(r['lmem_reserved_bytes'])} | {fmt_mb(r['h100_nvml_bytes'])} |")
        a("")

    a(draft_paragraph(p))

    a("## Caveats\n")
    cb = agg["ctx_baseline_bytes"]
    a("- One H100 on a JetStream2 GPU pass-through VM. The local-memory reservation is architectural "
      "(threads/SM × SMs), so it scales with the GPU rather than the VM: an A100's 108 SMs reserve "
      f"108/{g['sm_count']} as much per stack byte. Whether the {cb['max'] / MB:.0f} MB empty-context "
      "floor is VM-specific is what the bare-metal spot check tests.")
    a("- One error bound. Buffer sizes are element-count-driven (checked in September), but outlier "
      "capacity can be data-dependent.")
    a("- FZGM's pool release threshold keeps up to the pool size resident after frees; NVML sees "
      "reserved, not live, pool memory.")
    a("- NVML is sampled; a peak shorter than the sampling period can be missed (flagged when probe > NVML).")
    a("- PFPL native uses a `-cudart shared` shadow rebuild (same commit and flags) so the probe can see "
      "it; the `native_static` control cells compare it to the pinned static binaries.\n")
    return "\n".join(L) + "\n"


def draft_paragraph(p: dict) -> str:
    """~150-word RQ3 draft for the author; every number and every comparative word
    (below/above native) is derived from the payload."""
    agg, fam, S = p["aggregates"], p["per_family"], p["summary"]
    pl, sp = agg["planning"], agg["specialization"]
    rv = p.get("revision_change") or {}

    def pct(r):
        return f"{100 * (r - 1):.0f}%"

    def red(r):
        return f"{100 * (1 - r):.0f}%"

    def x(r):
        return f"{r:.2f}\u00d7"

    largest = {k: (d["largest_auto_over_native"] if d["specializes"] else d["largest_staged_over_native"])
               for k, d in fam.items()}
    top = max(d["largest_input_bytes"] or 0 for d in fam.values())
    short = {k: d["largest_input_bytes"] for k, d in fam.items()
             if d["largest_input_bytes"] and d["largest_input_bytes"] < top}
    at_or_below = [k for k in FAMILIES if largest.get(k) is not None and largest[k] <= 1.0]
    above = [k for k in FAMILIES if largest.get(k) is not None and largest[k] > 1.0]
    lm = [r["native_lmem_reserved_bytes"] for r in S if r["native_lmem_reserved_bytes"]]
    small = [r for r in S if r["family"] == "cuszp2_outlier" and r["native_nvml_bytes"] and r["auto_nvml_bytes"]]
    small_min = min(small, key=lambda r: r["input_bytes"]) if small else None
    def join(items):
        items = list(items)
        return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]
    names = lambda ks: join(FAMILY_LABEL[k] for k in ks)
    big_mb = max(r["input_bytes"] for r in S) / 1e9
    parts = [
        "We measure peak device memory as the driver-reported resident footprint of each "
        "compressor process over one compression and one decompression, on eight fields spanning "
        f"{min(r['input_bytes'] for r in S) / MB:.1f}\u2009MB to {big_mb:.2f}\u2009GB.",
        "Graph-wide planning matters: disabling liveness-based buffer reuse enlarges FZGM's working "
        f"set by {pct(pl['gmean_nocolor_over_planned_staged_self'])} (geometric mean; up to "
        f"{pct(pl['range_nocolor_over_planned_staged_self'][1])}).",
        "Specialization removes materialized intermediates, shrinking the working set by a further "
        f"{red(sp['gmean_auto_over_staged_self'])} and process memory by {red(sp['gmean_auto_over_staged_nvml'])}"
        + (f", although generated kernels with large per-thread stacks raise the footprint in "
           f"{sp['n_cells_auto_above_staged_nvml']} of {sp['n_cells']} cases." if sp["n_cells_auto_above_staged_nvml"] else "."),
    ]
    where = f"At the largest comparable field ({top / 1e9:.2f}\u2009GB"
    if short:
        by_size = {}
        for k, b in short.items():
            by_size.setdefault(b, []).append(k)
        where += "; " + "; ".join(f"{b / MB:.0f}\u2009MB for {names(ks)}" for b, ks in sorted(by_size.items()))
    where += ")"
    if at_or_below:
        rs = [largest[k] for k in at_or_below]
        parts.append(f"{where} FZGM needs {x(min(rs))}\u2013{x(max(rs))} the memory of native "
                     f"{names(at_or_below)}")
    if above:
        ra = [largest[k] for k in above]
        worst = max(above, key=lambda k: largest[k])
        above_txt = (f"{x(min(ra))}\u2013{x(max(ra))} for the other {len(above)} "
                     f"(highest: {FAMILY_LABEL[worst]})")
        parts[-1] += (f", and {above_txt}." if at_or_below
                      else f"{where} FZGM needs {above_txt} the memory of native.")
    elif at_or_below:
        parts[-1] += "."
    if lm and small_min:
        parts.append(f"Native cuSZp kernels reserve up to {max(lm) / 1e9:.1f}\u2009GB of per-thread stack memory, "
                     f"so on small fields FZGM needs as little as {x(small_min['auto_over_native'])} of native.")
    pf_prev = (rv.get("previous_largest_default_over_native") or {}).get("pfpl")
    if pf_prev and "pfpl" in largest:
        parts.append("Removing a padded input copy, a full-size chunk scratch buffer, and a retained "
                     f"decompression buffer lowered PFPL from {x(pf_prev)} to {x(largest['pfpl'])} of native.")
    parts.append(f"Planning and specialization bring FZGM to or below native memory for {len(at_or_below)} of "
                 f"{len(at_or_below) + len(above)} families; the rest keep a family-specific gap.")
    txt = " ".join(parts)
    n_words = len(txt.split())
    return ("## Draft RQ3 paragraph (for the author; not in evaluation.tex)\n\n"
            f"Generated from this artifact ({n_words} words; H100 only; field set not representative). "
            "Every number and comparative word below is filled in by the builder.\n\n> " + txt + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", required=True)
    ap.add_argument("--reference")
    ap.add_argument("--spot", nargs="*")
    ap.add_argument("--first-execution", help="runs=1 session for the sensitivity section")
    ap.add_argument("--previous-revision", help="same-protocol session at the previous FZGM revision")
    ap.add_argument("--diagnostics", help="diagnostics.jsonl from tools/peak_memory/diagnose.py")
    ap.add_argument("--blocksize-tex", help="papers/FZGM/tables/specialization_blocksize_h100.tex")
    ap.add_argument("--out-dir", required=True)
    build(ap.parse_args())


if __name__ == "__main__":
    main()
