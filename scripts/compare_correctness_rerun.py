#!/usr/bin/env python3
"""Compare a correctness-only rerun against published benchkit sessions.

For every FZGM cell, joined on (variant, pipeline, dataset, field, error bound):
  - status change (newly failing / newly passing)
  - compressed size change (compressed_bytes)
  - reconstruction change (decompressed_sha256)
  - bound check (eb_satisfied / err_over_bound)
and, within the rerun, staged (off) vs specialized (auto) compressed size and
reconstruction identity.

usage: compare_correctness_rerun.py --published-off DIR --published-auto DIR
                                    --new-off DIR --new-auto DIR [--out REPORT.md]
"""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path


def load(d: Path) -> dict:
    f = d / "runs.jsonl"
    files = [f] if f.exists() else sorted(d.glob("*.jsonl"))
    rows = {}
    for p in files:
        for line in p.open():
            r = json.loads(line)
            if r.get("compressor") != "fzgm":
                continue
            key = (r["variant"], r.get("pipeline_ref") or r.get("pipeline"), r["dataset"], r["field"],
                   float(r["error_bound"]))
            rows[key] = r  # later rows supersede earlier ones (resume/merge order)
    return rows


def payload(r):
    """Codec payload bytes. benchkit 7d87ebd started charging the full FZM archive
    (payload + container, recorded as compressed_archive_overhead_bytes); rows from
    before it recorded the payload only. Compare payload to payload."""
    return r.get("compressed_bytes", 0) - (r.get("compressed_archive_overhead_bytes") or 0)


def bound_ok(r):
    if r.get("eb_satisfied") is not None:
        return bool(r["eb_satisfied"])
    e = r.get("err_over_bound")
    return e is not None and e <= 1.001


def compare(pub: dict, new: dict, arm: str, out: list, issues: collections.Counter):
    for key in sorted(set(pub) | set(new), key=str):
        p, n = pub.get(key), new.get(key)
        tag = f"{arm} {key[0]} {key[2]}/{key[3]} eb={key[4]:g}"
        if p is not None and n is not None and p.get("pipeline_sha256") != n.get("pipeline_sha256"):
            # The pipeline definition itself changed (a deliberate preset change such as
            # 11ea87e's PFPL quantization match): differences are expected, not a
            # regression. Count, check the bound, and do not flag size/recon changes.
            issues["pipeline definition changed (expected differences)"] += 1
            if n.get("status") == "ok" and not bound_ok(n):
                issues["bound violated (changed pipeline)"] += 1
                out.append(f"- BOUND {tag}: err/bound {n.get('err_over_bound')} (pipeline changed)")
            continue
        if p is None:
            issues["new cell (not in published)"] += 1; continue
        if n is None:
            issues["missing in rerun"] += 1; continue
        ps, ns = p["status"] == "ok", n["status"] == "ok"
        if ps != ns:
            k = "newly failing" if ps else "newly passing"
            issues[k] += 1
            out.append(f"- {k.upper()} {tag}: {(n.get('error_message') or p.get('error_message') or '')[:160]}")
            continue
        if not ns:
            issues["failing in both"] += 1; continue
        if payload(p) != payload(n):
            issues["payload size changed"] += 1
            out.append(f"- SIZE {tag}: payload {payload(p)} -> {payload(n)}")
        if p.get("decompressed_sha256") != n.get("decompressed_sha256"):
            issues["reconstruction changed"] += 1
            out.append(f"- RECON {tag}: max_abs_err {p.get('max_abs_err')} -> {n.get('max_abs_err')}")
        if bound_ok(p) != bound_ok(n):
            k = "bound newly violated" if bound_ok(p) else "bound newly satisfied"
            issues[k] += 1
            out.append(f"- {k.upper()} {tag}: err/bound {p.get('err_over_bound')} -> {n.get('err_over_bound')}")
        issues["compared"] += 1


def main():
    ap = argparse.ArgumentParser()
    for a in ("--published-off", "--published-auto", "--new-off", "--new-auto"):
        ap.add_argument(a, type=Path, required=True)
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()
    po, pa, no, na = (load(x) for x in (a.published_off, a.published_auto, a.new_off, a.new_auto))
    lines, sections = [], {}
    for arm, pub, new in (("off", po, no), ("auto", pa, na)):
        issues, out = collections.Counter(), []
        compare(pub, new, arm, out, issues)
        sections[arm] = (issues, out)
    # staged vs specialized identity within the rerun
    ident = collections.Counter(); ident_out = []
    for key in sorted(set(no) & set(na), key=str):
        o, u = no[key], na[key]
        if o["status"] != "ok" or u["status"] != "ok":
            continue
        ident["pairs"] += 1
        if payload(o) != payload(u):
            ident["size differs"] += 1
            ident_out.append(f"- SIZE off/auto {key[0]} {key[2]}/{key[3]} eb={key[4]:g}: "
                             f"{o.get('compressed_bytes')} vs {u.get('compressed_bytes')}")
        if o.get("decompressed_sha256") != u.get("decompressed_sha256"):
            ident["reconstruction differs"] += 1
            ident_out.append(f"- RECON off/auto {key[0]} {key[2]}/{key[3]} eb={key[4]:g}")
    lines.append("# Correctness rerun vs published sessions\n")
    for arm, (issues, out) in sections.items():
        lines.append(f"## {arm} arm\n")
        lines += [f"- {k}: {v}" for k, v in sorted(issues.items())] + [""]
        if out:
            lines += ["<details><summary>cells</summary>\n"] + out[:500] + ["\n</details>\n"]
    lines.append("## Staged vs specialized within the rerun\n")
    lines += [f"- {k}: {v}" for k, v in sorted(ident.items())] + [""] + ident_out[:500]
    text = "\n".join(lines)
    if a.out:
        a.out.write_text(text)
    print(text[:4000])


if __name__ == "__main__":
    main()
