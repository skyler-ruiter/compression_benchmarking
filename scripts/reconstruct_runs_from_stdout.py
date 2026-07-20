#!/usr/bin/env python3
"""Reconstruct a partial runs.jsonl + provenance.json from a benchkit stdout
log (e.g. captured SLURM output) when the harness's own runs.jsonl was never
copied off the machine that produced it.

This is a LOSSY fallback, not a substitute for the real thing. It only
recovers what `report`'s printed summary table shows: comp, variant,
dataset, field, eb, CR, PSNR, cGB/s, dGB/s, eb_ok, tOK. Missing from a real
runs.jsonl: per-rep timing arrays, cv/stability detail, stages[],
gpu_sampling, eb_abs_effective, compressed_bytes, checksums, etc. If the
original runs.jsonl is ever recovered, replace the reconstructed one with it
and delete/ignore this reconstruction.

Usage:
    python scripts/reconstruct_runs_from_stdout.py <stdout.log> <output_dir>

Writes <output_dir>/runs.jsonl and <output_dir>/provenance.json (skeleton —
fill in anything the log didn't have, e.g. exact date, host name).
"""
import json
import re
import sys
from pathlib import Path


def parse_table(lines):
    """Find the 'comp variant dataset field eb CR PSNR cGB/s dGB/s eb_ok tOK graph'
    table (there may be more than one printed copy in a log; use the first) and
    return a list of row dicts."""
    rows = []
    in_table = False
    for line in lines:
        if re.match(r"^comp\s+variant\s+dataset\s+field\s+eb\s+CR\s+PSNR", line):
            in_table = True
            continue
        if in_table and set(line.strip()) <= {"-"} and line.strip():
            continue  # the '----' separator line
        if in_table:
            if not line.strip():
                if rows:
                    break  # blank line after at least one row ends the table
                continue
            parts = line.split()
            if len(parts) < 11:
                break
            comp, variant, dataset, field, eb, cr, psnr, cgbs, dgbs, eb_ok, tok = parts[:11]
            rows.append({
                "compressor": comp,
                "variant": variant,
                "dataset": dataset,
                "field": field,
                "error_bound": float(eb),
                "cr": float(cr),
                "psnr": float(psnr),
                "compress_throughput_gbs": float(cgbs),
                "decompress_throughput_gbs": float(dgbs),
                "eb_satisfied": eb_ok == "True",
                "timing_reliable": tok == "True",
                "status": "ok",
            })
    return rows


def parse_header(lines):
    info = {}
    for line in lines:
        m = re.match(r"\[gpu\]\s+(.+?),\s+(\d+\s*MiB),\s+([\d.]+)", line)
        if m and "gpu_name" not in info:
            info["gpu_name"] = m.group(1).strip()
            info["gpu_memory"] = m.group(2).strip()
            info["gpu_driver"] = m.group(3).strip()
        m = re.match(r"\[benchkit\]\s+session\s*=\s*(.+)", line)
        if m:
            info["session_id"] = m.group(1).strip()
        m = re.match(r"\[benchkit\]\s+experiment\s*=\s*(.+)", line)
        if m:
            info["experiment_config"] = m.group(1).strip()
        m = re.match(r"\[benchkit\]\s+results\s*=\s*(.+)", line)
        if m:
            info["original_results_path"] = m.group(1).strip()
        m = re.match(r"\[node-jobs\]\s+(.+)", line)
        if m:
            info["node_job_label"] = m.group(1).strip()
    return info


def parse_failures(lines):
    fails = []
    for line in lines:
        m = re.match(r"\s*FAIL\s+(\S+):\s*(.+)", line)
        if m:
            fails.append({"run_id": m.group(1), "error_message": m.group(2).strip()})
    return fails


def main():
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)
    log_path = Path(sys.argv[1])
    out_dir = Path(sys.argv[2])
    out_dir.mkdir(parents=True, exist_ok=True)

    lines = log_path.read_text(errors="replace").splitlines()
    rows = parse_table(lines)
    header = parse_header(lines)
    failures = parse_failures(lines)

    if not rows:
        print("No summary table found in this log — nothing to reconstruct.")
        sys.exit(1)

    for r in rows:
        r["reconstructed"] = True
        r["reconstruction_source"] = str(log_path.name)

    runs_path = out_dir / "runs.jsonl"
    with open(runs_path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    provenance = {
        "reconstructed": True,
        "reconstruction_source": str(log_path.name),
        "reconstruction_note": (
            "This provenance.json was reconstructed from a stdout log, not "
            "captured natively by benchkit. Fields not derivable from the log "
            "(exact date, CUDA toolkit patch version, per-compressor CLI paths, "
            "host details) are left blank or approximate — fill in by hand if "
            "known, or replace this file entirely if the original "
            "provenance.json is ever recovered from the source machine."
        ),
        "gpu": {
            "name": header.get("gpu_name"),
            "memory_total": header.get("gpu_memory"),
            "driver": header.get("gpu_driver"),
        },
        "session_id": header.get("session_id"),
        "experiment_config": header.get("experiment_config"),
        "original_results_path": header.get("original_results_path"),
        "node_job_label": header.get("node_job_label"),
    }
    prov_path = out_dir / "provenance.json"
    with open(prov_path, "w") as f:
        json.dump(provenance, f, indent=2)

    ok = len(rows)
    print(f"Reconstructed {ok} ok rows -> {runs_path}")
    print(f"Provenance skeleton -> {prov_path} (fill in blanks by hand)")
    if failures:
        print(f"{len(failures)} failure line(s) found in log (not added as rows, no full row data available):")
        for fl in failures:
            print(f"  {fl['run_id']}: {fl['error_message']}")


if __name__ == "__main__":
    main()
