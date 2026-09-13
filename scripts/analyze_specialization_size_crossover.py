#!/usr/bin/env python3
"""Pair staged/Auto size-crossover rows and expose runtime path transitions."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


VARIANT_RE = re.compile(r"^lorenzo_ab_b(32|64|96|128)_(plain|outlier)$")


def load(path: Path) -> list[dict]:
    with path.open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def runtime_path(row: dict) -> str | None:
    groups = row.get("fusion_groups") or []
    for group in groups:
        if group.get("implementation") == "warp-register":
            return group.get("execution_path")
    return None


def phase_stable(row: dict, phase: str) -> bool:
    return row.get(f"{phase}_stable", row.get("timing_reliable")) is not False


def pair(off_rows: list[dict], auto_rows: list[dict]) -> dict:
    off = {row["logical_cell_id"]: row for row in off_rows
           if row.get("logical_cell_id") and row.get("status") == "ok"}
    auto = {row["logical_cell_id"]: row for row in auto_rows
            if row.get("logical_cell_id") and row.get("status") == "ok"}
    rows = []
    for logical_id in sorted(off.keys() & auto.keys()):
        staged, specialized = off[logical_id], auto[logical_id]
        match = VARIANT_RE.match(specialized.get("variant", ""))
        if match is None:
            continue
        c_speedup = (specialized["compress_throughput_gbs"] /
                     staged["compress_throughput_gbs"])
        d_speedup = (specialized["decompress_throughput_gbs"] /
                     staged["decompress_throughput_gbs"])
        rows.append({
            "n": specialized["num_elements"],
            "bytes": specialized["original_bytes"],
            "block_size": int(match.group(1)),
            "mode": match.group(2),
            "execution_path": runtime_path(specialized),
            "compress_speedup": c_speedup,
            "decompress_speedup": d_speedup,
            "compress_stable": (phase_stable(staged, "compress") and
                                phase_stable(specialized, "compress")),
            "decompress_stable": (phase_stable(staged, "decompress") and
                                  phase_stable(specialized, "decompress")),
            "staged_compress_gbs": staged["compress_throughput_gbs"],
            "auto_compress_gbs": specialized["compress_throughput_gbs"],
            "staged_decompress_gbs": staged["decompress_throughput_gbs"],
            "auto_decompress_gbs": specialized["decompress_throughput_gbs"],
            "compressed_bytes_equal": (staged.get("compressed_bytes") ==
                                       specialized.get("compressed_bytes")),
            # This hashes the full FZM container, including execution-derived
            # internal-buffer metadata, rather than only the compressed payload.
            "container_sha256_equal": (staged.get("compressed_sha256") ==
                                      specialized.get("compressed_sha256")),
            "reconstruction_equal": (staged.get("decompressed_sha256") ==
                                     specialized.get("decompressed_sha256")),
        })
    rows.sort(key=lambda row: (row["block_size"], row["mode"], row["n"]))
    families = []
    for block, mode in sorted({(row["block_size"], row["mode"]) for row in rows}):
        family = [row for row in rows
                  if row["block_size"] == block and row["mode"] == mode]
        reliable = [row for row in family if row["compress_stable"]]
        losses = [row for row in reliable if row["compress_speedup"] < 1.0]
        last_loss_n = losses[-1]["n"] if losses else None
        wins = [row for row in reliable
                if row["compress_speedup"] >= 1.0 and
                (last_loss_n is None or row["n"] > last_loss_n)]
        transitions = []
        previous = None
        for row in family:
            if row["execution_path"] != previous:
                transitions.append({"n": row["n"], "path": row["execution_path"]})
                previous = row["execution_path"]
        families.append({
            "block_size": block,
            "mode": mode,
            "last_reliable_loss_n": last_loss_n,
            "first_reliable_win_n": wins[0]["n"] if wins else None,
            "first_reliable_win_speedup": wins[0]["compress_speedup"] if wins else None,
            "largest_n_speedup": family[-1]["compress_speedup"],
            "path_transitions": transitions,
        })
    return {
        "paired_cells": len(rows),
        "missing_runtime_path": sum(row["execution_path"] is None for row in rows),
        "compressed_size_mismatches": sum(not row["compressed_bytes_equal"] for row in rows),
        "container_sha256_mismatches": sum(
            not row["container_sha256_equal"] for row in rows),
        "reconstruction_mismatches": sum(not row["reconstruction_equal"] for row in rows),
        "reliable_compress_losses": sum(
            row["compress_stable"] and row["compress_speedup"] < 1.0 for row in rows),
        "reliable_decompress_losses": sum(
            row["decompress_stable"] and row["decompress_speedup"] < 1.0 for row in rows),
        "families": families,
        "rows": rows,
    }


def markdown(payload: dict, off_name: str, auto_name: str) -> str:
    lines = [
        "# Specialization size crossover",
        "",
        f"Staged source: `{off_name}`  ",
        f"Auto source: `{auto_name}`",
        "",
        "Speedups are Auto/staged for the same graph, prefix, and bound. An asterisk",
        "marks a phase that failed the paired timing-stability gate; retain its ratio",
        "only as directional evidence until rerun.",
        "",
        "## Sampled crossover summary",
        "",
        "| B | Mode | Last reliable loss | First reliable win | Win speedup | Path regimes |",
        "|---:|---|---:|---:|---:|---|",
    ]
    for family in payload["families"]:
        regimes = ", ".join(
            f"{entry['path']}@N={entry['n']}" for entry in family["path_transitions"])
        lines.append(
            f"| {family['block_size']} | {family['mode']} | "
            f"{family['last_reliable_loss_n'] or '--'} | "
            f"{family['first_reliable_win_n'] or '--'} | "
            f"{family['first_reliable_win_speedup']:.3f}x | {regimes} |"
        )
    lines += [
        "",
        "These are sampled brackets, not fitted crossover points. A gap can contain a",
        "timing-unreliable sample and must not be narrowed without a targeted rerun.",
        "",
        "## Paired measurements",
        "",
        "| B | Mode | N | Path | C speedup | D speedup | Auto C/D (GB/s) |",
        "|---:|---|---:|---|---:|---:|---:|",
    ]
    for row in payload["rows"]:
        c_mark = "" if row["compress_stable"] else "*"
        d_mark = "" if row["decompress_stable"] else "*"
        lines.append(
            f"| {row['block_size']} | {row['mode']} | {row['n']} | "
            f"{row['execution_path'] or '--'} | {row['compress_speedup']:.3f}x{c_mark} | "
            f"{row['decompress_speedup']:.3f}x{d_mark} | "
            f"{row['auto_compress_gbs']:.2f}/{row['auto_decompress_gbs']:.2f} |"
        )
    lines += [
        "",
        f"Paired cells: {payload['paired_cells']}; missing runtime paths: "
        f"{payload['missing_runtime_path']}; compressed-size mismatches: "
        f"{payload['compressed_size_mismatches']}; full-container SHA mismatches: "
        f"{payload['container_sha256_mismatches']}; reconstruction mismatches: "
        f"{payload['reconstruction_mismatches']}.",
        "",
        "All full-container hashes differ because specialization changes an",
        "execution-derived internal-buffer size entry in the FZM header. A direct",
        "Off/Auto byte diff found an identical payload CRC and differences only in",
        "that header entry plus the resulting header CRC. This one-case audit does",
        "not establish payload identity for every campaign row.",
        "",
        f"Reliable compression/decompression losses: "
        f"{payload['reliable_compress_losses']}/{payload['reliable_decompress_losses']}.",
        "",
    ]
    return "\n".join(lines)


def latex(payload: dict) -> str:
    def power(value: int | None) -> str:
        if value is None:
            return "--"
        return f"$2^{{{value.bit_length() - 1}}}$"

    lines = [
        "% Generated by scripts/analyze_specialization_size_crossover.py.",
        "% Do not edit measured values by hand.",
        "\\begin{table}[t]",
        "  \\centering",
        "  \\caption{Sampled H100 compression crossover for Auto versus staged",
        "  execution on checksummed HACC/vx prefixes at $10^{-3}$ relative-range",
        "  error. Unreliable timings are excluded when bracketing.}",
        "  \\label{tab:specialization-size-crossover}",
        "  \\small",
        "  \\begin{tabular}{rrlrr}",
        "    \\toprule",
        "    $B$ & Mode & Last loss & First win & Speedup \\\\",
        "    \\midrule",
    ]
    for family in payload["families"]:
        lines.append(
            f"    {family['block_size']} & {family['mode']} & "
            f"{power(family['last_reliable_loss_n'])} & "
            f"{power(family['first_reliable_win_n'])} & "
            f"{family['first_reliable_win_speedup']:.2f}$\\times$ \\\\"
        )
    lines += [
        "    \\bottomrule",
        "  \\end{tabular}",
        "\\end{table}",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--off", required=True, type=Path)
    parser.add_argument("--auto", required=True, type=Path)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-md", type=Path)
    parser.add_argument("--output-tex", type=Path)
    args = parser.parse_args()

    payload = pair(load(args.off / "runs.jsonl"), load(args.auto / "runs.jsonl"))
    payload["staged_source"] = args.off.name
    payload["auto_source"] = args.auto.name
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(payload, indent=2) + "\n")
    if args.output_md:
        args.output_md.parent.mkdir(parents=True, exist_ok=True)
        args.output_md.write_text(markdown(payload, args.off.name, args.auto.name))
    if args.output_tex:
        args.output_tex.parent.mkdir(parents=True, exist_ok=True)
        args.output_tex.write_text(latex(payload))
    if not args.output_json and not args.output_md and not args.output_tex:
        print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
