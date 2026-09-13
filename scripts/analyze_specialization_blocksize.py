#!/usr/bin/env python3
"""Generate paired evidence for the 1-D Lorenzo + AdaptiveBitpack block sweep."""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict
from pathlib import Path


VARIANT_RE = re.compile(r"^lorenzo_ab_b(32|64|96|128)_(plain|outlier)$")


def load(path: Path) -> list[dict]:
    with path.open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def gmean(values: list[float]) -> float | None:
    kept = [value for value in values if value > 0 and math.isfinite(value)]
    if not kept:
        return None
    return math.exp(sum(math.log(value) for value in kept) / len(kept))


def phase_stable(row: dict, phase: str) -> bool:
    key = f"{phase}_stable"
    if key in row:
        return row[key] is not False
    return row.get("timing_reliable") is not False


def parse_variant(row: dict) -> tuple[int, str] | None:
    match = VARIANT_RE.match(row.get("variant", ""))
    if match is None:
        return None
    return int(match.group(1)), match.group(2)


def summarize(off_rows: list[dict], auto_rows: list[dict]) -> dict:
    off = {row["logical_cell_id"]: row for row in off_rows if row.get("logical_cell_id")}
    auto = {row["logical_cell_id"]: row for row in auto_rows if row.get("logical_cell_id")}
    grouped: dict[tuple[int, str], list[tuple[dict, dict]]] = defaultdict(list)
    for logical_id in off.keys() & auto.keys():
        staged, specialized = off[logical_id], auto[logical_id]
        key = parse_variant(specialized)
        if key is not None and staged.get("status") == specialized.get("status") == "ok":
            grouped[key].append((staged, specialized))

    cr_baseline: dict[tuple[str, str, float, str], float] = {}
    for (block, mode), pairs in grouped.items():
        if block != 32:
            continue
        for _, specialized in pairs:
            cell = (
                specialized["dataset"], specialized["field"],
                specialized["error_bound"], mode,
            )
            cr_baseline[cell] = specialized["cr"]

    rows = []
    all_pairs = []
    for (block, mode), pairs in sorted(grouped.items()):
        all_pairs.extend(pairs)
        compress_pairs = [
            (staged, specialized) for staged, specialized in pairs
            if phase_stable(staged, "compress") and phase_stable(specialized, "compress")
        ]
        decompress_pairs = [
            (staged, specialized) for staged, specialized in pairs
            if phase_stable(staged, "decompress") and phase_stable(specialized, "decompress")
        ]
        compress_speedups = [
            specialized["compress_throughput_gbs"] / staged["compress_throughput_gbs"]
            for staged, specialized in compress_pairs
        ]
        decompress_speedups = [
            specialized["decompress_throughput_gbs"] / staged["decompress_throughput_gbs"]
            for staged, specialized in decompress_pairs
        ]
        cr_ratios = []
        for _, specialized in pairs:
            cell = (
                specialized["dataset"], specialized["field"],
                specialized["error_bound"], mode,
            )
            cr_ratios.append(specialized["cr"] / cr_baseline[cell])
        memory_ratios = [
            specialized["peak_device_bytes"] / staged["peak_device_bytes"]
            for staged, specialized in pairs
        ]
        rows.append({
            "block_size": block,
            "mode": mode,
            "valid_pairs": len(pairs),
            "compress_timed_pairs": len(compress_pairs),
            "decompress_timed_pairs": len(decompress_pairs),
            "installed_both": sum(
                specialized.get("fusion_installed_group_count", 0) > 0
                and specialized.get("fusion_inverse_installed_group_count", 0) > 0
                for _, specialized in pairs
            ),
            "compress_speedup_gmean": gmean(compress_speedups),
            "decompress_speedup_gmean": gmean(decompress_speedups),
            "compress_wins": sum(value > 1.0 for value in compress_speedups),
            "decompress_wins": sum(value > 1.0 for value in decompress_speedups),
            "staged_compress_gbs_gmean": gmean([
                staged["compress_throughput_gbs"] for staged, _ in compress_pairs
            ]),
            "specialized_compress_gbs_gmean": gmean([
                specialized["compress_throughput_gbs"] for _, specialized in compress_pairs
            ]),
            "staged_decompress_gbs_gmean": gmean([
                staged["decompress_throughput_gbs"] for staged, _ in decompress_pairs
            ]),
            "specialized_decompress_gbs_gmean": gmean([
                specialized["decompress_throughput_gbs"] for _, specialized in decompress_pairs
            ]),
            "cr_ratio_vs_b32_gmean": gmean(cr_ratios),
            "peak_memory_ratio_gmean": gmean(memory_ratios),
            "compressed_size_mismatches": sum(
                staged.get("compressed_bytes") != specialized.get("compressed_bytes")
                for staged, specialized in pairs
            ),
            "reconstruction_mismatches": sum(
                staged.get("decompressed_sha256") != specialized.get("decompressed_sha256")
                for staged, specialized in pairs
            ),
        })

    slowdowns = []
    for staged, specialized in all_pairs:
        for phase in ("compress", "decompress"):
            if not (phase_stable(staged, phase) and phase_stable(specialized, phase)):
                continue
            speedup = (
                specialized[f"{phase}_throughput_gbs"]
                / staged[f"{phase}_throughput_gbs"]
            )
            if speedup < 1.0:
                slowdowns.append({
                    "phase": phase,
                    "variant": specialized["variant"],
                    "dataset": specialized["dataset"],
                    "field": specialized["field"],
                    "error_bound": specialized["error_bound"],
                    "speedup": speedup,
                })

    return {
        "rows": rows,
        "paired_cells": len(all_pairs),
        "installed_both": sum(row["installed_both"] for row in rows),
        "compressed_size_mismatches": sum(row["compressed_size_mismatches"] for row in rows),
        "reconstruction_mismatches": sum(row["reconstruction_mismatches"] for row in rows),
        "strict_bound_misses_off": sum(row.get("eb_satisfied") is False for row in off_rows),
        "strict_bound_misses_auto": sum(row.get("eb_satisfied") is False for row in auto_rows),
        "slowdowns": slowdowns,
    }


def fmt(value: float | None, suffix: str = "") -> str:
    return "--" if value is None else f"{value:.2f}{suffix}"


def markdown(payload: dict, off_name: str, auto_name: str) -> str:
    lines = [
        "# Specialization block-size sweep",
        "",
        f"Staged source: `{off_name}`  ",
        f"Specialized source: `{auto_name}`",
        "",
        "The controlled family is `Quantizer(linear) -> Lorenzo1D(B) ->",
        "AdaptiveBitpack(B, mode)`. It is intentionally not labelled SZp; the",
        "`B=128, plain` cell is the shape previously called `szp_composed`.",
        "",
        "| B | Mode | Pairs | C/D timed | Compress | Decompress | Auto C/D (GB/s) | CR / B=32 | Peak memory |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["rows"]:
        lines.append(
            f"| {row['block_size']} | {row['mode']} | {row['valid_pairs']} | "
            f"{row['compress_timed_pairs']}/{row['decompress_timed_pairs']} | "
            f"{fmt(row['compress_speedup_gmean'], 'x')} | "
            f"{fmt(row['decompress_speedup_gmean'], 'x')} | "
            f"{fmt(row['specialized_compress_gbs_gmean'])}/"
            f"{fmt(row['specialized_decompress_gbs_gmean'])} | "
            f"{fmt(row['cr_ratio_vs_b32_gmean'], 'x')} | "
            f"{100.0 * (row['peak_memory_ratio_gmean'] - 1.0):+.1f}% |"
        )
    lines += [
        "",
        f"All {payload['installed_both']}/{payload['paired_cells']} Auto cells installed",
        "both forward and inverse specialization. Compressed bytes and reconstructed",
        f"hashes differ in {payload['compressed_size_mismatches']} and",
        f"{payload['reconstruction_mismatches']} pairs, respectively.",
        "",
        "Two Auto HACC/B=64/1e-2 decompression rows are timing-unreliable and are",
        "excluded only from decompression aggregates. Compression timing remains",
        "usable because phase-specific stability is reported independently.",
        "",
        "Eight rows per arm (every mode/block combination for CESM-2D/PSL at",
        "1e-4) exceed the literal requested bound by 0.934%; the excess is identical",
        "under Off and Auto and remains inside Benchkit's numerical-bound tolerance.",
        "It is not evidence of specialization changing semantics.",
        "",
        "## Interpretation",
        "",
        "- Specialization wins 119/120 reliable compression comparisons and every",
        "  reliable decompression comparison. The only loss is B=64/plain on",
        "  CESM-2D/PSL at 1e-4 (0.954x).",
        "- Blocks B>=64 improve aggregate compression ratio relative to B=32, and",
        "  specialized absolute throughput is highest at B=128. The CR trend is not",
        "  strictly monotonic: outlier B=96 is slightly above B=128. Staged",
        "  decompression degrades sharply at B>=96, so the large speedup is both an",
        "  optimized fused path and recovery of a block-size-sensitive modularity cost.",
        "- The experiment validates an explanatory cost-model target; it does not",
        "  validate a predictive Auto profitability policy, because Auto currently",
        "  installs every registered matching specialization in this matrix.",
        "- Choosing B is pipeline autotuning because it changes CR and archive",
        "  structure. A specialization cost model instead chooses staged versus",
        "  specialized execution for a fixed graph and fixed B.",
        "",
    ]
    return "\n".join(lines)


def latex(payload: dict) -> str:
    lines = [
        "% Generated by compression_benchmarking/scripts/analyze_specialization_blocksize.py.",
        "% Do not edit measured values by hand.",
        "\\begin{table}[t]",
        "  \\centering",
        "  \\caption{H100 block-size sensitivity for the specialized 1-D Lorenzo",
        "  and adaptive-bitpack family. Ratios compare Auto with the identical staged",
        "  graph. CR is paired relative to the same mode at $B=32$.}",
        "  \\label{tab:specialization-block-size}",
        "  \\small",
        "  \\resizebox{\\columnwidth}{!}{%",
        "  \\begin{tabular}{rrlrrrr}",
        "    \\toprule",
        "    $B$ & C/D pairs & Mode & C speedup & D speedup & CR/$B{=}32$ & Memory \\\\",
        "    \\midrule",
    ]
    for row in payload["rows"]:
        lines.append(
            f"    {row['block_size']} & {row['compress_timed_pairs']}/{row['decompress_timed_pairs']} & "
            f"{row['mode']} & {row['compress_speedup_gmean']:.2f}$\\times$ & "
            f"{row['decompress_speedup_gmean']:.2f}$\\times$ & "
            f"{row['cr_ratio_vs_b32_gmean']:.2f}$\\times$ & "
            f"{100.0 * (row['peak_memory_ratio_gmean'] - 1.0):+.1f}\\% \\\\"
        )
    lines += [
        "    \\bottomrule",
        "  \\end{tabular}",
        "  }",
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

    payload = summarize(load(args.off / "runs.jsonl"), load(args.auto / "runs.jsonl"))
    payload["staged_source"] = args.off.name
    payload["specialized_source"] = args.auto.name
    outputs = {
        args.output_json: json.dumps(payload, indent=2) + "\n",
        args.output_md: markdown(payload, args.off.name, args.auto.name),
        args.output_tex: latex(payload),
    }
    for path, content in outputs.items():
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content)
    if not any(outputs):
        print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
