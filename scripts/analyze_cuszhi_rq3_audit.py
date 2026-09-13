#!/usr/bin/env python3
"""Summarize the bounded cuSZ-Hi tuning probe and its effect on RQ3/X2."""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import geometric_mean, median
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchkit import validity  # noqa: E402


TUNED_TO_DEFAULT = {
    "cuszhi_tp_rd": "cuszhi_tp",
    "cuszhi_tp_r64": "cuszhi_tp",
    "cuszhi_cr_rd": "cuszhi_cr",
    "cuszhi_cr_r64": "cuszhi_cr",
}


def load_session(path: Path) -> list[dict]:
    with (path / "runs.jsonl").open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def finite_positive(value: object) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(value) and value > 0


def usable(row: dict) -> bool:
    return (
        row.get("status") == "ok"
        and validity.is_valid(row)
        and validity.quality_valid(row)
        and row.get("compress_stable", row.get("timing_reliable", True)) is not False
        and finite_positive(row.get("compress_throughput_gbs"))
        and finite_positive(row.get("cr"))
        and isinstance(row.get("psnr"), (int, float))
        and math.isfinite(row["psnr"])
    )


def cell(row: dict) -> tuple:
    return row.get("dataset"), row.get("field"), float(row.get("error_bound"))


def dominates(left: dict, right: dict, left_speed_multiplier: float = 1.0) -> bool:
    return (
        left["compress_throughput_gbs"] * left_speed_multiplier
        >= right["compress_throughput_gbs"]
        and left["cr"] >= right["cr"]
        and left["psnr"] >= right["psnr"] - 0.05
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--default", type=Path, action="append", required=True)
    parser.add_argument("--tuned", type=Path, required=True)
    parser.add_argument("--composed", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    args = parser.parse_args()

    default_raw = [row for path in args.default for row in load_session(path)]
    tuned_raw = load_session(args.tuned)
    composed_raw = [
        row
        for row in load_session(args.composed)
        if row.get("variant") == "x_lq_pfpl_ans"
    ]
    annotated = validity.annotate(default_raw + tuned_raw + composed_raw)
    n_default, n_tuned = len(default_raw), len(tuned_raw)
    defaults = [row for row in annotated[:n_default] if usable(row)]
    tuned = [row for row in annotated[n_default : n_default + n_tuned] if usable(row)]
    composed = [row for row in annotated[n_default + n_tuned :] if usable(row)]

    default_index = {(row["variant"], *cell(row)): row for row in defaults}
    variant_rows: dict[str, list[tuple[dict, dict]]] = defaultdict(list)
    for row in tuned:
        default_variant = TUNED_TO_DEFAULT.get(row.get("variant"))
        baseline = default_index.get((default_variant, *cell(row)))
        if baseline:
            variant_rows[row["variant"]].append((row, baseline))

    summaries = []
    for variant in sorted(TUNED_TO_DEFAULT):
        pairs = variant_rows[variant]
        summaries.append(
            {
                "variant": variant,
                "usable": sum(row.get("variant") == variant for row in tuned),
                "paired": len(pairs),
                "throughput_geomean_ratio": geometric_mean(
                    row["compress_throughput_gbs"] / baseline["compress_throughput_gbs"]
                    for row, baseline in pairs
                ) if pairs else None,
                "cr_geomean_ratio": geometric_mean(
                    row["cr"] / baseline["cr"] for row, baseline in pairs
                ) if pairs else None,
                "psnr_median_delta_db": median(
                    row["psnr"] - baseline["psnr"] for row, baseline in pairs
                ) if pairs else None,
                "tuned_dominates_default": sum(
                    dominates(row, baseline) for row, baseline in pairs
                ),
                "tuned_robustly_dominates_default": sum(
                    dominates(row, baseline) and
                    row["compress_throughput_gbs"] >=
                    1.05 * baseline["compress_throughput_gbs"]
                    for row, baseline in pairs
                ),
                "default_dominates_tuned": sum(
                    dominates(baseline, row) for row, baseline in pairs
                ),
            }
        )

    defaults_by_field: dict[tuple, list[dict]] = defaultdict(list)
    tuned_by_field: dict[tuple, list[dict]] = defaultdict(list)
    for row in defaults:
        defaults_by_field[(row.get("dataset"), row.get("field"))].append(row)
    for row in tuned:
        tuned_by_field[(row.get("dataset"), row.get("field"))].append(row)

    impact = []
    for row in composed:
        field_key = row.get("dataset"), row.get("field")
        if field_key not in tuned_by_field:
            continue
        default_dominated = any(
            dominates(native, row, 1.05) for native in defaults_by_field[field_key]
        )
        tuned_dominated = any(
            dominates(native, row, 1.05) for native in tuned_by_field[field_key]
        )
        impact.append(
            {
                "dataset": row["dataset"],
                "field": row["field"],
                "bound": float(row["error_bound"]),
                "default_dominated": default_dominated,
                "tuned_dominated": tuned_dominated,
                "newly_dominated_by_tuning": tuned_dominated and not default_dominated,
            }
        )

    payload = {
        "schema": "fzgm-rq3-cuszhi-audit-v1",
        "default_sources": [path.name for path in args.default],
        "tuned_source": args.tuned.name,
        "summaries": summaries,
        "x2_probe_rows": len(impact),
        "x2_default_dominated": sum(row["default_dominated"] for row in impact),
        "x2_tuned_dominated": sum(row["tuned_dominated"] for row in impact),
        "x2_newly_dominated_by_tuning": sum(
            row["newly_dominated_by_tuning"] for row in impact
        ),
        "impact": impact,
    }
    args.output_json.write_text(json.dumps(payload, indent=2) + "\n")

    lines = [
        "# RQ3 cuSZ-Hi coverage and tuning audit",
        "",
        "The source audit found that the ordinary native defaults already use CR-first",
        "interpolation autotuning, the maximum uint8 spline radius (128), and",
        "GPU/length-aware Huffman chunk autotuning. The bounded probe therefore tests",
        "only RD-first interpolation and the smaller legal radius 64.",
        "",
        "| Variant | Usable | Paired | Throughput ratio | CR ratio | Median PSNR delta | Tuned dominates | Robust tuned dominates | Default dominates |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summaries:
        def fmt(value: float | None, suffix: str = "") -> str:
            return "n/a" if value is None else f"{value:.3f}{suffix}"
        lines.append(
            f"| {row['variant']} | {row['usable']} | {row['paired']} | "
            f"{fmt(row['throughput_geomean_ratio'], 'x')} | "
            f"{fmt(row['cr_geomean_ratio'], 'x')} | "
            f"{fmt(row['psnr_median_delta_db'], ' dB')} | "
            f"{row['tuned_dominates_default']} | "
            f"{row['tuned_robustly_dominates_default']} | "
            f"{row['default_dominates_tuned']} |"
        )
    lines += [
        "",
        "## Effect on X2 in the probe",
        "",
        f"- Usable X2 probe rows: {payload['x2_probe_rows']}",
        f"- Dominated by default cuSZ-Hi: {payload['x2_default_dominated']}",
        f"- Dominated by a probed tuned cuSZ-Hi mode: {payload['x2_tuned_dominated']}",
        f"- Newly dominated only after tuning: {payload['x2_newly_dominated_by_tuning']}",
        "",
    ]
    args.output_md.write_text("\n".join(lines))


if __name__ == "__main__":
    main()
