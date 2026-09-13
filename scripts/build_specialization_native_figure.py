#!/usr/bin/env python3
"""Generate the H100 staged/Auto performance artifact normalized to native.

Each plotted coordinate must have valid, reliable measurements for native and
FZGM in both the Off and Auto sessions.  Both FZGM arms use the geometric mean
of the two native measurements as a common reference, so differences between
the bars cannot be caused by choosing a different native denominator.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchkit import validity  # noqa: E402
from benchkit.schema import load_result_file  # noqa: E402


FAMILIES = (
    ("cusz", "cuSZ", ("cusz", "cusz"), ("fzgm", "cusz")),
    ("cuszp2_plain", "cuSZp2\nplain", ("cuszp2", "cuszp2_plain"),
     ("fzgm", "cuszp2_plain_sp")),
    ("cuszp2_outlier", "cuSZp2\noutlier", ("cuszp2", "cuszp2_outlier"),
     ("fzgm", "cuszp2_outlier_sp")),
    ("cuszp3_fixed", "cuSZp3\nfixed", ("cuszp3", "cuszp3_fixed"),
     ("fzgm", "cuszp3_fixed")),
    ("cuszp3_plain", "cuSZp3\nplain", ("cuszp3", "cuszp3_plain"),
     ("fzgm", "cuszp3_plain_sp")),
    ("cuszp3_outlier", "cuSZp3\noutlier", ("cuszp3", "cuszp3_outlier"),
     ("fzgm", "cuszp3_outlier_sp")),
    ("fzgpu", "FZ-GPU", ("fzgpu", "fzgpu"), ("fzgm", "fzgpu")),
    ("pfpl", "PFPL", ("pfpl", "pfpl"), ("fzgm", "pfpl")),
    ("fsz", "FSZ", ("fsz", "fsz"), ("fzgm", "fsz")),
)

PHASES = (
    ("compression", "compress_throughput_gbs"),
    ("decompression", "decompress_throughput_gbs"),
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def coordinate(row: dict[str, Any]) -> tuple[Any, ...]:
    resolved = row.get("execution_context", {}).get("resolved_config", {})
    return (
        row.get("dataset"), row.get("field"), row.get("dtype", resolved.get("dtype")),
        tuple(row.get("dims", resolved.get("dims", []))),
        row.get("dim_order", resolved.get("dim_order")), row.get("error_mode"),
        row.get("error_bound"),
    )


def index(rows: list[dict[str, Any]], selector: tuple[str, str]) -> dict[tuple, dict]:
    selected = [r for r in rows if (r.get("compressor"), r.get("variant")) == selector]
    out = {coordinate(row): row for row in selected}
    if len(out) != len(selected):
        raise ValueError(f"duplicate coordinates for {selector}")
    return out


def gmean(values: list[float]) -> float | None:
    return math.exp(sum(math.log(value) for value in values) / len(values)) if values else None


def hierarchical_gmean(rows: list[dict[str, Any]], metric: str) -> float | None:
    """Give equal weight to bounds within fields, fields within datasets, and datasets.

    A raw coordinate mean lets datasets with many fields dominate the result.  Keep
    every valid coordinate, but reduce it in three explicit levels instead:

      1. geometric mean across error bounds for each field;
      2. geometric mean across fields for each dataset;
      3. geometric mean across datasets.

    dtype, shape, dimension order, and error mode are included in the field key so
    this remains well-defined if a future corpus reuses a field name across layouts.
    """
    by_dataset: dict[str, dict[tuple[Any, ...], list[float]]] = {}
    for row in rows:
        field_key = (
            row["field"], row["dtype"], tuple(row["dims"]), row["dim_order"],
            row["error_mode"],
        )
        by_dataset.setdefault(row["dataset"], {}).setdefault(field_key, []).append(
            float(row[metric])
        )
    dataset_means = []
    for fields in by_dataset.values():
        field_means = [gmean(values) for values in fields.values()]
        dataset_means.append(gmean([value for value in field_means if value is not None]))
    return gmean([value for value in dataset_means if value is not None])


def quantile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def identity_clean(rows: list[dict[str, Any]]) -> bool:
    hashes = {row.get("dataset_sha256") for row in rows}
    if None in hashes or len(hashes) != 1:
        return False
    effective = [row.get("eb_abs_effective") for row in rows]
    if any(value is None for value in effective):
        return False
    return all(math.isclose(float(effective[0]), float(value), rel_tol=1e-12, abs_tol=0.0)
               for value in effective[1:])


def specialization_installed(row: dict[str, Any], phase: str) -> bool:
    field = (
        "fusion_installed_group_count"
        if phase == "compression"
        else "fusion_inverse_installed_group_count"
    )
    return int(row.get(field, 0) or 0) > 0


def build(off_rows: list[dict[str, Any]], auto_rows: list[dict[str, Any]]):
    detail: list[dict[str, Any]] = []
    summary: list[dict[str, Any]] = []
    for family, label, native_selector, fzgm_selector in FAMILIES:
        native_off = index(off_rows, native_selector)
        native_auto = index(auto_rows, native_selector)
        fzgm_off = index(off_rows, fzgm_selector)
        fzgm_auto = index(auto_rows, fzgm_selector)
        common = set(native_off) & set(native_auto) & set(fzgm_off) & set(fzgm_auto)
        for phase, metric in PHASES:
            phase_rows = []
            for key in sorted(common, key=repr):
                rows = [native_off[key], native_auto[key], fzgm_off[key], fzgm_auto[key]]
                if not all(row.get("status") == "ok" and validity.is_valid(row)
                           for row in rows):
                    continue
                if not identity_clean(rows):
                    raise ValueError(f"identity mismatch for {family}: {key}")
                if not all(row.get("timing_reliable") is not False and
                           isinstance(row.get(metric), (int, float)) and row[metric] > 0
                           for row in rows):
                    continue
                baseline = math.sqrt(native_off[key][metric] * native_auto[key][metric])
                record = {
                    "family": family, "phase": phase, "dataset": key[0], "field": key[1],
                    "dtype": key[2], "dims": list(key[3]), "dim_order": key[4],
                    "error_mode": key[5], "error_bound": key[6],
                    "staged_over_native": fzgm_off[key][metric] / baseline,
                    "auto_over_native": fzgm_auto[key][metric] / baseline,
                    "auto_over_staged": fzgm_auto[key][metric] / fzgm_off[key][metric],
                    "native_auto_over_off": native_auto[key][metric] / native_off[key][metric],
                    "auto_specialization_installed": specialization_installed(
                        fzgm_auto[key], phase
                    ),
                }
                detail.append(record)
                phase_rows.append(record)
            staged = [row["staged_over_native"] for row in phase_rows]
            auto = [row["auto_over_native"] for row in phase_rows]
            staged_gmean = hierarchical_gmean(phase_rows, "staged_over_native")
            auto_gmean = hierarchical_gmean(phase_rows, "auto_over_native")
            recovered = None
            if staged_gmean is not None and auto_gmean is not None and staged_gmean < 1.0:
                recovered = (auto_gmean - staged_gmean) / (1.0 - staged_gmean)
            summary.append({
                "family": family, "label": label.replace("\n", " "), "phase": phase,
                "pairs": len(phase_rows),
                "datasets": len({row["dataset"] for row in phase_rows}),
                "fields": len({(row["dataset"], row["field"], row["dtype"],
                               tuple(row["dims"]), row["dim_order"], row["error_mode"])
                              for row in phase_rows}),
                "staged_over_native_gmean": staged_gmean,
                "auto_over_native_gmean": auto_gmean,
                "auto_over_staged_gmean": hierarchical_gmean(
                    phase_rows, "auto_over_staged"),
                "native_auto_over_off_gmean": hierarchical_gmean(
                    phase_rows, "native_auto_over_off"),
                "auto_specialization_installed_fraction": (
                    sum(row["auto_specialization_installed"] for row in phase_rows)
                    / len(phase_rows)
                    if phase_rows else None
                ),
                "staged_q25": quantile(staged, 0.25), "staged_median": quantile(staged, 0.5),
                "staged_q75": quantile(staged, 0.75), "auto_q25": quantile(auto, 0.25),
                "auto_median": quantile(auto, 0.5), "auto_q75": quantile(auto, 0.75),
                "auto_faster_than_staged_fraction": (
                    sum(row["auto_over_staged"] > 1.0 for row in phase_rows) / len(phase_rows)
                    if phase_rows else None),
                "auto_at_or_above_native_fraction": (
                    sum(row["auto_over_native"] >= 1.0 for row in phase_rows) / len(phase_rows)
                    if phase_rows else None),
                "aggregate_native_gap_recovered": recovered,
            })
    return summary, detail


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value) if isinstance(value, list) else value
                             for key, value in row.items()})


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# H100 specialization performance normalized to native", "",
        "Each row uses complete-case coordinates with valid, reliable native, staged,",
        "and Auto measurements in both sessions. The common native baseline is the",
        "geometric mean of the two native measurements. Reported geometric means",
        "give equal weight to bounds within each field, fields within each dataset,",
        "and datasets. Fractions and distribution quantiles remain coordinate-level.", "",
        "| Family | Phase | Datasets | Fields | Coordinates | Staged/native | Auto/native | Auto/staged | Auto wins | Auto >= native | Gap recovered |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["summary"]:
        pct = lambda value: "--" if value is None else f"{100 * value:.1f}%"
        ratio = lambda value: "--" if value is None else f"{value:.3f}x"
        lines.append(
            f"| {row['label']} | {row['phase']} | {row['datasets']} | "
            f"{row['fields']} | {row['pairs']} | "
            f"{ratio(row['staged_over_native_gmean'])} | "
            f"{ratio(row['auto_over_native_gmean'])} | "
            f"{ratio(row['auto_over_staged_gmean'])} | "
            f"{pct(row['auto_faster_than_staged_fraction'])} | "
            f"{pct(row['auto_at_or_above_native_fraction'])} | "
            f"{pct(row['aggregate_native_gap_recovered'])} |"
        )
    lines += ["", "`Gap recovered` is reported only when staged aggregate throughput is",
              "below native. Negative values mean Auto widened the aggregate native gap.", ""]
    path.write_text("\n".join(lines))


def plot(path: Path, detail: list[dict[str, Any]]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import FixedFormatter, FixedLocator, LogLocator, NullFormatter

    colors = {"staged": "#8b8b8b", "auto": "#2474b7"}
    specialized_ids = {
        row["family"] for row in detail if row["auto_specialization_installed"]
    }
    family_groups = (
        ("Specialization installed", [entry for entry in FAMILIES
                                       if entry[0] in specialized_ids]),
        ("No specialization", [entry for entry in FAMILIES
                               if entry[0] not in specialized_ids]),
    )
    families = family_groups[0][1] + family_groups[1][1]
    split_after = len(family_groups[0][1])
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 3.8), sharey=True)
    for axis, (phase, _) in zip(axes, PHASES):
        staged_data, auto_data, show_auto = [], [], []
        for family, *_ in families:
            selected = [
                row for row in detail
                if row["family"] == family and row["phase"] == phase
            ]
            staged_data.append([row["staged_over_native"] for row in selected])
            auto_data.append([row["auto_over_native"] for row in selected])
            show_auto.append(any(row["auto_specialization_installed"] for row in selected))
        centers = list(range(1, len(families) + 1))
        for name, all_data in (("staged", staged_data), ("auto", auto_data)):
            included = [
                index for index in range(len(families))
                if name == "staged" or show_auto[index]
            ]
            data = [all_data[index] for index in included]
            positions = [
                centers[index] + (
                    0.18 if name == "auto" else (-0.18 if show_auto[index] else 0.0)
                )
                for index in included
            ]
            bp = axis.boxplot(data, positions=positions, widths=0.31, whis=(10, 90),
                              showfliers=False, patch_artist=True, manage_ticks=False)
            for box in bp["boxes"]:
                box.set(facecolor=colors[name], edgecolor=colors[name], alpha=0.72)
            for key in ("whiskers", "caps", "medians"):
                for artist in bp[key]:
                    artist.set(color="#333333", linewidth=0.8)
            means = [hierarchical_gmean(
                [row for row in detail
                 if row["family"] == families[index][0] and row["phase"] == phase],
                f"{name}_over_native",
            ) for index in included]
            axis.scatter(positions, means, marker="D", s=18,
                         facecolor="white", edgecolor="#111111", linewidth=0.7, zorder=4)
        axis.axhline(1.0, color="#111111", linewidth=1.0, linestyle="--")
        axis.axvline(split_after + 0.5, color="#aaaaaa", linewidth=0.9,
                     linestyle=(0, (3, 3)))
        axis.text((split_after + 1) / 2, 7.8, "specialization installed",
                  ha="center", va="center", fontsize=7.5, color="#666666")
        axis.text((split_after + 1 + len(families)) / 2, 7.8, "no specialization",
                  ha="center", va="center", fontsize=7.5, color="#666666")
        axis.set_yscale("log")
        axis.set_ylim(0.1, 10.0)
        major_ticks = (0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0)
        axis.yaxis.set_major_locator(FixedLocator(major_ticks))
        axis.yaxis.set_major_formatter(
            FixedFormatter(("0.1", "0.2", "0.5", "1", "2", "5", "10"))
        )
        axis.yaxis.set_minor_locator(LogLocator(base=10, subs=(3, 4, 6, 7, 8, 9)))
        axis.yaxis.set_minor_formatter(NullFormatter())
        axis.set_xticks(centers, [entry[1] for entry in families], fontsize=7.3)
        axis.grid(axis="y", which="major", color="#d0d0d0", linewidth=0.65)
        axis.grid(axis="y", which="minor", color="#e8e8e8", linewidth=0.4)
        axis.set_title(phase.capitalize(), fontsize=10)
        axis.tick_params(axis="y", labelsize=8)
    axes[0].set_ylabel("Throughput / native throughput", fontsize=9)
    legend = [Line2D([0], [0], color=colors["staged"], lw=7, alpha=0.72, label="Staged"),
              Line2D([0], [0], color=colors["auto"], lw=7, alpha=0.72, label="Auto"),
              Line2D([0], [0], marker="D", color="none", markerfacecolor="white",
                     markeredgecolor="#111111", markersize=5,
                     label="Dataset-balanced geometric mean"),
              Line2D([0], [0], color="#111111", lw=1, linestyle="--", label="Native")]
    fig.legend(handles=legend, loc="lower center", bbox_to_anchor=(0.5, -0.005), ncol=4,
               frameon=False, fontsize=8)
    fig.tight_layout(rect=(0, 0.115, 1, 1), w_pad=1.5)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    if path.suffix.lower() == ".pdf":
        fig.savefig(path.with_suffix(".png"), dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--off", required=True, type=Path)
    parser.add_argument("--auto", required=True, type=Path)
    parser.add_argument("--policy", required=True, type=Path)
    parser.add_argument("--output-prefix", required=True, type=Path)
    parser.add_argument("--figure", required=True, type=Path)
    args = parser.parse_args()

    off_runs, auto_runs = args.off / "runs.jsonl", args.auto / "runs.jsonl"
    off_rows = validity.annotate(load_result_file(off_runs))
    auto_rows = validity.annotate(load_result_file(auto_runs))
    summary, detail = build(off_rows, auto_rows)
    payload = {
        "schema_version": 2,
        "method": (
            "four-way complete-case; common native baseline is geometric mean of Off "
            "and Auto native throughput; summary means aggregate bounds within fields, "
            "fields within datasets, then datasets with equal weight"
        ),
        "sources": {
            "off_session": args.off.name, "off_runs_sha256": sha256_file(off_runs),
            "auto_session": args.auto.name, "auto_runs_sha256": sha256_file(auto_runs),
            "policy_path": str(args.policy.resolve()), "policy_sha256": sha256_file(args.policy),
            "generator_sha256": sha256_file(Path(__file__)),
        },
        "summary": summary,
    }
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    args.output_prefix.with_suffix(".json").write_text(json.dumps(payload, indent=2) + "\n")
    write_csv(args.output_prefix.with_suffix(".csv"), detail)
    write_markdown(args.output_prefix.with_suffix(".md"), payload)
    plot(args.figure, detail)


if __name__ == "__main__":
    main()
