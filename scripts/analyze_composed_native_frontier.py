#!/usr/bin/env python3
"""Build the paired native-only frontier evidence for FZGM compositions.

For each composed row, compare its measured (compression throughput, CR, PSNR)
point with every valid, compression-stable native point for the same field across
all sampled error bounds.  This is deliberately stricter than comparing dataset
medians at the same requested bound.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchkit import validity  # noqa: E402
from benchkit.archive_size import (  # noqa: E402
    DEFAULT_CONFIG as DEFAULT_ARCHIVE_SIZE_CONFIG,
    correct_fzgm_archive_sizes,
)


NATIVE_VARIANTS = (
    ("cusz", "cusz"),
    ("cuszhi", "cuszhi_cr"),
    ("cuszhi", "cuszhi_tp"),
    ("cuszp2", "cuszp2_outlier"),
    ("cuszp2", "cuszp2_plain"),
    ("cuszp3", "cuszp3_outlier"),
    ("cuszp3", "cuszp3_plain"),
    ("cuszp3", "cuszp3_fixed"),
    ("pfpl", "pfpl"),
)

LABELS = {
    "x_lq_pfpl": "LQ--PFPL",
    "x_lq_pfpl_ans": "LQ--PFPL--ANS",
    "x_lq_lc_ans": "LQ--LC--ANS",
    "x_tl_pfpl": "TiledLorenzo--PFPL",
    "x_tl_pfpl_ans": "TiledLorenzo--PFPL--ANS",
    "x_gi_huff_ans": "GInterp--Huffman--ANS",
    "x_gi_huff_zstd": "GInterp--GPU-Zstd",
}

EPS = 1e-12
PSNR_TOL_DB = 0.05
THROUGHPUT_UNCERTAINTY = 0.05


def load_rows(path: Path) -> list[dict]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def apply_archive_size_correction(rows: list[dict], config_path: Path | None) -> list[dict]:
    if config_path is None:
        return rows
    return correct_fzgm_archive_sizes(rows, config_path=config_path)


def archive_size_metadata(config_path: Path | None) -> dict | None:
    if config_path is None:
        return None
    resolved = config_path.resolve()
    repo_root = Path(__file__).resolve().parent.parent
    try:
        display_path = resolved.relative_to(repo_root).as_posix()
    except ValueError:
        display_path = resolved.name
    return {
        "config_path": display_path,
        "config_sha256": hashlib.sha256(resolved.read_bytes()).hexdigest(),
    }


def select_composed_rows(rows: list[dict], variant: str | None = None) -> list[dict]:
    """Select composition candidates from a composed-only or paired session."""
    selected = [
        row for row in rows
        if row.get("compressor") == "fzgm" and row.get("variant") in LABELS
    ]
    if variant:
        selected = [row for row in selected if row.get("variant") == variant]
    return selected


def finite_positive(value: object) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(value) and value > 0


def compression_usable(row: dict) -> bool:
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


def point(row: dict) -> dict:
    return {
        "compress_gbs": float(row["compress_throughput_gbs"]),
        "cr": float(row["cr"]),
        "psnr_db": float(row["psnr"]),
    }


def dominates(native: dict, candidate: dict, native_speed_multiplier: float = 1.0) -> bool:
    """True when native is no worse on all three axes.

    The PSNR tolerance avoids treating sub-rounding differences as a quality
    distinction.  The robust frontier gives native throughput a 5% favorable
    adjustment, reflecting the measured independent-invocation envelope.
    """
    n, c = point(native), point(candidate)
    axes = (
        n["compress_gbs"] * native_speed_multiplier >= c["compress_gbs"] - EPS,
        n["cr"] >= c["cr"] - EPS,
        n["psnr_db"] >= c["psnr_db"] - PSNR_TOL_DB,
    )
    # Equality does not create a new frontier point, so a native point that is
    # no worse on every axis is sufficient; a strict improvement is not required.
    return all(axes)


def key(row: dict) -> tuple:
    return (row.get("dataset"), row.get("field"), float(row.get("error_bound")))


def data_signature(row: dict, include_hash: bool = False) -> tuple:
    signature = (
        tuple(row.get("dims") or ()),
        row.get("dtype"),
        row.get("original_bytes"),
    )
    if include_hash:
        signature += (row.get("dataset_sha256"),)
    return signature


def summarize(records: list[dict]) -> list[dict]:
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for record in records:
        grouped[(record["dataset"], record["bound"], record["variant"])].append(record)

    out = []
    for (dataset, bound, variant), rows in sorted(grouped.items()):
        out.append(
            {
                "dataset": dataset,
                "bound": bound,
                "variant": variant,
                "label": LABELS.get(variant, variant),
                "candidate_rows": len(rows),
                "available_frontier": sum(r["on_available_frontier"] for r in rows),
                "robust_available_frontier": sum(
                    r["on_robust_available_frontier"] for r in rows
                ),
                "complete_native_attempts": sum(
                    r["complete_native_attempts"] for r in rows
                ),
                "attempt_complete_frontier": sum(
                    r["complete_native_attempts"] and r["on_available_frontier"]
                    for r in rows
                ),
                "robust_attempt_complete_frontier": sum(
                    r["complete_native_attempts"]
                    and r["on_robust_available_frontier"]
                    for r in rows
                ),
                "complete_usable_native_curves": sum(
                    r["complete_usable_native_curves"] for r in rows
                ),
                "robust_all_usable_frontier": sum(
                    r["complete_usable_native_curves"]
                    and r["on_robust_available_frontier"]
                    for r in rows
                ),
            }
        )
    return out


def render_markdown(payload: dict) -> str:
    native_count = len(payload["definition"]["native_variants"])
    native_points_per_field = native_count * len(payload["definition"]["native_bounds"])
    compatibility = payload["source_compatibility"]
    lines = [
        "# RQ3 paired native-frontier analysis",
        "",
        f"Composed source: `{payload['composed_source']}`",
        "Native sources: " + ", ".join(f"`{s}`" for s in payload["native_sources"]),
        "",
        "A composed row adds a sampled point when no valid, compression-stable native",
        "measurement for the same field at any tested bound is at least as good in",
        "compression throughput, compression ratio, and PSNR. The robust result gives",
        "every native throughput a 5% favorable adjustment. `Attempt complete` requires",
        f"all {native_count} intended native variants to have been run at all three bounds. Native",
        "failures and invalid reconstructions count as completed attempts but cannot",
        "dominate a valid composition. `All usable` is the stricter sensitivity subset",
        f"where all {native_points_per_field} native points are valid and compression-stable.",
    ]
    correction = payload.get("archive_size_correction")
    if correction:
        lines += [
            "",
            "Archive-inclusive FZGM sizes were derived in memory using",
            f"`{correction['config_path']}` (SHA-256 `{correction['config_sha256']}`).",
        ]
    if compatibility["same_session"]:
        lines += [
            "Candidate and native rows come from the same publication session. Dataset",
            "hashes and complete execution/build provenance are therefore paired directly.",
        ]
    else:
        lines += [
            "Candidate and native rows come from separate sessions. Dataset/field name,",
            "dimensions, dtype, original byte count, and available dataset hashes are",
            "checked before rows enter the paired comparison.",
        ]
    lines += [
        "",
        "## Coverage audit",
        "",
        "| Dataset | Fields | Attempt complete | All native curves usable |",
        "|---|---:|---:|---:|",
    ]
    for row in payload["coverage"]:
        lines.append(
            f"| {row['dataset']} | {row['fields']} | "
            f"{row['attempt_complete_fields']}/{row['fields']} | "
            f"{row['all_usable_fields']}/{row['fields']} |"
        )

    lines += [
        "",
        "## Candidate summary",
        "",
        "`Frontier` is exact sampled dominance; `robust` includes the 5% unfavorable",
        "throughput sensitivity test.",
        "",
        "| Dataset | Bound | Composition | Valid | Frontier | Robust | Attempts complete | Robust, attempts complete | All usable | Robust, all usable |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["summary"]:
        lines.append(
            f"| {row['dataset']} | {row['bound']:.0e} | {row['label']} | "
            f"{row['candidate_rows']} | {row['available_frontier']} | "
            f"{row['robust_available_frontier']} | {row['complete_native_attempts']} | "
            f"{row['robust_attempt_complete_frontier']} | "
            f"{row['complete_usable_native_curves']} | "
            f"{row['robust_all_usable_frontier']} |"
        )

    lines += [
        "",
        "## Native attempt and usability audit",
        "",
        "`Missing attempts` means no row exists. `Unusable` means the run exists but",
        "failed, violated the validity gate, lacks finite quality, or has unstable",
        "compression timing. The latter is scientific/measurement disposition, not",
        "missing experimental coverage.",
        "",
        "| Dataset | Native variant | Missing attempts | Unusable rows |",
        "|---|---|---:|---:|",
    ]
    for row in payload["missing_native_rows"]:
        if row["missing_attempts"] or row["unusable"]:
            lines.append(
                f"| {row['dataset']} | {row['compressor']}:{row['variant']} | "
                f"{row['missing_attempts']} | {row['unusable']} |"
            )

    lines += [
        "",
        "## Interpretation boundary",
        "",
        "This artifact supports `adds a measured point to the sampled native frontier`,",
        "not `wins universally` or `beats every native compressor`. Failed or invalid",
        "native rows are reported outcomes, but systematic native failures should be",
        "described rather than converted into a rhetorical win. Dataset medians remain",
        "descriptive and are not substituted for paired field-level outcomes.",
        "",
    ]
    return "\n".join(lines)


def render_tex(payload: dict, selected_variant: str) -> str:
    selected = [row for row in payload["summary"] if row["variant"] == selected_variant]
    native_count = len(payload["definition"]["native_variants"])
    lines = [
        "% Generated by scripts/analyze_composed_native_frontier.py. Do not edit measured values.",
        "\\begin{table}[t]",
        "  \\centering",
        "  \\caption{Paired field-level results for the selected cross-family composition.",
        "  A frontier point is not dominated in compression throughput, CR, and PSNR by",
        "  any valid native measurement for that field across the sampled bounds. Robust",
        "  counts give native throughput a 5\\% favorable adjustment. Attempt-complete",
        f"  counts require all {native_count} native variants to have been run at all three bounds.}}",
        "  \\label{tab:composed-native-frontier}",
        "  \\small",
        "  \\resizebox{\\columnwidth}{!}{%",
        "  \\begin{tabular}{lrrrr}",
        "    \\toprule",
        "    Dataset / bound & Valid & Frontier & Robust & Attempt-complete robust \\\\",
        "    \\midrule",
    ]
    correction = payload.get("archive_size_correction")
    if correction:
        lines.insert(1, f"% Archive-size correction config SHA-256: {correction['config_sha256']}")
    for row in selected:
        lines.append(
            f"    {row['dataset']} / $10^{{{int(round(math.log10(row['bound'])))}}}$ & "
            f"{row['candidate_rows']} & "
            f"{row['available_frontier']} & {row['robust_available_frontier']} & "
            f"{row['robust_attempt_complete_frontier']} \\\\"
        )
    lines += ["    \\bottomrule", "  \\end{tabular}", "  }", "\\end{table}", ""]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--composed", type=Path, required=True)
    parser.add_argument(
        "--native",
        type=Path,
        action="append",
        required=True,
        help="native session, repeat in oldest-to-newest precedence order",
    )
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    parser.add_argument("--output-tex", type=Path, required=True)
    parser.add_argument("--selected-variant", default="x_lq_pfpl_ans")
    parser.add_argument(
        "--fzgm-archive-size-config",
        type=Path,
        help=("apply the audited historical FZGM archive-size correction using this "
              f"config (default mapping: {DEFAULT_ARCHIVE_SIZE_CONFIG})"),
    )
    parser.add_argument(
        "--composed-variant",
        help=("restrict the composed source to one variant; use this when a paired "
              "session contains both composition and native rows"),
    )
    args = parser.parse_args()

    composed_source_rows = load_rows(args.composed / "runs.jsonl")
    # Publication-paired sessions contain the candidate and native arms together.
    # Restrict this side to known FZGM compositions so native rows cannot become
    # candidate points when the same session is supplied to both inputs.
    composed_raw = select_composed_rows(composed_source_rows, args.composed_variant)
    composed_raw = apply_archive_size_correction(
        composed_raw, args.fzgm_archive_size_config
    )
    if not composed_raw:
        raise ValueError("composed source contains no matching FZGM composition rows")
    native_all = [row for path in args.native for row in load_rows(path / "runs.jsonl")]
    # Later sources supersede earlier measurements of the same native cell. This
    # preserves the historical multi-session workflow while also allowing one paired
    # publication session to be supplied on both sides.
    native_index = {}
    for row in native_all:
        native_index[
            (
                row.get("compressor"),
                row.get("variant"),
                row.get("dataset"),
                row.get("field"),
                row.get("error_mode"),
                float(row.get("error_bound")) if row.get("error_bound") is not None else None,
            )
        ] = row
    native_raw = [
        row for row in native_index.values()
        if (row.get("compressor"), row.get("variant")) in NATIVE_VARIANTS
        and row.get("error_mode") == "rel_range"
    ]
    annotated = validity.annotate(composed_raw + native_raw)
    composed = [r for r in annotated[: len(composed_raw)] if compression_usable(r)]
    native = [
        r
        for r in annotated[len(composed_raw) :]
        if (r.get("compressor"), r.get("variant")) in NATIVE_VARIANTS
        and compression_usable(r)
        and r.get("error_mode") == "rel_range"
    ]

    native_attempts = [
        r
        for r in annotated[len(composed_raw) :]
        if (r.get("compressor"), r.get("variant")) in NATIVE_VARIANTS
        and r.get("error_mode") == "rel_range"
    ]

    compare_dataset_hash = all(row.get("dataset_sha256") for row in composed_raw)
    candidate_signatures: dict[tuple, tuple] = {}
    for row in composed:
        field_key = (row.get("dataset"), row.get("field"))
        signature = data_signature(row, compare_dataset_hash)
        prior = candidate_signatures.setdefault(field_key, signature)
        if signature != prior:
            raise ValueError(f"inconsistent composed data signature for {field_key}")
    paired_native = [
        row
        for row in native
        if (row.get("dataset"), row.get("field")) in candidate_signatures
    ]
    signature_mismatches = [
        row
        for row in paired_native
        if data_signature(row, compare_dataset_hash)
        != candidate_signatures[(row.get("dataset"), row.get("field"))]
    ]
    if signature_mismatches:
        first = signature_mismatches[0]
        raise ValueError(
            "native/composed data signature mismatch for "
            f"{first.get('dataset')}/{first.get('field')}"
        )

    native_by_field: dict[tuple, list[dict]] = defaultdict(list)
    native_by_cell: dict[tuple, set[tuple[str, str]]] = defaultdict(set)
    native_by_curve: dict[tuple, set[tuple[str, str, float]]] = defaultdict(set)
    for row in native:
        field_key = (row.get("dataset"), row.get("field"))
        native_by_field[field_key].append(row)
        native_by_cell[key(row)].add((row["compressor"], row["variant"]))
        native_by_curve[field_key].add(
            (row["compressor"], row["variant"], float(row["error_bound"]))
        )

    native_attempt_by_curve: dict[tuple, set[tuple[str, str, float]]] = defaultdict(set)
    for row in native_attempts:
        native_attempt_by_curve[(row.get("dataset"), row.get("field"))].add(
            (row["compressor"], row["variant"], float(row["error_bound"]))
        )

    expected_curve = {
        (compressor, variant, bound)
        for compressor, variant in NATIVE_VARIANTS
        for bound in (1e-2, 1e-3, 1e-4)
    }

    records = []
    for candidate in composed:
        field_key = (candidate.get("dataset"), candidate.get("field"))
        natives = native_by_field[field_key]
        exact_dominators = [n for n in natives if dominates(n, candidate)]
        robust_dominators = [
            n
            for n in natives
            if dominates(n, candidate, 1.0 + THROUGHPUT_UNCERTAINTY)
        ]
        records.append(
            {
                "dataset": candidate["dataset"],
                "field": candidate.get("field"),
                "bound": float(candidate["error_bound"]),
                "variant": candidate["variant"],
                "candidate": point(candidate),
                "native_points": len(natives),
                "native_variants_at_bound": len(native_by_cell[key(candidate)]),
                "complete_native_attempts": native_attempt_by_curve[field_key]
                >= expected_curve,
                "complete_usable_native_curves": native_by_curve[field_key]
                >= expected_curve,
                "on_available_frontier": not exact_dominators,
                "on_robust_available_frontier": not robust_dominators,
                "exact_dominators": [
                    f"{n['compressor']}:{n['variant']}@{float(n['error_bound']):.0e}"
                    for n in exact_dominators
                ],
                "robust_dominators": [
                    f"{n['compressor']}:{n['variant']}@{float(n['error_bound']):.0e}"
                    for n in robust_dominators
                ],
            }
        )

    fields_by_dataset: dict[str, set[str]] = defaultdict(set)
    for row in composed:
        fields_by_dataset[row["dataset"]].add(row.get("field"))
    coverage = []
    missing_rows = []
    for dataset, fields in sorted(fields_by_dataset.items()):
        attempt_complete = sum(
            native_attempt_by_curve[(dataset, field)] >= expected_curve for field in fields
        )
        all_usable = sum(
            native_by_curve[(dataset, field)] >= expected_curve for field in fields
        )
        coverage.append(
            {
                "dataset": dataset,
                "fields": len(fields),
                "attempt_complete_fields": attempt_complete,
                "all_usable_fields": all_usable,
            }
        )
        for compressor, variant in NATIVE_VARIANTS:
            expected = {
                (field, bound) for field in fields for bound in (1e-2, 1e-3, 1e-4)
            }
            attempted = {
                (r.get("field"), float(r["error_bound"]))
                for r in native_attempts
                if r.get("dataset") == dataset
                and r.get("compressor") == compressor
                and r.get("variant") == variant
            }
            usable = {
                (r.get("field"), float(r["error_bound"]))
                for r in native
                if r.get("dataset") == dataset
                and r.get("compressor") == compressor
                and r.get("variant") == variant
            }
            missing_rows.append(
                {
                    "dataset": dataset,
                    "compressor": compressor,
                    "variant": variant,
                    "missing_attempts": len(expected - attempted),
                    "unusable": len(attempted - usable),
                }
            )

    payload = {
        "schema": "fzgm-rq3-native-frontier-v1",
        "archive_size_correction": archive_size_metadata(args.fzgm_archive_size_config),
        "composed_source": args.composed.name,
        "native_sources": [path.name for path in args.native],
        "definition": {
            "axes": ["compress_throughput_gbs", "cr", "psnr"],
            "native_bounds": [1e-2, 1e-3, 1e-4],
            "psnr_tolerance_db": PSNR_TOL_DB,
            "robust_native_throughput_multiplier": 1.0 + THROUGHPUT_UNCERTAINTY,
            "native_variants": [f"{a}:{b}" for a, b in NATIVE_VARIANTS],
        },
        "candidate_rows_raw": len(composed_raw),
        "candidate_rows_usable": len(composed),
        "native_rows_raw": len(native_raw),
        "native_rows_raw_in_scope": len(native_attempts),
        "native_rows_usable_in_scope": len(native),
        "source_compatibility": {
            "same_session": args.composed.resolve() in {path.resolve() for path in args.native},
            "paired_native_rows_checked": len(paired_native),
            "signature_fields": [
                "dataset", "field", "dims", "dtype", "original_bytes",
                *(["dataset_sha256"] if compare_dataset_hash else []),
            ],
            "signature_mismatches": len(signature_mismatches),
            "composed_rows_with_dataset_sha256": sum(
                bool(row.get("dataset_sha256")) for row in composed
            ),
            "paired_native_rows_with_dataset_sha256": sum(
                bool(row.get("dataset_sha256")) for row in paired_native
            ),
        },
        "coverage": coverage,
        "missing_native_rows": missing_rows,
        "summary": summarize(records),
        "records": records,
        "dominators": dict(
            Counter(d for record in records for d in record["exact_dominators"])
        ),
    }

    for path in (args.output_json, args.output_md, args.output_tex):
        path.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2) + "\n")
    args.output_md.write_text(render_markdown(payload))
    args.output_tex.write_text(render_tex(payload, args.selected_variant))
    print(
        f"usable candidates={len(composed)}/{len(composed_raw)}, "
        f"usable natives={len(native)}/{len(native_attempts)} in scope, "
        f"records={len(records)}"
    )


if __name__ == "__main__":
    main()
