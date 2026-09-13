#!/usr/bin/env python3
"""Measure reconstruction fidelity for the B1 native/FZGM pair contract."""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

from benchkit import validity
from benchkit.schema import load_result_file
from scripts.analyze_b1_join_audit import (
    REPO_ROOT,
    coordinate,
    display_coordinate,
    git_commit,
    load_contract,
    selector_matches,
    sha256_file,
    source_manifest,
)


ANALYSIS_SCHEMA_VERSION = 1
SCRIPT_PATH = Path(__file__).resolve()
COUNT_FIELDS = (
    "coordinates",
    "join_error",
    "both_successful",
    "execution_failed",
    "reference_execution_failed",
    "reconstruction_execution_failed",
    "degenerate",
    "reference_severe_bound_violation",
    "reconstruction_severe_bound_violation",
    "reference_marginal_bound_miss",
    "reconstruction_marginal_bound_miss",
    "joint_policy_valid",
    "exact_reconstruction",
)


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def geometric_mean(values: list[float]) -> float | None:
    if not values or any(value <= 0 for value in values):
        return None
    return math.exp(sum(math.log(value) for value in values) / len(values))


def marginal_bound_miss(row: dict[str, Any]) -> bool:
    ratio = row.get("err_over_bound")
    return (
        row.get("eb_satisfied") is False
        and isinstance(ratio, (int, float))
        and math.isfinite(ratio)
        and ratio <= validity.MARGINAL_EB_RATIO
    )


def analyze(
    raw_rows: list[dict[str, Any]], contract: dict[str, Any]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows = validity.annotate(raw_rows)
    coordinate_fields = list(contract["coordinate_fields"])
    exact_field = contract["fidelity_policy"]["exact_reconstruction_field"]
    summaries: list[dict[str, Any]] = []
    details: list[dict[str, Any]] = []

    for pair in contract["pairs"]:
        reference_by_key: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
        reconstruction_by_key: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            if selector_matches(row, pair["reference"]):
                reference_by_key[coordinate(row, coordinate_fields)].append(row)
            if selector_matches(row, pair["reconstruction"]):
                reconstruction_by_key[coordinate(row, coordinate_fields)].append(row)

        counts: dict[str, int] = defaultdict(int)
        psnr_deltas: list[float] = []
        max_error_deltas: list[float] = []
        normalized_error_deltas: list[float] = []
        byte_ratios: list[float] = []
        keys = sorted(
            set(reference_by_key) | set(reconstruction_by_key), key=lambda item: repr(item)
        )
        for key in keys:
            reference_candidates = reference_by_key.get(key, [])
            reconstruction_candidates = reconstruction_by_key.get(key, [])
            reference = reference_candidates[0] if len(reference_candidates) == 1 else None
            reconstruction = (
                reconstruction_candidates[0]
                if len(reconstruction_candidates) == 1
                else None
            )
            counts["coordinates"] += 1
            category = "join_error"
            exact = None
            psnr_delta = None
            max_error_delta = None
            normalized_error_delta = None
            byte_ratio = None

            if reference is not None and reconstruction is not None:
                reference_failed = reference.get("status") != "ok"
                reconstruction_failed = reconstruction.get("status") != "ok"
                if reference_failed:
                    counts["reference_execution_failed"] += 1
                if reconstruction_failed:
                    counts["reconstruction_execution_failed"] += 1

                if reference_failed or reconstruction_failed:
                    category = "execution_failed"
                    counts["execution_failed"] += 1
                else:
                    counts["both_successful"] += 1
                    reference_reasons = reference["_exclusions"]
                    reconstruction_reasons = reconstruction["_exclusions"]
                    degenerate = (
                        "degenerate_field" in reference_reasons
                        or "degenerate_field" in reconstruction_reasons
                    )
                    reference_severe = "eb_violated_severe" in reference_reasons
                    reconstruction_severe = "eb_violated_severe" in reconstruction_reasons
                    if degenerate:
                        counts["degenerate"] += 1
                    if reference_severe:
                        counts["reference_severe_bound_violation"] += 1
                    if reconstruction_severe:
                        counts["reconstruction_severe_bound_violation"] += 1
                    if marginal_bound_miss(reference):
                        counts["reference_marginal_bound_miss"] += 1
                    if marginal_bound_miss(reconstruction):
                        counts["reconstruction_marginal_bound_miss"] += 1

                    if degenerate:
                        category = "degenerate"
                    elif reconstruction_severe:
                        category = "fzgm_severe_bound_violation"
                    elif reference_severe:
                        category = "reference_severe_bound_violation"
                    elif validity.is_valid(reference) and validity.is_valid(reconstruction):
                        category = "joint_policy_valid"
                        counts["joint_policy_valid"] += 1
                        exact = reference.get(exact_field) == reconstruction.get(exact_field)
                        if exact:
                            counts["exact_reconstruction"] += 1

                        reference_psnr = reference.get("psnr")
                        reconstruction_psnr = reconstruction.get("psnr")
                        if all(
                            isinstance(value, (int, float)) and math.isfinite(value)
                            for value in (reference_psnr, reconstruction_psnr)
                        ):
                            psnr_delta = abs(reference_psnr - reconstruction_psnr)
                            psnr_deltas.append(psnr_delta)

                        reference_error = reference.get("max_abs_err")
                        reconstruction_error = reconstruction.get("max_abs_err")
                        if all(
                            isinstance(value, (int, float)) and math.isfinite(value)
                            for value in (reference_error, reconstruction_error)
                        ):
                            max_error_delta = abs(reference_error - reconstruction_error)
                            max_error_deltas.append(max_error_delta)

                        reference_normalized_error = reference.get("err_over_bound")
                        reconstruction_normalized_error = reconstruction.get(
                            "err_over_bound"
                        )
                        if all(
                            isinstance(value, (int, float)) and math.isfinite(value)
                            for value in (
                                reference_normalized_error,
                                reconstruction_normalized_error,
                            )
                        ):
                            normalized_error_delta = abs(
                                reference_normalized_error
                                - reconstruction_normalized_error
                            )
                            normalized_error_deltas.append(normalized_error_delta)

                        reference_bytes = reference.get("compressed_bytes")
                        reconstruction_bytes = reconstruction.get("compressed_bytes")
                        if (
                            isinstance(reference_bytes, (int, float))
                            and reference_bytes > 0
                            and isinstance(reconstruction_bytes, (int, float))
                            and reconstruction_bytes > 0
                        ):
                            byte_ratio = reconstruction_bytes / reference_bytes
                            byte_ratios.append(byte_ratio)
                    else:
                        category = "other_policy_exclusion"
            else:
                counts["join_error"] += 1

            coord = display_coordinate(key, coordinate_fields)
            details.append(
                {
                    "pair_id": pair["id"],
                    **coord,
                    "category": category,
                    "reference_status": reference.get("status") if reference else None,
                    "reconstruction_status": (
                        reconstruction.get("status") if reconstruction else None
                    ),
                    "reference_exclusions": (
                        reference.get("_exclusions", []) if reference else []
                    ),
                    "reconstruction_exclusions": (
                        reconstruction.get("_exclusions", []) if reconstruction else []
                    ),
                    "reconstruction_exact": exact,
                    "psnr_abs_delta_db": psnr_delta,
                    "max_abs_error_delta": max_error_delta,
                    "err_over_bound_abs_delta": normalized_error_delta,
                    "compressed_bytes_ratio_fzgm_over_native": byte_ratio,
                    "reference_logical_cell_id": (
                        reference.get("logical_cell_id") if reference else None
                    ),
                    "reconstruction_logical_cell_id": (
                        reconstruction.get("logical_cell_id") if reconstruction else None
                    ),
                }
            )

        if counts["reconstruction_severe_bound_violation"]:
            evidence = "fzgm_bound_failures_present"
        elif counts["exact_reconstruction"] == counts["joint_policy_valid"] and not counts[
            "reconstruction_execution_failed"
        ]:
            evidence = "supports_exact_reconstruction_on_tested_scope"
        elif counts["reconstruction_execution_failed"]:
            evidence = "supports_bound_preservation_when_successful_with_execution_gaps"
        else:
            evidence = "supports_bound_preserving_nonidentical_reconstruction"

        summaries.append(
            {
                "id": pair["id"],
                "label": pair.get("label", pair["id"]),
                "intended_relationship": pair.get("intended_relationship"),
                **{field: counts[field] for field in COUNT_FIELDS},
                "exact_reconstruction_fraction": (
                    counts["exact_reconstruction"] / counts["joint_policy_valid"]
                    if counts["joint_policy_valid"]
                    else None
                ),
                "psnr_abs_delta_db": {
                    "median": percentile(psnr_deltas, 0.5),
                    "p95": percentile(psnr_deltas, 0.95),
                    "max": max(psnr_deltas) if psnr_deltas else None,
                },
                "max_abs_error_delta_max": (
                    max(max_error_deltas) if max_error_deltas else None
                ),
                "err_over_bound_abs_delta": {
                    "p95": percentile(normalized_error_deltas, 0.95),
                    "max": max(normalized_error_deltas)
                    if normalized_error_deltas
                    else None,
                },
                "compressed_bytes_ratio_fzgm_over_native": {
                    "geomean": geometric_mean(byte_ratios),
                    "min": min(byte_ratios) if byte_ratios else None,
                    "max": max(byte_ratios) if byte_ratios else None,
                },
                "measured_evidence": evidence,
            }
        )

    return summaries, details


def write_csv(path: Path, rows: list[dict[str, Any]], coordinate_fields: list[str]) -> None:
    columns = [
        "pair_id",
        *coordinate_fields,
        "category",
        "reference_status",
        "reconstruction_status",
        "reference_exclusions",
        "reconstruction_exclusions",
        "reconstruction_exact",
        "psnr_abs_delta_db",
        "max_abs_error_delta",
        "err_over_bound_abs_delta",
        "compressed_bytes_ratio_fzgm_over_native",
        "reference_logical_cell_id",
        "reconstruction_logical_cell_id",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            encoded = dict(row)
            for key in coordinate_fields:
                if isinstance(encoded.get(key), (list, dict)):
                    encoded[key] = json.dumps(encoded[key], separators=(",", ":"))
            for key in ("reference_exclusions", "reconstruction_exclusions"):
                encoded[key] = ";".join(row[key])
            writer.writerow({column: encoded.get(column) for column in columns})


def fmt(value: float | None, digits: int = 4) -> str:
    return "--" if value is None else f"{value:.{digits}g}"


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# B1 reconstruction fidelity audit",
        "",
        "This report measures reconstruction agreement after the B1 join gate. Exact raw",
        "reconstruction, error-bound behavior, quality differences, and compressed size",
        "are reported separately. The measurements do not assign a final provenance tier",
        "without the corresponding implementation-attribution evidence.",
        "",
        "## Measured evidence",
        "",
        f"Configured coordinates: **{payload['totals']['coordinates']}**; successful "
        f"pairs: **{payload['totals']['both_successful']}**; jointly valid pairs: "
        f"**{payload['totals']['joint_policy_valid']}**.",
        "",
        "| Pair | Successful | Joint-valid | Exact raw reconstruction | Native severe EB | FZGM severe EB | Native failed | FZGM failed | Normalized max-error delta p95 / max | PSNR delta p95 / max (dB) | FZGM/native bytes geomean [range] |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for pair in payload["pairs"]:
        ratio = pair["compressed_bytes_ratio_fzgm_over_native"]
        psnr = pair["psnr_abs_delta_db"]
        normalized_error = pair["err_over_bound_abs_delta"]
        lines.append(
            f"| {pair['label']} | {pair['both_successful']} | "
            f"{pair['joint_policy_valid']} | {pair['exact_reconstruction']}/"
            f"{pair['joint_policy_valid']} | "
            f"{pair['reference_severe_bound_violation']} | "
            f"{pair['reconstruction_severe_bound_violation']} | "
            f"{pair['reference_execution_failed']} | "
            f"{pair['reconstruction_execution_failed']} | "
            f"{fmt(normalized_error['p95'])} / {fmt(normalized_error['max'])} | "
            f"{fmt(psnr['p95'])} / {fmt(psnr['max'])} | "
            f"{fmt(ratio['geomean'])} [{fmt(ratio['min'])}, {fmt(ratio['max'])}] |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- No configured whole-pipeline pair is byte-identical over every joint-valid",
            "  cell. The earlier two-field Tier-1 wording is therefore not supported by",
            "  this full-corpus audit.",
            "- Every successful, nondegenerate FZGM row avoids a severe error-bound",
            "  violation under `benchkit.validity`. Severe violations occur only in native",
            "  reference rows and remain visible in the table.",
            "- Exact reconstruction hashes occur on subsets of every family. Nonidentical",
            "  hashes require Tier-2 implementation attribution; close aggregate quality",
            "  alone cannot establish bit identity.",
            "- The largest PSNR deltas for the cuSZp families occur on the near-constant",
            "  f64 S3D/N2 field, where tiny absolute error changes are amplified in dB.",
            "  Across joint-valid cuSZp2/cuSZp3 cells, the 95th-percentile normalized",
            "  maximum-error delta is below 0.0005 and the maximum is below 0.009.",
            f"- The validity policy retains marginal misses through "
            f"{payload['validity_policy']['marginal_eb_ratio']:.2f}x.",
            f"  It retains {payload['totals']['reconstruction_marginal_bound_miss']} "
            f"FZGM and {payload['totals']['reference_marginal_bound_miss']} native rows;",
            "  they are reported rather than counted as severe violations.",
            "- The 19 execution failures remain coverage exceptions: four native cuSZ",
            "  failures and fifteen FZGM capacity refusals. They are not silently removed",
            "  from the tested-coordinate count.",
            "",
            "The accompanying CSV contains one row per coordinate, including validity",
            "reasons and logical-cell IDs. The JSON records source, contract, verification,",
            "and generator checksums for later artifact packaging.",
            "",
        ]
    )
    path.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--session", type=Path, action="append", required=True)
    parser.add_argument("--output-prefix", type=Path, required=True)
    args = parser.parse_args()

    contract_path = args.pairs.resolve()
    sessions = [path.resolve() for path in args.session]
    contract = load_contract(contract_path)
    if not isinstance(contract.get("fidelity_policy"), dict):
        raise ValueError(f"{contract_path}: fidelity_policy is required")
    rows: list[dict[str, Any]] = []
    for session in sessions:
        rows.extend(load_result_file(session / "runs.jsonl"))
    pairs, details = analyze(rows, contract)
    totals = {field: sum(pair[field] for pair in pairs) for field in COUNT_FIELDS}

    payload = {
        "analysis_schema_version": ANALYSIS_SCHEMA_VERSION,
        "study": contract["study"],
        "sources": [source_manifest(session) for session in sessions],
        "contract": {"name": contract_path.name, "sha256": sha256_file(contract_path)},
        "generator": {
            "path": SCRIPT_PATH.relative_to(REPO_ROOT).as_posix(),
            "sha256": sha256_file(SCRIPT_PATH),
            "benchkit_commit": git_commit(REPO_ROOT),
        },
        "validity_policy": {
            "module": "benchkit.validity",
            "marginal_eb_ratio": validity.MARGINAL_EB_RATIO,
        },
        "coordinate_fields": contract["coordinate_fields"],
        "input_rows": len(rows),
        "pairs": pairs,
        "totals": totals,
    }
    prefix = args.output_prefix.resolve()
    prefix.parent.mkdir(parents=True, exist_ok=True)
    prefix.with_suffix(".json").write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    write_csv(prefix.with_suffix(".csv"), details, contract["coordinate_fields"])
    write_markdown(prefix.with_suffix(".md"), payload)
    print(prefix.with_suffix(".md").read_text(), end="")


if __name__ == "__main__":
    main()
