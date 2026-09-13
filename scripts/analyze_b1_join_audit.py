#!/usr/bin/env python3
"""Audit native/FZGM B1 comparison joins before measuring fidelity.

The output is deterministic and artifact-oriented: a JSON analysis manifest, a
row-level CSV, and a Markdown report. It intentionally does not assign fidelity
tiers or aggregate CR/quality. Those operations are only meaningful after every
comparison coordinate has a unique, identity-clean native/FZGM pair.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
import subprocess
from typing import Any, Iterable

import yaml

from benchkit.schema import load_result_file


ANALYSIS_SCHEMA_VERSION = 1
REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = Path(__file__).resolve()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def git_commit(path: Path) -> str | None:
    proc = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    return proc.stdout.strip() if proc.returncode == 0 else None


def load_contract(path: Path) -> dict[str, Any]:
    raw = yaml.safe_load(path.read_text()) or {}
    if raw.get("schema_version") != 1:
        raise ValueError(f"{path}: unsupported or missing schema_version")
    pairs = raw.get("pairs")
    if not isinstance(pairs, list) or not pairs:
        raise ValueError(f"{path}: pairs must be a non-empty list")
    pair_ids = [pair.get("id") for pair in pairs]
    if any(not isinstance(pair_id, str) or not pair_id for pair_id in pair_ids):
        raise ValueError(f"{path}: every pair requires a non-empty string id")
    if len(pair_ids) != len(set(pair_ids)):
        raise ValueError(f"{path}: duplicate pair id")
    for pair in pairs:
        for side in ("reference", "reconstruction"):
            selector = pair.get(side)
            if not isinstance(selector, dict):
                raise ValueError(f"{path}: pair {pair['id']} lacks {side} selector")
            if "pipeline" in selector and "pipeline_any" in selector:
                raise ValueError(
                    f"{path}: pair {pair['id']} {side} specifies pipeline and pipeline_any"
                )
            if "pipeline_any" in selector and not isinstance(selector["pipeline_any"], list):
                raise ValueError(
                    f"{path}: pair {pair['id']} {side}.pipeline_any must be a list"
                )
    return raw


def selector_matches(row: dict[str, Any], selector: dict[str, Any]) -> bool:
    for key, expected in selector.items():
        if key == "pipeline_any":
            if row.get("pipeline") not in expected:
                return False
        elif row.get(key) != expected:
            return False
    return True


def freeze(value: Any) -> Any:
    if isinstance(value, list):
        return tuple(freeze(item) for item in value)
    if isinstance(value, dict):
        return tuple(sorted((key, freeze(item)) for key, item in value.items()))
    return value


def row_value(row: dict[str, Any], field: str) -> Any:
    """Return a canonical value, including fields retained only on failed runs."""
    value = row.get(field)
    if value is not None:
        return value

    logical_cell = row.get("logical_cell") or {}
    if field in logical_cell and logical_cell[field] is not None:
        return logical_cell[field]

    execution_context = row.get("execution_context") or {}
    resolved_config = execution_context.get("resolved_config") or {}
    if field in resolved_config and resolved_config[field] is not None:
        return resolved_config[field]

    dataset = execution_context.get("dataset") or {}
    dataset_fields = {
        "dataset_sha256": "sha256",
        "original_bytes": "bytes",
    }
    dataset_field = dataset_fields.get(field)
    if dataset_field and dataset.get(dataset_field) is not None:
        return dataset[dataset_field]

    if field == "num_elements":
        dims = row_value(row, "dims")
        if isinstance(dims, (list, tuple)) and dims:
            return math.prod(dims)
    return None


def coordinate(row: dict[str, Any], fields: Iterable[str]) -> tuple[Any, ...]:
    return tuple(freeze(row_value(row, field)) for field in fields)


def display_coordinate(key: tuple[Any, ...], fields: list[str]) -> dict[str, Any]:
    def thaw(value: Any) -> Any:
        if isinstance(value, tuple):
            return [thaw(item) for item in value]
        return value

    return {field: thaw(value) for field, value in zip(fields, key)}


def values_equal(field: str, reference: Any, reconstruction: Any) -> bool:
    if reference is None or reconstruction is None:
        return reference is reconstruction
    if field == "eb_abs_effective" and isinstance(reference, (int, float)) and isinstance(
        reconstruction, (int, float)
    ):
        return math.isclose(float(reference), float(reconstruction), rel_tol=1e-12, abs_tol=0.0)
    return freeze(reference) == freeze(reconstruction)


def audit(
    rows: list[dict[str, Any]], contract: dict[str, Any]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    coordinate_fields = list(contract["coordinate_fields"])
    identity_fields = list(contract["identity_checks"])
    pair_summaries: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []

    for pair in contract["pairs"]:
        reference_rows = [row for row in rows if selector_matches(row, pair["reference"])]
        reconstruction_rows = [
            row for row in rows if selector_matches(row, pair["reconstruction"])
        ]
        reference_by_key: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
        reconstruction_by_key: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
        for row in reference_rows:
            reference_by_key[coordinate(row, coordinate_fields)].append(row)
        for row in reconstruction_rows:
            reconstruction_by_key[coordinate(row, coordinate_fields)].append(row)

        counts = defaultdict(int)
        for key in sorted(
            set(reference_by_key) | set(reconstruction_by_key), key=lambda item: repr(item)
        ):
            reference_candidates = reference_by_key.get(key, [])
            reconstruction_candidates = reconstruction_by_key.get(key, [])
            reference = reference_candidates[0] if len(reference_candidates) == 1 else None
            reconstruction = (
                reconstruction_candidates[0] if len(reconstruction_candidates) == 1 else None
            )
            issues: list[str] = []
            if not reference_candidates:
                issues.append("reference_missing")
            elif len(reference_candidates) > 1:
                issues.append("reference_duplicate")
            if not reconstruction_candidates:
                issues.append("reconstruction_missing")
            elif len(reconstruction_candidates) > 1:
                issues.append("reconstruction_duplicate")

            identity_mismatches: list[str] = []
            identity_unverifiable: list[str] = []
            if reference is not None and reconstruction is not None:
                for field in identity_fields:
                    reference_value = row_value(reference, field)
                    reconstruction_value = row_value(reconstruction, field)
                    if reference_value is None or reconstruction_value is None:
                        identity_unverifiable.append(field)
                    elif not values_equal(field, reference_value, reconstruction_value):
                        identity_mismatches.append(field)
                if identity_mismatches:
                    issues.append("identity_mismatch")
                if identity_unverifiable:
                    counts["identity_incomplete"] += 1

                reference_failed = reference.get("status") == "fail"
                reconstruction_failed = reconstruction.get("status") == "fail"
                if reference_failed:
                    counts["reference_execution_failed"] += 1
                if reconstruction_failed:
                    counts["reconstruction_execution_failed"] += 1
                if reference_failed and reconstruction_failed:
                    counts["both_execution_failed"] += 1

            if not issues:
                statuses = {reference.get("status"), reconstruction.get("status")}
                if statuses == {"ok"} and not identity_unverifiable:
                    outcome = "matched_identity_clean"
                elif "fail" in statuses:
                    outcome = "matched_execution_failed"
                else:
                    outcome = "identity_unverifiable"
            elif "reference_duplicate" in issues or "reconstruction_duplicate" in issues:
                outcome = "duplicate"
            elif "reference_missing" in issues:
                outcome = "reference_missing"
            elif "reconstruction_missing" in issues:
                outcome = "reconstruction_missing"
            else:
                outcome = "identity_mismatch"
            counts[outcome] += 1
            counts["coordinates"] += 1

            coord = display_coordinate(key, coordinate_fields)
            audit_rows.append(
                {
                    "pair_id": pair["id"],
                    **coord,
                    "outcome": outcome,
                    "issues": issues,
                    "identity_mismatches": identity_mismatches,
                    "identity_unverifiable": identity_unverifiable,
                    "reference_count": len(reference_candidates),
                    "reconstruction_count": len(reconstruction_candidates),
                    "reference_status": reference.get("status") if reference else None,
                    "reconstruction_status": (
                        reconstruction.get("status") if reconstruction else None
                    ),
                    "reference_fail_phase": (
                        reference.get("fail_phase") if reference else None
                    ),
                    "reconstruction_fail_phase": (
                        reconstruction.get("fail_phase") if reconstruction else None
                    ),
                    "reference_error_type": (
                        reference.get("error_type") if reference else None
                    ),
                    "reconstruction_error_type": (
                        reconstruction.get("error_type") if reconstruction else None
                    ),
                    "reference_error_message": (
                        reference.get("error_message") if reference else None
                    ),
                    "reconstruction_error_message": (
                        reconstruction.get("error_message") if reconstruction else None
                    ),
                    "reference_logical_cell_id": (
                        reference.get("logical_cell_id") if reference else None
                    ),
                    "reconstruction_logical_cell_id": (
                        reconstruction.get("logical_cell_id") if reconstruction else None
                    ),
                    "reference_provenance_id": (
                        reference.get("provenance_id") if reference else None
                    ),
                    "reconstruction_provenance_id": (
                        reconstruction.get("provenance_id") if reconstruction else None
                    ),
                }
            )

        pair_summaries.append(
            {
                "id": pair["id"],
                "label": pair.get("label", pair["id"]),
                "intended_relationship": pair.get("intended_relationship"),
                "reference_rows": len(reference_rows),
                "reconstruction_rows": len(reconstruction_rows),
                "coordinates": counts["coordinates"],
                "matched_identity_clean": counts["matched_identity_clean"],
                "matched_execution_failed": counts["matched_execution_failed"],
                "identity_unverifiable": counts["identity_incomplete"],
                "successful_identity_unverifiable": counts["identity_unverifiable"],
                "reference_execution_failed": counts["reference_execution_failed"],
                "reconstruction_execution_failed": counts[
                    "reconstruction_execution_failed"
                ],
                "both_execution_failed": counts["both_execution_failed"],
                "reference_missing": counts["reference_missing"],
                "reconstruction_missing": counts["reconstruction_missing"],
                "duplicates": counts["duplicate"],
                "identity_mismatches": counts["identity_mismatch"],
            }
        )

    return pair_summaries, audit_rows


def source_manifest(session: Path) -> dict[str, Any]:
    runs = session / "runs.jsonl"
    if not runs.is_file():
        raise ValueError(f"{session}: runs.jsonl is missing; merge the session first")
    provenance = session / "provenance.json"
    metadata = session / "metadata.yaml"
    verification = session / "verification.json"
    return {
        "session_id": session.name,
        "runs_sha256": sha256_file(runs),
        "provenance_sha256": sha256_file(provenance) if provenance.is_file() else None,
        "metadata_sha256": sha256_file(metadata) if metadata.is_file() else None,
        "verification_sha256": (
            sha256_file(verification) if verification.is_file() else None
        ),
    }


def write_csv(path: Path, rows: list[dict[str, Any]], coordinate_fields: list[str]) -> None:
    columns = [
        "pair_id",
        *coordinate_fields,
        "outcome",
        "issues",
        "identity_mismatches",
        "identity_unverifiable",
        "reference_count",
        "reconstruction_count",
        "reference_status",
        "reconstruction_status",
        "reference_fail_phase",
        "reconstruction_fail_phase",
        "reference_error_type",
        "reconstruction_error_type",
        "reference_error_message",
        "reconstruction_error_message",
        "reference_logical_cell_id",
        "reconstruction_logical_cell_id",
        "reference_provenance_id",
        "reconstruction_provenance_id",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            encoded = dict(row)
            for key in coordinate_fields:
                if isinstance(encoded.get(key), (list, dict)):
                    encoded[key] = json.dumps(encoded[key], sort_keys=True, separators=(",", ":"))
            encoded["issues"] = ";".join(row["issues"])
            encoded["identity_mismatches"] = ";".join(row["identity_mismatches"])
            encoded["identity_unverifiable"] = ";".join(row["identity_unverifiable"])
            writer.writerow({column: encoded.get(column) for column in columns})


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    totals = payload["totals"]
    lines = [
        "# B1 reconstruction join audit",
        "",
        "This report audits comparison coverage and input identity before any fidelity",
        "or performance aggregation. It does not assign reconstruction tiers.",
        "",
        "## Sources",
        "",
    ]
    for source in payload["sources"]:
        lines.append(
            f"- `{source['session_id']}`: `runs.jsonl` SHA-256 "
            f"`{source['runs_sha256']}`"
        )
    lines.extend(
        [
            "",
            "## Coverage",
            "",
            "| Pair | Reference rows | FZGM rows | Coordinates | Fidelity-ready | "
            "Reference failed | FZGM failed | Reference missing | FZGM missing | "
            "Duplicates | Identity mismatches |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for pair in payload["pairs"]:
        lines.append(
            f"| {pair['label']} | {pair['reference_rows']} | "
            f"{pair['reconstruction_rows']} | {pair['coordinates']} | "
            f"{pair['matched_identity_clean']} | {pair['reference_execution_failed']} | "
            f"{pair['reconstruction_execution_failed']} | {pair['reference_missing']} | "
            f"{pair['reconstruction_missing']} | {pair['duplicates']} | "
            f"{pair['identity_mismatches']} |"
        )
    lines.extend(
        [
            "",
            "## Totals",
            "",
            f"- Comparison coordinates: **{totals['coordinates']}**",
            f"- Unique, identity-clean pairs: **{totals['matched_identity_clean']}**",
            f"- Paired coordinates with an execution failure: "
            f"**{totals['matched_execution_failed']}**",
            f"- Paired coordinates with incomplete identity fields: "
            f"**{totals['identity_unverifiable']}**",
            f"- Successful pairs with incomplete identity fields: "
            f"**{totals['successful_identity_unverifiable']}**",
            f"- Missing reference rows: **{totals['reference_missing']}**",
            f"- Missing FZGM rows: **{totals['reconstruction_missing']}**",
            f"- Duplicate coordinates: **{totals['duplicates']}**",
            f"- Identity mismatches: **{totals['identity_mismatches']}**",
            "",
            "## Gate",
            "",
            (
                "**PASS:** every configured coordinate has one reference row and one "
                "FZGM row, with no contradictory identity fields. Fidelity may be "
                "measured for the identity-clean, successful subset above."
                if payload["join_gate_passed"]
                else "**FAIL:** resolve missing, duplicate, identity-mismatched, or "
                "identity-incomplete successful rows before measuring reconstruction "
                "fidelity."
            ),
            "",
            "The accompanying CSV contains one row per comparison coordinate. The JSON",
            "records source, contract, and generator checksums for artifact packaging.",
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
    sessions = [session.resolve() for session in args.session]
    contract = load_contract(contract_path)
    rows: list[dict[str, Any]] = []
    for session in sessions:
        runs = session / "runs.jsonl"
        if not runs.is_file():
            raise ValueError(f"{session}: runs.jsonl is missing; merge the session first")
        rows.extend(load_result_file(runs))

    pairs, audit_rows = audit(rows, contract)
    total_keys = (
        "coordinates",
        "matched_identity_clean",
        "matched_execution_failed",
        "identity_unverifiable",
        "successful_identity_unverifiable",
        "reference_execution_failed",
        "reconstruction_execution_failed",
        "both_execution_failed",
        "reference_missing",
        "reconstruction_missing",
        "duplicates",
        "identity_mismatches",
    )
    totals = {key: sum(pair[key] for pair in pairs) for key in total_keys}
    gate_passed = all(
        totals[key] == 0
        for key in (
            "reference_missing",
            "reconstruction_missing",
            "duplicates",
            "identity_mismatches",
            "successful_identity_unverifiable",
        )
    )
    payload = {
        "analysis_schema_version": ANALYSIS_SCHEMA_VERSION,
        "study": contract["study"],
        "join_gate_passed": gate_passed,
        "sources": [source_manifest(session) for session in sessions],
        "contract": {
            "name": contract_path.name,
            "sha256": sha256_file(contract_path),
        },
        "generator": {
            "path": SCRIPT_PATH.relative_to(REPO_ROOT).as_posix(),
            "sha256": sha256_file(SCRIPT_PATH),
            "benchkit_commit": git_commit(REPO_ROOT),
        },
        "coordinate_fields": contract["coordinate_fields"],
        "identity_checks": contract["identity_checks"],
        "input_rows": len(rows),
        "selected_rows": sum(
            pair["reference_rows"] + pair["reconstruction_rows"] for pair in pairs
        ),
        "pairs": pairs,
        "totals": totals,
    }

    prefix = args.output_prefix.resolve()
    prefix.parent.mkdir(parents=True, exist_ok=True)
    json_path = prefix.with_suffix(".json")
    csv_path = prefix.with_suffix(".csv")
    markdown_path = prefix.with_suffix(".md")
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")
    write_csv(csv_path, audit_rows, contract["coordinate_fields"])
    write_markdown(markdown_path, payload)
    print(markdown_path.read_text(), end="")
    if not gate_passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
