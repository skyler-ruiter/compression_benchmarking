from scripts.analyze_b1_join_audit import audit


def contract():
    return {
        "coordinate_fields": [
            "dataset",
            "field",
            "dtype",
            "dims",
            "error_mode",
            "error_bound",
        ],
        "identity_checks": ["dataset_sha256", "original_bytes"],
        "pairs": [
            {
                "id": "demo",
                "label": "Demo",
                "reference": {"compressor": "native", "variant": "plain"},
                "reconstruction": {"compressor": "fzgm", "variant": "plain"},
                "intended_relationship": "faithful_reconstruction",
            }
        ],
    }


def row(compressor, *, sha="a" * 64, field="x"):
    return {
        "compressor": compressor,
        "variant": "plain",
        "dataset": "D",
        "field": field,
        "dtype": "f32",
        "dims": [4, 4],
        "error_mode": "rel_range",
        "error_bound": 1e-3,
        "dataset_sha256": sha,
        "original_bytes": 64,
        "status": "ok",
        "logical_cell_id": f"{compressor}-{field}",
        "provenance_id": f"provenance-{compressor}",
    }


def test_audit_reports_clean_missing_hash_mismatch_and_duplicate_coordinates():
    rows = [
        row("native", field="clean"),
        row("fzgm", field="clean"),
        row("native", field="missing"),
        row("native", field="hash"),
        row("fzgm", field="hash", sha="b" * 64),
        row("native", field="duplicate"),
        row("native", field="duplicate"),
        row("fzgm", field="duplicate"),
    ]

    summaries, details = audit(rows, contract())

    assert summaries == [
        {
            "id": "demo",
            "label": "Demo",
            "intended_relationship": "faithful_reconstruction",
            "reference_rows": 5,
            "reconstruction_rows": 3,
            "coordinates": 4,
            "matched_identity_clean": 1,
            "matched_execution_failed": 0,
            "identity_unverifiable": 0,
            "successful_identity_unverifiable": 0,
            "reference_execution_failed": 0,
            "reconstruction_execution_failed": 0,
            "both_execution_failed": 0,
            "reference_missing": 0,
            "reconstruction_missing": 1,
            "duplicates": 1,
            "identity_mismatches": 1,
        }
    ]
    outcomes = {detail["field"]: detail["outcome"] for detail in details}
    assert outcomes == {
        "clean": "matched_identity_clean",
        "duplicate": "duplicate",
        "hash": "identity_mismatch",
        "missing": "reconstruction_missing",
    }
    hash_detail = next(detail for detail in details if detail["field"] == "hash")
    assert hash_detail["identity_mismatches"] == ["dataset_sha256"]


def test_audit_recovers_failed_run_coordinates_from_execution_context():
    reference = row("native")
    reconstruction = row("fzgm")
    reconstruction.update(
        {
            "status": "fail",
            "dtype": None,
            "dims": None,
            "dataset_sha256": None,
            "original_bytes": None,
            "execution_context": {
                "resolved_config": {"dtype": "f32", "dims": [4, 4]},
                "dataset": {"sha256": "a" * 64, "bytes": 64},
            },
        }
    )

    summaries, details = audit([reference, reconstruction], contract())

    assert summaries[0]["coordinates"] == 1
    assert summaries[0]["matched_execution_failed"] == 1
    assert summaries[0]["reference_missing"] == 0
    assert summaries[0]["reconstruction_missing"] == 0
    assert details[0]["outcome"] == "matched_execution_failed"


def test_audit_rejects_successful_pair_with_incomplete_identity():
    reference = row("native")
    reconstruction = row("fzgm")
    reconstruction["original_bytes"] = None

    summaries, details = audit([reference, reconstruction], contract())

    assert summaries[0]["identity_unverifiable"] == 1
    assert summaries[0]["successful_identity_unverifiable"] == 1
    assert details[0]["outcome"] == "identity_unverifiable"
