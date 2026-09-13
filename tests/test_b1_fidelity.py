from scripts.analyze_b1_fidelity import analyze, percentile


def contract():
    return {
        "coordinate_fields": ["dataset", "field", "error_bound"],
        "fidelity_policy": {"exact_reconstruction_field": "decompressed_sha256"},
        "pairs": [
            {
                "id": "demo",
                "label": "Demo",
                "reference": {"compressor": "native"},
                "reconstruction": {"compressor": "fzgm"},
            }
        ],
    }


def row(compressor, *, sha="same", status="ok"):
    return {
        "compressor": compressor,
        "dataset": "D",
        "field": "x",
        "error_mode": "rel_range",
        "error_bound": 1e-3,
        "status": status,
        "decompressed_sha256": sha,
        "eb_satisfied": True,
        "err_over_bound": 0.9,
        "max_abs_err": 9e-4,
        "psnr": 60.0,
        "cr": 2.0,
        "compressed_bytes": 50,
        "logical_cell_id": compressor,
    }


def test_percentile_interpolates():
    assert percentile([0.0, 10.0], 0.5) == 5.0


def test_analyze_reports_exact_joint_valid_pair():
    summaries, details = analyze([row("native"), row("fzgm")], contract())
    assert summaries[0]["joint_policy_valid"] == 1
    assert summaries[0]["exact_reconstruction"] == 1
    assert summaries[0]["measured_evidence"] == (
        "supports_exact_reconstruction_on_tested_scope"
    )
    assert details[0]["category"] == "joint_policy_valid"


def test_analyze_keeps_fzgm_bound_failure_visible():
    reconstruction = row("fzgm", sha="different")
    reconstruction.update({"eb_satisfied": False, "err_over_bound": 2.0})
    summaries, details = analyze([row("native"), reconstruction], contract())
    assert summaries[0]["reconstruction_severe_bound_violation"] == 1
    assert summaries[0]["measured_evidence"] == "fzgm_bound_failures_present"
    assert details[0]["category"] == "fzgm_severe_bound_violation"
