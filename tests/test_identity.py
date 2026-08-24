import copy
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from benchkit.identity import (
    IdentityError,
    assert_no_id_collisions,
    canonical_json,
    execution_id,
    logical_cell_id,
    logical_cell_id_from_row,
    make_execution_context,
    make_logical_cell,
    normalize_pipeline_ref,
    sha256_prefix,
)


def logical():
    return make_logical_cell(
        compressor="fzgm", variant="cusz", pipeline="configs/pipelines/cusz.toml",
        dataset="CESM-2D", field="CLDHGH", error_mode="rel_range",
        error_bound=1e-3,
    )


def execution(logical_id_value):
    return make_execution_context(
        logical_id=logical_id_value,
        run_parameters={"run_entry": {"graph": False}, "repetitions": 5},
        resolved_config={"source_sha256": "a" * 64, "error_bound": 1e-3},
        dataset={"sha256": "b" * 64, "bytes": 16},
        tool={"artifacts": {"cli_path": {"sha256": "c" * 64}}},
        harness={"git_sha": "deadbeef", "python": "3.12"},
        environment={"gpu": {"name": "H100"}},
    )


class IdentityTests(TestCase):
    def test_canonical_json_is_independent_of_mapping_order(self):
        self.assertEqual(canonical_json({"b": 2, "a": 1}),
                         canonical_json({"a": 1, "b": 2}))

    def test_logical_id_reconstructs_from_legacy_fields(self):
        payload = logical()
        legacy = {key: value for key, value in payload.items()
                  if key != "identity_schema_version"}
        self.assertEqual(logical_cell_id_from_row(legacy), logical_cell_id(payload))

    def test_logical_id_reconstructs_from_legacy_failure_cell_key(self):
        payload = logical()
        legacy_failure = {
            "cell_key": (
                "fzgm|cusz|configs/pipelines/cusz.toml|"
                "CESM-2D|CLDHGH|rel_range|0.001")
        }
        self.assertEqual(logical_cell_id_from_row(legacy_failure),
                         logical_cell_id(payload))

    def test_execution_id_changes_for_every_execution_input_family(self):
        lid = logical_cell_id(logical())
        base = execution(lid)
        baseline = execution_id(base)
        mutations = [
            ("run_parameters", "repetitions", 6),
            ("resolved_config", "source_sha256", "d" * 64),
            ("dataset", "sha256", "e" * 64),
            ("harness", "git_sha", "cafebabe"),
            ("environment", "gpu", {"name": "A100"}),
        ]
        for section, key, value in mutations:
            changed = copy.deepcopy(base)
            changed[section][key] = value
            self.assertNotEqual(execution_id(changed), baseline, (section, key))

    def test_graph_request_changes_execution_but_not_logical_cell(self):
        payload = logical()
        lid = logical_cell_id(payload)
        off = execution(lid)
        on = copy.deepcopy(off)
        on["run_parameters"]["run_entry"]["graph"] = True
        self.assertEqual(logical_cell_id(payload), lid)
        self.assertNotEqual(execution_id(off), execution_id(on))

    def test_same_id_with_different_payload_is_rejected_as_collision(self):
        rows = [
            {"logical_cell_id": "forced", "logical_cell": {"field": "a"}},
            {"logical_cell_id": "forced", "logical_cell": {"field": "b"}},
        ]
        with self.assertRaises(IdentityError):
            assert_no_id_collisions(rows)

    def test_repo_absolute_and_relative_pipeline_refs_match(self):
        root = Path("/tmp/repo")
        self.assertEqual(
            normalize_pipeline_ref(root / "configs/p.toml", root),
            "configs/p.toml",
        )
        self.assertEqual(
            normalize_pipeline_ref("/old/machine/repo/configs/pipelines/p.toml"),
            "configs/pipelines/p.toml",
        )

    def test_dataset_digest_covers_only_declared_bytes(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "field.bin"
            path.write_bytes(b"scientific-bytes" + b"ignored-padding")
            first = sha256_prefix(path, len(b"scientific-bytes"))
            path.write_bytes(b"scientific-bytes" + b"different-padding")
            self.assertEqual(sha256_prefix(path, len(b"scientific-bytes")), first)
