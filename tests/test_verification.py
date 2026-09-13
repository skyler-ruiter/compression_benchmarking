import json
import tempfile
import unittest
from argparse import Namespace
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

from benchkit.identity import (execution_id, logical_cell_id, make_execution_context,
                               make_logical_cell)
from benchkit.artifact import build_artifact, verify_artifact
from benchkit.provenance import assign_provenance_id
from benchkit.store import ResultStore
from benchkit.cli import cmd_merge, cmd_verify
from benchkit.schema import dumps_result, load_result_file, load_session_text
from benchkit.verification import load_verification_report, verify_session


def _row(provenance_id: str, *, status="ok", reliable=True):
    logical = make_logical_cell(
        compressor="fzgm", variant="test", pipeline="default", dataset="d", field="f",
        error_mode="rel_range", error_bound=1e-3)
    lid = logical_cell_id(logical)
    context = make_execution_context(
        logical_id=lid, run_parameters={},
        resolved_config={"dtype": "f32", "dims": [4], "dim_order": "fast-to-slow"},
        dataset={"sha256": "a" * 64, "bytes": 16}, tool={}, harness={}, environment={})
    base = {
        "run_id": "run", "session_id": "session", "compressor": "fzgm",
        "variant": "test", "pipeline": "default", "dataset": "d", "field": "f",
        "error_mode": "rel_range", "error_bound": 1e-3,
        "identity_schema_version": 1, "logical_cell_id": lid,
        "execution_id": execution_id(context), "logical_cell": logical,
        "execution_context": context, "dataset_sha256": "a" * 64,
        "provenance_id": provenance_id, "status": status,
    }
    if status == "ok":
        base.update({"timestamp": "2026-08-24T00:00:00+00:00", "dtype": "f32",
                     "dims": [4], "num_elements": 4, "original_bytes": 16,
                     "compressed_bytes": 8, "cr": 2.0, "psnr": 60.0,
                     "max_abs_err": 0.0005, "eb_satisfied": True,
                     "timing_reliable": reliable})
    else:
        base.update({"fail_phase": "compress", "error_type": "RuntimeError",
                     "error_message": "expected test failure"})
    return base


class VerificationTests(unittest.TestCase):
    def _session(self, root: Path, verification=None, row_status="ok", reliable=True):
        store = ResultStore(root, "session")
        experiment = {
            "name": "test", "datasets": ["d"], "fields": "all",
            "error": {"mode": "rel_range", "bounds": [1e-3]},
            "runs": [{"compressor": "fzgm", "variant": "test", "pipeline": "default"}],
        }
        if verification is not None:
            experiment["verification"] = verification
        datasets = {"d": {"dtype": "f32", "fields": {"f": {"dims": [4],
                                                                 "path": "f.bin"}}}}
        artifacts = {
            "experiment": store.archive_bytes("experiment", _yaml(experiment), ".yaml"),
            "datasets": store.archive_bytes("datasets", _yaml(datasets), ".yaml"),
            "dataset_checksums": store.archive_bytes("dataset-checksums", b"lock\n", ".yaml"),
            "site": store.archive_json("site", {}),
            "resolved_datasets": store.archive_json("resolved-datasets", {}),
            "harness_patch": store.archive_bytes("harness", b"", ".patch"),
        }
        manifest = {
            "session_schema_version": 1, "session_kind": "native", "session_id": "session",
            "timestamp": "2026-08-24T00:00:00+00:00", "shard": None, "gpu": {},
            "host": {}, "scheduler": {}, "software": {}, "harness": {},
            "compressors": {"fzgm:test": {"build_provenance_complete": True}},
            "nvidia_smi": None, "provenance_schema_version": 1,
            "input_artifacts": artifacts,
            "dataset_integrity": {
                "all_verified": True, "publication_grade": True,
                "datasets": [{"dataset": "d", "field": "f", "verified": True,
                              "expected_sha256": "a" * 64,
                              "observed_sha256": "a" * 64}],
            },
        }
        manifest["provenance_id"] = assign_provenance_id(manifest)
        store.write_provenance(manifest)
        store.append(_row(manifest["provenance_id"], status=row_status, reliable=reliable))
        return store.dir

    def test_complete_session_passes_and_writes_strict_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            session = self._session(Path(tmp))
            report = verify_session(session)
            self.assertTrue(report["publication_grade"])
            saved = load_verification_report(session / "verification.json")
            self.assertTrue(saved["publication_grade"])

    def test_unaccepted_failure_fails_but_named_policy_passes(self):
        with tempfile.TemporaryDirectory() as tmp:
            session = self._session(Path(tmp), row_status="fail")
            report = verify_session(session)
            self.assertFalse(report["publication_grade"])
        with tempfile.TemporaryDirectory() as tmp:
            policy = {"accepted_failures": [{"match": {"error_type": "RuntimeError"},
                                               "reason": "known fixture failure"}]}
            session = self._session(Path(tmp), verification=policy, row_status="fail")
            self.assertTrue(verify_session(session)["publication_grade"])

    def test_unreliable_timing_requires_named_policy(self):
        with tempfile.TemporaryDirectory() as tmp:
            session = self._session(Path(tmp), reliable=False)
            self.assertFalse(verify_session(session)["publication_grade"])
        with tempfile.TemporaryDirectory() as tmp:
            policy = {"accepted_unreliable_timing": [
                {"match": {"dataset": "d"}, "reason": "fixture variance"}]}
            session = self._session(Path(tmp), verification=policy, reliable=False)
            self.assertTrue(verify_session(session)["publication_grade"])

    def test_missing_canonical_cell_fails_matrix_coverage(self):
        with tempfile.TemporaryDirectory() as tmp:
            session = self._session(Path(tmp))
            (session / "runs.jsonl").write_text("")
            report = verify_session(session)
            self.assertFalse(report["publication_grade"])
            failed = {c["name"] for c in report["checks"] if c["status"] == "fail"}
            self.assertIn("matrix_coverage", failed)
            with redirect_stdout(StringIO()):
                self.assertEqual(cmd_verify(Namespace(session_dir=str(session))), 1)

    def test_stale_merge_is_detected(self):
        with tempfile.TemporaryDirectory() as tmp:
            session = self._session(Path(tmp))
            (session / "runs.jsonl").rename(session / "runs.shard-0-of-1.jsonl")
            with redirect_stdout(StringIO()):
                cmd_merge(Namespace(session_dir=str(session)))
            newer = load_result_file(session / "runs.jsonl")[0]
            newer["run_id"] = "newer-attempt"
            with (session / "runs.shard-0-of-1.jsonl").open("a") as handle:
                handle.write(dumps_result(newer) + "\n")
            report = verify_session(session)
            merge = next(c for c in report["checks"] if c["name"] == "merge_state")
            self.assertEqual(merge["status"], "fail")

    def test_corrupted_archived_input_is_detected(self):
        with tempfile.TemporaryDirectory() as tmp:
            session = self._session(Path(tmp))
            provenance_file = next(session.glob("provenance.provenance-v1-*.json"))
            manifest = load_session_text(provenance_file.read_text())
            artifact = session / manifest["input_artifacts"]["experiment"]["path"]
            artifact.write_text("corrupted: true\n")
            report = verify_session(session)
            checks = {c["name"]: c["status"] for c in report["checks"]}
            self.assertEqual(checks["artifact_checksums"], "fail")

    def test_unreferenced_provenance_does_not_change_measured_matrix(self):
        with tempfile.TemporaryDirectory() as tmp:
            session = self._session(Path(tmp))
            source = next(session.glob("provenance.provenance-v1-*.json"))
            orphan = load_session_text(source.read_text())
            store = ResultStore(Path(tmp), "session")
            other_experiment = {
                "name": "unmeasured-initialization", "datasets": ["d"],
                "fields": "all", "error": {"mode": "rel_range", "bounds": [1e-2]},
                "runs": [{"compressor": "fzgm", "variant": "test",
                          "pipeline": "default"}],
            }
            orphan["input_artifacts"]["experiment"] = store.archive_bytes(
                "experiment", _yaml(other_experiment), ".yaml")
            orphan.pop("provenance_id")
            orphan["provenance_id"] = assign_provenance_id(orphan)
            store.write_provenance(orphan)

            report = verify_session(session)
            self.assertTrue(report["publication_grade"])
            matrix = next(c for c in report["checks"]
                          if c["name"] == "matrix_coverage")
            self.assertEqual(matrix["summary"], "1/1 expected cells")

    def test_gating_exclusion_requires_code_scoped_policy(self):
        with tempfile.TemporaryDirectory() as tmp:
            session = self._session(Path(tmp))
            row = load_result_file(session / "runs.jsonl")[0]
            row.update({"eb_satisfied": False, "err_over_bound": 2.0})
            (session / "runs.jsonl").write_text(dumps_result(row) + "\n")
            self.assertFalse(verify_session(session)["publication_grade"])
        with tempfile.TemporaryDirectory() as tmp:
            policy = {"accepted_exclusions": [{"codes": ["eb_violated_severe"],
                                                  "match": {"dataset": "d"},
                                                  "reason": "known fixture behavior"}]}
            session = self._session(Path(tmp), verification=policy)
            row = load_result_file(session / "runs.jsonl")[0]
            row.update({"eb_satisfied": False, "err_over_bound": 2.0})
            (session / "runs.jsonl").write_text(dumps_result(row) + "\n")
            self.assertTrue(verify_session(session)["publication_grade"])

    def test_artifact_build_and_offline_verify_golden_layout(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            session = self._session(root)
            verify_session(session)
            bundle = root / "bundle"
            manifest = build_artifact(session, bundle)
            self.assertEqual(verify_artifact(bundle)["artifact_id"], manifest["artifact_id"])
            paths = {entry["path"] for entry in manifest["files"]}
            required = {"runs.jsonl", "verification.json", "publication/rows.csv",
                        "publication/rows.metadata.json", "reproduction/smoke.yaml",
                        "reproduction/datasets.yaml",
                        "reproduction/datasets.checksums.yaml",
                        "reproduction/run-smoke.sh", "licenses/BENCHKIT-LICENSE",
                        "licenses/NOTICE.md", "software/pyproject.toml"}
            self.assertTrue(required <= paths)
            metadata = json.loads((bundle / "publication/rows.metadata.json").read_text())
            self.assertEqual(metadata["source_session_id"], "session")
            self.assertEqual(metadata["generator"], "benchkit artifact-v1")

    def test_artifact_verifier_rejects_tampering_and_builder_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            session = self._session(root)
            verify_session(session)
            bundle = root / "bundle"
            build_artifact(session, bundle)
            with self.assertRaisesRegex(ValueError, "already exists"):
                build_artifact(session, bundle)
            with (bundle / "publication/rows.csv").open("a") as handle:
                handle.write("tampered\n")
            with self.assertRaisesRegex(ValueError, "file mismatch"):
                verify_artifact(bundle)


def _yaml(document) -> bytes:
    import yaml
    return yaml.safe_dump(document, sort_keys=False).encode()


if __name__ == "__main__":
    unittest.main()
