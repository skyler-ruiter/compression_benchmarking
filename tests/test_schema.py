import json
import math
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from benchkit.schema import (
    RESULT_SCHEMA_VERSION,
    SESSION_SCHEMA_VERSION,
    SchemaError,
    dumps_result,
    dumps_session,
    load_result_file,
    load_result_line,
    load_session_text,
)
from benchkit.identity import execution_id, logical_cell_id
from benchkit.store import ResultStore


def measurement_row(**updates):
    row = {
        "run_id": "run-0001",
        "session_id": "session",
        "cell_key": "fzgm|x|p|d|f|lossless|toml",
        "compressor": "fzgm",
        "variant": "x",
        "pipeline": "p.toml",
        "dataset": "d",
        "field": "f",
        "status": "ok",
        "timestamp": "2026-08-24T00:00:00+00:00",
        "dtype": "f32",
        "dims": [4],
        "num_elements": 4,
        "original_bytes": 16,
        "error_mode": "lossless",
        "error_bound": None,
        "compressed_bytes": 8,
        "cr": 2.0,
        "psnr": math.inf,
        "max_abs_err": 0.0,
        "eb_satisfied": True,
        "timing_reliable": True,
    }
    row.update(updates)
    return row


def native_session():
    return {
        "session_id": "session",
        "timestamp": "2026-08-24T00:00:00+00:00",
        "shard": None,
        "gpu": {},
        "host": {},
        "scheduler": {},
        "software": {},
        "harness": {},
        "compressors": {},
        "nvidia_smi": None,
    }


class VersionedSchemaTests(TestCase):
    def test_exact_psnr_is_strict_json_and_round_trips(self):
        encoded = dumps_result(measurement_row())

        self.assertNotIn("Infinity", encoded)
        raw = json.loads(encoded, parse_constant=lambda value: self.fail(value))
        self.assertEqual(raw["result_schema_version"], RESULT_SCHEMA_VERSION)
        self.assertEqual(raw["record_kind"], "measurement")
        self.assertIsNone(raw["psnr"])
        self.assertEqual(raw["psnr_kind"], "exact")
        self.assertEqual(raw["nonfinite_values"]["/psnr"], "positive_infinity")

        loaded = load_result_line(encoded)
        self.assertTrue(math.isinf(loaded["psnr"]))
        self.assertEqual(loaded["psnr_kind"], "exact")

    def test_nested_nan_is_encoded_explicitly(self):
        encoded = dumps_result(measurement_row(run_notes={"stage": [math.nan]}))
        raw = json.loads(encoded, parse_constant=lambda value: self.fail(value))

        self.assertIsNone(raw["run_notes"]["stage"][0])
        self.assertEqual(
            raw["nonfinite_values"]["/run_notes/stage/0"], "nan")
        self.assertTrue(math.isnan(load_result_line(encoded)["run_notes"]["stage"][0]))

    def test_legacy_row_loads_as_v0_without_rewriting(self):
        text = '{"run_id":"old","status":"ok","psnr":Infinity}'

        loaded = load_result_line(text)

        self.assertEqual(loaded["result_schema_version"], 0)
        self.assertTrue(math.isinf(loaded["psnr"]))
        self.assertEqual(text, '{"run_id":"old","status":"ok","psnr":Infinity}')

    def test_legacy_file_synthesizes_missing_ids_deterministically(self):
        legacy = {
            "compressor": "cusz", "variant": "cusz", "dataset": "CESM",
            "field": "CLDHGH", "status": "ok", "reconstructed": True,
            "reconstruction_source": "stdout.log", "error_bound": 0.01,
            "cr": 12.0, "psnr": 42.0, "eb_satisfied": True,
            "timing_reliable": False,
        }
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "runs.jsonl"
            path.write_text(json.dumps(legacy) + "\n")

            first = load_result_file(path)[0]
            second = load_result_file(path)[0]

            self.assertEqual(first["session_id"], second["session_id"])
            self.assertEqual(first["run_id"], second["run_id"])
            self.assertTrue(first["legacy_identity_synthesized"])
            migrated = json.loads(dumps_result(first))
            self.assertEqual(migrated["source_result_schema_version"], 0)
            self.assertEqual(migrated["record_kind"], "reconstructed_measurement")

    def test_malformed_new_row_is_rejected_before_file_creation(self):
        with TemporaryDirectory() as tmp:
            store = ResultStore(Path(tmp), "session")
            bad = measurement_row()
            del bad["cell_key"]

            with self.assertRaises(SchemaError):
                store.append(bad)

            self.assertFalse(store.runs_path.exists())

    def test_native_session_is_versioned_and_strict(self):
        encoded = dumps_session(native_session(), indent=2)
        raw = json.loads(encoded, parse_constant=lambda value: self.fail(value))

        self.assertEqual(raw["session_schema_version"], SESSION_SCHEMA_VERSION)
        self.assertEqual(raw["session_kind"], "native")
        loaded = load_session_text(encoded)
        self.assertEqual(loaded["session_schema_version"], SESSION_SCHEMA_VERSION)

    def test_wrong_optional_type_is_rejected(self):
        with self.assertRaises(SchemaError):
            dumps_result(measurement_row(graph_active="yes"))

    def test_unknown_non_json_value_is_rejected_instead_of_stringified(self):
        with self.assertRaises(SchemaError):
            dumps_result(measurement_row(run_notes={"bad": object()}))

    def test_identity_hash_mismatch_is_rejected(self):
        logical = {
            "identity_schema_version": 1, "compressor": "fzgm", "variant": "x",
            "pipeline": "p.toml", "dataset": "d", "field": "f",
            "error_mode": "lossless", "error_bound": None,
        }
        lid = logical_cell_id(logical)
        context = {
            "identity_schema_version": 1, "logical_cell_id": lid,
            "run_parameters": {}, "resolved_config": {},
            "dataset": {"sha256": "a" * 64, "bytes": 16}, "tool": {},
            "harness": {}, "environment": {},
        }
        row = measurement_row(
            identity_schema_version=1, logical_cell_id=lid,
            execution_id=execution_id(context), logical_cell=logical,
            execution_context=context, dataset_sha256="a" * 64)
        row["logical_cell"]["field"] = "changed-without-rehashing"

        with self.assertRaises(SchemaError):
            dumps_result(row)

    def test_identity_payload_must_join_top_level_scientific_fields(self):
        logical = {
            "identity_schema_version": 1, "compressor": "fzgm", "variant": "x",
            "pipeline": "p.toml", "dataset": "d", "field": "f",
            "error_mode": "lossless", "error_bound": None,
        }
        lid = logical_cell_id(logical)
        context = {
            "identity_schema_version": 1, "logical_cell_id": lid,
            "run_parameters": {}, "resolved_config": {},
            "dataset": {"sha256": "a" * 64, "bytes": 16}, "tool": {},
            "harness": {}, "environment": {},
        }
        row = measurement_row(
            field="different", identity_schema_version=1, logical_cell_id=lid,
            execution_id=execution_id(context), logical_cell=logical,
            execution_context=context, dataset_sha256="a" * 64)

        with self.assertRaises(SchemaError):
            dumps_result(row)

    def test_failure_record_has_a_distinct_contract(self):
        row = {
            "run_id": "run-0002",
            "session_id": "session",
            "cell_key": "fzgm|x|p|d|f|rel_range|0.001",
            "compressor": "fzgm",
            "variant": "x",
            "pipeline": "p.toml",
            "dataset": "d",
            "field": "f",
            "status": "fail",
            "error_message": "compress failed",
            "error_type": "AdapterError",
            "fail_phase": "compress",
        }

        raw = json.loads(dumps_result(row))

        self.assertEqual(raw["record_kind"], "failure")
        self.assertNotIn("psnr_kind", raw)

    def test_reconstructed_measurement_is_explicitly_lower_fidelity(self):
        row = {
            "run_id": "reconstructed-0000",
            "session_id": "reconstructed-session",
            "compressor": "cusz",
            "variant": "cusz",
            "dataset": "d",
            "field": "f",
            "status": "ok",
            "error_bound": 1e-3,
            "cr": 2.0,
            "psnr": 80.0,
            "eb_satisfied": True,
            "timing_reliable": False,
            "reconstructed": True,
            "reconstruction_source": "slurm.out",
        }

        raw = json.loads(dumps_result(row))

        self.assertEqual(raw["record_kind"], "reconstructed_measurement")
        self.assertNotIn("cell_key", raw)
        self.assertNotIn("pipeline", raw)
