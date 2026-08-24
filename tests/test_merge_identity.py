from argparse import Namespace
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from benchkit.cli import cmd_merge
from benchkit.identity import execution_id, logical_cell_id, make_execution_context, make_logical_cell
from benchkit.schema import dumps_result, load_result_file


def row(run_id: str, tool_sha: str, cr: float) -> dict:
    logical = make_logical_cell(
        compressor="fzgm", variant="cusz", pipeline="configs/pipelines/cusz.toml",
        dataset="d", field="f", error_mode="rel_range", error_bound=1e-3)
    lid = logical_cell_id(logical)
    context = make_execution_context(
        logical_id=lid, run_parameters={}, resolved_config={},
        dataset={"sha256": "a" * 64, "bytes": 16},
        tool={"sha256": tool_sha}, harness={}, environment={})
    return {
        "run_id": run_id, "session_id": "session",
        "cell_key": "fzgm|cusz|configs/pipelines/cusz.toml|d|f|rel_range|0.001",
        "identity_schema_version": 1, "logical_cell_id": lid,
        "execution_id": execution_id(context), "logical_cell": logical,
        "execution_context": context, "dataset_sha256": "a" * 64,
        "compressor": "fzgm", "variant": "cusz",
        "pipeline": "configs/pipelines/cusz.toml", "dataset": "d", "field": "f",
        "status": "ok", "timestamp": "2026-08-24T00:00:00+00:00",
        "dtype": "f32", "dims": [4], "num_elements": 4, "original_bytes": 16,
        "error_mode": "rel_range", "error_bound": 1e-3,
        "compressed_bytes": 8, "cr": cr, "psnr": 80.0, "max_abs_err": 1e-3,
        "eb_satisfied": True, "timing_reliable": True,
    }


class LogicalMergeTests(TestCase):
    def test_newer_execution_supersedes_same_logical_cell(self):
        with TemporaryDirectory() as tmp:
            session = Path(tmp)
            old = row("old-attempt", "b" * 64, 2.0)
            new = row("new-attempt", "c" * 64, 3.0)
            shard = session / "runs.shard-0-of-1.jsonl"
            shard.write_text(dumps_result(old) + "\n" + dumps_result(new) + "\n")

            with redirect_stdout(StringIO()):
                self.assertEqual(cmd_merge(Namespace(session_dir=str(session))), 0)

            merged = load_result_file(session / "runs.jsonl")
            self.assertEqual(len(merged), 1)
            self.assertEqual(merged[0]["run_id"], "new-attempt")
            self.assertEqual(merged[0]["cr"], 3.0)
            self.assertNotEqual(old["execution_id"], new["execution_id"])
