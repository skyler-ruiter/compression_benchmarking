import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from benchkit.store import ResultStore


class ResultStoreLoadingTests(TestCase):
    @staticmethod
    def _write(path: Path, *rows: dict) -> None:
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))

    def test_unmerged_session_loads_all_shards(self):
        with TemporaryDirectory() as tmp:
            store = ResultStore(Path(tmp), "session")
            self._write(store.dir / "runs.shard-0-of-2.jsonl", {"cell_key": "a"})
            self._write(store.dir / "runs.shard-1-of-2.jsonl", {"cell_key": "b"})

            self.assertEqual(
                {row["cell_key"] for row in store.load_rows()}, {"a", "b"})

    def test_merged_session_does_not_reload_raw_shards(self):
        with TemporaryDirectory() as tmp:
            store = ResultStore(Path(tmp), "session")
            row = {"cell_key": "a", "status": "ok"}
            self._write(store.dir / "runs.shard-0-of-1.jsonl", row)
            self._write(store.dir / "runs.jsonl", row)

            loaded = store.load_rows()
            self.assertEqual(len(loaded), 1)
            self.assertEqual(loaded[0]["cell_key"], "a")
            self.assertEqual(loaded[0]["result_schema_version"], 0)
            self.assertTrue(loaded[0]["legacy_identity_synthesized"])

    def test_current_shard_can_be_loaded_after_an_earlier_merge(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            merged_store = ResultStore(root, "session")
            self._write(merged_store.runs_path, {"cell_key": "old"})
            shard_store = ResultStore(root, "session", shard=(0, 1))
            self._write(shard_store.runs_path, {"cell_key": "new"})

            loaded = shard_store.load_rows(canonical=False)
            self.assertEqual(len(loaded), 1)
            self.assertEqual(loaded[0]["cell_key"], "new")
            self.assertEqual(loaded[0]["result_schema_version"], 0)

    def test_resume_uses_execution_id_and_ignores_legacy_cell_key(self):
        with TemporaryDirectory() as tmp:
            store = ResultStore(Path(tmp), "session")
            self._write(store.runs_path,
                        {"status": "ok", "cell_key": "legacy-only"},
                        {"status": "ok", "cell_key": "same-cell",
                         "execution_id": "execution-v1-exact"})

            self.assertEqual(store.completed_execution_ids(), {"execution-v1-exact"})
