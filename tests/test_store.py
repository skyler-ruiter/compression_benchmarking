import json
import threading
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from benchkit.store import ResultStore, _write_immutable_once


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


class WriteImmutableOnceTests(TestCase):
    def test_detects_a_genuine_content_mismatch(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "input.yaml"
            _write_immutable_once(path, b"first")
            with self.assertRaises(RuntimeError):
                _write_immutable_once(path, b"second")

    def test_repeated_identical_writes_are_a_no_op(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "input.yaml"
            _write_immutable_once(path, b"same")
            _write_immutable_once(path, b"same")  # must not raise
            self.assertEqual(path.read_bytes(), b"same")

    def test_concurrent_identical_writes_never_collide(self):
        # Regression test for the race that killed 2 of 4 shards mid-launch on
        # Delta: every shard archives the same experiment/dataset YAML at
        # session start, so N threads/processes call this for the *same*
        # path with *identical* content at effectively the same instant. The
        # old "exists? read, compare, write" sequence let a reader observe
        # another writer's file mid-write and misread it as a mismatch.
        for _ in range(50):  # a race is timing-dependent; iterate to catch flakes
            with TemporaryDirectory() as tmp:
                path = Path(tmp) / "input.yaml"
                content = b"x" * 4096  # large enough that a naive write is not one syscall
                barrier = threading.Barrier(8)
                errors: list[BaseException] = []

                def worker():
                    barrier.wait()
                    try:
                        _write_immutable_once(path, content)
                    except BaseException as exc:  # noqa: BLE001
                        errors.append(exc)

                threads = [threading.Thread(target=worker) for _ in range(8)]
                for t in threads:
                    t.start()
                for t in threads:
                    t.join()

                self.assertEqual(errors, [])
                self.assertEqual(path.read_bytes(), content)

    def test_archive_bytes_concurrent_shards_do_not_collide(self):
        # Same race, exercised through the public ResultStore API the runner
        # actually calls (store.archive_bytes for experiment.yaml / datasets.yaml).
        for _ in range(20):
            with TemporaryDirectory() as tmp:
                root = Path(tmp)
                content = json.dumps({"experiment": "specialization_vs_native_full"}).encode()
                barrier = threading.Barrier(4)
                errors: list[BaseException] = []
                results: list[dict] = []

                def worker(shard_idx: int):
                    store = ResultStore(root, "session", shard=(shard_idx, 4))
                    barrier.wait()
                    try:
                        results.append(store.archive_bytes("experiment", content, ".yaml"))
                    except BaseException as exc:  # noqa: BLE001
                        errors.append(exc)

                threads = [threading.Thread(target=worker, args=(k,)) for k in range(4)]
                for t in threads:
                    t.start()
                for t in threads:
                    t.join()

                self.assertEqual(errors, [])
                self.assertEqual(len({r["sha256"] for r in results}), 1)
