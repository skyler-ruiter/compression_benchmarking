import hashlib
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from benchkit.config import DatasetCatalog
from benchkit.dataset_checksums import dump_checksum_lock
from benchkit.provenance import _software_env, assign_provenance_id, capture_git_state
from benchkit.runner import verify_dataset_inputs
from benchkit.schema import SchemaError, dumps_session
from benchkit.site import Site
from benchkit.store import ResultStore


class H3ProvenanceTests(unittest.TestCase):
    def test_software_env_captures_portable_gpu_build_target(self):
        with patch.dict("os.environ", {
                "BENCHKIT_GPU_ARCH": "sm_89", "CUDA_ARCH": "89",
                "FZGMOD_BACKEND": "CUDA"}, clear=False):
            build = _software_env()["build_environment"]
        self.assertEqual(build["BENCHKIT_GPU_ARCH"], "sm_89")
        self.assertEqual(build["CUDA_ARCH"], "89")
        self.assertEqual(build["FZGMOD_BACKEND"], "CUDA")

    def test_catalog_retains_exact_manifest_and_declared_digest(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = root / "field.bin"
            data.write_bytes(b"abcd")
            digest = hashlib.sha256(b"abcd").hexdigest()
            manifest = root / "datasets.yaml"
            text = ("sample:\n  dtype: u8\n  root: " + str(root) +
                    "\n  fields:\n    f:\n      path: field.bin\n      dims: [4]\n" +
                    f"      sha256: {digest}\n")
            manifest.write_text(text)
            catalog = DatasetCatalog.load(manifest)
            self.assertEqual(catalog.source_text, text)
            self.assertEqual(catalog.resolve("sample", "f").expected_sha256, digest)

    def test_adjacent_checksum_lock_supplies_and_validates_digest(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "field.bin").write_bytes(b"abcd")
            manifest = root / "datasets.yaml"
            manifest.write_text(
                f"sample:\n  dtype: u8\n  root: {root}\n  fields:\n"
                "    f: {path: field.bin, dims: [4]}\n")
            digest = hashlib.sha256(b"abcd").hexdigest()
            lock = root / "datasets.checksums.yaml"
            lock.write_text(dump_checksum_lock({"sample": {"f": digest}}))
            catalog = DatasetCatalog.load(manifest)
            self.assertEqual(catalog.resolve("sample", "f").expected_sha256, digest)
            self.assertEqual(catalog.checksum_text, lock.read_text())

    def test_inline_and_locked_digest_disagreement_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "field.bin").write_bytes(b"abcd")
            manifest = root / "datasets.yaml"
            manifest.write_text(
                f"sample:\n  dtype: u8\n  root: {root}\n  fields:\n"
                f"    f: {{path: field.bin, dims: [4], sha256: {'0' * 64}}}\n")
            lock = root / "datasets.checksums.yaml"
            lock.write_text(dump_checksum_lock({"sample": {"f": "1" * 64}}))
            catalog = DatasetCatalog.load(manifest)
            with self.assertRaisesRegex(ValueError, "differ"):
                catalog.resolve("sample", "f")

    def test_dataset_verification_is_strict_with_explicit_opt_out(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "field.bin"
            path.write_bytes(b"abcdtrailing")
            from benchkit.config import FieldSpec
            spec = FieldSpec("d", "f", "u8", "fast-to-slow", [4], path)
            cells = [(0, (None, spec, None))]
            with self.assertRaisesRegex(RuntimeError, "no declared sha256"):
                verify_dataset_inputs(cells)
            _, records = verify_dataset_inputs(cells, allow_unverified=True)
            self.assertFalse(records[("d", "f")]["verified"])
            spec.expected_sha256 = "0" * 64
            with self.assertRaisesRegex(RuntimeError, "checksum mismatch"):
                verify_dataset_inputs(cells)
            spec.expected_sha256 = hashlib.sha256(b"abcd").hexdigest()
            _, records = verify_dataset_inputs(cells)
            self.assertTrue(records[("d", "f")]["verified"])

    def test_dataset_verification_hashes_each_distinct_input_once(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "field.bin"
            path.write_bytes(b"abcd")
            from benchkit.config import FieldSpec
            spec = FieldSpec("d", "f", "u8", "fast-to-slow", [4], path,
                             expected_sha256=hashlib.sha256(b"abcd").hexdigest())
            # The same field appears once per pipeline/bound in a real matrix.
            cells = [(i, (None, spec, None)) for i in range(8)]
            with patch("benchkit.runner.sha256_prefix",
                       return_value=spec.expected_sha256) as digest:
                verify_dataset_inputs(cells)
            digest.assert_called_once_with(path, 4)

    def test_dirty_identity_covers_tracked_patch_and_untracked_content(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            subprocess.run(["git", "init", "-q", str(repo)], check=True)
            subprocess.run(["git", "-C", str(repo), "config", "user.email", "a@b"], check=True)
            subprocess.run(["git", "-C", str(repo), "config", "user.name", "test"], check=True)
            (repo / "tracked").write_text("one\n")
            subprocess.run(["git", "-C", str(repo), "add", "tracked"], check=True)
            subprocess.run(["git", "-C", str(repo), "commit", "-qm", "base"], check=True)
            clean, _ = capture_git_state(repo)
            self.assertFalse(clean["dirty"])
            self.assertEqual(len(clean["commit"]), 40)
            (repo / "tracked").write_text("two\n")
            (repo / "new").write_text("content\n")
            dirty, patch = capture_git_state(repo)
            self.assertTrue(dirty["dirty"])
            self.assertTrue(patch)
            self.assertEqual(dirty["untracked"][0]["path"], "new")
            self.assertNotEqual(clean["dirty_identity_sha256"],
                                dirty["dirty_identity_sha256"])

    def test_archives_are_content_addressed_and_site_secrets_redacted(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = ResultStore(Path(tmp), "session")
            first = store.archive_bytes("experiment", b"name: x\n", ".yaml")
            second = store.archive_bytes("experiment", b"name: x\n", ".yaml")
            self.assertEqual(first, second)
            self.assertTrue((store.dir / first["path"]).is_file())
            site = Site("/bin/tool", Path("/results"),
                        {"api_token": "secret", "partition": "gpu"})
            sanitized = site.sanitized_manifest()
            self.assertEqual(sanitized["site_local"]["api_token"], "<redacted>")
            self.assertEqual(sanitized["site_local"]["partition"], "gpu")

    def test_provenance_id_binds_shard_and_schema_rejects_malformed_id(self):
        base = {"session_schema_version": 1, "session_kind": "native",
                "session_id": "s", "timestamp": "now", "shard": [0, 2],
                "gpu": {}, "host": {}, "scheduler": {}, "software": {},
                "harness": {}, "compressors": {}, "nvidia_smi": None,
                "provenance_schema_version": 1, "input_artifacts": {},
                "dataset_integrity": {}}
        first = assign_provenance_id(base)
        base["shard"] = [1, 2]
        self.assertNotEqual(first, assign_provenance_id(base))
        base["provenance_id"] = "bad"
        with self.assertRaises(SchemaError):
            dumps_session(base)

    def test_provenance_write_preserves_id_addressed_history(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = ResultStore(Path(tmp), "session", shard=(0, 2))
            manifest = {"session_schema_version": 1, "session_kind": "native",
                        "session_id": "session", "timestamp": "now", "shard": [0, 2],
                        "gpu": {}, "host": {}, "scheduler": {}, "software": {},
                        "harness": {}, "compressors": {}, "nvidia_smi": None,
                        "provenance_schema_version": 1, "input_artifacts": {},
                        "dataset_integrity": {}}
            manifest["provenance_id"] = assign_provenance_id(manifest)
            store.write_provenance(manifest)
            addressed = store.dir / f"provenance.{manifest['provenance_id']}.json"
            self.assertTrue(addressed.is_file())
            self.assertTrue((store.dir / "provenance.shard-0-of-2.json").is_file())


if __name__ == "__main__":
    unittest.main()
