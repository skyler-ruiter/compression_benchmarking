from pathlib import Path
from subprocess import CompletedProcess
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import patch

from benchkit.adapters.base import AdapterError, Prepared
from benchkit.adapters.fsz import FszAdapter


class FszCompressExitCodeTests(TestCase):
    def setUp(self):
        self.adapter = object.__new__(FszAdapter)
        self.adapter.cli = "/fake/fsz"
        self.spec = SimpleNamespace(field=SimpleNamespace(original_bytes=4096))
        self.prep = Prepared(
            config_args=["-i", "/fake/input", "-eb", "rel", "1e-6"],
            eb=1e-6,
            native_mode="rel",
            basis="range",
            pipeline_ref="fsz:default",
            pipeline_path=None,
            pipeline_sha256=None,
        )

    def test_exit_one_with_artifact_is_a_completed_result(self):
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as tmp:
            workdir = Path(tmp)

            def completed_with_quality_miss(argv, log):
                Path(argv[argv.index("-o") + 1]).write_bytes(b"artifact")
                return CompletedProcess(argv, 1)

            with patch("benchkit.adapters.fsz.run_cli",
                       side_effect=completed_with_quality_miss):
                result = self.adapter.compress(self.spec, self.prep, workdir)

            self.assertEqual(result.compressed_bytes, len(b"artifact"))
            self.assertEqual(result.raw_json["native_exit_code"], 1)

    def test_exit_two_is_a_harness_failure(self):
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as tmp:
            with patch("benchkit.adapters.fsz.run_cli",
                       return_value=CompletedProcess([], 2)):
                with self.assertRaises(AdapterError):
                    self.adapter.compress(self.spec, self.prep, Path(tmp))
