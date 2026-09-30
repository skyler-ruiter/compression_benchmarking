from pathlib import Path
from subprocess import CompletedProcess
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import patch

from benchkit.adapters.base import AdapterError, Prepared
from benchkit.adapters.fzgm import FzgmAdapter
from benchkit.runner import _add_fzgm_size_breakdown


class FzgmCompressedSizeTests(TestCase):
    def _benchmark_report(self):
        return {
            "size": {"original_bytes": 16, "compressed_bytes": 7},
            "timing": {
                "compress": {"device_ms": {"all": [1.0]},
                             "host_wall_ms": {"all": [1.1]}},
                "decompress": {"device_ms": {"all": [2.0]},
                               "host_wall_ms": {"all": [2.1]}},
            },
        }

    def _benchmark_args(self, workdir):
        source = workdir / "input.bin"
        source.write_bytes(b"input-data")
        adapter = object.__new__(FzgmAdapter)
        adapter.cli = "/fake/fzgmod-cli"
        spec = SimpleNamespace(field=SimpleNamespace(
            path=source, dtype="f32", dim_arg="4", original_bytes=16), graph=False)
        prep = Prepared(
            config_args=[], eb=1e-3, native_mode="NOA", basis="range",
            pipeline_ref="fixture", pipeline_path=None, pipeline_sha256=None)
        return adapter, spec, prep

    def test_compress_counts_complete_archive_and_keeps_payload_report(self):
        with TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            source = workdir / "input.bin"
            source.write_bytes(b"input-data")
            adapter = object.__new__(FzgmAdapter)
            adapter.cli = "/fake/fzgmod-cli"
            spec = SimpleNamespace(field=SimpleNamespace(
                path=source, dtype="f32", dim_arg="4", original_bytes=16))
            prep = Prepared(
                config_args=[], eb=1e-3, native_mode="NOA", basis="range",
                pipeline_ref="fixture", pipeline_path=None, pipeline_sha256=None)
            report = {"size": {"original_bytes": 16, "compressed_bytes": 7}}

            def fake_run_cli(argv, log):
                Path(argv[argv.index("-o") + 1]).write_bytes(b"header+payload")
                return CompletedProcess(argv, 0)

            with patch("benchkit.adapters.fzgm.run_cli", side_effect=fake_run_cli), \
                    patch("benchkit.adapters.fzgm.load_report_json", return_value=report):
                result = adapter.compress(spec, prep, workdir)

            self.assertEqual(result.compressed_bytes, len(b"header+payload"))
            self.assertEqual(result.raw_json["size"]["compressed_bytes"], 7)

    def test_benchmark_reports_complete_archive_and_preserves_payload_report(self):
        with TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            archive = workdir / "c.fzm"
            archive.write_bytes(b"header+payload")
            adapter, spec, prep = self._benchmark_args(workdir)
            with patch("benchkit.adapters.fzgm.run_cli",
                       return_value=CompletedProcess([], 0)), \
                    patch("benchkit.adapters.fzgm.load_report_json",
                          return_value=self._benchmark_report()):
                result = adapter.benchmark(spec, prep, 1, workdir)

            self.assertEqual(result.compressed_bytes, archive.stat().st_size)
            self.assertEqual(result.raw_json["size"]["compressed_bytes"], 7)

    def test_benchmark_fails_if_complete_archive_is_missing(self):
        with TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            adapter, spec, prep = self._benchmark_args(workdir)
            with patch("benchkit.adapters.fzgm.run_cli",
                       return_value=CompletedProcess([], 0)), \
                    patch("benchkit.adapters.fzgm.load_report_json",
                          return_value=self._benchmark_report()):
                with self.assertRaisesRegex(AdapterError, "cannot report complete archive size"):
                    adapter.benchmark(spec, prep, 1, workdir)

    def test_result_row_keeps_payload_and_archive_overhead_for_fzgm(self):
        row = {"compressed_bytes": 14}
        entry = SimpleNamespace(compressor="fzgm")
        comp = SimpleNamespace(
            compressed_bytes=14,
            raw_json={"size": {"compressed_bytes": 7}},
        )

        _add_fzgm_size_breakdown(row, entry, comp)

        self.assertEqual(row["compressed_payload_bytes"], 7)
        self.assertEqual(row["compressed_archive_overhead_bytes"], 7)

    def test_result_row_size_breakdown_is_absent_for_native_adapters(self):
        row = {"compressed_bytes": 14}
        entry = SimpleNamespace(compressor="pfpl")
        comp = SimpleNamespace(compressed_bytes=14, raw_json={})

        _add_fzgm_size_breakdown(row, entry, comp)

        self.assertNotIn("compressed_payload_bytes", row)
        self.assertNotIn("compressed_archive_overhead_bytes", row)
