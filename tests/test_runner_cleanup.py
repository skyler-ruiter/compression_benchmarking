import tempfile
import unittest
from pathlib import Path

from benchkit.runner import _cleanup_cell_artifacts


class RunnerCleanupTests(unittest.TestCase):
    def test_removes_all_cell_data_but_preserves_diagnostics(self):
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            for name in ("c.sz3", "d.bin", "d_bench.bin", "compressed.dat"):
                (workdir / name).write_bytes(b"regenerable data")
            for name in ("compress.log", "report.json", "pipeline.toml"):
                (workdir / name).write_text("diagnostic")

            _cleanup_cell_artifacts(workdir)

            self.assertEqual(
                sorted(path.name for path in workdir.iterdir()),
                ["compress.log", "pipeline.toml", "report.json"],
            )


if __name__ == "__main__":
    unittest.main()
