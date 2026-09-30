import tempfile
import unittest
from pathlib import Path

from scripts.analyze_composed_native_frontier import (
    apply_archive_size_correction as apply_frontier_correction,
    point,
)
from scripts.analyze_specialization_blocksize import (
    apply_archive_size_correction as apply_blocksize_correction,
)


def correction_config(directory: Path) -> Path:
    path = directory / "archive_sizes.yaml"
    path.write_text(
        "schema_version: 1\n"
        "selectors:\n"
        "  demo|pipeline.toml: {header_bytes: 1104}\n"
    )
    return path


def fzgm_row():
    return {
        "compressor": "fzgm",
        "variant": "demo",
        "pipeline": "pipeline.toml",
        "status": "ok",
        "compressed_bytes": 896,
        "original_bytes": 8000,
        "num_elements": 1000,
        "compress_throughput_gbs": 1.0,
        "psnr": 50.0,
    }


class ArchiveSizeConsumerTests(unittest.TestCase):
    def test_frontier_consumer_recomputes_cr_without_mutating_source(self):
        source = fzgm_row()
        with tempfile.TemporaryDirectory() as directory:
            corrected = apply_frontier_correction(
                [source], correction_config(Path(directory))
            )

        self.assertEqual(source["compressed_bytes"], 896)
        self.assertEqual(corrected[0]["compressed_payload_bytes"], 896)
        self.assertEqual(corrected[0]["compressed_archive_overhead_bytes"], 1104)
        self.assertEqual(corrected[0]["compressed_bytes"], 2000)
        self.assertEqual(corrected[0]["cr"], 4.0)
        self.assertEqual(point(corrected[0])["cr"], 4.0)

    def test_blocksize_consumer_corrects_both_arms_and_leaves_native_rows_alone(self):
        staged = fzgm_row()
        auto = fzgm_row()
        native = {"compressor": "cusz", "compressed_bytes": 10}
        with tempfile.TemporaryDirectory() as directory:
            config = correction_config(Path(directory))
            corrected_staged = apply_blocksize_correction([staged, native], config)
            corrected_auto = apply_blocksize_correction([auto], config)

        self.assertEqual(corrected_staged[0]["compressed_bytes"], 2000)
        self.assertEqual(corrected_auto[0]["compressed_bytes"], 2000)
        self.assertEqual(corrected_staged[1], native)
        self.assertEqual(staged["compressed_bytes"], 896)
        self.assertEqual(auto["compressed_bytes"], 896)


if __name__ == "__main__":
    unittest.main()
