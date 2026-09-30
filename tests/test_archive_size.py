import copy
import unittest

from benchkit.archive_size import (
    ArchiveSizeCorrectionError,
    correct_fzgm_archive_sizes,
)


def _row(**overrides):
    row = {
        "compressor": "fzgm",
        "variant": "pfpl",
        "pipeline": "configs/pipelines/pfpl.toml",
        "status": "ok",
        "compressed_bytes": 640,
        "original_bytes": 4096,
        "num_elements": 1024,
        "cr": 6.4,
        "bitrate_bits_per_elem": 5.0,
    }
    row.update(overrides)
    return row


class ArchiveSizeCorrectionTests(unittest.TestCase):
    def test_correction_adds_audited_header_and_recomputes_size_metrics(self):
        source = _row()
        original = copy.deepcopy(source)

        [row] = correct_fzgm_archive_sizes([source])

        self.assertEqual(source, original)
        self.assertIsNot(row, source)
        self.assertEqual(row["compressed_payload_bytes"], 640)
        self.assertEqual(row["compressed_archive_overhead_bytes"], 1360)
        self.assertEqual(row["compressed_bytes"], 2000)
        self.assertEqual(row["cr"], 4096 / 2000)
        self.assertEqual(row["bitrate_bits_per_elem"], 2000 * 8 / 1024)

    def test_already_corrected_row_is_idempotent(self):
        corrected = _row(
            compressed_bytes=2000,
            compressed_payload_bytes=640,
            compressed_archive_overhead_bytes=1360,
            cr=4096 / 2000,
            bitrate_bits_per_elem=2000 * 8 / 1024,
        )

        [result] = correct_fzgm_archive_sizes([corrected])

        self.assertEqual(result, corrected)

    def test_unknown_pipeline_or_variant_fails_closed(self):
        for row in (
            _row(pipeline="configs/pipelines/new_pipeline.toml"),
            _row(variant="new_variant"),
        ):
            with self.subTest(row=row), self.assertRaises(ArchiveSizeCorrectionError):
                correct_fzgm_archive_sizes([row])

    def test_inconsistent_existing_overlay_fails_closed(self):
        row = _row(
            compressed_bytes=2001,
            compressed_payload_bytes=640,
            compressed_archive_overhead_bytes=1360,
        )

        with self.assertRaises(ArchiveSizeCorrectionError):
            correct_fzgm_archive_sizes([row])

    def test_fixed_cuszp3_dimension_topologies_have_distinct_audited_sizes(self):
        expected = {
            "configs/pipelines/cuszp3_fixed.toml": 848,
            "configs/pipelines/cuszp3_2d_fixed.toml": 1104,
            "configs/pipelines/cuszp3_3d_fixed.toml": 1104,
        }
        for pipeline, overhead in expected.items():
            with self.subTest(pipeline=pipeline):
                [row] = correct_fzgm_archive_sizes([_row(
                    variant="cuszp3_fixed", pipeline=pipeline,
                )])
                self.assertEqual(row["compressed_archive_overhead_bytes"], overhead)

    def test_validated_fsz_composition_and_blocksize_selectors(self):
        selectors = {
            ("fsz", "configs/pipelines/fsz.toml"): 1104,
            ("fzgm-fsz-bpt8", "configs/pipelines/fsz.toml"): 1104,
            ("x_lq_pfpl_ans", "configs/pipelines/x_lorenzoquant_pfpl_ans.toml"): 1872,
            ("lorenzo_ab_b128_plain", "configs/pipelines/lorenzo_ab_b128_plain.toml"): 1104,
            ("lorenzo_ab_b128_outlier", "configs/pipelines/lorenzo_ab_b128_outlier.toml"): 1104,
            ("lorenzo_ab_b32_plain", "configs/pipelines/lorenzo_ab_b32_plain.toml"): 1104,
            ("lorenzo_ab_b32_outlier", "configs/pipelines/lorenzo_ab_b32_outlier.toml"): 1104,
            ("lorenzo_ab_b64_plain", "configs/pipelines/lorenzo_ab_b64_plain.toml"): 1104,
            ("lorenzo_ab_b64_outlier", "configs/pipelines/lorenzo_ab_b64_outlier.toml"): 1104,
            ("lorenzo_ab_b96_plain", "configs/pipelines/lorenzo_ab_b96_plain.toml"): 1104,
            ("lorenzo_ab_b96_outlier", "configs/pipelines/lorenzo_ab_b96_outlier.toml"): 1104,
        }
        for (variant, pipeline), overhead in selectors.items():
            with self.subTest(variant=variant, pipeline=pipeline):
                [row] = correct_fzgm_archive_sizes([_row(variant=variant, pipeline=pipeline)])
                self.assertEqual(row["compressed_archive_overhead_bytes"], overhead)

    def test_validated_b1_selector_aliases(self):
        selectors = {
            ("cuszhi_cr_b1", "configs/pipelines/cusz_hi_cr_b1.toml"): 1872,
            ("cuszhi_cr_b1_singlecall", "configs/pipelines/cusz_hi_cr_b1.toml"): 1872,
            ("cuszhi_tp_b1", "configs/pipelines/cusz_hi_tp_b1.toml"): 2640,
            ("cuszp3_fixed_b1", "configs/pipelines/cuszp3_2d_fixed.toml"): 1104,
            ("cuszp3_fixed_b1", "configs/pipelines/cuszp3_3d_fixed.toml"): 1104,
        }
        for (variant, pipeline), overhead in selectors.items():
            with self.subTest(variant=variant, pipeline=pipeline):
                [row] = correct_fzgm_archive_sizes([_row(variant=variant, pipeline=pipeline)])
                self.assertEqual(row["compressed_archive_overhead_bytes"], overhead)

    def test_non_fzgm_row_is_copied_without_changes(self):
        source = _row(compressor="pfpl")
        [row] = correct_fzgm_archive_sizes([source])

        self.assertEqual(row, source)
        self.assertIsNot(row, source)


if __name__ == "__main__":
    unittest.main()
