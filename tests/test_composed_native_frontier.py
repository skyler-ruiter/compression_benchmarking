import unittest

from scripts.analyze_composed_native_frontier import (
    NATIVE_VARIANTS,
    data_signature,
    select_composed_rows,
)


class ComposedNativeFrontierTests(unittest.TestCase):
    def test_paired_session_selection_keeps_only_requested_composition(self):
        rows = [
            {"compressor": "fzgm", "variant": "x_lq_pfpl_ans"},
            {"compressor": "fzgm", "variant": "x_lq_pfpl"},
            {"compressor": "cusz", "variant": "cusz"},
        ]
        self.assertEqual(select_composed_rows(rows, "x_lq_pfpl_ans"), [rows[0]])

    def test_publication_native_set_excludes_fzgpu(self):
        self.assertNotIn(("fzgpu", "fzgpu"), NATIVE_VARIANTS)
        self.assertEqual(len(NATIVE_VARIANTS), 9)

    def test_data_signature_can_require_archived_dataset_hash(self):
        row = {
            "dims": [4, 5, 6],
            "dtype": "f32",
            "original_bytes": 480,
            "dataset_sha256": "abc",
        }
        self.assertEqual(data_signature(row), ((4, 5, 6), "f32", 480))
        self.assertEqual(
            data_signature(row, include_hash=True),
            ((4, 5, 6), "f32", 480, "abc"),
        )


if __name__ == "__main__":
    unittest.main()
