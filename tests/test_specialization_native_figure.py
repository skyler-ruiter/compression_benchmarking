import unittest

from scripts.build_specialization_native_figure import build


def row(compressor, variant, throughput, *, reliable=True, forward=0, inverse=0,
        dataset="D", field="f", error_bound=1e-3):
    return {
        "compressor": compressor,
        "variant": variant,
        "dataset": dataset,
        "field": field,
        "dtype": "f32",
        "dims": [16],
        "dim_order": "fast-to-slow",
        "error_mode": "rel_range",
        "error_bound": error_bound,
        "dataset_sha256": "a" * 64,
        "eb_abs_effective": 0.25,
        "status": "ok",
        "timing_reliable": reliable,
        "compress_throughput_gbs": throughput,
        "decompress_throughput_gbs": throughput,
        "fusion_installed_group_count": forward,
        "fusion_inverse_installed_group_count": inverse,
        "_exclusions": [],
    }


class SpecializationNativeFigureTests(unittest.TestCase):
    def test_uses_one_geometric_native_baseline_for_both_arms(self):
        off = [row("cusz", "cusz", 4.0), row("fzgm", "cusz", 3.0)]
        auto = [row("cusz", "cusz", 9.0), row("fzgm", "cusz", 12.0)]
        summary, detail = build(off, auto)
        compression = next(entry for entry in summary
                           if entry["family"] == "cusz" and
                           entry["phase"] == "compression")
        point = next(entry for entry in detail
                     if entry["family"] == "cusz" and
                     entry["phase"] == "compression")
        self.assertEqual(compression["pairs"], 1)
        self.assertAlmostEqual(point["staged_over_native"], 0.5)
        self.assertAlmostEqual(point["auto_over_native"], 2.0)
        self.assertAlmostEqual(point["auto_over_staged"], 4.0)
        self.assertAlmostEqual(point["native_auto_over_off"], 2.25)

    def test_requires_reliable_timing_in_all_four_rows(self):
        off = [row("cusz", "cusz", 4.0), row("fzgm", "cusz", 3.0)]
        auto = [row("cusz", "cusz", 9.0, reliable=False),
                row("fzgm", "cusz", 12.0)]
        summary, detail = build(off, auto)
        compression = next(entry for entry in summary
                           if entry["family"] == "cusz" and
                           entry["phase"] == "compression")
        self.assertEqual(compression["pairs"], 0)
        self.assertFalse(any(entry["family"] == "cusz" for entry in detail))

    def test_records_phase_specific_specialization_installation(self):
        off = [row("cusz", "cusz", 4.0), row("fzgm", "cusz", 3.0)]
        auto = [row("cusz", "cusz", 4.0),
                row("fzgm", "cusz", 6.0, forward=1, inverse=0)]
        summary, detail = build(off, auto)
        compression = next(entry for entry in summary
                           if entry["family"] == "cusz" and
                           entry["phase"] == "compression")
        decompression = next(entry for entry in summary
                             if entry["family"] == "cusz" and
                             entry["phase"] == "decompression")
        self.assertEqual(compression["auto_specialization_installed_fraction"], 1.0)
        self.assertEqual(decompression["auto_specialization_installed_fraction"], 0.0)
        self.assertTrue(next(entry for entry in detail
                             if entry["family"] == "cusz" and
                             entry["phase"] == "compression")
                        ["auto_specialization_installed"])

    def test_summary_mean_weights_bounds_fields_and_datasets_hierarchically(self):
        off = []
        auto = []
        coordinates = [
            ("one-field", "f", 1e-2, 16.0),
            ("three-fields", "a", 1e-2, 1.0),
            ("three-fields", "b", 1e-2, 1.0),
            ("three-fields", "c", 1e-2, 1.0),
        ]
        for dataset, field, bound, ratio in coordinates:
            off += [
                row("cusz", "cusz", 1.0, dataset=dataset, field=field,
                    error_bound=bound),
                row("fzgm", "cusz", ratio, dataset=dataset, field=field,
                    error_bound=bound),
            ]
            auto += [
                row("cusz", "cusz", 1.0, dataset=dataset, field=field,
                    error_bound=bound),
                row("fzgm", "cusz", ratio, dataset=dataset, field=field,
                    error_bound=bound),
            ]

        summary, _ = build(off, auto)
        compression = next(entry for entry in summary
                           if entry["family"] == "cusz" and
                           entry["phase"] == "compression")
        # Dataset means are 16 and 1, whose equal-weight geometric mean is 4.
        # A coordinate-weighted mean would instead be 2.
        self.assertAlmostEqual(compression["staged_over_native_gmean"], 4.0)
        self.assertEqual(compression["datasets"], 2)
        self.assertEqual(compression["fields"], 4)
        self.assertEqual(compression["pairs"], 4)


if __name__ == "__main__":
    unittest.main()
