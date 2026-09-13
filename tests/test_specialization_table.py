import unittest

from scripts.build_specialization_table import hierarchical_speedup_gmean


def pair(dataset, field, bound, speedup):
    base = {
        "dataset": dataset,
        "field": field,
        "dtype": "f32",
        "dims": [16],
        "dim_order": "fast-to-slow",
        "error_mode": "rel_range",
        "error_bound": bound,
        "compress_throughput_gbs": 1.0,
    }
    specialized = dict(base)
    specialized["compress_throughput_gbs"] = speedup
    return base, specialized


class SpecializationTableTests(unittest.TestCase):
    def test_speedup_mean_weights_datasets_equally(self):
        pairs = [
            pair("one-field", "f", 1e-2, 16.0),
            pair("three-fields", "a", 1e-2, 1.0),
            pair("three-fields", "b", 1e-2, 1.0),
            pair("three-fields", "c", 1e-2, 1.0),
        ]
        self.assertAlmostEqual(
            hierarchical_speedup_gmean(pairs, "compress_throughput_gbs"), 4.0
        )

    def test_speedup_mean_weights_bounds_within_fields(self):
        pairs = [
            pair("D", "two-bounds", 1e-2, 4.0),
            pair("D", "two-bounds", 1e-3, 16.0),
            pair("D", "one-bound", 1e-2, 1.0),
        ]
        # Field means are 8 and 1, so the dataset mean is sqrt(8).
        self.assertAlmostEqual(
            hierarchical_speedup_gmean(pairs, "compress_throughput_gbs"), 8 ** 0.5
        )


if __name__ == "__main__":
    unittest.main()
