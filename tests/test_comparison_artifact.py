from unittest import TestCase

from scripts.build_comparison_artifact import build_data, index_rows


def _row(field, compressor, cr):
    return {
        "status": "ok",
        "variant": "cusz",
        "compressor": compressor,
        "dataset": "CESM-2D",
        "field": field,
        "error_bound": 1e-3,
        "cr": cr,
        "psnr": 80.0,
        "compress_throughput_gbs": 10.0,
        "decompress_throughput_gbs": 20.0,
        "eb_satisfied": True,
        "timing_reliable": True,
    }


class ComparisonArtifactFieldKeyTests(TestCase):
    def test_index_keeps_fields_from_the_same_dataset_distinct(self):
        rows = [_row("CLDHGH", "cusz", 2.0), _row("CLDLOW", "cusz", 3.0)]

        indexed = index_rows(rows)

        self.assertEqual(len(indexed), 2)
        self.assertEqual(
            indexed[("cusz", "CESM-2D", "CLDHGH", "0.001")]["native"]["cr"],
            2.0,
        )
        self.assertEqual(
            indexed[("cusz", "CESM-2D", "CLDLOW", "0.001")]["native"]["cr"],
            3.0,
        )

    def test_payload_renders_dataset_and_field_labels(self):
        rows_a = [
            _row("CLDHGH", "cusz", 2.0),
            _row("CLDHGH", "fzgm", 2.1),
            _row("CLDLOW", "cusz", 3.0),
            _row("CLDLOW", "fzgm", 3.1),
        ]
        rows_b = [
            _row("CLDHGH", "cusz", 2.0),
            _row("CLDHGH", "fzgm", 2.1),
            _row("CLDLOW", "cusz", 3.0),
            _row("CLDLOW", "fzgm", 3.1),
        ]
        base_a = {"rows": rows_a, "label": "A", "dir": "a", "meta": {}}
        base_b = {"rows": rows_b, "label": "B", "dir": "b", "meta": {}}

        payload, anomalies = build_data(base_a, base_b)

        self.assertEqual(
            payload["meta"]["datasets"],
            ["CESM-2D/CLDHGH", "CESM-2D/CLDLOW"],
        )
        self.assertEqual(payload["meta"]["matched_cells"], 2)
        self.assertEqual(
            payload["data"]["cusz"]["CESM-2D/CLDHGH"]["0.001"]
            ["native"]["a"]["cr"],
            2.0,
        )
        self.assertEqual(anomalies, [])

    def test_pairing_refuses_different_explicit_logical_cells(self):
        row_a = _row("CLDHGH", "cusz", 2.0)
        row_b = _row("CLDHGH", "cusz", 2.0)
        row_a["logical_cell_id"] = "logical-v1-a"
        row_b["logical_cell_id"] = "logical-v1-b"
        base_a = {"rows": [row_a], "label": "A", "dir": "a", "meta": {}}
        base_b = {"rows": [row_b], "label": "B", "dir": "b", "meta": {}}

        with self.assertRaises(ValueError):
            build_data(base_a, base_b)

    def test_default_omits_gated_rows_and_withholds_unreliable_throughput(self):
        good = _row("CLDHGH", "cusz", 2.0)
        unreliable = _row("CLDLOW", "cusz", 3.0)
        unreliable["timing_reliable"] = False
        invalid = _row("TS", "cusz", 4.0)
        invalid["eb_satisfied"] = False
        invalid["err_over_bound"] = 2.0

        indexed = index_rows([good, unreliable, invalid])

        self.assertEqual(set(indexed), {
            ("cusz", "CESM-2D", "CLDHGH", "0.001"),
            ("cusz", "CESM-2D", "CLDLOW", "0.001"),
        })
        self.assertIsNone(
            indexed[("cusz", "CESM-2D", "CLDLOW", "0.001")]["native"]["cgbs"]
        )
        self.assertTrue(
            indexed[("cusz", "CESM-2D", "CLDHGH", "0.001")]["native"]["tok"]
        )

    def test_variant_can_be_explicitly_excluded_for_incompatible_baselines(self):
        fixed = _row("CLDHGH", "cusz", 2.0)
        fixed["variant"] = "cuszp3_fixed"
        comparable = _row("CLDLOW", "cusz", 3.0)

        indexed = index_rows([fixed, comparable], excluded_variants=["cuszp3_fixed"])

        self.assertEqual(set(indexed), {("cusz", "CESM-2D", "CLDLOW", "0.001")})
