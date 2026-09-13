from unittest import TestCase

from benchkit.adapters.base import AdapterError
from benchkit.adapters.cuszhi import _parse_pipeline
from benchkit.adapters.cuszp import _parse_speed, _speed_to_ms
from benchkit.adapters.fzgpu import _parse_all_times_s, _parse_psnr


class AdapterParserFixtureTests(TestCase):
    def test_cuszhi_bounded_native_tuning_syntax(self):
        self.assertEqual(_parse_pipeline("default"), ("cr", None))
        self.assertEqual(
            _parse_pipeline("tp;radius=64,auto_tuning=rd-first"),
            ("tp", "auto_tuning=rd-first,radius=64"),
        )
        self.assertEqual(
            _parse_pipeline("cr;huffchunk=1024"),
            ("cr", "huffchunk=1024"),
        )

    def test_cuszhi_rejects_semantically_invalid_tuning(self):
        for pipeline in (
            "tp;huffchunk=1024",
            "cr;radius=256",
            "cr;auto_tuning=fast",
            "cr;unknown=1",
        ):
            with self.subTest(pipeline=pipeline), self.assertRaises(AdapterError):
                _parse_pipeline(pipeline)

    def test_cuszp_spacing_and_scientific_notation(self):
        output = """
cuSZp compression    end-to-end speed: 1.25e+02 GB/s
cuSZp decompression end-to-end speed: 250.0 GB/s
"""
        self.assertEqual(_parse_speed(output, "compression"), 125.0)
        self.assertEqual(_parse_speed(output, "decompression"), 250.0)
        self.assertAlmostEqual(_speed_to_ms(125.0, 125 * 1024 * 1024), 1.0)

    def test_cuszp_rejects_unrecognized_output(self):
        with self.assertRaises(AdapterError):
            _parse_speed("compression completed", "compression")

    def test_fzgpu_repeated_timing_and_quality_fixture(self):
        output = """
compression e2e time: 1.5e-04 s
decompression e2e time: 0.00020 s
compression e2e time: 0.00030 s
decompression e2e time: 4e-04 s
PSNR: 72.125 dB
"""
        self.assertEqual(_parse_all_times_s(output, "compression"), [0.00015, 0.0003])
        self.assertEqual(_parse_all_times_s(output, "decompression"), [0.0002, 0.0004])
        self.assertEqual(_parse_psnr(output), 72.125)

    def test_fzgpu_rejects_missing_timing(self):
        with self.assertRaises(AdapterError):
            _parse_all_times_s("PSNR: 80", "compression")
