from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

import numpy as np

from benchkit.adapters.base import RunSpec
from benchkit.adapters.mans import MansAdapter
from benchkit.adapters.quantized_integer import dequantize, quantize
from benchkit.config import FieldSpec


class QuantizedIntegerWrapperTests(TestCase):
    def test_codec_padding_is_zero_and_not_dequantized(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            src = np.array([1.0, 1.25, 1.5], dtype=np.float32)
            src.tofile(root / "input.bin")
            meta = {
                "original_dtype": "f32",
                "integer_dtype": "u16",
                "num_elements": 3,
                "codec_num_elements": 8,
                "offset": 1.0,
                "step": 0.25,
                "constant": False,
            }

            quantize(root / "input.bin", root / "q.bin", meta)
            q = np.fromfile(root / "q.bin", dtype=np.uint16)
            self.assertEqual(q.tolist(), [0, 1, 2, 0, 0, 0, 0, 0])

            dequantize(root / "q.bin", root / "output.bin", meta)
            out = np.fromfile(root / "output.bin", dtype=np.float32)
            self.assertEqual(out.tolist(), src.tolist())

    def test_mans_forces_the_verified_u32_path(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            np.array([0.0, 1.0], dtype=np.float32).tofile(root / "input.bin")
            field = FieldSpec(
                dataset="test", field="x", dtype="f32",
                dim_order="fast-to-slow", dims=[2], path=root / "input.bin")
            spec = RunSpec(field=field, error_mode="abs", error_bound=1e-3,
                           pipeline="default", variant="mans")
            adapter = object.__new__(MansAdapter)

            adapter.prepare(spec, root / "work")
            import json
            meta = json.loads((root / "work" / "quantization.json").read_text())
            self.assertEqual(meta["integer_dtype"], "u32")
            self.assertIn("not bit-exact", meta["integer_dtype_reason"])
