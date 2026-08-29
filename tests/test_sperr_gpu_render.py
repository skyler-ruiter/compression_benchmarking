"""FZGM's GPU SPERR pipeline needs a dual-basis eb render fix: Cdf97OutlierCorrect
has no error_bound_mode key, so it always treats its error_bound as a literal
absolute value, unlike its mode-aware Quantizer sibling. See
benchkit/pipelines.py's PipelineToml.has_mode_agnostic_bound_stage() and
benchkit/adapters/fzgm.py's FzgmAdapter._prepare_toml for the fix, and
docs/RUN_LEDGER.md sec.3.5 / row EBLC-SPERR-GPU for the history.
"""
import struct
import tempfile
from pathlib import Path
from unittest import TestCase

from benchkit.adapters.base import RunSpec
from benchkit.adapters.fzgm import FzgmAdapter
from benchkit.config import FieldSpec
from benchkit.pipelines import PipelineToml

SPERR_GPU_TOML = Path(__file__).resolve().parent.parent / "configs/pipelines/sperr_gpu.toml"


class Cdf97OutlierCorrectDualBasisDetectionTests(TestCase):
    def test_sperr_gpu_toml_is_detected_as_mode_agnostic(self):
        tpl = PipelineToml.load(SPERR_GPU_TOML)
        self.assertTrue(tpl.has_mode_agnostic_bound_stage())

    def test_single_mode_aware_stage_pipeline_is_not_flagged(self):
        # A plain Quantizer-only lossy pipeline (no Cdf97OutlierCorrect-style
        # sibling) must not trip the dual-basis path.
        import benchkit.pipelines as pipelines_mod
        tpl = pipelines_mod.PipelineToml(
            path=Path("synthetic.toml"),
            text='[[stage]]\ntype = "Quantizer"\nerror_bound = 1e-4\n'
                 'error_bound_mode = "ABS"\n',
            doc={"stage": [{"type": "Quantizer", "error_bound": 1e-4,
                            "error_bound_mode": "ABS"}]},
        )
        self.assertFalse(tpl.has_mode_agnostic_bound_stage())


class FzgmAdapterSperrGpuRenderTests(TestCase):
    """Verifies the adapter pre-converts rel_range/rel_maxabs to an absolute
    error_bound for BOTH lossy stages before rendering, rather than pushing the
    literal requested eb (and the requested mode) onto Cdf97OutlierCorrect, which
    would silently desync it from Quantizer's NOA-rescaled bound."""

    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmpdir.cleanup)
        workdir = Path(self.tmpdir.name)

        # 100 f32 values, range [-2.0, 8.0] -> range=10.0, maxabs=8.0.
        data_path = workdir / "field.f32"
        values = [-2.0 + 10.0 * i / 99 for i in range(100)]
        data_path.write_bytes(struct.pack(f"<{len(values)}f", *values))

        self.field = FieldSpec(dataset="synthetic", field="ramp", dtype="f32",
                                dim_order="fast-to-slow", dims=[100],
                                path=data_path)
        self.workdir = workdir

    def _rendered_error_bound_lines(self, spec: RunSpec) -> list[str]:
        adapter = FzgmAdapter.__new__(FzgmAdapter)  # skip resolve_cli()
        prep = adapter._prepare_toml(spec, self.workdir)
        return [ln.strip() for ln in prep.pipeline_path.read_text().splitlines()
                if ln.strip().startswith("error_bound = ")]

    def test_rel_range_renders_the_same_converted_absolute_bound_on_both_stages(self):
        spec = RunSpec(field=self.field, error_mode="rel_range", error_bound=1e-3,
                        pipeline=str(SPERR_GPU_TOML), variant="fzgm")
        lines = self._rendered_error_bound_lines(spec)
        self.assertEqual(len(lines), 2)
        self.assertEqual(lines[0], lines[1])
        # eb * range = 1e-3 * 10.0 = 0.01, not the raw requested 1e-3.
        self.assertAlmostEqual(float(lines[0].split("=")[1]), 0.01)

    def test_rel_maxabs_converts_by_maxabs_not_range(self):
        spec = RunSpec(field=self.field, error_mode="rel_maxabs", error_bound=1e-3,
                        pipeline=str(SPERR_GPU_TOML), variant="fzgm")
        lines = self._rendered_error_bound_lines(spec)
        self.assertEqual(lines[0], lines[1])
        # eb * maxabs = 1e-3 * 8.0 = 0.008.
        self.assertAlmostEqual(float(lines[0].split("=")[1]), 0.008)

    def test_abs_mode_renders_the_requested_bound_verbatim_unconverted(self):
        spec = RunSpec(field=self.field, error_mode="abs", error_bound=1e-4,
                        pipeline=str(SPERR_GPU_TOML), variant="fzgm")
        lines = self._rendered_error_bound_lines(spec)
        self.assertEqual(lines[0], lines[1])
        self.assertAlmostEqual(float(lines[0].split("=")[1]), 1e-4)

    def test_harness_eb_basis_stays_the_original_request_not_the_converted_value(self):
        # prep.eb/prep.basis feed the harness's OWN independent eb_ok check
        # (metrics.py recomputes eb_abs = eb * basis_val itself) -- they must
        # reflect what was requested, not what got rendered into the TOML.
        spec = RunSpec(field=self.field, error_mode="rel_range", error_bound=1e-3,
                        pipeline=str(SPERR_GPU_TOML), variant="fzgm")
        adapter = FzgmAdapter.__new__(FzgmAdapter)
        prep = adapter._prepare_toml(spec, self.workdir)
        self.assertEqual(prep.eb, 1e-3)
        self.assertEqual(prep.basis, "range")
        self.assertIn("rel_range", prep.native_mode)
