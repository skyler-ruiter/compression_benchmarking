from unittest import TestCase

from benchkit import validity


class TightBoundExpansionTests(TestCase):
    def test_quality_satisfying_expansion_is_retained(self):
        row = {"status": "ok", "dataset": "d", "field": "f",
               "error_mode": "rel_range", "cr": 0.8,
               "eb_satisfied": True, "psnr": 120.0}
        annotated = validity.annotate([row])[0]
        self.assertIn("expansion", annotated["_exclusions"])
        self.assertTrue(validity.is_valid(annotated))
        self.assertTrue(validity.quality_valid(annotated))

    def test_expansion_does_not_rescue_bound_violation(self):
        row = {"status": "ok", "dataset": "d", "field": "f",
               "error_mode": "rel_range", "cr": 0.8,
               "eb_satisfied": False, "err_over_bound": 20.0, "psnr": 80.0}
        annotated = validity.annotate([row])[0]
        self.assertIn("eb_violated_severe", annotated["_exclusions"])
        self.assertFalse(validity.is_valid(annotated))
