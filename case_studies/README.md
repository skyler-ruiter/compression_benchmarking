# FZGM case studies

This directory holds application-facing studies built on top of the reusable
Benchkit datasets, pipelines, and experiment runner.  A case study may add
domain-specific exploration and quality metrics, but benchmark executions still
belong in `configs/experiments/` and raw run output still belongs outside the
repository under the configured results root.

Current studies:

- [`combustion/`](combustion/) — S3D statistically planar premixed-flame pilot.
- [`roibin/`](roibin/) — ROIBIN-SZ-style dual-error-bound (region-of-interest)
  pipeline for serial-crystallography frames (experiments.md E1 / F1 study 1).
  Previously a standalone tree under `~/compressors/ROIBIN`; its prior git history
  is preserved as `roibin/roibin-prior-history.gitbundle`. Note its `results/*.csv`
  are committed here (small, curated) — unlike the repo-wide results-root rule.

