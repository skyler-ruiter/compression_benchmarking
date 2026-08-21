# FZGM case studies

This directory holds application-facing studies built on top of the reusable
Benchkit datasets, pipelines, and experiment runner.  A case study may add
domain-specific exploration and quality metrics, but benchmark executions still
belong in `configs/experiments/` and raw run output still belongs outside the
repository under the configured results root.

Current studies:

- [`combustion/`](combustion/) — S3D statistically planar premixed-flame pilot.

