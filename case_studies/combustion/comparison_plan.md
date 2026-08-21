# Staged comparison plan

## Stage 0: dataset and metric validation (current)

The first snapshot shows a clear wrinkled reaction front.  Temperature is almost
perfectly correlated with CO2 and H2O and anticorrelated with CH4 and O2 in the
common spatial sample.  That makes a shared detector plausible.  Pressure and N2
have extremely small dynamic ranges and should be treated separately: they are
useful error-bound stress tests but poor fields from which to infer a flame mask.

Before benchmarking, ask a combustion collaborator or dataset owner to confirm:

1. a reaction-progress definition for these six species;
2. the desired flame isovalue and brush width (physical distance or grid cells);
3. whether the reported `1e-4`/`1e-3` NRMSE values are field-normalized targets,
   mask-specific targets, or illustrative orders of magnitude;
4. the minimum downstream checks expected for a credible result.

## Stage 1: uniform-bound baseline

Start with `TEMP`, `CH4`, `O2`, `CO2`, `U`, and `V`; they span progress-like,
minor-species, and velocity behavior without paying for all eleven fields.

- Bounds: relative-to-range `1e-4`, `3e-4`, `1e-3`, `3e-3`, and `1e-2`.
- Compressors: all correct f64 FZGM pipelines, cuSZ-family native baselines that
  truly support f64, FZ-GPU if its f64 path is valid, and CPU SZ3/ZFP as quality
  references when available.
- Record: compression ratio, compression/decompression throughput, observed
  pointwise error, NRMSE, and device memory.
- Reject a cell rather than silently casting to f32.  Historical S3D numbers in
  the paper workspace are explicitly not valid baseline evidence.

This stage identifies the 2--10x feasible region and selects two or three Pareto
points for expensive QoI analysis.

### Stage 1b: use the general tight-bound robustness sweep

The `1e-6`/`1e-7` diagnostic is a corpus-wide benchmark, not a combustion
experiment. Its executable configs are
`configs/experiments/tight_bounds_full_gpu.yaml` and
`configs/experiments/tight_bounds_full_references.yaml`; filter their S3D rows
when combustion-specific failure evidence is useful. These are numerically
**lower/tighter** bounds intended to expose precision, quantization-range,
outlier-capacity, and metadata failures. They are not application-facing S3D
operating points and must not be folded into the Stage 1 QoI table.

Within the S3D subset, retain `N2` and `PRES`: their tiny ranges around large
offsets are valuable numerical stress cases even though they are poor flame
indicators. Preserve failed rows as robustness evidence; do not relax a bound
or silently drop a field merely to make the matrix complete.

## Stage 2: QoI preservation

For each selected reconstruction, compute:

- isosurface/flame-front displacement and area change;
- conditional means and PDFs of temperature/species versus signed distance;
- scalar-gradient magnitude and direction error;
- velocity gradient, vorticity, and strain-derived error;
- connected-component and topology summaries if the necessary tooling is
  available.

Report these inside the flame brush, outside it, and globally.  A global metric
alone can hide a thin-front failure.

## Stage 3: composed adaptive pipeline

Compare three policies at matched total compressed size and again at matched QoI:

1. one stringent bound everywhere;
2. one relaxed bound everywhere;
3. a validated flame mask with stringent inside / relaxed outside bounds.

For policy 3, account for mask/detector bytes and time.  Report the FZGM pipeline
unfused and with eligible finalize-time automatic fusion, plus the closest native
fused compressor.  The paper-worthy result is a new useful tradeoff or comparable
QoI at higher throughput—not merely that fusion beats an intentionally
materialized version of the same composition.

## Stage 4: temporal check

Repeat the reduced matrix on at least the five locally available S3D timesteps.
If the mask must be stored, test temporal reuse or delta coding; if it is derived,
test detector stability and cost.  One visually compelling snapshot is a pilot,
not an application case study.
