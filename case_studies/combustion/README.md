# Combustion case study: S3D planar premixed flame

## Purpose

This pilot asks whether an FZGM composition can provide a useful operating point
for reacting-flow data: high GPU throughput, modest useful compression, and
tighter fidelity in a flame region than elsewhere.  It starts with the SDRBench
S3D snapshot already registered as dataset `S3D`.

This is an exploration package, not yet evidence for a paper claim.  In
particular, the snapshot has temperature, pressure, velocity, and six species
mass fractions, but it does **not** provide a trusted flame-surface mask,
reaction-progress variable, mixture fraction, reaction rate, or coordinates.
The temperature-gradient mask emitted by the script is consequently a diagnostic
proxy only.

## Reproduce the initial survey

From the repository root:

```bash
.venv/bin/python case_studies/combustion/analyze_s3d.py \
  --data-root /media/volume/Compression_Data/sdrbench_data
```

The script uses memory maps and deterministic samples, so it does not load the
11 GB split snapshot into RAM.  It writes to `case_studies/combustion/artifacts/`:

- `field_summary.csv` and `field_summary.md`: sampled distributions and slice
  gradient statistics;
- `sample_correlations.csv`: cross-field Pearson correlations from common
  spatial samples;
- `figures/s3d_orthogonal_slices.png`: three mid-plane views of every field;
- `figures/correlation_matrix.png`: a visual cross-field correlation matrix;
- `figures/temp_gradient_proxy.png`: temperature slice, gradient magnitude, and
  a top-decile gradient mask (explicitly not a validated flame mask).

Use `--sample-count` to trade accuracy for analysis time and `--force` to replace
an existing artifact directory.

The orthogonal-slice montage has one row per field in this order: `CH4`, `O2`,
`CO`, `CO2`, `H2O`, `N2`, `TEMP`, `PRES`, `U`, `V`, `W`.  Its columns are the
central `XY`, `XZ`, and `YZ` planes.  Each scalar panel uses its own robust color
range; velocity panels use a zero-centered diverging range, so colors should not
be compared numerically across panels.

## Questions this dataset can answer now

1. Which fields and spatial regions dominate dynamic range and gradient content?
2. Do species and temperature share enough structure to motivate a common
   block-local detector or mask?
3. At NRMSE targets of roughly `1e-4` (stringent) and `1e-3` (relaxed), which
   native and FZGM pipelines reach useful 2--10x compression?
4. Does a composed detect/mask -> dual-error-bound -> encode pipeline preserve
   conditional statistics and gradients better at the same size and throughput?

## Evidence needed before a combustion claim

- Obtain or derive a domain-validated reaction-progress variable and flame
  surface.  Temperature gradient alone is not sufficient validation.
- Evaluate level-set displacement, conditional means/PDFs versus signed distance
  to the flame, scalar gradients, velocity/vorticity quantities, and preferably
  topology (merge trees or Morse--Smale summaries), not just PSNR.
- Use multiple timesteps spanning different flame states.  Five packed snapshots
  are locally available; only the first is currently split for Benchkit.
- Compare against native GPU error-bounded compressors and a strong CPU reference
  where appropriate.  Do not reuse the historical S3D ratios flagged as invalid
  in the FZGM paper notes.

See [`literature.md`](literature.md) for the requirements trace and nearby work.
See [`domain_primer.md`](domain_primer.md) for the combustion concepts and metric
definitions needed to interpret this dataset responsibly.
See [`comparison_plan.md`](comparison_plan.md) for the staged benchmark plan.
