# Combustion compression: requirements and nearby work

Checked 2026-08-21.  This is a working trace, not a systematic review.

## Sarasota requirements confirmed

Primary source: Cappello et al., [*Lossy Compression of Scientific Data:
Applications Constrains and Requirements*](https://arxiv.org/html/2503.20031v1),
NSF FZ Project Sarasota Workshop Report (2025), especially Section II-B.

- The two named use cases are S3D simulation output (checkpointing and in situ
  event capture) and BlastNet (ML/super-resolution).
- S3D output is already on GPUs, so compression for checkpoints and on-the-fly
  snapshots calls for high GPU throughput.
- Requested analyses span composition and physical space: concentrations,
  temperature, progress/mixture variables, scalar dissipation, velocity,
  vorticity, gradients, reaction rates, level sets, conditional statistics, merge
  trees, and Morse--Smale structure.
- The flame brush should receive stringent normalized errors on the order of
  `1e-4` to `1e-3`, with relaxed control outside it.
- Fidelity outranks a heroic compression ratio: 5--10x, and even 2--3x, can be
  useful.  Partial decompression and random access are also desired.

The report reflects application information from March 2024 and compression
technology from January 2025, so these are requirements evidence rather than a
claim about the current compressor leaderboard.

## Relevant prior work

### Application-aware combustion studies

- Rai et al., [*Randomized Functional Sparse Tucker Tensor for Compression and
  Fast Visualization of Scientific Data*](https://arxiv.org/abs/1907.05884)
  describes what appears to be this exact dataset: a statistically planar
  premixed methane-air S3D flame with six species, eleven variables, a 500^3
  spatial grid, and 400 snapshots.  It studies low-rank spatial-temporal tensor
  compression.  This is essential related work and a possible quality/ratio
  reference, but it targets whole-dataset structure rather than an online
  block-local GPU compressor.
- Ali et al., [*Optimal Compressed Sensing and Reconstruction of Unstructured
  Mesh Datasets*](https://doi.org/10.1007/s41019-017-0042-4) (2018), evaluates
  S3D RCCI fields before and after ignition.  It varies local compression with
  gradient statistics, measures CO2/velocity and their gradients, and finds
  errors concentrate around flame fronts.  This is the closest precedent for
  our proposed spatially adaptive story, although its data and CPU-oriented
  compressed-sensing/wavelet methods differ from this 3-D SDRBench snapshot.
- Bode et al., [*BLASTNet: A call for community-involved big data in combustion
  machine learning*](https://doi.org/10.1016/j.dche.2022.100087) (2022), studies
  lossy-label effects on combustion ML and shows that derivative-like labels can
  amplify pointwise perturbations.  It supports evaluating downstream quantities,
  not treating a pointwise bound as sufficient.

### Compressor baselines that include combustion data

- [SDRBench](https://doi.org/10.1109/BigData50022.2020.9378449) registers
  combustion S3D as a standard scientific-compression dataset,
  so many compressor papers report generic ratio/error results on it.  Those
  results establish baselines but generally do not validate flame-specific QoIs.
- Recent GPU compressors such as
  [VGC](https://doi.org/10.1145/3712285.3759817) include HCCI and S3D among broad benchmark
  suites.  They are relevant throughput/ratio baselines, not substitutes for the
  domain metrics above.
- Learned scientific compressors such as
  [CAESAR](https://doi.org/10.3390/app15168977) also evaluate a dataset called
  S3D, but that tensor is a 2-D-in-space HCCI sequence with 58 species and 50
  timesteps.  It must not be conflated with our 11-field, 500^3 statistically
  planar premixed-flame snapshot.

## Working opportunity for FZGM

The potentially novel result is not “S3D compresses well.”  It is that a modular
GPU pipeline can compose a flame-region detector or supplied mask with two error
policies and an encoder, automatically fuse the eligible block-local path, and
occupy a useful throughput--quality--ratio point while preserving combustion QoIs.
The adaptive-error idea has precedent; the differentiation must come from usable
GPU composition, automatic fusion, and evidence across more than one application.
