# compression_benchmarking

Standardized benchmarking and analysis toolkit for **GPU-accelerated, error-bounded
lossy compressors (EBLCs)** — cuSZ, cuSZ+, cuSZp, cuSZ-Hi, FSZ, FZ-GPU, PFPL,
SZ3, zfp, MGARD-X, SPERR, MANS, lsCOMP, and modularized
[FZGPUModules](https://github.com/szcompressor/FZGPUModules) (FZGM) pipelines.

It exists to answer one question cheaply and repeatedly:

> Does the FZGM port of a compressor roughly match the original on **compression ratio**
> and **quality**, without losing too much **speed** — under identical datasets, error
> bounds, and error-mode semantics?

…and, more broadly, to support rapid, reproducible compression experiments for research
papers: provenance-tracked runs, tidy results, and paper-ready tables and figures.

## Status

**M1 complete** — the core loop runs end-to-end on FZGM. The architecture, schemas,
metric definitions, and roadmap live in **[docs/DESIGN.md](docs/DESIGN.md)** — start
there. Reference-compressor adapters (cuSZ, cuSZp, …) land in M3.

## Usage

```bash
# Requires: python3 + numpy + pyyaml, a built fzgmod-cli, an NVIDIA GPU.
# Set paths once: copy configs/site.example.yaml -> configs/site.local.yaml (gitignored)
# and point fzgmod_cli at your build, or export FZGMOD_CLI / BENCHKIT_RESULTS_ROOT.

python -m benchkit run    configs/experiments/smoke.yaml   # run a matrix
python -m benchkit report results/<session>/               # re-print the table
```

Each run writes a session dir: `runs.jsonl` (one tidy row per measurement),
`provenance*.json` (GPU/driver/host/scheduler/git), `logs/`, and `work/`. Configure
experiments in [configs/experiments/](configs/experiments/), datasets in
[configs/datasets.yaml](configs/datasets.yaml), pipelines in
[configs/pipelines/](configs/pipelines/).

### HPC (SLURM)

The same configs run on a cluster. A job array shards the matrix; each task is
resumable. Template: [scripts/submit.slurm](scripts/submit.slurm).

```bash
# one array task (k of N) into a shared, resumable session
python -m benchkit run configs/experiments/smoke.yaml \
    --session-id "$SLURM_ARRAY_JOB_ID" --shard "${SLURM_ARRAY_TASK_ID}/${N}"
python -m benchkit merge "$BENCHKIT_RESULTS_ROOT/$SLURM_ARRAY_JOB_ID"   # after array
```

FZGM adapter quirks (the `rel`-basis finding, huffman/zigzag, TOML-first) are in
[docs/adapters/fzgm.md](docs/adapters/fzgm.md); the full design in
[docs/DESIGN.md](docs/DESIGN.md).

### General reference compressors

SZ3, zfp, MGARD-X, and SPERR are first-class standalone adapters; they do not
need a corresponding FZGM pipeline. A ready-to-run coverage check is:

```bash
source scripts/env-jetstream2.sh
python3 -m benchkit run configs/experiments/smoke-cpu-refs.yaml \
  --session-id general-refs-smoke
```

| Adapter | Execution and timing | Error-bound support | Dimensions |
|---|---|---|---|
| SZ3 | CPU, native self-reported wall time | native `abs`, native `rel_range` | 1–4D |
| zfp | CPU serial, external process wall time | native `abs`; emulated `rel_range`/`rel_maxabs` | 1–4D |
| MGARD-X | CUDA, self-reported low-level time | native `abs`; emulated `rel_range`/`rel_maxabs` | dataset-dependent; large 1-D HACC is excluded |
| SPERR | CPU, external process wall time | guarded native `abs`; emulated `rel_range`/`rel_maxabs` | 2-D/3-D only |

The CPU/external-wall figures are useful within an adapter, but are not a GPU
device-throughput ranking. Benchkit still owns CR and reconstructed-quality
measurement for every adapter. Detailed contracts live in
[`docs/adapters/`](docs/adapters/).

### Lossless codecs (nvCOMP)

nvCOMP's Zstd/LZ4/Deflate/GDeflate/ANS take no error bound, so they run under the
`lossless` error mode against FZGM pipelines built from coder stages alone:

```bash
./scripts/build-nvcomp-cli.sh          # nvCOMP ships no CLI; this repo builds one
python -m benchkit run configs/experiments/nvcomp_vs_fzgm_lossless.yaml

# coder-vs-coder on identical Lorenzo quant codes:
python scripts/extract_quant_codes.py --dataset CESM-2D --fields all --eb 1e-3
python -m benchkit run configs/experiments/nvcomp_vs_fzgm_backend.yaml
```

**Do not** compare FZGM's lossy `gpu_zstd` preset to nvCOMP Zstd directly — the gap
is the Lorenzo predictor, not the coder. [docs/adapters/nvcomp.md](docs/adapters/nvcomp.md)
explains both framings and the validity-gate consequences of lossless rows.

## Why it's structured this way (the short version)

- **The harness owns the metrics.** CR, PSNR, NRMSE, and error-bound checks are computed
  by the toolkit from raw artifacts, not scraped from each tool's self-report — so the
  comparison is fair. The only number trusted from a tool is its device kernel time.
- **Reference compressors are pinned git submodules**; FZGM is driven via an installed
  `fzgmod-cli`.
- **Datasets are the SDRBench standard set**, described by a checksummed manifest.
- **Results are append-only JSONL**, one row per atomic run, each carrying full
  environment provenance.

See [docs/DESIGN.md](docs/DESIGN.md) for the full design, schemas, and milestone plan.
