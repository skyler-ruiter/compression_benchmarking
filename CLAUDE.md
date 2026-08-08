# compression_benchmarking — Claude Code Instructions

Benchmarking toolkit (`benchkit`) for GPU error-bounded lossy compressors. Its job:
validate FZGPUModules (FZGM) modular ports against the original compressors (cuSZ, cuSZp,
cuSZ-Hi, MANS, PFPL) on compression ratio, quality, throughput, and memory — and support
reproducible experiments for papers, on the local desktop *and* on HPC clusters.

## Start of every session

Read these before doing work (they are the source of truth — this repo's docs travel
with it; the host's Claude `memory/` does not, so trust the docs):

1. `docs/DESIGN.md` — architecture, schemas, the 15-entry **decision log**, roadmap, and
   **§12 Execution on HPC**. This is the living contract.
2. `docs/adapters/fzgm.md` — the FZGM adapter contract and confirmed gotchas.
3. `README.md` — how to run (local + SLURM).

## What to run

```bash
# Set paths once: copy configs/site.example.yaml -> configs/site.local.yaml (gitignored),
# point fzgmod_cli at your build; or export FZGMOD_CLI / BENCHKIT_RESULTS_ROOT.
python -m benchkit run    configs/experiments/smoke.yaml      # run a matrix
python -m benchkit report results/<session>/                  # re-print the table
python -m benchkit merge  results/<session>/                  # combine shard files

# HPC: a SLURM job array (template: scripts/submit.slurm)
python -m benchkit run <exp> --session-id "$SLURM_ARRAY_JOB_ID" --shard "$SLURM_ARRAY_TASK_ID/$N"
```

## Project layout

- `benchkit/` — the package: `config`, `datasets`(in config.py), `pipelines` (TOML
  load/render), `adapters/{base,fzgm,cusz_ref,cuszhi,cuszp,fsz,fzgpu,pfpl,mans,sz3,zfp,
  mgard,sperr,lscomp,nvcomp}`, `metrics` (harness-owned), `fzm` (.fzm archive
  reader), `gpu` (throttle sampler), `provenance`, `store` (JSONL), `runner`,
  `analysis`, `cli`, `site`.
- `tools/nvcomp_cli/` — the only compressor CLI built *in this repo*. nvCOMP ships
  as a library with no vendor binary that reports device time; build with
  `scripts/build-nvcomp-cli.sh`.
- `configs/` — `datasets.yaml`, `experiments/*.yaml`, `pipelines/*.toml`,
  `site.example.yaml` (copy to gitignored `site.local.yaml`).
- `docs/` — `DESIGN.md`, `adapters/`. `scripts/submit.slurm` — SLURM array template.
  `scripts/build_comparison_artifact.py`, `scripts/reconstruct_runs_from_stdout.py` —
  see `results/baselines/README.md`.
- `results/` — gitignored run output (one dir per session), **except**
  `results/baselines/` — curated, git-tracked cross-machine snapshots (see
  `results/baselines/README.md`, `docs/DESIGN.md` D24).

## Key facts (don't relearn these the hard way)

- **Harness owns the metrics.** It computes CR/PSNR/NRMSE/eb-satisfaction itself; the only
  number trusted from a tool is device kernel time. Recompute throughput in one unit
  (decimal GB/s) — tools disagree (GB/s vs GiB/s vs MiB/ms).
- **Error-mode names collide.** Canonical modes are `abs`/`rel_range`/`rel_maxabs`/
  `from_toml`/`lossless`. **FZGM `REL` = `eb·max(|data|)` (Lorenzo), NOT range** — the
  cross-tool comparable is `rel_range` (= FZGM `NOA` = cuSZ `REL`). See DESIGN §5.4.
- **`lossless` is a mode, not a bound** (D31). For nvCOMP and for FZGM coder-only
  pipelines: no `error.bounds`, contract is bit-exactness. Every such row has
  `max_abs_err == 0` and `psnr == inf` *by construction*, so `validity.py` carves
  them out of the degeneracy detector and treats `cr <= 1` as a real measurement
  (`lossless_expansion`, retained) rather than corruption. `report --aggregate`
  showing "0 usable for quality" on a lossless session is correct, not a failure.
- **Never compare FZGM's `gpu_zstd` preset to nvCOMP Zstd end-to-end** (D32). The
  preset is lossy, nvCOMP is lossless; the 10.87x-vs-1.14x gap on CESM-2D/CLDHGH is
  the Lorenzo predictor, not the coder. Use `nvcomp_vs_fzgm_lossless.yaml` (predictor
  stripped) or `nvcomp_vs_fzgm_backend.yaml` (identical quant codes into both).
  See `docs/adapters/nvcomp.md`.
- **`device_ms` excludes different amounts of work per tool — check
  `*_host_over_device` before quoting a throughput ratio** (D33). FZGM's `device_ms`
  is `dag_elapsed_ms` (CUDA events around `dag->execute()` only); for a **split-mode**
  pipeline it omits the host-side assembly of the coded ports into an archive, which
  measured **3.53x** on raw f32 (single-stream control: 1.02x). nvCOMP runs 1.00x.
  Device-only said FZGM compresses 3.5x faster than nvCOMP Zstd; host wall says
  nvCOMP is *ahead*. Both are now in every row.
- **Pin, record, and re-check reference-tool versions** (D37). nvCOMP is pinned to
  5.3.0.16 and its version is recorded per session as `nvcomp_version`; the work
  started a release behind on 5.2.0.10 (CR bit-identical, but ANS compress +34% in
  5.3). Every reference compressor here is a hand-built tree — nothing keeps them
  current. `docs/adapters/nvcomp.md` has the redist check commands.
- **Give a reference tool its best config, not its documented default** (D34).
  nvCOMP's header recommends chunk 65536; on this H100 that costs Zstd up to 2.1x
  (16 KB is the optimum) because chunk count is the parallelism. FZGM's side is
  tuned by measurement, so nvCOMP is swept too.
- **FZGM is TOML-first** (`-c config.toml`), not `--stages`: full DAGs, and the rendered
  TOML is archived per run. The PATH `fzgmod-cli` may be **stale** (no `--report-json`) —
  point `FZGMOD_CLI` at the intended build.
- **HPC timing:** clocks usually can't be locked, so trust the variance flag
  (`timing_reliable`, cv ≤ 0.15) and prefer `*_device_ms_min`; a throttle sampler records
  why. Results from different GPUs are partitioned by provenance, never pooled.
- **`status: ok` is not "usable".** Reported means go through the validity gate in
  `benchkit/validity.py` (D30): constant/degenerate fields, `cr <= 1` expansions, and
  severe error-bound misses are excluded; marginal (≤1.01x) misses are retained and
  counted. `report --aggregate` gates by default and prints the audit; `--exclusions`
  prints the audit alone. Never quote an aggregate without it — on the full corpus the
  gate drops 798 of 9,416 `ok` rows.
- **Disk:** both `retain_decompressed` and `retain_compressed` default to `false` —
  `d.bin` and `c.fzm`/`c.cuszp`/etc. are deleted after each cell's row is written;
  sizes and checksums are recorded regardless. A single full `fzgm_vs_native.yaml`
  session left `retain_compressed` unset (i.e. always-kept) and ate 16GB of disk in
  compressed artifacts nobody read again — see D25.

## Full-corpus sweep

The paper-scale experiment is `configs/experiments/fzgm_vs_native_full.yaml` (~9.8k
cells, 186 fields, ~27 h on an H100) with an FZGM-only counterpart
`fzgm_only_full.yaml` for machines without native builds. **Read
`docs/running-the-full-corpus.md` before running it on any machine** — it covers the
data prerequisites (two datasets need splitting before they are usable), the per-site
commands, the sharding/resume workflow, and the memory ceilings.

Don't re-run all ~9.8k cells for a one-stage FZGM change — ask which cells it affects:

```bash
python -m benchkit stale <session>/                          # stage -> cells using it
python -m benchkit stale <session>/ --stage AdaptiveBitpack  # what a change invalidates
python -m benchkit stale <session>/ --against-build $FZGMOD_CLI   # detect drift automatically
```

`--against-build` compares each row's recorded stage fingerprints (`stage_versions`,
emitted by FZGM's `--report-json`) against a build, so you needn't know what you
edited. Rows from before 2026-07-29 carry none and are excluded, not assumed clean.

Then re-measure just those cells and collapse to one row each:

```bash
python -m benchkit run <exp> --session-id <S> --only-stale --against-build $FZGMOD_CLI
python -m benchkit merge $BENCHKIT_RESULTS_ROOT/<S>/     # newest row per cell wins
```

Re-runs **append**; the superseded row stays in the raw file so throughput changes
stay traceable. `merge` keeps the last row per `cell_key` within a file, and shard
files still beat `runs.jsonl`.

Backed by `configs/pipeline_stages.json` (regenerate with
`scripts/probe_pipeline_stages.py` whenever a preset changes; `--check` fails on
drift). Design + limits: `docs/stage-level-invalidation.md`.

## Status

M1 (core loop) + M2 (HPC execution + timing reliability) complete. M3 (reference
adapters) well underway: cuSZ, cuSZ-Hi, cuSZp2/3, FZ-GPU, PFPL (GPU) and now SZ3, zfp,
MGARD, SPERR (CPU/GPU, added on the JetStream2 H100 node — see docs/adapters/*.md) all
have working adapters, plus **nvCOMP** (GPU lossless: zstd/lz4/deflate/gdeflate/ans,
via the repo-built `tools/nvcomp_cli` — see docs/adapters/nvcomp.md) and **FSZ**
(SC'26, released 2026-08 — the compressor FZGM's `AdaptiveLorenzoStage` was
reconstructed from the paper alone; see docs/adapters/fsz.md). MANS and lsCOMP
remain stubs (quantized-integer compressors that don't map onto the
abs/rel_range/rel_maxabs model without a quantization-wrapper design — see
docs/adapters/mans.md, docs/adapters/lscomp.md). Note the `lossless` mode added for
nvCOMP (D31) is the missing half of what those two stubs need: they still lack the
quantizer, but no longer lack a way to express "no error bound".
See the roadmap in `docs/DESIGN.md` §9.

## Conventions

- No new Python deps beyond numpy + pyyaml (stdlib `tomllib` for TOML read). No pydantic.
- New experiment = a YAML in `configs/experiments/`; new pipeline = a `.toml`. Config over
  code. When you change behavior, update `docs/DESIGN.md` (and the decision log if it's a
  design choice) and the relevant `docs/adapters/*.md`.
