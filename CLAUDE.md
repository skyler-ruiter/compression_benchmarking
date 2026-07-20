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
  load/render), `adapters/{base,fzgm,cusz_ref,cuszhi,cuszp,fzgpu,pfpl,mans,sz3,zfp,
  mgard,sperr,lscomp}`, `metrics` (harness-owned), `gpu` (throttle sampler),
  `provenance`, `store` (JSONL), `runner`, `analysis`, `cli`, `site`.
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
  `from_toml`. **FZGM `REL` = `eb·max(|data|)` (Lorenzo), NOT range** — the cross-tool
  comparable is `rel_range` (= FZGM `NOA` = cuSZ `REL`). See DESIGN §5.4.
- **FZGM is TOML-first** (`-c config.toml`), not `--stages`: full DAGs, and the rendered
  TOML is archived per run. The PATH `fzgmod-cli` may be **stale** (no `--report-json`) —
  point `FZGMOD_CLI` at the intended build.
- **HPC timing:** clocks usually can't be locked, so trust the variance flag
  (`timing_reliable`, cv ≤ 0.15) and prefer `*_device_ms_min`; a throttle sampler records
  why. Results from different GPUs are partitioned by provenance, never pooled.
- **Disk:** `retain_decompressed: false` by default (deletes ~original-sized `d.bin`,
  checksum kept, regenerable from `c.fzm`).

## Status

M1 (core loop) + M2 (HPC execution + timing reliability) complete. M3 (reference
adapters) well underway: cuSZ, cuSZ-Hi, cuSZp2/3, FZ-GPU, PFPL (GPU) and now SZ3, zfp,
MGARD, SPERR (CPU/GPU, added on the JetStream2 H100 node — see docs/adapters/*.md) all
have working adapters. MANS and lsCOMP remain stubs (lossless/quantized-integer
compressors that don't map onto the abs/rel_range/rel_maxabs model without a
quantization-wrapper design — see docs/adapters/mans.md, docs/adapters/lscomp.md).
See the roadmap in `docs/DESIGN.md` §9.

## Conventions

- No new Python deps beyond numpy + pyyaml (stdlib `tomllib` for TOML read). No pydantic.
- New experiment = a YAML in `configs/experiments/`; new pipeline = a `.toml`. Config over
  code. When you change behavior, update `docs/DESIGN.md` (and the decision log if it's a
  design choice) and the relevant `docs/adapters/*.md`.
