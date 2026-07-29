# Running the full-corpus sweep

The full-corpus experiment (`configs/experiments/fzgm_vs_native_full.yaml`, ~9.8k cells
over 186 fields) replaces the 4-field `fzgm_vs_native.yaml` for paper-scale results.
This page is the per-machine runbook.

Started 2026-07-28. See `docs/DESIGN.md` D26/D27 for the bugs that had to be fixed
before this was runnable at all.

---

## 0. What changed vs. the old 210-cell runs

Results from this sweep are **not** throughput-comparable with baselines captured before
2026-07-28, for one specific reason: the cuSZ-Hi presets moved from
`MemoryStrategy::MINIMAL` to `PREALLOCATE` (D27). CR and PSNR are unaffected —
PREALLOCATE is byte-identical — but cuSZ-Hi throughput changes (it gets faster). Every
other pipeline was already PREALLOCATE and is unaffected.

Two correctness fixes also landed, both of which silently corrupted results before:

- f64 fields were fed to float32 pipeline stages, producing `nan` PSNR and degenerate CR
  on half the affected cells and hard failures on the rest. Only matters if you have
  MIRANDA / S3D / NWCHEM / BROWN registered — i.e. now.
- cuSZ-Hi corrupted memory on large 3-D fields under MINIMAL.

So: **re-run, do not merge new shards into an old session.**

---

## 1. Prerequisites on every machine

### 1a. Sync the repo

The configs, the two extraction scripts, and `benchkit/` itself all changed. Pull before
running; a stale `benchkit/metrics.py` will still work but is ~4x slower per cell, and a
stale `configs/pipelines/cusz_hi_*.toml` will hit the MINIMAL memory-corruption bug.

### 1b. Rebuild FZGPUModules

`modules/fused/ginterp/` changed (bounds-checked outlier compaction). Rebuild with the
preset that machine already uses — `release` / `cuda-h200` / the HIP build. Confirm with
`ctest` (46/46 expected on CUDA).

### 1c. Get the data

The corpus grew from 8 dataset families to 13. On a machine that only has the original
eight, fetch the rest:

```bash
source scripts/<your-site>.sh          # sets BENCHKIT_DATA_ROOT
SDRBENCH_DATASETS="SCALE NWCHEM S3D BROWN EXAFEL" \
    bash scripts/download-sdrbench.sh "$BENCHKIT_DATA_ROOT"
```

Then run the two split scripts — **neither dataset is usable without them**, because
SDRBench does not ship either one as individual field files:

```bash
python scripts/extract_qmcpack_orbitals.py --n 8     # 4-D blob -> 8 3-D orbitals
python scripts/extract_s3d_variables.py              # 11 GB blob -> 11 3-D variables
```

**Run one invocation per dataset, in parallel.** The Globus mirror throttles *per
connection*, not per client: a single sequential `SDRBENCH_DATASETS="SCALE NWCHEM S3D
BROWN EXAFEL"` run sits at 1–2 MB/s, which is ~10 h for these five (S3D alone is a 46 GB
tarball). Five concurrent one-dataset invocations pull ~100 MB/s aggregate and finish in
minutes. They touch disjoint tarballs and destination directories, so this is safe:

```bash
for ds in SCALE NWCHEM S3D BROWN EXAFEL; do
    SDRBENCH_DATASETS="$ds" nohup bash scripts/download-sdrbench.sh \
        "$BENCHKIT_DATA_ROOT" > ~/download-$ds.log 2>&1 &
done; wait
```

Budget roughly **120 GB** for the full corpus, plus transient space for tarballs
(delete them after extraction). Verify before you queue an 8-hour job:

```bash
python - <<'EOF'
import os, yaml
cfg = yaml.safe_load(os.path.expandvars(open('configs/datasets.yaml').read()))
bad = 0
for ds, sp in cfg.items():
    if not isinstance(sp, dict) or 'fields' not in sp: continue
    w = 8 if sp['dtype'] == 'f64' else 4
    for n, f in sp['fields'].items():
        p = os.path.join(sp['root'], f['path']); e = 1
        for d in f['dims']: e *= d
        if not (os.path.exists(p) and os.path.getsize(p) == e * w):
            bad += 1; print('MISSING/WRONG SIZE:', ds, n, p)
print('problems:', bad)
EOF
```

---

## 2. Per-machine

### JetStream2 H100 — full vs-native matrix, single process

Natives are built; clocks can be locked. This is the reference machine.

```bash
cd ~/compression_benchmarking
bash scripts/lock_clocks.sh                     # needs sudo; JetStream2 has it
source scripts/env-jetstream2.sh
source .venv/bin/activate
nohup python -m benchkit run configs/experiments/fzgm_vs_native_full.yaml \
      --session-id 20260728-fullcorpus-skyler-h100 > ~/fullcorpus.log 2>&1 &
```

~27 h. **Do not shard this across concurrent processes on the one GPU.** The GPU idles
near 0% (per-cell kernel time is milliseconds against seconds of host work), so it looks
like free parallelism, but overlapping processes contend during the timed kernels and
corrupt the throughput numbers the sweep exists to produce. Sharding is for separate
GPUs, not for one.

Run `bash scripts/unlock_clocks.sh` when finished.

### BigRed200 A100 — full vs-native matrix, job array

Natives are built here too, so it runs the same experiment as the H100 and the two are
cell-for-cell comparable. Clock control is admin-only, so `--exclusive` is the only lever
on timing variance (D15).

```bash
# edit SESSION_ID in the script, or export it; it MUST be stable across resubmissions
sbatch --array=0-7 scripts/submit_full_corpus.slurm
# requeue the same array until no task has cells left:
sbatch --array=0-7 scripts/submit_full_corpus.slurm
python -m benchkit merge  $BENCHKIT_RESULTS_ROOT/<session>/
python -m benchkit report $BENCHKIT_RESULTS_ROOT/<session>/ --aggregate --by-dataset
```

The submit script's BigRed200 block is uncommented by default.

### NCSA Delta H200 and MI100 — FZGM-only

The native references are **not built** on Delta, and on the MI100 they cannot be:
cuSZ / cuSZ-Hi / cuSZp / FZ-GPU / PFPL are CUDA-only codebases, so an FZGM-vs-native
matrix there is structurally impossible, not merely absent. Both Delta GPUs therefore run
`configs/experiments/fzgm_only_full.yaml`, which is the `fzgm` half of the vs-native
config, verified 1:1 (4,860 cells).

**Do not use `scripts/submit_full_corpus.slurm` here** — its `--array=0-7 --exclusive`
geometry is wrong for Delta, because both of that script's premises fail:

- `MaxTime` on `gpuH200x8` and `gpuMI100x8` is **2-00:00:00**, not 8 h. Sharding is not
  forced by the wall clock, so the requeue-until-done loop is unnecessary.
- `gpuMI100x8` is a **single node** (8 MI100s) and `gpuH200x8` is only **8 nodes**. An
  8-way exclusive array therefore *serialises* on the MI100 — each task wants the whole
  node — and claims the entire partition on the H200.

Use `scripts/submit_delta_multigpu.slurm` instead: one job, one `--exclusive` node, and
8 concurrent shards each pinned to its own GPU on that node. That still honours the
"shard across separate GPUs, never several processes onto one GPU" rule from §2, while
holding one node instead of eight.

```bash
sbatch --export=ALL,SITE=h200  scripts/submit_delta_multigpu.slurm
sbatch --export=ALL,SITE=mi100 scripts/submit_delta_multigpu.slurm
```

`SESSION_ID` defaults to `fullcorpus-delta-<site>` and is stable across resubmissions by
construction, so resume works if a job does hit the wall limit.

**MI100 caveat — expect cuSZ-Hi failures on MIRANDA and S3D.** GInterp's
double-precision 3-D path exceeds MI100's fixed 64 KB LDS ceiling (the f32 path is well
under it). The old 102-cell `fzgm_only.yaml` never hit this because it had no f64 3-D
dataset; the full corpus has two. Either accept the `status: fail` rows — they are a real,
documented backend limitation and worth recording — or add
`skip_datasets: [MIRANDA, S3D]` to the two `fzgm:cuszhi_*` entries in an MI100-local copy
and say so in that baseline's `metadata.yaml`. It is deliberately not baked into the
shared config, which would desynchronise it from the H200 run.

---

## 3. Memory ceilings

Measured peak device memory is roughly 4–11x the field payload depending on pipeline
(PREALLOCATE sizes to worst case). The corpus is scoped so that everything in the
experiment fits the smallest GPU in the fleet (32 GB MI100) — with one deliberate
exclusion:

`NWCHEM` has two fields far larger than SDRBench advertises, `acd` (6.4 GB) and `ccd`
(5.7 GB). Measured peak on an H100: 20.6 GB (cusz) to **35.0 GB (pfpl)**. That does not
fit the MI100 and leaves no headroom on the 40 GB A100, so the experiment scopes NWCHEM
to its one normal-sized field (`t631`) via the `fields:` block. Run the big two
separately on H100/H200 if you want them.

If you test any f64 field by hand with `fzgmod-cli`, render `input_type` to `float64`
first — invoking a preset straight out of `configs/pipelines/` with `-t f64` silently
reads the data as f32 and gives plausible-looking but wrong numbers (D27).

---

## 4. After the run

```bash
python -m benchkit merge  <session>/                       # shards -> runs.jsonl
python -m benchkit report <session>/ --aggregate --by-dataset
```

`--by-dataset` is what you want for a paper table: it reports, per dataset x pipeline x
error bound, both the size-weighted **ratio-of-sums** CR and the equal-weight
**geometric-mean** CR, plus a geometric spread and the min/max field. Pooling datasets
instead is misleading — ratio-of-sums is size-weighted, so a 21 GB CESMATM would simply
*be* the "overall" number.

Then snapshot to `results/baselines/<id>/` per `results/baselines/README.md` (D24):
`runs.jsonl` + `provenance.json` + a `metadata.yaml` recording GPU, site, date, clock
state, and **known issues** — including, for this generation of baselines, the
MINIMAL→PREALLOCATE throughput discontinuity described in §0.
