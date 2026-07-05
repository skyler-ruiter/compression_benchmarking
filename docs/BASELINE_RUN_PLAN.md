# Baseline Reference Run — Planning

**Goal:** one large, paper-grade SLURM job (spanning multiple GPU nodes on BigRed200's
A100 partition) that produces a trustworthy baseline of CR / quality / throughput /
memory for every compressor and FZGM modular composition we have, across the full
SDRBench corpus. This is the "real" run that `docs/BENCHMARK_PLAN.md` §B has been
gating on the FZGM-vs-native validation work.

**Status:** planning only. Not ready to submit — the open-decision list below needs
answers (some from the user, some from a dry run) before an experiment YAML gets
written. Gated on finishing the in-flight bugfix confirmations (E17, E16, E4) tracked
in `docs/BENCHMARK_PLAN.md`.

This doc is scoped to *this* run's decisions. Architecture stays in `docs/DESIGN.md`
(§12 covers HPC execution mechanics already built); this doc is about what matrix to
point that machinery at.

---

## 1. Open decisions, by axis

### 1.1 SLURM / job mechanics

- Existing template (`scripts/submit.slurm`): 1 GPU per array task, `N` shards share
  one session dir (`--session-id $SLURM_ARRAY_JOB_ID`), `--exclusive` for steady
  clocks. "Multiple GPU nodes" falls out of this for free — SLURM schedules each
  1-GPU task wherever a node is free, no multi-node code needed. Open question is
  just whether that's *sufficient*, or whether we want node-affinity/SKU grouping
  (provenance already partitions by GPU so heterogeneous hardware can't silently pool,
  per DESIGN §12 — but confirm BigRed200's gpu partition is uniform A100-40GB and
  doesn't mix in other SKUs).
- **Unknown, needs checking:** BigRed200 gpu partition allocation limits — max
  concurrent jobs/GPUs under our account, max array size, walltime ceiling, QOS. Check
  with `sacctmgr show assoc -p user=$USER` or ask HPC support before picking `N`.
- **Unknown, needs data:** per-cell runtime. Pull actual device-timing numbers from
  the `fzgm_vs_native.yaml` / `smoke-m3-refs.yaml` runs already done (2026-07-02/03)
  to extrapolate total walltime = (cells per shard) × (reps + warmup) × (avg per-rep
  time), then pick `N` and per-task `--time`.
- Resume is already idempotent via `cell_key` (DESIGN §12) — conservative walltime +
  resubmit-on-timeout is an acceptable fallback, don't need to over-provision `N`.
- `lock_clocks: true` given `--exclusive`, matching `sdrbench.yaml` convention.

### 1.2 Datasets

- Full SDRBench f32 matrix (`sdrbench.yaml`'s dataset list): CESM-2D (79) +
  CESMATM-3D (33) + HURR (13) + NYX (6) + HACC (6) + EXAALT (6) + QMCPACK (1) =
  **144 fields**. Plus MIRANDA (7 f64 fields), which must live in a separate
  experiment file (`sdrbench-miranda.yaml`-style) since the runner applies every
  `runs:` entry to every listed dataset and f32/f64 pipelines can't mix in one file.
- **Decide:** does "baseline reference for all these compressors" mean the *full*
  144(+7) fields, or one-field-per-shape-class the way `fzgm_vs_native.yaml` did for
  validation? The user's framing ("strong baseline performance reference") reads as
  full-corpus, not spot-check — but that's a scope call worth confirming since it's
  the single biggest lever on total cell count.
- **Decide:** is MIRANDA (f64) in scope for this run, or a separate follow-up? f64
  pipelines only confirmed to exist for `cusz_f64`/`fzgpu_f64`; f64 support for
  pfpl/cuszp2/cuszp3/cuszhi is unconfirmed — check adapter code before committing.
- `SCALING-SYNTH` is out of scope here — that dataset exists for the throughput-vs-size
  question (`docs/BENCHMARK_PLAN.md` category C), a different question than this run.

### 1.3 Compressors / modular compositions

| Native tool | FZGM port | Status per BENCHMARK_PLAN.md (2026-07-03) |
|---|---|---|
| cusz | `cusz.toml` | ✅ validated |
| cuszhi (tp mode) | `cusz_hi_tp.toml` | ✅ validated (E11 closed) |
| cuszhi (cr mode) | `cusz_hi_cr.toml` | ⚠️ mostly validated, E17 open (~36% CR gap on NYX only) |
| cuszp2 | `cuszp2.toml` | ✅ validated |
| cuszp3 | `cuszp3.toml` | ⚠️ 2-D/3-D only, no 1-D preset (E12) |
| fzgpu | `fzgpu.toml` | ✅ validated (E16 eb-overshoot curiosity, low priority) |
| pfpl | `pfpl.toml` | ✅ validated |
| mans | — | stub, excluded |

- **Decide:** gate the full run on E17/E4 closing, or run anyway and flag known-gap
  cells the way `fzgm_vs_native.yaml` did explicitly in its header comment? The
  BENCHMARK_PLAN open-items list already leans toward "closer to ready" once those
  close — worth finishing those first per the user's stated plan for this session.
  E16 is low-priority and shouldn't block.
  - **Note (2026-07-03 mid-session):** none of E17/E16/E4 confirmed closed yet as of
    this doc; treat as still-open gating items until BENCHMARK_PLAN.md says otherwise.
- **Decide:** does "modular compositions" mean just the direct 1:1 FZGM ports of each
  native tool (current scope of everything built so far), or does the user also want
  novel cross-family FZGM DAGs (e.g. swapping a predictor from one tool's pipeline
  with an encoder from another's) as part of the baseline? That's a real scope
  expansion beyond what's been validated — flag for explicit confirmation, don't
  assume.
- cuszp3's 1-D gap (E12): either add a 1-D preset before this run, or exclude
  HACC/EXAALT from cuszp3's rows and say so in the experiment file's comments (as
  `sdrbench-miranda.yaml` does for its own scope note).

### 1.4 Error bounds

- Existing convention (`sdrbench.yaml`): `rel_range` (the canonical cross-tool
  comparable — DESIGN §5.4: `rel_range` = FZGM `NOA` = cuSZ `REL`), bounds
  `[1e-2, 1e-3, 1e-4]`.
- **Decide:** enough for "strong baseline," or add `1e-5` (common in SDRBench-derived
  papers) or a second abs-mode pass? More bounds multiplies total cells linearly, so
  this interacts directly with the walltime budget in §1.1.
- Worth a quick audit before trusting a big matrix: `rel_range` plumbing has only been
  smoke-tested end-to-end for cusz/cuszp/fzgpu so far (per BENCHMARK_PLAN smoke
  table) — confirm cuszhi/pfpl's rel_range → CLI translation has actually been
  exercised too, not just abs/default modes.

### 1.5 Modes / settings / parameter sweeps

- cuSZ-Hi tp vs cr: already split as two separate `runs:` rows — keep both.
- cuSZp v2 (outlier-selection) vs v3 (plain tiled-Lorenzo, 2-D/3-D only) — keep both,
  gate v3 out of 1-D fields per E12 above.
- Memory strategy (PREALLOCATE vs MINIMAL) and outlier-capacity tuning
  (`cusz_minimal_lowoutlier.toml`) exist as levers from the **scaling** work
  (category C), not the comparison work — these are about surviving large single
  allocations, not standard SDRBench field sizes. **Recommend: out of scope for this
  run**, confirm with user rather than silently sweeping them in.
- `proc_dim` (cuszp3's dimensionality mode) needs one preset per dataset
  dimensionality (1-D/2-D/3-D) if cuszp3 stays in scope for non-2-D data.
- Repetitions: `sdrbench.yaml`'s existing convention (`repetitions: 10, warmup_reps: 3,
  timing_cv_threshold: 0.10`) is the current "trustworthy" bar — keep unless the
  walltime budget forces a cut, in which case cut bounds/fields before cutting reps
  (reps is what makes `timing_reliable` meaningful).

### 1.6 Other setup details

- `retain_decompressed: false` (existing convention) — keep. Re-check the ~200 GB
  scratch estimate in `docs/hpc-setup.md` against this run's actual field/pipeline
  count; more compressor rows means more `c.fzm` + checksum artifacts kept even
  without `d.bin`.
- Provenance: confirm every adapter (not just fzgm) records tool build/version info
  in `provenance.json`, not only the `fzgmod_cli` path — matters for reproducing a
  "baseline reference" later.
- Write-up format: given the scale (144+ fields × ~14 compressor/port rows × several
  bounds), the flat `benchkit report` table won't be readable as one artifact — plan
  for a per-compressor summary rollup in the eventual `docs/experiments/*.md`
  write-up, not just the raw table.

---

## 2. Suggested sequencing

1. Finish the in-flight bugfix confirmations already planned for this session (E4
   re-verify at scale, E17 root-cause, E16 decision) — those are explicitly on the
   BENCHMARK_PLAN.md gating list for "SDRBench full matrix ready."
2. Resolve the scope questions in §1.2/§1.3/§1.5 with the user (full corpus vs subset,
   MIRANDA in/out, novel cross-family DAGs in/out, memory-strategy sweeps in/out) —
   these change total cell count by an order of magnitude and shouldn't be guessed.
3. Estimate total cells × per-cell time from existing timing data; pick shard count
   `N` and per-task walltime.
4. Confirm BigRed200 gpu-partition limits (`sacctmgr`/HPC support) before submitting
   an array sized from step 3.
5. Author `configs/experiments/baseline_reference.yaml` (+ a MIRANDA companion file
   if in scope), reusing `sdrbench.yaml`'s structure with the full compressor/FZGM
   pairing list from `fzgm_vs_native.yaml`.
6. Dry-run at smoke scale (one field per shape class, every compressor row) to
   validate the full run list end-to-end and capture real per-cell timing before
   committing to the full array.
7. Submit, merge, report, write up in `docs/experiments/`.

---

## 3. Questions to put to the user before writing the experiment YAML

- [ ] Full 144(+7 MIRANDA) field corpus, or a representative subset?
- [ ] Include MIRANDA (f64) in this run, or as a separate follow-up?
- [ ] Direct 1:1 FZGM ports only, or also novel cross-family FZGM DAG compositions?
- [ ] Error bounds: keep `[1e-2, 1e-3, 1e-4]`, or add `1e-5` / an abs-mode pass?
- [ ] Memory-strategy / outlier-capacity sweeps in scope, or reserved for category C?
- [ ] Any walltime/allocation budget constraints (account SUs, deadline) that should
      cap total scope before we even estimate cell count?
