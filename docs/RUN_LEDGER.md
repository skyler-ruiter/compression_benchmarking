# Benchmark Run Ledger

Last updated: 2026-08-24

This is the canonical operational record for benchmark work: what we intend to
run, what has actually run, on which machine, and what evidence is still
missing. [`BENCHMARK_PLAN.md`](BENCHMARK_PLAN.md) retains the longer historical
plan and design discussion.

## Status vocabulary

| Status | Meaning |
|---|---|
| `IDEA` | Desired comparison, but its experimental contract is not fixed. |
| `BLOCKED` | A named design, implementation, dependency, or data decision is missing. |
| `CONFIGURED` | A checked-in config exists, but the intended campaign has not run. |
| `RUNNING` | A resumable session is active. |
| `PARTIAL` | Useful results exist, but do not satisfy the stated campaign. |
| `COMPLETE` | Intended cells ran, accepted failures are documented, and results are curated. |
| `SUPERSEDED` | Kept for provenance, but a newer campaign is authoritative. |

A campaign is not `COMPLETE` merely because its process exited. Record the
config and intended cell count, success/failure disposition, machine and
software provenance, timing validity, curated baseline path, and result note.

## Running and resuming safely

Use a stable session name for the lifetime of a run:

```bash
python -m benchkit run configs/experiments/EXPERIMENT.yaml \
  --session-id RUN_ID --results-root results/runs

# Reissuing the same command resumes the session and skips completed cells.
python -m benchkit run configs/experiments/EXPERIMENT.yaml \
  --session-id RUN_ID --results-root results/runs
```

For overnight work, keep the same session directory and use `tmux`, `screen`,
or the site scheduler. Do not launch overlapping processes against one session.
For sequential shards, keep the run contract identical and give each shard a
distinct session ID; merge only after checking that matrix keys do not overlap.
See [`running-the-full-corpus.md`](running-the-full-corpus.md) for the established
full-corpus workflow.

## Machine registry

| Machine ID | Accelerator | Demonstrated scope | Current role |
|---|---|---|---|
| `js2-h100` | NVIDIA H100 80 GB | Full FZGM/native corpus, nvCOMP parity, all current third-party adapter smoke tests | Primary CUDA reference and next-run target |
| `br200-a100` | NVIDIA A100 40 GB | Full FZGM/native corpus with the 2026-07 toolchain | Cross-machine validation; new adapters need revalidation |
| `delta-h200` | NVIDIA H200 141 GB | Full FZGM-only corpus | Scaling/portability; native toolchain not established |
| `delta-mi100` | AMD MI100 32 GB | Full HIP FZGM-only corpus | HIP portability; CUDA-native and nvCOMP comparisons are structurally unavailable |

Use these IDs in new result notes. Add exact hostname, GPU model, driver,
runtime, commit, adapter executable, and library version to session provenance;
the short ID is only a human-readable join key.

## Master dashboard

| ID | Campaign | Status | Machines with useful results | Next action |
|---|---|---|---|---|
| `EBLC-BASE` | FZGM vs matching native GPU EBLC, standard bounds | `COMPLETE` | H100, A100; FZGM-only H200/MI100 | Preserve as reference; rerun only for a material code/toolchain change |
| `EBLC-TIGHT` | GPU pairs plus FSZ at `1e-6` and `1e-7` | `PARTIAL` | H100 diagnostic sweep | Rerun after the FSZ exit-1 and FZGM linear-overflow fixes; quarantine native cuSZ/cuSZ-Hi defects |
| `EBLC-MODES` | `abs`, range-relative, max-absolute-relative comparisons | `IDEA` | Range-relative baseline only | Build a capability/semantics matrix before authoring configs |
| `EBLC-LOG` | Positive-domain log transform plus EBLC | `BLOCKED` | None | Specify transform, zero/negative policy, inverse, and original-domain quality gates |
| `EBLC-FSZ` | Native FSZ vs matching FZGM pipeline | `PARTIAL` | H100 subset | Tight full-corpus pair is configured; add standard-bound full-corpus coverage |
| `EBLC-SZX` | Native SZx/SZp family comparison | `BLOCKED` | cuSZp2/cuSZp3 pairs already covered | Name the exact repository/version and distinguish it from existing cuSZp2/3 |
| `NVC-PARITY` | FZGM lossless stages vs closest nvCOMP counterparts | `COMPLETE` | H100 scientific subset | Keep bit-exact stage gate with future backend changes |
| `NVC-CORPUS` | Broader general-lossless corpus | `PARTIAL` | H100 scientific subset only | Select, license, checksum, and register raw log/genomic/general-byte datasets |
| `NVC-PROP` | Unpaired nvCOMP Bitcomp/Cascaded and other native codecs | `COMPLETE` for current subset | H100 | Extend with `NVC-CORPUS`; do not imply an FZGM pair |
| `3P-ADAPTERS` | FSZ, SZ3, zfp, MGARD-X, SPERR, MANS, lsCOMP execution support | `COMPLETE` for H100 smoke | H100 | Re-smoke on other CUDA machines as needed |
| `3P-CORPUS` | Standalone third-party EBLC on the general scientific corpus | `PARTIAL` | Small H100 tests only | Write standard-bound full-corpus config distinct from the tight stress run |
| `FEAT-HUFF` | Adaptive Huffman feature highlight | `PARTIAL` | H100 subset | Add controlled full-corpus cuSZ/cuSZ-Hi CR-mode preset |
| `FEAT-BITPACK` | Adaptive bitpack/outlier-selection ablation | `PARTIAL` | H100 baseline modes | Name one knob at a time and avoid relabeling existing TP/CR pairs |
| `FEAT-GINTERP` | GInterp adaptivity ablation | `IDEA` | No clean adaptive-only comparison | Identify an actual independent adaptive switch before scheduling |
| `FEAT-PRED` | Adaptive vs fixed predictor/coder behavior | `PARTIAL` | H100 FSZ subset | Fold a controlled full-corpus comparison into the FSZ campaign |
| `INFRA-AE` | Publication and AD/AE hardening | `COMPLETE` | CPU CI plus H100 publication smoke | Use H0-H6 for new paper sessions; preserve compatibility gates |

## Publication and AD/AE infrastructure (`INFRA-AE`)

The technical contract and acceptance criteria live in
[`DESIGN.md` M6](DESIGN.md#m6-publication-hardening-path). This ledger records only
execution status so infrastructure work does not become a second planning system.

| Package | Status | Completed evidence | Next action |
|---|---|---|---|
| H0 canonical loading and comparison identity | `COMPLETE` | Merged directories prefer `runs.jsonl`; unmerged directories load shards; comparison artifacts retain `dataset/field`; regression tests cover both paths | Preserve behavior while changing identities |
| H1 versioned schema and strict JSON | `COMPLETE` | v1 schema module and strict writers; native/failure/reconstructed contracts; explicit non-finite encoding; legacy-v0 readers; real legacy merge verified; compatibility readers wired into reporting/artifact scripts | Preserve legacy reads while adding H2 IDs |
| H2 logical vs execution identity | `COMPLETE` | Stored canonical payloads/SHA-256 IDs; execution-keyed resume; logical-keyed merge with explicit supersession/collision failure; publication pairing checks logical IDs; graph/config/dataset/tool/harness sensitivity; legacy transition tests | Preserve identity semantics while adding provenance joins |
| H3 provenance and dataset integrity | `COMPLETE` | Exact config/manifest/checksum-lock/site/rendered-pipeline archives; 238 locked field hashes; safe atomic checksum update/verification command; full dirty-tree identity; declared/observed dataset checksums; tool build declarations; shard-bound provenance IDs on every new row; explicit non-publication-grade opt-out; regression tests | Preserve provenance joins while implementing H4 |
| H4 session verification | `COMPLETE` | `benchkit verify`; strict machine-readable report; raw/canonical schema and identity audit; provenance/artifact/dataset joins; matrix and merge completeness; explicit named exception policy; nonzero incomplete exit; regression tests; 2026-08-24 real H100/FZGM publication smoke completed 1/1 cell and passed all 12 checks | Preserve report semantics while implementing H5 |
| H5 AE bundle | `COMPLETE` | Content-addressed build/offline verify; canonical rows and metadata; archived inputs/provenance/logs/recipes/licenses; one-command reproduction; final H100/FZGM source and reproduced sessions each passed 12/12 H4 checks | Use `benchkit artifact build` after each authoritative session |
| H6 tests/CI/packaging closure | `COMPLETE` | Fake-CLI end-to-end and parser fixtures; schema/dataset/golden artifact tests; Python 3.10/3.12 CPU CI; exact dev lock; fresh-venv install passed 55 tests; tracked machine ELFs removed; external tool acquisition documented truthfully | Maintain these gates as adapters and schemas evolve |

Do not regenerate an authoritative full-corpus baseline merely because an H1-H3 storage
or provenance change lands. First validate backward reads against the curated baselines,
then run a small publication-grade smoke session through H4. Schedule a costly baseline
rerun only when its scientific result or required provenance cannot be recovered.

## Active sessions

| Campaign | Session | Machine | Started | Intended cells | State and paths |
|---|---|---|---|---:|---|
| `EBLC-TIGHT` | `tight-full-gpu-20260822-skyler-h100` | `js2-h100` | 2026-08-22 16:43 UTC | 7,080 | `COMPLETE-DIAGNOSTIC`: 4,790 ok / 2,290 failed; results: `/home/exouser/benchkit-results/tight-full-gpu-20260822-skyler-h100`; log: `/home/exouser/benchkit-results/_run_logs/tight-full-gpu-20260822-skyler-h100.log` |
| `EBLC-TIGHT` repair | `tight-full-gpu-20260822-skyler-h100`, shard `0/1` | `js2-h100` | 2026-08-23 15:50 UTC | 807 retried | `COMPLETE-DIAGNOSTIC`: 319 ok / 488 failed. FSZ: 319 measured bound misses + 4 constant-field exit-2 failures; FZGM: 484 representable-spacing refusals. Config: `configs/experiments/tight_bounds_targeted_rerun.yaml`; log: `/home/exouser/benchkit-results/_run_logs/tight-full-gpu-20260822-targeted-repair.log` |

Early-run alert: native cuSZ's first tight-bound cells exited successfully but
the harness marked their bounds unsatisfied. Several non-degenerate CESM-2D
fields expanded (`CR < 1`) and reconstructed at only about 7--10 dB PSNR. Keep
these rows as failure evidence; do not count `status: ok` as a valid result.

Post-run disposition (2026-08-23):

- FSZ exit 1 is a completed result with a valid artifact and a failed native
  bound check, not a harness failure. The adapter now decompresses and measures
  it; exit 2+ remains a hard CLI/file error. These cells should be rerun.
- FZGM linear quantization could wrap an int32 bin on offset-valued fields at
  tight NOA bounds. FZGM now detects and refuses the overflow. Rerun affected
  linear/cuszp-style cells; a refusal means the configuration needs wider codes,
  centering, an outlier-capable mode, or a looser bound.
- Existing representable-spacing and outlier-capacity failures are intentional
  correctness guards, not adapter faults. Rescue them only with a scientifically
  equivalent configuration; do not suppress the guards.
- Native cuSZ-Hi produced illegal-memory-access/invalid-argument failures and
  explicit outlier-buffer overflow. Native cuSZ also returned severe quality
  violations under tight bounds. Preserve these as unsupported/failure evidence
  until their upstream implementations are changed; an adapter-only rerun will
  not repair them.
- Lossy expansion (`CR <= 1`) is now reported but retained when quality is valid.
  Tight bounds can legitimately expand data; deleting those rows biases CR.

## 1. FZGM vs native error-bounded compressors

### Existing standard-bound baseline (`EBLC-BASE`)

The authoritative config is
[`fzgm_vs_native_full.yaml`](../configs/experiments/fzgm_vs_native_full.yaml):
13 datasets / 186 fields at range-relative bounds `1e-2`, `1e-3`, and `1e-4`.

| Machine | Session/baseline | Cells | Outcome | Scope |
|---|---|---:|---:|---|
| H100 | `h100-jetstream2-20260808-fullcorpus-postfix` | 9,684 | 9,401 ok / 283 failed | Current native + FZGM reference after precision/Huffman fixes |
| A100 | `a100-bigred200-fullcorpus-slurm7828800` | 9,684 | 9,416 ok / 268 failed | Cross-machine native + FZGM reference |
| H200 | `h200-delta-fullcorpus-20559887` | 4,860 | 4,860 ok | FZGM half only |
| MI100 | `mi100-delta-fullcorpus-20568712` | 4,860 | 4,752 ok / 108 failed | FZGM HIP; failures are accepted GInterp LDS-limit cases |

The H100/A100 failure sets include deliberate FZGM outlier-overflow refusals
and native-tool failures; retain their disposition with the curated baselines
rather than silently dropping them.

### Tight bounds (`EBLC-TIGHT`)

[`tight_bounds_full_gpu.yaml`](../configs/experiments/tight_bounds_full_gpu.yaml)
contains 7,080 intended cells across all 13 datasets at `1e-6` and `1e-7`,
covering the GPU native/FZGM pairs and FSZ. Its first H100 session started on
2026-08-22; see the active-session table above.

Recommended interpretation:

- `1e-6` is the required extension of the accuracy curve.
- `1e-7` is a stress/failure-finding tier unless the resulting precision is
  scientifically required. Report unsupported or ineffective bounds, rather
  than treating all failures as infrastructure defects.
- Run on H100 first. Promote a second-machine rerun only after inspecting
  failures, runtime, and whether `1e-7` adds useful separation.

[`tight_bounds_full_references.yaml`](../configs/experiments/tight_bounds_full_references.yaml)
adds SZ3, zfp, MGARD-X, SPERR, MANS, and lsCOMP (2,188 intended cells). Keep its
CPU and GPU timings stratified; this is coverage, not a claim that CPU wall time
is directly comparable to GPU kernel time.

### Error-bound semantics (`EBLC-MODES`)

Use semantic names in reports; tool labels are inconsistent:

| Benchkit mode | Scale | Important note |
|---|---|---|
| `abs` | absolute tolerance | Direct only when the adapter supports absolute bounds |
| `rel_range` | `eb * (max - min)` | Often called `REL`; FZGM calls this `NOA` |
| `rel_maxabs` | `eb * max(abs(x))` | FZGM native `REL` semantics |

Before a mixed-mode campaign, record which adapters implement each mode
natively, which are converted to an absolute tolerance, and which cannot make
the requested guarantee. Never pool modes solely because their CLI option is
named `REL`.

A log transform is a separate pipeline experiment, not another error-bound
type. It remains blocked until the corpus subset and handling of zeros,
negatives, non-finite values, inverse transformation, and original-domain error
validation are specified.

### Compressor scope notes

- FSZ has a completed H100 subset comparison and a full tight-bound pair ready.
  Its adapter uses codec-reported timings so setup/file-wrapper time is not
  charged selectively; the same timing layer must be used for both sides of a
  reported comparison. See [`adapters/fsz.md`](adapters/fsz.md).
- cuSZp2 and cuSZp3 already have native/FZGM pairs in the main baseline.
- SZ3 is supported as a standalone CPU reference and structural analogue; it
  is not an exact native implementation of an FZGM pipeline.
- “SZx/SZp” is not yet a runnable item beyond cuSZp2/3. Record the exact project,
  commit/version, executable, precision support, and intended FZGM pairing.

## 2. FZGM vs nvCOMP

### Completed scientific-subset evidence

- [`nvcomp_stage_parity.yaml`](../configs/experiments/nvcomp_stage_parity.yaml)
  established 54/54 bit-exact stage-parity cells on H100. See
  [`nvcomp-stage-parity.md`](results/nvcomp-stage-parity.md).
- The broad raw and quantized-code campaigns completed 885/885 cells on H100
  with nvCOMP 5.3.0.16: 210 raw-lossless and 675 Lorenzo-quantized-code cells.
  See [`nvcomp-parity-broad.md`](results/nvcomp-parity-broad.md).
- Current scientific coverage is a selected set of CESM-2D, HURR, NYX,
  MIRANDA-f64, and EXAALT-1D fields.
- FZGM ANS, GPULZ, GPULZ+Huffman, and GPU-Zstd were compared with appropriate
  nvCOMP ANS/LZ4/Deflate/GDeflate/Zstd candidates. Snappy, Gzip, Bitcomp, and
  Cascaded are retained as unpaired native baselines where applicable.

Raw-byte lossless ratios and ratios over quantized integer codes answer
different questions and must remain separate. Bitcomp and Cascaded are
type-aware lossless codecs, not error-bounded lossy compressors. A complete
end-to-end lossy predictor-to-nvCOMP comparison has not yet been established.

### General-lossless corpus expansion (`NVC-CORPUS`)

The following are candidate categories, not approved datasets:

| Category | Candidate representation | Selection concerns |
|---|---|---|
| Logs | Raw public log streams | License, timestamp/identifier policy, chronology, realistic chunk sizes |
| Genomics | Raw FASTA/FASTQ; optionally uncompressed SAM | Avoid `.gz`, BAM, and CRAM as primary inputs because they pre-compress or transform the comparison |
| General bytes | Established corpora such as Silesia/Canterbury | Provenance, redistribution, age, and whether files are large enough for GPU timing |
| Scientific integers | Masks, IDs, labels, connectivity, quantized codes | Preserve type/shape semantics and do not pool with raw-byte streams |

Admission gate: public and redistributable provenance, raw/checksummed inputs,
no codec-specific preprocessing, enough data for stable timing, and documented
chunk/request sizes. Because backend behavior depends on request size, include
at least a small and a throughput-scale chunk tier rather than reporting only a
whole-file result.

## 3. Standalone third-party EBLC

These adapters are useful even without an exact FZGM match:

| Adapter | Current execution | Intended role |
|---|---|---|
| FSZ | GPU, f32/f64 | Paired FZGM comparison and standalone baseline |
| SZ3 | CPU | General standalone reference |
| zfp | CPU | Error-bounded standalone reference |
| MGARD-X | CUDA | GPU standalone reference |
| SPERR | CPU, 2D/3D | Standalone reference with dimensional constraints |
| MANS | CPU quantization wrapper | Component/reference experiment; label timing boundary explicitly |
| lsCOMP | GPU integer backend plus CPU quantization wrapper | Component/reference experiment; separate GPU codec from wrapper time |

All have working H100 smoke coverage through the adapter suite, including
[`smoke-cpu-refs.yaml`](../configs/experiments/smoke-cpu-refs.yaml). The remaining
work is a standard-bound general-corpus campaign and cross-machine installation
verification, not basic adapter creation.

Future additions such as cuZFP should get their own adapter identity and
provenance even if a CPU zfp adapter exists; do not treat CPU zfp results as a
proxy for a CUDA implementation.

## 4. Feature-highlighting campaigns

### Adaptive Huffman (`FEAT-HUFF`)

Subset evidence on H100 already shows about a 1.14x compression-throughput
geometric mean for the tested adaptive-Huffman set with unchanged quality; the
benefit is size-dependent. GPU-Zstd shapes showed roughly 1.09--1.18x, while
Huffman on literal streams was nearly free and Huffman on codes had a small,
data-dependent compression-ratio cost. Evidence:
[`new-stage-features-h100.md`](results/new-stage-features-h100.md) and
[`adaptive-huffman-gpu-zstd.md`](results/adaptive-huffman-gpu-zstd.md).

The requested full-corpus cuSZ and/or cuSZ-Hi CR-mode ablation is still an
`IDEA`. Its config must hold predictor, error mode, backend, launch geometry,
and repetition policy fixed while toggling only adaptive Huffman. Report CR,
compress/decompress throughput, peak memory, and quality; stratify by input size.

### Other adaptive features

- cuSZp2/3 plain-versus-outlier modes already exercise outlier selection
  (`outlier_selection=false/true`). A new campaign should name the exact
  adaptive-bitpack control and avoid duplicating this comparison under a new
  label.
- AdaptiveLorenzo versus fixed Lorenzo has subset FSZ evidence and can be
  promoted within `EBLC-FSZ` if the coder and timing boundary are held fixed.
- GInterp TP versus CR changes more than one backend choice and is not a clean
  “adaptive on/off” ablation. `FEAT-GINTERP` stays an idea until an independent
  switch is identified.
- The measured `auto_shift` subset was negative; preserve that killed
  hypothesis rather than rerunning it without a changed mechanism.

## Immediate run queue

1. Monitor the active H100 `EBLC-TIGHT` session. Treat `1e-6` as the accuracy
   extension and `1e-7` as a failure-finding stress tier during interpretation.
2. Build the adapter capability table for `abs` and `rel_maxabs`; generate only
   semantically legal cells.
3. Author the controlled full-corpus `FEAT-HUFF` cuSZ/cuSZ-Hi preset.
4. Select and register the raw external lossless corpus before expanding
   `NVC-CORPUS`.
5. Author a standard-bound `3P-CORPUS` config; keep it distinct from tight-bound
   stress testing and stratify CPU/GPU timing.

## Per-run record template

Copy one row per session (or link a result document when details do not fit):

| Field | Value |
|---|---|
| Campaign ID | `EBLC-TIGHT` |
| Session ID | Stable unique name |
| Status | `RUNNING`, `PARTIAL`, or `COMPLETE` |
| Machine ID / exact host |  |
| GPU / CPU |  |
| Git commit + dirty diff |  |
| Config + config checksum |  |
| Dataset manifest + checksums |  |
| Tool/library versions |  |
| Intended / ok / failed cells |  |
| Accepted failure classes |  |
| Timing boundary and repetitions |  |
| Session path |  |
| Curated baseline path |  |
| Result note |  |
| Follow-up |  |
