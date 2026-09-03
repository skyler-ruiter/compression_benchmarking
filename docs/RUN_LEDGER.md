# Benchmark Run Ledger

Last updated: 2026-09-03

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
| `EBLC-SPEC` | Performance-first FZGM Pipeline Specialization vs native, including cuSZp3 fixed | `PARTIAL` | Refreshed off/auto smoke complete on H100; full matrix status on other machines not yet reconciled here | Preserve the clean H100 gate, identify every remote full-session ID/shard, then merge and verify the 186-field pairs; retain no compressed or decompressed artifacts; use FZGM-only pool peaks for memory |
| `EBLC-TIGHT` | GPU pairs plus FSZ at `1e-6` and `1e-7` | `COMPLETE-DIAGNOSTIC` | H100 | Preserve the 55 explicit refusals as failure evidence; accept four f64 `PRES` ratios of `1.000000080109` as marginal relative-mode boundary noise and continue to `EBLC-MODES` |
| `EBLC-MODES` | `abs`, range-relative, max-absolute-relative comparisons | `IDEA` | Range-relative baseline only | Build a capability/semantics matrix before authoring configs |
| `EBLC-LOG` | Positive-domain log transform plus EBLC | `BLOCKED` | None | Specify transform, zero/negative policy, inverse, and original-domain quality gates |
| `EBLC-FSZ` | Native FSZ vs matching FZGM pipeline | `COMPLETE-DIAGNOSTIC` | H100 full corpus | Curate the 544 reliable pairs; preserve six native constant-field failures and target only eight timing-unreliable cells if publication requires them |
| `EBLC-SZX` | Native SZx comparison | `SUPERSEDED` | None | Out of the current paper scope; revisit only after a supported modular SZx pipeline exists |
| `NVC-PARITY` | FZGM lossless stages vs closest nvCOMP counterparts | `COMPLETE` | H100 scientific subset | Keep bit-exact stage gate with future backend changes |
| `NVC-CORPUS` | Broader general-lossless corpus | `PARTIAL` | H100 scientific subset only | Select, license, checksum, and register raw log/genomic/general-byte datasets |
| `NVC-PROP` | Unpaired nvCOMP Bitcomp/Cascaded and other native codecs | `COMPLETE` for current subset | H100 | Extend with `NVC-CORPUS`; do not imply an FZGM pair |
| `EBLC-SPERR-GPU` | FZGM's new GPU SPERR pipeline (CDF97->Quantizer->Cdf97OutlierCorrect->SPECK2D, no Tee since 2026-08-29) as a comparable GPU EBLC | `COMPLETE` for both `abs`-mode bound verification (16/16 `eb_ok=True`) and the `rel_range` cross-tool CR/PSNR/throughput table (12/12 `eb_ok=True`, fixed 2026-08-29) | H100: bound-guarantee smoke 16/16 `eb_ok=True` (`sperr_gpu_bounded_smoke.yaml`); `rel_range` cross-tool table 60/60 ran, 12/12 `fzgm_sperr_gpu` cells `eb_ok=True` (`sperr-gpu-relrange-fixed-20260829`, harness-level rel_range->abs conversion in `FzgmAdapter._prepare_toml`) — see sec.3.5 | None — closed. `benchkit verify` flags this smoke session as not publication-grade (ad-hoc `FZGMOD_CLI` build provenance, one cv-unreliable row); re-run through the pinned `build_benchmarking` tree before citing numbers in a paper |
| `3P-ADAPTERS` | FSZ, SZ3, zfp, MGARD-X, SPERR, MANS, lsCOMP execution support | `COMPLETE` for H100 smoke | H100 | Re-smoke on other CUDA machines as needed |
| `3P-CORPUS` | Standalone third-party EBLC on the general scientific corpus | `RUNNING` | H100 gate complete; full corpus active | Complete the 3,282-cell standard-bound session; keep CPU, GPU, and mixed-wrapper timing strata separate |
| `FEAT-HUFF` | Adaptive Huffman feature highlight | `COMPLETE-DIAGNOSTIC` | H100 full f32 corpus | Deferred from the current rerun while the cross-timestep/refit hypothesis is refined |
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
| `EBLC-TIGHT` linear postfix | `tight-fzgm-linear-postfix-20260826-skyler-h100` | `js2-h100` | 2026-08-26 19:46 UTC | 1,488 | `COMPLETE-DIAGNOSTIC`: 992 ok / 496 explicit refusals; 12 signed-code overflows caught, 484 representable-spacing refusals, 784 severe bound misses, 28 marginal misses, 192 validity-usable rows. H3 eligible and complete coverage, but H4 correctly fails until failures/exclusions are resolved or explicitly accepted. Config: `configs/experiments/tight_bounds_fzgm_linear_postfix.yaml`; log: `/home/exouser/benchkit-results/_run_logs/tight-fzgm-linear-postfix-20260826-skyler-h100.log` |
| `EBLC-TIGHT` strict arithmetic smoke | `tight-quantizer-precision-smoke-v2-20260827` | `js2-h100` | 2026-08-27 15:55 UTC | 18 | `COMPLETE-GATE`: 18/18 status ok and 18/18 requested bounds satisfied across captured CESM FLNTC, HURR V, and NYX velocity_z failures at `1e-6`/`1e-7`; includes strict-double cuSZp2/cuSZp3 and strict-double+power2 cuSZp2. Results: `/home/exouser/benchkit-results/tight-quantizer-precision-smoke-v2-20260827` |
| `EBLC-TIGHT` strict full diagnostic | `tight-quantizer-precision-full-20260827` | `js2-h100` | 2026-08-27 16:06 UTC | 1,860 | `COMPLETE-DIAGNOSTIC`, superseded: 1,805 ok / 55 explicit failures; all successful rows pass Benchkit's bound gate. Exact auditing found an at-most `1.000000080109` ratio on a few f64 rows because the runtime requested bound was stored as f32 metadata. Preserve for diagnosis, but use the final rerun below for comparisons. Results: `/home/exouser/benchkit-results/tight-quantizer-precision-full-20260827`; log: `/home/exouser/benchkit-results/_run_logs/tight-quantizer-precision-full-20260827.log` |
| `EBLC-TIGHT` strict final rerun | `tight-quantizer-precision-final-20260827` | `js2-h100` | 2026-08-27 18:45 UTC | 1,860 | `COMPLETE-DIAGNOSTIC`: 1,805 ok / 55 explicit failures (40 reconstruction-spacing refusals and 15 signed-code overflows); 1,799 timing-reliable rows. Benchkit's bound gate passes every successful row. A stricter audit finds four f64 S3D `PRES` rows at ratio `1.000000080109`; accepted for relative-mode interpretation as immaterial boundary noise, with no rerun. Power2 completes 369/372 and avoids every spacing refusal. Config: `configs/experiments/tight_bounds_quantizer_precision_full.yaml`; results: `/home/exouser/benchkit-results/tight-quantizer-precision-final-20260827`; log: `/home/exouser/benchkit-results/_run_logs/tight-quantizer-precision-final-20260827.log` |
| `FEAT-HUFF` smoke | `feature-huffman-smoke-20260828` | `js2-h100` | 2026-08-28 01:06 UTC | 8 | `COMPLETE-GATE`: 8/8 ok with matched quality in every PerBlock/Adaptive pair. Adaptive compression throughput improved on all four smoke comparisons; cuSZ CR was unchanged and cuSZ-Hi CR moved slightly. Results: `/home/exouser/benchkit-results/feature-huffman-smoke-20260828` |
| `FEAT-HUFF` full f32 | `feature-huffman-full-f32-20260828` | `js2-h100` | 2026-08-28 01:07 UTC | 1,896 | `COMPLETE-DIAGNOSTIC`: 1,888 ok / 8 known LorenzoQuant overflow failures; 29 timing-unreliable rows. Adaptive versus PerBlock compression geomean: cuSZ 1.181x over 483 reliable pairs, cuSZ-Hi CR 1.131x over 432; quality identical within every valid pair. Assumption: subsequent data has a sufficiently similar distribution. Results: `/home/exouser/benchkit-results/feature-huffman-full-f32-20260828` |
| `EBLC-FSZ` chained gate/full | `fsz-standard-smoke-20260828` then `fsz-standard-full-20260828` | `js2-h100` | 2026-08-28 03:30 UTC | 4 gate + 1,116 full | `COMPLETE-DIAGNOSTIC`: gate 4/4; full 1,110 ok / 6 native constant-field failures, with 8 timing-unreliable rows and 544 reliable native/FZGM pairs. FZGM device compression is 0.456x native while CR and quality match closely; this FZGM AdaptiveLorenzo pipeline is staged, not fused. Results: `/home/exouser/benchkit-results/fsz-standard-full-20260828` |
| `3P-CORPUS` repaired gate | `third-party-eblc-standard-smoke-v2-20260828` | `js2-h100` | 2026-08-28 18:51 UTC | 17 | `COMPLETE-GATE`: 17/17 ok and every requested bound satisfied across f32/2-D, f64/3-D, and f32/1-D paths. The superseded first gate exposed non-bit-exact MANS u16 output and lsCOMP partial-grid/skinny-1D failures; MANS now uses its verified u32 path and lsCOMP uses padded complete codec tiles. Results: `/home/exouser/benchkit-results/third-party-eblc-standard-smoke-v2-20260828` |
| `3P-CORPUS` full standard | `third-party-eblc-full-standard-20260828` | `js2-h100` | 2026-08-28 18:55 UTC; resumed 21:50 UTC | 3,282 | `RUNNING-RESUMED`: SZ3, zfp, MGARD-X, SPERR, MANS, and lsCOMP over 186 fields at `rel_range` `1e-2`, `1e-3`, `1e-4`. The first process stopped after 367 valid cells because adapter-owned `d_bench.bin` timing scratch accumulated to 48.98 GiB and filled `/`; 196 subsequent failure rows are retained as disk-exhaustion evidence and will be superseded by successful retries. The scratch was removed, the malformed crash-tail fragment was archived under the session logs, and resume confirmed all 367 successes were skipped. The resumed process has a scratch reaper, and the runner now centrally enforces the no-retention contract for future processes. MGARD excludes known HACC 1-D OOM; SPERR excludes unsupported 1-D families. Config: `configs/experiments/third_party_eblc_full_standard.yaml`; tmux: `thirdparty-eblc-20260828`; log: `/home/exouser/benchkit-results/_run_logs/third-party-eblc-full-standard-20260828.log` |
| `EBLC-SPERR-GPU` smoke (pre-fix) | `sperr-gpu-smoke-20260828` | `js2-h100` | 2026-08-28 02:4x UTC | 60 | `SUPERSEDED` (pre-fix diagnostic, kept for provenance): 60/60 cells ran — `fzgm:fzgm_sperr_gpu` (3-stage, no bound guarantee) alongside `fzgm:fzgm_cusz`, native `cusz`, native `cuszp2_outlier`, native `sperr`, 4 CESM-2D fields x 3 `rel_range` bounds. All 12 `fzgm_sperr_gpu` cells `eb_ok=False`, severely. This is what motivated the `Cdf97OutlierCorrectStage` fix — see sec.3.5. Results: `/home/exouser/benchkit-results/sperr-gpu-smoke-20260828`. |
| `EBLC-SPERR-GPU` smoke (post-fix, rel_range table) | `sperr-gpu-bounded-smoke-20260828` | `js2-h100` | 2026-08-28 | 60 | `PARTIAL`: 60/60 ran, `fzgm_sperr_gpu` re-pointed at the DAG-integrated bound-guaranteed pipeline (5-stage: Tee/CDF97/Quantizer/Cdf97OutlierCorrect/SPECK2D). 8/12 `eb_ok=True` (up from 0/12), but 4/12 still `False` and CR collapsed to ~0.5x everywhere — diagnosed as the `rel_range` mode itself being invalid for this pipeline (Quantizer's NOA mode rescales its bound by a coefficient-domain `value_base` internally; `Cdf97OutlierCorrectStage` has no such rescaling and always treats its `error_bound` as literal absolute, so the two stages' bounds silently diverge under `rel_range`). Kept as the CR/PSNR/throughput cross-tool table; its `fzgm_sperr_gpu`/`eb_ok` values are NOT meaningful — use the `abs`-mode session below for the actual guarantee check. Results: `/home/exouser/benchkit-results/sperr-gpu-bounded-smoke-20260828`. |
| `EBLC-SPERR-GPU` bound verification (abs mode) | `sperr-gpu-abs-bounded-20260828` | `js2-h100` | 2026-08-28 | 16 | `SUPERSEDED` (correct at the time; pipeline later redesigned, re-verified below): 16/16 ok, **16/16 `eb_ok=True`** (`sperr_gpu_bounded_smoke.yaml`, `abs` mode — the semantically correct mode for this pipeline, matching how native SPERR's own adapter is handled: native pointwise absolute, `rel_range` emulated by external conversion, never internal rescaling). 4 CESM-2D fields x bounds `1e-2..1e-5`. CR is genuinely data/bound-dependent (2.5x-106x on CLDHGH/CLDLOW; expansion, CR 0.45x-0.69x, on FLDSC/TS at the two tightest bounds) — the guarantee holds in every case, only its cost varies, exactly matching what `sperr_gpu_bounded.cu`'s standalone validation predicted. Results: `/home/exouser/benchkit-results/sperr-gpu-abs-bounded-20260828`. |
| `EBLC-SPERR-GPU` bound verification (post-Tee-removal, broken) | `sperr-gpu-generic-outliercorrect-20260829` | `js2-h100` | 2026-08-29 | 16 | `SUPERSEDED` (diagnostic, kept for provenance): 16/16 ok but **16/16 `eb_ok=False`** — same `sperr_gpu_bounded_smoke.yaml`, run immediately after replacing `TeeStage` with `Pipeline::bindExternalInput()`. Root cause was in `decompressFromFile()` (the two-process file round trip benkchit always uses), not in the live in-DAG path FZGM's own tests exercise — see sec.3.5. `max_abs_err` ~2.5-2.9x the bound on CLDHGH/CLDLOW (the uncorrected-baseline signature) and literally identical between the `1e-4`/`1e-5` cells on FLDSC/TS (correction not applying at all). Fixed FZGM-side same day; see the next row. Results: `/home/exouser/benchkit-results/sperr-gpu-generic-outliercorrect-20260829`. |
| `EBLC-SPERR-GPU` bound verification (post-Tee-removal, fixed) | `sperr-gpu-generic-outliercorrect-fixed-20260829` | `js2-h100` | 2026-08-29 | 16 | `COMPLETE`: 16/16 ok, **16/16 `eb_ok=True`**, same `sperr_gpu_bounded_smoke.yaml`, after FZGM's `.fzm` file-format primary-source flag fix. CR/PSNR figures unchanged from `sperr-gpu-abs-bounded-20260828` (pure internal refactor, no behavior change once the file-path regression was fixed). `compute-sanitizer` clean. Results: `/home/exouser/benchkit-results/sperr-gpu-generic-outliercorrect-fixed-20260829`. |
| `EBLC-SPERR-GPU` `rel_range` cross-tool table (fixed) | `sperr-gpu-relrange-fixed-20260829` | `js2-h100` | 2026-08-29 | 60 | `COMPLETE`: 60/60 ok, **12/12 `fzgm_sperr_gpu` cells `eb_ok=True`** (up from 8/12 pre-fix), same `sperr_gpu_smoke.yaml`, after `benchkit/pipelines.py`'s `has_mode_agnostic_bound_stage()` + `benchkit/adapters/fzgm.py`'s harness-level `rel_range`/`rel_maxabs`->`abs` pre-conversion for dual-basis pipelines (see sec.3.5 close-out). CR now realistic (3.54x-588.02x) instead of collapsed ~0.5x. `benchkit verify`: FAIL on `timing_reliability` (1 cv-unreliable row) and `h3_eligibility` (ad-hoc `FZGMOD_CLI` build, not the pinned `build_benchmarking` tree) — smoke-grade, not publication-grade; re-run through the pinned build before citing these numbers. Results: `/home/exouser/benchkit-results/sperr-gpu-relrange-fixed-20260829`. |

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

Current adapter capability matrix (`native` means the compressor itself applies
that semantic; `converted` means Benchkit reads the field basis and supplies an
absolute tolerance):

| Adapter family | `abs` | `rel_range` | `rel_maxabs` | Comparison note |
|---|---|---|---|---|
| FZGM | native `ABS` | native `NOA` | native `REL` | Direct three-mode reference |
| PFPL | native `ABS` | native `NOA` | native `REL` | `REL` is documented as approximate per-element maxabs mode |
| cuSZ, cuSZ-Hi, cuSZp2/3 | native | native range-relative | unsupported | Do not reinterpret their `REL`/`r2r` label as maxabs-relative |
| FSZ | native | native range-relative | unsupported | Same range basis as FZGM `NOA` |
| SZ3 | native | native range-relative `REL` | unsupported | SZ3 `REL` is not FZGM `REL` |
| FZ-GPU | unsupported | native `NOA` | not wired | Current adapter exposes only range-relative `NOA`; FZ-GPU's differently named `REL` path is not yet mapped |
| zfp | native | converted to native absolute | converted to native absolute | CPU serial accuracy-mode timing |
| MGARD-X | native with `s=inf` | converted to native absolute | converted to native absolute | Native MGARD relative norm is not either canonical relative mode |
| SPERR | native pointwise absolute | converted to native absolute | converted to native absolute | Output-cast ULP guard is included |
| MANS, lsCOMP | Benchkit quantizer | Benchkit quantizer | Benchkit quantizer | Lossless integer codecs behind a uniform-quantization wrapper; not native float EBLC modes |

Therefore a native-only three-mode comparison is limited to FZGM and PFPL. A
broader `abs`/`rel_range` campaign can include the native GPU tools; zfp,
MGARD-X, SPERR, MANS, and lsCOMP must be labelled as converted or wrapped, and
`rel_maxabs` cannot be reported as a native cross-tool sweep beyond FZGM/PFPL.

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
- `szp_composed.toml` is an FZGM-only SZp/fZ-light-inspired composition for the
  specialization ablation. It shares the upstream quantize/Lorenzo/fixed-width
  structure, but differs in quantization and predictor partition boundaries and is
  not a native-parity or container-compatibility claim.
- SZx is out of the current paper campaign because there is no supported modular
  FZGM SZx pipeline to evaluate.

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

## 3.5. FZGM GPU SPERR pipeline (`EBLC-SPERR-GPU`)

FZGM gained a full GPU reimplementation of SPERR's pipeline structure (CDF 9/7
DWT -> quantizer -> SPECK bit-plane coding, all on-device — SPECK2D's
encode/decode are both parallel, unlike the reference's serial coder). See
`FZGPUModules/memory/speck_algorithm_writeup.md` and
`FZGPUModules/memory/speck_gpu_design.md` for the derivation and the
measured raw encode/decode throughput vs. native SPERR (27x-349x). Pipeline:
`configs/pipelines/sperr_gpu.toml`.

**Not yet a fair `eb_ok` comparison.** The smoke above found the pipeline's
quantizer applies its bound directly to DWT COEFFICIENTS with no
subband/level-aware scaling and no outlier-correction pass — the two
mechanisms native SPERR actually uses to turn a coefficient-domain threshold
into a guaranteed POINTWISE bound in the reconstructed field after the
inverse transform. Consequences, both confirmed:

1. **`rel_range`/`abs` mode is severely wrong**, not marginally: FZGM's `NOA`
   (rel_range) mode computes `value_base` from its OWN stage's input range —
   here that's the DWT COEFFICIENT range, not the original field's range — so
   the effective bound bears no fixed relationship to the requested one.
   Observed CR inflation of 20x-500x over cuSZ at the "same" nominal bound,
   `max_abs_err` up to 47.3 in reconstructed-field units against an intended
   tiny bound.
2. **Direct `abs` mode is closer but still wrong**: passing the bound straight
   through to the coefficient-domain threshold ignores that CDF 9/7's
   synthesis-filter gain differs by subband/level, so a uniform coefficient
   threshold does not map to a uniform reconstructed-domain error. Measured
   miss: requested `1e-3`, achieved `max_abs_err=2.72e-3` (2.7x over) on
   CLDHGH — a real violation, not ULP-guard-scale noise like SPERR's own
   adapter emulation footnote.

**What this pipeline IS good evidence for right now:** raw encode/decode GPU
throughput (device-timed, no bound-satisfaction dependency — this is what
`memory/speck_gpu_design.md`'s 27x-349x-vs-native numbers rest on), and
CR/PSNR as an uncalibrated data point at a *reported*, not *guaranteed*,
distortion level.

**UPDATE (2026-08-28): the fix is designed and validated, but not yet wired
into the runnable TOML pipeline benchkit invokes.** Two candidates were
measured on real CLDHGH data before picking one:
- **Subband/level-scaled quantization (weight each coefficient's threshold by
  its level's synthesis-filter gain) — REJECTED.** Measured the actual CDF 9/7
  per-level gain (a real impulse-response calibration, not a guess) and tried
  it: max reconstruction error got WORSE, not better, because many
  coefficients across levels jointly influence any given pixel, so bounding
  each one's isolated worst case doesn't bound their sum.
- **Sparse outlier correction (matching native SPERR's own `Outlier_Coder`
  mechanism) — the fix.** Quantize normally; separately dequantize + inverse-
  transform a copy and compare to the original (cheap: SPECK2D is lossless
  w.r.t. the codes, so this needs no actual encode/decode round trip); every
  pixel over bound gets an exact sparse correction applied at decompress. This
  gives a mathematically exact guarantee. Validated in FZGPUModules'
  `examples/sperr_gpu_bounded.cu`: every bound now guaranteed exactly, at a
  measured cost of 0.06%-8.2% of pixels needing correction on CLDHGH/CLDLOW
  (cheap) but up to 95% on FLDSC at the tightest bound tested (correction
  stream bigger than the main archive — a real, data-dependent cost the
  mechanism surfaces honestly, not a case where the guarantee is violated).
  `compute-sanitizer` clean. Full writeup: FZGPUModules'
  `memory/speck_gpu_design.md` sec.9.

**DAG-integrated and verified (2026-08-28).** `configs/pipelines/sperr_gpu.toml`
now runs the real 5-stage pipeline: `Tee -> CDF97 -> Quantizer ->
Cdf97OutlierCorrect -> SPECK2D`. The `Tee` stage (structural, 1-in/N-out
forward, N-in/1-out inverse) exists because `Pipeline::compress()` allows
exactly one true source stage, and `buildInverseDAG()`'s wiring requires
`inverse_input_count == forward_output_count` / `inverse_output_count ==
forward_input_count` for every stage — getting the correction stage both the
ORIGINAL raw field (compress time) and the DOWNSTREAM reconstructed field
(decompress time) needs this exact shape; see FZGPUModules'
`modules/coders/cdf97_outlier_correct/cdf97_outlier_correct_stage.h` for the
full edge-by-edge trace that arrived at it (not an arbitrary topology).
Verified through `fzgmod-cli`, `compute-sanitizer` clean, full FZGM test
suite green (`tests/pipeline/test_sperr_gpu_bounded.cpp`).

**One more real finding closing this out: `rel_range` mode is invalid for
this pipeline, `abs` mode is required.** `sperr_gpu_bounded_smoke.yaml`'s
20260828 rerun through the real DAG got 8/12 `eb_ok=True` (up from 0/12) but
4/12 still failed and CR collapsed to ~0.5x everywhere under `rel_range` —
diagnosed as `QuantizerStage`'s `NOA` mode rescaling its bound internally by
a coefficient-domain `value_base`, while `Cdf97OutlierCorrectStage` has no
such rescaling and always treats `error_bound` as literal absolute — the two
stages' effective bounds silently diverge whenever `rel_range` is used. A
dedicated `abs`-mode session (`sperr_gpu_bounded_smoke.yaml`,
`sperr-gpu-abs-bounded-20260828`) confirms the fix cleanly: **16/16 ok,
16/16 `eb_ok=True`**, across 4 CESM-2D fields and bounds `1e-2..1e-5`. This
exactly mirrors how native SPERR's OWN adapter is already handled in this
repo (native pointwise absolute; `rel_range`/`rel_maxabs` emulated by
external conversion to absolute, never internal tool-side rescaling) — see
`docs/adapters/sperr.md`. CR is genuinely data/bound-dependent under `abs`
too: 2.5x-106x on CLDHGH/CLDLOW, but real expansion (CR 0.45x-0.69x) on
FLDSC/TS at the two tightest bounds — the guarantee holds in every case,
only its cost varies, exactly matching what `sperr_gpu_bounded.cu`'s
standalone prototype predicted before the DAG integration.

**CLOSED (2026-08-29): `rel_range` cross-tool table fixed via harness-level
conversion, matching the zfp/MGARD/SPERR "converted" pattern exactly.**
`PipelineToml.has_mode_agnostic_bound_stage()` (`benchkit/pipelines.py`)
detects a template where some lossy stage's `error_bound` has no
`error_bound_mode` sibling key (`Cdf97OutlierCorrect`) alongside one that
does (`Quantizer`) — the same structural situation the earlier root-cause
diagnosis above named. `FzgmAdapter._prepare_toml` (`benchkit/adapters/
fzgm.py`) now pre-converts a non-`abs` request to a literal absolute bound
itself (`eb * range` or `eb * maxabs`, via the same `read_range_stats` helper
`sperr.py` already uses) before rendering, so BOTH stages get the identical
converted value and `render_mode="ABS"` — never the raw requested eb under a
mode `Cdf97OutlierCorrect` can't interpret. `prep.eb`/`prep.basis` stay the
ORIGINAL requested values (the harness's own `metrics.py` eb check
recomputes `eb_abs = eb * basis_val` independently, so it needs the request,
not what got rendered); `native_mode` is recorded as
`ABS-emulated(rel_range)` for provenance. Verified: `sperr-gpu-relrange-fixed-
20260829` (`sperr_gpu_smoke.yaml`, H100) — 60/60 ok, **12/12
`fzgm_sperr_gpu` cells `eb_ok=True`** (up from the earlier 8/12 with 4/12
false and CR collapsed to ~0.5x everywhere), CR now realistic and
bound-dependent (3.54x-588.02x across CLDHGH/CLDLOW/FLDSC/TS x
1e-2/1e-3/1e-4), matching the `abs`-mode session's own CR/PSNR figures at
matching effective bounds. 6 new unit tests in
`tests/test_sperr_gpu_render.py` cover the detection helper and all three
render paths (rel_range, rel_maxabs, abs) plus the eb/basis-stays-unconverted
invariant. Not re-run through this repo's pinned `build_benchmarking` tree
(used `FZGMOD_CLI` pointed at FZGPUModules' own fresh `build/release` instead,
post-merge-to-main) — `benchkit verify` correctly flags the session as not
publication-grade on build provenance; re-run before citing these exact
numbers in a paper, though the fix's correctness does not depend on which
build produced them.

**REDESIGNED (2026-08-29): `TeeStage` removed, replaced with
`Pipeline::bindExternalInput()` + a genericized, transform-agnostic
correction stage.** Pushback on `TeeStage`/`Cdf97OutlierCorrectStage` as
too SPERR-specific to justify as Pipeline primitives (comparable to
`MergeStage`, which has a genuine general use case) led to two real
FZGPUModules-side changes, not just a rename:
- **`Pipeline::bindExternalInput(Stage*)`** binds the pipeline's raw input
  directly to a specific stage's input port, even when that stage has other
  real connections too (`Cdf97OutlierCorrectStage` needs the raw field on
  one port and `Quantizer`'s codes on another). No duplicate-copy node is
  needed any more — `configs/pipelines/sperr_gpu.toml` is now 4 stages,
  `CDF97 -> Quantizer -> Cdf97OutlierCorrect -> SPECK2D`, with `CDF97` and
  `Cdf97OutlierCorrect` both bound directly to the same external buffer.
  Peak device memory on CLDHGH dropped 303.5 MB -> 254.0 MB (the removed
  duplicate buffer).
- **`OutlierCorrectStage<Reconstructor>`** (`FZGPUModules/modules/coders/
  outlier_correct/`) is now a template: all the diffing/sparse-pack/apply
  logic is transform-agnostic, parametrized only by a small `Reconstructor`
  policy supplying the one CDF97-specific step (inverse-transform the
  dequantized trial). `Cdf97OutlierCorrectStage` is the one shipped
  instantiation; a future non-CDF97 bound-guarantee pipeline reuses the
  whole mechanism by writing one small policy struct, not a new stage.

**A real regression was found and fixed while re-verifying this**, not
assumed clean from the FZGM-side test suite alone: benchkit always runs
`fzgmod-cli` as two SEPARATE processes (`-z` compress-to-file, then `-x`
decompress-from-file), which goes through a DIFFERENT, static
file-header-reconstruction code path (`decompressFromFile()`) than the
in-process `Pipeline::compress()`/`decompress()` round trip FZGM's own
tests exercise. That path's "which stage is the answer" heuristic (a plain
"nothing else in the DAG produced any of my inputs" topology test) has no
way to recognize a *mixed* stage — one bound via `bindExternalInput()` on
one port and connected normally on another — as the intended answer stage;
it silently fell back to `CDF97`'s own uncorrected reconstruction instead
of `Cdf97OutlierCorrect`'s corrected one. Symptom: **16/16 `eb_ok=False`**,
with `max_abs_err` ~2.5-2.9x the bound (CLDHGH/CLDLOW — the exact
signature of the pre-fix, uncorrected baseline) and, on FLDSC/TS,
`max_abs_err` literally IDENTICAL between the `1e-4` and `1e-5` bound
cells — the tell that no correction was being applied at all, independent
of what bound was requested. Fixed FZGM-side by adding an explicit
"primary source" flag to the `.fzm` file format (`FZMStageInfo::stage_flags`,
repurposing 2 bytes of prior padding — no format-version break, absent/0 on
older archives which fall back to the original heuristic unchanged) and
having `Pipeline::writeToFile()` set it on whichever stage
`Pipeline::setPrimarySource()` designates. Re-verified via the exact
two-process compress/decompress command benchkit issues (not just FZGM's
in-process tests, which could not have caught this): `Max Abs Error`
returned to the exact requested bound, `compute-sanitizer` clean. Session
`sperr-gpu-generic-outliercorrect-fixed-20260829` confirms **16/16 ok,
16/16 `eb_ok=True`**, CR/PSNR figures unchanged from the pre-redesign
`sperr-gpu-abs-bounded-20260828` session (same guarantee, same cost
profile — this was a pure internal refactor with one real bug caught and
fixed before shipping, not a behavior change). The intermediate,
broken-decompressFromFile() session (`sperr-gpu-generic-outliercorrect-20260829`,
16/16 `eb_ok=False`) is kept for provenance below, superseded.

## 4. Feature-highlighting campaigns

### Automatic pipeline fusion (`FEAT-FUSION`)

The publication experiment records finalize-time specialization directly in
each result row (`fusion_policy`, legal/installed group counts, implementation,
stage membership, and fallback reason). `FZ_FUSION` and `FZ_FUSION_NVRTC` are
also execution-identity inputs, preventing staged and Auto rows from sharing a
resume identity.

The H100 gate uses locked clocks, five timed repetitions after one warmup, and
the standard `rel_range` bounds. Sessions
`fusion-standard-smoke-staged-paired-20260831` and
`fusion-standard-smoke-auto-valid-20260831` verified publication-grade: 36/36
controlled staged rows and 60/60 Auto rows passed validity and timing checks.
Every Auto row installed exactly one group (warp-register for cuSZp, chunk-coop
for PFPL). Across the 36 identical-semantics pairs, compression-throughput
geometric-mean speedups were 2.095x for cuSZp2 outlier, 1.321x for cuSZp3
outlier, and 1.494x for PFPL; compressed bytes and PSNR were identical in every
pair. Decompression remained near 1x because inverse-DAG fusion is not
implemented.

PlainBitpackCoder modes are fused-versus-native evidence only. Arming their
current warp op uses the adaptive archive container, while fusion-off executes
the adaptive plain/outlier stage; those are not an algorithm-controlled A/B and
must not be reported as staged/fused speedups. Float64 datasets are likewise
excluded from the fusion-performance campaign because the current device ops
are float32-only; fallback coverage belongs in a separate compatibility test.

Full-f32 sessions are launched sequentially by
`scripts/run-fusion-standard-full.sh`: staged controls first, then Auto. Native
comparisons reuse `h100-jetstream2-20260808-fullcorpus-postfix` after its
validity/provenance filters.

### Pipeline specialization refresh (`EBLC-SPEC`)

The refreshed H100 gate at FZGPUModules `6897f66` and Benchkit `808870e`
completed on 2026-09-03. Sessions
`specialization-vs-native-smoke-skyler-h100-{off,auto}-20260903` contain
252/252 `status: ok` rows per arm; the memory sessions
`specialization-memory-smoke-skyler-h100-{off,auto}-20260903` contain 96/96
`status: ok` rows per arm. Across the FZGM pairs, compression ratio and
reconstructed-output hashes are unchanged, every intended `_sp`, PFPL, and
`szp_composed` arm installs one forward and one inverse group under Auto, and
the non-specializing controls remain near 1x.

This is a successful functional/performance gate, not yet the publication
corpus. The Auto vs-native session has 12 known native error-bound exclusions,
10 timing-unreliable rows, and incomplete native build provenance; the Auto
memory session has 18 timing-unreliable rows because it is a low-repetition
smoke. The Off memory arm alone passes all current verification checks. Do not
pool these smoke rows into a headline aggregate. At the 2026-09-03 local audit,
the H100 was idle and no full specialization session directory was present
under `/home/exouser/benchkit-results`; remote machine/session status still
needs reconciliation by exact session ID.

### Adaptive Huffman (`FEAT-HUFF`, deferred)

Subset evidence on H100 already shows about a 1.14x compression-throughput
geometric mean for the tested adaptive-Huffman set with unchanged quality; the
benefit is size-dependent. GPU-Zstd shapes showed roughly 1.09--1.18x, while
Huffman on literal streams was nearly free and Huffman on codes had a small,
data-dependent compression-ratio cost. Evidence:
[`new-stage-features-h100.md`](results/new-stage-features-h100.md) and
[`adaptive-huffman-gpu-zstd.md`](results/adaptive-huffman-gpu-zstd.md).

The existing results remain diagnostic evidence, but adaptive-Huffman book reuse is
outside the current paper rerun. A future campaign must first define cross-timestep
distribution drift and refit policy; identical-field warm reuse is not sufficient.

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

1. Build the adapter capability table for `abs` and `rel_maxabs`; generate only
   semantically legal cells.
2. Author the controlled full-corpus `FEAT-HUFF` cuSZ/cuSZ-Hi preset.
3. Select and register the raw external lossless corpus before expanding
   `NVC-CORPUS`.
4. Author a standard-bound `3P-CORPUS` config; keep it distinct from tight-bound
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
