# Benchmark Plan

Living document tracking what benchmark work has been done, what's in flight, and
where we're taking it next. Complements `docs/DESIGN.md` (architecture/roadmap) and
`docs/experiments/*.md` (dated result write-ups) — this doc is the higher-level map
across experiment *categories*, updated as direction changes rather than per-run.

---

## Benchmark categories

We've settled on two main categories of benchmark, plus smoke tests as a prerequisite
gate for both.

### A. Smoke tests (infra validation, not a result in themselves)

Confirm the loop (`config → adapter → metrics → JSONL → report`) works for a given
compressor/adapter before trusting any real numbers from it.

| Config | Covers | Status (2026-07-02) |
|---|---|---|
| `configs/experiments/smoke.yaml` | M1 local sanity, FZGM only | Previously validated (M1) |
| `configs/experiments/smoke-bigred200.yaml` | HPC sanity, fzgpu + cusz FZGM pipelines | ✅ 8/8 on BigRed200 A100 |
| `configs/experiments/smoke-cusz.yaml` | Native cuSZ vs `fzgm:cusz` pairing, plumbing check | ✅ 8/8; PSNR matches exactly, CR differs up to ~28% (HACC 4.92 native vs 6.32 fzgm) — flagged for follow-up, not a bug |
| `configs/experiments/smoke-m3-refs.yaml` | All 5 ready native adapters (pfpl, cuszhi, cuszp2, cuszp3, fzgpu) | ✅ 20/20 after fixing a cuSZp adapter bug (see below) |

**Run 2026-07-02 findings:**
- **Fixed:** cuszp2/cuszp3 failed 8/8 cells on first run — `_parse_speed()` in
  `benchkit/adapters/cuszp.py` hardcoded 3 spaces before "end-to-end speed:" for both
  phases, but cuSZp only pads `compression` (to align with the longer word
  `decompression`, which gets 1 space). Fixed to a whitespace-tolerant regex; re-run
  confirmed 20/20 pass with identical CR/PSNR to before. See `observed_errors.md` E7.
- **Open (not a harness bug):** `fzgpu` reference adapter deterministically overshoots
  its own error bound by 0.18% on CESM-2D/CLDHGH (`err_over_bound: 1.00183`,
  bit-identical across two independent re-runs) — all other fields pass. Likely a real
  quantization-boundary property of the FZ-GPU tool on this specific field, not
  measurement noise (ruled out via re-run). Matters for fairness: CR comparisons "at
  the same eb" aren't strictly apples-to-apples if FZ-GPU quietly overshoots. See
  `observed_errors.md` E8.
- **Confirmed correct by design:** timing methodology is consistent where it needs to
  be — pfpl, cuszhi, cuszp2, cuszp3, and both cusz variants all use CUDA-event
  device-only timing (directly comparable); only the `fzgpu` reference adapter uses
  wall-clock e2e (documented, not comparable to the others' throughput numbers — CR/
  quality still comparable). Small fields (CESM/CLDHGH, 25.9 MB) show the most
  wall-clock variance, as expected for wall-clock timing of fast kernels.
- `mans` remains a stub (not in the smoke matrix) — lossless-only, needs a
  quantization wrapper before it's comparable to the others.

### B. FZGM vs Native comparison

**Question:** for the same algorithm, how close is the FZGM modular port to the
original compressor on CR, quality (PSNR/NRMSE/eb-satisfaction), and throughput?

- Adapters wired (M3): `pfpl`, `cuszhi`, `cuszp2`, `cuszp3`, `fzgpu` fully working;
  `mans` is a stub (lossless-only tool, needs a quantization wrapper before it's
  comparable).
- Smoke-tested via `smoke-cusz.yaml` / `smoke-m3-refs.yaml` — plumbing confirmed.
- **Done (2026-07-02):** `configs/experiments/fzgm_vs_native.yaml` — first real
  cross-tool validation pass, 7 algorithms (cuSZ, cuSZ-Hi TP, cuSZ-Hi CR, cuSZp2,
  cuSZp3, FZ-GPU, PFPL; MANS excluded, stub), one field per shape class, eb=1e-3
  rel_range. Results: **cuSZ / cuSZp2 / FZ-GPU validated** (CR/PSNR track closely);
  **cuSZ-Hi CR mode entirely broken** (upstream FZGPUModules RRE crash, E10);
  **cuSZ-Hi TP mode has a real ~65% CR deficit** + an eb violation (E11); **PFPL has
  a real 22–45% CR deficit** at matched quality (E13); **cuSZp3 preset is 2-D-only**,
  mismatched against 1-D HACC (E12); native cuSZp2 separately violates its own eb on
  HURR/TC, unrelated to FZGM (E14). Also required two harness fixes to even run
  cleanly: `benkit/pipelines.py` now renders `[pipeline]` `dims`/`input_size` per
  field (previously hardcoded placeholders broke any preset that declared them), and
  the `cuszhi` adapter gained `-s cr`/`-s tp` mode support. Full comparison tables:
  see the validation artifact from that session (per-algorithm CR/PSNR/throughput,
  linked in that conversation) — not yet copied into a `docs/experiments/*.md` file.
- **Retest (2026-07-02, same day):** Skyler pushed upstream FZGPUModules preset fixes
  for cuSZ-Hi (`memory_strategy` PREALLOCATE→MINIMAL) and PFPL (`Bitshuffle
  element_width` 2→4). Mixed outcome:
  - **PFPL: mostly fixed.** CR now matches native on 3/4 fields (CESM, HURR, HACC —
    was 22–45% low, now within noise or slightly ahead). **But NYX/temperature now
    decodes to garbage** (PSNR −84.64 dB, `err_over_bound` ~4.3 billion×) — a silent
    correctness regression, no crash, `status: ok`. New: **E15**. Don't roll
    `element_width=4` out further until this is fixed.
  - **cuSZ-Hi CR: still fully broken**, same RRE illegal-memory-access, just at a
    different line/phase. The MINIMAL switch didn't touch the actual bug.
  - **cuSZ-Hi TP: regressed.** CESM defect is bit-identical to before (confirms it's
    independent of memory strategy); HURR and NYX, which previously *passed* (with
    the ~65% CR deficit), now **crash outright** in RRE/RZE. 2/3 previously-working
    fields broke.
- **Retest 2 (2026-07-02, same day) — root cause found and fixed, FZGPUModules commit
  `8e581a8`.** Real fixes, not preset tuning this time: (1) `rre_stage.cu`/`rze_stage.cu`
  encode kernels were missing a `__syncthreads()` before copy-out, racing the last writers
  and producing non-deterministic compressed blobs — this was the root cause of **both**
  E10's crash and E15's silent corruption (E15 was misdiagnosed as a `Bitshuffle` bug;
  actually `RZEStage`, shared with `pfpl.toml`). (2) `ginterp_stage.cu` left the
  outlier-port tail uninitialized, so a recycled memory pool fed garbage into `-b`'s rep
  2+ — root cause of the E11 `-b`-only crash. (3) `dag.cpp`'s `MINIMAL`-mode buffer-restore
  had a related sizing bug on rep 2+. (4) The cuSZ-Hi presets' placeholder `dims` field was
  removed entirely — it was overriding the CLI's real `-l` dims via `setDims()` this whole
  time, previously masked in our testing because benchkit's own dims-rendering fix (above)
  happened to substitute the correct value. **Confirmed fixed:**
  - **PFPL: all 4 fields now match native** (CESM 0.0%, HURR +0.6%, NYX −0.6%, HACC 0.0%
    CR delta). E13 and E15 both closed.
  - **cuSZ-Hi CR: no longer crashes** — CESM and HURR now match native almost exactly
    (13.26/13.30, 41.18/41.21); NYX has a real but much smaller ~36% CR gap remaining
    (new: **E17**). HACC still unsupported (1-D, expected, E9). E10 closed.
  - **cuSZ-Hi TP: no longer crashes on HURR/NYX**, and NYX's corruption is gone (PSNR back
    to 75.75 dB). **But the original ~63–67% CR deficit on all 3 fields is completely
    unchanged** — this was never touched by the crash/corruption fixes; still open (E11
    partially closed — crash/corruption resolved, CR deficit is a separate remaining bug).
  - **New cross-cutting finding (E16):** the tiny (~0.19%) eb overshoot on CESM/CLDHGH
    (`err_over_bound = 1.0018662878045856`, exact to 16 sig figs) now confirmed identical
    across 4 independent tools/pipelines (native fzgpu, FZGM pfpl/cuszhi_tp/cuszhi_cr) —
    almost certainly a harness-side `eb_abs_effective` precision question or a genuine
    property of this field's data, not a per-tool bug. Low priority given the magnitude.
- **Retest 3 (2026-07-03, FZGPUModules commit `bb96edb`) — E11 fixed.** Root cause: the
  `cusz_hi_tp.toml` `GInterp` stage used `code_type = "uint16"` / `quant_radius = 32768`, but
  native cuSZ-Hi's actual spline error-control emits 1-byte codes with `radius = 128` — the
  FZGM preset was using 2× the code width with no outlier routing. Fixed to
  `code_type = "uint8"` / `quant_radius = 128`, matching native exactly. **CR deficit closed
  to within a few percent on all 3 fields** (CESM −6.5%, HURR −1.5%, NYX +0.3% — actually
  ahead of native), PSNR unchanged. **E11 closed** — only the pre-existing E16 eb-overshoot
  pattern remains on CESM, unrelated to this fix. **E17 (cuSZ-Hi CR mode's NYX gap) was not
  addressed** — `cusz_hi_cr.toml` still uses the old `uint16`/wide-radius config; same root
  cause family suspected but not yet applied there.
- **Current state of the FZGM-vs-native validation: 6 of 7 algorithms fully validated**
  (cuSZ, cuSZ-Hi TP, cuSZ-Hi CR, cuSZp2, FZ-GPU, PFPL all within single-digit % of native CR
  at matched quality); **cuSZp3 partial** (2-D/3-D fine, 1-D preset mismatch, E12); one small
  open gap (E17, cuSZ-Hi CR mode on NYX specifically); one low-priority cross-tool curiosity
  (E16). This is a dramatically different picture than the first pass on 2026-07-02, which
  had 2 of 7 algorithms broken/regressed and 2 more with large real CR deficits.
- **Still not done:** the full SDRBench matrix (`sdrbench.yaml` / `sdrbench-miranda.yaml`,
  144 fields × 3 bounds × 2 pipelines = 864 cells) — configured for SLURM array
  submission but no results doc exists yet. This is the "real" paper-scale run that
  the smoke tests and this validation pass are gating.

### C. Scalability (data size vs. failure mode)

**Question:** as input size grows, where does each pipeline/memory-strategy
combination fail, and why?

- `configs/experiments/fzgm_scaling.yaml` — run and fully documented in
  `docs/experiments/fzgm_scaling_2026-06-30.md`. Swept synthetic 0.5–16 GB data
  across `cusz`, `fzgpu`, `cusz_minimal`, `cusz_minimal_lowoutlier`.
- `configs/experiments/fzgm_scaling_v2.yaml` — run, added `fzgpu_minimal`, `pfpl`,
  `pfpl_minimal`. Failure modes catalogued in `docs/experiments/observed_errors.md`
  (E1–E6): one real bug fixed (missing stage type in `fzgmod-cli` decompression),
  several expected OOM ceilings characterized, one **open upstream bug** (E4 —
  illegal memory access in RZE decompression ≥8 GB, unresolved).
- Key findings so far: fzgpu/BitplaneRZE beats cusz/Huffman on CR (194 vs 30) and
  throughput; MINIMAL + reduced `outlier_capacity` is the main lever for pushing the
  VRAM ceiling; Huffman's worst-case output buffer is a hard ~12 GB wall regardless
  of memory strategy.
- Open items tracked at the bottom of `fzgm_scaling_2026-06-30.md` (fzgpu+MINIMAL
  follow-up, lowoutlier@12GB, capturing 12/16 GB single-shot throughput numbers).

---

## Gaps / open questions to resolve

- [x] Smoke tests for all registered compressors (2026-07-02) — all pass; see the
      smoke test table above. One real adapter bug found and fixed (E7); one open
      tool-level question flagged (E8, fzgpu eb overshoot on CESM/CLDHGH).
- [x] A full FZGM-vs-native *comparison* run (2026-07-02, `fzgm_vs_native.yaml`) —
      done for 7 algorithms × 4 fields. cuSZ/cuSZp2/FZ-GPU validated; cuSZ-Hi CR
      broken (E10), cuSZ-Hi TP (E11) and PFPL (E13) show large real CR deficits in
      the FZGM port; cuszp3 preset is 2-D-only (E12); native cuszp2 has its own eb
      bug on HURR/TC (E14). Not yet turned into a `docs/experiments/*.md` write-up.
- [x] Root-cause attempt on E10/E11/E13 (2026-07-02 retest, upstream preset fixes) —
      **PFPL mostly fixed** (3/4 fields now match native CR) but surfaced a new,
      more serious **silent correctness bug on NYX/temperature** (E15, no crash,
      PSNR −84.64 dB) — don't consider E13 closed until E15 is fixed. **cuSZ-Hi CR
      still fully broken** (E10, same bug, different line). **cuSZ-Hi TP regressed**
      — 2 of 3 previously-working fields now crash outright.
- [x] Real fix + retest 2 (2026-07-02, commit `8e581a8`) — **E10, E13, E15 confirmed
      fixed**; **E11 partially fixed** (crash/corruption resolved, but the original
      ~63–67% CR deficit on all 3 fields is unchanged — a genuinely separate bug,
      still open). New: **E16** (cross-tool eb overshoot, low priority) and **E17**
      (cuSZ-Hi CR mode ~36% CR gap on NYX, smaller than E10's original finding).
- [x] Root-cause + fix E11 (2026-07-03, commit `bb96edb`) — the `cusz_hi_tp.toml`
      `GInterp` stage's `code_type`/`quant_radius` (`uint16`/32768) didn't match
      native cuSZ-Hi's actual spline error-control (`uint8`/128). Fixed; CR deficit
      closed to within a few percent on all 3 fields (NYX now slightly ahead of
      native). **E11 closed.**
- [ ] Root-cause E17 (cuSZ-Hi CR mode, ~36% CR gap on NYX/temperature specifically,
      CESM/HURR are fine) — `cusz_hi_cr.toml` still uses the pre-fix `uint16`/wide
      radius config that caused E11; same root-cause family suspected (see the E17
      update in observed_errors.md) but not yet applied to this preset.
- [ ] Investigate E16 (cross-tool eb overshoot, CESM/CLDHGH, identical across 4
      tools) — check whether it's a harness-side `eb_abs_effective` precision issue
      or a genuine property of this field's data. Low priority (0.19% overshoot).
- [ ] Re-verify E4 (RZE decompression illegal memory access at ≥8 GB scale) against
      the fixed build — commit `8e581a8` fixed a related `rze_stage.cu` race
      condition, but at SDRBench field sizes (≤512 MB), not E4's multi-GB scale;
      needs `fzgm_scaling_v2.yaml` re-run to confirm before closing.
- [ ] `mans` adapter still a stub — decide if/when it's worth the quantization
      wrapper to bring it into the comparison category.
- [ ] No 1-D FZGM preset exists for cuszp3/cuSZ-Hi (E12) — either add one or
      permanently exclude 1-D fields from those pairings.
- [ ] SDRBench full matrix (864 cells) not yet submitted — now that E10/E13/E15 are
      closed and the remaining fidelity gaps (E11, E17) are well-scoped, this is
      much closer to ready; still blocked on the E4 re-verification above if pfpl
      is in scope for larger fields.

---

## Future direction

<!-- To be filled in as we plan next steps. -->
