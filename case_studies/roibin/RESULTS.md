# ROIBIN dual-error-bound pipeline in FZGM — results

Companions: `notes/data_provenance.md` (data), `notes/design_and_method.md` (design,
constraints, tuning), `README.md` (layout, reproduction).

**What was measured.** A pipeline that gives Bragg-peak regions a tight error bound and
the background a loose one, in one pass, against the strongest tuned single-bound
compressors applying the tight bound globally — which is what a single-bound compressor
must do to protect the peaks.

**Corpus.** 409 detector frames from two independent LCLS serial-crystallography
experiments, 19,492 published Bragg peaks. The two are reported separately and never
pooled (one is calibrated f32, the other raw ADU from a different experiment).

**Fixed across every arm:** ABS bound mode, PREALLOCATE, `fzgmod-cli -b`, one process
per frame, warm-up rep discarded, `roi_half_width = 4` (9×9 box, justified in
`design_and_method.md` §2).

---

## 1. Configurations under test

| arm | what it is |
|---|---|
| `roibin_b1` | ROIBinSplit, `bin_factor=1` — ROI at eb 10, background at eb 100. Both bounds real. |
| `roibin_b2` | ROIBinSplit, `bin_factor=2` — the ROIBIN configuration. Background binned 2×2. |
| `cuszhi` | native cuSZ-Hi, **tuned** (8-point grid; `spline/cr-first/cr` won on both corpora), eb 10 globally |
| `pfpl` | native PFPL (LC), eb 10 globally |
| `cuszp3` | FZGM cuSZp3 preset, eb 10 globally — throughput reference |

`eb_roi = 10.0` was **forced by the baseline, not chosen**: native cuSZ-Hi aborts on
84/130 EXAFEL frames at eb=1.0 and tuning does not rescue it (§5). At eb=10 every
baseline runs on every frame, so the comparison has full coverage.

---

## 2. Results table

Aggregate CR is total original bytes / total compressed bytes; the bracketed figures are
the per-frame median and p10/p90. Compressed sizes include the peak table stored in the
archive, so the ROIBIN arms are charged for their own metadata.

### EXAFEL — 130 frames

| arm | frames | fail | aggregate CR | CR med [p10,p90] | ROI PSNR dB | bg PSNR dB | cmp GB/s | dec GB/s |
|---|---:|---:|---:|---|---:|---:|---:|---:|
| ROIBIN dual-eb (bin=1) | 130 | 0 | **11.37×** | 11.21 [10.06, 12.77] | 68.91 | 48.73 | 86.6 | 89.1 |
| ROIBIN binned (bin=2) | 130 | 0 | **50.27×** | 48.54 [42.02, 62.90] | 68.91 | 43.26 | 114.3 | 102.9 |
| cuSZ-Hi (tuned, global tight eb) | 130 | 0 | 6.70× | 6.57 [6.18, 7.36] | 68.86 | 68.86 | 26.8 | 28.2 |
| PFPL (global tight eb) | 130 | 0 | 5.84× | 5.75 [5.41, 6.35] | 68.91 | 68.94 | 175.8 | 161.3 |
| cuSZp3 (global tight eb) | 130 | 0 | 5.69× | 5.61 [5.29, 6.20] | 68.91 | 68.94 | 135.2 | 133.0 |

### CXIDB 21 — 279 frames

| arm | frames | fail | aggregate CR | CR med [p10,p90] | ROI PSNR dB | bg PSNR dB | cmp GB/s | dec GB/s |
|---|---:|---:|---:|---|---:|---:|---:|---:|
| ROIBIN dual-eb (bin=1) | 279 | 0 | **14.99×** | 14.25 [12.02, 22.61] | 65.11 | 45.88 | 87.5 | 89.3 |
| ROIBIN binned (bin=2) | 279 | 0 | **61.50×** | 57.36 [47.75, 102.55] | 65.11 | 41.98 | 115.6 | 104.1 |
| cuSZ-Hi (tuned, global tight eb) | 279 | 0 | 8.61× | 8.36 [7.46, 10.93] | 64.87 | 65.47 | 26.7 | 23.8 |
| PFPL (global tight eb) | 279 | 0 | 7.87× | 7.65 [6.88, 9.68] | 65.11 | 64.79 | 177.8 | 162.3 |
| cuSZp3 (global tight eb) | 279 | 0 | 6.70× | 6.52 [5.79, 8.32] | 65.11 | 64.79 | 137.0 | 134.3 |

### Paired per-frame comparison vs tuned cuSZ-Hi

Aggregates can hide a mixed result; this pairs by frame.

| corpus | arm | frames | CR ratio med [min, max] | frames with higher CR | ROI PSNR delta dB med [min, max] |
|---|---|---:|---|---:|---|
| EXAFEL | ROIBIN bin=1 | 130 | 1.70 [1.56, 3.16] | 100 % | +0.022 [−0.320, +1.658] |
| EXAFEL | ROIBIN bin=2 | 130 | 7.41 [5.78, 42.03] | 100 % | +0.022 [−0.320, +1.658] |
| EXAFEL | PFPL | 130 | 0.87 [0.85, 0.89] | 0 % | +0.022 |
| EXAFEL | cuSZp3 | 130 | 0.85 [0.74, 0.86] | 0 % | +0.022 |
| CXIDB 21 | ROIBIN bin=1 | 279 | 1.70 [1.54, 3.95] | 100 % | +0.274 [−0.406, +1.074] |
| CXIDB 21 | ROIBIN bin=2 | 279 | 6.87 [6.08, 20.62] | 100 % | +0.274 [−0.406, +1.074] |
| CXIDB 21 | PFPL | 279 | 0.91 [0.81, 0.96] | 0 % | +0.274 |
| CXIDB 21 | cuSZp3 | 279 | 0.78 [0.66, 0.82] | 0 % | +0.274 |

The two corpora are independent experiments, different samples, different calibration
state — and the bin=1 paired ratio lands on **1.70 in both**. That agreement is a
stronger signal than either corpus alone.

The ROI PSNR delta column is identical across the three non-cuSZ-Hi arms because every
round-to-nearest ABS quantizer at the same bound produces the *same* ROI reconstruction;
only cuSZ-Hi's spline path differs. It is not a copy-paste artifact.

---

## 3. Per-region error-bound verification

The gate that matters. A global max-abs-error on a dual-bound pipeline reports only the
looser bound and says nothing about the peaks, so ROI and background are checked
separately, and every arm — baselines included — goes through the same check.

### EXAFEL — 130 frames

| arm | ROI bound satisfied | ROI max abs err (worst frame) | bg bound satisfied | bg max abs err (worst frame) |
|---|---:|---:|---:|---:|
| ROIBIN dual-eb (bin=1) | **130/130** | 10.0000 | **130/130** | 100.0001 |
| ROIBIN binned (bin=2) | **130/130** | 10.0000 | *n/a — binned* | 10518.03 |
| cuSZ-Hi (tuned) | 130/130 | 10.0000 | 130/130 | 10.0000 |
| PFPL | 130/130 | 10.0000 | 130/130 | 10.0000 |
| cuSZp3 | 130/130 | 10.0000 | 130/130 | 10.0000 |

### CXIDB 21 — 279 frames

| arm | ROI bound satisfied | ROI max abs err (worst frame) | bg bound satisfied | bg max abs err (worst frame) |
|---|---:|---:|---:|---:|
| ROIBIN dual-eb (bin=1) | **279/279** | 10.0000 | **279/279** | 100.0000 |
| ROIBIN binned (bin=2) | **279/279** | 10.0000 | *n/a — binned* | 11054.00 |
| cuSZ-Hi (tuned) | 279/279 | 10.0000 | 279/279 | 10.0000 |
| PFPL | 279/279 | 10.0000 | 279/279 | 10.0000 |
| cuSZp3 | 279/279 | 10.0000 | 279/279 | 10.0000 |

**The ROI bound holds on 409/409 frames in every arm and every configuration.** That is
the invariant the science depends on and it is what this table exists to establish.

**`bin_factor = 2` claims no background bound.** Its background error reaches 1.05e4
(EXAFEL) and 1.11e4 (CXIDB) against a *nominal* 100 — because binning is a resolution
reduction, not an error bound: the error is binning error plus quantization error. The
gate records `n/a` rather than a pass or a fail, and background fidelity for that arm is
reportable **only** as PSNR. Quoting the bin=2 ratio beside a bound-satisfying
baseline's without this distinction would compare a lossy-resolution scheme against an
error-bounded one.

Only `bin_factor = 1` gives a background that genuinely satisfies its stated bound
pixel-wise, and it does so on every frame. That arm is the like-for-like comparison.

---

## 4. Bound sensitivity (EXAFEL, 40 frames)

| eb_roi | eb_bg | bin=1 CR | bin=1 bg PSNR | bin=2 CR | bin=2 bg PSNR | ROI PSNR | cuSZ-Hi CR |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 10 | 50 | 9.14× | 54.78 | 40.96× | 44.91 | 68.77 | 6.80× |
| 10 | 100 | 11.37×* | 48.73* | 50.27×* | 43.26* | 68.91* | 6.70×* |
| 10 | 1000 | **139.27×** | 35.38 | **397.69×** | 35.34 | 68.77 | 6.80× |
| 5 | 50 | 9.13× | 54.78 | 40.77× | 44.91 | 74.75 | 5.61× |
| 1 | 10 | 5.71× | 68.86 | 24.78× | 45.45 | 88.74 | 3.88×† |

\* full 130-frame corpus, not the 40-frame subset. † 17/40 frames only — see §5.

The ratio is governed almost entirely by the **background** bound: holding `eb_roi = 10`
and moving `eb_bg` from 50 to 1000 takes bin=1 from 9.14× to 139×, while **ROI PSNR does
not move** (68.77 dB throughout). That is the mechanism doing exactly what it was built
to do, and it is also why a loose bound must always be quoted with its background PSNR —
at `eb_bg = 1000` the background is down to 35 dB and is no longer scientifically
useful for anything but peak-finding context.

Tightening the ROI bound costs ratio as expected (5.71× at eb_roi=1) and buys ROI
quality (88.74 dB). The advantage over the baseline persists at every bound tested.

---

## 5. The baseline's failure envelope

Native cuSZ-Hi (`spline/cr-first/cr`) on all 130 EXAFEL frames:

| ABS bound | frames compressed |
|---:|---|
| 1.0 | **46/130 (35.4 %)** |
| 2.0 | 130/130 |
| 5.0 | 130/130 |
| 10.0 | 130/130 |

Hard aborts (`psz_gpu_exception`, *GPU API failed at `compressor.inl:312`, invalid
argument*), deterministic per frame, not degraded output.

**Tuning does not rescue it.** With the full 8-point grid tried per frame and the best
survivor kept: **17/40 (42.5 %)**, against 35.4 % pinned — ~7 points, not a fix. Of the
17 survivors **13 won with `-s tp` and only 4 with `-s cr`**, so the abort is
concentrated in the high-ratio LC path, which is also the setting that wins on ratio
wherever it runs.

**Selection-bias check.** A comparison at eb=1.0 would be restricted to the 46 frames
cuSZ-Hi survives, and those are not a random half — they are dimmer (frame mean median
128.8 vs 268.0) and more compressible (cuSZ-Hi CR at eb=10: 7.18 vs 6.41). Effect on the
paired claim, measured at eb=10 where all 130 run:

| paired CR ratio vs cuSZ-Hi | all 130 frames | restricted to the 46 survivors |
|---|---:|---:|
| ROIBIN bin=1 | 1.702 | 1.732 (**+1.8 %**) |
| ROIBIN bin=2 | 7.415 | 8.163 (**+10.1 %**) |

The bias runs **in ROIBIN's favour**, so reporting the full corpus at eb=10 is the
conservative choice. Note this is the *opposite sign* from the same defect on
HURR/SCALE-LETKF, where the fields cuSZ-Hi cannot process are the most compressible ones
and the survivor subset flatters the baseline. There is no general rule for which way a
coverage gap cuts; it has to be measured per study.

This is one build on one machine and should be reported as such. It is characterised
corpus-wide in `experiments.md` §F1 (38 fields abort unconditionally across 6 datasets,
range-driven; 10 are bound-dependent; the EXAFEL case is the latter class — two
triggers, possibly one defect, not demonstrated).

---

## 6. Whole-volume run

Per-frame runs never exercise `nz > 1`. The full 130-frame volume (1.19 GB, all 13,837
peaks, one pipeline, `bin_factor = 2`):

| metric | value |
|---|---|
| compression ratio | **50.95×** (vs 50.27× per-frame aggregate) |
| ROI max abs err | **10.0000** — bound satisfied across all 130 slices |
| ROI PSNR / background PSNR | 69.15 dB / 44.02 dB |
| compress / decompress | 601.3 / 459.9 GB/s |
| peak device memory | 3.96 GB for a 1.19 GB input (3.3×) |

The ratio agreeing with the per-frame aggregate is the check that the `z` indexing is
right — a broken slice stride would show up as a ratio and bound discrepancy here and
nowhere else.

**Do not read the per-frame throughput as the pipeline's throughput.** A 9.2 MB frame is
launch-overhead-dominated: the same pipeline runs at 114 GB/s per frame and **601 GB/s**
on the volume. The per-frame figures in §2 are comparable *to each other* (all arms pay
the same overhead) but understate absolute throughput by ~5×.

---

## 7. Correctness

- 8 stage unit tests, all passing; full repo suite 48/48 (33 stage + 15 pipeline).
- `compute-sanitizer --tool memcheck` clean on the unit tests and on the full pipeline
  in both directions, including `eb_bg = 1e5` where the background quantizes to entirely
  zeros — the near-constant condition that previously triggered a Huffman illegal-memory
  access in this codebase.
- Archive self-containment verified by deleting the `.roi` peak file before decompress:
  reconstruction succeeds within the background bound.
- Two silent-failure hazards found and fixed during the build, both of which produced
  plausible wrong numbers rather than errors. The worse one cost **13.4× of ratio**
  (44.83 → 3.34) with correct round-trip, satisfied ROI bound and normal PSNR. Both are
  documented in `notes/design_and_method.md` §7.

---

## 7b. Ablations — and one number these results understate

Audit of design choices made by reasoning rather than measurement, prompted by a peer
session finding `TiledLorenzo` shape-sensitive on 2-D climate fields. Full detail in
`notes/design_and_method.md` §2b.

| choice | measured effect |
|---|---|
| `TiledLorenzo` on the background | **+9.3 %** (bin=1) / **+13.6 %** (bin=2) — earns its place, does not collapse |
| predictor on the ROI branch | +0.4 % — correctly omitted |
| `bin_factor` 2 → 4 | CR 44.8× → 122.8× for **0.66 dB** of background PSNR, but max error 7290 → 10290 |
| `AdaptiveBitpack` block size 64 → 32 | **+3.4 %** |

Two of these change how the headline should be read:

- **ROI PSNR is 69.07 dB at every bin factor (1, 2, 3, 4).** The ROI branch is fully
  insulated from whatever is done to the background. That is the design's central claim,
  and here it is observed rather than argued.
- **The shipped configs use `block_size = 64`, which is not optimal — 32 is ~3.4 %
  better.** The 64 was inherited from the cuSZp3 preset and never tuned. It has *not*
  been changed, because every number above was produced at 64 and re-tuning one arm's
  coder after measuring would make the comparison inconsistent. So the reported ROIBIN
  ratios **understate this pipeline by ~3 %**.

Two further choices were measured after the first pass, and one of them corrected a
claim in this report's own design note:

| choice | measured effect |
|---|---|
| `linear_mode` true → false | 46.12× → **33.49×** — the inherited default is worth **+37.7 %** |
| `MergeStage` → one coder, vs separate coders | 46.00× vs 46.12× — **a wash (0.26 %)** |

The design note had claimed separate coders were "strictly better" because merging would
force two very different distributions through one set of coder statistics. **That was
wrong for the coder actually used**: `AdaptiveBitpack` is block-local (a bit width per
64-element block), so concatenation costs only the blocks straddling the seam. The
argument was correct for a global entropy coder and irrelevant here. Separate coders
still ship — marginally better and one stage simpler — but on the honest grounds of "no
reason to add a stage", not on a ratio claim.

Still unmeasured, and stated as such: the choice of `AdaptiveBitpack` over
Huffman / ANS / RZE on either branch.

## 8. What these numbers do not establish

- **Throughput is not host-inclusive.** cuSZ-Hi's figures are kernel-only from its
  `-R time` `(total)` row; FZGM's come from `--report-json`. Cross-tool throughput here
  is indicative — the host-inclusive numbers are what experiments B1/B3 exist to produce.
- **The two corpora are not pooled** and should not be averaged together.
- **EXAFEL is one run from one sample.** Splitting the single registered 3-D field into
  130 per-frame fields is what lifts it above a handful of rows, but it does not make it
  more than one experiment. CXIDB 21 is included for exactly this reason, and it is
  itself a 2.57 GB prefix of a 32 GB deposition covering two runs — complete valid
  frames, but not a random sample.
- **The comparison is against single-bound compressors doing what they must.** It says
  nothing about how they would perform if they could express two bounds; the claim is
  about what the graph structure permits, not about codec quality.
- **`bin_factor = 2` is not error-bounded** on the background. See §3.
