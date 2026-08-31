# ROIBIN in FZGM — design, method, and the constraints that shaped both

Companion to `data_provenance.md` (corpus) and `RESULTS.md` (numbers).

---

## 1. The pipeline

One new stage, `ROIBinSplitStage`, does the whole job of turning a detector frame
into two independently-bounded branches. Everything downstream is existing FZGM
stages.

```
                    ┌── roi ───> Quantizer(eb_roi)  ─────────────────> AdaptiveBitpack ──┐
input ──> ROIBinSplit ── bg ────> Quantizer(eb_bg)  ──> TiledLorenzo ──> AdaptiveBitpack ─┼─> .fzm
                    └── peaks ────────────────────────────────────────────────────────────┘
```

`ROIBinSplit` is `1 → 3` forward and `3 → 1` inverse. That asymmetry is what makes
the two error bounds possible: because the ROI pixels and the background leave the
stage on *different ports*, they become different DAG branches, and a branch is
where a `Quantizer` with its own `error_bound` lives. A linear compressor has one
place to put the bound; this has two.

**Why the two branches differ in more than their bound.** The `bg` branch keeps
`TiledLorenzo` because the background is still an image and spatially decorrelating
it pays (+9.3 % / +13.6 %, measured below). The `roi` branch does not: `roi` is a
peak-major concatenation of 9×9 boxes rather than an image, so a *2-D* predictor has
no meaningful stride over it. A **1-D** predictor is expressible, and was measured
rather than dismissed: inserting a 1-D `TiledLorenzo` on the ROI branch moves CR from
44.83× to 45.00× — **+0.4 %, noise-level**, not worth a stage. `AdaptiveBitpack` is
block-local and dimension-agnostic, so it applies to both branches.

(An earlier draft of this note asserted the ROI branch had "no valid stride". That
was stronger than the evidence — a 1-D stride is perfectly valid, it just does not
pay, because quantized peak-box residuals at eb=10 have little run-to-run structure.)

**Ablation: is `TiledLorenzo` on the background actually earning its place?** It was
chosen by reasoning ("the background is still an image"), which is not evidence, and a
peer session found cuSZp3's TiledLorenzo to be *shape-sensitive* — strong on HURR,
collapsing on CESM-2D — which is a direct caution for a branch whose data is a binned,
non-native shape. Checked on 3 EXAFEL frames at eb 10/100, removing the stage and
wiring `AdaptiveBitpack` straight to the quantizer codes:

| arm | with TiledLorenzo | without | contribution |
|---|---:|---:|---:|
| bin=1 | 11.14× | 10.19× | **+9.3 %** |
| bin=2 | 46.13× | 40.62× | **+13.6 %** |

It helps on this data and does not collapse, so the headline numbers are not resting on
a mis-chosen predictor. Note it is a *modest* contributor — most of the ratio comes from
the dual bound and the binning, not from the predictor — so a future corpus where
TiledLorenzo does collapse would cost ~10 %, not the result.

**No `MergeStage` — but not for the reason first given.** The task sketch had the
branches converging through a merge. They do converge, but at the archive: FZGM
assembles the `.fzm` from the DAG's leaf outputs, so a `MergeStage` adds a
device-to-device copy for nothing.

The original note went further and claimed that merging before a coder would force the
ROI and background codes through *one* set of coder statistics — their distributions
being very different — so separate coders were "strictly better for ratio". **That was
measured and it is wrong.** Merging both streams into a single `AdaptiveBitpack`:

| | CR (3 EXAFEL frames, eb 10/100) |
|---|---:|
| separate coders (shipped) | 46.12× |
| Merge → one coder | 46.00× |

A **0.26 % difference — a wash, not "strictly better"**.

The replacement explanation reached for next was that `AdaptiveBitpack` is block-local
(a bit width per 64-element block), so concatenation only costs the blocks straddling
the seam. True, but still not the general rule — a parallel study measured the *opposite*
result on a pipeline terminating in **ANS**, a global entropy coder:

| HURR/CLOUD, X6 pipeline | CR @1e-3 | CR @1e-2 |
|---|---:|---:|
| Merge (shipped) | 44.46 | 109.49 |
| separate coder per port | 28.00 | 44.74 |
| | **1.59×** | **2.45×** |

Merging was strongly load-bearing there — and the mechanism was **neither** distribution
mixing nor block locality. It was **per-stream framing overhead**: ANS emits a header per
invocation, and that pipeline's small ports (`anchor`, `outlier_idxs`) each pay a full
header when coded alone, plus an RRE pass over the concatenation finds cross-port
redundancy that independent coders cannot see. Distribution mixing is real and gets
swamped.

**The general rule, from both results together:** whether to merge is a property of the
**coder's per-invocation framing cost and the port-size distribution**, not of whether
the streams "belong together". This pipeline has two large ports and a coder with
negligible framing, so merging is a wash; a pipeline with small ports and a globally-
framed coder can gain 1.6–2.5×.

Separate coders are still what ships — marginally better, and one stage simpler — but
the honest justification is "no reason to add a stage", not a ratio claim. Merge remains
the right tool when a codec is *defined* over a concatenation (cuSZ-Hi's LC blob), and
now also when the ports are small relative to the coder's framing cost.

## 2. Three design decisions worth defending

### Redundancy instead of stream compaction

Each peak owns a fixed 9×9 box and `roi` is those boxes concatenated. Overlapping
boxes store shared pixels twice. Measured redundancy at `half_width = 4`:

| corpus | ROI slots | duplicated | unique ROI coverage (mean / max) |
|---|---:|---:|---:|
| EXAFEL | 1,120,797 | 5,448 (**0.49 %**) | 0.374 % / 2.130 % |
| CXIDB 21 | 458,055 | 12,830 (**2.80 %**) | 0.069 % / 0.203 % |

The alternative — a per-pixel mask plus a device-wide exclusive scan — needs a
4-byte offset per pixel, i.e. **1.2 GB** on the 130-frame volume, to recover 0.5–2.8 %
of a stream that is itself under 1 % of the frame. Storing duplicates also makes
the output size *exactly* `npeaks * box * sizeof(T)`, known before any kernel runs,
so `estimateOutputSizes()` is exact and PREALLOCATE needs no slack.

Correctness rests on scatter being **idempotent**: every copy of a source pixel
takes the same value, goes through the same quantizer, and reconstructs to the same
number, so write order cannot matter. Edge boxes are clamped rather than truncated
for the same reason. Both are pinned by tests (`OverlappingBoxesAreIdempotent`,
`EdgeBoxesAreClamped`).

### The box size is measured, not assumed

`roi_half_width = 4` was chosen by measuring the mean intensity on square rings
around the published peak positions (`scripts/peak_profile.py`). EXAFEL, frame
median 211.3 ADU:

| radius | box | mean ring | excess |
|---:|---|---:|---:|
| 0 | 1×1 | 3731.2 | 3519.9 |
| 1 | 3×3 | 1242.0 | 1030.7 |
| 2 | 5×5 | 507.4 | 296.0 |
| 3 | 7×7 | 406.8 | 195.5 |
| **4** | **9×9** | **394.2** | **182.8** |
| 5 | 11×11 | 393.6 | 182.2 |
| ≥6 | — | ~389 | ~178–188 |

The peak decays into a plateau by radius 3–4 and the profile is flat beyond it, so
the residual ~180 ADU excess is locally elevated background around peaks (they
cluster on bright rings), not peak signal. CXIDB 21 plateaus by radius 2–3. A 9×9
box therefore contains the peak with a ring of margin on both corpora. Had this not
been checked, a too-small box would have silently demoted real peak signal to the
loose-bound branch — the one failure the study must not have.

### The peak list is carried in the archive

The peak table is re-emitted on the `peaks` port, so it is stored inside the `.fzm`
and **counted in every compressed size reported here**. At 8 B/peak this is ≈0.01 %
of a frame. Verified directly: deleting the `.roi` file and decompressing still
reconstructs the frame to within the background bound.

This matters for fairness. Feeding the decompressor an out-of-band ROI mask and not
counting its bytes would be a way of making the ratio look better than it is.

## 2b. Ablations — which design choices were measured, and which were not

Prompted by a peer session's finding that `TiledLorenzo` is shape-sensitive on 2-D
climate fields. Auditing this pipeline for choices made by *reasoning* rather than
measurement turned up several. All figures are aggregate CR over 3 EXAFEL frames at
eb 10/100 unless noted.

**`TiledLorenzo` on the background** — earns its place, modestly:

| arm | with | without | contribution |
|---|---:|---:|---:|
| bin=1 | 11.14× | 10.19× | **+9.3 %** |
| bin=2 | 46.13× | 40.62× | **+13.6 %** |

It helps here and does not collapse, so the headline does not rest on it. It is a
*modest* contributor — most of the ratio comes from the dual bound and the binning —
so a corpus where it does collapse would cost ~10 %, not the result.

**A predictor on the ROI branch** — 44.83× → 45.00×, **+0.4 %**. Not worth a stage.

**`bin_factor`** (single frame, so CR differs slightly from the 3-frame figures):

| bin | bg shape | CR | ROI PSNR | bg PSNR | bg max err (nominal bound 100) |
|---:|---|---:|---:|---:|---:|
| 1 | 1552×1480 | 11.34× | 69.07 | 48.92 | **100** |
| 2 | 776×740 | 44.83× | 69.07 | 44.76 | 7290 |
| 3 | 518×494 | 85.37× | 69.07 | 44.27 | 9690 |
| 4 | 388×370 | 122.80× | 69.07 | 44.10 | 10290 |

Two things fall out of this that were not obvious in advance:

1. **ROI PSNR is identical (69.07 dB) at every bin factor.** The ROI branch is
   completely insulated from what happens to the background — which is the design's
   central claim, here observed rather than asserted.
2. **Beyond bin=2, ratio is nearly free in RMS terms.** bin=2 → bin=4 is 2.7× the
   ratio for 0.66 dB of background PSNR, because at `eb_bg = 100` the quantization
   error already dominates the binning error. But the background **max** error grows
   7290 → 10290, so the cost lands on sharp features, not on the bulk. Anything doing
   background subtraction cares about that tail, which is why this is a real
   trade-off and not free ratio. `bin_factor = 2` is what the study reports, matching
   ROIBIN-SZ; it is a conservative choice, and 3–4 are available if a downstream
   analysis can tolerate the tail.

**`AdaptiveBitpack` block size** (bin=2): 32 → **47.71×**, 64 → 46.12×, 256 → 42.98×.

**The shipped configs use 64, which is not optimal — 32 is ~3.4 % better.** The 64 was
inherited from the cuSZp3 preset and never tuned. It is *not* changed here, because
every number in `RESULTS.md` was produced at 64 and silently re-tuning one arm's coder
after measuring would make the comparison inconsistent. Recorded as known unexploited
headroom: the reported ratios understate this pipeline by ~3 % for that reason.

**`linear_mode` on the quantizers** — inherited from the cuSZp3 preset, and it turns
out to matter: `true` → 46.12×, `false` → **33.49×**, i.e. **+37.7 %** for the shipped
setting. A correct inherited default, not a no-op.

**`zigzag_codes`** is mutually exclusive with `linear_mode` and the framework rejects
the combination cleanly (*"QuantizerStage: linear mode is incompatible with zigzag
codes"*), so there is no choice to make here — worth recording because on a
non-linear-mode pipeline zigzag is worth ~4× and its absence would be a silent loss.

**`MergeStage` vs separate coders** — measured, see §1. A wash (0.26 %), and the
original justification was wrong.

**Still unmeasured, and stated as such:** the choice of `AdaptiveBitpack` over
Huffman / ANS / RZE on either branch.

## 3. Binning bounds nothing — and both arms are reported

`bin_factor = b > 1` replaces each `b×b` background block with its mean. **That is a
resolution reduction, not an error bound.** With `b > 1` the background error is
binning error plus quantization error, and it is *not* bounded by `eb_bg`.

Two arms are therefore reported separately and mean different things:

| arm | `bin_factor` | background claim |
|---|---|---|
| `roibin_b1` | 1 | genuinely satisfies `eb_bg` pixel-wise; the per-region gate is meaningful on both regions |
| `roibin_b2` | 2 | **no bound claimed**; background reported as PSNR only, gate records `n/a` |

The ROI branch satisfies `eb_roi` in both arms — that is the invariant the science
depends on, and it is what the verification table checks.

`b2` is the configuration ROIBIN-SZ actually describes and gets much the better
ratio. Presenting its ratio next to a bound-satisfying baseline's without this
distinction would be comparing a lossy-resolution scheme against an error-bounded
one and calling it a ratio win. The stage emits a `getRunNotes()` warning whenever
`bin_factor > 1` so the caveat travels with the run.

## 4. Constraints discovered while measuring

### An f32 precision floor sets the tightest usable bound

At `eb = 1e-2` ABS the ROI bound was violated (`max_abs_err` 0.01074). This is
**not** a defect of the split stage: a plain `cuszp3` pipeline with no ROIBIN stage
at all violates identically on the same frame (`Max Abs Error 1.1e-02`), while
`1e-1` and `1e0` are satisfied exactly.

The cause is representational. Linear-mode quantization reconstructs
`x' = q · 2·eb`; with `|x|` up to ~1.6e4 and `eb = 1e-2` that is ~1.6 M levels, and
the f32 rounding of the product is ~8.7e-4 — the same order as the 7.4e-4 by which
the bound is exceeded. The usable floor for this data is roughly
`eb > |x|_max · 2⁻²⁴ ≈ 1e-3`, with `eb ≥ 1e-1` comfortably safe.

Consequence for the study: **all arms use ABS bounds ≥ 1.0**, and the reported
bound pairs sit well inside the safe region. This is a property of f32 quantization
at this data scale, not of any arm, and it applies equally to the baselines.

### The baseline cannot run at every bound the ROI might want

Native cuSZ-Hi (`spline`, `cr-first`, `-s cr`) on all 130 EXAFEL frames:

| ABS bound | frames compressed |
|---:|---|
| 1.0 | **46/130 (35.4 %)** |
| 2.0 | 130/130 |
| 5.0 | 130/130 |
| 10.0 | 130/130 |

Failures are hard aborts (`psz_gpu_exception`, `GPU API failed at
compressor.inl:312, invalid argument`), not degraded output. The headline bound pair
was chosen at `eb_roi = 10.0` **specifically so every baseline runs on every frame**
— a comparison against a baseline that crashed on two thirds of the corpus would be
worthless. The `eb_roi = 1.0` arm is run separately, with cuSZ-Hi's full tuning grid
unpinned, to answer whether *any* tuning survives there.

This should be reported as an observed limit of one build on one machine, not as a
general claim about cuSZ-Hi.

## 5. How the baselines were tuned

The task requires tuned baselines. cuSZ-Hi exposes three knobs — `--predictor`
(spline/lorenzo), `-a` (cr-first/rd-first), `-s` (cr/tp) — giving 8 combinations,
all swept by `scripts/tune_cuszhi.py` over 12 sampled frames per corpus:

| setting | EXAFEL agg. CR | CXIDB 21 agg. CR |
|---|---:|---:|
| **spline/cr-first/cr** | **6.682** | **8.469** |
| spline/rd-first/cr | 6.504 | 8.264 |
| spline/cr-first/tp | 6.165 | 8.120 |
| spline/rd-first/tp | 6.043 | 7.948 |
| lorenzo/cr-first/cr | 5.937 | 7.326 |
| lorenzo/rd-first/cr | 5.937 | 7.326 |
| lorenzo/cr-first/tp | 5.599 | 7.172 |
| lorenzo/rd-first/tp | 5.599 | 7.172 |

All 8 ran on all 12 frames at `eb = 10`. **The winner is cuSZ-Hi's own default** —
the tuning pass confirms the default is the right setting for this data rather than
improving on it, which is the honest outcome to report.

One configuration is chosen per (corpus, bound) and pinned for the whole sweep,
rather than picking the best per frame. A per-frame oracle would flatter the
baseline beyond anything a real deployment does, and would also make the baseline's
cost model incomparable to the single fixed ROIBIN config.

PFPL's only extra argument is a sentinel-preservation `threshold`, which is not a
ratio knob and is left unset.

## 6. Measurement rules honoured

- **PREALLOCATE** in every FZGM config.
- **`fzgmod-cli -b`** for all FZGM timing — never a cold single run.
- **Repeated invocations**, one process per frame, not in-process `runs=N` only.
- **One bound mode (ABS)** across every arm in every comparison.
- **Warm-up discarded** in every throughput figure: cuSZ-Hi's first `-R time`
  rep is 2.78 ms against 0.34 ms warm, and PFPL's first of 9 internal reps is
  likewise cold; both are dropped before taking the median.
- **Aggregate CR is total bytes / total bytes**, not the mean of per-frame ratios,
  which over-weights frames that happen to compress well. Per-frame spread is
  reported separately as median and p10/p90.
- **Per-region gate, not a global one.** A global `max_abs_err` on a dual-bound
  pipeline just reports the looser bound and says nothing about the peaks. Every
  arm — baselines included — goes through the same per-region check so the numbers
  are comparable.

## 7. Correctness evidence

- 8 stage unit tests (`tests/stages/test_roibin_split.cpp`), all passing: exact
  round-trip at `bin=1`, background-is-the-mean at `bin=2`, ROI exact *despite*
  binning, overlapping-box idempotence, edge clamping, port counts, header
  round-trip (including that a later pipeline dims push cannot clobber
  archive-restored geometry), and exact output sizes.
- `compute-sanitizer --tool memcheck` clean on the unit tests **and** on the full
  pipeline in both directions, including `eb_bg = 1e5` where the background
  quantizes to entirely zeros — the near-constant condition that previously
  triggered a Huffman illegal-memory-access in this codebase.
- Archive self-containment verified by deleting the `.roi` file before decompress.

### Two silent-failure hazards found and fixed during the build

Both would have produced *plausible wrong numbers* rather than errors:

1. **`Pipeline::finalize()` re-pushes global dims to every stage**, overwriting
   anything set at construction. A dimension-aware predictor on the binned
   background would have used the full-resolution row stride, produced garbage
   residuals, and *still round-tripped* — because the inverse repeats the same
   mistake. Fixed with `TiledLorenzoStage::setDimsOverride()` (pinned dims,
   serialized in the stage header) and by having `ROIBinSplitStage` ignore pipeline
   dim pushes once its geometry came from the archive.

   **Measured cost of getting this wrong — 13.4× of ratio, with no other symptom:**

   | `roibin_b2`, eb 10/100 | with `dim_x/dim_y` | without |
   |---|---:|---:|
   | frame f005 CR | 44.83 | **3.34** |
   | frame f012 CR | 43.54 | **3.36** |

   The un-pinned run decompresses cleanly, satisfies the ROI bound
   (`roi_max_abs_err` 9.9999 ≤ 10), and reports ROI PSNR 69.07 dB / background
   44.76 dB — all indistinguishable from correct. Only the ratio moves. Had this
   not been caught, the binned arm would have been reported at roughly the
   baseline's ratio and the study would have concluded the ROIBIN approach barely
   helps.
2. **A 22-byte header declared as 24 bytes** made `deserializeHeader()` return early
   every time, so the decompress side never learned its geometry — and the stage
   factory read the element type from the wrong offset. This one did fail loudly
   (`dimensions not set`), but only because of an explicit guard.

## 8. Caveats

- The two corpora are **not pooled**: one is calibrated f32, the other raw ADU from
  a different experiment.
- EXAFEL's 130 frames are one run from one sample. Splitting the single registered
  3-D field into per-frame fields is what makes it more than a handful of rows, but
  it does not make it more than one experiment — which is why CXIDB 21 is included.
- The CXIDB 21 frames are a **2.57 GB prefix** of a 32 GB archive (the server
  ignores HTTP Range), covering runs 0015–0016 only. They are complete, valid
  per-event files, but they are not a random sample of the deposition.
- cuSZ-Hi throughput is parsed from its `-R time` `(total)` row, which is
  kernel-only time, and is not host-inclusive. FZGM figures come from
  `--report-json`. Cross-tool throughput comparisons here are indicative, not the
  host-inclusive numbers experiment B1/B3 exists to produce.
- A pre-existing `MemoryPool::free - pointer not found in allocations` warning
  appears on decompress for **all** stock presets (`cuszp3`, `pfpl`, `cusz_hi_cr`)
  as well as the ROIBIN configs, so it is not introduced by this work.
