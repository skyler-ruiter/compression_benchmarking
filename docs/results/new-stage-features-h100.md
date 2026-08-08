# New FZGM stage features on H100 — measured

**Date:** 2026-08-05/06 · **GPU:** NVIDIA H100 80GB HBM3 (driver 595.84), JetStream2
`skyler-h100` · **FZGM:** `1b0e074` (Release, sm_90) · **Sessions:**
`20260805-213128-newfeat-{fzgm_huffman_book,fzgm_fsz,fzgm_rle_bitpack}` in
`$BENCHKIT_RESULTS_ROOT` (380 cells, 379 ok).

Evaluates the features added in `f3e338f`, `579e4ad`, `1b0e074`. Log transform was
explicitly out of scope for this round; bugfixes were not performance-measured.

Reproduce any table below with:

```bash
python scripts/build_variant_ratio.py <session> --baseline <variant> \
       --metric ctp|dtp|cr [--stage Huffman] [--by-field]
```

Ratios are formed **per cell and then averaged geometrically**, never as a ratio of two
independently-computed means: the corpus spans 11 MB to 1.12 GB, so a mean of raw
throughputs would just report which fields were in the subset.

---

## Headline

1. **Adaptive Huffman codebooks are worth taking, but the throughput win is
   size-dependent and modest: 1.14x compress geomean, 1.2–1.45x on inputs ≤100 MB,
   ~1.0x at ≥500 MB.** Compression ratio is unchanged to four digits (1.000x) and PSNR is
   bit-identical. The stronger argument for `Adaptive` turns out not to be speed at all —
   see (2).
2. **`Adaptive` compresses data that `PerBlock` cannot.** PerBlock aborts on skewed and on
   degenerate inputs; Adaptive's frequency floor makes both encodable. One sweep cell and
   two entire real SDRBench fields are affected. **A pre-existing crash bug was found in
   the process — see [Bug](#bug).**
3. **`Fixed` should not be used.** It costs **-55% compression ratio** for +6% throughput,
   and is *slower* than `Adaptive`, which gives up nothing. There is no operating point
   where it wins.
4. **FSZ's `AdaptiveLorenzo` is the strongest result in this round: +28% compression ratio
   AND +26% compress throughput** against the same coder with a plain Lorenzo predictor.
   It also out-compresses native cuSZp2/cuSZp3 by 7%. It costs ~19% decompress throughput.
5. **Chunked RLE is a large throughput win (1.85x compress, 6.4x decompress) at a real
   ratio cost (-20…-26%)**, not the "negligible" cost the stage docs claim. `chunk_size =
   8192` dominates `4096`: same throughput, better ratio.
6. **Bitpack `auto_shift` costs 12.5% throughput and buys exactly nothing (1.000x ratio)**
   on predictor-residual data. Correct behaviour, wrong pipeline — don't enable it here.

---

## 1. Huffman codebook source

Session `…-fzgm_huffman_book`, baseline `fzgm-huf-perblock` (= `configs/pipelines/cusz.toml`).
19–20 cells per arm; 10 fields × 2 bounds (`rel_range` 1e-2, 1e-4).

| arm | compress | decompress | ratio | Huffman stage only (compress) |
|---|---|---|---|---|
| `Adaptive` | **1.140x** | 1.000x | **1.000x** | 1.172x |
| `Adaptive`, no range guard | **1.233x** | 0.999x | 1.000x | **1.298x** |
| `Adaptive`, refit every 4 | 1.120x | 0.999x | 1.000x | 1.160x |
| `Fixed` (Laplace, scale 16) | 1.064x | 0.986x | **0.447x** | 1.080x |
| native cuSZ | 1.261x | 1.011x | 0.860x | — |

**Ratio is exactly preserved.** `Adaptive` measures 1.000x CR over all 19 cells and PSNR
matched `PerBlock` to within 1e-6 in **all 76 comparisons** — as it must, since the
codebook changes only how symbols are spelled. This is the cleanest possible confirmation
that the mode is safe to adopt.

**The win is inversely proportional to input size.** This is the per-call barrier being
amortised, and it is the honest way to report the number — a single geomean hides it:

| field | size | baseline compress GB/s | `Adaptive` |
|---|---|---|---|
| EXAALT `xx` | 11.5 MB | 21.4 | **1.34x** |
| CESM-2D `LHFLX` | 25.9 MB | 49.9 | **1.45x** |
| HURR `CLOUD` | 100 MB | 129.3 | 1.26x |
| NYX `baryon_density` | 537 MB | 235.5 | 1.04x |
| CESMATM-3D `CLOUD` | 674 MB | 193.0 | 1.04x |
| HACC `vx` | 1.12 GB | 83.4 | 1.02x |

(`eb=1e-4` rows. At ≥500 MB the baseline is already running at 190–240 GB/s and the
per-call histogram is noise against the payload.)

**The symbol-range guard costs 8%** (1.233 / 1.140). It is on by default and is what
replaces the `sum(h_freq) == inlen` check once a book is pinned. Disabling it is only safe
when the upstream guarantees the range — with `LorenzoQuant`, `zigzag_codes = true` and
`bklen = 2 × quant_radius`, which is exactly `cusz.toml`'s shape. Off by hand it is
undefined behaviour and takes down the CUDA context.

**Refitting every 4 calls costs 1.7%** (1.140 → 1.120) — cheap insurance, and the
measurement is only the cost side: benchkit re-compresses identical data, so a refit here
can never recover ratio. The ratio side is §1.2.

**`Fixed` is a trap.** -55.3% ratio against PerBlock, worse than the -31% the stage docs
report for this preset shape, and it is *slower than `Adaptive`* (1.064x vs 1.140x). Both
modes skip the histogram after the first call, so `Fixed` buys no additional throughput
while giving up half the compression. `Adaptive` strictly dominates it.

**Against native cuSZ**, FZGM's Huffman path is ~11% behind on compress throughput
(1.261 / 1.140, both measured against the same PerBlock baseline) but compresses
**15.5% better** — measured directly against `cusz-native` as baseline, `Adaptive` scores
**1.155x** CR over 18 shared cells. Native cuSZ additionally produced no usable row on 2
cells that every FZGM arm completed.

### 1.2 Codebook drift across genuinely different data

benchkit cannot see drift — it reuses one pipeline on identical data, which is both the
correct steady-state throughput measurement and the best case for ratio. Measured
separately with `fzgmod-profile-huffman-drift` (build `build_profiling/`):

| sequence | `Adaptive` mean loss | worst | `Fixed` mean |
|---|---|---|---|
| CESM-ATM `CLOUD`, 23 non-constant levels | **0.86%** | 3.06% | 19.5% |
| 5 different CESM-ATM fields | **2.27%** | 7.96% | 49.2% |
| CESM-2D `CLDHGH`, 8 slabs | 4.92% | 15.1% | 4.11% |

Better than the 4.7% / 8.3% the stage docs record from the development box. The refit
machinery fired 2 times in each run, so the degradation is bounded rather than
monotonically growing. `Fixed`'s 19–49% independently confirms the ratio verdict above.

---

## <a name="bug"></a>2. Bug found: CUDA IMA on constant input (pre-existing, not from this round)

> **Fixed in §5.3** — and the investigation found a second, worse bug underneath it.
> The characterisation below is what was visible *before* the root cause was known;
> read §5.3 for what was actually wrong. In particular, the "constant + not a multiple
> of 1024" framing describes the trigger, not the defect.

`HuffmanStage` in the default `PerBlock` mode dies with an illegal memory access on
**constant (single-symbol) input whose element count is not a multiple of 1024**:

```
[fzgmod] CUDA error at modules/coders/huffman/phf/hf_buf.cc:170 —
cudaMemcpyAsync(...) → an illegal memory access was encountered
```

Both conditions are required, and the boundary is exact:

| n (all-zero f32) | multiple of 1024 | result |
|---|---|---|
| 5,242,879 | no | **IMA** |
| 5,242,880 | yes | ok (CR 482.95) |
| 5,242,881 | no | **IMA** |
| 5,243,904 | yes | ok (CR 482.86) |
| 6,480,000 | no | **IMA** |

Non-constant data at a non-multiple-of-1024 length is fine — CESM `CLDHGH` is also exactly
6,480,000 elements and compresses normally. The IMA is reported at the next sync point,
not at the faulting kernel, so the message names `hf_buf.cc:170` rather than the encode
kernel; the fault is a tail/remainder case in the encode path when the codebook degenerates
to a single symbol.

**This hits real data.** CESM-2D `SFCLDICE` and `SFCLDLIQ` are entirely zero at 3600×1800
= 6,480,000 elements, and **cannot be compressed at all by the default `cusz.toml`
pipeline**. `cusz_hufbook_adaptive.toml` compresses both at 30.03x. The same fault
aborts `fzgmod-profile-huffman-drift` on any sequence containing a constant step
(CESM-ATM `CLOUD` levels 0–2, Hurricane `CLOUDf48` slabs 18–19), which is why the drift
runs above use `--skip 3`.

Minimal repro (no dataset needed):

```bash
python3 -c "import numpy as np; np.zeros(6480000,dtype='<f4').tofile('/tmp/z.f32')"
fzgmod-cli -c configs/pipelines/cusz.toml -i /tmp/z.f32 -l 6480000 -b --report
```

Not caused by this round's changes — `PerBlock` is the pre-existing default path. It went
unnoticed because the benchkit validity gate classifies these fields as
`degenerate_field` and drops them from aggregates, so they never produced a visible number.

---

## 3. FSZ / `AdaptiveLorenzo`

Session `…-fzgm_fsz`, baseline `fzgm-cuszp2` — the controlled reference: **same
`AdaptiveBitpack` coder, plain Lorenzo predictor**. 20 cells per arm, no failures.

| arm | compress | decompress | ratio |
|---|---|---|---|
| `fsz` (bpt 8, order2+centering) | **1.257x** | 0.809x | **1.282x** |
| `fsz` bpt 16 | 1.174x | 0.753x | **1.325x** |
| `fsz` no order-2 | 1.259x | 0.820x | 1.262x |
| `fsz` no centering | 1.567x | 0.830x | 1.198x |
| FZGM cuszp3 | 1.285x | 0.650x | 0.766x |
| native cuSZp2 | 2.749x | 1.627x | 1.193x |
| native cuSZp3 | 2.778x | 1.619x | 1.193x |

**FSZ wins on both axes at once against its control** — +28.2% ratio *and* +25.7% compress
throughput. That combination is unusual and is the strongest single result in this round.

**It also out-compresses the native fused implementations**, confirmed by re-running the
comparison with `cuszp2-native` as the baseline directly: `fsz` bpt 8 scores **1.075x** and
bpt 16 **1.111x** CR over all 20 cells. It runs at ~46% of their compress throughput; that
remaining gap is the known modular-DAG materialization tax (three stages, three DRAM
round-trips vs one fused kernel), not a property of `AdaptiveLorenzo`.

**Ablation.** The stage does three things at once; they are not equally valuable:

| contribution | ratio effect |
|---|---|
| cross-block prediction state (chain spans the tile) | **≈ +19 pp** (residual) |
| centering | +7.0% (1.282 / 1.198) |
| second-order residual | +1.6% (1.282 / 1.262) |

The stage docs' claim that the cross-block prediction state is "most of the compression
win" is confirmed: turning off *both* adaptive residual variants still leaves the large
majority of the gain. Second-order prediction earns almost nothing here and costs nothing
either. **Centering is the expensive option**: it buys 7% ratio but is responsible for a
25% throughput drop (`nocenter` runs at 1.567x vs `fsz` at 1.257x).

**`blocks_per_tile = 16` is the better ratio setting** (+3.4% over bpt 8) at a 7%
throughput cost, confirming the docs' expectation on H100.

**Decompress is the weak side** — every FSZ arm is ~19% *slower* to decompress than plain
Lorenzo, and cuszp3 is 35% slower. If decompress throughput is the binding constraint,
this predictor is not the right trade.

---

## 4. Chunked RLE and bitpack shift

Session `…-fzgm_rle_bitpack`. Bounds are 1e-2/1e-3 here (not 1e-4): RLE needs runs, and at
1e-4 even CESM `CLDHGH` expands, which would gate the entire ratio side away.

### Chunked RLE — baseline `fzgm-rle-whole` (`chunk_size = 0`)

| arm | compress | decompress | ratio |
|---|---|---|---|
| `chunk_size = 4096` | 1.851x | **6.363x** | 0.744x |
| `chunk_size = 8192` | 1.852x | 6.222x | **0.802x** |

**The throughput win is large and the decompress win is dramatic** (6.4x) — chunked decode
replaces the whole-array path's device-to-host readback with header-carried metadata, which
is also what makes chunked decode CUDA Graph-capturable.

**But the ratio cost is real: -20% to -26%, not the "negligible" the stage docs claim.**
The docs measured 1.50x CR in both modes on synthetic 88%-zero data. On real predictor
residuals the forced run boundary at every chunk head plus the
`4 × (num_chunks + 1)` offset table costs a fifth of the compression. It bites hardest
exactly where RLE works best: NYX `baryon_density` compresses 792x whole-array, and long
runs are precisely what chunk boundaries chop up.

**Use 8192, not 4096.** Identical throughput (1.852 vs 1.851), 7.8% better ratio. The docs
call 4096–8192 a "sweet spot" and treat them as equivalent; on H100 they are not.

**Gate audit for this session:** 100 rows → 89 usable. 9 excluded as `expansion` (the
1-D-particle RLE cells described below) and 2 as `eb_violated_severe`. Both codes are
throughput-neutral; the ratio table above is computed over the surviving 17 shared cells,
and the throughput tables deliberately retain `expansion` rows (see
`scripts/build_variant_ratio.py` — a coder that expanded still has a valid speed).

**RLE's ratio is violently data-dependent** and it is not a general-purpose coder here — at
eb 1e-2: NYX 792x, HURR `CLOUD` 12.2x, CESM `LHFLX` 8.7x, but HACC `vx` 1.24x and EXAALT
`xx` **0.72x (expansion)**. The 1-D particle fields have no runs to find. Those cells are
retained in the session for their throughput rows and hard-excluded from CR aggregates by
the validity gate as `expansion`.

### Bitpack `auto_shift` — baseline `fzgm-bitpack-base`

| arm | compress | ratio |
|---|---|---|
| `auto_shift = true` | **0.875x** | **1.000x** |

**Costs 12.5% throughput, buys nothing.** `auto_shift` does an OR-reduce to find the
largest provably-lossless right shift; after Zigzag, a residual stream's low bits are
exactly the bits that vary, so the answer is always shift 0. The knob is working correctly
— it is declining to fire — and the 1.000x confirms it never silently discarded data. It
is simply the wrong pipeline for it. `auto_shift` also disables CUDA Graph compatibility
while active.

---

## Method notes and gotchas

- **`fzgmod-cli -e` / `-m` do NOT override a TOML `error_bound`.** A hand-run bound sweep
  against a `-c config.toml` silently reports the TOML's baked-in bound every time. benchkit
  is unaffected because `benchkit/pipelines.py` rewrites the TOML text. This cost real time
  and produces plausible-looking wrong tables.
- **Never run sweeps concurrently on this VM.** One GPU; parallel runs corrupt every timing.
  `ctest -j4` also produced two spurious failures (`test_pipeline_errors`,
  `test_memory_strategies`) that pass serially — **48/48 pass with `ctest` unparallelised**.
- **`test_ans` `PartialBlock` now passes.** It was a known pre-existing SEGFAULT as of
  2026-07-23; the bugfix commits in this round resolved it.
- **Per-stage `device_ms` understates `Adaptive`'s benefit.** The barrier it removes is a
  *host* histogram D2H + serial tree build. Device timers see it only as a gap. The
  Huffman-stage ratio (1.172x) and the end-to-end ratio (1.140x) bracket it; the CLI's
  DAG-level number on a single 26 MB field showed 49.7 → 70.0 GB/s (1.41x), consistent
  with the size-dependence table rather than contradicting it.
- **`build_benchmarking/` had `BUILD_PROFILING=OFF` and stays that way.** The drift harness
  lives in `build_profiling/`, per `scripts/env-jetstream2.sh`'s split, so NVTX
  instrumentation never contaminates the timing binary.
- Drift-harness GB/s figures are one unwarmed wall-clock call per cell and are labelled
  indicative by the harness itself; they are not quoted as results above.

---

## 5. Optimizations made in response to these results

Two changes landed after the measurements above; both are in the FZGM working tree
(`hf_bk.cc`, `adaptive_lorenzo_stage.cu`) and neither is a format change.

### 5.1 Huffman `sublen` is now size-adaptive — closes the large-input gap

§1 found `Adaptive`'s throughput win decaying to ~1.0x above 500 MB. The cause was not
the codebook path at all. `capi_phf_coarse_tune_sublen(size_t inlen)` **ignored `inlen`
and returned the constant 768**, fixing `pardeg = ceil(n/768)` — which *is* the coarse
path's entire parallel decomposition, and drives cost in three places simultaneously:
pardeg-sized grids in encode phase2/phase4 and in decode; an O(pardeg) **serial host**
prefix-sum plus two accumulates behind two stream syncs; and two pardeg-sized arrays
written into the encoded stream.

A first hypothesis — that the host barrier dominates — was **refuted**: a 32x `pardeg`
sweep moved throughput by nothing. The real finding came from sweeping `sublen` across
input sizes, which showed the optimum moving with `n` and 768 being far off at the top end.

Now targets `pardeg ≈ 131072`, floored at the historical 768. The floor is what makes
every change a strict improvement rather than a trade. Validated as two full benchkit
sessions (`sublen-old`, `sublen-new`, 67 matched cells, `configs/experiments/
fzgm_huffman_sublen.yaml`) over a field set deliberately **wider than the six fields the
rule was tuned on**:

| dataset | n | compress | decompress | ratio |
|---|---|---|---|---|
| EXAALT | 2.9M | 1.002x | 1.001x | 1.000x |
| CESM-2D | 6.5M | 0.998x | 0.999x | 1.000x |
| HURR | 25M | 0.996x | 0.998x | 1.000x |
| NYX | 134M | **1.144x** | 1.101x | 1.018x |
| CESMATM-3D | 168M | **1.310x** | 1.100x | 1.014x |
| HACC | 281M | **1.962x** | **1.470x** | 1.026x |
| **overall** | | **1.152x** | **1.077x** | 1.008x |

**Held-out fields only** (EXAALT `yy`, CESM `TMQ`/`PRECT`, HURR `P`, NYX `velocity_x`,
CESMATM `RELHUM`, HACC `xx`; 28 cells): **1.171x / 1.085x / 1.009x** — slightly better
than the overall mean, so the rule is not overfitted to its tuning set.

Ratio **never regressed on any cell** (min 1.000x) and PSNR is unchanged in all 67.
Cross-geometry decode was verified explicitly: a stream written with the old geometry and
one written with the new decode to *identical* reconstructions under either build, because
`sublen` travels in `phf_header`.

Below ~26 MB a smaller sublen is a genuine throughput/ratio **trade** (sublen 256: +27%
compress, +161% decompress, −3.9% ratio), so it is left opt-in via `FZ_HF_SUBLEN` rather
than made the default.

**Confound worth knowing:** `Buf` computes `sublen = use_HFR ? 1024 : tune(inlen)`, so a
preset declaring `encode_mode = "Fine"` pins sublen to 1024 and **opts out of the tuning
entirely** — while still never running the fine kernel on realistic data. The shipped
`cusz.toml` declares `"Fine"`, so it takes fine's partition geometry and none of fine's
benefit. Presets that want the tuning must say `"Coarse"` (hence the `_coarse` variants
added here).

### 5.2 `AdaptiveLorenzo` inverse rewritten as a warp scan — fixes FSZ's weak axis

§3 found FSZ losing ~19% decompress, its only bad axis. `adaptive_lorenzo_inverse_kernel`
ran a shared-memory Hillis-Steele scan with **two `__syncthreads()` per stride** —
`2·log2(tile_size)` barriers per pass, doubled for order-2 tiles, so 32 barriers at the
default 256-element tile. Barrier-bound, not memory-bound: the same failure mode already
fixed twice in this codebase (`TiledLorenzoStage` 3.2x, 1-D `LorenzoStage` 5.1x).

Replaced with a two-level warp scan (`fz::backend::shflUp`): the intra-warp scan is
warp-synchronous and needs no barriers, so a pass costs 2 barriers regardless of tile
size, and shared memory drops from `tile_size` elements to at most 32 warp totals.

| | old | new | |
|---|---|---|---|
| AdaptiveLorenzo inverse, NYX 512³ | 0.8137 ms | 0.6150 ms | **1.32x** |
| FSZ end-to-end decompress | — | — | 1.07–1.20x |

The other two stages are unmoved (AdaptiveBitpack 0.3896→0.3882, Quantizer 0.4471→0.4460),
confirming the change is localized. The gain is largest at `blocks_per_tile = 16`
(1.15–1.20x), where the old barrier count was highest — the signature the diagnosis
predicts.

**Output is bit-identical**, verified by a git-stash A/B decoding the same streams under
both builds across all four FSZ preset variants plus a 3-D field. `compute-sanitizer`
memcheck, racecheck and synccheck are all clean — racecheck and synccheck specifically
because the barrier structure changed. Full suite 48/48.

Bit-exactness is not incidental: the old kernel accumulated in `T` and wrapped at each
step, the new one accumulates in a widened type and truncates on store. Those agree
because two's-complement addition is modular and truncation is a ring homomorphism
`Z/2^32 → Z/2^16`.

---

### 5.3 The constant-input crash (§2) was two bugs, and the second is worse

Both are fixed in the FZGM working tree. §2's IMA turned out to be the shallower of the
two, and chasing it surfaced a silent data-corruption bug underneath.

**Bug A — out-of-bounds shared read (the crash).**
`KERNEL_CUHIP_Huffman_ReVISIT_lite` stages its chunk under `if (id < len)`, so when `len`
is not a multiple of `ChunkSize` (1024) the tail slots of `s_to_encode` are never written.
The reduce-merge loop read them anyway and used the value as a **codebook index**
(`s_book[p_key]`); its only guard, `(idx < allowed_len())`, suppressed the bit *count*, not
the *read*. That is why the trigger looked like "constant data whose length is not a
multiple of 1024": the fine path only engages when every code is ≤ 8 bits, which in
practice means a degenerate alphabet. Fixed by stopping the loop at the chunk's valid
length. Sanitizers clean.

**Bug B — a single-symbol alphabet got a ZERO-BIT code (silent corruption).**
`phf_stack::inorder_traverse` assigns codes by walking the tree, but with one distinct
symbol the root *is* a leaf, so it emitted with `len == 0`. The encoder then contributed 0
bits per symbol, the partition encoded to an empty bitstream, and the decoder returned
zeros. **Wrong data, not a failure, and in both encode modes** — Coarse as well as Fine.
Measured before the fix on a constant `uint16_t` stream of symbol 7: **1,279 of 3,000
symbols wrong** in each mode. Fixed by giving a one-symbol alphabet the conventional 1-bit
code.

**Why B hid for so long, and a caution about how §2 was reported.** The corrupt value is
`0`. An all-zero field decodes to 0 either way, and *any* constant field behind a Lorenzo
predictor has residual 0, so its single symbol is 0 as well — corruption is a no-op in
exactly the cases that produce it. Only a stage-level test with a non-zero symbol exposes
it. Concretely: after fixing Bug A alone, `SFCLDICE` compressed at 488x with PSNR `inf`
and max-abs-error 0, which looks like proof of correctness and is not. That result is
consistent with Bug B still being present. Pipeline-level round-trip testing cannot
distinguish these; the regression test (HF35) is at stage level for that reason.

**Bug C — small inputs overran the encoded-blob buffer.** Found alongside, now also fixed.
`d_encoded` aliases `d_scratch4`, sized at one `H4` word per input element — but the blob
it must hold is header + reverse codebook + two `pardeg` arrays + bitstream, and that is
*not* bounded by input length: the reverse codebook is fixed by `bklen` (2,304 bytes at
1024) and does not shrink. Below `inlen ≈ 1,224` it dominated and `memcpy_merge` wrote past
the allocation (`cudaMemcpyAsync ... invalid argument`). The predicted boundary
`128 + revbk4_bytes + 2·pardeg·4 + (inlen/2)·4 > inlen·4` matched measurement exactly —
1,223 worked, 1,200 failed. Now sized as the max of both requirements; a no-op at any size
where the old sizing sufficed, so no extra memory at scale. Every size from n=64 up now
compresses.

Two things worth noting from chasing it. First, the fix initially did *not* work because I
used `sizeof(phf_header)` (~64 B) where the format reserves a fixed
`PHFHEADER_FORCED_ALIGN` of 128 B — under-reserving by exactly enough to keep the overrun
alive. Instrumenting the actual offsets found that in one run; reasoning about it had
already produced one wrong answer. Second, `Buf::clear_buffer()` was passing element counts
to `cudaMemset`, which takes bytes, so every buffer wider than a byte was only a quarter
cleared. Fixed while there.

---

## Follow-ups

1. Consider making `Adaptive` the default `book_source` — it is strictly better than
   `PerBlock` on ratio (equal), robustness (strictly better) and throughput (1.14x), with
   the drift caveat bounded at ~1–3% and self-correcting via refit.
2. Re-examine the chunked-RLE ratio claim in `docs/stages/rle.md` — the "negligible"
   characterization does not survive contact with real predictor residuals.
3. Switch `cusz.toml` (and any other preset declaring `encode_mode = "Fine"`) to
   `"Coarse"`, so it stops paying fine's partition geometry for none of fine's benefit
   and picks up §5.1. Alternatively, route the `use_HFR` path through the tuner too —
   but only for the geometry the fine kernel can actually accept.
4. §5.1's floor leaves the sub-26 MB throughput/ratio trade unexploited. If small-input
   latency matters for a workload, `FZ_HF_SUBLEN=256` is measured and available.
