# Native ROIBIN-SZ baseline (LibPressio) — setup and smoke test

*2026-09-28. Baseline for the FZGM RQ4 light-source case study.*

## Build
- Spack env `roibin-sz` (`~/spack/var/spack/environments/roibin-sz/spack.yaml`):
  `libpressio+core+sz3+openmp+python~cuda~fzgpumodules build_type=Release`, `sz3@3.3.0`
  (commit 5d2fd26), dev-built from `~/src/libpressio-fork`. `+core` sets
  `LIBPRESSIO_BUILD_MODE=FULL`, which compiles the `roibin`/`binning` plugins.
- The fork's `roibin.cc`, `roibin_impl.h`, `binning.cc`, `masked_binning.cc`, `sz3.cc`
  are identical to upstream robertu94/libpressio master `868a3a7` (the fork only adds
  the fzgpumodules plugin), so this is an unmodified native baseline.
- Run with: `V=~/spack/var/spack/environments/roibin-sz/.spack-env/view;
  PYTHONPATH=$V/lib/python3.14/site-packages LD_LIBRARY_PATH=$V/lib $V/bin/python3 smoke.py 000`

## Configuration (matches FZGM `roibin_b2`)
`roibin` { roi = `sz3` abs 10; background = `binning` {shape 2x2x1} -> `sz3` abs 100 },
`roi_size = {4,4,0}` (half-width -> 9x9 box, same as FZGM `roi_half_width=4`),
coordinate strategy with the published `.roi` peak lists. Scoped option names:
`/rb/roi:pressio:abs`, `/rb/background/sz3:pressio:abs`.

## Upstream bug found (worked around, not patched)
`roibin_impl.h` `restore_omp` for **1-D and 2-D** loops over `bins[2]` and `bins[3]`
of a 1-/2-element indexer (copy-paste from the 4-D version): out-of-bounds reads give
garbage trip counts, so `binning` **decompression hangs** on 2-D frames. The 3-D and
4-D versions are correct (upstream `test_roibin` is 3-D). Workaround: pass each frame
as 3-D `(1552, 1480, 1)` with bins `{2,2,1}` — identical result, no native code change.
Worth reporting upstream.

## Fairness notes
- Native ROIBIN-SZ does **not** store the peak coordinates in its archive; decompression
  reads them from the compressor options. FZGM stores the peak table in-archive
  (8 B/peak). For a like-for-like CR, add `8 * npeaks` bytes to native's size or report
  both.
- Timing here is Python wall time around `encode`/`decode`, min of 3, 20-core Xeon 8468;
  SZ3 itself runs single-threaded in this configuration (`roibin:nthreads`/
  `binning:nthreads` only parallelize ROI extraction and binning).

## Smoke result (EXAFEL, eb_roi=10, eb_bg=100, bin 2x2)

| frame | peaks | native CR | FZGM b2 CR | native ROI max err | native bg PSNR | FZGM b2 bg PSNR | native cmp GB/s | FZGM b2 cmp GB/s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| f000 | 108 | 403.4 | 97.3 | 9.9995 | 45.26 | 43.16 | 0.55 | 110.6 |
| f001 | 68 | 273.1 | 77.6 | 9.9985 | 45.98 | 44.16 | 0.62 | 113.4 |
| f050 | 219 | 92.2 | 41.9 | 9.9999 | 43.06 | 42.75 | 0.59 | 114.8 |

FZGM columns from `../results/exafel_eb10_100.csv`. Native wins CR by 2.2-4.2x with
slightly better background PSNR; FZGM is ~190x faster per frame (GPU vs CPU). The CR
gap is the coder: SZ3 (interpolation + Huffman + zstd) on a mostly-zero binned
background vs FZGM's TiledLorenzo + AdaptiveBitpack (per-block fixed width, which has a
floor on near-zero blocks). Three frames only — run the full 409 before quoting.

---

# 2026-09-28 (cont.) — specialization and background-coder sweep, EXAFEL volume

All FZGM numbers: whole 130-frame EXAFEL volume (1.19 GB, `roibin_volume_b2.toml`,
eb_roi=10, eb_bg=100, bin 2x2), H100, `fzgmod-cli -b --runs 5`, device-time GB/s, GPU
otherwise idle. Binary: worktree `~/FZGPUModules-paper-d511ebc` (d511ebc) **plus the
uncommitted TiledLorenzo fix below**. Scratch configs/results:
`/tmp/claude-1001/.../scratchpad/spec/` (var_*.toml, b_*_{off,auto}.json).

## Bug: TiledLorenzo warp-register specialization corrupts stacked 2-D slices
`modules/predictors/tiled_lorenzo/tiled_lorenzo_stage.h` `getFusedOp()` /
`getInverseFusedOp()` take the 2-D op whenever `tile_z == 1`, even when `dim_z > 1`.
The 2-D op sizes the work as `ntx*nty*64` (one slice) and its params carry no `dz`, so
only slice 0 is coded correctly. Symptom on the volume: Auto archive 1.41 MB vs 23.4 MB
staged (CR 848 vs 51), background max error ~4.3e11 on slices > 0 (PSNR -141 dB), ROI
still correct, exit code 0 — a silent failure. Reproduced on b08c406 and d511ebc.
Per-frame (dz=1) runs are unaffected and byte-identical.

Fix tested (uncommitted, in the d511ebc worktree): 2-D op only when
`tz == 1 && dz == 1`; otherwise the existing 3-D op with tz=1 (which never predicts
along z). Result: Auto archive byte-identical to staged (23,446,688 B), reconstruction
identical, staged output unchanged.

Paper impact: none found. `specialization_vs_native_full.yaml` uses 8x8x1 tiling only
on CESM-2D (dz=1); all 3-D datasets use 4x4x4 presets.

## Specialization on the ROIBIN pipeline
Installs one forward and one inverse warp-register group on the background branch
(Quantizer -> TiledLorenzo -> AdaptiveBitpack); the ROI branch (no predictor) and the
ROIBinSplit stage stay staged.
- Per 9.2 MB frame: no gain (0.080 ms staged vs 0.114 ms Auto compress) — below the
  small-input crossover; launch overhead dominates.
- Volume (with fix): compress 631 -> 830 GB/s (1.32x), decompress 479 -> 589 (1.23x).

## Background-coder sweep (volume; ROI branch unchanged; quantizer + TiledLorenzo fixed)
Reconstruction byte-identical to the baseline for every variant that decodes.

| background coder | CR | C GB/s off / auto | D GB/s off / auto | specializes |
|---|---:|---:|---:|---|
| AdaptiveBitpack (current) | 50.95 | 631 / 830 | 479 / 589 | yes |
| Zigzag-Bitshuffle-RZE | 68.08 | 468 / 467 | 372 / 372 | no |
| GolombRice | 71.65 | 460 / 461 | decode fails | no |
| Zigzag-Huffman (16-bit codes) | 74.44 | 409 / 409 | 321 / 321 | no |
| Zigzag-Bitshuffle-RZE-ANS (32-bit) | 85.36 | 436 / 436 | 367 / 367 | no |
| Zigzag-Bitshuffle-RZE-ANS (16-bit) | **85.81** | 445 / 444 | 363 / 363 | no |

GolombRice decompress: CUDA illegal memory access (mempool.cpp:221 reports it at the
next sync) — second bug, not investigated.

## Native ROIBIN-SZ, all 130 EXAFEL frames (per-frame, CPU, SZ3)
Aggregate CR **152.3** (150.2 if charged 8 B/peak for the coordinate table), max ROI
error 10.0000, median background PSNR 43.73 dB, median 0.555 GB/s compress /
0.521 GB/s decompress.

## Reading
Native keeps a 1.8x CR lead over the best FZGM back end (86 vs 150) at ~800x lower
throughput. Specialization only installs for the AdaptiveBitpack coder, so the
highest-throughput and highest-ratio FZGM points are different configurations.
Remaining CR gap is most likely SZ3's interpolation predictor + Huffman + zstd;
untested FZGM options: GInterp (cuSZ-Hi) predictor on the background, a GPU-Zstd
stage after Huffman. bin=1 (both bounds real) has not been swept yet.

---

# 2026-09-28 (cont. 2) — fixes, bin=1 sweep, PFPL back end

## Commits / fixes (FZGM worktree `~/FZGPUModules-paper-d511ebc`, branch `fix-tiled-lorenzo-stacked-slices`)
- `df85bad` fix: TiledLorenzo specialization on stacked 2-D slices (committed; not on main,
  whose checkout had another session's uncommitted work).
- GolombRice, **uncommitted**, two bugs, both only in standalone (file) decompress:
  1. Inverse output sized from `cached_orig_bytes_`, which only an in-process forward run
     sets; a `-x` decode fell back to the compressed size and wrote out of bounds
     (volume: crash; single frame: 2,177 silent memcheck errors). Fix: store the original
     byte count in the stage header (5 -> 9 bytes, as RZE does); 5-byte headers still load.
  2. `GrBitReader::refill()` prefetched up to 8 bytes past the stream end; the in-process
     buffer had a 16-byte tail pad, a file-loaded archive does not. Fix: per-chunk read
     limit (`comp_size - 4 - 4*kIntervalsPerChunk`), zero past the end.
  After both: 0 memcheck errors, recon byte-identical to baseline (frame and volume),
  compressed output unchanged, `test_golomb_rice` 12/12 (HeaderSerialization updated).
  Archives written before the fix still cannot be decoded standalone.
- GolombRice ran **staged** in every sweep (0 groups installed).

## bin=1 (both bounds real), EXAFEL volume, eb_roi=10, eb_bg=100
Every FZGM variant: ROI max err 10.0000, bg max err 100.0001 (float rounding), bg PSNR
49.09, recon byte-identical across coders.

| config | CR | specializes |
|---|---:|---|
| native ROIBIN-SZ (CPU, 130 frames; bg = SZ3 directly) | **22.73** (22.68 w/ peak table) | — |
| FZGM Zigzag-Huffman (16-bit) | **16.92** | no |
| FZGM Zigzag-Bitshuffle-RZE-ANS (16-bit) | 16.55 | no |
| FZGM Zigzag-Bitshuffle-RZE-ANS (32-bit) | 16.48 | no |
| FZGM GolombRice | 15.84 | no |
| FZGM TiledLorenzo-Zigzag-Bitshuffle-RZE | 13.79 | no |
| FZGM PFPL back end (Quant-Difference-Bitshuffle-RZE) | 13.52 | **yes** (chunk-cooperative) |
| FZGM TiledLorenzo-AdaptiveBitpack | 11.40 | **yes** (warp-register) |

Native bin=1 timing: median 0.191 GB/s compress / 0.307 decompress (CPU), max bg err 100.0.
bin=2 PFPL back end: CR 66.40, specializes; (bin=2 TiledLorenzo+RZE was 68.08, not
specialized).

**FZGM throughput for the bin=1 sweep is NOT valid**: the peak-memory session
(`~/FZGPUModules-memory-d511ebc`) was running fzgmod-cli on the same GPU. Rerun timing
when the GPU is free. CR and bounds are unaffected.

## Why the higher-ratio coders don't specialize
Planner verdict for every RZE/ANS/Huffman/GolombRice variant: 2 legal groups,
`no_profitable_implementation` — the chunk-cooperative strategy only has a registered
implementation for the PFPL chain (Quant -> Difference -> Bitshuffle -> RZE).

## GolombRice through chunk-cooperative specialization (variant H)
GolombRice is a chunk-cooperative SegmentCodec (forward only, int32, 16 KiB chunk). It
did not install after TiledLorenzo because TiledLorenzo's 64-element region does not
match the 16 KiB chunk. The designed chain — Quantizer (zigzag_codes, inplace_outliers,
ABS 100) -> Difference (int32, chunk 16384) -> GolombRice — installs one chunk-coop
forward group (inverse stays staged, by design), byte-identical to staged:
bin=1 CR 11.06 (bounds: ROI 10.0000, bg 100.0001), bin=2 CR 54.26. The 1-D Difference
predictor costs ratio vs TiledLorenzo (GolombRice after TiledLorenzo: 15.84 / 71.65).
Closing that needs a chunk-cooperative TiledLorenzo op (8x8 tiles are tile-major, so a
16 KiB chunk holds 64 whole tiles).

Fixes are on FZGPUModules main as 80d2e1a (TiledLorenzo) and 0bfbb6f (GolombRice).

---

# 2026-09-30 — clean timing campaign (paper candidate) + lsCOMP

**Build:** FZGPUModules `381a45e` (origin/main, includes the TiledLorenzo/GolombRice
fixes and the chunk-fusion single-pass encode), worktree
`~/FZGPUModules-roibin-381a45e/build_rel`. GPU verified idle before every cell (the
driver aborts otherwise). EXAFEL volume, 1.19 GB, eb_roi=10, eb_bg=100, `-b --runs 7`,
median device time; max phase CV 0.016. Driver: `clean_timing.py`; configs:
`variant_configs/`; results: `clean_timing_results.json`.
Every row: staged and specialized archives decode to byte-identical output; ROI max err
10.0000; bin=1 bg max err 100.0001 (bg PSNR 49.09); bin=2 bg PSNR 44.02 (no bg bound).

| background coder | bin=1 CR | bin=1 C/D GB/s (staged -> spec) | bin=2 CR | bin=2 C/D GB/s (staged -> spec) | installed |
|---|---:|---|---:|---|---|
| TiledLorenzo-AdaptiveBitpack | 11.40 | 205/209 -> **375/266** | 50.95 | 600/461 -> **818/547** | warp-register fwd+inv |
| Quant-Difference-Bitshuffle-RZE (PFPL) | 13.52 | 198/140 -> **326/264** | 66.41 | 579/369 -> **798/542** | chunk-coop fwd+inv |
| Quant-Difference-GolombRice | 11.06 | 154/99 -> 228/100 | 54.26 | 487/293 -> 641/293 | chunk-coop fwd only |
| TiledLorenzo-Bitshuffle-RZE | 13.79 | 141/139 | 68.08 | 445/356 | none |
| TiledLorenzo-GolombRice | 15.84 | 134/98 | 71.65 | 431/282 | none |
| TiledLorenzo-RZE-ANS (32-bit) | 16.48 | 133/137 | 85.36 | 415/351 | none |
| TiledLorenzo-RZE-ANS (16-bit) | 16.55 | 136/139 | 85.81 | 423/351 | none |
| TiledLorenzo-Huffman (16-bit) | **16.92** | 125/108 | 74.44 | 395/311 | none |
| **native ROIBIN-SZ** (CPU, SZ3) | **22.73** | 0.191/0.307 | **152.3** | 0.555/0.521 | — |

Specialization speedups at bin=1: bitpack 1.83x C / 1.27x D; PFPL back end 1.65x C /
1.89x D (+128 / +124 GB/s).

## lsCOMP (SC'25 GPU light-source compressor) on CXIDB 21
Input: 279 CXIDB 21 frames stacked as uint16 (raw int16 ADU + global offset 1717;
lossless conversion, all values integer, span 16,571). `lsCOMP_uint16 -d 279 1480 1552`.
Caveat: lsCOMP targets non-negative photon counts near zero; these are uncalibrated raw
frames with negative noise, and the offset raises the background level, which hurts its
ratio. Report as "on uncalibrated raw frames".

| mode | CR | C/D GB/s (lsCOMP end-to-end) | ROI max err | ROI frac err>10 | bg max err |
|---|---:|---|---:|---:|---:|
| lossless (-b 1 1 1 1 -p 1) | 1.37 | 342/301 | 2048 (7 of 640M values wrong) | — | — |
| uniform bins 20, no pooling | 2.19 | 445/317 | 19 | 0.49 | 2579 |
| paper bins 3/5/10/15, pool 0.5 | 2.06 | 432/318 | 14 | 0.24 | 1934 |

lsCOMP has no region-aware or pointwise bound; its lossless mode mis-decodes 7 values
(not magnitude-correlated) — a defect worth reporting upstream.

---

# 2026-09-30 (cont.) — CXIDB 21 replication, native parallel modes, coder overflow fix

## FZGM on CXIDB 21 (279 frames, 2.56 GB, 5,655 peaks, raw int16 ADU as f32)
Volume + z-indexed peaks built from the per-frame files (sorted filename order,
`cxidb21_volume_frame_order.txt`): `derived/CXIDB21_ROIBIN/cxidb21_volume_279x1480x1552.f32`,
`peaks/cxidb21_full.roi`. Same build (381a45e), protocol and driver
(`clean_timing.py CXIDB21`); results `clean_timing_results_CXIDB21.json`.
bin=1 rows: ROI max 10.0, bg max 100.0, bg PSNR 49.97; bin=2 bg PSNR 44.64.

| background coder | bin=1 CR | bin=1 C/D staged -> spec | bin=2 CR | bin=2 C/D staged -> spec |
|---|---:|---|---:|---|
| TiledLorenzo-AdaptiveBitpack (warp-register) | 15.05 | 212/217 -> **418/277** | 62.51 | 624/477 -> **927/569** |
| PFPL back end (chunk-coop fwd+inv) | 20.75 | 203/142 -> **342/271** | 107.89 | 605/381 -> **862/564** |
| Difference-GolombRice (chunk-coop fwd) | 14.81 | 158/101 -> 237/100 | 71.06 | 502/286 -> 676/299 |
| TiledLorenzo-GolombRice | 19.03 | 136/98 | 84.95 | 441/286 |
| TiledLorenzo-Huffman16 | 20.86 | 128/108 | 90.93 | 416/312 |
| TiledLorenzo-RZE | 21.40 | 144/142 | 112.41 | 462/370 |
| TiledLorenzo-RZE-ANS32 | 26.08 | 139/140 | 145.07 | 444/366 |
| TiledLorenzo-RZE-ANS16 | **26.26** | 142/141 | **146.64** | 451/366 |

**The two GolombRice bin=1 rows in `clean_timing_results_CXIDB21.json` are invalid as
recorded** (TiledLorenzo-GolombRice: bg max err 14,854, last 13 frames decoded as zeros;
Difference-GolombRice: off/auto recon differed). Cause: `golombRicePackKernel` computed
`cid * scratch_stride` in uint32; stride ~28.8 KB wraps 2^32 at chunk ~149k (~2.4 GB of
int32 input). Fixed (64-bit offsets; RZE's pack kernel had the same pattern, wrapping at
4 GB) — FZGPUModules `db387d3` on branch `fix-coder-pack-offset-overflow` (pushed), and
`a062d88` on local main. After the fix both rows: 0 bound violations, off/auto recon
identical, CR unchanged (19.03 / 14.81). Their timing was not re-measured.

## Native ROIBIN-SZ, both corpora (`run_native.py`)
SZ3 3.3.0 via LibPressio, 20-core Xeon 8468. Codec GB/s = frame bytes / summed
encode (decode) time. "20 frames" = 20 worker processes compressing whole frames;
its aggregate lies between the wall-time bound (encode + decode + quality checks, so
conservative) and 20x the per-worker codec rate.

| corpus | bin | CR (w/ peak table) | serial C/D GB/s | 20 workers: per-worker C/D | 20 workers: wall bound | 20x per-worker C |
|---|---|---:|---|---|---:|---:|
| EXAFEL | 1 | 22.73 (22.68) | 0.191/0.307 | 0.161/0.266 | 0.95 | ~3.2 |
| EXAFEL | 2 | 152.3 (150.2) | 0.555/0.521 | 0.455/0.426 | 1.29 | ~9.1 |
| CXIDB 21 | 1 | 36.88 (36.85) | 0.205/0.413 | 0.172/0.316 | 1.17 | ~3.4 |
| CXIDB 21 | 2 | 234.3 (233.3) | 0.598/0.587 | 0.523/0.465 | 1.72 | ~10.5 |

All native runs: ROI max err 10.0001 (CXIDB 21; float rounding) / 10.0000 (EXAFEL),
bin=1 bg max 100.00. SZ3 OpenMP (`sz3:openmp`, 20 threads) gave no speedup on 9 MB
frames (0.186 / 0.170 GB/s compress) and decompression stays single-threaded; frame-level
parallelism is the realistic CPU scaling path.

## Remaining ratio gap (FZGM best / native), bin=1: EXAFEL 16.92/22.73 = 0.74,
CXIDB 21 26.26/36.88 = 0.71. bin=2: 85.81/152.3 = 0.56, 146.64/234.3 = 0.63.
