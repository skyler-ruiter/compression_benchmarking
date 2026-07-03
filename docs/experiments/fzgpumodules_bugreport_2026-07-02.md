# FZGPUModules bug reports — from FZGM-vs-native validation, 2026-07-02

> **Update, later the same day:** commit `8e581a8` ("fixes cusz-hi bugs and rre/rze bugs
> found in benchmarking") fixed issues 1, 2, and 3b/3c below. Root causes, confirmed:
> a missing `__syncthreads()` in the `RREStage`/`RZEStage` encode kernels (shared root
> cause of issue 1's silent corruption *and* issue 2's crash — the corruption was
> misdiagnosed here as a `Bitshuffle` bug; it wasn't), an uninitialized outlier-port tail
> in `ginterp_stage.cu` causing the `-b`-repeated-run crashes, and a `MINIMAL`-mode
> buffer-restore bug in `dag.cpp`. Retested and confirmed: cuSZ-Hi CR/TP no longer crash,
> PFPL's NYX corruption is gone. **Issue 3a (cuSZ-Hi TP's ~65% CR deficit) is explicitly
> NOT fixed** — bit-for-bit unchanged, a separate bug from the crash/corruption ones. A new,
> smaller CR gap (~36%) also surfaced on cuSZ-Hi CR mode's NYX field once it became
> reachable. See `docs/experiments/observed_errors.md` E10/E11/E13/E15/E16/E17 for the full,
> current status — the section below is preserved as the original report for reference.

Handoff doc for the FZGPUModules side. These are library-level findings from running
`fzgmod-cli` against real SDRBench-scale data through the `cusz_hi_tp`, `cusz_hi_cr`, and
`pfpl` presets — all repro commands below use `fzgmod-cli` directly, no benkit involved.
Cross-referenced to `docs/experiments/observed_errors.md` (E10/E11/E12/E15) in the
compression_benchmarking repo if more context is needed.

Environment: BigRed200, A100-SXM4-40GB, driver 595.71.05, CUDA 12.6.
`fzgmod-cli` build: `/N/u/sruiter/BigRed200/research/FZGPUModules/build_bench/bin/fzgmod-cli`,
built from commit `e30d0cc` ("fixes configs again for cusz-hi").

Datasets used (all under `$BENCHKIT_DATA_ROOT`, SDRBench originals):

| Dataset/field | Path | dims (fast&rarr;slow) | dtype | bytes |
|---|---|---|---|---|
| CESM-2D/CLDHGH | `CESM_1800x3600/CLDHGH_1_1800_3600.f32` | 3600x1800 | f32 | 25,920,000 |
| HURR/TC | `HURR_100x500x500/TCf48.bin.f32` | 500x500x100 | f32 | 100,000,000 |
| NYX/temperature | `NYX_512x512x512/temperature.f32` | 512x512x512 | f32 | 536,870,912 |
| HACC/vx | `HACCM_280953867/vx.f32` | 280953867 (1-D) | f32 | 1,123,815,468 |

---

## 1. [CRITICAL — new] `pfpl.toml`: silent data corruption on NYX/temperature after Bitshuffle element_width=4 fix

**Severity:** Critical. No crash, no non-zero exit, `--report-json` reports `status: ok`. Only
independently recomputed quality metrics reveal the corruption. This is more dangerous than a
crash because nothing in the tool's own output signals a problem.

**Preset:** `examples/presets/pfpl.toml`, current version (commit `79dc3ba`, `Bitshuffle`
stage changed from `element_width = 2` to `element_width = 4`).

**Repro (exact commands, pulled verbatim from the benchkit run logs — note there is
no `-m`/`-e`: the error bound and mode are already baked into the TOML's `error_bound =
0.001` / `error_bound_mode = "NOA"`, and passing `-c` uses those, not CLI flags):**
```bash
fzgmod-cli -z -i $BENCHKIT_DATA_ROOT/NYX_512x512x512/temperature.f32 \
    -o /tmp/nyx_temp.fzm -l 512x512x512 -t f32 \
    -c examples/presets/pfpl.toml --report-json /tmp/nyx_z.json

fzgmod-cli -x -i /tmp/nyx_temp.fzm -o /tmp/nyx_temp.dec.f32 \
    -c examples/presets/pfpl.toml \
    --compare $BENCHKIT_DATA_ROOT/NYX_512x512x512/temperature.f32 \
    --report-json /tmp/nyx_x.json
```
(An earlier version of this report incorrectly included `-m noa -e 1e-3` on these commands —
drop them; passing both `-c` and `-m`/`-e` may behave unpredictably. If reproducing manually,
copy the `[[stage]]` blocks from `examples/presets/pfpl.toml` verbatim rather than trying to
reconstruct the eb/mode via CLI flags.)

**Symptom:** Both commands exit 0. The **tool's own** `[Decompress Report]` (`-x`'s stdout)
already shows the corruption directly, no external comparison needed:
```
[Decompress Report]
  Output size:     536870912 bytes
  Time:            263.382 ms
  Throughput:      2.04 GB/s
  Value Range:     [2.28e+03, 4.78e+06] (Span: 4.78e+06)
  Max Abs Error:   2.05e+13
  PSNR:            -84.64 dB
  NRMSE:           1.71e+04
```
`compressed_bytes`/`cr` from the `-z` step look completely normal (CR ≈ 48.04, matching
native PFPL's 48.32 on the same field/bound) — only the decompress-side quality numbers
reveal anything is wrong. Independently recomputing quality from the raw bytes (benchkit's
own harness-side check, not the tool's self-report) gives the same magnitude:

```
psnr        = -84.64 dB      (every other field/pipeline this session: 63.9–75.8 dB)
nrmse       = 17067.35       (every other field/pipeline this session: ~0.0006)
max_abs_err = 20531243901466.38     (requested eb_abs ≈ 4780.30 — off by ~4.3 billion×)
```

This is not "quality slightly worse" — the reconstructed array is unrelated to the input.

**What's known:**
- CESM/CLDHGH, HURR/TC, and HACC/vx all decode *correctly* through the exact same preset
  (`element_width = 4`) at the same error bound (eb=1e-3, NOA/rel_range) in the same session.
  Only NYX/temperature — the one 3-D, 512×512×512 field — fails.
- With the *previous* preset (`element_width = 2`), NYX/temperature decoded correctly (just
  with a real ~27% CR deficit vs. native, before the fix that's the subject of this whole
  issue). So `element_width=4` is the variable that introduced the corruption, isolated to
  this field.
- Unlike the `cusz_hi_cr`/`cusz_hi_tp` bugs below (issues 2 and 3), **this one is not limited
  to `-b`** — the benchmark call also completes without a crash (`status: ok`, populated
  per-stage timing) despite the corrupted data. There is no invocation mode where this
  pipeline fails loudly on NYX/temperature; every path (`-z`+`-x`, or `-b`) reports success.
- Pipeline: `Quantizer(linear=false, zigzag, radius=32768) → Difference(chunk_size=16384) →
  Bitshuffle(element_width=4, block_size=16384) → RZE(word_size=1)`.

**Hypothesis (not confirmed — needs actual investigation):** `Bitshuffle` groups its input
into `element_width`-byte-wide shuffle planes; NYX/temperature is the only field of the four
whose total element count (512³ = 134,217,728) combined with `block_size=16384` and
`element_width=4` might produce a block/tail-handling edge case that the other three field
sizes don't hit (their sizes: 6,480,000 / 25,000,000 / 280,953,867 elements — none share a
common divisibility property with 134,217,728 that would obviously explain it, so this needs
checking against the actual code, not assumed from the sizes). Check specifically:
- Does `BitshuffleStage` (or wherever `element_width` changes shuffle-plane sizing) have a
  fast path or SIMD-width assumption for `element_width=4` that silently mis-handles a block
  boundary for this input size?
- Does the *inverse* (decompress-side) Bitshuffle correctly recover the same block layout
  the forward pass used, or is there a symmetry bug that only manifests for particular
  `(total_size, block_size, element_width)` combinations?
- Is the `Difference` stage's `chunk_size=16384` output feeding `Bitshuffle` at a size/stride
  that only breaks alignment for 512³ data?

**Suggested first step:** add a golden round-trip test for `Bitshuffle(element_width=4,
block_size=16384)` over exactly 134,217,728 `uint32` elements (matching NYX/temperature's
post-Quantizer/Difference code count) and confirm bit-for-bit round trip in isolation, without
the rest of the pipeline — that will confirm/deny whether this is a `BitshuffleStage` bug
specifically vs. an interaction with `Difference`/`RZE`.

---

## 2. [OPEN] `cusz_hi_cr.toml`: `RREStage` illegal-memory-access / invalid-argument crash on every multi-D field

**Severity:** High — this preset cannot complete a single run on real (non-toy) data.

**Preset:** `examples/presets/cusz_hi_cr.toml`, current version (commit `e30d0cc`,
`memory_strategy = "MINIMAL"`, no `input_size` hint).

**Repro — three separate calls per field, exactly as benchkit runs them (no `-m`/`-e`; eb and
mode are baked into the TOML):**
```bash
# 1. compress — succeeds on all 3 multi-D fields
fzgmod-cli -z -i $BENCHKIT_DATA_ROOT/CESM_1800x3600/CLDHGH_1_1800_3600.f32 \
    -o /tmp/cesm.fzm -l 3600x1800 -t f32 \
    -c examples/presets/cusz_hi_cr.toml --report-json /tmp/cesm_z.json

# 2. standalone decompress — succeeds for CESM and HURR, FAILS for NYX
fzgmod-cli -x -i /tmp/cesm.fzm -o /tmp/cesm.dec.f32 \
    -c examples/presets/cusz_hi_cr.toml \
    --compare $BENCHKIT_DATA_ROOT/CESM_1800x3600/CLDHGH_1_1800_3600.f32 \
    --report-json /tmp/cesm_x.json

# 3. benchmark (repeated in-process runs) — FAILS for CESM and HURR (never reached for NYX,
#    since step 2 already failed there)
fzgmod-cli -b -i $BENCHKIT_DATA_ROOT/CESM_1800x3600/CLDHGH_1_1800_3600.f32 \
    -l 3600x1800 -t f32 -c examples/presets/cusz_hi_cr.toml --runs 6 \
    --compare $BENCHKIT_DATA_ROOT/CESM_1800x3600/CLDHGH_1_1800_3600.f32 \
    --report-json /tmp/cesm_b.json
```
(Substitute `-l 500x500x100` / `HURR_100x500x500/TCf48.bin.f32` or `-l 512x512x512` /
`NYX_512x512x512/temperature.f32` for the other two fields.)

**Symptom — this is the important part, and corrects an earlier version of this report that
conflated two different failure modes:**

```
CESM/CLDHGH:
  step 1 (-z compress)          exit 0, succeeds
  step 2 (-x decompress, alone) exit 0, succeeds — correct reconstruction (PSNR 66.81 dB)
  step 3 (-b, 6 in-process reps) exit 1 — CRASHES:
    [fzgmod] CUDA error at modules/coders/rre/rre_stage.cu:448 —
    cudaStreamSynchronize(stream) → an illegal memory access was encountered

HURR/TC:
  step 1 (-z compress)          exit 0, succeeds
  step 2 (-x decompress, alone) exit 0, succeeds — correct reconstruction (PSNR 68.54 dB)
  step 3 (-b, 6 in-process reps) exit 1 — CRASHES, same error as above (rre_stage.cu:448)

NYX/temperature:
  step 1 (-z compress)          exit 0, succeeds
  step 2 (-x decompress, alone) exit 1 — CRASHES immediately:
    [fzgmod] CUDA error at modules/coders/rre/rre_stage.cu:448 —
    cudaStreamSynchronize(stream) → an illegal memory access was encountered
  step 3 (-b)                   never reached (benchkit aborts the cell after step 2 fails)
```

**So there are two distinct bugs sharing one crash signature, not one bug on three fields:**
1. **CESM and HURR: a single compress+decompress round trip is entirely correct.** The crash
   only appears when running multiple repetitions in one process (`-b --runs N`). This points
   at state that isn't reset/reinitialized between reps inside `RREStage` (or wherever `-b`'s
   loop reuses pipeline/stage objects across reps) — e.g. a buffer, offset table, or stream
   left in a stale state from rep *k* that rep *k+1* reads incorrectly.
2. **NYX: fails on the very first, single, standalone decompress** — a fundamentally different
   (and more serious) bug specific to this field's size/shape, unrelated to repeated-run state.

Note: on an *earlier* build (before the `MINIMAL` memory-strategy fix, commit prior to
`e30d0cc`), CESM's step 3 instead failed with a different error at a different line:
```
[fzgmod] CUDA error at modules/coders/rre/rre_stage.cu:444 —
cudaMemcpyAsync(d_out + h_out_off[i], d_in + h_in_off[i], h_orig_sz[i],
cudaMemcpyDeviceToDevice, stream) → invalid argument
```
**This is the important clue:** switching `memory_strategy` from `PREALLOCATE` to `MINIMAL`
changed *where* it crashes (444→448) but did not fix it — and it was always the `-b` step, not
step 1 or 2, that failed for CESM/HURR, both before and after that change. That rules out a
PREALLOCATE buffer-sizing issue as the root cause; the bug is in the actual offset/size logic
inside `RREStage`'s handling of repeated runs.

**Revised hypothesis, given the `-b`-only failure mode above:** since `cusz_hi_tp.toml` (see
issue 3 below) crashes with the *identical* `rre_stage.cu:448` error, also only in `-b` mode,
also only on the fields where the single-pass round trip works fine — the common factor is
**not** the Huffman/variable-length-segment structure (an earlier version of this report
guessed that; it doesn't hold since `cusz_hi_tp` has no Huffman stage and hits the same bug).
The common factor across both presets is `RREStage` itself running inside a `-b` repeated-run
loop. Both presets share `RREStage` in their chain (`cusz_hi_cr`: `RRE(word_size=4)` after the
Huffman merge; `cusz_hi_tp`: `RRE1`/`RRE2` in the codes/outlier chains) — `RREStage` is the
one stage common to both crash sites.

**Hypothesis:** `RREStage` likely holds or derives some per-run state (an offset table, a
scratch buffer, or stream-associated resource) during `finalize()`/first execution that is not
correctly reset or re-derived on the second and subsequent reps of a `-b` loop — rep 1 succeeds
(consistent with the standalone `-x` call succeeding), but something left over from rep 1
corrupts rep 2 onward. This would explain why a single `-z`+`-x` round trip is clean but `-b
--runs N` (N≥2) is not.

**Suggested first step:** Run `-b --runs 2` (the minimum that exercises a second rep) on CESM
against `cusz_hi_cr.toml` with `--bounds-check` (per `docs/cli.md`) and/or debug logging around
`RREStage`'s per-rep setup — specifically, diff whatever `RREStage` computes/allocates on rep 1
vs. rep 2 to find what's stale. Compare against a stage *not* implicated (e.g. `Bitshuffle`,
which appears in `cusz_hi_tp`'s working codes-chain) to confirm the bug is specific to
`RREStage` and not something upstream (e.g. `MergeStage`) leaking bad state into it.

---

## 3. [OPEN — regressed by the MINIMAL memory-strategy change] `cusz_hi_tp.toml`: silent corruption on one field, `-b`-only crashes on two others, real CR deficit on the field that fully completes

**Severity:** High. Went from "usable with a known CR deficit" to "silently corrupts one field
and crashes `-b` on two others" after the same `PREALLOCATE`→`MINIMAL` change described in
issue 2.

**Preset:** `examples/presets/cusz_hi_tp.toml`, current version (commit `e30d0cc`).

### 3a. CESM/CLDHGH — the only field where `-b` also completes; CR is 65% below native and violates its own eb (unchanged by the recent fix)

**Repro (no `-m`/`-e` — see the note in issue 1 about why):**
```bash
fzgmod-cli -z -i $BENCHKIT_DATA_ROOT/CESM_1800x3600/CLDHGH_1_1800_3600.f32 \
    -o /tmp/cesm.fzm -l 3600x1800 -t f32 \
    -c examples/presets/cusz_hi_tp.toml --report-json /tmp/cesm_z.json
fzgmod-cli -x -i /tmp/cesm.fzm -o /tmp/cesm.dec.f32 \
    -c examples/presets/cusz_hi_tp.toml \
    --compare $BENCHKIT_DATA_ROOT/CESM_1800x3600/CLDHGH_1_1800_3600.f32 \
    --report-json /tmp/cesm_x.json
```

**Symptom:** Runs to completion, exit 0. But:
```
CR (this pipeline)   = 4.66
CR (native cuszhi -s tp, same field/eb) = 13.33     →  -65.0% relative
PSNR (this pipeline) = 66.81 dB
PSNR (native)        = 66.54 dB   (fzgm side is not lower quality — rules out "less
                                    compression for more headroom" as the explanation)
eb_satisfied         = false      (max_abs_err slightly exceeds eb_abs_effective)
```
Since PSNR is *not* lower on the FZGM side, whatever's causing the CR shortfall isn't a
tuning/headroom tradeoff — the encoding chain (`GInterp → Zigzag → Bitshuffle → RRE` for
codes, `Merge → Bitshuffle → RRE → RZE` for the outlier segment) is producing a
larger-than-expected bitstream for the same reconstruction quality. Worth comparing
per-stage output byte counts (`--profile`/`--print-pipeline`) between this pipeline and
native cuszhi's own TP-mode backend to see which stage is responsible.

**Confirmed unaffected by the recent preset fix:** CR, PSNR, and eb_satisfied are all
bit-for-bit identical before and after the `memory_strategy` PREALLOCATE→MINIMAL change —
this is a real fidelity bug in the encoding chain itself, not a memory-strategy artifact.

### 3b. HURR/TC — single round-trip is clean; `-b` crashes (same pattern as issue 2)

**Repro:** Same 3-step pattern as issue 2's repro (compress, standalone decompress, then
`-b --runs N`), substituting `-l 500x500x100` / `HURR_100x500x500/TCf48.bin.f32` and
`cusz_hi_tp.toml`.

**Symptom, verified from the actual run logs:**
```
step 1 (-z compress)           exit 0, succeeds
step 2 (-x decompress, alone)  exit 0, succeeds — correct reconstruction (PSNR 68.54 dB,
                                matches the CR=12.89/PSNR=68.54 result from the original,
                                pre-retest pass)
step 3 (-b, 6 in-process reps) exit 1 — CRASHES:
    [fzgmod] CUDA error at modules/coders/rre/rre_stage.cu:448 —
    cudaStreamSynchronize(stream) → an illegal memory access was encountered
```
Identical crash signature and identical failure pattern (works standalone, fails only in
`-b`) to issue 2's CESM/HURR cases — see issue 2's revised hypothesis; this is very likely
the same underlying `RREStage` repeated-run bug, not a separate issue.

### 3c. NYX/temperature — a second, distinct bug: standalone decompress "succeeds" but silently corrupts, then `-b` crashes on top of that

**Repro:** Same 3-step pattern, `-l 512x512x512`, `NYX_512x512x512/temperature.f32`.

**Symptom, verified from the actual run logs — this is worse than a crash:**
```
step 1 (-z compress)           exit 0, succeeds
step 2 (-x decompress, alone)  exit 0 — but silently wrong:
    Max Abs Error:   3.21e+08
    PSNR:            13.33 dB      (original pre-retest pass, same field/preset: 75.75 dB)
    NRMSE:           2.16e-01      (original pass: ~0.0001-scale)
step 3 (-b, 6 in-process reps) exit 1 — CRASHES with a *different* stage than 3b:
    [fzgmod] CUDA error at modules/coders/rze/rze_stage.cu:446 —
    cudaMemcpyAsync(d_out + h_out_off[i], d_in + h_in_off[i], h_orig_sz[i],
    cudaMemcpyDeviceToDevice, stream) → invalid argument
```
**Two things are wrong here, and they need to be treated as separate bugs:**
1. The standalone decompress itself regressed from PSNR 75.75 dB (original pass, same
   preset+field, before the `MINIMAL` change) to 13.33 dB now — a real, silent correctness
   regression on this field, same family of bug as issue 1's pfpl/NYX corruption (E15), just
   less extreme in magnitude. This is **not** the RRE/RZE crash — it happens even in a plain,
   single `-z`+`-x` call with no repeated runs involved. Since `cusz_hi_tp.toml`'s only change
   was `memory_strategy` PREALLOCATE→MINIMAL, that's the prime suspect for this one.
2. Separately, `-b` crashes with an `RZEStage` error — a different stage than 3b's `RREStage`
   crash, but the same general shape (offset mismatch in a `cudaMemcpyAsync`). This may or may
   not share a root cause with issue 2/3b's `-b`-repeated-run bug; worth checking once 1 is
   fixed and `-b` can actually be reached with a correct rep-1 baseline to compare against.

**Suggested next step:** first isolate #1 by reverting just `cusz_hi_tp.toml`'s
`memory_strategy` back to `PREALLOCATE` and re-running the standalone `-z`+`-x` on NYX only —
if PSNR goes back to ~75.75 dB, that confirms `MINIMAL` is responsible for this field's
decompress corruption specifically. (Note: this is a *different* preset/pipeline from issue
1's pfpl/NYX corruption — `pfpl.toml` still uses `PREALLOCATE` and was never touched on that
axis, only its `Bitshuffle element_width` changed, so the two corruption bugs are probably
unrelated in cause even though both happen to land on the NYX/temperature field. Don't assume
one fix addresses both.) Once #1 is isolated, address the `-b`-only crashes (2, 3b, and
whatever's left of 3c) as the repeated-run bug described in issue 2.

---

## 4. [Design gap, not a crash] No 1-D preset for cuSZ-Hi or cuSZp3's 2-D `TiledLorenzo` variant

**Severity:** Low — not a bug, but blocks fair comparison on 1-D data (e.g. HACC particle
fields) until addressed.

- `cusz_hi_tp.toml` / `cusz_hi_cr.toml`: `GInterpStage::setDims` hard-requires `dims[1] > 1`
  (multi-level spline interpolation is inherently ≥2-D by design — this part is expected,
  not a bug, per `docs/stages/ginterp.md`). There's no 1-D fallback path or preset, so cuSZ-Hi
  simply cannot be benchmarked against 1-D particle data at all right now.
- `cuszp3.toml`'s `TiledLorenzo(tile_x=8, tile_y=8)` assumes ≥2-D structure; fed a `[N, 1, 1]`
  shaped field (HACC), it produces degenerate 8×1 tiles and the compressed output actually
  *grows* (CR 0.69) rather than erroring — silently wrong rather than loudly wrong, which is
  worth a guard/warning even if a 1-D preset isn't added.

**Ask:** either (a) add a 1-D-specific preset for cuSZp3 (skip `TiledLorenzo`, go straight
`Quantizer → AdaptiveBitpack`, which `cuszp3_fixed.toml` already does — may just need
promoting/documenting as the 1-D path), or (b) have `TiledLorenzo`/`AdaptiveBitpack` detect
and warn on degenerate tile shapes (`tile_y=8` against `dims[1]=1`) rather than silently
producing expansion.

---

## Priority order for the module library agent

1. **Issue 1 (E15) and issue 3c's decompress corruption** — silent, non-crashing corruption is
   the most dangerous class of bug in this list; both happen to land on NYX/temperature but
   are two different presets/pipelines (`pfpl.toml` unchanged `PREALLOCATE`, `cusz_hi_tp.toml`
   switched to `MINIMAL`) — treat as two separate bugs, don't assume a shared cause. Fix or at
   least add an assertion so each fails loudly instead of passing silently.
2. **Issue 2 / 3b's `-b`-only crash** — now understood to be one bug shared by both presets
   (`RREStage` breaking specifically on rep 2+ of a `-b` loop, not on a single pass) — a single
   fix here likely closes issue 2 (CESM/HURR), 3b, and unblocks re-testing 3c's separate `-b`
   crash (`RZEStage`, possibly the same class of bug in a different stage).
3. **Issue 3c's `-b` crash (`RZEStage`)** — can't be usefully investigated until 3c's
   standalone-decompress corruption (priority 1) is fixed, since `-b` never gets a correct
   rep-1 baseline to compare against on this field today.
4. **Issue 3a** — the CR/eb fidelity gap in `cusz_hi_tp`'s CESM case; lower urgency since
   nothing crashes or corrupts, but it means the TP-mode port isn't representative of the real
   algorithm yet.
5. **Issue 4** — preset/documentation gap, cheapest to close, no urgency.
