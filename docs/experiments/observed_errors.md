# Observed Errors — Causes and Status

Running log of failure modes encountered across experiments, with root cause analysis
and open questions. Add new entries as failures are diagnosed.

---

## E1 — Unknown stage type: N (fzgmod-cli decompression)

**Status:** Fixed (binary patch, 2026-07-02)

**Symptom:**
```
[fzgmod-cli] error: Unknown stage type: 14
```
Decompression exits 1. Compression succeeds. `x.json` has `"status": "error"`.

**Affected pipelines:** pfpl, pfpl_minimal (any pipeline using `Quantizer` or `Difference` stages)

**Root cause:**
`fzgmod-cli -x` reconstructs the pipeline by reading stage-type integers from the `.fzm`
header and dispatching through `createStage()` in `stage_factory.h`. `Quantizer` (enum 14)
and `Difference` (enum 15) were not registered in `createStage()`, so decompression failed
for any `.fzm` produced by a Quantizer-based pipeline.

Note: `-c pipeline.toml` is **silently ignored** in `-x` mode — `run_decompress()` in
`cli.cpp` calls `Pipeline::decompressFromFile()` unconditionally regardless of
`config_path`. Passing `-c` to the adapter's decompress call has no effect on the current
binary; it is retained as a forward-compat hint.

**Fix:** Added `case StageType::QUANTIZER:` (and `DIFFERENCE`) to `createStage()` in
`stage_factory.h` and rebuilt `build_bench/bin/fzgmod-cli`.

**First seen:** fzgm_scaling_v2, all pfpl/pfpl_minimal cells, 2026-06-30

---

## E2 — OOM: failed to preallocate pipeline input pad buffer

**Status:** Expected / by design (VRAM ceiling hit)

**Symptom:**
```
[fzgmod-cli] error: Failed to preallocate pipeline input pad buffer (12000002048 bytes);
pool may be exhausted
```
Compression exits 1. Only occurs with `memory_strategy = "PREALLOCATE"`.

**Affected pipelines:** pfpl (PREALLOCATE strategy)

**Root cause:**
PREALLOCATE mode reserves all pipeline buffers upfront before any kernel launches. For a
12 GB input, this requires allocating the full 12 GB input pad buffer in one shot, which
combined with other pipeline allocations exceeds the A100-40 GB VRAM budget.

**VRAM ceiling:** pfpl compress fails at 12 GB input, passes at 4 GB. Ceiling is somewhere
in the 4–12 GB range (8 GB compress succeeds but decompression hits E4 before this bound
can be confirmed cleanly).

**Fix:** None — this is the memory ceiling the scaling experiment was designed to find.
Use `memory_strategy = "MINIMAL"` (pfpl_minimal) to push the ceiling higher, though that
has its own limit (see E3).

**First seen:** fzgm_scaling_v2, pfpl synth_12GB/synth_16GB, 2026-07-02

---

## E3 — OOM: failed to allocate intermediate pipeline buffer (MINIMAL strategy)

**Status:** Expected / by design (VRAM ceiling hit)

**Symptom:**
```
[fzgmod-cli] error: Failed to allocate buffer: Difference_to_Bitshuffle
```
(Buffer name reflects the stage boundary where allocation failed.)

**Affected pipelines:** pfpl_minimal (MINIMAL strategy)

**Root cause:**
MINIMAL mode allocates buffers on demand as each stage runs, rather than preallocating
the whole pool. For large inputs the per-stage intermediate buffers still exceed VRAM at
some point; for pfpl_minimal at 12 GB, the `Difference_to_Bitshuffle` intermediate buffer
could not be allocated.

**VRAM ceiling:** Same effective ceiling as pfpl for this experiment (fails at 12 GB
compress). MINIMAL strategy does not help here — the intermediate buffer at this stage is
comparable in size to the input.

**Fix:** None — expected finding. The ceiling for pfpl_minimal is similar to pfpl on
the Quantizer+Difference+Bitshuffle+RZE topology.

**First seen:** fzgm_scaling_v2, pfpl_minimal synth_12GB/synth_16GB, 2026-07-02

---

## E4 — CUDA illegal memory access in RZE decompression

**Status:** Open — binary bug in FZGPUModules, needs upstream fix

**Symptom:**
```
[fzgmod] CUDA error at .../modules/coders/rze/rze_stage.cu:450 —
cudaStreamSynchronize(stream) → an illegal memory access was encountered
```
Decompression exits 1. Compression of the same file succeeds. Only observed at large
input sizes.

**Affected pipelines:** pfpl, pfpl_minimal (both use RZE as the final entropy coder)

**Root cause (hypothesis):**
An out-of-bounds memory access in the RZE decompression kernel at `rze_stage.cu:450`.
Likely a buffer sizing issue that only manifests above a certain input-size threshold —
compress at 8 GB works but decompress at 8 GB crashes, suggesting the decompressor
allocates or indexes differently than the compressor for this data volume.

**Possibly related fix (2026-07-02, unverified for this entry):** FZGPUModules commit
`8e581a8` added a missing `__syncthreads()` in `rze_stage.cu`'s *encode* kernel that was
racing the copy-out loop and producing non-deterministic blobs (root cause of E10/E15,
confirmed at SDRBench field sizes, not the multi-GB scale this entry is about). Same file,
same `cudaStreamSynchronize → illegal memory access` signature, but a different kernel
(encode vs. this entry's decode-side crash) and a very different data scale (E4 is ≥8 GB;
E10/E15 were confirmed at ≤512 MB) — plausible the fix helps here too, but **not confirmed**;
needs the scaling experiment (`fzgm_scaling_v2.yaml`, pfpl/pfpl_minimal ≥8 GB) re-run against
the fixed build before closing this entry.

**VRAM ceiling interaction:** This error appears before the OOM ceiling (E2/E3), so the
true VRAM limit for pfpl decompression is lower than for compression. The 8 GB size is
the first where this is triggered; sizes ≤ 4 GB decompress cleanly.

**Workaround:** None currently. Avoid pfpl on inputs ≥ 8 GB until the upstream RZE
decompression kernel is fixed.

**Needs:** Investigation at `rze_stage.cu:450` in FZGPUModules — check buffer allocation
size for the decompression path vs. the compression path at the same input size. A
minimal repro is: compress any ~8 GB f32 array with a Quantizer→RZE pipeline, then
decompress the resulting `.fzm`.

**First seen:** fzgm_scaling_v2, pfpl/pfpl_minimal synth_8GB decompress, 2026-07-02

---

## E5 — OOM: failed to allocate buffer (fzgpu_minimal benchmark)

**Status:** Expected / by design (VRAM ceiling hit)

**Symptom:**
```
[fzgmod-cli] error: Failed to allocate buffer: BitplaneRZE_to_LorenzoQuant
```
Benchmark step exits 1. Occurs during the `-b` (benchmark) invocation, not compress/decompress.

**Affected pipelines:** fzgpu_minimal (BitplaneRZE + Lorenzo pipeline)

**Root cause:**
The benchmark allocates both the compression and decompression buffers simultaneously for
in-process timing (it runs N compress+decompress reps without freeing between phases on
each rep). For the fzgpu_minimal topology, the combined buffer footprint at 12 GB input
exceeds the A100-40 GB VRAM budget.

**VRAM ceiling:** fzgpu_minimal benchmark passes at 8 GB, fails at 12 GB. Ceiling is
in the 8–12 GB range. Confirmed by the 8 GB success in fzgm_scaling_v2.

**Fix:** None for this experiment. If 12 GB+ is needed, investigate whether `-b` can be
run with a reduced buffer pool, or run compress and decompress separately (benchkit's
single-shot paths) for size/quality at cost of timing reliability.

**First seen:** fzgm_scaling_v2, fzgpu_minimal synth_12GB/synth_16GB, 2026-06-30 (confirmed 2026-07-02)

---

## E6 — eb_satisfied = False on synth_2GB (pfpl, NOA mode)

**Status:** Under investigation — likely floating-point accumulation at scale, not a real violation

**Symptom:**
`eb_satisfied: False` in the result row for pfpl SCALING-SYNTH/synth_2GB at eb=0.001,
despite PSNR=92.8 dB and CR=1601.6. All other synth sizes at the same eb pass.

**Affected pipelines:** pfpl only (pfpl_minimal at 2 GB passes)

**Root cause (hypothesis):**
The error-bound check compares harness-computed `max_abs_err` against the effective
absolute bound derived from the range. A marginal violation (err_over_bound slightly > 1)
at exactly the 2 GB size may reflect floating-point accumulation in the Quantizer or
Difference stages at that chunk boundary, or a harness rounding difference at this data
volume. The fact that pfpl_minimal passes at the same size is suspicious — the two
pipelines use identical stage parameters, differing only in `memory_strategy`.

**Needs:** Check `err_over_bound` value for this row; re-run with `--bounds-check` to
see if the binary itself reports a violation; compare max_abs_err between pfpl and
pfpl_minimal at 2 GB to see if they differ.

**First seen:** fzgm_scaling_v2, pfpl SCALING-SYNTH/synth_2GB, 2026-07-02

---

## E7 — cuSZp adapter: decompression speed line never matched (100% failure)

**Status:** Fixed (2026-07-02)

**Symptom:**
```
No 'cuSZp decompression   end-to-end speed:' line found in cuSZp output.
Is this the cuSZp binary? Check the log.
```
Every cuszp2 and cuszp3 cell failed — compress succeeded (log shows the line printed
correctly) but the benchmark-phase parser raised on every single call.

**Affected pipelines:** cuszp2, cuszp3 (both versions share `_parse_speed` in
`benchkit/adapters/cuszp.py`)

**Root cause:**
`_parse_speed()` built its match key as `f"cuSZp {phase}   end-to-end speed:"` — three
literal spaces — for *both* phases. cuSZp's own stdout pads only `compression` with three
spaces so its printed table lines up with the longer word `decompression`, which gets a
single space:
```
cuSZp compression   end-to-end speed: 52.088753 GB/s
cuSZp decompression end-to-end speed: 48.519325 GB/s
```
The hardcoded three-space key matched the compression line but never the decompression
line, so 100% of cuszp2/cuszp3 cells failed at the benchmark step (compress-only smoke
via `compress()`/`decompress()` was unaffected since those don't call `_parse_speed`).

**Fix:** Changed the match to `re.compile(rf"cuSZp\s+{phase}\s+end-to-end speed:\s*...")`
— whitespace-tolerant regex instead of a literal fixed-width key.

**Verification:** Re-ran `smoke-m3-refs.yaml` under a fresh session id after the fix —
20/20 cells passed (previously 12/20, with all 8 failures on cuszp2/cuszp3). CR/PSNR
identical to the first (failed-at-benchmark) run, confirming the fix only touched timing
parsing, not correctness.

**First seen:** smoke-m3-refs, cuszp2/cuszp3 all cells, 2026-07-02

---

## E8 — fzgpu (reference): deterministic eb violation on CESM/CLDHGH

**Status:** Open — appears to be a real property of the FZ-GPU reference tool, not a
harness bug

**Symptom:**
`eb_satisfied: false`, `err_over_bound: 1.00183` (0.18% over the nominal bound) on the
`fzgpu` reference adapter (the original FZ-GPU tool, not the FZGM port) for
CESM-2D/CLDHGH at eb=0.001, rel_range. All other fields (HURR, NYX, HACC) pass cleanly
at the same bound.

**Affected pipelines:** `fzgpu` reference adapter only (`compressor: fzgpu`, not
`fzgm:fzgpu`)

**Root cause (status):** Not yet root-caused. Ruled out as a timing/measurement
artifact — `err_over_bound` is bit-identical (`1.0018329522656289`) across two
independent re-runs (fresh sessions, separate subprocess calls), so this is
deterministic given the same input/eb, not sampling noise. The first run also showed
huge `compress_cv` (3.89) on this same cell, which looked suspicious, but the re-run
came back `compress_stable: true` (cv 0.054) with the identical eb violation — so the
timing instability and the eb violation are unrelated; the instability was transient
wall-clock jitter (expected per `docs/adapters/fzgpu.md`, small 25.9 MB field is most
exposed to CPU-scheduling noise), while the eb violation is a fixed property of this
tool+field+bound combination.

**Why this matters for fairness:** if FZ-GPU's actual max error can exceed its own
nominal bound by a small margin while other reference tools (pfpl, cuszhi, cuszp2/3,
cusz) enforce it strictly, then CR comparisons at "the same eb" are not strictly
apples-to-apples for this field — FZ-GPU may be trading a small, undetected bound
overshoot for its reported ratio.

**Needs:** Investigate FZ-GPU's internal Lorenzo/bitshuffle quantization boundary
handling on CESM/CLDHGH specifically (small dynamic range 2-D field) — check whether the
tool's own NOA threshold computation rounds differently than the harness's independent
recomputation. A minimal repro: `fz-gpu <CLDHGH path> 3600 1800 1 0.001` and compare the
harness's independently computed max_abs_err against FZ-GPU's own reported error.

**First seen:** smoke-m3-refs, fzgpu CESM-2D/CLDHGH, 2026-07-02 (confirmed reproducible
on re-run same day)

---

## E9 — GInterp (cuSZ-Hi) requires ≥2 spatial dimensions

**Status:** Expected / by design — not a bug

**Symptom:**
```
GInterpStage::setDims: dims[1] must be > 1; 1-D input is not supported (got y=1)
```
`compress()` fails immediately for any field whose dims collapse to `[N, 1, 1]`.

**Affected pipelines:** `cusz_hi_tp.toml`, `cusz_hi_cr.toml` (both use the fused `GInterpStage`
predictor) on HACC/vx (1-D particle data).

**Root cause:** GInterp's multi-level spline-interpolation pyramid is inherently multi-dimensional
(`docs/stages/ginterp.md`); there is no 1-D fallback. This is a real constraint of the algorithm,
not a defect — cuSZ-Hi itself is a scientific-field compressor, not designed for 1-D particle data.

**Fix:** None needed. Exclude 1-D fields from cuSZ-Hi (native and FZGM) comparisons.

**First seen:** fzgm_vs_native, cusz_hi_tp/cusz_hi_cr HACC/vx, 2026-07-02

---

## E10 — FZGM cuSZ-Hi CR-mode pipeline crashes on every 2-D/3-D field

**Status:** Fixed (2026-07-02, FZGPUModules commit `8e581a8`)

**Symptom:**
```
[fzgmod] CUDA error at modules/coders/rre/rre_stage.cu:444 —
cudaMemcpyAsync(d_out + h_out_off[i], d_in + h_in_off[i], h_orig_sz[i],
cudaMemcpyDeviceToDevice, stream) → invalid argument
```
(CESM/CLDHGH — fails during `compress()`.) On HURR/TC and NYX/temperature the same stage fails
one line later, during `decompress()`/benchmark:
```
[fzgmod] CUDA error at modules/coders/rre/rre_stage.cu:448 —
cudaStreamSynchronize(stream) → an illegal memory access was encountered
```

**Affected pipelines:** `cusz_hi_cr.toml` only (GInterp → Huffman → Merge → RRE4 → Zigzag → RZE
chain). `cusz_hi_tp.toml` (Huffman-free) does not hit this.

**Root cause (hypothesis):** Not yet root-caused, but the failure is 100% reproducible across every
multi-dimensional field tried (3/3), always inside `RREStage`, always immediately after the `Merge`
stage concatenates `[Huffman(codes) | anchor | outlier_vals | outlier_idxs]` into one blob. Likely
an offset/size bookkeeping bug in how `RREStage` reads the merged blob's segment boundaries — the
Huffman-coded segment has a size that's only known at runtime (unlike the other three, fixed-size
segments), which is the one thing that differs from the (working) TP-mode pipeline's merge step.

**Why this matters:** blocks any FZGM cuSZ-Hi CR-mode comparison entirely — not a fidelity question,
the pipeline cannot complete a single run on real data.

**Needs:** Investigation in FZGPUModules at `modules/coders/rre/rre_stage.cu:444-448` — check how
`RREStage` computes segment offsets/sizes when the upstream `Merge` includes a variable-length
Huffman-coded segment. A minimal repro: `fzgmod-cli -c examples/presets/cusz_hi_cr.toml -i
<any CESM field> -l 3600x1800 -b`.

**First seen:** fzgm_vs_native, cusz_hi_cr CESM-2D/HURR/NYX, 2026-07-02

**Retest (2026-07-02, after upstream preset fix):** Skyler switched `cusz_hi_cr.toml`'s
`memory_strategy` from `PREALLOCATE` to `MINIMAL` and dropped the stale `input_size` hint
(FZGPUModules commits `79dc3ba`, `e30d0cc`). **Still broken on all 4 fields** — the crash
site moved (CESM now fails at `rre_stage.cu:448` during `benchmark()`, previously `:444`
during `compress()`; NYX now fails during `decompress()` instead of `benchmark()`) but the
underlying `RREStage` illegal-memory-access is unchanged. The memory-strategy switch did
not touch the actual bug — confirms this is the Merge→RRE offset bookkeeping issue
hypothesized above, not a PREALLOCATE-sizing artifact.

**Root cause (confirmed, FZGPUModules commit `8e581a8`):** Two independent bugs, both in the
FZGM LC-coder integration, not specific to the Huffman/variable-length-segment theory above:
1. `rre_stage.cu`/`rze_stage.cu` encode kernels wrote their final bitmap level to `s_out`
   without a trailing `__syncthreads()`, so the copy-out loop raced the last writers
   (compute-sanitizer flagged thousands of hazards). The inverse kernels already synced
   externally; the forward path didn't. This made the compressed blob non-deterministic and
   fed malformed chunks to the decoder.
2. `ginterp_stage.cu`'s forward pass left the tail of the `outlier_vals`/`outlier_idxs` ports
   uninitialized beyond the actual outlier count; `MergeStage` read those ports at full
   worst-case capacity mid-pipeline (before the end-of-pipeline size trim), including the
   stale tail — which on a *recycled* pool page (i.e. after a prior `decompress()`) held
   garbage, making repeated `-b` runs produce different merged blobs for identical input.

**Retest 2 (2026-07-02, later — after the actual fix):** All 3 multi-D fields (CESM, HURR,
NYX) now run cleanly through `-z`/`-x`/`-b` with no crash. CESM: CR 13.26 vs native 13.30
(matches almost exactly — down from completely broken); HURR: CR 41.18 vs native 41.21
(near-perfect match); NYX: CR 185.10 vs native 287.82 (a real ~36% gap remains — see new
finding E17 below, much smaller than "doesn't run at all" but not fully closed). HACC/vx
still fails — expected, see E9 (GInterp requires ≥2-D). CESM shows the same marginal eb
overshoot as E8/E13/E16 (see E16) — not a new issue.

---

## E11 — FZGM cuSZ-Hi TP-mode CR is 63–67% below native, one field also violates eb

**Status:** Fixed (2026-07-03, FZGPUModules commit `bb96edb`) — CR deficit closed to within
a few percent of native on all 3 fields; only the pre-existing E16 eb-overshoot pattern on
CESM remains, unrelated to this entry

**Symptom:** On every field where `cusz_hi_tp.toml` completes (CESM, HURR, NYX — HACC blocked by
E9), FZGM's CR is 63–67% lower than native `cuszhi -s tp` at the same eb:

| Field | native CR | fzgm CR | Δ |
|---|---|---|---|
| CESM/CLDHGH | 13.33 | 4.66 | −65.0% |
| HURR/TC | 39.55 | 12.89 | −67.4% |
| NYX/temperature | 209.75 | 77.02 | −63.3% |

CESM additionally shows `eb_satisfied: false`.

**Root cause:** Not yet root-caused. PSNR is *not* lower on the FZGM side (in fact slightly higher:
+0.27 to +1.36 dB) — ruling out "FZGM trades ratio for extra headroom" as the explanation. The gap
is too large and too consistent across 3 independent fields to be outlier-capacity noise; points at
the TP-mode lossless chain (Zigzag → Bitshuffle → RRE over the quantization codes, plus a separate
Merge → Bitshuffle → RRE → RZE chain over the outlier segment) under-compressing relative to
cuSZ-Hi's own TP backend.

**Needs:** Compare intermediate stage output sizes (`--profile`/`--print-pipeline`) between the FZGM
pipeline and native cuszhi's TP mode to isolate which stage is responsible for the size gap.

**First seen:** fzgm_vs_native, cusz_hi_tp CESM/HURR/NYX, 2026-07-02

**Retest (2026-07-02, after upstream preset fix):** Same `MINIMAL` + eb/radius fix as E10.
CESM is **bit-for-bit unchanged** (CR 4.66, PSNR 66.81, eb still violated — identical to
before, confirming this defect is independent of memory strategy). HURR and NYX, which
both *passed* before the preset change, now **crash outright**: HURR fails in `RREStage`
(`rre_stage.cu:448`, illegal memory access — the same error as E10's cuSZ-Hi CR pipeline,
despite TP mode's chain not sharing CR mode's Huffman/Merge structure); NYX fails in a
different stage, `RZEStage` (`rze_stage.cu:446`, invalid `cudaMemcpyAsync` argument). The
preset change traded "2/3 fields work with a large CR deficit" for "0/3 fields complete" —
a regression, not a fix. Likely `MINIMAL`'s on-demand allocation exposes a sizing bug in
the TP-mode outlier chain (Merge→Bitshuffle→RRE→RZE) that `PREALLOCATE`'s generous
worst-case buffers had been masking.

**Root cause (confirmed, FZGPUModules commit `8e581a8`):** Same two bugs as E10's final root
cause — the missing `__syncthreads()` in `rre_stage.cu`/`rze_stage.cu` encode kernels, and
`ginterp_stage.cu` leaving the outlier-port tail uninitialized across `-b` reps. Neither is
specific to `cusz_hi_cr`'s Huffman/Merge structure — both stages are shared by `cusz_hi_tp`'s
own Merge→Bitshuffle→RRE→RZE outlier chain, which is why the identical crash signature showed
up in both presets. A separate fix (`src/pipeline/dag.cpp`'s `CompressionDAG::reset()`) also
addressed a `MINIMAL`-mode buffer-size restore bug that under-allocated buffers on rep 2+.

**Retest 2 (2026-07-02, later — after the actual fix):** HURR and NYX no longer crash — both
complete `-z`/`-x`/`-b` cleanly, and NYX's PSNR is back to 75.75 dB (matching the original,
pre-regression pass exactly — the E15-adjacent silent corruption on this field is also gone).
**But the core CR deficit from the original finding is completely unchanged** — this fix
target the crashes/corruption, not compression ratio:

| Field | native CR | fzgm CR | Δ | status |
|---|---|---|---|---|
| CESM/CLDHGH | 13.33 | 4.66 | −65.0% | unchanged from original finding; `eb_satisfied: false` |
| HURR/TC | 39.55 | 12.88 | −67.4% | unchanged; now completes without crashing |
| NYX/temperature | 209.75 | 76.87 | −63.4% | unchanged; now completes without crashing/corruption |

So the CR deficit itself (this entry's original headline finding) is a genuinely separate,
still-open bug from the crash/corruption issues that shared its symptoms — needs the
stage-by-stage size comparison suggested below. HACC/vx still fails, expected (E9).

**Root cause (confirmed, FZGPUModules commit `bb96edb`):** The FZGM preset's `GInterp` stage
used `code_type = "uint16"` with `quant_radius = 32768` — but native cuSZ-Hi's actual spline
error-control (`ErrCtrlTrait<1>` in the vendored MVP code) emits **1-byte** codes in `[0, 256)`
(`context.h` default `dict_size = 256, radius = 128`), escaping any residual outside ±128 to
the outlier buffer. The FZGM preset's wide radius kept every residual inline as a 2-byte code
— literally twice the bytes per code, and higher-entropy per code too since nothing was
routed to outliers. Skyler verified this wasn't an auto-tuning gap (`auto_tuning = 3/4` adds
< 0.5% CR on top of the fix) or a lossless-coder gap (reimplementing `RREStage`'s bitshuffle
as native's exact `d_BIT_1` gave zero change) — it was purely the code width/radius mismatch.
Fix: `code_type = "uint8"`, `quant_radius = 128`, matching native exactly.

**Retest 3 (2026-07-03, after the fix):** CR deficit closed to within a few percent on all 3
fields — NYX is now actually *ahead* of native:

| Field | native CR | fzgm CR | Δ |
|---|---|---|---|
| CESM/CLDHGH | 13.33 | 12.47 | −6.5% |
| HURR/TC | 39.55 | 38.97 | −1.5% |
| NYX/temperature | 209.75 | 210.33 | +0.3% |

PSNR unchanged (66.81/68.54/75.75 dB) — quality wasn't traded away for the CR gain, confirming
this was a genuine efficiency bug, not a bound-tightness difference. CESM still shows
`eb_satisfied: false` with the identical E16 `err_over_bound` (`1.0018662878045856`) — that's
the cross-tool pattern tracked separately, not a regression from this fix. **Consider this
entry closed** — HACC remains excluded (1-D unsupported, E9).

---

## E12 — 2-D-only FZGM presets silently mismatched against 1-D HACC data

**Status:** Expected once understood — an experiment-design gap, not a library bug

**Symptom:** `fzgm:cuszp3` on HACC/vx gives `CR = 0.69` — the "compressed" output is *larger* than
the original.

**Affected pipelines:** `cuszp3.toml` (explicitly documented in its own header as "cuSZp3 (plain,
**2-D**)", uses `TiledLorenzo` with `tile_x=8, tile_y=8`).

**Root cause:** HACC's dims collapse to `[N, 1, 1]` (confirmed via the GInterp error message in E9:
"got y=1"). Feeding that into an 8×8 2-D tiling scheme produces degenerate 8×1 tiles — the predictor
gets essentially no spatial correlation to exploit, and `AdaptiveBitpack`'s per-tile overhead
(rate byte + sign bitmap) is paid on tiny, poorly-predicted tiles, netting expansion. Native cuSZp3's
`plain` mode (no `-d` flag given) takes a genuinely 1-D code path internally, which is why it stays
healthy (CR 5.18) on the same data — the two sides were not running comparable algorithms for this
field.

**Fix:** Not a bug to fix — a comparison-design issue. Exclude 1-D fields from `cuszp3.toml` /
`cusz_hi_tp.toml` / `cusz_hi_cr.toml` pairings until a 1-D-specific FZGM preset exists for each.

**First seen:** fzgm_vs_native, cuszp3 HACC/vx, 2026-07-02

---

## E13 — FZGM pfpl.toml runs 22–45% below native PFPL's CR at matched quality

**Status:** Fixed (2026-07-02, FZGPUModules commit `79dc3ba`, `Bitshuffle element_width` 2→4)

**Symptom:** Across all 4 fields, FZGM's CR trails native PFPL substantially at identical PSNR
(matched to 2 decimal places):

| Field | native CR | fzgm CR | Δ |
|---|---|---|---|
| CESM/CLDHGH | 11.19 | 6.47 | −42.2% |
| HURR/TC | 15.58 | 10.35 | −33.6% |
| NYX/temperature | 48.32 | 35.30 | −27.0% |
| HACC/vx | 5.68 | 3.14 | −44.7% |

CESM additionally shows a marginal eb overshoot (`err_over_bound ≈ 1.0019`), the same pattern as E8.

**Root cause:** Not yet root-caused. The two implementations are structurally different — FZGM's
`pfpl.toml` is Quantizer → Difference → Bitshuffle → RZE, while native PFPL uses its own
LC-framework backend — so some gap between "faithful port" and "same algorithm, different
lossless back-end" is expected, but a consistent 27–45% CR deficit at matched quality is large
enough to warrant checking whether the FZGM port is actually equivalent or is leaving compression
on the table in one of its stages.

**Needs:** Compare per-stage output sizes between the FZGM chain and PFPL's own backend
(`--profile`) to find which stage accounts for the gap.

**First seen:** fzgm_vs_native, pfpl all 4 fields, 2026-07-02

**Retest (2026-07-02, after upstream preset fix):** Skyler changed `pfpl.toml`'s
`Bitshuffle` stage from `element_width = 2` to `element_width = 4` (FZGPUModules commit
`79dc3ba`). **CR deficit is resolved on 3/4 fields** — CESM now matches native exactly
(11.19 vs 11.19, was 6.47), HURR slightly exceeds native (15.68 vs 15.58, was 10.35), HACC
matches exactly (5.68 vs 5.68, was 3.14). This confirms the hypothesis: the old
`element_width=2` Bitshuffle grouping was leaving compression on the table relative to
PFPL's native backend. **But NYX/temperature now decodes to garbage** — see new entry E15,
a correctness regression introduced by this same change, more serious than the CR gap it
fixes. Do not adopt `element_width=4` project-wide until E15 is understood.

**Retest 2 (2026-07-02, later — after E15's root cause was fixed, commit `8e581a8`):** NYX now
also matches native (48.04 vs 48.32, CR gap −0.6%, PSNR 63.89 dB — correct, no more
corruption). **All 4 fields now confirmed fixed:**

| Field | native CR | fzgm CR | Δ |
|---|---|---|---|
| CESM/CLDHGH | 11.19 | 11.19 | 0.0% |
| HURR/TC | 15.58 | 15.68 | +0.6% |
| NYX/temperature | 48.32 | 48.04 | −0.6% |
| HACC/vx | 5.68 | 5.68 | 0.0% |

CESM still shows the marginal eb overshoot shared with E8/E11/E16 — see E16, not specific to
this pipeline. Consider this entry closed.

---

## E14 — Native cuSZp2 (outlier mode) violates its own error bound on HURR/TC

**Status:** Open — native tool finding, independent of FZGM

**Symptom:** `cuszp2 -m outlier` on HURR/TC gives `PSNR = 50.93 dB` and `eb_satisfied: false`, vs.
64–68 dB and passing on every other field/tool combination tested this session (including FZGM's
`cuszp2.toml` port on the *same* field, which passes at PSNR 64.79 dB).

**Affected pipelines:** native `cuszp2`, `-m outlier` mode only (v2 reference binary).

**Root cause:** Not investigated. Since FZGM's port of the same algorithm passes cleanly on this
exact field/eb, this looks like a genuine quirk or bug in the native cuSZp2 binary's outlier-mode
handling on 3-D data, not a comparison artifact.

**Needs:** Reproduce directly (`cuSZp -i <HURR/TC path> -t f32 -m outlier -eb rel 0.001`) and check
whether it's specific to this field's outlier density or a broader 3-D-outlier-mode issue.

**First seen:** fzgm_vs_native, cuszp2 HURR/TC, 2026-07-02

---

## E15 — FZGM pfpl.toml decodes NYX/temperature to garbage after Bitshuffle element_width fix

**Status:** Fixed (2026-07-02, FZGPUModules commit `8e581a8`)

**Symptom:** After changing `pfpl.toml`'s `Bitshuffle` stage to `element_width = 4` (the change
that fixed E13's CR gap on 3/4 fields), NYX/temperature compresses and decompresses without a
CUDA error or non-zero exit — but the decompressed array is garbage:
```
psnr           = -84.64 dB      (everything else this session: 63.9–75.8 dB)
nrmse          = 17067.35       (everything else: ~0.0006)
max_abs_err    = 20531243901466.38   (eb_abs_effective was 4780.30 — 4.3 billion× over)
```
`cr` and `compressed_bytes` look completely normal (48.04, matching native's 48.32), and the run
exits 0 with `status: ok` — there is no crash to signal that anything is wrong, only the quality
numbers reveal it.

**Affected pipelines:** `pfpl.toml` with `Bitshuffle element_width = 4`, NYX/temperature only.
CESM/CLDHGH, HURR/TC, and HACC/vx all decode correctly with the same preset.

**Root cause (hypothesis):** Not investigated. NYX/temperature is a 512×512×512 f32 volume
(134,217,728 elements); the other three fields have different sizes. `element_width = 4` groups
the `Difference` stage's `uint32` output into 4-byte-wide shuffle planes — a size/alignment edge
case specific to NYX's element count interacting with `block_size = 16384` is the most likely
culprit (e.g. a not-evenly-divisible last block handled incorrectly at width 4 but tolerated at
width 2), but this needs to be confirmed against the `BitshuffleStage` source, not guessed from
the symptom.

**Why this matters:** this is worse than the CR gap it replaced — a silent, non-crashing
correctness failure that only shows up by checking quality metrics. If harness-owned quality
checks (`eb_satisfied`, `psnr`) were ever skipped or thresholds loosened, this would ship as a
passing row. Reinforces why the harness independently recomputes quality from raw bytes (D4)
rather than trusting a tool's own exit code.

**Needs:** Investigate `BitshuffleStage` (or wherever `element_width` changes shuffle-plane
sizing) in FZGPUModules for a size/alignment bug at `element_width=4` specific to large,
evenly-sized-but-not-block-aligned inputs. A minimal repro: `fzgmod-cli -c
examples/presets/pfpl.toml -i <NYX/temperature path> -l 512x512x512 -m noa -e 1e-3 -z -x
--compare <path>`. Do not roll `element_width=4` out further (e.g. to other presets) until
this is fixed — treat E13's 3/4 fix as provisional.

**First seen:** fzgm_vs_native retest, pfpl NYX/temperature, 2026-07-02

**Root cause (confirmed, FZGPUModules commit `8e581a8`):** Not a `Bitshuffle`/`element_width`
bug as hypothesized — the actual cause was the missing `__syncthreads()` in `rze_stage.cu`'s
encode kernel (see E10's confirmed root cause; `pfpl.toml`'s final stage is `RZE`, the same
stage implicated in E10/E11's cuSZ-Hi crashes). The race made the encoded blob
non-deterministic specifically on data that exercises the compressible path heavily — NYX's
highly-regular temperature field — while smaller/less-regular fields happened not to trigger
it. `Bitshuffle element_width` was a red herring; correlation (the corruption appeared right
after that preset change) wasn't causation (the actual bug was already latent in `RZEStage`,
just not triggered until the input to it changed enough to expose the race).

**Retest 2 (2026-07-02, later — after the fix):** NYX/temperature decodes correctly, PSNR
63.89 dB (matches native exactly), CR 48.04 vs native 48.32 (−0.6%, within noise). Confirmed
fixed — see E13's retest 2 for the full 4-field table. Consider this entry closed.

---

## E16 — Marginal eb overshoot on CESM/CLDHGH is shared by every tool tested, not a per-tool bug

**Status:** Open — likely harness-side or field-specific, low priority (0.19% overshoot)

**Symptom:** `eb_satisfied: false` on CESM/CLDHGH at eb=1e-3 rel_range, with
`err_over_bound = 1.0018662878045856` (a 0.19% overshoot of the nominal bound) — and this
exact value, to 16 significant figures, has now been observed independently on:
- native `fzgpu` (E8)
- FZGM `pfpl.toml` (E13, before and after its fix)
- FZGM `cusz_hi_tp.toml` (E11, both before and after the crash fix)
- FZGM `cusz_hi_cr.toml` (E10, after the crash fix — first time this pipeline could even reach
  a quality check on this field)

**Why this matters:** four independently-implemented tools/pipelines, sharing no code path
apart from this being the same input field, all reproduce the identical overshoot ratio. That
rules out "bug in tool X's quantizer" as the explanation (already weak given how many
different quantizer implementations are involved) — the common factor is CESM/CLDHGH's actual
data plus how the *harness* computes `eb_abs_effective` (`= eb × (max − min)` of the original
array) and compares it against the realized `max_abs_err`. Two live hypotheses, neither
confirmed:
1. The harness computes `range = max − min` in a precision (or via a code path) that differs
   subtly from what each tool's own internal range computation produces, so `eb_abs_effective`
   ends up a hair tighter than what every tool is actually targeting — i.e. every tool is
   correct relative to its own bound, and the harness's cross-tool `eb_abs_effective` is very
   slightly wrong for this specific field.
2. CLDHGH's specific value distribution (cloud fraction data, likely a narrow range with many
   values very close to the range boundary) puts the true quantization error right at the
   rounding edge for any reasonable uniform quantizer at this exact eb, so a ~0.19% overshoot
   is a genuine, expected property of the data/bound combination, not a bug anywhere.

**Needs:** Compute CESM/CLDHGH's `min`/`max` independently (e.g. via `numpy.float64` on the
raw array) and compare bit-for-bit against what the harness uses in `eb_abs_effective` — if
they differ at all, that's hypothesis 1 confirmed and a harness-side precision fix. If they
match exactly, it's hypothesis 2, and the fix (if any) is to loosen `eb_tol` for this
class of near-boundary case rather than chase a bug that isn't there.

**First seen (as a distinct, cross-tool pattern):** fzgm_vs_native retest 2, 2026-07-02 —
individual instances go back to E8 (2026-07-02, same day, earlier).

---

## E17 — FZGM cuSZ-Hi CR-mode CR is ~36% below native on NYX/temperature (post-crash-fix)

**Status:** Open — real gap, not yet root-caused; much smaller than E10's original finding

**Symptom:** After FZGPUModules commit `8e581a8` fixed E10's crash, `cusz_hi_cr.toml` now
completes on NYX/temperature but with a real CR gap: 185.10 vs native's 287.82 (−35.7%), while
CESM and HURR are within noise of native (see E10's retest 2 table). PSNR is correct (75.75 dB,
matches native's 74.39 dB) — this is a compression-efficiency gap, not a correctness issue.

**Root cause:** Not investigated. Given E11's CESM/HURR/NYX CR deficits in TP mode remain
open and unrelated to the crash fixes, and this is the CR-mode pipeline's own Huffman+Merge
chain, this may be a related-but-distinct instance of the same class of issue as E11 (the
lossless back-end chain leaving compression on the table relative to native cuSZ-Hi's own
implementation) — but on this pipeline, only manifesting significantly on NYX's highly
compressible data (CESM/HURR are close to native, suggesting the gap scales with how much the
Huffman stage's codebook or the RRE/RZE chain has to work).

**Needs:** Compare per-stage output sizes (`--profile`) between this pipeline and native
cuszhi's CR-mode backend specifically on NYX/temperature to isolate which stage loses ground
on highly compressible data.

**Update (2026-07-03):** E11's analogous TP-mode deficit turned out to be a `GInterp`
`code_type`/`quant_radius` mismatch (uint16/32768 in the FZGM preset vs. native's actual
uint8/128 spline error-control) — fixed in commit `bb96edb`, closing that gap to within a few
percent. **`cusz_hi_cr.toml` still uses `quant_radius = 2048` with `code_type = "uint16"`**
(see its own header comment: `quant_radius` explicit "so Huffman bklen can match") — this
preset was not touched by the TP-mode fix and this entry's gap remains open. Given CR mode
runs `GInterp` through a `Huffman` stage rather than direct Bitshuffle/RRE, the fix may not
transfer directly (a `code_type=uint8` change here would also need `Huffman`'s `bklen` and
CESM/HURR's already-matching numbers re-verified), but the same root-cause family (code
width/radius not matching native's actual spline error-control) is the first thing to check.

**First seen:** fzgm_vs_native retest 2, cusz_hi_cr NYX/temperature, 2026-07-02
