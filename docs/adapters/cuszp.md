# cuSZp adapter contract (v2 and v3)

**Tool:** cuSZp (GPU error-bounded lossy compressor, fixed-length encoding)  
**Adapter:** `benchkit/adapters/cuszp.py` · `CuszpAdapter`  
**Compressor keys in experiments:** `cuszp2`, `cuszp3`

---

## CLI

**v2:**
```bash
cuSZp -i <input> -t f32|f64 -m plain|outlier \
      -eb abs|rel <bound> [-x compressed] [-o decompressed]
```

**v3 adds `-d`:**
```bash
cuSZp -i <input> -t f32|f64 -m plain|outlier|fixed \
      -d 1|2|3 [dz dy dx] \
      -eb abs|rel <bound> [-x compressed] [-o decompressed]
```

Set `CUSZP2_CLI` / `CUSZP3_CLI` to the respective binary paths, or pass
`cli_path` in the run entry. The two versions have separate binaries.

---

## Error-mode semantics

| Canonical | Native (`-eb`) | eb basis |
|---|---|---|
| `abs` | `abs <eb>` | `eb` |
| `rel_range` | `rel <eb>` | `eb × (max − min)` |

cuSZp's `rel` is **range-relative**: it internally computes `max−min` and
multiplies the error bound — same semantics as cuSZ's `r2r` and FZGM's
`NOA`. Use `rel_range` for cross-tool comparisons.

`rel_maxabs` is not supported (raises `AdapterError`).

---

## Pipeline string

The `pipeline:` field in an experiment run entry encodes the encoding mode
and (for v3) the processing dimension:

| Pipeline string | Encoding mode | Dim processing |
|---|---|---|
| `plain` | fixed-length plain | v2: implicit 1D; v3: `−d 1` |
| `outlier` | fixed-length with outlier | same |
| `fixed` | no-delta fixed-length | v3 only |
| `plain:2d` | plain | v3: `−d 2 1 dim_y dim_x` |
| `plain:3d` | plain | v3: `−d 3 dim_z dim_y dim_x` |
| `outlier:3d` | outlier | v3: `−d 3 dim_z dim_y dim_x` |

Dims for 2D/3D processing come from the `FieldSpec.dims` in fast-to-slow
order: `dims[0]` is fastest (x), `dims[1]` is y, `dims[2]` is z.
cuSZp v3 takes `dz dy dx` in slow-to-fast order.

**Default:** 1D processing. Use 2D/3D for better CR on structured data,
but verify the dim order matches the actual layout of each dataset.

---

## Round-trip model

cuSZp does **compress + decompress in a single invocation**. There is no
separate decompress-only binary.

- `compress()`: runs the full round-trip, saves `-x c.cuszp -o d.bin`.
- `decompress()`: returns the already-written `d.bin`. No subprocess.
- `benchmark()`: runs N invocations **without `-x`/`-o`** (cuSZp skips
  file writes when these are omitted — no disk I/O overhead per call).

---

## Timing

- **Method:** CUDA events (`cudaEventElapsedTime`), via `TimingGPU`.
- **Per invocation:** 10 GPU warmup iterations + 1 timed compress + 1 timed
  decompress. Single warm run ≈ `*_device_ms_min` behavior.
- **Printed output:** `cuSZp compression   end-to-end speed: X GB/s` where
  X is actually **MiB/ms** (mislabeled). The adapter recovers device_ms:
  ```
  device_ms = (original_bytes / 1024²) / X
  ```
- **n_runs subprocesses** each yield one timing value per phase. Set
  `warmup_reps: 0` in experiments (each run is already warm), or `1` to
  discard any clock-startup effects across subprocess boundaries.

---

## Build

```bash
# v2
cd ~/research/compressors/cuSZp-V2.0.1
cmake -S . -B build \
  -DCMAKE_CUDA_ARCHITECTURES="80" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_COMPILER=$(which g++) \
  -DCMAKE_C_COMPILER=$(which gcc)
cmake --build build -j8
# binary: build/examples/bin/cuSZp

# v3
cd ~/research/compressors/cuSZp-V3.0.0
# (same cmake flags as v2)
cmake --build build -j8
# binary: build/examples/bin/cuSZp
```

---

## Known quirks

- **Mislabeled throughput:** the binary prints `GB/s` but computes `MiB/ms`.
  Mixing this printed value with other tools introduces a ~7% error. The
  adapter always recovers raw ms and lets the harness compute throughput.
- **No decompress-only mode:** cuSZp always does the full round-trip. The
  adapter caches the decompressed file from compress() rather than re-running.
- **Single timed rep per subprocess:** unlike FZGM/PFPL, each cuSZp call
  yields one timing value. Set `repetitions: 5–10` in experiments.
- **1D default for v3:** multi-dim processing may improve CR but requires
  verifying the dim order convention per dataset.

---

## FZGM pairing: preset ↔ native pipeline string

FZGM has one preset per (algorithm, mode, dimensionality) triple; there is no
single preset that's correct for every field, since `TiledLorenzoStage`'s tile
shape is dimensionality-specific (`docs/stages/tiled_lorenzo.md`, FZGM repo).
Match native `pipeline:` strings to FZGM presets like this:

| Native `pipeline:` | Applies to | FZGM preset | Notes |
|---|---|---|---|
| `plain` (cuszp2) | all fields | `cuszp2_plain.toml` | `Lorenzo(block=32)`, no outlier |
| `outlier` (cuszp2) | all fields | `cuszp2.toml` | `Lorenzo(block=32)` + outlier |
| `plain:2d` (cuszp3) | CESM-2D | `cuszp3.toml` | `TiledLorenzo(8x8)`, no outlier |
| `outlier:2d` (cuszp3) | CESM-2D | `cuszp3_outlier.toml` | `TiledLorenzo(8x8)` + outlier |
| `plain:3d` (cuszp3) | HURR, NYX | `cuszp3_3d.toml` | `TiledLorenzo(4x4x4)`, no outlier |
| `outlier:3d` (cuszp3) | HURR, NYX | `cuszp3_3d_outlier.toml` | `TiledLorenzo(4x4x4)` + outlier |
| `plain` (cuszp3) | HACC | `cuszp3_1d.toml` | `Lorenzo(block=32)`, same stage chain as cuszp2 |
| `outlier` (cuszp3) | HACC | `cuszp3_1d_outlier.toml` | `Lorenzo(block=32)` + outlier |

cuSZp3 is dimension-matched end to end: `configs/experiments/fzgm_vs_native.yaml`
scopes each native/FZGM pair to the fields whose true dimensionality they're
built for, via `only_datasets` (D17, `docs/DESIGN.md`).

**Why cuszp3 needs a 1-D preset (E12, fixed):** `cuszp3.toml`'s `TiledLorenzo(8x8)`
is a 2-D preset; feeding it 1-D data (e.g. HACC, dims `[N,1,1]`) collapses tiles
to degenerate 8×1 shapes and badly inflates the output (observed: CR 0.69, i.e.
the file *grows*). FZGM's own docs are explicit that 1-D data should go through
plain `LorenzoStage(block=32)` instead — identical to cuszp2's stage chain,
since cuSZp3's 1-D delta *is* cuSZp2's delta.

**Why cuszp3 needs a 3-D preset too (E12 follow-up, fixed):** the 2-D `8x8` tile
also isn't the right shape for genuinely 3-D fields (HURR, NYX) — `tile_z=1`
degrades to per-z-slice 2-D tiling rather than real 3-D locality, and pairing it
against native's *default* (`-d 1`, fully flattened) pipeline string wasn't
apples-to-apples on the native side either. Both sides now use FZGM's documented
3-D default (`tile_x=tile_y=tile_z=4`) and native's `-d 3 dz dy dx`.

---

## Known issue (fixed, 2026-07-19): native cuSZp2/cuSZp3 throughput was ~20-100x
## understated on the JetStream2 H100 — not a GPU or hardware problem

The first full `fzgm_vs_native.yaml` run on JetStream2 (BigRed200 A100 comparison
artifact vs. this run) showed native cuSZp2/cuSZp3 compress/decompress throughput
around **7-30 GB/s** — dramatically below the A100 numbers for the *identical*
cells (e.g. CESM/CLDHGH plain eb=1e-3: A100 58.3/60.0 GB/s vs. JetStream2 H100
8.0/7.8 GB/s), while every other native tool (cuSZ, cuSZ-Hi, FZ-GPU, PFPL) showed
H100 numbers matching or beating A100 as expected. Root-caused, not a
clock-locking, sm_90-build, or platform issue (all ruled out empirically —
see below) — **`cuSZp_compress`/`cuSZp_decompress` call `cudaMalloc`+`cudaFree`
for 3 small scratch buffers on every single invocation**
(`cuSZp_entry_{f32,f64}.cu` in cuSZp-V2, `cuSZp_entry_{1D,2D,3D}_{f32,f64}.cu` in
cuSZp-V3). That's cheap on bare-metal GPUs but on this GPU-passthrough cloud VM,
`cudaMalloc`/`cudaFree` latency is large and highly variable — `nsys` profiling
showed a single `cudaMalloc` call taking up to **451 ms**, while the actual
compression kernel (also measured via `nsys`, hardware timestamps) ran in
**~145 μs**. The self-reported "GB/s" folds the allocator's latency into the
timed region, dwarfing the real kernel time — worst for cuSZp specifically
because it's the *fastest* compressor in the comparison (sub-millisecond
kernels), so a large fixed per-call tax that's noise for cuSZ/PFPL (tens of ms
per call) completely dominates cuSZp's numbers.

**Ruled out before finding the real cause** (in this order): GPU clock locking
(reproduced with clocks fully unlocked); missing native `sm_90` SASS (both
cuSZp-V2 and cuSZp-V3 hardcode `set(CMAKE_CUDA_ARCHITECTURES 60 61 62 70 75 80
86)` in their CMakeLists.txt — **silently overriding** any `-DCMAKE_CUDA_
ARCHITECTURES=90` passed on the command line, since it's a plain `set()` not
`set(... CACHE ...)`; confirmed via `cuobjdump -lelf` no sm_90 cubin was
present, but rebuilding with `90` appended made no measurable difference on its
own — ruling out the missing-native-code theory too); a generic per-kernel-
launch dispatch-latency artifact (disproven by batching 100 kernel launches
between one CUDA-event pair — no improvement, meaning the cost recurs *inside*
each `cuSZp_compress()` call, not around the timer).

**Fix applied** (both `cuSZp-V2.0.1` and `cuSZp-V3.0.0`, in `~/compressors/`,
not this repo):
1. Added `90` to `CMAKE_CUDA_ARCHITECTURES` in both CMakeLists.txt (now builds
   real sm_90 SASS, confirmed via `cuobjdump`).
2. Replaced the per-call `cudaMalloc`+`cudaFree` scratch buffers with a
   grow-only cache (`CuszpScratch`/`ensure(cmpOffSize)`) — one static instance
   per compress/decompress/plain/outlier/fixed function, allocated once on
   first call and reused thereafter (still `cudaMemset` to 0 every call, so
   correctness is unaffected; verified `[Pass error check!]` on every test
   field). V3 has 36 such call sites across 6 files (1D/2D/3D × f32/f64) —
   fixed with a scripted transformation, spot-checked before building.
3. Batched the example driver's (`examples/cuSZp.cpp`) single-shot
   `cudaEventElapsedTime` timing into 100 back-to-back launches per event pair
   (`TIMING_REPEATS`), for a smoother average — not the root-cause fix (the
   scratch-buffer cache is), but kept since it's a straightforward
   robustness improvement matching the `--repeat` pattern already used to fix
   cuSZ/FZ-GPU's cold-start timing on BigRed200 (see `cusz.md`, `fzgpu.md`).

**Result**, same CESM/CLDHGH plain eb=1e-3 cell, through the real benchkit
adapter: **8.0/7.8 GB/s → 159.6/207.1 GB/s** (cuSZp2); NYX/temperature 3-D:
**~124/131 GB/s → 653/1205 GB/s** (cuSZp3) — now solidly in H100-class range
and consistent with (mostly exceeding) the BigRed200 A100 numbers for the same
cells. All `eb_ok=True`, CR/PSNR unchanged (deterministic, timing-independent).

If you rebuild cuSZp-V2/V3 from a fresh clone, re-apply this fix (source
patches are in the machine's `~/compressors/{cuSZp-V2.0.1,cuSZp-V3.0.0}/src/`
and `examples/cuSZp.cpp`, not upstreamed) — a naive rebuild will silently
reintroduce ~20-100x understated throughput without any error or warning.

## Known issue (fixed, 2026-07-19): native cuSZp2 (not v3) corrupted output
## on this H100 for ~11/24 cells — an uninitialized-shared-memory bug in the
## decoupled look-back scan, already fixed upstream in v3 but not backported to v2

While building an A100-vs-H100 performance comparison, CR/PSNR were expected
to match exactly between platforms (deterministic algorithm, identical
input/eb). They mostly do — except for **native cuSZp2** (both `plain` and
`outlier` variants; **cuSZp3 was unaffected**), where 11 of 24 native-cuSZp2
cells on this H100 showed severely corrupted PSNR relative to the BigRed200
A100 run of the identical cell (e.g. HACC/vx outlier eb=1e-4: A100 84.8dB →
H100 35.6dB; CESM/CLDHGH outlier eb=1e-4: A100 84.8dB → H100 6.9dB). cuSZp2's
own internal correctness self-check (`examples/cuSZp.cpp`, the "Fail error
check!" line) caught it and printed e.g. `Exceeding data count: 34340` —
not a harness-side miss, cuSZp2 flagged its own output as wrong.

### Root cause: `excl_sum` read uninitialized for block/warp 0

Each of the 4 kernel variants (`compress`/`decompress` × `plain`/`outlier`,
duplicated again for f64 — 8 functions total in `cuSZp_kernels_{f32,f64}.cu`)
computes a GPU-wide compressed-byte-offset prefix sum via a single-pass
"decoupled look-back" scan (one CUDA block == one warp == 32 threads here;
`cmp_tblock_size = 32`). Each block's final byte offset is
`excl_sum + rate_ofs`, where `excl_sum` (`__shared__ unsigned int`) is meant
to hold that block's *exclusive* prefix — the sum of every preceding block's
contribution.

For block/warp `w > 0`, `excl_sum` is computed by walking backward through
`flag[]`/`locOffset[]`/`cmpOffset[]` until an already-published value is
found. **For block/warp 0 — which by definition has no predecessor and
should just use `excl_sum = 0`, exactly like `cmpOffset[0]` implicitly does
via the pre-kernel `cudaMemset` — no code path ever assigns `excl_sum` at
all.** It's read uninitialized at `base_idx = excl_sum + rate_ofs`, which
every block (including block 0) executes unconditionally. Reading an
uninitialized `__shared__` variable is undefined behavior: its value is
whatever physically remained in that SM's shared-memory bank, which is not
part of the CUDA spec and is free to differ across GPU architectures, driver
versions, or even scheduling-dependent residue from a *different* kernel
that happened to reuse the same physical bank. That fully explains the
symptom profile: wrong from element 0 of the affected file, bit-for-bit
reproducible on a given piece of hardware/driver, absent on A100, and
present on H100.

**Confirmed by direct instrumentation**, not just code reading: adding a
one-off block-index counter to the correctness-check loop in
`examples/cuSZp.cpp` (temporary, reverted after use) showed the corruption
for HACC/vx outlier eb=1e-4 was confined to exactly **2 of 8575 blocks**
(block 0 and block 5) — not scattered across the file. Block 0's own
(garbage) `excl_sum` sends its compressed bytes to a essentially-random
offset in `cmpData`, corrupting both its own decode *and*, by landing in the
wrong place, whatever legitimate block's byte range it happened to
overwrite (block 5, in this case) — a two-block blast radius from one
uninitialized read, matching the earlier ruled-out theories' inability to
explain a scattered/grid-size-proportional pattern.

**Also ruled out along the way (empirically, not just by inspection):**
this session's own scratch-buffer-cache and `TIMING_REPEATS=100` changes
(reverted each independently, bug persisted bit-for-bit both times — see
git history of this file for the original writeup), and int32 overflow in
the quantization arithmetic (HACC/vx's actual worst-case quantization code
is ~2547, six orders of magnitude below `INT32_MAX`).

### The fix

```c
if(warp==0)
{
    excl_sum = 0;   // <-- added; warp 0 has no predecessor
    flag[0] = 2;
    __threadfence();
    flag[1] = 1;
    __threadfence();
}
```

One line, added to the pre-existing `warp==0` special case in all 8 kernel
functions (`cuSZp_compress_kernel_{plain,outlier}_{f32,f64}`,
`cuSZp_decompress_kernel_{plain,outlier}_{f32,f64}`). The existing
`__syncthreads()` immediately after this block already makes the write
visible to lane 0 before the later unconditional read — no additional
synchronization needed. **cuSZp-V3 already has this exact fix** (verified:
`grep -n "excl_sum = 0;" cuSZp-V3.0.0/src/cuSZp_kernels_*.cu` finds it in
the same `warp==0` branch of every kernel) — this is a one-line backport
from v3 to v2, not a novel fix. Worth upstreaming.

**Verified:** all 24 previously-checked native-cuSZp2 cells (4 datasets ×
2 modes × 3 error bounds) now pass cuSZp2's own internal error check with
zero failures, on a clean rebuild with no other changes. Both `cuSZp_test_f32`
and `cuSZp_test_f64` self-tests still pass (f64 isn't in this repo's actual
benchmark matrix, so this is a smoke check, not full sweep coverage, but the
fix is textually identical to the verified f32 case).

**Scope:** native cuSZp2 only. FZGM's independent port of the same
algorithm (`fzgm:cuszp2_plain`/`fzgm:cuszp2_outlier`, run on the exact same
cells) never reproduced this — matched A100 PSNR to 4-5 decimal places on
every cell even before this fix, since it's an independent reimplementation
that doesn't share cuSZp2's kernel code. cuSZp3 was also already clean
(it has its own, already-correct `excl_sum = 0`).

If you rebuild cuSZp-V2 from a fresh clone, re-apply this fix too (same
caveat as the D20 throughput fix above — not upstreamed, source patch lives
only in `~/compressors/cuSZp-V2.0.1/src/cuSZp_kernels_{f32,f64}.cu` on this
machine).
