# Peak device memory: how the numbers were obtained and compared

Provenance record for the FZGM-vs-native peak-memory comparison (paper `RQ2`
memory subsection; see `docs/DESIGN.md` D46 for the conclusion this supports).
This is a **standalone measurement**, run outside benchkit's adapter/session
machinery — the tool and field/CLI details below are exactly what a benchkit
adapter integration would need to formalize, if that becomes worth doing.

## What is being compared

For each compressor family, three arms on the same field/bound:

- **native** — the reference binary, unmodified algorithm.
- **fzgm staged** (`FZ_SPECIALIZE=off`) — the FZGM port, one kernel per stage.
- **fzgm specialized** (`FZ_SPECIALIZE=auto`) — the same port, finalize-time
  kernel fusion.

"Best FZGM" in the results/artifact is `min(staged, specialized)` peak at that
cell — the fairer number to compare against native, since a reader comparing
architectures shouldn't be penalized by picking the worse of two FZGM policies.

## Measurement method (what actually produces a number)

Two instruments, run simultaneously on one process invocation per cell:

1. **`tools/cuda_mem_probe/`** — an `LD_PRELOAD` interposer on the
   `cudaMalloc` family (see its README for the interception mechanism).
   Reports the peak **live, logically-tracked** allocation. This is a
   **decomposition/validation** signal, not the primary number: it is
   provably a subset of the true footprint (validated by construction —
   it only sees calls to symbols it wraps), and it undercounts by a
   **fixed, per-tool, non-scaling amount** for at least one family (see
   Caveats). It was used first to calibrate FZGM's own self-reported
   `peak_device_bytes` (`Pipeline::getPeakMemoryUsage()`): probe peak =
   self-report + the CLI's raw input buffer, exactly, to the MB, confirming
   the self-report is accurate for what it claims to cover (the pool's
   working set) and identifying precisely what it excludes.
2. **NVML** (`pynvml`, direct bindings — **not** subprocess calls to
   `nvidia-smi`) polling `nvmlDeviceGetComputeRunningProcesses` for the
   target PID's `usedGpuMemory`, in a tight loop (~7us/call, effectively
   free; polled every 0.5ms) from a background thread for the process's
   lifetime, keeping the running max. **This is the primary/authoritative
   number.** It is the driver's own per-process resident-memory accounting,
   so it catches everything regardless of which API (if any) a library used
   to get there — CUDA context, module/JIT loading, `cudaMalloc`-family
   calls, driver VMM, allocator-internal churn.
   - An earlier pass (superseded) shelled out to `nvidia-smi` per sample
     from bash; each spawn cost enough wall time that sub-second cells could
     be under-sampled (observed as `probe_peak > nvml_peak`, which is
     structurally impossible if NVML is sampled correctly, since the probe's
     scope is a subset of NVML's). The `pynvml` rewrite eliminated this
     entirely — 0 such inversions across the full re-run.
3. Each cell runs **3 repetitions**; the reported peak is the **max across
   reps** (peak-of-peaks — a mean would understate a real worst case).
   Observed rep-to-rep spread was 0-2 MB on almost every cell, i.e. these
   measurements are highly reproducible at one error bound on one machine.

## Tool provenance (same builds as the throughput/CR campaign)

| tool | version | commit | build |
|---|---|---|---|
| FZGModules | 2.0.0-dev | `d511ebc` | Release, `build_paper/` (isolated, self-contained RUNPATH) |
| cuSZ | 0.17.3 | `e1c0135f` | Release, `PSZ_BACKEND=cuda`, `recipe=compressors/build/cusz.sh` |
| cuSZp2 | 2.0.1 | `240240b2` | Release, `recipe=compressors/build/cuszp2.sh` |
| cuSZp3 | 3.0.0 | `da90afb4` | Release, `recipe=compressors/build/cuszp3.sh` |
| FZ-GPU | 0.1-3-g | `c7e83f76` | Release, **shadow rebuild** (below) |
| PFPL | 36f5aaef | `36f5aaef` | Release, **shadow rebuild** (below) |
| FSZ | 1.0.0 | `43240ed1` | Release, `CMAKE_CUDA_ARCHITECTURES=90` |

Commits/patches match the pinned `x-<tool>-tool` anchors in
`configs/experiments/specialization_vs_native_full.yaml`.

**Shadow rebuilds — FZ-GPU and PFPL.** Both ship statically linked against
`libcudart`, which makes `LD_PRELOAD` interposition a no-op (the loader never
resolves `cudaMalloc` at runtime, so the probe's wrapper is never in the
symbol-resolution chain — see `tools/cuda_mem_probe/README.md` Known
Limitations). Rebuilt both from the *same pinned commit*, with the *same*
recipe, adding only `-cudart shared` to the `nvcc` invocation (no source
patch). Round-trip correctness verified against the pinned binary's output
(matching CR/PSNR) before use. Shadow copies live outside the repo, at
`/tmp/memprobe_fzgpu/fz-gpu-dyn` and `/tmp/memprobe_pfpl/bin/{f32,f64}/gpu/`
— **not persisted**; anyone reproducing this needs to redo the shadow build
(commands: FZ-GPU — `nvcc -gencode arch=compute_90,code=sm_90 -gencode
arch=compute_90,code=compute_90 -cudart shared src/claunch_cuda.cu -Iinclude
--extended-lambda -c` then link the same way; PFPL — same `-cudart shared`
flag added to each `f{32,64}_{noa,abs,rel}_{compress,decompress}_cuda` nvcc
invocation in `PFPL/makefile`'s `gpu:` target).

## Field selection

Eight fields spanning ~100x in size, chosen specifically to resolve where the
native/FZGM comparison crosses over (see D46) — not a representative corpus
sample:

| field | bytes | dtype | dims (fast-to-slow) |
|---|---|---|---|
| EXAALT/vx | 11.48 MB | f32 | `[2869440]` (1-D) |
| CESM-2D/CLDHGH | 25.92 MB | f32 | `[3600, 1800]` |
| HURR/CLOUD | 100.0 MB | f32 | `[500, 500, 100]` |
| MIRANDA/density | 302.0 MB | f64 | `[384, 384, 256]` |
| NYX/temperature | 536.9 MB | f32 | `[512, 512, 512]` |
| CESMATM-3D/CLDICE | 673.9 MB | f32 | `[3600, 1800, 26]` |
| NWCHEM/t631 | 823.6 MB | f64 | `[102953248]` (1-D) |
| HACC/vx | 1123.8 MB | f32 | `[280953867]` (1-D) |

Dims/paths/dtype are from `configs/datasets.yaml`; one error bound only
(`1e-3`) — a pre-flight check (P2 in the working notes) confirmed peak memory
is **invariant to error bound** for every pipeline tested (buffers are
worst-case-sized from element count, not compressed output size), so sweeping
bounds would not have added information.

## Per-tool invocation specifics (gotchas that cost time to find)

- **FZGM dtype retargeting.** FZGM's checked-in pipeline TOMLs are all
  written `input_type = "float32"` (documented as D27 in benchkit); the
  adapter patches this to the field's real dtype at TOML-render time. A
  naive standalone invocation that skips this and feeds `.d64` data through
  an unpatched `input_type = "float32"` stage produces silent garbage or a
  crash (`cudaFreeAsync invalid argument`) that looks like a specialization
  bug but is not one — confirmed by patching `input_type` and re-running
  clean. The standalone script replicates this patch (`input_type =
  "float32"` → `"float64"` string substitution) for every f64 field.
- **cuSZp3's `-d` parser always reads 3 numbers** (z, y, x) after `-d 2` or
  `-d 3`, even for true 2-D data — `-d 2 <y> <x>` is a parse error; it must
  be `-d 2 1 <y> <x>` (z=1 placeholder). Source: `examples/cuSZp.cpp`'s
  argument loop unconditionally does `i+=4` and reads `argv[i+2..i+4]`
  regardless of whether `processingDim` is 2 or 3.
- **FSZ needs an explicit `-t f64`** before `-eb` for double data; omitting
  it silently mis-sizes the expected element count (its own error message
  reports "float32 values" for an f64 file when the flag is missing).
- **PFPL has no single round-trip binary** — separate
  `f{32,64}_{noa,abs,rel}_{compress,decompress}_cuda` executables, so "PFPL
  native peak" for one cell is `max(compress-process peak, decompress-process
  peak)`, not a single process's number the way every other family is.
- **Dimension order**: FZGM/cuSZ take fast-to-slow dims joined by `x` (e.g.
  `-l 512x512x512`, matching `configs/datasets.yaml`'s `dim_order: fast-to-slow`).
  cuSZp3 and FZ-GPU want slow-to-fast (z, y, x). Getting this backwards
  changes predictor tiling / can trip a boundary correctness check (observed:
  cuSZp3 2-D "Exceeding data count: 8" on a placeholder-z of the wrong
  value) but does **not** change total element count or the peak-memory
  number, since buffer sizes depend on total elements, not the axis
  labeling — worth knowing if a future run hits a similar boundary warning
  and wants to know whether to trust the memory number anyway (yes).

## Exclusions (6 of 140 cells; none are bugs)

| cell | cause | class |
|---|---|---|
| `native:cusz` @ EXAALT | segfault | known native limit (matches the earlier throughput campaign's documented `cusz EXAALT @1e-3` segfault) |
| `fzgm:cusz`, `fzgm:fzgpu` @ EXAALT | 45.6% outlier overflow (quant radius / outlier capacity) | known port capability boundary, same class as the campaign's documented EXAALT radius-limit cells |
| `fzgm:{cuszp2,cuszp3,szp_composed}_..._sp:off` @ NYX/temperature | ABS-mode `error_bound=1e-3` literal exceeds NYX/temperature's TCode range | known preset/field mismatch class (same as the main campaign's `accepted_failures`, e.g. CESM-2D/PSL) |

All three classes are pre-existing, already-documented failure modes from the
main vs-native campaign, encountered again here because this sweep touches
fields/tools the main campaign's `accepted_failures` list doesn't itemize by
name. None required a code change; all are excluded from the comparison, not
silently dropped.

## Platform caveat: this host is a GPU-passthrough VM, and that may inflate the fixed-overhead magnitude

The entire sweep ran on one JetStream2 H100 instance. `nvidia-smi -q` confirms
`Virtualization Mode: Pass-Through` (a dedicated physical GPU, not a time-sliced
vGPU), but it is still a full KVM/QEMU guest: `lspci` shows a `virtio-pci`
memory-balloon device, and `dmesg` shows the IOMMU running with
`DMA domain TLB invalidation policy: lazy mode` — every `cudaMalloc`/`cudaFree`
is mediated by a VFIO/IOMMU page-table update, not a direct path to the GPU's
own memory manager the way bare metal would be.

This matters specifically because **this repo already has a documented finding
that this class of platform makes repeated `cudaMalloc`/`cudaFree` calls
anomalous**: `compressors/cuSZp-V2.0.1/src/cuSZp_entry_f32.cu` carries a local
patch, with a comment stating plainly that per-call `cudaMalloc`/`cudaFree` "is
cheap on bare-metal GPUs but on some platforms (observed on a GPU-passthrough
cloud VM) ... carry large, highly variable latency (seen up to several hundred
ms for a single call)". That finding was about latency; the mechanism (IOMMU-
mediated allocate/free churn under lazy invalidation) could plausibly also
inflate NVML's guest-visible resident-memory reading for a process that churns
many small device allocations internally — which is exactly what cuSZp2/3's
**outlier** mode does via CUB's internal temp-storage allocation, and exactly
the mode that showed the large (~1.4GB), field-size-invariant "hidden" gap this
methodology's headline finding rests on (its own **plain**-mode isolation,
same binary, same host, dropped that gap to ~550MB — comparable to every other
tool's baseline — so the effect is real and mode-specific, but its *absolute
magnitude* has not been checked off this one host).

**What is likely robust vs. likely platform-sensitive:**
- Likely robust (architecture-driven, should reproduce elsewhere): the
  *qualitative* shape — native tools that repeatedly allocate/free scratch pay
  a fixed per-process cost that FZGM's single persistent memory pool avoids by
  design, and specialization narrows FZGM's own gap to native. This is the
  same mechanism class as the throughput story (RQ2), not something this
  platform would manufacture from nothing.
- Likely platform-sensitive (not yet cross-checked): the *magnitude* of the
  fixed overhead (~1.4GB for cuSZp2/3-outlier, ~500-650MB baseline for
  everything else) and therefore the *exact crossover point* (~400-500MB).
  These should be stated as measured-on-this-host, not quoted as a portable
  constant, unless corroborated elsewhere.

**Recommended before quoting an exact crossover number in the paper:** spot-
check one or two cells (e.g. cuSZp2-outlier on CLDHGH and HACC/vx) on a
non-passthrough-VM machine in the project's existing fleet (BigRed200 A100 or
Delta H200/MI100 — see `docs/running-the-full-corpus.md`). A similar-magnitude
result there would settle this; a much smaller gap would mean the crossover
point (not the qualitative direction) needs restating as host-specific. Given
D46 scopes this to one paragraph and one figure, the safer default language
absent that cross-check is qualitative ("native's advantage grows with field
size; below a few hundred MB, FZGM's pool design avoids a fixed cost native
pays repeatedly") rather than a specific quoted crossover size.

## What this is *not*

- Not integrated into benchkit (`benchkit/adapters/`) — the script and driver
  are standalone (in the working scratchpad), reusing benchkit's dataset
  paths/dims and mirroring its dtype-retargeting convention, but outside its
  session/provenance/validity-gate machinery. Formalizing this into a real
  adapter (so native `peak_device_bytes` becomes a first-class benchkit row
  field) is future work, not required for the paper's current scope (D46:
  one paragraph, one single-column figure).
- Not a corpus-representative sample — the 8 fields were chosen to resolve
  the crossover, not to estimate an aggregate "typical" ratio.
- Not swept across bounds (justified above) or across GPUs/architectures.
