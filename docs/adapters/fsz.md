# FSZ adapter

`benchkit/adapters/fsz.py` — reference adapter for **FSZ 1.0.0** (Jiajun Huang,
*FSZ: Breaking the Prediction-Throughput Trade-off in GPU Lossy Compression*,
SC'26, arXiv:2607.15413; BSD-3-Clause, released 2026-08).

FSZ matters here for a specific reason: FZGM's `AdaptiveLorenzoStage` and the
`fsz.toml` preset were **reconstructed from the paper before any FSZ source
existed**. This adapter is what makes that reconstruction checkable.

## Build

```bash
source ~/load-env                       # nvhpc 25.7 -> CUDA 12.9, needed for sm_90
cd ~/compressors/FSZ
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=90
cmake --build build -j
export FSZ_CLI=$HOME/compressors/FSZ/build/fsz

cd -                                    # back to this repo
./scripts/build-fsz-hosttime.sh         # host-timing harness; see below
```

CMake finds nvcc through `CUDAToolkit_ROOT`/PATH only — without `load-env` it
fails at configure time with "Failed to find nvcc", not at build time.

**Build `fsz_hosttime` too.** Without it the adapter silently falls back to the
stock CLI and every FSZ row carries **no host time**, which is exactly the D33
blind spot. The adapter finds it automatically at
`tools/fsz_hosttime/fsz_hosttime`; override with `FSZ_HOSTTIME_CLI`. Which path
ran is recorded per session as `provenance.timing_method`
(`cuda_events_plus_host_wall` vs `cuda_events_device_only`) — check it before
quoting a host-time comparison.

## CLI contract

| phase | invocation |
|---|---|
| benchmark | `fsz -i data -t f32 -d D1 D2 D3 -eb rel V --csv` |
| compress | `fsz -z -i data -t f32 -d … -eb rel V -o c.fsz` |
| decompress | `fsz -x -i c.fsz -o d.bin` |

With neither `-z` nor `-x`, FSZ does a full in-memory round trip, reports both
device times and its own bound check, and **writes nothing** — so `benchmark()`
has no file I/O inside the measurement.

- **`-d` is C order, D1 slowest-varying.** `FieldSpec.dims` is fast-to-slow, so
  the adapter reverses them. FSZ's algorithm is 1-D over the flattened array
  (256-element tiles); dims only reach the file header and `--ssim`.
- **A `.fsz` container is self-describing**: `-x` needs no `-t`, `-d` or `-eb`.
- **No pipeline options.** FSZ exposes no modes or tuning flags by design, so
  `pipeline` must be the literal `default`; anything else is rejected.

## Error modes

| canonical | native | basis |
|---|---|---|
| `abs` | `-eb abs V` | abs |
| `rel_range` | `-eb rel V` | range |

FSZ's `-eb rel` is documented as a fraction of `max - min`, i.e. canonical
`rel_range` — the same basis as FZGM `NOA` and cuSZ `REL`. `rel_maxabs` has no
equivalent and raises.

## Timing

3 warmup iterations, then **one** timed launch per phase, via a `cudaEvent`
pair. Unlike cuSZp — which averages 100 back-to-back launches inside a single
subprocess and thereby deflates the harness's cv — each FSZ subprocess yields a
genuine single-shot measurement, so the cv across `repetitions` is a real
variance estimate. The `fsz_hosttime` path preserves that shape (`-r 1` per
subprocess) rather than collapsing the repetitions into one process.

Both `device_ms` and `host_ms` are reported for **f32 and f64** when
`fsz_hosttime` is built. The harness instantiates the matching `FSZ<float>` or
`FSZ<double>` overload and places CUDA events inside a host-wall bracket around
the same call and synchronization. Allocation, workspace construction, and H2D
input transfer are outside both timings, matching the FZGM benchmark bracket.
Without the helper the adapter falls back to the stock device-only CLI and
records that fact in session provenance.

Measured host/device is **1.003–1.08**, decaying with field size — launch
overhead and nothing else, since compression is one fused kernel writing one
contiguous device buffer with no host-side archive assembly. That is a
*measured* result, not an assumption: it is the reason FSZ's device figure can
be trusted, and it only became checkable once the harness existed.

## CR basis

The `--csv` line reports the **bare bitstream** size; the `.fsz` container adds
a fixed 56-byte header (`static_assert` in `tools/fsz_file_format.hpp`). The
adapter reports the container — the artifact on disk when `compress()` ran,
else bitstream + 56 — so an FSZ cell's CR includes its header exactly as an
FZGM cell's includes the `.fzm` header. Under 1e-5 relative at corpus sizes,
but it keeps one CR convention across tools.

## Exit status

`0` pass, `1` **error bound violated**, `2` usage/file error. The adapter raises
only on `>= 2`: a reference tool failing its own contract is a result to record,
not a harness error. The verdict is surfaced in `native_quality.native_status`
alongside the tool's self-reported CR and max error, for cross-checking against
the harness's own metrics.

## Measured against FZGM — session `20260807-183830-skyler-h100`

`configs/experiments/fsz_vs_native.yaml`, 20 cells (10 fields x 2 bounds), H100.

- **Quality is identical.** `|ΔPSNR| = 0.0000 dB` on every cell. Both sides
  quantize `q = round(x / 2eb)` and reconstruct `q * 2eb`.
- **CR: FZGM/native geomean 0.9928** (min 0.970 on NYX, max 0.999 on EXAALT).
- **Throughput: native is 2.63x on compress, 1.94x on decompress** (geomean,
  device time — the reported figure, matching how cuSZp/cuSZ/FSZ all publish).
  Host wall time puts it at 2.87x / 2.07x; that column is the **audit** that
  says the device number is safe to quote here, not a competing headline. See
  "Host wall time" below for the one case where it isn't.

### The CR gap is entirely the mode side channel

On NYX/baryon_density the `.fzm` `output` port is **4,277,784 bytes — the exact
byte count of FSZ's bitstream**. The predictor and coder make identical
decisions; the deficit is framing:

| | bytes |
|---|---|
| FZGM excess over FSZ, summed over 20 cells | 1,867,716 |
| accounted for by the `modes` port (2 bits/tile) | 1,859,738 (99.6%) |

FSZ carries the same two flags **free**, in the spare bits of block 0's rate
byte (`FSZ_LZ2_FLAG 0x80`, `FSZ_MEAN_FLAG 0x40`; the rate itself needs 5 bits).
FZGM's `AdaptiveLorenzoStage` cannot: it does not know what its downstream coder
emits. This is the modular DAG's cost, priced exactly — 0.0078 bits/element,
which is invisible at CR 3 and worth 3% at CR 128.

### The throughput gap is materialization, decomposed

**Two different "GB/s" appear below; do not confuse them.**

- **Throughput** = *uncompressed bytes / time* — what a user waits for, and what
  benchkit's `compress_throughput_gbs` means. On NYX/`baryon_density` FZGM
  compresses at **196 GB/s**, FSZ at **978 GB/s**.
- **HBM traffic rate** = *all bytes read+written / kernel time* — a bandwidth
  utilization figure used only in the two tables here to separate "moves more
  data" from "moves it less efficiently". It counts 2688.7 MB for FZGM against
  536.87 MB of user data, so it is ~5x larger than the throughput number and is
  **not** a rate anything is compressed at.

NYX 512³ (536.87 MB), stage times from `--profile`:

| compress | ms | MB moved | HBM GB/s |
|---|---|---|---|
| Quantizer | 0.657 | 1073.7 | 1634 |
| AdaptiveLorenzo | 1.258 | 1073.9 | **854** |
| AdaptiveBitpack | 0.805 | 541.1 | 672 |
| **FZGM total** | **2.720** | **2688.7** | **988** |
| **FSZ (fused)** | **0.546** | **1078.0** | **1974** |

FZGM moves **2.49x** the bytes (two materialized int32 arrays) and runs at
**2.00x** lower aggregate bandwidth. 2.49 x 2.00 = 4.99x, against 5.04x
measured on device time. FSZ reads the input twice, recomputing quantization
rather than spilling it.

| decompress | ms | MB moved | HBM GB/s |
|---|---|---|---|
| AdaptiveBitpack | 0.391 | 541.1 | 1384 |
| AdaptiveLorenzo | 0.615 | 1073.9 | 1746 |
| Quantizer | 0.447 | 1073.7 | 2402 |
| **FZGM total** | **1.453** | **2688.7** | **1850** |
| **FSZ (fused)** | **0.393** | **541.1** | **1377** |

**On decompress FZGM's kernels are 1.34x more bandwidth-efficient than FSZ's**;
the whole 3.73x gap is the 4.97x in bytes moved. Compress is the asymmetric
side, and `AdaptiveLorenzo` forward is the outlier at 854 GB/s against its own
inverse at 1746 GB/s — consistent with the forward kernel never having had the
barrier-removal pass its inverse got.

### Host wall time — the audit, not the headline

**Device time is what this repo reports**, here and generally: it is what cuSZp,
cuSZ, cuSZ-Hi and FSZ itself all publish, so it is the only figure that makes
FZGM comparable to a number in someone else's paper. Host time exists to answer
D33's question — *is the device figure hiding work?* — and is checked, not
quoted. For FSZ vs FZGM the answer is that it is not hiding much, which is
precisely what licenses the device comparison above.


Everything above is device time. Per D33 that is only half an answer, and the
stock `fsz` CLI reports no host figure — so
`tools/fsz_hosttime/fsz_hosttime.cu` (built against `libfsz`) times
`std::chrono` around `fsz::compress(...)` + `cudaStreamSynchronize`, the same
bracket FZGM's "Host elapsed" uses, with allocation, workspace and H2D hoisted
out on both sides. Device events are recorded inside the *same* iteration, so
host and device come from one launch.

**The adapter now emits these into every f32 and f64 row**
(`compress_host_ms_all` / `decompress_host_ms_all`) whenever the harness is
built, so new sessions carry the host column without any out-of-band step. The
numbers below were first collected out-of-band and then reproduced through the
adapter.

**FSZ's `device_ms` is honest: host/device is 1.003–1.08**, decaying with field
size — it is launch overhead and nothing else. One fused kernel, one contiguous
output buffer, no host-side assembly. FZGM's ranges **1.02–1.39**: three kernel
launches plus port bookkeeping, which is a fixed cost and so hurts small fields
most.

Throughput = uncompressed bytes / time, decimal GB/s, geomean over the 20 cells:

| | FSZ / FZGM on **device** time | FSZ / FZGM on **host** time |
|---|---|---|
| compress | 2.63x | 2.87x |
| decompress | 1.94x | 2.07x |

The two agree to within ~9%, so the device figure is a fair summary of this
comparison and is the one to report. The gap tracks field size, because FZGM's
per-launch overhead amortizes.

**The one exception, and the reason to keep checking.** On EXAALT/`xx` (2.87M
elements, the smallest field here) device time says FZGM decompresses **faster**
than FSZ — 166.2 vs 138.5 GB/s at 1e-4. Host time erases it: 124.9 vs 129.3
GB/s, FSZ ahead. FZGM's 1.33x host-over-device eats the entire device-side win.
So a *ranking* claim on a small field needs the ratio checked first; an
aggregate over fields of this size does not. Same pattern as D33, milder.

## Buffer-state methodology (per the author, Jiajun Huang)

> I measure end-to-end performance with no assumptions about GPU buffer state.
> cuSZp's decompression skips zero blocks and needs a pre-zeroed output buffer,
> which cudaMalloc does not guarantee, so I time that memset as part of its
> pipeline; FSZ zeroes inside the decompression call, so its timing already
> includes the cost.

Checked against both sides of this adapter (2026-08-08); no correction needed.

- **FSZ**: confirmed in `src/fsz.cu` — `fsz_decompress_with_ws` issues
  `cudaMemsetAsync(d_out, 0, origSize, stream)` on the *same stream*, before the
  decompress kernel, gated on `high_cr` (`original_bytes / compressed_bytes >
  100`). Both timing paths bracket the *whole* call (the stock CLI's own
  `cudaEvent` pair, and `fsz_hosttime`'s), so the memset was already inside
  every `device_ms`/`host_ms` this adapter has ever recorded. It fires on 2 of
  the 20 `fsz_vs_native` cells: NYX `baryon_density`@1e-2 (CR 127.8x) and NYX
  `temperature`@1e-2 (CR 115.2x).
- **FZGM has no matching dependency to omit.** None of the three `fsz.toml`
  stages assume a pre-zeroed buffer: `AdaptiveBitpackStage` decode writes zero
  per-element for a zero-rate block rather than skipping the write
  (`decode_unpack_kernel_warp`); `AdaptiveLorenzoStage` inverse writes every
  live element directly; `QuantizerStage` in the in-place/linear mode this
  preset uses writes every element directly too ("no memset needed" per its own
  comment at `quantizer.cu:601` — the memset-before-scatter path is only for
  non-inplace outlier mode, unused here).

**One asymmetry worth flagging, not fixing:** at CR > 100 this genuinely favors
FSZ beyond bytes-moved — a single bulk `cudaMemsetAsync` runs closer to peak HBM
bandwidth than FZGM paying the zero-rate cost element-by-element in its decode
kernel. If the corpus grows more triple-digit-CR fields, expect FSZ's
decompress edge to widen further there for this reason, independent of the
fusion-vs-materialization story above.

## Caveats

- **Throughput here is not a like-for-like compressor comparison.** It prices
  fusion against a modular DAG, the same known property tracked for the cuSZp
  natives. CR and quality *are* like-for-like.
- FSZ clamps the per-block rate to 31 bits (`FSZ_RATE_MASK`) with no outlier
  path; FZGM's quantizer has a radius and an outlier mechanism the `fsz.toml`
  preset disables (`linear_mode = true`). The presets agree, but the stages are
  not interchangeable outside this configuration.
- FSZ supports f32 and f64 only, and requires compute capability >= 8.0.
