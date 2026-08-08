# nvCOMP adapter contract

**Tool:** nvCOMP **5.3.0.16** (NVIDIA GPU lossless compression SDK), driven through
`nvcomp_cli` — a CLI built from this repo (`tools/nvcomp_cli/`)
**Adapter:** `benchkit/adapters/nvcomp.py` · `NvcompAdapter`
**Compressor key in experiments:** `nvcomp`
**Status:** ✅ Working (JetStream2 H100, CUDA 12.9, sm_90). Verified bit-exact
round-trip on CESM-2D / HURR / NYX.

---

## Why this adapter exists

To answer one question: **how does FZGM's GPU-Zstd back end compare to NVIDIA's?**

FZGM has a `gpu_zstd` preset that reproduces Zstandard's block structure on the
GPU (LorenzoQuant → GPULZ split → Huffman + ANS×3). nvCOMP ships a GPU Zstd. The
obvious comparison is a trap, and most of this document is about why.

### The trap

FZGM's `gpu_zstd` preset is an **error-bounded lossy compressor**. nvCOMP's Zstd
is a **lossless byte-stream codec**. On CESM-2D/CLDHGH at `rel_range` 1e-3:

| | CR |
|---|---|
| FZGM `gpu_zstd.toml` (lossy, eb=1e-3) | 10.87x |
| nvCOMP Zstd (lossless) | 1.14x |

That 9.5x gap is **the Lorenzo predictor**, not the Zstd implementation. Quoting
it as "FZGM's Zstd beats nvCOMP's Zstd by 9.5x" would be wrong.

### The two honest framings

Both are implemented; run whichever the claim needs.

**1. Lossless head-to-head** — `configs/experiments/nvcomp_vs_fzgm_lossless.yaml`

Strip the predictor. `configs/pipelines/gpu_zstd_lossless.toml` is the same DAG
without LorenzoQuant, so both sides consume the identical raw f32 bytes under the
identical bit-exactness contract. Answers: *whose lossless GPU codec is better on
raw scientific data?*

**2. Back-end isolation** — `configs/experiments/nvcomp_vs_fzgm_backend.yaml`

Give both coders the bytes FZGM's own preset feeds its back end: the uint16
Lorenzo quant codes. `scripts/extract_quant_codes.py` runs a predictor-only
pipeline and pulls the `codes` port out of the `.fzm` archive as a derived
dataset; `configs/pipelines/gpu_zstd_codes.toml` is `gpu_zstd.toml`'s coder half.
Answers: *whose Zstd-family coder is better on the input FZGM's actually sees?*
This is the fairer comparison for FZGM's back end, which was tuned on Lorenzo
residuals rather than on raw mantissa bytes.

### Measured results (JetStream2 H100, 2026-08-06)

Both tables quote **device and host** throughput, because the two are far apart on
one side and identical on the other — see "Timing" below before reading these.
nvCOMP appears at both the vendor-default 64 KB chunk and its measured-best 16 KB.

**Framing 1 — raw f32 fields.** 10 fields across CESM-2D / HURR / NYX, 10 reps,
70/70 cells ok, all bit-exact, all `timing_reliable`. Geometric-mean CR, median
throughput:

| | geo CR | compress **host** | decompress **host** |
|---|---|---|---|
| FZGM `gpu_zstd_lossless.toml` | 1.100 | **23.8** | 44.7 |
| nvCOMP Zstd (64 KB) | **1.155** | 7.8 | 18.7 |
| nvCOMP Zstd (16 KB) | 1.153 | 9.0 | **46.4** |
| nvCOMP GDeflate | 1.151 | 8.0 | 46.9 |
| nvCOMP ANS | 1.120 | 207.3 | 348.4 |
| nvCOMP LZ4 | 0.999 | 9.7 | 126.9 |

**FZGM wins compress ~2.6x; decompress is a tie; nvCOMP Zstd keeps a ~5% ratio
edge.** Nobody does *well* on ratio here — ~1.15x is what byte-oriented
lossless compression is worth on f32 mantissas, which is the argument for the
error-bounded approach, not for either Zstd.

> **These numbers changed on 2026-08-06.** Before the `gather_kernel` fix
> (FZGPUModules CHANGELOG, "concat gather_kernel used one block per segment"),
> FZGM's host-wall compress here was **7.6 GB/s with a 3.53x host/device gap**, and
> nvCOMP Zstd at 16 KB *beat* it. The gap was one 256-thread block copying the
> whole 23 MB archive. It is now 1.04x. Any comparison from before that date
> understates FZGM's end-to-end compress by ~3.4x.

**Framing 2 — identical Lorenzo quant codes.** 3 CESM-2D fields, 10 reps:

| | geo CR | compress **host** | decompress **host** |
|---|---|---|---|
| FZGM `gpu_zstd_codes.toml` | **8.144** | **13.9** | **15.5** |
| nvCOMP Zstd (64 KB) | 6.636 | 0.9 | 3.6 |
| nvCOMP Zstd (16 KB) | 6.460 | 3.4 | 10.7 |
| nvCOMP ANS | 6.081 | 79.9 | 105.6 |

**This is where FZGM's back end genuinely wins, and the win survives every
correction**: +26% ratio over nvCOMP Zstd, 4.1x faster compress and 1.5x faster
decompress *on host wall time*, against nvCOMP at its best chunk size.

The reversal between the two framings is the actual finding. GPULZ plus the
uint16-symbol Huffman exploit structure that exists in Lorenzo residuals and not in
raw mantissas — so on the input FZGM's back end was designed for it wins clearly,
and on bytes it was not designed for it does not.

Both host/device gaps are now ~1.05x. Before the `gather_kernel` fix this table's
gap was 1.10x while the raw-field table's was 3.53x — same code, 8x fewer output
bytes to assemble — which is what identified the concat gather as the culprit.

---

## SDK version currency

**Pinned to 5.3.0.16** (released 2026-07-14), the newest published redist. Check for
a newer one — the SDK is a hand-unpacked tarball and nothing keeps it current:

```bash
curl -s https://developer.download.nvidia.com/compute/nvcomp/redist/ | grep -o 'redistrib_[0-9.]*\.json'
curl -s https://developer.download.nvidia.com/compute/nvcomp/redist/redistrib_<ver>.json
# tar.xz under nvcomp/linux-x86_64/, sha256 in the same JSON
```

`NvcompAdapter.provenance()` reads the version out of
`$NVCOMP_ROOT/lib/cmake/nvcomp/nvcomp-config-version.cmake` and records it as
`nvcomp_version` in every session manifest, so a result can always be traced to the
build that produced it. `~/compressors/nvcomp` is a symlink; 5.2.0.10 is kept beside
it for A/B.

### 5.2.0.10 → 5.3.0.16, measured (CESM-2D/CLDHGH, 8 reps, 3 trials)

The first version used here was 5.2.0.10, which was already one minor release
behind. It mattered for one algorithm:

| algorithm | chunk | 5.2 compress | 5.3 compress | Δ | CR |
|---|---|---|---|---|---|
| Zstd | 64 KB | 4.22 | 4.16 | −1.4% | identical |
| Zstd | 16 KB | 8.88 | 8.65 | **−2.6%** | identical |
| LZ4 | 16 KB | 14.07 | 14.10 | +0.2% | identical |
| GDeflate | 16 KB | 9.23 | 9.20 | −0.3% | identical |
| **ANS** | 64 KB | 82.8 | 106.6 | **+29%** | identical |
| **ANS** | 16 KB | 83.1 | 111.2 | **+34%** | identical |

- **Compression ratios are bit-identical in all 8 configurations** — same
  bitstreams, so no result that quotes CR is affected by the version.
- **ANS gained 29–34% on compress** (three non-overlapping trials each). Decompress
  also moved: 243 → 348 GB/s on the raw-field corpus.
- **Zstd compress regressed ~2.6%**, consistently. Small, but it means the earlier
  5.2 measurements were, if anything, slightly *favourable* to nvCOMP Zstd — so the
  FZGM-vs-Zstd conclusions were never at risk from being a version behind.
- Headers are API-identical for everything `nvcomp_cli` uses (diffs are brace style
  and pointer placement only) and the exported Zstd symbol set is unchanged, so the
  tool rebuilt without modification.

The Python wheel (`nvidia-nvcomp-cu12`) ships only the Python bindings — the C++
headers and `libnvcomp.so` come from the redist tarball above.

---

## The `lossless` error mode

Every nvCOMP algorithm compresses a byte stream and takes **no error bound**, so
`nvcomp` accepts exactly one canonical error mode, `lossless` (added for this;
see `benchkit/config.py` `CANONICAL_MODES` and DESIGN.md D31). It takes no
`error.bounds`, like `from_toml`.

```yaml
error:
  mode: lossless      # no `bounds` key
```

`prepare()` reports `eb=0`, `basis="lossless"`. `metrics.compute_quality` then
checks **bit-exactness** instead of bound satisfaction: `eb_abs_effective=0`,
`err_over_bound=0`, and `eb_satisfied` is `max_abs_err == 0` with **no `eb_tol`
slack** — there is no bound for round-off to hide under, and a lossless codec that
returns anything but its input is broken.

The FZGM adapter accepts `lossless` too, for coder-only pipelines. It **refuses**
a pipeline that declares an `error_bound` under this mode rather than shipping it:
`error_mode` is what the validity gate keys its carve-outs on, so one mislabeled
row would corrupt the gate for every other row in the session.

### Validity-gate consequences (important)

Lossless rows have `max_abs_err == 0` and `psnr == inf` **by construction**. Two
carve-outs in `benchkit/validity.py` exist because of that:

- **`degenerate_fields()` ignores lossless rows entirely.** The detector infers
  "this field is constant" from "no compressor produced any error on it" — sound
  over lossy runs, meaningless over lossless ones. Without this, an all-lossless
  session marks *every* field degenerate and the gate drops the whole session.
- **`cr <= 1` is `lossless_expansion`, not `expansion`, and is RETAINED.** For a
  lossy codec, `cr <= 1` is a corruption signal. For a lossless one it is an
  ordinary measurement: nvCOMP LZ4 compresses CESM-2D/CLDHGH to **0.996x**,
  because raw f32 mantissas are near-incompressible byte-wise and framing overhead
  is real. Dropping those rows would bias every lossless CR mean upward by
  deleting exactly the cases where the codec lost.

`psnr == inf` is reported as `lossless_exact` (quality aggregates only, CR and
throughput retained) rather than the unexplained-anomaly code `psnr_nonfinite`.
So `report --aggregate` on a lossless session shows *N usable for CR/throughput,
0 usable for quality* — that is correct, not a failure.

---

## CLI

`nvcomp_cli` is ours, not NVIDIA's. nvCOMP ships as a library with no vendor
binary that reads a file and reports device time, so `tools/nvcomp_cli/` provides
one under the harness's contract.

```bash
nvcomp_cli --compress   -i <in>     -o <out.nvc> -a zstd [--chunk-size N] [--level N]
nvcomp_cli --decompress -i <in.nvc> -o <out.bin> -a zstd
nvcomp_cli --benchmark  -i <in>                  -a zstd --reps N [--warmup N]
# common: --report-json <file>  --quiet
# typed coders: --dtype <name>;  cascaded: --rles N --deltas N --bp 0|1
```

Algorithms: `zstd`, `lz4`, `deflate`, `gdeflate`, `ans`, `snappy`, `gzip`,
`bitcomp`, `cascaded`.

### Byte coders vs type-aware coders

The first seven treat the input as an undifferentiated byte stream. **Bitcomp and
Cascaded do not** — both model it as an array of a declared element type, and
nvCOMP defaults that type to `NVCOMP_TYPE_UCHAR`. On this corpus that default is
not neutral, it is wrong (D38); CESM-2D/CLDHGH raw f32:

| dtype | Bitcomp CR | Cascaded CR |
|---|---|---|
| `uchar` (**nvCOMP's default**) | 0.996 | 0.998 |
| `uint` (correct 4-byte width) | **1.437** | **1.515** |

So `dtype=` is **required** for those two and **rejected** for the rest. Two
things follow that belong in any writeup quoting these rows:

- **nvCOMP 5.3 has no f32 element type.** `NVCOMP_TYPE_FLOAT` (enum value 8) was
  removed; the enum is char/uchar/short/ushort/int/uint/longlong/ulonglong plus
  float16. f32 fields can only be described by 4-byte *integer* width, and f64 by
  8-byte. That is a modelling limit of the comparison, not a choice on our side —
  which is why no `float`/`f32` spelling is accepted anywhere: writing
  `dtype=uint` forces the limitation to be visible in the config.
- **Correctly typed, Cascaded is the strongest nvCOMP entry on raw f32** (1.539
  at its best scheme, vs Zstd 1.184 / Deflate 1.183 / FZGM GPU-Zstd 1.131),
  because it is the only one that predicts. It collapses on quant codes (2.308 vs
  FZGM's 8.144) for the same reason — its delta pass re-differences an already
  decorrelated stream.

Cascaded's own default scheme is also not its best. Sweeping raw f32: `bp=0` gives
0.998 in *every* configuration (bitpack is the only pass that compresses), one
delta pass is worth 1.300 → 1.539, and RLE hurts monotonically (1.539 / 1.531 /
1.515 / 1.508 for 0/1/2/3 passes), so the documented `rles=2` costs 1.6%. Per D34
both the default and the swept best are run.

Cascaded is also the one nvCOMP algorithm with a **composable** FZGM counterpart
rather than a single-coder pairing: `{num_RLEs, num_deltas, use_bp}` maps directly
onto `rle`/`rre`/`rze`/`rare`/`raze`, Lorenzo, and `bitpack`/`adaptive_bitpack`,
so FZGM can mirror any Cascaded configuration as a DAG.

### Pipeline string

nvCOMP has no config file, so the experiment's `pipeline` field carries the
algorithm and its knobs:

```
nvcomp:<algo>[:chunk=<bytes>][:level=<N>][:dtype=<name>][:rles=N][:deltas=N][:bp=0|1]

nvcomp:zstd                       # chunk=65536 (nvCOMP's documented Zstd sweet spot)
nvcomp:zstd:chunk=131072
nvcomp:gdeflate:level=5
nvcomp:bitcomp:dtype=uint         # f32 field, 4-byte integer width
nvcomp:bitcomp:dtype=ushort:level=1          # uint16 quant codes, sparse algorithm
nvcomp:cascaded:dtype=uint:rles=0:deltas=1:bp=1
```

`level` maps to nvCOMP's `algorithm` field on the Deflate/Gdeflate opts structs
(0 = entropy-only … 5 = highest ratio) and on Bitcomp's (0 = default, 1 = sparse).
Zstd, LZ4, ANS, Snappy and Gzip expose no level in 5.3. Every option that does not
apply to the chosen algorithm is a **hard error, not a no-op** — otherwise a
config that thinks it is sweeping something quietly emits identical rows under
distinct variant names.

### ⚠ Bitcomp's `--level` is needed on DECOMPRESS too (D39)

Deflate/Gdeflate's `algorithm` is an encoder-side choice the NVCOMP_NATIVE header
describes, so decompression needs only `-a`. **Bitcomp's is not.** A stream
written with `algorithm=1` (sparse) and decoded by a manager built with the
default `algorithm=0` decodes to **wrong bytes at exit 0** — measured on
EXAALT-qcodes-noa0.001/xx as `status: ok`, a plausible CR of 1.89, and **PSNR
20.85 dB on a codec that is lossless by construction**. Nothing in nvCOMP errors;
only the harness's bit-exactness check catches it. `prep_algo_args()` therefore
re-emits `--level` for bitcomp specifically. Do not simplify it back to `-a`
alone. Generalizes: an option that is encoder-only in one algorithm may be
structural in another — verify per algorithm, never by family.

---

## Timing — read this before quoting a throughput ratio

`timing_method: cuda_events_device_only_in_process`. The *method* matches FZGM's
and cuSZp's `device_ms` — a CUDA event pair on the stream, excluding H2D/D2H —
unlike the zfp/SPERR CPU adapters, which are external wall-clock.

**But the two brackets do not exclude the same amount of work**, and on this
comparison the difference is large enough to change the conclusion. Measured on
CESM-2D/CLDHGH compress:

| | device ms | host ms | host/device |
|---|---|---|---|
| FZGM `gpu_zstd_lossless.toml` (split mode), **before** gather fix | 1.26 | 3.60 | **2.85x** |
| FZGM `gpu_zstd_lossless.toml` (split mode), **after** | 1.26 | 1.33 | 1.06x |
| FZGM `gpulz->ans` (single stream) | 0.73 | 0.75 | 1.02x |
| nvCOMP Zstd (manager) | 6.14 | 6.14 | 1.00x |

FZGM's `device_ms` is `dag_elapsed_ms`, a CUDA event pair around `dag->execute()`
only. Host work inside `pipeline->compress()` but outside that bracket is not
counted — and for a **split-mode** pipeline that is 2.33 ms of assembling GPULZ's
four coded ports into one archive on the host. The single-stream row above is the
control: same tool, same measurement, 0.016 ms of gap. nvCOMP's manager writes one
contiguous device buffer, so its bracket excludes essentially nothing.

The `gpu_zstd_lossless` row above is what this section was written to catch: it was
one 256-thread block copying the whole archive, invisible in `dag_elapsed_ms`
because `concatOutputs()` runs after the DAG event bracket. Fixed 2026-08-06 (see
"Measured results"), and the single-stream row is the control that localized it.

The mechanism is general even though this instance is fixed: a tool's device figure
is only as meaningful as the fraction of its work the bracket contains.

Device-only is still what benchkit reports — it is the standard kernel-cost metric
and the right denominator for stage attribution. But **an end-to-end compress
throughput claim must use the host number or state the exclusion.** Every row now
carries `compress_host_ms_median` and `compress_host_over_device` so this is
checkable without re-deriving it (DESIGN.md D33). Whether FZGM's archive assembly
is inherent or simply not yet moved onto the device is an implementation question —
but until it moves, it is real time a caller pays.

### Chunk size is nvCOMP's biggest throughput knob

nvCOMP's header recommends 65536. On this H100 that costs Zstd up to 2.1x, because
chunk count *is* the parallelism — 25.9 MB / 64 KB is 396 chunks across 132 SMs,
about 3 per SM. Compress on CESM-2D/CLDHGH:

| chunk | 4 KB | 8 KB | 16 KB | 32 KB | 64 KB | 128 KB | 256 KB |
|---|---|---|---|---|---|---|---|
| GB/s | 7.55 | 8.08 | **8.84** | 6.87 | 4.22 | 2.67 | 1.61 |

CR spread across that whole range is ~0.7% (16 KB is 0.2% worse than 64 KB). The
effect shrinks where chunks are already plentiful — NYX at 536 MB gives 8.80 vs
7.70, only 1.14x — so it bites hardest on the small fields that dominate a corpus
by count. Since FZGM's side is tuned by measurement, the experiments run **both**
`nvcomp:zstd` and `nvcomp:zstd:chunk=16384` and quote the better one.

### The manager (HLIF) API is not the bottleneck

Checked directly against the low-level `nvcompBatchedZstdCompressAsync` batched
API that NVIDIA's own benchmarks use, same data and chunk size:

| | device ms | GB/s |
|---|---|---|
| HLIF `ZstdManager` | 6.137 | 4.22 |
| low-level batched | 5.989 | 4.33 |

**+2.5%**, and the HLIF number includes writing its metadata header (22,733,105 vs
22,725,351 raw chunk bytes). Switching to the low-level API would buy ~2%, not 2x,
and would cost the self-describing bitstream that makes `--decompress` work from a
file. Not worth it; the manager stays.

`--benchmark` holds the input on the device and brackets only the nvCOMP manager's
`compress()`/`decompress()` call with CUDA events. Specifically:

- **One manager reused across all reps.** nvCOMP allocates its scratch buffer
  lazily on first `compress()`; re-creating the manager per rep would time that
  allocation instead of the codec. An untimed compress runs before the loop for
  the same reason.
- **`configure_decompression()` is hoisted out of the loop.** It parses the
  bitstream header and synchronizes the stream — setup, not decode work.
- **No H2D/D2H, no file I/O, no process startup** inside the timed region.
- **Both phases are pre-warmed.** nvCOMP allocates decompression scratch lazily on
  the first `decompress()` exactly as it does for compress; rep 0 measured 20.4 ms
  against a 13.8 ms steady state (+48%) on CLDHGH. benchkit's warmup-rep drop hid
  that from the median, but a caller running `--reps 1` would have silently
  reported the cold number, and the compress path was already being warmed — so the
  asymmetry was *ours*, not nvCOMP's. Fixed; rep 0 now matches steady state.
- One subprocess yields all N reps, so `--warmup 0`: benchkit already runs
  `warmup_reps + repetitions` and drops the first `warmup_reps` in
  `metrics.summarize_timing`. Doing it in the tool too would hide the ramp from
  the machinery built to measure it.

## Compressed size

The manager runs in `NVCOMP_NATIVE` bitstream mode, whose header carries the
chunking metadata needed to decompress. That header is part of the real artifact,
so its bytes count in `compressed_bytes` — the same rule applied to every other
tool here. Checksums are off (`NoComputeNoVerify`), so nothing is paid for
integrity checking the harness does itself.

---

## Build

nvCOMP is a prebuilt SDK download, not a source tree. Unpack it and point
`NVCOMP_ROOT` at the directory containing `include/` and `lib/`:

```bash
export NVCOMP_ROOT=$HOME/compressors/nvcomp     # 5.2.0.10
./scripts/build-nvcomp-cli.sh                   # CUDA_ARCH=90 by default
export NVCOMP_CLI=$PWD/tools/nvcomp_cli/build/nvcomp_cli
```

Both are set by `scripts/env-jetstream2.sh`.

**Toolchain gotcha.** The build script pins `nvcc` and `g++` rather than taking
PATH order. On the JetStream2 node, `nvcc` resolves through the nvhpc bundle to a
CUDA **11.8** wrapper and nvhpc's default C++ is `nvc++`; the configure step fails
with a confusing compiler-identification error. The script finds a CUDA ≥ 12
`nvcc`, skipping the `compilers/bin/nvcc` redirector, and forces `g++` as host
compiler. (Independently, `nvc++` has a stack-alignment codegen bug in large CUDA
translation units — avoid it here regardless.)

Zstd requires nvCOMP ≥ 3.0. The `CMakeLists.txt` bakes `NVCOMP_ROOT/lib` into the
binary's RPATH, because the harness invokes it as a bare subprocess and should not
depend on `LD_LIBRARY_PATH` being right in whatever shell or SLURM job launched it.

---

## Derived quant-codes datasets

`scripts/extract_quant_codes.py` builds framing 2's input:

```bash
python scripts/extract_quant_codes.py --dataset CESM-2D --fields all \
    --eb 1e-3 --mode NOA --radius 2048
# writes $BENCHKIT_DATA_ROOT/derived/CESM-2D_qcodes_noa0.001/<field>.u16
# and prints a configs/datasets.yaml stanza
```

It runs a predictor-only LorenzoQuant pipeline and reads the `codes` output port
out of the `.fzm` archive via `benchkit/fzm.py` (a reader for FZGM's archive
format — reader only; fzgmod-cli owns that format and a second writer would be a
second thing to keep in sync).

Three things to know about rows measured on these datasets:

1. **`cr` is a back-end ratio**, over an already-2x-reduced intermediate. Never
   quote it as a compression ratio for the field.
2. **`original_bytes` is the codes size**, so throughput is per byte of codes.
   Do not pool these rows with raw-field rows.
3. **Outlier streams are excluded.** LorenzoQuant also emits `outlier_errors` /
   `outlier_indices` for values its prediction can't represent within the bound;
   both back ends would carry them identically, so they are not part of the
   comparison. The script reports the outlier fraction per field — 0% for CESM-2D
   CLDHGH/PRECT/TS at NOA 1e-3, but a field with a large fraction has a codes
   stream representing less of its data, and a correspondingly less
   representative back-end ratio.

The `u16` dtype required two small additions: `_ELEMENT_SIZE` in `config.py` and
`_NP_DTYPE` in `metrics.py`. `fzgmod-cli`'s `-t` only takes `f32`/`f64`, so the
FZGM adapter presents an integer-typed field to it as an equivalent f32 *length*
(`FzgmAdapter._io`) — a coder-only DAG never interprets its input as numbers, and
the harness still reads both files back as `u16` for its own bit-exactness check.

---

## Gotchas

- **nvCOMP Zstd compression is slow** — 0.8–8 GB/s on the H100 depending on input,
  well under FZGM's GPULZ+Huffman+ANS. Decompression is closer but still behind.
  ANS is the outlier in nvCOMP's family (60–250 GB/s) and is the one to reach for
  if throughput is the constraint.
- **LZ4 expands raw f32 data** — 0.996x, on 8 of 10 fields tested. Expected; see
  the `lossless_expansion` carve-out above. Verified that the carve-out is
  load-bearing on this session: 60/60 rows usable with it, **0/60 without**
  (60 `degenerate_field` + 8 `expansion`).
- **`decompress()` re-derives the algorithm from the pipeline string**, since
  `Adapter.decompress` receives a path rather than a `Prepared`. Chunk size and
  level are compression-side only and come back from the NVCOMP_NATIVE header.
- **`gpu_zstd_lossless.toml` uses Huffman\<uint8\>/bklen 256; `gpu_zstd_codes.toml`
  uses Huffman\<uint16\>/bklen 4096.** Not an inconsistency: quant codes *are*
  uint16 symbols whose high and low bytes correlate, and raw f32 field data has no
  such 16-bit structure. Same reasoning, opposite answer, because the input differs.
