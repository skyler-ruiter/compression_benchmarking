# FZ-GPU adapter contract

**Tool:** FZ-GPU (Lorenzo + bitshuffle GPU compressor, HPDC'23)  
**Adapter:** `benchkit/adapters/fzgpu.py` · `FzgpuAdapter`  
**Compressor key in experiments:** `fzgpu`  
**Status:** ✅ Functional — source patched; rebuild required before first use.

---

## CLI (patched)

```bash
fz-gpu <input_file> <x> <y> <z> <eb> [compressed_out] [decompressed_out] [repeat]
```

- `<x> <y> <z>`: data dimensions (fastest to slowest).
- `<eb>`: **range-relative** error bound (internally: `eb × range`). Only NOA mode.
- `[compressed_out]`: optional path to write the compressed bitstream (`.fzg`).
- `[decompressed_out]`: optional path to write the decompressed float array (`.bin`).
- When output paths are omitted the binary does the original in-memory round-trip only.
- `[repeat]`: optional in-process repeat count (default 1) — loops the full
  round-trip N times sharing one CUDA stream/context. See "In-process repeat"
  below.
- Only float32 supported.

---

## Adapter model

FZ-GPU does compress+decompress in a single invocation (same as cuSZp). The adapter
follows the same pattern:

- `compress()`: calls with both output paths → writes `c.fzg` + `d.bin`.
- `decompress()`: returns the already-written `d.bin`; no subprocess.
- `benchmark()`: one subprocess call with `repeat=n_runs` (empty output-path
  args to stay in-memory); parses N compress/decompress timing pairs from
  that one process's stdout.

### In-process repeat: fairness fix

Like cuSZ and cuSZ-Hi, `runFzgpu()` originally created and destroyed its own
CUDA stream inside a single call, and `benchmark()` compensated by spawning N
cold subprocesses — each paying fresh CUDA-context creation, module load, and
whatever clock state the GPU happened to be in (clocks are unlocked on
BigRed200), the same failure mode `docs/adapters/fzgm.md` documents for FZGM
itself. A local patch (unmerged, `~/research/compressors/FZ-GPU`) hoists
stream creation out of `runFzgpu()` into `main()` (created once, passed in,
destroyed once) and loops the round-trip `repeat` times sharing that stream.

Unlike cuSZ (which had a real use-after-free bug hiding behind an identical
symptom — see `docs/adapters/cusz.md`), this fix worked on the first attempt:
`runFzgpu()` has no aliased/externally-owned pointers passed between calls
(all arguments are simple values; every `cudaMalloc`/`cudaFree` pair is fully
local to one call), so there was no equivalent ownership bug to trip over.
Verified with 20-rep and 5x-repeated-trial runs on CESM-2D/CLDHGH: zero
crashes, correct compressed/decompressed output (PSNR/CR match a
non-repeated run) across all trials. Rep 1 (cold) reads ~2.0ms compress /
~0.20ms decompress; reps 2+ settle to a stable ~0.22ms compress / ~0.18ms
decompress — roughly a **9x** gap on compress between what the old
N-cold-subprocess method was reporting and the true warm number.
`warmup_reps` (dropped by `metrics.summarize_timing`) discards the first
(cold) rep, same as FZGM/cuSZ-Hi/cuSZ.

---

## Compressed file format

`c.fzg` is a raw binary with a 12-byte header followed by three arrays:

| Field | Size | Notes |
|---|---|---|
| `dataChunkSize` | 4 bytes (uint32) | Number of uint32 words in bitFlagArr |
| `nChunks` | 4 bytes (uint32) | Number of start-position entries |
| `offsetSum` | 4 bytes (uint32) | Number of uint32 words of compressed data |
| `bitFlagArr` | `dataChunkSize × 4` bytes | Bitshuffle flag array |
| `startPosition` | `nChunks × 4` bytes | Per-chunk start positions |
| `compressedData` | `offsetSum × 4` bytes | Bitshuffle-compressed quant codes |

FZ-GPU's own compressed-size formula excludes the 12-byte header. The harness uses
`c.fzg` file size, which is 12 bytes larger — negligible for any real dataset.

---

## Error mode

| FZ-GPU mode | Canonical | eb basis |
|---|---|---|
| (only mode) | `rel_range` | `eb × (max − min)` |

Same basis as FZGM `NOA`, cuSZ `r2r`, cuSZp `rel`.

---

## Timing

Wall-clock via `std::chrono::system_clock`. The timers bracket the CUDA kernel launches
plus `cudaDeviceSynchronize()`. The `VERIFICATION` block (always compiled in) runs
**after** both timers close, so verification does not inflate reported times.

Wall-clock timing is less precise than CUDA-event adapters (extra variance from CPU
scheduling jitter). The harness `timing_reliable` flag (cv ≤ 0.15) still applies;
prefer `*_device_ms_min` for comparison.

---

## Source patch summary (`src/fz.cu`)

Five changes were made to the original source:

1. `runFzgpu()` signature: added `std::string compress_out` and `std::string decompress_out`
   (originally defaulted to `""`; now required params, see #5).
2. After compression sync: moved `offsetSum` memcpy here; added conditional compressed-file write
   (copies `deviceBitFlagArr`, `deviceStartPosition`, and `deviceCompressedOutput` to host, writes
   header + three arrays).
3. After decompression sync: added conditional decompressed-file write (copies `deviceDecompressedOutput`
   to host, writes via existing `write_array_to_binary()`).
4. `main()`: reads `argv[6]` and `argv[7]` as optional output paths.
5. **In-process repeat:** `runFzgpu()` no longer creates/destroys its own
   `cudaStream_t` — it now takes `cudaStream_t stream` as a parameter.
   `main()` creates one stream, reads an optional `argv[8]` repeat count
   (default 1), and loops the `runFzgpu()` call that many times sharing the
   one stream before destroying it. See "In-process repeat" above.

---

## Build

```bash
cd ~/research/compressors/FZ-GPU
# Edit Makefile SM if needed: nvcc ... -arch=sm_80 (A100)
make main
# binary: ./fz-gpu
```

Set `FZGPU_CLI` in `scripts/env-bigred200.sh` (already set to the project directory binary).
