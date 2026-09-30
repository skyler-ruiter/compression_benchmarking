# PFPL adapter contract

**Tool:** PFPL GPU compressor (LC-framework synthesis, Texas State University)  
**Adapter:** `benchkit/adapters/pfpl.py` · `PfplAdapter`  
**Compressor key in experiments:** `pfpl`

---

## Executables

PFPL builds one executable per (dtype × mode × direction). The adapter
uses the GPU variants from `<bin_dir>/<dtype>/gpu/`:

| Canonical mode | Exe (f32, compress) | Exe (f32, decompress) |
|---|---|---|
| `abs` | `f32_abs_compress_cuda` | `f32_abs_decompress_cuda` |
| `rel_range` | `f32_noa_compress_cuda` | `f32_noa_decompress_cuda` |
| `rel_maxabs` | `f32_rel_compress_cuda` | `f32_rel_decompress_cuda` |

Set `PFPL_BIN_DIR` to the `bin/` directory (e.g.
`~/research/compressors/PFPL/bin`), or pass `cli_path` in the run entry.

The adapter selects the corresponding executables under `bin/f64/gpu/` for
f64 fields.

---

## CLI

```
# Compress
f32_abs_compress_cuda input_file compressed_file error_bound [threshold]

# Decompress
f32_abs_decompress_cuda compressed_file decompressed_file
```

The `threshold` argument (optional) losslessly preserves values at or
above it (sentinel-value protection). The adapter does not pass it; data
with sentinel values should be filtered or handled upstream.

---

## Error-mode semantics

| PFPL mode | Canonical | eb basis (`eb_abs =`) |
|---|---|---|
| ABS | `abs` | `eb` |
| NOA | `rel_range` | `eb × (max − min)` |
| REL | `rel_maxabs` | `eb × max(|data|)` (per-element approx) |

`NOA` is range-relative — same semantics as cuSZ's `r2r` and cuSZp's
`rel`. Use `rel_range` for cross-tool comparisons.

`REL` is approximate per-element (like FZGM's `REL`). See DESIGN §5.4.

---

## Timing

- **Method:** CUDA events (cudaEventElapsedTime → `0.001 × ms = seconds`).
- **Internally runs 9 iterations** (`NUM_RUNS = 9` hardcoded in source).
- **Per-run output:** `lc comp ecltime,  X.XXXXXXXXX` (seconds) ×9 for
  compress; `lc decomp ecltime,  X.XXXXXXXXX` (seconds) ×9 for decompress.
- The adapter converts seconds → ms and returns all 9 values per phase.
- **n_runs is advisory** — PFPL always produces 9 values. Set
  `warmup_reps ≤ 8` in experiment configs; the runner discards leading
  values before computing statistics.
- The timing covers only the GPU kernel (encode/decode), not file I/O.

---

## Output files

- **Compressed:** written to `<workdir>/c.pfpl` after all 9 timing runs.
- **Decompressed:** written to `<workdir>/d.bin` (compress()) or
  `<workdir>/d_bench.bin` (benchmark()) after all 9 timing runs.

PFPL writes no compressed-size header; compressed size is measured from
file size. The compressed format is opaque (LC-framework bitstream).

## FZGPUModules correspondence

The canonical `configs/pipelines/pfpl.toml` preset reproduces the native f32
quantization representation, including the `2^22` code radius, raw-float
in-place escapes, and the post-quantization reconstruction check. Its staged
and chunk-specialized paths deliberately preserve native's float operation
order (`0.5f / eb`, `roundf`, and a separately rounded reconstruction
multiplication); allowing CUDA to contract the reconstruction into an FMA
changes which boundary values escape.

Focused checks on the two CESM-2D fields that formerly exposed opposite size
extremes now produce byte-identical reconstructed fields in native PFPL and
both FZGPUModules execution modes. FZGPUModules archives remain approximately
0.1% larger because FZM framing and RZE's 32-bit per-chunk size entries differ
from native PFPL's container and 16-bit entries. This is reconstruction and
algorithmic correspondence, not archive byte identity.

The current FZGPUModules preset is f32-specific. Native PFPL's f64 path uses
64-bit quantization codes, `DIFFNB_8`, and `BIT_8`; simply rendering the f32
graph with a double input does not reproduce that pipeline.

---

## build

```bash
# Update SM in Makefile to match GPU (80 for A100, 86 for RTX4090)
cd ~/research/compressors/PFPL
make all
```

Build artifacts land under `bin/<dtype>/<backend>/`.

---

## Known quirks

- **No machine-readable output:** size and ratio are printed as text;
  the harness ignores tool-printed values and measures from files.
- **9 reps fixed:** warmup is implicit in PFPL's loop. The first rep is
  the coldest; setting `warmup_reps: 1` in experiments is recommended.
- **No repeat-mode flag:** cannot request more or fewer reps from CLI.
- **f32 and f64 both work.** (This note previously read "f32 only … extend
  `_MODE_MAP` and `_check_exe` if needed" — that was stale. `_exe()` already
  builds `<bin_dir>/<dtype>/gpu/<dtype>_<mode>_<direction>_cuda` from
  `spec.field.dtype`, so f64 needs no code change, only the f64 executables
  present under `bin/f64/gpu/`. Exercised by MIRANDA, the one f64 dataset in
  `fzgm_vs_native_full.yaml`.) The class docstring's "(f32 only)" was stale for
  the same reason.
- **`benchmark()` cleans up its own scratch (D26).** The benchmark phase writes
  `d_bench.bin`, a full-original-size decompressed file, which is *not* covered
  by `retain_decompressed` — that setting governs the separate `decompress()`
  call's output. It is now unlinked in a `finally`. Note the asymmetry: `c.pfpl`
  is deliberately **not** deleted there, because the runner reads it after
  `benchmark()` returns and applies `retain_compressed` to it itself; deleting it
  in the adapter fails the cell with a missing-`c.pfpl` `FileNotFoundError`.
  Before this fix a 210-cell session left 4.99 GB of `d_bench.bin` behind, and a
  full-corpus sweep would have left ~110 GB.
