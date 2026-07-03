# MANS adapter contract

**Tool:** MANS (Multi-dimensional Adaptive Non-uniform Superposition)  
**Adapter:** `benchkit/adapters/mans.py` · `MansAdapter`  
**Compressor key in experiments:** `mans`  
**Status:** ⚠️ Stub — all interface methods raise AdapterError. Integration design below.

---

## Why MANS requires special handling

MANS is a **lossless integer compressor**, not a direct error-bounded float
compressor. Its CLI:

```bash
nv_mans_compress <-u2|-u4> <input_file> <output_file> [--mode p|r] [--dims x [y z]]
nv_mans_decompress <-u2|-u4> <compressed_file> <output_file> [--dims x [y z]]
```

- `-u2` = uint16, `-u4` = uint32 (not float32/float64)
- No error bound parameter — MANS is lossless on its integer input
- The compressed output is a MANS bitstream
- The decompressed output is the reconstructed integers (identical to input)

To use MANS as an **error-bounded float compressor**, an external quantization
wrapper is needed:

```
float array  →  [quantize at eb]  →  integer array  →  MANS compress  →  bitstream
bitstream  →  MANS decompress  →  integer array  →  [dequantize]  →  float array
```

---

## Quantization wrapper design

### Option A: Uniform midtread quantizer (simplest)

For `abs` mode with error bound `eb`:
```
quant_int = round(float_value / (2 * eb))   # integer in range [~-32768, 32767]
float_reconstructed = quant_int * (2 * eb)
max_abs_err ≤ eb  (by construction)
```

This is how Lorenzo predictors work internally in cuSZ/FZGM. The quantized
integers fit in u16 (uint16_t) for most scientific data with typical bounds.

For `rel_range` mode: compute `eb_abs = eb * range`, then apply the abs quantizer.

### Option B: Lorenzo delta quantization

Apply 1D/2D/3D Lorenzo differencing before quantization (captures spatial
correlation, improves MANS CR). This is how cuSZ uses MANS internally.

---

## CLI

Once the wrapper is implemented:

```bash
nv_mans_compress -u2 <quant_int16_file> <compressed_file> --mode p [--dims x [y z]]
nv_mans_decompress -u2 <compressed_file> <recon_int16_file> [--dims x [y z]]
```

`--mode p` = prediction mode (better CR for structured data).
`--mode r` = residual mode.

### Timing

MANS currently prints **no timing information**. The `mans_api.cpp` timing
(via `std::chrono`) is only used internally by the autotune function, not
exposed in the CLI. To time MANS:
- Wrap the `nv_mans_compress` invocations with the harness wall-clock timer
  (low precision) — OR —
- Add CUDA event timing to `nv/nv_mans_compress.cpp` around the
  `mans::compress(...)` call and print `mans comp device_ms X.XXX`.

---

## Data type and dim convention

MANS accepts `u2` (uint16) or `u4` (uint32). For scientific float data:
- u16 quantization requires the dynamic range fits in ±32767 quant bins.
  At `eb = 1e-3`, a field with range 1.0 has 500 bins on each side — well
  within u16. At `eb = 1e-5`, that's 50000 — also fine.
- If the data range / (2*eb) > 32767, use u32 or reduce the error bound.

`--dims x [y z]` follows MANS's own convention: `x` is the first dimension
(innermost in MANS's spatial processing). Check the MANS documentation for
whether this is fast-to-slow or slow-to-fast.

---

## Build

```bash
cd ~/research/compressors/MANS
cmake -S . -B build \
  -DTARGET_PLATFORM=cpu_nv \
  -DCMAKE_CUDA_COMPILER="/N/soft/sles15sp6/cuda/gnu/12.6/bin/nvcc" \
  -DBUILD_HDF5_PLUGIN=OFF \
  -DCMAKE_CUDA_ARCHITECTURES=80 \
  -DCMAKE_CXX_COMPILER=$(which g++) \
  -DCMAKE_C_COMPILER=$(which gcc)
cmake --build build -j8
# binaries: build/bin/nv/nv_mans_compress, nv_mans_decompress
```

Set `MANS_CLI` to the `nv_mans_compress` binary path.
