# cuSZ-Hi adapter contract

**Tool:** cuSZ-Hi (psz project, high-performance Lorenzo+Huffman variant)  
**Adapter:** `benchkit/adapters/cuszhi.py` · `CuszhiAdapter`  
**Compressor key in experiments:** `cuszhi`

---

## CLI

cuSZ-Hi uses the same flag style as cuSZ:

```bash
# Compress
cuszhi -z -i <input> -t f32 -l <dims> -m abs|r2r -e <eb> [-R time]

# Decompress
cuszhi -x -i <compressed.cusza> [-R time]
```

Set `CUSZHI_CLI` to the full path to the `cuszhi` binary, or pass
`cli_path` in the run entry.

---

## Error-mode semantics

| Canonical | Native flag (`-m`) | eb basis |
|---|---|---|
| `abs` | `abs` | `eb` |
| `rel_range` | `r2r` | `eb × (max − min)` |

`rel_maxabs` is not supported — raises `AdapterError`.

cuSZ-Hi's `r2r` is range-relative, identical to cuSZ's `r2r` semantics.

---

## Output file naming

cuSZ-Hi hardcodes output paths relative to the input file:
- **Compressed:** `<input>.cusza` (same directory as `-i` input)
- **Decompressed:** `<compressed_stem>.cuszx` (same directory as compressed)

The adapter symlinks the original data file into the workdir so that
both `.cusza` and `.cuszx` land inside the workdir.

---

## Timing

- **Method:** CUDA events summed over all pipeline stages, reported via
  `TimeRecordViewer` when `-R time` is passed.
- **Output format (text table, per run):**
  ```
  (c) COMPRESSION REPORT
    compression ratio X.XX

    kernel         time, ms     GiB/s
    pred           X.XXXX       XX.XX
    histogram      X.XXXX       XX.XX
    (subtotal)     X.XXXX       XX.XX
    book           X.XXXX       XX.XX
    (total)        X.XXXX       XX.XX
  ```
- The adapter parses the `(total)` row (ms column) for both compress and
  decompress. Locale thousand-separators (commas from `%'f`) are stripped.
- **n_runs subprocess calls** each yield one timing value per phase.
  No `-S write2disk` equivalent: each call writes `.cusza`/`.cuszx`.
  This adds file I/O overhead; use `*_device_ms_min` for throughput.

### Comparison note

The cuSZ adapter (patched `executor.cc`) emits a JSON line with
`compress_device_ms` covering only `psz_compress_float` GPU kernels.
cuSZ-Hi's `(total)` row covers all pipeline stages (same scope). They
should be comparable, but this has not been verified empirically.

---

## Build

```bash
cd ~/research/compressors/cuSZ-Hi
mkdir -p build
cmake -S . -B build \
  -DPSZ_BACKEND=cuda \
  -DPSZ_BUILD_EXAMPLES=off \
  -DCMAKE_CUDA_ARCHITECTURES="80" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_COMPILER=$(which g++) \
  -DCMAKE_C_COMPILER=$(which gcc)
cmake --build build -j8
# binary: build/cuszhi
```

---

## Known quirks

- **No write2disk skip:** benchmark() always writes compressed and
  decompressed files to disk on every call. For large datasets this
  adds latency; accept higher CV or reduce `repetitions`.
- **Locale-dependent `%'f`:** timing table uses locale-aware digit
  grouping. The adapter strips commas, but if a non-English locale adds
  other separators, parsing may break. Force `LC_NUMERIC=C` in the job
  environment if this becomes an issue.
- **-R time only for benchmark:** compress() and decompress() do not
  pass `-R time`; they only get functional correctness (file output).
