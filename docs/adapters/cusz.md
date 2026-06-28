# cuSZ Adapter Contract

Adapter class: `benchkit.adapters.cusz_ref.CuszAdapter`  
Compressor key in YAML: `cusz`  
Binary: resolved from `CUSZ_CLI` env var → `cli_path` in the run entry → `cusz` on PATH.

---

## What it wraps

[cuSZ / psz](https://github.com/szcompressor/cuSZ) — Lorenzo predictor + Huffman codec,
CUDA backend. Built from source on BigRed200; see `scripts/env-bigred200.sh` for the
build commands and binary path.

---

## CLI interface summary

```
cusz -z -i <input> -t <dtype> -l <dims> -m <mode> -e <eb> [-R cr]   # compress
cusz -x -i <input.cusza> [--compare <original>]                       # decompress
```

There is no `-o` output path flag for compression — the output is always
`<input>.cusza` next to the input file. The adapter works around this by symlinking
the data file into the workdir so the `.cusza` file lands there.

Decompressed output is written to `<compressed_stem>.cuszx`; the adapter renames
it to `d.bin` for the harness.

---

## Error mode mapping

| Canonical mode | cusz flag | Basis for eb-check |
|---|---|---|
| `abs` | `-m abs` | absolute |
| `rel_range` | `-m r2r` | `eb × (max−min)` |
| `rel_maxabs` | **not supported** | — |

`r2r` (range-relative) is cuSZ's default and maps directly to `rel_range` / NOA.

---

## Timing methodology

The `-R time` flag was originally stubbed out in cuSZ's CLI (`executor.cc`) with
`"Reporting time is disabled/to be updated"`. The cuSZ source used in this project
has been patched (see `scripts/env-bigred200.sh` build notes) to add CUDA event
pairs around `psz_compress_float/double` and `psz_decompress_float/double` and
emit a JSON line to stdout when `-R time` is passed.

**Benchmark runs pass `-R time` and parse the JSON line from stdout:**
```
{"compress_device_ms": 12.3456, "original_bytes": 1048576, "compressed_bytes": 52428}
{"decompress_device_ms": 8.2345}
```

**CUDA event placement:** `ev_start` is recorded on the stream immediately before
the compress/decompress library call; `ev_stop` immediately after. This excludes:
- PCIe H2D for the input data (cudaMemcpy before the library call)
- PCIe D2H for the output (skipped by `-S write2disk` during timing runs)
- File I/O, process startup

The result is comparable to fzgm's `cuda_event_device_only` timing.

The `provenance.timing_method` field is set to `"cuda_event_device_only"` for cusz rows.

---

## Benchmark mode

`benchmark()` runs:
- N compress subprocesses with `-S write2disk` (skip disk write — GPU kernels
  only, no PCIe D2H for compressed data)
- N decompress subprocesses with `-S write2disk` (skip writing `.cuszx`)

The compressed `.cusza` file produced by `compress()` is reused for all decompress
timing runs (not overwritten).

Warmup semantics: same as fzgm — the first `warmup_reps` timing values are
discarded before computing the median.

---

## Known limitations

- `rel_maxabs` mode not available — raises `AdapterError` at `prepare()` time.
- No per-stage timing breakdown (`stages: []` in every row).
- `native_psnr` is always `null` (cuSZ does not output PSNR in JSON form).
- Device timing requires the patched `executor.cc` (CUDA events added); a stock
  cuSZ binary will fail `benchmark()` with an AdapterError about missing JSON.
- f64 (Miranda) not tested at time of writing — cuSZ supports f64 via `-t f64`.
