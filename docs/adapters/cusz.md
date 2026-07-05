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

`benchmark()` makes one subprocess call per phase, each looping `--repeat
n_runs` reps inside the binary, sharing one CUDA stream/context across reps
(see "In-process `--repeat`" below) — mirrors FZGM's `-b --runs N` and
cuSZ-Hi's `--repeat`. `-S write2disk` skips disk writes during timing (GPU
kernels only, no PCIe D2H). The compressed `.cusza` file produced by
`compress()` is reused for all decompress timing reps (not overwritten).

Warmup semantics: same as fzgm — the first `warmup_reps` timing values are
discarded before computing the median.

### In-process `--repeat`: fairness fix and the bug it took to get there

This adapter used to run N cold subprocesses per phase — each paying fresh
CUDA-context creation, module load, and whatever clock state the GPU
happened to be in (clocks are unlocked on BigRed200), the same cold-start
failure mode `docs/adapters/fzgm.md` documents for FZGM itself. A local
patch (unmerged, `~/research/compressors/cuSZ`) adds `--repeat N`: `cli.cc`
loops the existing `psz_compress_task`/`psz_decompress_task` calls N times
in one process, sharing one CUDA stream across reps.

Getting there took two attempts:

1. **First attempt (insufficient):** just hoisting stream creation out of
   the per-call task functions into `cli.cc` (matching cuSZ-Hi's
   `dispatch()`, which creates one stream and reuses it). This still
   segfaulted deterministically on rep 3 of every trial (rep 2 silently
   produced no output — see next point).
2. **Actual root cause, found via debug prints + `compute-sanitizer`:**
   `psz_compress_task` (`executor.cc`) sets `m->cli = args->cli;` — an
   *alias* to the caller's `psz_cli_config`, not a copy. `psz_release_resource()`
   (`libcusz.cc:75`) unconditionally does `if (manager->cli) delete
   manager->cli;`. Harmless for a single-shot process (it exits right
   after), but fatal under `--repeat`: rep 1's cleanup deletes `args->cli`
   out from under the still-running process. Rep 2 then reads a
   just-freed pointer (glibc usually hasn't overwritten it yet, so it
   silently reads stale-but-plausible values — no crash, no output). By
   rep 3 that freed memory has been reused by later allocations, so
   reading/writing through it corrupts real state and segfaults. Fixed by
   setting `m->cli = nullptr;` right before `psz_release_resource(m)` in
   `psz_compress_task`, so release only frees what it actually owns.
   `psz_decompress_task` was never affected — its resource manager's
   `cli` field defaults to `nullptr` and is never aliased.

Verified with 20-rep and 5x-repeated-trial runs on CESM-2D/CLDHGH, both
phases, zero crashes after the fix (previously: deterministic segfault by
rep 3 on every trial). Compress settled to a stable ~0.43ms, decompress
~1.56ms (cv well under the 0.15 threshold). `warmup_reps` (dropped by
`metrics.summarize_timing`) discards the first (cold) rep, same as FZGM
and cuSZ-Hi.

---

## Known limitations

- `rel_maxabs` mode not available — raises `AdapterError` at `prepare()` time.
- No per-stage timing breakdown (`stages: []` in every row).
- `native_psnr` is always `null` (cuSZ does not output PSNR in JSON form).
- Device timing requires the patched `executor.cc` (CUDA events added); a stock
  cuSZ binary will fail `benchmark()` with an AdapterError about missing JSON.
- f64 (Miranda) not tested at time of writing — cuSZ supports f64 via `-t f64`.
