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
- **One subprocess call per phase, `--repeat n_runs` reps in-process.** This
  build of `cuszhi` is patched (local, unmerged fork at
  `~/research/compressors/cuSZ-Hi`) to add a `--repeat N` flag: it loops the
  compress/decompress task N times inside one process, sharing one CUDA
  stream/context across reps, and prints N `(total)` rows to stdout. The
  adapter parses all N rows out of one process's output instead of spawning N
  subprocesses. No `-S write2disk` equivalent: each rep still writes
  `.cusza`/`.cuszx` (same file, overwritten each rep) — this adds file I/O
  overhead outside the timed CUDA-event window, so it doesn't contaminate
  `*_device_ms`.

### Fairness: why `--repeat` replaced the old N-subprocess loop

The old adapter spawned N separate `cuszhi` subprocesses per phase — each one
paying fresh CUDA-context creation, module load, and (clocks unlocked on
BigRed200) whatever clock state the GPU happened to be in. That's the same
cold-start failure mode `docs/adapters/fzgm.md` documents for FZGM itself
("a single-shot compress reported 31.7ms vs in-process benchmark min
0.93ms"), which is why FZGM has its own `-b --runs N`. Confirmed empirically
on CESM-2D/CLDHGH eb=1e-3: the old N-cold-process loop measured a stable
~1.5-1.8ms compress / (not measured for decompress in the old loop, but the
first rep of the new in-process loop — itself still cold — read 3.89ms
decompress); the new `--repeat 6` in-process loop reads rep 1 (cold, same
process but first CUDA work) at ~1.64ms compress / 3.89ms decompress, then
reps 2-6 settle to a stable ~0.70ms compress / ~0.81ms decompress — roughly
2.3x and 4.8x faster than what the old method was reporting as "the"
number. `warmup_reps` (dropped by `metrics.summarize_timing`) discards rep 1,
so the harness now sees the true warm numbers, matching how FZGM's own single
internal warmup rep is handled.

### Comparison note

The cuSZ adapter (patched `executor.cc`) emits a JSON line with
`compress_device_ms` covering only `psz_compress_float` GPU kernels.
cuSZ-Hi's `(total)` row covers all pipeline stages (same scope). They
should be comparable, but this has not been verified empirically. cuSZ's own
`--repeat` patch has since been fixed too (a use-after-free in
`psz_release_resource`, not a stream issue — see `docs/adapters/cusz.md`
"In-process `--repeat`"), so both adapters now measure via the same
in-process-warm method.

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
  decompressed files to disk on every rep (same file, overwritten). For
  large datasets this adds latency; accept higher CV or reduce
  `repetitions`. Now only one subprocess launch per phase regardless of
  `repetitions` (see `--repeat` above), so this is cheaper than it used to be.
- **Locale-dependent `%'f`:** timing table uses locale-aware digit
  grouping. The adapter strips commas, but if a non-English locale adds
  other separators, parsing may break. Force `LC_NUMERIC=C` in the job
  environment if this becomes an issue.
- **-R time only for benchmark:** compress() and decompress() do not
  pass `-R time`; they only get functional correctness (file output).

---

## FZGM pairing: 1-D data is structurally unsupported

Native cuSZ-Hi handles 1-D fields *inconsistently* with the `tp`/`cr` (spline)
presets — but there is no fair FZGM counterpart regardless, and this is **not a
preset bug**. Both FZGM cuSZ-Hi presets (`cusz_hi_tp.toml`, `cusz_hi_cr.toml`)
are built entirely around `GInterpStage` (the spline-interpolation predictor,
cuSZ-Hi's LC "Spline" path), and `GInterpStage::setDims` throws by design on
1-D input — confirmed in FZGM's own test suite (`tests/stages/test_ginterp.cpp`:
`EXPECT_THROW(s.setDims(1024, 1, 1), std::runtime_error)`). The spline predictor
needs a real 2-D+ neighborhood to interpolate against; there's no reduced 1-D
mode to fall back to inside `GInterpStage` itself.

Native cuSZ-Hi's CLI presumably dispatches to a different (non-spline, likely
Lorenzo) predictor internally for 1-D inputs — FZGM's cuSZ-Hi port only
implements the spline path, so it cannot express that fallback. Making this pairing
fair for 1-D would mean adding a genuinely different predictor to the FZGM
cuSZ-Hi pipeline (real upstream work), not tweaking a preset's dims. Until then,
`configs/experiments/fzgm_vs_native.yaml` scopes both FZGM cuSZ-Hi run entries
with `skip_datasets: [HACC]` — native cuSZ-Hi still runs against HACC (useful
reference data on its own), it just has no FZGM row to pair against.

**Update (JetStream2 H100, full `fzgm_vs_native.yaml` run, 2026-07-19):** native
cuSZ-Hi's 1-D handling is not just a different-predictor fallback — it's
outright unstable at some (preset, eb) combinations on HACC's 280,953,867-element
field, aborting with a C++ exception (exit -6) rather than producing a result:

| Preset | eb=1e-2 | eb=1e-3 | eb=1e-4 |
|---|---|---|---|
| `tp` | ok (CR 9.77) | ok (CR 5.08) | **crash** — `psz_gpu_exception`, "invalid argument" at `compressor.inl:272` |
| `cr` | **crash** — `std::runtime_error`, "exceeding max len: 27" | ok (CR 5.65) | **crash** — `psz_gpu_exception`, "invalid argument" at `compressor.inl:309` |

Only eb=1e-3 survived for both presets — not a clean "1-D unsupported"
failure mode, more likely internal array-length/quantization-level bookkeeping
in the spline path that happens to fit for some eb-derived parameter ranges and
overflow for others on a field this large. Recorded as `status: fail` rows in
`runs.jsonl` (not swallowed) — this is itself a legitimate robustness data point
for the native tool, not a benchkit bug. Not root-caused further; if you need
the `tp`/`cr` presets to run cleanly against 1-D data, this is where to start
digging in cuSZ-Hi's own source.
