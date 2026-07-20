# MGARD adapter contract

**Tool:** MGARD / `mgard-x` (multilevel/wavelet GPU compressor)
**Adapter:** `benchkit/adapters/mgard.py` · `MgardAdapter`
**Compressor key in experiments:** `mgard`
**Status:** ✅ Functional — GPU (`-d cuda`).

---

## CLI

```bash
mgard-x -z -i <input> -o <compressed> -dt s|d -dim N <d_slow ... d_fast> \
        -em abs -e <bound> -s inf -l huffman|huffman-lz4|huffman-zstd -d cuda -v 2
mgard-x -x -i <compressed> -o <decompressed> -d cuda -v 2
```

- `-dt s|d`: single/double precision.
- **Dims are slowest-first** (`-dim N d_slow ... d_fast`) — the *opposite* of
  `FieldSpec.dims` (fastest-first). The adapter reverses them
  (`_dim_args`).
- `-l`: lossless back end. `pipeline: default` → `huffman`; `huffman-lz4` /
  `huffman-zstd` also valid pipeline strings.
- `-s inf`: **always** used by this adapter — see "Smoothness" below.

Set `MGARD_CLI` to the `mgard-x` binary path, or pass `cli_path` in the run
entry. **Also requires MGARD's `lib/` directory on `LD_LIBRARY_PATH`**
(`libmgard.so` + bundled nvcomp/protobuf/zstd) — set in
`scripts/env-jetstream2.sh`.

---

## Error-mode semantics

| Canonical | MGARD native | eb basis |
|---|---|---|
| `abs` | `-em abs -e <eb>` | `eb` |
| `rel_range` | `-em abs -e (eb × range)` *(emulated)* | `eb × (max − min)` |
| `rel_maxabs` | `-em abs -e (eb × maxabs)` *(emulated)* | `eb × max|x|` |

MGARD's native `-em rel` is **not** range- or maxabs-relative: it scales the
tolerance by an s-norm of the data computed in `CalculateNorm()`
(`Compressor.hpp`), which has no equivalent to this harness's canonical
bases. `rel_range`/`rel_maxabs` are instead emulated the same way as the zfp
adapter: read the input file (`read_range_stats`), compute range or maxabs,
and pass the product as an **absolute** bound via `-em abs`.

---

## Smoothness: `-s inf`, not `-s 0` (important correctness note)

The `-s` (smoothness) parameter also controls how the ABS tolerance is
distributed across decomposition levels
(`CalcQuantizers` in `Quantization/LinearQuantization.hpp`) — independent of
`-em`, i.e. it matters even in `abs` mode. This was found the hard way while
validating this adapter:

- With `-s 0`, `mgard-x` allocates per-level quantizers via the
  smoothness-weighted branch (`s != inf`), which does **not** bound the
  pointwise max error. Empirically, on CESM-2D/CLDHGH and HURR/TC at
  `eb=1e-3` (rel_range), the realized max|error| ran **~1.7x over** the
  nominal bound (`err_over_bound` 1.72–1.75), even though MGARD's own printed
  "Absolute L_2 error" line said "Satisfied" — that's a different norm than
  this harness's max-error `eb_ok` check.
- With `-s inf`, `mgard-x` takes the `s == infinity` branch, allocating
  `abs_tol / ((l_target+1) * (1 + 3^D))` per level — this does bound the
  pointwise max error. MGARD's own report then prints "Absolute L_inf error"
  and satisfies it with margin (e.g. `3.9e-05` realized vs. `1e-3` nominal on
  CLDHGH — MGARD's L∞ guarantee is quite conservative, so expect higher PSNR
  / lower CR than tools with a tighter pointwise guarantee at the same
  nominal bound).

This adapter always passes `-s inf`. If you ever see `eb_ok=False` with a
consistent, large `err_over_bound` on an MGARD row, check this first.

---

## Timing

`mgard-x -v 2` prints many `[time]` phase lines per call. This adapter uses
**"Aggregated low-level compression/decompression time"** — GPU kernel +
memory work, excluding CUDA context init ("Prepare device environment") and
serialization framing. This is the least-contaminated number mgard-x prints,
but:

- There is **no in-process repeat** — every subprocess call pays a fresh
  CUDA context init (tens to hundreds of ms observed here), unlike
  cuSZ/FZGM/FZ-GPU where that cost was eliminated with a source patch (see
  `cusz.md`/`fzgpu.md`). Not attempted for MGARD. Expect more run-to-run
  variance than those adapters; `warmup_reps` does not remove this cost since
  every rep is an independent cold process.
- `compress()` always runs an **internal decompress-for-verification**
  (to print its own L∞/PSNR line), which prints a second, later set of
  decompression timing lines in the same stdout. The adapter's regex takes
  the *first* "Aggregated low-level compression time" match, which precedes
  the verification pass — this does not contaminate the reported compress
  timing, but be aware extra GPU work happens inside every `compress()` call
  regardless.

---

## Known limitation: large 1-D arrays

`mgard-x` ran out of GPU memory on HACC's 280M-element (1.1 GB) 1-D field on
an 80 GB H100:

```
GPUassert: out of memory .../RuntimeX/DeviceAdapters/DeviceAdapterCuda.h 811
```

3-D fields with a similar or larger total element count (HURR, NYX) compress
fine, so this looks specific to the 1-D decomposition path — plausibly
padding to the next power of 2 (280,953,867 → 536,870,912, ~1.9x) combined
with several full-size intermediate buffers (original/decomposed/quantized/
norm arrays), rather than a fundamental capacity limit. Not investigated
further (would require reading `mgard-x`'s 1-D `Hierarchy`/`Decompose` memory
estimation code). Exclude very large 1-D fields with `skip_datasets` until/
unless this is root-caused — see `configs/experiments/smoke-cpu-refs.yaml`.

---

## Build

This machine has a pre-built `install-cuda-hopper` tree
(`~/compressors/MGARD/build-cuda-hopper` + `install-cuda-hopper`), built via
`build_scripts/build_mgard_cuda_hopper.sh` (sm_90). See that script for the
full CMake invocation (bundles nvcomp/protobuf/zstd as submodule builds).
