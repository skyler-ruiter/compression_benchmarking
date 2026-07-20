# SPERR adapter contract

**Tool:** SPERR (wavelet/SPECK-based CPU compressor)
**Adapter:** `benchkit/adapters/sperr.py` · `SperrAdapter`
**Compressor key in experiments:** `sperr`
**Status:** ✅ Functional — CPU-only, **2D/3D only**.

---

## CLI

```bash
sperr2d -c --ftype 32|64 --dims <nx> <ny>     --bitstream <out> --pwe <bound> <input>
sperr3d -c --ftype 32|64 --dims <nx> <ny> <nz> --bitstream <out> --pwe <bound> <input>

sperr2d -d --decomp_f|--decomp_d <output> <bitstream>
sperr3d -d --decomp_f|--decomp_d <output> <bitstream>
```

- `--ftype`: bit width (`32`/`64`), not a `f32`/`f64` string.
- `--dims`: fastest-first, identical to `FieldSpec.dims` — no reordering.
- The data volume/bitstream is a **positional** argument, given last.
- `--decomp_f`/`--decomp_d` selects the output precision on decompress.

Set `SPERR_BIN_DIR` to the directory containing both `sperr2d` and
`sperr3d` (same directory in this repo's build), or pass `cli_path`.

---

## Dimensionality: 2D/3D only

There is no 1D or 4D SPERR binary. The adapter picks `sperr2d` vs. `sperr3d`
from `len(field.dims)` and raises `AdapterError` for anything else:

```
SPERR only supports 2D or 3D data; got 1D: [2869440].
Exclude this dataset with skip_datasets/only_datasets.
```

Exclude HACC, EXAALT (1D) and QMCPACK (4D) from SPERR run entries with
`skip_datasets: [HACC, EXAALT, QMCPACK]` or the inverse `only_datasets:` —
the same pattern already used for cuszhi vs. HACC in `fzgm_vs_native.yaml`.

---

## Error-mode semantics

| Canonical | SPERR native | eb basis |
|---|---|---|
| `abs` | `--pwe <eb>` | `eb` |
| `rel_range` | `--pwe (eb × range)` *(emulated)* | `eb × (max − min)` |
| `rel_maxabs` | `--pwe (eb × maxabs)` *(emulated)* | `eb × max|x|` |

SPERR's only error-control mode is `--pwe` (point-wise error, absolute) —
the alternatives are `--psnr` (target PSNR) and `--bpp` (target bitrate),
neither of which is an error bound. `rel_range`/`rel_maxabs` are emulated the
same way as zfp/MGARD: `read_range_stats` computes range/maxabs from the
input file, multiplied by the canonical `eb` to get the `--pwe` value passed
to the tool.

---

## Timing

Neither `sperr2d` nor `sperr3d` report elapsed time — `--print_stats` prints
input range, bitrate, PSNR, and "Accuracy Gain", no timing. Timing is
measured externally with `time.perf_counter()` around each subprocess call
(includes process startup) — not comparable to the GPU adapters' device_ms.

`benchmark()` makes N separate subprocess calls per phase (no in-process
repeat support).

---

## Build

```bash
cd ~/compressors/SPERR && mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
# binaries: build/bin/sperr2d, build/bin/sperr3d
```

No CUDA toolchain needed — CPU-only (OpenMP via `--omp`, unused by this
adapter — always runs with SPERR's default thread count).
