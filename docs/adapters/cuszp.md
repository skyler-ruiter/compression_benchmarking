# cuSZp adapter contract (v2 and v3)

**Tool:** cuSZp (GPU error-bounded lossy compressor, fixed-length encoding)  
**Adapter:** `benchkit/adapters/cuszp.py` · `CuszpAdapter`  
**Compressor keys in experiments:** `cuszp2`, `cuszp3`

---

## CLI

**v2:**
```bash
cuSZp -i <input> -t f32|f64 -m plain|outlier \
      -eb abs|rel <bound> [-x compressed] [-o decompressed]
```

**v3 adds `-d`:**
```bash
cuSZp -i <input> -t f32|f64 -m plain|outlier|fixed \
      -d 1|2|3 [dz dy dx] \
      -eb abs|rel <bound> [-x compressed] [-o decompressed]
```

Set `CUSZP2_CLI` / `CUSZP3_CLI` to the respective binary paths, or pass
`cli_path` in the run entry. The two versions have separate binaries.

---

## Error-mode semantics

| Canonical | Native (`-eb`) | eb basis |
|---|---|---|
| `abs` | `abs <eb>` | `eb` |
| `rel_range` | `rel <eb>` | `eb × (max − min)` |

cuSZp's `rel` is **range-relative**: it internally computes `max−min` and
multiplies the error bound — same semantics as cuSZ's `r2r` and FZGM's
`NOA`. Use `rel_range` for cross-tool comparisons.

`rel_maxabs` is not supported (raises `AdapterError`).

---

## Pipeline string

The `pipeline:` field in an experiment run entry encodes the encoding mode
and (for v3) the processing dimension:

| Pipeline string | Encoding mode | Dim processing |
|---|---|---|
| `plain` | fixed-length plain | v2: implicit 1D; v3: `−d 1` |
| `outlier` | fixed-length with outlier | same |
| `fixed` | no-delta fixed-length | v3 only |
| `plain:2d` | plain | v3: `−d 2 1 dim_y dim_x` |
| `plain:3d` | plain | v3: `−d 3 dim_z dim_y dim_x` |
| `outlier:3d` | outlier | v3: `−d 3 dim_z dim_y dim_x` |

Dims for 2D/3D processing come from the `FieldSpec.dims` in fast-to-slow
order: `dims[0]` is fastest (x), `dims[1]` is y, `dims[2]` is z.
cuSZp v3 takes `dz dy dx` in slow-to-fast order.

**Default:** 1D processing. Use 2D/3D for better CR on structured data,
but verify the dim order matches the actual layout of each dataset.

---

## Round-trip model

cuSZp does **compress + decompress in a single invocation**. There is no
separate decompress-only binary.

- `compress()`: runs the full round-trip, saves `-x c.cuszp -o d.bin`.
- `decompress()`: returns the already-written `d.bin`. No subprocess.
- `benchmark()`: runs N invocations **without `-x`/`-o`** (cuSZp skips
  file writes when these are omitted — no disk I/O overhead per call).

---

## Timing

- **Method:** CUDA events (`cudaEventElapsedTime`), via `TimingGPU`.
- **Per invocation:** 10 GPU warmup iterations + 1 timed compress + 1 timed
  decompress. Single warm run ≈ `*_device_ms_min` behavior.
- **Printed output:** `cuSZp compression   end-to-end speed: X GB/s` where
  X is actually **MiB/ms** (mislabeled). The adapter recovers device_ms:
  ```
  device_ms = (original_bytes / 1024²) / X
  ```
- **n_runs subprocesses** each yield one timing value per phase. Set
  `warmup_reps: 0` in experiments (each run is already warm), or `1` to
  discard any clock-startup effects across subprocess boundaries.

---

## Build

```bash
# v2
cd ~/research/compressors/cuSZp-V2.0.1
cmake -S . -B build \
  -DCMAKE_CUDA_ARCHITECTURES="80" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_COMPILER=$(which g++) \
  -DCMAKE_C_COMPILER=$(which gcc)
cmake --build build -j8
# binary: build/examples/bin/cuSZp

# v3
cd ~/research/compressors/cuSZp-V3.0.0
# (same cmake flags as v2)
cmake --build build -j8
# binary: build/examples/bin/cuSZp
```

---

## Known quirks

- **Mislabeled throughput:** the binary prints `GB/s` but computes `MiB/ms`.
  Mixing this printed value with other tools introduces a ~7% error. The
  adapter always recovers raw ms and lets the harness compute throughput.
- **No decompress-only mode:** cuSZp always does the full round-trip. The
  adapter caches the decompressed file from compress() rather than re-running.
- **Single timed rep per subprocess:** unlike FZGM/PFPL, each cuSZp call
  yields one timing value. Set `repetitions: 5–10` in experiments.
- **1D default for v3:** multi-dim processing may improve CR but requires
  verifying the dim order convention per dataset.
