# SZ3 adapter contract

**Tool:** SZ3 (CPU error-bounded lossy compressor)
**Adapter:** `benchkit/adapters/sz3.py` · `Sz3Adapter`
**Compressor key in experiments:** `sz3`
**Status:** ✅ Functional — CPU-only.

---

## CLI

```bash
sz3 -f|-d -i <input> -z <compressed> -1|-2|-3|-4 <dims...> -M ABS|REL <bound>
sz3 -f|-d -z <compressed> -o <decompressed> -1|-2|-3|-4 <dims...>
```

- `-f`/`-d`: float/double (switch flags, not value options).
- Dims are fastest-first (`-3 nx ny nz` matches `data[nz][ny][nx]`), identical
  to `FieldSpec.dims` — no reordering needed.
- Compress and decompress are separate invocations (no combined round-trip).
- Dims must be passed again on decompress — SZ3's own binary format doesn't
  carry them in this CLI's usage pattern (validated: decompress works with
  just the dims, no `-M` needed).

Set `SZ3_CLI` to the `sz3` binary path, or pass `cli_path` in the run entry.

---

## Error-mode semantics

| Canonical | Native (`-M`) | eb basis |
|---|---|---|
| `abs` | `ABS <eb>` | `eb` |
| `rel_range` | `REL <eb>` | `eb × (max − min)` |

SZ3 calls `REL` "value-range-based" (`VR_REL` in `-h2` SZ2-compat help) —
identical semantics to cuSZ `r2r`, cuSZp `rel`, FZGM `NOA`.

`rel_maxabs` has no native equivalent — SZ3's other modes are `PSNR`, `NORM`
(L2), `ABS_AND_REL`, `ABS_OR_REL`, none of which are a maxabs-relative bound.
Raises `AdapterError`.

---

## Pipeline

`pipeline:` is ignored beyond `"default"` — no config-file (`-c sz.config`)
support is wired up; all behavior comes from the canonical error mode/bound
and the field's own dtype/dims.

---

## Timing

SZ3 self-reports elapsed time directly:

```
compression time = 0.283263
decompression time = 0.093027 seconds.
```

This is the tool's own internal timer (CPU wall-clock around the algorithm,
not process startup). **Not comparable to the GPU adapters' device_ms** —
treat SZ3 as a CPU baseline for CR/quality, not a throughput peer.

`benchmark()` makes N separate subprocess calls per phase (no in-process
repeat flag exists). Process-launch overhead is negligible next to SZ3's own
compute time for the field sizes in this repo's datasets, unlike the
CUDA-context-init cost that made naive subprocess loops unusable for the GPU
adapters (see `fzgm.md` / `cusz.md`).

---

## Build

```bash
cd ~/compressors/SZ3 && mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
# binary: build/tools/sz3/sz3
```

No CUDA toolchain needed — CPU-only.
