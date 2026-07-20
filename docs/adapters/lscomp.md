# lsCOMP adapter contract

**Tool:** lsCOMP (GPU compressor for unsigned integers, light-source data)
**Adapter:** `benchkit/adapters/lscomp.py` · `LscompAdapter`
**Compressor key in experiments:** `lscomp`
**Status:** 🛑 Stub — same status as `mans` (see `docs/adapters/mans.md`).

---

## Why this is a stub

lsCOMP's CLI operates on `uint32`/`uint16` data with two lossy knobs:

```bash
lsCOMP_uint32 -i <input> -d <dims.x dims.y dims.z> \
              -b <bins.x bins.y bins.z bins.w> -p <pooling threshold> \
              [-x <compressed>] [-o <decompressed>]
```

- `-b x y z w`: per-level adaptive scalar quantization bins (`x <= y <= z <=
  w`; `-b 1 1 1 1` disables quantization).
- `-p`: selective-pooling threshold (`-p 1` disables pooling).

Neither knob is an error bound in the abs/rel_range/rel_maxabs sense this
harness uses, and the input is integer, not the float32/float64 SDRBench
fields this harness benchmarks. To use lsCOMP here, a quantization pre-step
(float → uint32/16 at a target error bound, decompress → dequantize
post-hoc) would be needed, with `eb_ok` attributed through that quantization
step rather than checked directly against lsCOMP's own output — the same
shape of problem as MANS (see `docs/adapters/mans.md`), and not attempted for
the same reason: it doesn't map onto the single-adapter
prepare/compress/decompress/benchmark model the other tools use without a
real design decision about where the quantization error budget goes.

`LscompAdapter` is registered (key `lscomp`) purely to reserve the slot and
raise a clear, actionable `AdapterError` from every interface method,
matching `MansAdapter`'s pattern exactly.

---

## What would need to change to support it

1. A quantization scheme mapping (canonical error mode, bound) → integer
   quantization parameters, with a documented error contribution from
   rounding at the chosen bin width.
2. A wrapper that quantizes the float field to `uint32`/`uint16` before
   calling `lsCOMP_uint32`/`lsCOMP_uint16`, and dequantizes the decompressed
   integers back to float before handing them to the harness's quality check
   — so `metrics.compute_quality` still operates on the tool's *contribution*
   to error, not the quantization step's.
3. A decision on which of `-b`/`-p` (or both) absorb the requested error
   budget, and whether that decision is exposed via the `pipeline:` field.

None of this is implemented. Set `LSCOMP_CLI` if you want `is_available()` to
resolve a real binary path (useful once someone picks up the wrapper design),
but every other adapter method will raise.

---

## Build (for reference — not required to use lsCOMP today)

```bash
cd ~/compressors/lsCOMP && mkdir -p build && cd build
cmake .. -DCMAKE_CUDA_ARCHITECTURES=90 && make -j$(nproc)
# binaries: lsCOMP_uint32, lsCOMP_uint16
```

### `sm_90` fix (2026-07-19)

`CMakeLists.txt` had an unconditional `set(CMAKE_CUDA_ARCHITECTURES 80 86)` —
the identical cache-shadowing bug found in cuSZp (D20 in `docs/DESIGN.md`): a
plain `set()` of this variable in CMakeLists.txt creates a normal variable
that shadows any cache value from `-DCMAKE_CUDA_ARCHITECTURES=90`, so the
actual build silently targeted only `sm_80`/`sm_86` no matter what was passed
on the configure line. Confirmed via `cuobjdump -lelf` before the fix (only
`sm_80`/`sm_86` cubins present). Fixed by adding `90` to the `set()` call
directly (`80 86 90`) and rebuilding from scratch; `sm_90` now present
alongside the other two. Not urgent for benchmark correctness today since
this adapter is a stub (see above — no real timing numbers are produced by
it yet), but worth having fixed before anyone builds out the quantization
wrapper.
