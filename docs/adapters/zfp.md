# zfp adapter contract

**Tool:** zfp (fixed-rate / fixed-precision / fixed-accuracy compressor)
**Adapter:** `benchkit/adapters/zfp.py` · `ZfpAdapter`
**Compressor key in experiments:** `zfp`
**Status:** ✅ Functional — CPU-only for error-bounded modes (see below).

---

## CLI

```bash
zfp -f|-d -1|-2|-3|-4 <dims...> -i <input> -z <compressed> -a <tolerance> -x serial -h
zfp -z <compressed> -o <decompressed> -h -x serial
```

- `-f`/`-d`: single/double precision.
- Dims are fastest-first (`-2 nx ny` matches `a[ny][nx]`), identical to
  `FieldSpec.dims` — no reordering.
- `-h`: embeds type/dims/mode in the compressed file header, so decompress
  needs only `-z`/`-o` (no need to repeat dims/type).
- `-x serial`: **always** used by this adapter — see "CUDA limitation" below.

Set `ZFP_CLI` to the `zfp` binary path, or pass `cli_path` in the run entry.

---

## CUDA limitation (why this adapter is CPU-only)

zfp's `-x cuda` execution policy only implements **fixed-rate** compression.
Confirmed empirically on this machine:

```
zfp -f -3 200 100 300 -i r.f32 -z r.zfp -a 1e-3 -x cuda -s   # "compression failed"
zfp -f -3 200 100 300 -i r.f32 -z r.zfp -r 16     -x cuda -s   # works
```

Fixed-rate compression takes a bitrate, not an error bound, so it has no
mapping onto this harness's abs/rel_range/rel_maxabs model (which specifies a
bound and checks whether it was met). Building a rate-vs-error bisection
search to emulate error-bounded compression via fixed-rate was judged out of
scope. This adapter always runs `-x serial` (CPU) instead, so zfp participates
in the same CR/quality comparisons as the GPU tools, but its timing is not a
GPU-throughput data point — see "Timing" below.

---

## Error-mode semantics

| Canonical | zfp native | eb basis |
|---|---|---|
| `abs` | `-a <eb>` | `eb` |
| `rel_range` | `-a (eb × range)` *(emulated)* | `eb × (max − min)` |
| `rel_maxabs` | `-a (eb × maxabs)` *(emulated)* | `eb × max|x|` |

zfp has no native relative-error mode (only `-R` reversible, `-r` fixed-rate,
`-p` fixed-precision, `-a` fixed-accuracy/absolute). `rel_range`/`rel_maxabs`
are **emulated**: `prepare()` reads the input file itself
(`read_range_stats`, shared with the MGARD/SPERR adapters), computes
`max-min` or `max|x|`, multiplies by the canonical `eb`, and passes that as
`-a`. The harness's own independent quality check reads the same file and
computes the same statistic, so the emulated bound and `eb_ok` agree.

---

## Timing

zfp prints only a one-line size/rate/error summary — **no elapsed time**.
Timing is measured externally with `time.perf_counter()` around each
subprocess call, so it includes process startup, not just the compression
work. Combined with the CPU-only constraint above, treat zfp's throughput
numbers as informational, not a precise or GPU-comparable measurement.

`benchmark()` makes N separate subprocess calls per phase (zfp has no
in-process repeat flag).

---

## Build

```bash
cd ~/compressors/zfp && mkdir -p build && cd build
cmake .. -DZFP_WITH_CUDA=ON -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
# binary: build/bin/zfp
```

`-DZFP_WITH_CUDA=ON` is only needed if you also want the fixed-rate CUDA path
for other purposes — this adapter never invokes it.
