# MANS adapter

MANS is a lossless u16/u32 codec, so benchkit supplies the error-bounded float
transform:

```text
q = round((x - min(x)) / (abs_eb / 4))
x_hat = min(x) + q * (abs_eb / 4)
```

For relative modes, `abs_eb` is resolved from the field range or maximum
absolute value first. Ideal quantization error is at most one eighth of the
requested bound; the remaining headroom covers floating-point reconstruction
rounding at tight f32 settings. u16 is used when the realized code range fits,
otherwise u32. MANS codes those values
losslessly. A self-describing benchkit header stores dtype, dimensions, offset,
step, and codec settings, and its bytes count toward CR.

`pipeline: default` selects MANS mode `p`; `p` and `r` can also be requested
explicitly. The wrapper currently flattens fields. The installed
multidimensional MANS mapping segfaults on valid odd-shaped arrays, while the
1-D path round-trips the same integer sequence bit-exactly.

## Timing and installed backend

Timing is external wall clock over the actually runnable pipeline:

- compression: CPU quantization and q-file write + MANS CLI;
- decompression: MANS CLI + CPU dequantization and output write.

It includes process startup and intermediate I/O and is not comparable to a
native float compressor's CUDA-event device time. Both the device and host
columns carry this same end-to-end wall figure because the current result
schema requires a primary timing series; provenance makes the meaning explicit.

This machine intentionally uses `cpu_mans_compress`/`cpu_mans_decompress`.
The installed NVIDIA backend works for u16 but reproducibly crashes on u32
codes (`map_values_kernel_thrust ... illegal memory access`), exactly the code
width needed at `1e-6` and `1e-7`. Do not silently mix GPU u16 rows and CPU u32
rows in one sweep.

The CPU source also shipped with a u32 ADM capacity bug: it reserved three
signal bytes per uint32 even though the encoder can write four. Tight codes then
reported `adm_buf overflow` after already corrupting the heap. This machine has
the one-line capacity correction applied. Reproduce and rebuild it with:

```bash
bash scripts/build-mans-benchkit.sh
```

The source diff is archived at `patches/mans-u32-adm-capacity.patch`.

Environment:

```bash
export MANS_CLI=$HOME/compressors/MANS/build/bin/cpu/cpu_mans_compress
export MANS_DECOMPRESS_CLI=$HOME/compressors/MANS/build/bin/cpu/cpu_mans_decompress
```

The adapter is a standalone general baseline; no FZGM analogue is required.
