# SZ3 (CPU) vs GInterp + Huffman + GPU-Zstd

**Experiment:** `configs/experiments/sz3_vs_ginterp_gpu_zstd.yaml`
**Session:** `sz3-vs-gpu`, JetStream2 H100, 2026-08-06. 84/84 cells ok, 0 failed.
**Question:** does the GPU pipeline land in the same CR/PSNR neighbourhood as the
CPU reference running the same algorithm?

SZ3's default `ALGO_INTERP` path is spline interpolation + quantization → Huffman →
Zstd. `configs/pipelines/ginterp_huffman_gpu_zstd.toml` is that structure built from
FZGM stages (GInterp → Huffman → GPULZ-split + Huffman/ANS), so this is the closest
CPU-to-GPU like-for-like available.

7 fields × 3 bounds (`rel_range` 1e-2/1e-3/1e-4) across CESM-2D, HURR, NYX.

---

## Headline: compare at matched PSNR, not at matched error bound

At a **matched error bound**, the GPU pipeline gets a geomean **0.409x** of SZ3's CR.
That number is misleading. It also gets **+2.0 dB median PSNR** — the two tools are
not at the same operating point. FZGM's GInterp quantizes more conservatively and
spends bits SZ3 does not, so it buys quality SZ3 did not ask for.

Comparing rate against distortion instead — CR at matched PSNR, log-interpolated
within each tool's measured range:

| | geomean CR ratio (GPU / SZ3) |
|---|---|
| matched error bound | 0.409 |
| **matched PSNR** | **0.557** |
| **matched PSNR, excluding NYX/baryon_density** | **0.799** |

Per field, at the midpoint of the overlapping PSNR range:

| field | PSNR | SZ3 CR | GPU CR | GPU/SZ3 |
|---|---|---|---|---|
| CESM-2D/CLDHGH | 68.3 | 15.75 | 12.55 | 0.80 |
| CESM-2D/PRECT | 71.1 | 43.09 | 38.32 | 0.89 |
| CESM-2D/TS | 72.1 | 56.80 | 46.89 | 0.83 |
| HURR/QVAPOR | 69.8 | 30.69 | 26.63 | 0.87 |
| HURR/U | 68.6 | 27.44 | 23.31 | 0.85 |
| NYX/temperature | 76.3 | 323.35 | 226.36 | 0.70 |
| **NYX/baryon_density** | 92.3 | **12303.83** | **706.22** | **0.06** |

**So: on 6 of 7 fields the GPU pipeline is within 11–30% of SZ3's rate-distortion
curve** — the port is faithful, and the remaining gap is ordinary back-end quality,
not an algorithmic defect. One field is not close at all, and it is worth its own
section.

Throughput is deliberately not compared here: SZ3 is CPU-only and its adapter
measures external wall clock including process startup (`docs/adapters/sz3.md`).

## Context rows (geomean CR over the same 21 cells)

| variant | CR | median PSNR |
|---|---|---|
| SZ3 (CPU) | 152.05 | 68.31 |
| GInterp → Huffman → **GPU-Zstd** | 62.14 | 72.15 |
| GInterp → Huffman → **LC chain** (`cusz_hi_cr.toml`) | 60.71 | 72.15 |
| Lorenzo → Huffman (`cusz.toml`) | 18.76 | 64.79 |

Two things fall out:

- **The interpolation predictor is worth 3.3x** over Lorenzo+Huffman (62.14 vs
  18.76) at higher PSNR. That is by far the largest single lever on this data.
- **GPU-Zstd beats the cuSZ-Hi LC back end by 2.4%** (62.14 vs 60.71) at an
  identical front end. Real but small — the back end is not where the CR is.

---

## NYX/baryon_density: a 17x gap with a diagnosable cause

SZ3 reaches 12,304x where the GPU pipeline gets 706x at matched PSNR. That is not
back-end polish; it is structural. The field:

```
range [0.058, 1.159e+05]
at rel_range 1e-2 -> 22 distinct quant codes over 134,217,728 elements
   the single most common code covers 99.999% of the field
   1,700 runs -> mean run length 78,952 elements (158 KB at uint16)
```

Two independent limits bind:

1. **Huffman's ≥1-bit-per-symbol floor.** A field that is 99.999% one symbol still
   costs 1 bit per element through a Huffman coder: 134M elements → **16.8 MB
   minimum**, before the GPU-Zstd back end sees anything. SZ3's total output for
   this cell is ~20 KB.
2. **GPULZ's 2048-byte independent chunks.** A 158 KB run spans ~77 chunks, none of
   which can reference another. CPU Zstd has a multi-megabyte window and collapses
   the whole run into a handful of tokens.

### Measured fixes (eb=1e-2, same field, CR)

| pipeline | CR |
|---|---|
| GInterp → **Huffman** → GPU-Zstd (current) | 855 |
| GInterp → **RLE** → Huffman → GPU-Zstd | 3,890 |
| GInterp → **RZE** → GPU-Zstd | 926 |
| GInterp → **RLE** → GPU-Zstd | **3,927** |
| GPULZ `chunk_size` 2048 → 4096 (either variant) | no change |
| *SZ3 reference* | *27,218* |

**Replacing Huffman with RLE is worth 4.6x on this field** and closes the gap from
32x to 6.9x. Huffman after RLE adds nothing (3,890 vs 3,927) — once runs are
collapsed there is little entropy left for it. Raising the GPULZ chunk size does
nothing, which is expected: 4096 is still three orders of magnitude below the run
length, so limit (2) needs long-range matching, not a bigger chunk.

The remaining 6.9x is limit (2). Closing it means either cross-chunk matching in
GPULZ or a dedicated long-run path; a bigger `chunk_size` will not do it.

### What this implies beyond one field

Highly-skewed, mostly-constant fields are not rare in this corpus (CESM-2D's
SFCLDICE and SFCLDLIQ are entirely zero — see `validity.py`'s degenerate-field
detector). A pipeline that hard-codes Huffman pays the 1-bit floor on every one of
them. An adaptive choice — RLE when the top symbol dominates, Huffman otherwise —
is a cheap, well-targeted win, and the measurement above says how much it is worth.
