# Stage-for-stage: FZGM coders vs their nvCOMP counterparts

**Experiment:** `configs/experiments/nvcomp_stage_parity.yaml` · **Session:**
`stage-parity-v2` · H100, nvCOMP **5.3.0.16**, 2026-08-06 · **54/54 cells ok,
all bit-exact**.

> The first run of this experiment (`stage-parity`) was 52/54 and had one row
> that was `ok` but not bit-exact. Both were real FZGM bugs on the same trigger
> — a non-chunk-multiple input into GPULZ — and both are now fixed (E22, E23).
> Ratios are unchanged (the compress side was never wrong); the GPULZ-vs-LZ4
> raw-f32 row is now n=3 rather than n=1.

The `nvcomp_vs_fzgm_*` experiments compare whole back ends. This one asks whether
each *individual* FZGM coder holds up against NVIDIA's implementation of the same
algorithm. Both sides run lossless on identical bytes; nvCOMP is at `chunk=16384`,
its measured-best on this GPU (D34).

Two inputs on purpose — raw f32 fields, and the derived Lorenzo quant codes FZGM's
coders were actually tuned on. Ratios are geometric means over 3 fields each;
throughput is median **host** wall (D33).

---

## Raw f32 fields (CESM-2D CLDHGH/PRECT/TS)

| pairing | FZGM CR | nvC CR | CR | compress | decompress |
|---|---|---|---|---|---|
| rANS vs **ANS** | 1.116 | 1.093 | 1.02x | **0.78x** | **2.17x** |
| GPULZ vs **LZ4** | 0.996 | 0.997 | 1.00x | **4.03x** | 1.32x |
| GPULZ+Huffman vs **Deflate** | 1.127 | 1.183 | 0.95x | **2.70x** | **2.69x** |
| GPULZ+Huffman vs **GDeflate** | 1.127 | 1.167 | 0.97x | **2.54x** | 0.73x |
| GPU-Zstd vs **Zstd** | 1.131 | 1.184 | 0.96x | **2.21x** | 1.79x |

## Lorenzo quant codes (the input FZGM's coders were built for)

| pairing | FZGM CR | nvC CR | CR | compress | decompress |
|---|---|---|---|---|---|
| rANS vs **ANS** | 5.133 | 6.008 | **0.85x** | **0.63x** | **2.23x** |
| GPULZ vs **LZ4** | 4.048 | 2.464 | **1.64x** | **10.68x** | **6.59x** |
| GPULZ+Huffman vs **Deflate** | 6.628 | 4.082 | **1.62x** | **3.89x** | 1.36x |
| GPULZ+Huffman vs **GDeflate** | 6.628 | 3.891 | **1.70x** | **3.88x** | 0.91x |
| GPU-Zstd vs **Zstd** | 8.144 | 6.460 | **1.26x** | **4.13x** | 1.46x |

---

## Reading it

**The LZ-family coders are the strong ones.** On quant codes GPULZ beats LZ4 by
1.64x ratio at 10.7x the compress throughput, and GPULZ+Huffman beats Deflate and
GDeflate by 1.6–1.7x ratio at 3.9x. On raw f32 the ratio advantage disappears
(FZGM trails Deflate by 5%) but the throughput lead holds. That split is the same
pattern the whole-back-end experiments show, and for the same reason: GPULZ's short
match window is a liability on unstructured mantissa bytes and an asset on
residuals.

**ANS is the outlier, and the one place NVIDIA is ahead.** It is the only pairing
where nvCOMP wins on *both* ratio and compress throughput:

| | FZGM ANS | nvCOMP ANS (best chunk) |
|---|---|---|
| raw f32, CR | **1.093** | 1.087 (@131072) |
| raw f32, compress | 92.0 | **108.8** |
| raw f32, decompress | **365.1** | 136.9 |
| quant codes, CR | 4.061 | **4.375** (@65536) |
| quant codes, compress | 52.8 | **79.2** |
| quant codes, decompress | **274.9** | 109.9 |

(CLDHGH alone, so nvCOMP can be shown at its ratio-best chunk rather than the
throughput-best 16384 used in the tables above — at 16384 its CR drops to 1.056 /
4.336 and the comparison would flatter FZGM.)

So, precisely:

- **Ratio is at parity on raw f32** (FZGM +0.6%) and **nvCOMP is ~8% ahead on quant
  codes** — the input that matters more for this library.
- **nvCOMP compresses 1.3–1.6x faster**, and that gap *widened* recently: 5.3.0.16
  improved nvCOMP ANS compress by 29–34% over 5.2.0.10 (see nvcomp.md).
- **FZGM decompresses 2.5–2.7x faster**, consistently, on both inputs.

FZGM's ANS is a dietGPU rANS port; the profile reads as decompress-optimised where
NVIDIA's is compress-optimised. If ANS throughput on the compress side ever matters
(it is the fastest coder in either library — 50–110 GB/s against GPULZ's 14–62), the
gap is worth a look. The ratio gap on quant codes is the more interesting one, since
ANS exists in these pipelines precisely to code the small skewed sequence streams
where Huffman's ≥1-bit floor hurts.

**No counterpart either way:** FZGM's standalone Huffman has no nvCOMP equivalent
(nvCOMP ships no bare Huffman), and nvCOMP's Bitcomp and Cascaded have no FZGM
equivalent. Neither is in the table.

## The two bugs this experiment found

Running GPULZ *alone* — rather than behind a predictor, as every other preset
does — put a non-chunk-multiple byte count into it for the first time, and that
shape was broken two different ways. Both are fixed in FZGM; see
`docs/experiments/observed_errors.md` (E22, E23) for the full write-ups.

- **E22** — GPULZ's `estimateOutputSizes()` bounded the output by the *input*
  size, but `execute()` zero-pads a partial tail chunk and encodes it as a full
  one, so the encode wrote 4–40 B past its buffer. Loud as a terminal stage
  (`cudaMemcpyAsync failed: invalid argument`, the 2 failed cells), **silent
  mid-pipeline**.
- **E23** — the inverse DAG *overrode* each sink buffer's size with the recorded
  uncompressed size, which is smaller than the padded extent GPULZ's decode
  kernel writes. The overrun landed on the compressed buffer and raced the other
  chunks reading it, so whole chunks decoded to zeros. This is the one that
  matters: it exited 0 and reported `status: ok`, and only the harness's own
  bit-exactness check caught it (`qcodes/PRECT`, 14,480 B across 372 of 6,329
  chunks).

**Nothing published elsewhere depended on either.** E23 only reaches a pipeline
whose *first* stage is GPULZ — that is `gpulz_only`, `gpulz_huffman`,
`gpu_zstd_lossless`, `gpu_zstd_codes`, all introduced for this comparison. The
lossy presets (`gpu_zstd.toml`, `ginterp_*`) start with LorenzoQuant/GInterp and
are structurally out of reach, so the full-corpus sweep is unaffected. And
because both bugs are decode-side or bounds-side, no compression *ratio* ever
changed — the CR columns above are identical to the first run's.
