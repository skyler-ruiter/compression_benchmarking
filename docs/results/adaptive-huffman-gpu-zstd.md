# Adaptive Huffman codebooks in the GPU-Zstd pipelines

**Date:** 2026-08-06 · **GPU:** H100 80GB HBM3, JetStream2 `skyler-h100` ·
**Measured after** the `gather_kernel` fix (D35), which changed the profile enough
that earlier Huffman numbers on these presets do not carry over.

`HuffmanBookSource::Adaptive` histograms the first `compress()` call, builds one
codebook, and reuses it — removing a per-call histogram *and* the host-side
codebook build with its stream sync. `docs/results/new-stage-features-h100.md` §1
evaluated it on `cusz.toml` (Lorenzo→Huffman, where Huffman *is* the whole back
end) and found 1.14x geomean. This measures it where the question actually is: the
GPU-Zstd presets, where Huffman shares the work with GPULZ and ANS.

**Verdict: take it.** 1.09–1.18x compress depending on preset, ratio unchanged to
within 0.05% on the literals coder, decompress unaffected, and measured drift across
genuinely different fields is 0.19%.

---

## Measured, per preset (CESM-2D/CLDHGH, 25.9 MB)

| preset | Huffman shape | compress | speedup | CR |
|---|---|---|---|---|
| `gpu_zstd_lossless.toml` | literals, uint8/256 | 20.6 → 22.6 GB/s | **1.09x** | 1.11x → 1.11x (0.00%) |
| `gpu_zstd.toml` (lossy) | literals, uint16/4096 | 26.6 → 30.7 GB/s | **1.15x** | 10.87x → 10.87x (0.00%) |
| `gpu_zstd_codes.toml` | literals, uint16/4096 | 14.7 → 16.6 GB/s | **1.15x** | −0.05% |
| `ginterp_huffman_gpu_zstd.toml` | **two** Huffmans | 12.5 → 14.8 GB/s | **1.18x** | 13.39x → 13.33x (−0.4%) |

Decompress is unchanged everywhere (31.2 → 31.2, 29.7 → 29.9, 15.5 → 15.6 GB/s) —
as it must be, since the codebook travels in the stream and the decoder does no
book building. PSNR is identical in every cell (66.83 dB on the ginterp rows).

`gpu_zstd_codes.toml` is the preset that carries the nvCOMP head-to-head, so its
1.15x moves that result: FZGM's compress lead over nvCOMP Zstd (16 KB chunk, its
best) goes from **4.1x to ~4.8x**.

### The two-Huffman case is worth splitting

`ginterp_huffman_gpu_zstd.toml` has a *codes* Huffman (uint16/4096, over GInterp
quant codes) and a *literals* Huffman (uint8/256, over GPULZ output):

| | compress | CR |
|---|---|---|
| both `PerBlock` | 12.5 GB/s | 13.39x |
| codes Adaptive only | 13.3 (1.06x) | 13.33x (**−0.4%**) |
| literals Adaptive only | 13.8 (**1.10x**) | 13.39x (**0.00%**) |
| both Adaptive | 14.8 (**1.18x**) | 13.33x (−0.4%) |

**The literals coder is the free one** — more speedup than the codes coder *and* no
ratio cost. That is the sensible default. The codes Huffman's 0.4% is the price of
pinning a book to a data-dependent quant-code distribution; take it or not, but it
is a separate decision from the literals coder.

## The gain shrinks with input size

The per-call histogram and codebook build are fixed costs, so they amortise
(`gpu_zstd_lossless.toml`):

| field | size | PerBlock | Adaptive | speedup |
|---|---|---|---|---|
| CESM-2D/CLDHGH | 25.9 MB | 20.7 | 22.5 | **1.09x** |
| HURR/U | 100 MB | 26.8 | 27.6 | 1.03x |
| NYX/temperature | 536 MB | 31.8 | 32.4 | 1.02x |

Same shape as the `cusz.toml` evaluation, smaller magnitude — there Huffman was the
entire back end (1.45x at 25.9 MB); here GPULZ encode and three ANS stages carry
most of the work, so Amdahl caps what a Huffman-only change can return.

## Drift: the risk that matters, measured

Every number above re-compresses identical data, which is Adaptive's best case —
the book is always perfectly fitted. The real question is what a pinned book costs
across genuinely *different* inputs. Measured with
`build_profiling/bin/profiling/fzgmod-profile-huffman-drift`, 5 different CESM-2D
fields at `eb=1e-3` through one resident pipeline:

| step | PerBlock CR | Adaptive CR | loss |
|---|---|---|---|
| CLDHGH (book fitted here) | 13.80 | 13.80 | 0.00% |
| PRECT | 21.44 | 21.41 | 0.13% |
| TS | 21.56 | 21.56 | 0.02% |
| FLDSC | 21.85 | 21.80 | 0.26% |
| LHFLX | 21.08 | 21.00 | 0.35% |

**Mean 0.19%, worst 0.35%, 0 refits triggered.** Better than the 2.27% the earlier
round measured over 5 CESM-ATM fields — these five are more alike. `Fixed` on the
same sequence: **59.84%**, confirming it remains a trap.

**Caveat on scope:** this profiler builds a `LorenzoQuant → Huffman` pipeline, so
it characterises the *codes*-Huffman shape. The *literals* Huffman in the GPU-Zstd
presets sits downstream of GPULZ and codes LZ literal bytes, whose distribution has
good reason to be more stable across fields than raw quant codes — but that is an
argument, not a measurement. Extending the profiler to the split-mode shape is the
open item here.

## What this does not fix

Adaptive is a **throughput** lever. It does nothing for the ratio limitation found
in `docs/results/sz3-vs-ginterp-gpu-zstd.md`: on highly-skewed fields, Huffman's
≥1-bit-per-symbol floor costs 16.8 MB on NYX/baryon_density regardless of how the
book is built. That needs RLE (worth 4.6x there), not a better codebook.

## How to enable

Uncomment `book_source = "Adaptive"` on the literals Huffman — the presets carry the
line and the measured effect. It changes `pipeline_sha256`, so `benchkit stale` will
correctly flag the affected cells for re-measurement:

```bash
python -m benchkit stale <session>/ --stage Huffman
```
