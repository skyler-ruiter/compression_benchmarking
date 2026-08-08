# Broadened stage parity: do the FZGM-vs-nvCOMP claims survive more data?

**Experiments:** `configs/experiments/nvcomp_parity_raw_broad.yaml`,
`nvcomp_parity_codes_broad.yaml` · **Sessions:** `parity-raw-broad` (210 cells),
`parity-codes-broad` (675 cells) · H100, nvCOMP **5.3.0.16**, 2026-08-07 ·
**885/885 ok, 0 bit-exactness failures.**

`nvcomp-stage-parity.md` made strong claims — GPULZ beats LZ4 1.64x on ratio,
GPULZ+Huffman beats Deflate 1.62x, GPU-Zstd beats Zstd 1.26x — from **three fields
of one dataset at one error bound**. This is the check. Five datasets (CESM-2D,
HURR, NYX, MIRANDA **f64**, EXAALT 1-D), three fields each, and on the codes tier
three error bounds (NOA 1e-2 / 1e-3 / 1e-4): **59x more cells.**

Ratios below are geometric means of per-cell FZGM/nvCOMP, **>1 = FZGM ahead**.
Throughput is median **host** wall (D33). nvCOMP is at `chunk=16384` (D34).

---

## Verdict per claim

| claim (codes tier) | original | broadened | verdict |
|---|---|---|---|
| GPULZ vs **LZ4**, CR | 1.64x | **1.33x** | direction holds, magnitude overstated |
| GPULZ+Huf vs **Deflate**, CR | 1.62x | **1.73x** | confirmed, was conservative |
| GPULZ+Huf vs **GDeflate**, CR | 1.70x | **2.01x** | confirmed, was conservative |
| GPU-Zstd vs **Zstd**, CR | 1.26x | **1.39x** | confirmed, most robust of all |
| rANS vs **ANS**, CR | 0.85x | **0.84x** | confirmed — nvCOMP ahead |

**The headline survives.** On the input FZGM's coders were built for, the three
LZ-family pairings are ahead on ratio on 4 of 5 datasets and at all three error
bounds, at 3–6x the compress throughput. FZGM's GPU-Zstd wins the compression
ratio outright on **38 of 45** codes cells against every nvCOMP algorithm in the
library, Bitcomp included.

Four things the narrow experiment got wrong or could not see.

### 1. Every LZ-family ratio lead erodes as the error bound tightens

| pairing (codes) | 1e-2 | 1e-3 | 1e-4 |
|---|---|---|---|
| GPULZ vs LZ4 | 1.509 | 1.382 | **1.131** |
| GPULZ+Huf vs Deflate | 2.249 | 1.717 | **1.344** |
| GPULZ+Huf vs GDeflate | 2.823 | 1.964 | **1.473** |
| GPU-Zstd vs Zstd | 1.549 | 1.376 | **1.252** |

Monotonic in all four. A tighter bound widens and flattens the code distribution,
which is exactly the regime where GPULZ's short match window stops paying. The
original 1e-3 measurement sat in the middle, so it read as a stable number when it
is the midpoint of a trend. **Any claim of this form needs its error bound
attached**; at 1e-4 the GPULZ-vs-LZ4 lead is 1.13x, not 1.64x.

### 2. EXAALT breaks the GPULZ ratio lead

GPULZ vs LZ4 on codes by dataset: CESM-2D 1.537, NYX 1.608, HURR 1.420,
MIRANDA 1.370, **EXAALT 0.869**. 1-D molecular-dynamics data has no spatial
locality for Lorenzo to exploit, so its residuals stay wide and LZ4's longer window
wins. The lead is a property of *gridded* data, not of the coder.

### 3. GPULZ's raw-f32 decompress advantage is not real

The original reported GPULZ decompressing 1.32x faster than LZ4 on raw f32 — that
was CESM-2D only. Broadened: **0.83x**, i.e. LZ4 is faster, and it worsens with
field size (CESM 1.25, HURR 0.78, MIRANDA 0.56, NYX 0.38). Same for
GPULZ+Huffman vs GDeflate on raw (0.58x) and codes (0.70x). **FZGM's decompress
lead is real against Deflate and Zstd, and absent against LZ4 and GDeflate** — the
old table's single dataset happened to pick the favourable case.

### 4. The ANS ratio gap is worse than measured, and dataset-dependent

rANS vs ANS on codes: CESM-2D 1.019 and EXAALT 1.066 (FZGM *ahead*), but HURR
0.762, MIRANDA 0.701, NYX 0.735. The original CESM-only number (0.85x, from a
dataset where FZGM is at parity) **understated** the deficit on everything else.
The profile is otherwise exactly as described: nvCOMP compresses 1.3x faster,
FZGM decompresses 1.86x faster, consistently everywhere. This remains the one
pairing where NVIDIA is ahead, and it is the clearest target for work.

---

## Raw tier: ratio parity, throughput lead

| pairing | CR | compress | decompress |
|---|---|---|---|
| rANS vs **ANS** | 1.012 | 0.80x | **1.81x** |
| GPULZ vs **LZ4** | 1.000 | **4.83x** | 0.83x |
| GPULZ+Huf vs **Deflate** | 0.976 | **3.17x** | **2.65x** |
| GPULZ+Huf vs **GDeflate** | 0.996 | **2.99x** | 0.58x |
| GPU-Zstd vs **Zstd** | 0.964 | **2.52x** | 1.13x |

On raw field bytes every pairing is within 4% on ratio while FZGM compresses
2.5–4.8x faster. This is the same split the narrow experiment found and it holds
across all five datasets and both precisions: **GPULZ's short match window is a
liability on unstructured mantissa bytes and an asset on residuals.** FZGM never
wins raw-f32 CR outright — Bitcomp takes 7 of 15 cells and Cascaded 6.

## Bitcomp and Cascaded

Correctly typed (D38), these are the strongest nvCOMP entries on **raw f32**,
because they are the only ones that predict. On **quant codes** they lose to
FZGM's GPU-Zstd almost everywhere (7 of 45 cells to Bitcomp-sparse, all on NYX),
since their delta pass re-differences an already decorrelated stream.

Bitcomp's sparse algorithm is worth having: **1.072x** ratio over the default on
codes, better in 32 of 45 cells.

### Correction: "RLE hurts" was a one-field artefact

The scheme sweep behind D38 was run on CESM-2D/CLDHGH and found RLE hurting
monotonically, suggesting `rles=0`. **That does not generalise.** Across the
broadened sweep `rles=0` is a geomean **0.917x** of the default on raw and
**0.847x** on codes. The distribution is bimodal rather than centred — within 0.1%
on run-free fields, catastrophic on the rest:

| cell | default (`rles=2`) | `rles=0` |
|---|---|---|
| HURR / CLOUD (raw) | 5.81 | **1.30** |
| NYX-qcodes-1e-4 / baryon_density | 56.75 | **10.04** |
| CESM-2D-qcodes-1e-2 / TS | 14.47 | **4.59** |
| NYX-qcodes-1e-2 / baryon_density | 81.63 | **127.78** |

RLE is load-bearing wherever the data has runs, which CLDHGH does not. The `bp=0`
and dtype findings from the same sweep did generalise. **A knob sweep on one field
establishes that a knob matters, never which setting to standardise on** — sweep
it on a field chosen for the property the knob exploits.

---

## What this does not establish

These are **back-end ratios on quant codes**, not compression ratios for a field
(D32) — `original_bytes` is the codes stream, and rows from the two tiers must
never be pooled. Nothing here is an end-to-end lossy comparison against nvCOMP;
that still needs a hybrid LorenzoQuant → nvCOMP round-trip measured against the
original field bytes at real PSNR.
