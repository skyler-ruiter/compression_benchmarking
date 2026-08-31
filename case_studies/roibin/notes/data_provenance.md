# Light-source corpus for the ROIBIN study — provenance and preprocessing

Two independent serial-crystallography datasets, both LCLS / CSPAD, both 1480×1552
detector frames, both shipping their experiment's **own peak-finder output** so the
ROI definition is not something we invented from the pixel values.

Totals: **409 frames, 19,492 Bragg peaks.**

| corpus | frames | peaks | dtype | source |
|---|---:|---:|---|---|
| EXAFEL (SDRBench) | 130 | 13,837 | f32, calibrated | SDRBench ds131.2 |
| CXIDB ID 21 (5HT2B GPCR) | 279 | 5,655 | int16→f32, raw ADU | cxidb.org |

They are reported **separately and never pooled**: one is calibrated, the other is
raw ADU, and they come from different experiments, samples, and beamtimes.

---

## 1. EXAFEL — SDRBench

**Files** (already present on this machine, `/media/volume/Compression_Data/sdrbench_data/EXAFEL_130x1480x1552/`):

```
SDRBENCH-EXAFEL-data-130x1480x1552.f32     1,194,419,200 B   130 × 1480 × 1552 f32
SDRBENCH-EXAFEL-nPeaks.i64                         1,040 B   130 × int64
SDRBENCH-EXAFEL-peakXPosRaw-130x2048.d64       2,129,920 B   130 × 2048 f64
SDRBENCH-EXAFEL-peakYPosRaw-130x2048.d64       2,129,920 B   130 × 2048 f64
```

Upstream URL recorded in the local `metadata.yaml`:
`https://g-8d6b0.fd635.8443.data.globus.org/ds131.2/Data-Reduction-Repo/raw-data/EXAFEL/SDRBENCH-EXAFEL-130x1480x1552.tar.gz`

**Acquisition note (2026-08-09).** The other two SDRBench EXAFEL builds — the 4-D
`986x32x185x388` (8.5 GB) and `10x32x185x388` — could **not** be retrieved. That
Globus HTTPS collection now refuses anonymous access: directory listing returns
`LOGIN_DENIED` (`permission_denied`, identity does not map to a valid username for
the connector) and every file URL, including the one for the build we already have,
returns 404. So the 986-event build is unavailable from here without Globus
credentials. This is why the second dataset was taken from CXIDB instead.

### Peak metadata: the coordinate convention was verified, not assumed

`peakXPosRaw`/`peakYPosRaw` are f64 arrays padded to 2048 columns; only the first
`nPeaks[e]` entries of event `e` are valid (the tail is 0, which is a legal pixel
coordinate, so the count must be respected). Values are integral pixel indices —
`prep_exafel.py` asserts that, so a fractional convention cannot slip through as a
silent rounding bug.

The indexing order was determined empirically by comparing intensity at the claimed
peak positions against random pixels:

```
e0  k=108  A=frame[y,x] mean=1393.93  med= 961.6 | B=frame[x,y] mean= 53.59 med= 42.9 | random mean= 44.49 med= 37.9
e5  k=355  A=frame[y,x] mean=3893.49  med=2643.6 | B=frame[x,y] mean=244.77 med=161.1 | random mean=168.69 med=138.8
e12 k=264  A=frame[y,x] mean=4290.62  med=3070.2 | B=frame[x,y] mean=291.91 med=248.0 | random mean=238.76 med=203.0
```

**`frame[e][ y[e,i], x[e,i] ]` is correct**: 20–30× the frame median. The transposed
reading is statistically indistinguishable from random pixels. Getting this backwards
would have placed every ROI box on background and still produced a pipeline that
round-tripped and reported plausible ratios.

### Derived layout

`scripts/prep_exafel.py --frames` writes to
`/media/volume/Compression_Data/sdrbench_data/derived/EXAFEL_ROIBIN/`:

- `frames/exafel_f000.f32` … `f129.f32` — 130 single-event images, 1552×1480 f32,
  9,187,840 B each, byte-for-byte slices of the source volume (no rescaling).
- `peaks/exafel_f000.roi` … — per-frame peak list in the `.roi` format below.
- `peaks/exafel_full.roi` — all 13,837 peaks indexed by `z`, for the whole 3-D volume.
- `frame_stats.csv` — per frame: npeaks, min, max, mean, mean intensity at peaks.

Splitting the single registered 3-D field into 130 per-frame fields is the point:
the corpus previously carried EXAFEL as **one** field, which is what made it a
54-row slice rather than a domain study. Each frame is a genuinely independent
diffraction pattern with its own peak list. It remains **one experimental run from
one sample**, which is why the CXIDB set was added.

Per-frame statistics: npeaks 15 / 71.5 / 618 (min/median/max), value range roughly
[−1749, +14732], mean intensity at peak pixels 961–7287 vs frame means 16–476.

---

## 2. CXIDB ID 21 — 5HT2B GPCR serial femtosecond crystallography

**Citation.** Liu et al., *Serial femtosecond crystallography of G protein-coupled
receptors*, Science **342**:1521 (2013). CXIDB ID 21, DOI `10.11577/1169541`,
released **CC0**. LCLS, CXI instrument, CSPAD detector. PDB `4NC3`.

**Retrieved** from `https://www.cxidb.org/data/21/cxidb-21-run0010.tar` (32.08 GB).

**Partial-fetch disclosure.** The server does **not honour HTTP Range requests** — a
`Range: 0-262143` request returns the full stream. The transfer was cut off at
**2.57 GB**, which is a prefix of the tar, and the complete member files inside that
prefix were extracted. This yielded **280 per-event HDF5 files, 279 of which are
complete** (the 280th is truncated at the cut and is excluded by `prep_cxidb21.py`,
which simply lets the `h5py` open fail and skips it). The frames come from two runs:

- `cxidb-21-run0015` — 165 frames
- `cxidb-21-run0016` — 114 frames

These are Cheetah per-event outputs. Relevant datasets in each file:

```
data/rawdata0                          (1480, 1552)  int16    raw CSPAD frame
processing/cheetah/peakinfo-raw        (N, 4)        float64  Cheetah peak finder
processing/cheetah/peakinfo-assembled  (N, 4)        float64  assembled-detector coords
processing/cheetah/pixelmasks          (1480, 1552)  int8
```

### Peak convention: also verified, and one candidate rejected

`peakinfo-raw` columns are `(x, y, integrated_intensity, npix)`. Verified over all
5,655 peaks in the 279 frames:

| reading | mean | median | fraction > 1000 ADU |
|---|---:|---:|---:|
| `frame[c1, c0]` = `frame[y, x]` | 1615.0 | **1129.0** | **55.3 %** |
| `frame[c0, c1]` (transposed) | 250.6 | 148.0 | 2.0 % |
| `peakinfo-assembled[c0, c1]` | 142.6 | 0.0 | 2.8 % |
| random pixels | 321.3 | 149.0 | 7.0 % |

`frame[y, x]` — same convention as the SDRBench set. **`peakinfo-assembled` must not
be used**: it is in assembled-detector coordinates, does not index the raw frame, and
its median intensity of 0 shows it landing in inter-panel gaps.

### Preprocessing applied

`scripts/prep_cxidb21.py` writes to `derived/CXIDB21_ROIBIN/` with the same layout as
the EXAFEL corpus. Two deliberate choices:

1. **`int16` → `float32`, no rescaling.** The compressors under test are float codecs
   and the SDRBench build is f32, so widening keeps the arms comparable. Values stay
   raw ADU.
2. **No calibration.** No dark, gain, or common-mode correction is applied, so these
   frames are *not* equivalent to the calibrated SDRBench build. That difference is
   exactly why the two corpora are reported separately.

Per-frame peak counts: 15 / 18 / 59 (min/median/max) — a lower-multiplicity sample
than the EXAFEL set, which makes it a useful contrast: ROI coverage is smaller, so
the headroom for a dual-bound scheme is correspondingly smaller.

---

## 3. The `.roi` peak-list format

Little-endian, consumed by `ROIBinSplitStage::setPeaksFile()` and by
`scripts/validate_regions.py`:

```
magic   char[8]  "FZROI1\0\0"
nx      uint32   fast axis
ny      uint32   slow axis
nz      uint32   frames
npeaks  uint32
records npeaks × { uint32 z; uint16 x; uint16 y; }    (8 bytes each)
```

Records are sorted by `(z, y, x)` so the file is byte-deterministic for a given input.
The stage validates every record against the field bounds at load time and refuses a
file whose `nx/ny/nz` disagree with the pipeline's dimensions — a mismatch there would
silently relocate every ROI box.

The peak table is re-emitted on the stage's `peaks` output port, so it is stored
**inside the `.fzm` archive and counted in the compressed size**. At 8 B/peak this is
≈0.01 % of a frame (618 peaks → 4.9 kB against 9.19 MB). The decompressor never needs
the `.roi` file.

---

## 4. Reproducing

```bash
python scripts/prep_exafel.py --frames
python scripts/prep_cxidb21.py --src <dir containing cxidb-21-run*/data1/*.h5>
```

`prep_exafel.py` needs only numpy; `prep_cxidb21.py` also needs `h5py`.
