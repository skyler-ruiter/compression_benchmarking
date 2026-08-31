# ROIBIN-SZ-style dual-error-bound pipeline in FZGM (experiments.md E1 / F1 study 1)

A region-of-interest-aware compression pipeline for serial-crystallography detector
frames: Bragg-peak regions get a tight error bound, background gets a loose one, in
**one pipeline and one pass**. No monolithic GPU compressor applies two error bounds in
one pass — expressing it is the point of the study.

Start with **`RESULTS.md`**. The two supporting documents are
`notes/data_provenance.md` (where the data came from and what was done to it) and
`notes/design_and_method.md` (why the pipeline is shaped this way and what the numbers
may and may not be used to claim).

## Layout

```
configs/     pipeline TOMLs (the artifact's headline object)
scripts/     data prep, sweep driver, per-region gate, analysis
results/     sweep CSVs + JSON  (raw, one row per frame x arm)
notes/       provenance and method
RESULTS.md   the tables
```

### Configs

| file | what it is |
|---|---|
| `roibin_dual_eb.toml` | the annotated reference pipeline, `bin_factor = 1` |
| `roibin_sweep_b1.toml` | same, with `EB_ROI`/`EB_BG`/`PEAKS_FILE` placeholders for the driver |
| `roibin_sweep_b2.toml` | `bin_factor = 2` — the ROIBIN configuration |
| `roibin_volume_b2.toml` | `bin_factor = 2` over the whole 130-frame volume (exercises `nz > 1`) |
| `baseline_cuszp3.toml` | cuSZp3 at a single global bound, throughput reference |

### Scripts

| file | what it does |
|---|---|
| `prep_exafel.py` | SDRBench EXAFEL → 130 per-frame fields + `.roi` peak lists |
| `prep_cxidb21.py` | CXIDB ID 21 Cheetah HDF5 → 279 frames + `.roi` peak lists |
| `peak_profile.py` | measures the radial peak profile — justifies `roi_half_width` |
| `tune_cuszhi.py` | sweeps cuSZ-Hi's 8-point tuning grid, picks one setting per corpus |
| `run_sweep.py` | runs every arm over a corpus, validates per region, writes CSV |
| `validate_regions.py` | the per-region gate: ROI vs background error and PSNR |
| `selection_bias_check.py` | does the baseline's failure set bias the comparison? |
| `summarize.py` | CSVs → the results, verification and paired-comparison tables |

## Reproducing

Needs an H100-class GPU, `fzgmod-cli` from a Release build, native `cuszhi` and PFPL,
and a Python env with numpy (+ h5py for the CXIDB prep).

```bash
# 1. Build the corpora (see notes/data_provenance.md for sources)
python scripts/prep_exafel.py --frames
python scripts/prep_cxidb21.py --src <dir with cxidb-21-run*/data1/*.h5>

# 2. Choose the baseline's configuration on this data, once per corpus
python scripts/tune_cuszhi.py --root <corpus> --eb 10.0

# 3. Sweep. Arms: roibin_b1, roibin_b2, cuszhi, pfpl, cuszp3
python scripts/run_sweep.py --dataset EXAFEL --root <corpus> \
    --eb-roi 10.0 --eb-bg 100.0 --cuszhi-tuning spline/cr-first/cr \
    --out results/exafel_eb10_100.csv

# 4. Tables
python scripts/summarize.py results/*_eb10_100.csv --split-by-dataset --paired
```

The GPU must be **quiet**: a concurrent sweep silently corrupts throughput (it happened
during this study and cost an EXAFEL re-run). CR, PSNR and the bound checks are
deterministic and survive contention.

## Three things to read before quoting a number

1. **`bin_factor > 1` is a resolution reduction, not an error bound.** The `roibin_b2`
   background error reaches ~1.1e4 against a nominal bound of 100. Background fidelity
   for that arm is reportable as PSNR only; the gate records `n/a`. The `roibin_b1` arm
   is the one whose background genuinely satisfies its stated bound. Both are reported
   because they answer different questions.
2. **The ROI bound is the invariant**, and it holds in every arm and every
   configuration — that is what the verification table exists to show.
3. **The bound pair was forced by the baseline, not chosen for effect.** Native cuSZ-Hi
   aborts on 84/130 EXAFEL frames at ABS eb=1.0 and no tuning rescues it, so
   `eb_roi = 10.0` is where every baseline runs on every frame. See
   `notes/design_and_method.md` §4.
