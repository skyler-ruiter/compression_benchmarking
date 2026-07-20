# compression_benchmarking — Design & Roadmap

> Standardized benchmarking and analysis toolkit for GPU-accelerated, error-bounded
> lossy compressors (EBLCs). Built to compare reference compressors (cuSZ, cuSZ+,
> cuSZp, cuSZ-Hi, MANS, PFPL, …) against their modularized **FZGPUModules (FZGM)**
> ports across compression ratio, throughput, quality, and memory — and to support
> rapid, reproducible experiments for research papers.

Status: **M1 complete; M2 (HPC execution) in progress.** The `benchkit` package runs the
core loop on FZGM, TOML-first, with sharding/resume for clusters. This document is the
living contract. Decisions are recorded in the [Decision Log](#10-decision-log).

---

## 1. The headline question

FZGM re-implements and modernizes the kernels of several published GPU EBLCs as
composable pipeline stages. The central question this toolkit must answer cheaply and
repeatedly:

> **Does the FZGM port of compressor *X* roughly match the original *X* on compression
> ratio and quality, without losing too much speed — under identical datasets, error
> bounds, and error-mode semantics?**

Everything here is in service of producing a defensible, reproducible answer to that
question, and of making the next experiment (new stage, new dataset, new bound) a
config edit rather than a scripting project.

### Secondary goals
- Cross-compare *all* compressors against each other (rate–distortion, throughput),
  not just FZGM-vs-original pairs.
- Generate paper-ready tables and figures from a single results store.
- Capture enough provenance that any number in a paper can be traced to the exact
  binary, GPU, clocks, dataset checksum, and config that produced it.

### Non-goals (initially)
- Not a compressor. We orchestrate and measure; we do not implement compression.
- Not a CI gate for FZGM correctness (that lives in the FZGM repo's tests).
- Not a distributed/cluster scheduler. Single-node, single-GPU first; multi-GPU and
  job-array submission are later extensions.

---

## 2. Design principles

1. **The harness owns the metrics.** We do *not* trust each tool's self-reported CR,
   PSNR, or throughput. Tools differ in PSNR value-range conventions, whether timing
   includes PCIe transfers, and what they count as "size." The harness computes CR,
   bit-rate, PSNR, NRMSE, and error-bound satisfaction itself from raw artifacts
   (original bytes, compressed bytes, decompressed output). The *only* number we must
   accept from a tool is **device kernel time**, because that is the one thing the
   harness cannot observe from outside the process — and we record exactly how each
   tool measured it. See [§5 Metrics](#5-metrics-harness-owned).

2. **Normalize error-mode semantics before comparing.** "Relative" means different
   things across tools (relative to value-range vs. per-value vs. L∞). A comparison is
   only fair once both compressors are configured to the *same* effective bound. This
   normalization is explicit and recorded, not assumed. See [§5.4](#54-error-mode-normalization).

3. **Every result row carries full provenance.** A result is meaningless without the
   GPU, clocks, driver, commit SHA, and dataset checksum that produced it. Provenance
   is captured once per run-session and foreign-keyed into every row.

4. **Tidy, append-only results.** One row = one atomic measurement
   (compressor × dataset-field × error-bound × mode × repetition). Stored as JSONL
   (newline-delimited JSON), append-only, trivially loadable into pandas/polars. No
   in-place mutation; re-runs append and are disambiguated by `run_id`.

5. **Adapters isolate per-compressor messiness.** Each compressor is wrapped by an
   adapter implementing one interface. Adding a compressor = writing one adapter +
   one build script. The runner, metrics, provenance, and analysis layers never know
   which compressor they're driving.

6. **Reproducibility is a feature, not an afterthought.** Pinned commits (submodules),
   locked GPU clocks, warmup runs, repetition with spread reporting, recorded
   environment. A run should be re-creatable months later for a paper revision.

7. **Config over code.** A new experiment is a YAML file describing the run matrix.
   The Python is generic; the science is declarative.

---

## 3. Architecture overview

```
                    configs/experiments/*.yaml   configs/datasets.yaml
                                 │                        │
                                 ▼                        ▼
   ┌───────────────────────────────────────────────────────────────────┐
   │                            RUNNER                                   │
   │  expands run matrix → for each cell: warmup, repeat, collect        │
   └───────────────┬───────────────────────────────┬───────────────────┘
                   │ drives via uniform interface   │ records once/session
                   ▼                                ▼
        ┌─────────────────────┐            ┌──────────────────────┐
        │  ADAPTERS            │            │  PROVENANCE          │
        │  fzgm / cusz / cuszp │            │  GPU, clocks, driver │
        │  cusz-hi / mans /    │            │  CUDA, host, commits │
        │  pfpl  …             │            │  build flags         │
        │  compress()/decomp() │            └──────────┬───────────┘
        └──────────┬──────────┘                       │
                   │ artifacts (compressed, decompressed) + device times
                   ▼                                   │
        ┌─────────────────────┐                        │
        │  METRICS (harness)  │                        │
        │  CR, bitrate, PSNR, │                        │
        │  NRMSE, eb-check,   │                        │
        │  throughput, mem    │                        │
        └──────────┬──────────┘                        │
                   ▼                                    ▼
            ┌──────────────────────────────────────────────┐
            │  RESULTS STORE   results/<session>/runs.jsonl │
            │                  + provenance.json + logs/    │
            └───────────────────────┬──────────────────────┘
                                    ▼
            ┌──────────────────────────────────────────────┐
            │  ANALYSIS   rate–distortion, throughput bars, │
            │  FZGM-vs-reference delta report, LaTeX/figs   │
            └──────────────────────────────────────────────┘
```

---

## 4. Repository layout (target)

```
compression_benchmarking/
├── README.md
├── docs/
│   ├── DESIGN.md                  ← this document
│   └── adapters/                  ← per-compressor integration notes & quirks
│       └── <compressor>.md
├── configs/
│   ├── datasets.yaml              ← SDRBench dataset manifest
│   └── experiments/
│       ├── smoke.yaml             ← tiny end-to-end sanity matrix
│       └── fzgm_vs_reference.yaml ← the headline validation matrix
├── benchkit/                      ← the Python package (orchestration + analysis)
│   ├── config.py                  ← load/validate experiment + dataset configs
│   ├── datasets.py                ← resolve, fetch, checksum datasets
│   ├── provenance.py              ← capture environment manifest
│   ├── metrics.py                 ← harness-owned metric computation
│   ├── runner.py                  ← expand matrix, warmup/repeat, orchestrate
│   ├── store.py                   ← append-only JSONL results + artifact paths
│   └── adapters/
│       ├── base.py                ← Adapter ABC + shared subprocess helpers
│       ├── fzgm.py                ← wraps fzgmod-cli
│       ├── cusz.py
│       ├── cuszp.py
│       ├── cusz_hi.py
│       ├── mans.py
│       └── pfpl.py
├── external/                      ← git submodules: reference compressor sources
│   ├── cuSZ/  cuSZp/  cuSZ-Hi/  MANS/  PFPL/  …
│   └── (FZGM consumed via its own install / PATH, not vendored here)
├── scripts/
│   ├── build_all.sh               ← build each submodule, record commit+flags
│   ├── fetch_datasets.sh          ← download SDRBench, verify checksums
│   ├── lock_clocks.sh             ← pin GPU sm/mem clocks for stable timing
│   └── unlock_clocks.sh
├── results/                       ← gitignored; per-session run output
│   └── <YYYYMMDD-HHMMSS-host>/
│       ├── provenance.json
│       ├── runs.jsonl
│       └── logs/<run_id>.log
└── analysis/
    ├── load.py                    ← results → tidy DataFrame
    ├── figures.py                 ← rate–distortion, throughput, deltas
    └── notebooks/
```

Notes:
- **FZGM is not vendored as a submodule.** It is your library; the adapter calls an
  installed `fzgmod-cli` (path configurable). Reference compressors *are* vendored as
  pinned submodules so their exact source is reproducible.
- `external/` build artifacts and `results/` are gitignored. Only sources/recipes,
  configs, and the package are tracked.

---

## 5. Metrics (harness-owned)

All quality/size metrics are computed by `benchkit/metrics.py` from raw bytes. Inputs:
the original array, the compressed file size, the decompressed array, and the dataset's
declared dtype/dims/value-range.

### 5.1 Size / ratio
- **Compression ratio** `CR = original_bytes / compressed_bytes`.
- **Bit-rate** `bitrate = compressed_bits / num_elements` (bits per value). Report both;
  rate–distortion plots use bit-rate, headline tables often use CR.
- `compressed_bytes` is measured by the harness from the output file, never parsed from
  the tool. (Document any container/header overhead each tool adds.)

### 5.2 Distortion / quality
Let `r = max(original) − min(original)` be the value range, `MSE` the mean squared error
between original and decompressed.
- **PSNR** `= 20·log10(r) − 10·log10(MSE)` (∞ when MSE = 0). Value-range convention is
  fixed here so it is identical across all compressors.
- **NRMSE** `= sqrt(MSE) / r`.
- (Optional, later) SSIM for visualization-oriented fields.

### 5.3 Error-bound satisfaction
- **max_abs_err** `= max |original − decompressed|`.
- **max_rel_err** `= max |original − decompressed| / r` (range-relative).
- **eb_satisfied** — boolean: did the realized error actually respect the requested
  bound under the requested mode? A compressor that "wins" on CR while violating its
  bound is disqualified, so this flag is first-class.

### 5.4 Error-mode normalization
Before a run, the runner translates the experiment's canonical bound into each
compressor's native flags, recording both. Canonical mode is **range-relative ABS**
(`eb_abs = rel · r`) as the comparison baseline, since most SDRBench studies report this
way. Per-compressor quirks (e.g. a tool whose "REL" is per-value, or that bounds L∞ vs
L2) are documented in `docs/adapters/<compressor>.md` and encoded in the adapter so the
*effective* bound matches across tools. If a tool cannot express the canonical bound,
that is recorded and the row is flagged `mode_mismatch`.

**Canonical modes vs. native names — the collision is real.** "REL" means *different
things* across tools, so the harness uses a tool-agnostic canonical vocabulary and each
adapter translates it to native flags + an eb basis:

| Canonical | eb basis (`eb_abs =`) | ABS/REL tools (cuSZ, cuSZp, cuSZ-Hi, MANS) | ABS/NOA/REL tools (FZGM, PFPL) |
|---|---|---|---|
| `rel_range` *(cross-tool comparable)* | `eb·(max−min)` | **REL** | **NOA** |
| `rel_maxabs` | `eb·max(\|data\|)` | — | **REL** (Lorenzo; approx per-elem) |
| `abs` | `eb` | ABS | ABS |
| `from_toml` | (read from the config) | — | use the config's declared bound, no sweep |

Key facts (M1, empirically caught then verified in source):
- **FZGM/PFPL `NOA` = range-relative** (= what cuSZ etc. call `REL`); **FZGM `REL` =
  `eb·max(|data|)`** (`predictor_utils.cuh`), *not* range, and approximate per-element
  (≈1.0002× overshoot at tight bounds). FZGM's `REL` for QuantizerStage is a different
  (exact per-element) thing — revisit when those pipelines are benchmarked.
- The harness first assumed FZGM `rel` was range-relative and (correctly) flagged every
  row as bound-violating until the basis was fixed — the §5.4 hazard, concrete on day
  one. The cross-tool comparable `rel_range` maps to FZGM `NOA` and is *exact*
  (`err_over_bound`≈1.000003), so headline experiments use it.
- `metrics.compute_quality` takes the `basis` directly from the adapter's mode
  translation. `err_over_bound` (realized max error ÷ requested abs bound) is recorded
  per row so a near-miss is visible; `eb_tol` defaults to 1e-3 to absorb documented
  approximate-REL slack without masking real violations.

**Bound rendering (FZGM, TOML-first).** For TOML pipelines the harness renders the swept
bound into every lossy stage (`error_bound` + `error_bound_mode`) by text substitution
(preserving comments), validates the result re-parses, and **archives the rendered TOML
into the run's work dir** alongside the compressed/decompressed artifacts and JSON
reports — a self-contained, shippable bundle (D9).

### 5.5 Throughput
- **compress_throughput** `= original_bytes / compress_device_ms` → GB/s.
- **decompress_throughput** `= original_bytes / decompress_device_ms` → GB/s.
- Timing source is **device/kernel time reported by the tool** (the one trusted
  number), with the measurement method recorded per adapter (e.g. CUDA events,
  DAG-elapsed, includes/excludes H2D/D2H). We additionally capture **end-to-end
  wall time** from the harness subprocess for an apples-to-apples lower bound and to
  detect tools whose self-timing excludes large transfers. **Never** compare one
  tool's device time against another's host wall time (a single-shot decompress wall
  time can be 10–100× the device time because it includes pipeline construction + file
  I/O).
- **Recompute throughput in one unit convention from raw bytes + a chosen device time —
  never tabulate printed throughput.** Tools disagree on units: FZGM reports decimal
  GB/s (bytes / 1e9 / s), cuSZ reports GiB/s (1024³), cuSZp reports MiB/ms. 1 GiB/s =
  1.0737 GB/s, so mixing printed numbers bakes in a silent ~7% skew. The harness fixes
  one convention (decimal GB/s) and derives every number itself; per-adapter native
  units are documented in `docs/adapters/<x>.md`.
- Report **median over repetitions** with min/max (or IQR), never a single run. When a
  reference tool only reports a single warm run (cuSZp) or externally-aggregated runs
  (PFPL), use a *consistent* statistic from the tools that give arrays (FZGM `median` or
  `min`) and record the other tool's method — a single warm run ≈ our `min`.

### 5.6 Memory (best-effort)
- **peak_gpu_mem_bytes** via NVML sampling during the run (poll `nvmlDeviceGetMemoryInfo`
  on a side thread) or `nsys`/`--print-gpu-trace` where available. Uniform peak-memory
  capture across heterogeneous tools is hard; this is explicitly best-effort and
  nullable, with the capture method recorded.

---

## 6. Schemas

### 6.1 Result row (one atomic measurement → one JSONL line)
```jsonc
{
  "run_id": "smoke-0007",                  // unique within session
  "session_id": "20260617-141500-hostname",
  "timestamp": "2026-06-17T14:15:03Z",

  "compressor": "cusz",                    // logical name
  "variant": "reference",                  // "reference" | "fzgm"
  "pipeline": "lorenzo->huffman",          // for fzgm: the stage chain / preset
  "version": "cuSZ 0.x",                   // tool-reported version string
  "commit": "a1b2c3d",                     // submodule SHA (null for fzgm install)

  "dataset": "CESM-ATM",
  "field": "CLDHGH",
  "dtype": "f32",
  "dims": [26, 1800, 3600],
  "dim_order": "fast-to-slow",
  "num_elements": 168480000,
  "original_bytes": 673920000,

  "error_mode": "rel",                     // canonical mode
  "error_bound": 1e-3,
  "eb_abs_effective": 4.21e-4,             // normalized absolute bound actually used
  "native_flags": "-m r2r -e 1e-3",        // exactly what was passed to the tool

  "rep": 2,
  "warmup_reps": 3,

  "compressed_bytes": 5230112,
  "cr": 128.85,
  "bitrate": 0.248,

  "compress_device_ms": 1.82,
  "decompress_device_ms": 1.10,
  "compress_throughput_gbs": 370.3,
  "decompress_throughput_gbs": 612.7,
  "compress_walltime_ms": 41.0,            // harness-observed, end to end
  "timing_method": "cuda_events_d2d",      // how the tool measured device time

  "psnr": 84.21,
  "nrmse": 6.1e-5,
  "max_abs_err": 4.20e-4,
  "max_rel_err": 9.98e-4,
  "eb_satisfied": true,

  "peak_gpu_mem_bytes": 1342177280,
  "mem_method": "nvml_poll",

  "provenance_id": "20260617-141500-hostname",
  "log_path": "logs/smoke-0007.log",
  "status": "ok",                          // "ok" | "fail" | "mode_mismatch"
  "error_message": null
}
```

### 6.2 Provenance manifest (one per session → `provenance.json`)
```jsonc
{
  "session_id": "20260617-141500-hostname",
  "gpu": {
    "name": "NVIDIA A100-SXM4-40GB", "uuid": "GPU-...", "driver": "550.xx",
    "cuda_runtime": "12.4", "vbios": "...", "ecc": true, "persistence": true,
    "sm_clock_locked_mhz": 1410, "mem_clock_locked_mhz": 1215, "power_limit_w": 400
  },
  "host": { "cpu": "AMD EPYC ...", "cores": 64, "ram_gb": 512,
            "os": "Ubuntu 24.04", "kernel": "6.x" },
  "harness": { "git_sha": "...", "config_hash": "sha256:...",
               "python": "3.12", "numpy": "2.x" },
  "compressors": {
    "cusz":   { "repo": "https://github.com/szcompressor/cuSZ", "commit": "a1b2c3d",
                "build_flags": "-DPSZ_BACKEND=cuda ...", "compiler": "nvcc 12.4 / gcc 13",
                "built_at": "2026-06-15T..." },
    "fzgm":   { "cli_path": "/usr/local/bin/fzgmod-cli", "version": "2.0" }
  },
  "nvidia_smi": "<captured snapshot>",
  "env": { "CUDA_VISIBLE_DEVICES": "0" }
}
```

### 6.3 Experiment config (`configs/experiments/*.yaml`)
```yaml
name: fzgm_vs_reference
description: Validate FZGM ports against originals at matched bounds.

datasets: [CESM-ATM, NYX, Hurricane-ISABEL]   # keys into configs/datasets.yaml
fields: all                                    # or explicit list per dataset

error:
  mode: rel                                    # canonical mode
  bounds: [1e-2, 1e-3, 1e-4, 1e-5]

repetitions: 5
warmup_reps: 3
lock_clocks: true

# Each entry is a (compressor, variant, pipeline) the runner will drive.
runs:
  - {compressor: cusz,    variant: reference, pipeline: lorenzo+huffman}
  - {compressor: fzgm,    variant: fzgm,      pipeline: "lorenzo->huffman"}
  - {compressor: cuszp,   variant: reference, pipeline: default}
  - {compressor: fzgm,    variant: fzgm,      pipeline: "lorenzo->bitshuffle->rze"}

# Pairings for the FZGM-vs-original delta report (§8).
pairings:
  - {reference: cusz,  fzgm_pipeline: "lorenzo->huffman",            label: cuSZ}
  - {reference: cuszp, fzgm_pipeline: "lorenzo->bitshuffle->rze",    label: cuSZp}
```

### 6.4 Dataset manifest (`configs/datasets.yaml`)
```yaml
CESM-ATM:
  source: https://sdrbench.github.io/   # download URL / instructions
  dtype: f32
  dim_order: fast-to-slow
  fields:
    CLDHGH: {dims: [1800, 3600],        sha256: "...", path: "CESM-ATM/CLDHGH_1_1800_3600.f32"}
    # ...
NYX:
  dtype: f32
  fields:
    baryon_density: {dims: [512, 512, 512], sha256: "...", path: "NYX/baryon_density.f32"}
```

---

## 7. Adapter interface

Each compressor implements `benchkit/adapters/base.py::Adapter`:

```python
class Adapter(ABC):
    name: str
    variant: str                    # "reference" | "fzgm"

    def is_available(self) -> bool: ...
    def provenance(self) -> dict: ...                 # version, commit, build flags

    # Translate canonical (mode, bound, dataset) → native flags. Records both.
    def native_flags(self, spec: RunSpec) -> NativeInvocation: ...

    # Run compression. Returns compressed artifact path + device time + raw log.
    def compress(self, spec: RunSpec) -> CompressResult: ...

    # Run decompression. Returns decompressed artifact path + device time + raw log.
    def decompress(self, spec: RunSpec) -> DecompressResult: ...
```

The adapter's *only* jobs: build the command line, run the subprocess, and parse two
things from stdout — **device time** and **tool version**. Compressed size, CR, all
quality metrics, and eb-satisfaction are computed downstream by `metrics.py` from the
artifacts. This keeps every compressor honest against the same definitions.

Per-adapter quirks (flag meanings, REL semantics, header overhead, timing method) live
in `docs/adapters/<compressor>.md` so the knowledge is captured, not buried in code.

---

## 8. The FZGM-vs-reference delta report

The product that answers the headline question. For each `pairing` in the config, at
each matched (dataset, field, bound):

| metric | computed as | PASS criterion (default, configurable) |
|---|---|---|
| ΔCR | `(CR_fzgm − CR_ref) / CR_ref` | within ±5% |
| ΔPSNR | `PSNR_fzgm − PSNR_ref` (dB) | within ±0.5 dB |
| Δcompress throughput | `(T_fzgm − T_ref) / T_ref` | ≥ −20% (not *too* much slower) |
| Δdecompress throughput | same | ≥ −20% |
| eb_satisfied | both must be true | both true |

Output: a per-pairing table (CSV + LaTeX) and a roll-up PASS/FAIL with the cells that
fail and by how much — so "did the port hold up?" is a glance, and the offending
dataset/bound is immediately visible for debugging the FZGM stage.

Thresholds are config-driven; they encode "roughly matches … without losing too much
speed" numerically so the standard is explicit and consistent across papers.

---

## 9. Roadmap / milestones

- **M0 — Design (this doc).** Architecture, schemas, decisions. ✅ Done.
- **M1 — Core loop on one compressor.** ✅ **Done (2026-06-18).** `config → runner →
  fzgm adapter → metrics → JSONL → comparison table`, driven by
  `configs/experiments/smoke.yaml` on local CLDHGH. The `benchkit` package ships:
  config/dataset loaders, the `Adapter` ABC + `FzgmAdapter`, harness-owned `metrics`,
  lightweight `provenance` capture, append-only `store`, the `runner`, and a `report`
  table — run with `python -m benchkit run configs/experiments/smoke.yaml`. Validated:
  harness PSNR matches FZGM's own to 5 decimals; per-stage timing captured; the
  rel-basis hazard (§5.4) surfaced and fixed. Deferred to later milestones: skip-
  completed resumability, clock locking, TOML-preset (huffman/cuSZ-equivalent) sweeps.
- **M2 — Reproducibility & HPC execution.** ✅ **Done (2026-06-18).** Site config
  (de-hardcoded `fzgmod-cli`/results-root paths; `${ENV}` dataset roots); matrix
  **sharding** (`--shard k/N` for SLURM job arrays); **resume** (skip completed cells by
  `cell_key`); per-shard provenance capturing scheduler (SLURM/PBS) + software
  (modules/Spack/nvcc) + GPU; a `merge` command; `scripts/submit.slurm`. **Timing
  reliability** (since clocks can't be locked on shared nodes): per-cell coefficient of
  variation over the kept reps flags unstable throughput (`*_stable`, `timing_reliable`,
  default cv ≤ 0.15), plus a concurrent `GpuSampler` recording clocks + thermal/power
  throttle reasons during the benchmark. See [Execution on HPC](#12-execution-on-hpc).
  Optional clock-lock hook (where permitted) is the only deferred piece.
- **M3 — Reference adapters (incremental).** Add submodules + build scripts + adapters
  one at a time: cuSZ → cuSZp → cuSZ-Hi → MANS → PFPL. Each lands with a
  `docs/adapters/<x>.md` and passes the smoke matrix before the next is added.
- **M4 — Analysis layer.** Tidy loader, rate–distortion curves, throughput bars, and
  the FZGM-vs-reference delta report (§8).
- **M5 — Paper-support polish.** LaTeX table export, figure styling, run archiving,
  SDRBench fetch automation with checksum verification. Optionally: multi-GPU / cluster
  job-array submission.

Each milestone is independently useful and leaves a working artifact.

---

## 10. Decision log

| # | Decision | Rationale |
|---|---|---|
| D1 | **Python** orchestration + analysis; compressors driven as **subprocesses**. | pandas/matplotlib/pydantic ecosystem; subprocess isolation matches heterogeneous CLIs and keeps the harness language-agnostic about compressors. |
| D2 | Reference compressors vendored as **git submodules + build scripts** (pinned commits). | Self-contained, no external package manager dependency; exact source reproducible. |
| D3 | Datasets: **SDRBench standard set**, described by a checksummed manifest. | Field-standard, paper-comparable; checksums guard against silent data drift. |
| D4 | **Harness owns all size/quality metrics**; only device time is trusted from tools. | Eliminates per-tool PSNR/CR/timing convention skew → fair comparison. |
| D5 | Canonical error mode = **range-relative**, normalized into native flags per adapter. | "REL" is defined inconsistently across tools; one baseline makes bounds comparable. |
| D6 | Results stored as **append-only JSONL**, one row per atomic run, with provenance FK. | Tidy, mergeable, trivially loadable; re-runs append rather than clobber. |
| D7 | FZGM consumed via **installed `fzgmod-cli`**, not vendored. | It's the home library; avoid duplicating its source/build here. |
| D8 | First deliverable: **design doc only**, scaffold in M1 after review. | Agreed scope for this pass. |
| D9 | **TOML-first pipelines** for FZGM (not CLI `--stages`); rendered config archived per run. | TOML exposes the full DAG (branches, fused stages) the CLI text path can't; lets a hand-tuned config be benchmarked as-is and shipped with its results+provenance. `--stages` kept only for quick linear tests. |
| D10 | **Canonical, tool-agnostic error modes** (`abs`/`rel_range`/`rel_maxabs`/`from_toml`); adapters translate to native flags + eb basis. | "REL"/"NOA" names collide across tools; one canonical vocabulary makes bounds comparable and the eb-check correct. |
| D11 | **Decompressed output deleted after metrics by default** (`retain_decompressed: false`); its sha256 is recorded and `c.fzm` is kept. | Keeps the local repo under a ~20 GB budget — `d.bin` is ~original-sized and regenerable from `c.fzm`; at ~2–3 MB/run retained, ~7k runs fit. Toggle on per-experiment when the array itself is needed. |
| D12 | **No hardcoded paths** — `fzgmod-cli` + results-root from a site config (env > `configs/site.local.yaml` > default); dataset roots via `${ENV}` expansion. | The same configs must run unchanged on the desktop and on HPC (scratch filesystems, module/Spack-provided binaries). |
| D13 | **Sharding + resume** keyed by a deterministic `cell_key`; each shard writes its own `runs.shard-k-of-N.jsonl`; a `merge` step dedupes. | SLURM job arrays split a big matrix across tasks with no append contention; jobs that hit walltime resume idempotently. |
| D14 | **Per-shard provenance** (not one shared manifest). | Each array task may land on a different node/GPU — capturing GPU+scheduler+software per shard is correct, and avoids a write race. |
| D15 | **Timing reliability = variance-primary, throttle-reasons-secondary.** `cv` over kept reps decides `timing_reliable`; concurrent GPU sampling is diagnostic. | Clocks can't be locked on shared nodes; cv catches sub-sample-rate clock bounce that a clock query misses, while throttle reasons explain *why* when something is detectably throttling. |
| D16 | **Graph-mode plumbing**: a per-run-entry `graph: true` sets `RunSpec.graph` → adapter passes `--graph` → row records `graph_requested`/`graph_active`/`graph_reason` (parsed from report-json's `"graph"` object). **benchkit does not maintain a stage compatibility matrix** — the FZGM library validates a captured DAG stage-by-stage (`CompressionDAG::setCaptureMode`) and throws a descriptive error naming incompatible stages; the CLI's `--graph` (added 2026-07-03) catches that and falls back to normal execution, reporting the outcome in `report-json` schema 1.1. Confirmed live on BigRed200: `cusz.toml` (Huffman) falls back cleanly; `cuszp2.toml`/`cuszp3.toml` (linear ABS Quantizer → Lorenzo/TiledLorenzo → AdaptiveBitpack forward) capture and replay correctly. | benchkit's plumbing (2026-07-02) predated the CLI flag (2026-07-03) intentionally, so the contract could be specified up front; one mismatch surfaced on first real integration test — the adapter expected flat `graph_active`/`graph_incompatible_reason` keys but the shipped schema nests them under `"graph": {...}` — fixed in the adapter once seen against the real binary. NOA-mode Quantizer pipelines (pfpl, quantizer_lorenzo_bitpack) still need a precomputed value base (`setValueBase`) to avoid a D2H scan; not yet tested. See `docs/adapters/fzgm.md` "Graph mode". |
| D17 | **Per-run-entry dataset scoping** (`only_datasets` / `skip_datasets` on a `RunEntry`, mutually exclusive). | Some FZGM presets are dimensionality-specific (e.g. `TiledLorenzo`-based cuSZp3 presets are 2-D/3-D only; `GInterp`-based cuSZ-Hi presets are structurally 2-D+ only, see `docs/adapters/cuszhi.md`). Forcing one `pipeline:` across the whole dataset x field matrix silently produced nonsense on mismatched fields (E12: cuszp3's 2-D preset fed 1-D HACC data collapsed to degenerate tiles, CR 0.69). Scoping a run entry to the fields its preset actually supports is config, not a runner special-case — consistent with D7 "config over code." |
| D18 | **Aggregate CR report** (`report --aggregate`) computes both **ratio-of-sums** (`sum(original_bytes)/sum(compressed_bytes)` across fields, size-weighted) and **geometric mean of per-field CR** (`(Π CR_i)^(1/n)`, every field weighted equally), grouped by (compressor, variant, pipeline, error_bound). | Both appear in the compression literature as "the" multi-field CR and answer different questions — ratio-of-sums is what SDRBench-style papers usually report as an "overall CR" (a big field dominates); geomean is more robust to one huge/tiny field skewing the number. Reps of a cell are collapsed via the *median* compressed_bytes first (deterministic compressors shouldn't vary CR across reps; median is a defensive statistic against a rep-level fluke, not an assumption that they will). Implemented stdlib-only in `benchkit/analysis.py`, ahead of the pandas-based M4 analysis layer, because the FZGM-vs-native validation matrix needed it now. |
| D19 | **Emulated relative bounds for tools with no native range/maxabs-relative mode** (`read_range_stats` helper in `adapters/base.py`): the adapter reads the input file itself, computes `max-min`/`max\|x\|`, multiplies by the canonical `eb`, and passes the product as the tool's native **absolute** bound. Used by zfp, MGARD, and SPERR (all `abs`-only natively). | Nearly every experiment config in this repo sweeps `rel_range`; tools that only accept an absolute tolerance would otherwise be unusable in those same configs. Since the emulation reads the identical file the harness's own `eb_ok` check reads, the two computations of range/maxabs agree, so the emulated bound is exactly as cross-tool-comparable as a native rel_range mode. Surfaced a real correctness bug in the process: MGARD's `-s 0` (smoothness) does not bound pointwise max error even in `-em abs` mode (see `docs/adapters/mgard.md`) — must pass `-s inf`. |
| D20 | **cuSZp2/cuSZp3 source patched** (in `~/compressors/`, outside this repo) to cache their internal scratch-buffer `cudaMalloc`s across calls instead of allocating+freeing on every compress/decompress, plus `sm_90` added to both CMakeLists.txt. | On the JetStream2 H100 (GPU-passthrough cloud VM), native cuSZp2/cuSZp3 throughput measured 20-100x below expectation (~8 GB/s vs. an A100 baseline of ~58 GB/s on the same cell) while every other tool measured correctly. Root-caused with `nsys`: a single `cudaMalloc` call took up to 451 ms on this VM vs. ~145 μs of actual kernel time — cuSZp calls `cudaMalloc`/`cudaFree` for 3 small scratch arrays on *every* compress/decompress call, an allocator-latency tax that's noise on bare-metal hardware but dominates cuSZp's sub-millisecond kernels specifically. Fixed by caching the scratch allocation (grow-only, reused across calls); confirmed 8.0/7.8 → 159.6/207.1 GB/s (CESM) and ~124/131 → 653/1205 GB/s (NYX 3-D) through the real adapter, CR/PSNR unaffected. See `docs/adapters/cuszp.md`. Lesson for any future cloud/passthrough GPU site: a tool whose own self-reported throughput looks implausibly low relative to its known bare-metal numbers is a `cudaMalloc`-in-the-hot-path smell before it's a "this cloud GPU is slow" conclusion — profile with `nsys`/hardware counters, don't trust the tool's own timer at face value. |
| D21 | **Full audit of all native reference compressors for the two D20 failure modes** (missing `sm_90` SASS, `cudaMalloc` inside the timed region), following the cuSZp fix. Findings: **sm_90 missing/silently overridden** in `FZ-GPU` (no `-arch` flag at all in the Makefile → nvcc's ancient sm_52 default, ran only via PTX JIT on this box) and `lsCOMP` (unconditional `set(CMAKE_CUDA_ARCHITECTURES 80 86)` in CMakeLists.txt — the *identical* cache-shadowing bug cuSZp had, silently discarding any `-DCMAKE_CUDA_ARCHITECTURES=90` passed at configure time). Both fixed (`FZ-GPU/Makefile` gets an explicit `-gencode arch=compute_90,code=sm_90`; `lsCOMP/CMakeLists.txt`'s `set()` now includes `90`) and rebuilt; verified via `cuobjdump -lelf`. cuSZ, cuSZ-Hi, PFPL, and MGARD were already correctly targeting `sm_90` (either no hardcoded `set()` at all, or one guarded with `if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)`). **`cudaMalloc`-in-hot-path**: none of the other tools have it — cuSZ and cuSZ-Hi already use CUDA-event device-only timing with an in-process `--repeat` (a *different*, pre-existing bugfix: an aliased-pointer use-after-free, not an allocator issue — see `docs/adapters/cusz.md`); FZ-GPU allocates once per `runFzgpu()` call, outside its `compressionStart`/`compressionEnd` window, with its own in-process `repeat` loop; PFPL allocates once before its `NUM_RUNS` loop and times only the kernel launches inside it. cuSZp2/cuSZp3 were the only tools with allocation *inside* the timed region. | The user asked, after the cuSZp fix, whether the same two problems (missing Hopper codegen, malloc-in-hot-path) were silently present in the other reference tools before trusting any of them for paper-quality numbers on this VM. Worth recording that `set(CMAKE_CUDA_ARCHITECTURES ...)` with no `if(NOT DEFINED ...)` guard is a recurring anti-pattern across this codebase's vendored compressors (cuSZp, lsCOMP) — anyone adding a new reference tool should grep its CMakeLists for a bare `set(CMAKE_CUDA_ARCHITECTURES` and either add `90` directly or gate it behind `if(NOT DEFINED ...)` before trusting `-DCMAKE_CUDA_ARCHITECTURES=90` to reach the compiler. Also worth recording the contrast with FZGPUModules itself (see below) — it never had this class of bug because it was designed against a stream-ordered pool allocator from the start. |
| D22 | **How FZGPUModules avoids the D20 class of bug by construction.** `Pipeline` buffers use `MemoryStrategy::PREALLOCATE` and a `MemoryPool` (`src/mem/mempool.cpp`) built on CUDA's native stream-ordered pool allocator (`cudaMallocFromPoolAsync`, with a synchronous `cudaMalloc` fallback only when the pool is unavailable, e.g. vGPU). The CLI's `-b --runs N` loop (`cli.cpp`) runs one **untimed warmup** `pipeline->compress()` call first (warms the pool, JIT, GPU clocks), then times each subsequent `compress()`/`decompress()` call with `std::chrono` bracketing just the call + `cudaDeviceSynchronize()` — no allocation happens inside that window because the pool-owned buffers persist across calls (freeing pool-owned pointers is explicitly forbidden — see the Memory Ownership table in FZGPUModules' `docs/architecture.md`). Each call also independently records a device-only `dag_elapsed_ms` via internal CUDA events (`pipeline->getLastPerfResult()`), giving two corroborating timing sources per run. | Directly answers "how does FZGPUModules time in relation to cudaMalloc calls" (asked alongside the D21 audit): unlike cuSZp's ad-hoc fix (a hand-rolled grow-once scratch cache retrofitted after the fact), FZGPUModules never allocates inside its timed region in the first place — allocation-vs-timing safety was a design constraint (`PREALLOCATE` + pool allocator + ownership rules), not a bug that had to be found. This is the reason FZGM's own numbers were never in question during the D20 investigation even though they run on the same VM as the broken cuSZp numbers. |
| D23 | **Native cuSZp2 (not v3) corrupted output on this H100 for ~11/24 cells — root-caused and fixed.** Found while building an A100-vs-H100 comparison (CR/PSNR were expected to match exactly; they didn't for native cuSZp2's `plain`/`outlier` modes on HACC/HURR). Empirically ruled out this session's own scratch-buffer-cache and `TIMING_REPEATS` changes (both independently reverted, bug persisted bit-for-bit each time) and int32 quantization overflow (checked HACC/vx's actual worst-case code, ~2547, six orders below `INT32_MAX`). **Root cause, confirmed by direct instrumentation** (a temporary per-block failure counter added to `examples/cuSZp.cpp`): the `excl_sum` `__shared__` variable in the decoupled-look-back GPU-wide prefix-sum scan is only ever assigned for `warp>0` — block/warp 0 (which every grid has) reads it uninitialized at `base_idx = excl_sum + rate_ofs`, undefined behavior whose value depends on physical shared-memory bank residue, plausibly differing by GPU architecture/driver. Instrumentation showed exactly 2 of 8575 blocks corrupted for the worst cell (block 0 itself, plus block 5 — collateral damage from block 0's compressed bytes landing at a garbage offset and clobbering block 5's). **Fix:** one line, `excl_sum = 0;`, added to the pre-existing `if(warp==0)` branch in all 8 kernel functions (compress/decompress × plain/outlier × f32/f64). Confirmed **cuSZp-V3 already has this exact line** in the same branch — a one-line backport from v3, not a novel fix. Verified: all 24 previously-checked native-cuSZp2 cells now pass cuSZp2's own internal error check (0 failures), `cuSZp_test_f32`/`cuSZp_test_f64` self-tests still pass. Patch lives only in `~/compressors/cuSZp-V2.0.1/src/cuSZp_kernels_{f32,f64}.cu` (not upstreamed). See `docs/adapters/cuszp.md`. | The user's instinct to sanity-check "CR/PSNR should be the same between platforms" before trusting a performance-only comparison caught a real, previously-unknown correctness bug — worth verifying that assumption quantitatively rather than assuming it, even (especially) when the numbers being compared are throughput. The user is a coworker of cuSZp's author and explicitly wants bugs like this surfaced upstream, which is why this got a full fix rather than just a documented caveat. |
| D24 | **`results/baselines/` — curated, git-tracked cross-machine result snapshots**, carved out of the otherwise-gitignored `results/` (`.gitignore`: `results/*` + `!results/baselines/`). Each `results/baselines/<id>/` holds `runs.jsonl` + `provenance.json` (harness schema, copied verbatim when possible) plus a `metadata.yaml` (GPU/site/date/known-issues, schema documented in `results/baselines/README.md`). Added `scripts/reconstruct_runs_from_stdout.py` (best-effort `runs.jsonl`/`provenance.json` from a stdout/SLURM log when the harness's own output was never copied off the source machine — explicitly marked lower-fidelity, missing per-rep timing/stages/gpu_sampling) and `scripts/build_comparison_artifact.py` (generalized version of this session's one-off comparison-artifact scripts: takes two baseline directories, cross-checks CR/PSNR agreement before charting throughput, flags disagreements generically rather than assuming which side is at fault). Seeded with the two baselines from the A100-vs-H100 investigation: `a100-bigred200-slurm7562329` (reconstructed from a stdout log, carries its own uninitialized-`excl_sum` residual per D23) and `h100-jetstream2-20260719` (native, full-fidelity, the first baseline with verified-correct native cuSZp2 output). | Building the A100-vs-H100 comparison required manually locating a stray SLURM log on this machine, hand-parsing its summary table, and writing one-off scripts against hardcoded absolute paths — none of which would work from a different machine or survive this session ending. The user wants to keep doing cross-machine comparisons as more GPUs/sites are benchmarked, so the reusable pieces (data format, parsing script, artifact-building script) needed to move from scratchpad one-offs into the repo, not just the data. `metadata.yaml`'s `known_issues` field exists specifically because D23 was found by two baselines *disagreeing* on cells that should have matched — future baselines should record that kind of finding right next to the data, not just in a decision-log entry that's easy to miss when comparing against an old snapshot later. |

---

## 11. Open questions (to resolve before/within M1)

1. **~~`fzgmod-cli` machine-readable output.~~ RESOLVED (2026-06-18).** FZGM now ships
   `--report-json <path>` (schema_version 1.0): a standalone JSON file with `tool`,
   `status`, `config`, `size`, `timing` (device_ms + host_wall_ms, per-rep `all` arrays),
   `throughput`, `memory`, `quality`, and an FZGM-only `stages[]` per-stage device-time
   breakdown. `--report-json` auto-enables profiling, so device timing and stages are
   always populated. Adapter integration rules captured in
   [`docs/adapters/fzgm.md`](adapters/fzgm.md); full schema lives in the FZGM repo at
   `memory/report_json_spec.md`. Other adapters still need stdout scraping; FZGM is the
   one we control and it is now clean.
2. **Decompressed-output retention.** Harness-owned quality metrics require each tool to
   write the decompressed array to disk. Confirm every reference tool can emit raw
   decompressed output (most can); note any that only self-report PSNR.
3. **Peak-memory method.** Decide NVML-poll vs `nsys` as the default; NVML-poll is
   lighter and per-process-attributable, `nsys` is more precise but heavier. Default
   proposal: NVML-poll, nullable, method recorded.
4. **Clock policy.** Confirm we can `nvidia-smi -lgc`/`-lmc` on the target GPUs (needs
   permissions); if not, record clocks-as-observed and flag throughput as unlocked.
5. **Dim-order convention.** Lock fast-to-slow vs slow-to-fast across the manifest and
   all adapters (FZGM uses `-l fast x mid x slow`); mismatches silently wreck quality
   metrics, so this must be asserted, not assumed.
```

---

## 12. Execution on HPC

The toolkit runs unchanged on the local desktop and on SLURM/PBS clusters; the cluster
is where the paper-grade matrices and reference GPUs (A100/H100) live.

**Paths are never hardcoded.** `fzgmod-cli` and the results root resolve from a site
config (env var > `configs/site.local.yaml` (gitignored) > default); dataset manifest
`root`s support `${ENV}` expansion (e.g. `${BENCHKIT_DATA_ROOT}`). On HPC set these from
the job script (`module load` / `spack load`); commit only `site.example.yaml`.

**Sharding (job arrays).** The cell matrix is enumerated in a deterministic order;
`--shard k/N` runs only cells where `index % N == k`. A SLURM array of `N` tasks shares
one session dir (`--session-id $SLURM_ARRAY_JOB_ID`) and each task writes its own
`runs.shard-k-of-N.jsonl` + `provenance.shard-k-of-N.json` — no append contention, and
per-task provenance because each task may be a different node/GPU. `benchkit merge
<session>` dedupes the shard files into `runs.jsonl`. Template: `scripts/submit.slurm`.

**Resume.** Each row carries a deterministic `cell_key`
(`compressor|variant|pipeline|dataset|field|mode|eb`). On start the runner scans all run
files in the session dir and skips cells already `status: ok`, so a task that hits
walltime resumes idempotently on resubmit.

**Timing without clock-lock privileges.** `nvidia-smi -lgc/-lmc` is usually admin-only on
shared clusters, so the harness does not assume it. Two complementary signals make
throughput honest instead of silently wrong:
- **Variance (primary).** Per cell, the coefficient of variation (`cv = std/median`) over
  the kept reps is computed for each phase; `cv > timing_cv_threshold` (default 0.15)
  sets `*_stable = false` and `timing_reliable = false`. This catches fast clock-bounce
  that coarse sampling misses (observed: a cell read a steady 1710 MHz yet had cv 0.25).
  On unlocked GPUs prefer the recorded `*_device_ms_min` (best warm rep).
- **Throttle reasons (diagnostic).** A `GpuSampler` thread polls clocks + NVML throttle
  reasons *during* the benchmark (a post-hoc query only ever sees GpuIdle). It records
  observed SM-clock min/mean/max, max temp, and any thermal/HW/power throttle reason;
  `throttled_thermal` also forces `timing_reliable = false`.

The runner prints a per-cell flag and an end-of-run roll-up of unreliable cells. Results
from different GPUs are partitioned by provenance, never silently pooled.

```bash
# local
python -m benchkit run configs/experiments/smoke.yaml

# one HPC array task (k of N), shared session, resumable
python -m benchkit run configs/experiments/sdrbench.yaml \
    --session-id "$SLURM_ARRAY_JOB_ID" --shard "${SLURM_ARRAY_TASK_ID}/${N}"

# after the array finishes
python -m benchkit merge "$BENCHKIT_RESULTS_ROOT/$SLURM_ARRAY_JOB_ID"
```
