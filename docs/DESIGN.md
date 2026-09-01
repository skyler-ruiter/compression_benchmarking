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
├── scripts/
│   ├── build-*.sh                 ← selected helper/SDK build recipes
│   ├── download-sdrbench.sh       ← download SDRBench; checksum lock is separate
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
- **FZGM and reference compressors are not vendored.** Adapters call installed builds
  selected by environment/per-run paths. Their source/build/patch identity is mandatory
  publication provenance; `docs/reference-tools.md` records the acquisition gap.
- `results/` is gitignored except curated legacy baselines. Sources/recipes, configs,
  and the package are tracked; machine-specific helper ELF outputs are not.

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

On-disk schemas are enforced by `benchkit/schema.py`. New native sessions use
`session_schema_version: 1`; new rows use `result_schema_version: 1`. Unversioned
historical baselines load as legacy v0 and are not modified. The v1 writer uses strict
JSON (`allow_nan=False`): a non-finite float is stored as `null` plus a
`nonfinite_values` JSON-pointer map, then restored to the corresponding Python float by
the reader. `psnr_kind` separately records the scientific meaning as `finite`, `exact`,
or `undefined`.

Rows have an explicit `record_kind`: `measurement`, `failure`, or
`reconstructed_measurement`. The last has a deliberately smaller required-field set so
a stdout reconstruction cannot masquerade as a full-fidelity native result. Core
identity/status/metric fields and their nullability are validated at every store/merge
read and write boundary; additive adapter-specific nested fields remain allowed.

### 6.1 Logical-cell and execution identity

Identity payloads use sorted-key, compact, UTF-8 strict JSON and SHA-256. Each payload
is stored beside its hash so merge and the later H4 verifier can reject collisions or a
row whose inputs were edited without re-hashing.

- `logical_cell_id = logical-v1-<sha256>` identifies the requested experimental arm:
  compressor, variant, normalized requested pipeline label, dataset, field, canonical
  error mode, and bound. It excludes binaries, resolved files, graph request, and
  machine state. Those are different executions of the same named scientific cell. A
  graph or other ablation arm therefore needs its own `variant`, as the existing graph
  configs already do.
- `execution_id = execution-v1-<sha256>` binds the logical ID to behavior-affecting
  run-entry parameters (including `graph` and `cli_path`), repetitions/warmups/timing
  policy, normalized pipeline source plus its SHA-256 and resolution inputs, the
  SHA-256 of exactly the dataset bytes consumed, executable/tool identity,
  harness/config identity, and GPU/host/software state. The row also retains the actual
  `pipeline_sha256` produced by adapter preparation as an independent audit value; H3
  archives and verifies that file and completes build provenance.
  FZGM's `FZ_FUSION` and `FZ_FUSION_NVRTC` environment variables are included in the
  execution environment payload because they change the selected kernels without
  changing the TOML or executable. Fusion ablation arms still require distinct variant
  names because logical identity, not execution identity, defines scientific matching.
- `run_id` identifies one attempt and is unique even when the same exact execution is
  retried. It is not used for scientific matching.

Resume skips only a successful matching `execution_id`. A legacy `cell_key` cannot
prove that the dataset, binary, graph request, or harness still matches, so legacy rows
remain readable but never suppress an H2 execution. `cell_key` is emitted temporarily
as a compatibility alias for older scripts.

Merge groups attempts by `logical_cell_id` (reconstructed from complete legacy fields
when possible) and applies one supersession order: success beats failure; among equal
statuses in one source file, the later appended attempt wins; across files, a raw shard
beats the older `runs.jsonl` from a previous merge. One ID associated with two different
canonical payloads is a hard error.
Publication comparison first locates candidates by the shared display coordinates
`(variant, dataset, field, bound)`, then requires each native/FZGM side's
`logical_cell_id` to agree across baselines before comparing its metrics.

### 6.2 Result row (one atomic measurement → one JSONL line)
```jsonc
{
  "result_schema_version": 1,
  "record_kind": "measurement",
  "run_id": "smoke-0007-a81f09d42e1b",     // unique attempt
  "session_id": "20260617-141500-hostname",
  "identity_schema_version": 1,
  "logical_cell_id": "logical-v1-...",
  "execution_id": "execution-v1-...",
  "logical_cell": {"identity_schema_version": 1, "compressor": "cusz",
                   "variant": "reference", "pipeline": "lorenzo+huffman",
                   "dataset": "CESM-ATM", "field": "CLDHGH",
                   "error_mode": "rel_range", "error_bound": 0.001},
  "execution_context": {"identity_schema_version": 1,
                        "logical_cell_id": "logical-v1-...",
                        "run_parameters": {}, "resolved_config": {},
                        "dataset": {"sha256": "...", "bytes": 673920000},
                        "tool": {}, "harness": {}, "environment": {}},
  "dataset_sha256": "...",
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
  "psnr_kind": "finite",
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

An exact reconstruction is strict JSON rather than a non-standard `Infinity` token:

```json
{"psnr": null, "psnr_kind": "exact",
 "nonfinite_values": {"/psnr": "positive_infinity"}}
```

### 6.3 Provenance manifest (one per session → `provenance.json`)
```jsonc
{
  "session_schema_version": 1,
  "session_kind": "native",
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

### 6.4 Experiment config (`configs/experiments/*.yaml`)
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

### 6.5 Dataset manifest (`configs/datasets.yaml`)
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
  **sharding** (`--shard k/N` for SLURM job arrays); **resume** (originally by
  `cell_key`, upgraded by H2 to exact `execution_id`); per-shard provenance capturing
  scheduler (SLURM/PBS) + software
  (modules/Spack/nvcc) + GPU; a `merge` command; `scripts/submit.slurm`. **Timing
  reliability** (since clocks can't be locked on shared nodes): per-cell coefficient of
  variation over the kept reps flags unstable throughput (`*_stable`, `timing_reliable`,
  default cv ≤ 0.15), plus a concurrent `GpuSampler` recording clocks + thermal/power
  throttle reasons during the benchmark. See [Execution on HPC](#12-execution-on-hpc).
  Optional clock-lock hook (where permitted) is the only deferred piece.
- **M3 — Reference adapters (incremental).** ✅ Adapters exist for the functional
  families documented under `docs/adapters/`; sources are external builds rather than
  the originally planned submodules. MANS/lsCOMP remain deliberately limited coder
  scaffolding. See `docs/reference-tools.md`.
- **M4 — Analysis layer.** Tidy loader, rate–distortion curves, throughput bars, and
  the FZGM-vs-reference delta report (§8).
- **M5 — Paper-support polish.** LaTeX table export, figure styling, run archiving,
  SDRBench fetch automation with checksum verification. Optionally: multi-GPU / cluster
  job-array submission.
- **M6 — Publication and AD/AE hardening.** ✅ **Complete (2026-08-24).**
  Turn the working research harness into a mechanically verifiable publication artifact.
  The ordered work packages and their acceptance criteria are below. They are sequenced
  so each package establishes the contract required by the next one; do not jump directly
  to bundle generation while session identity and verification remain ambiguous.

Each milestone is independently useful and leaves a working artifact.

### M6 publication-hardening path

| ID | Status | Work package | Required outcome / acceptance criteria |
|---|---|---|---|
| H0 | **DONE** | Canonical result loading and true comparison-cell keys | A merged session directory reads only its deduplicated `runs.jsonl`; an unmerged session reads its shards. Resume still scans all raw attempts. Comparison artifacts key by `(variant, dataset, field, error_bound)` and retain every field in a multi-field dataset. Regression tests cover both bugs. |
| H1 | **DONE** | Versioned session/result schema and strict serialization | `benchkit/schema.py` defines and validates v1 native-session, measurement, failure, and reconstructed-measurement contracts. Store, merge, reporting, reconstruction, and publication scripts use compatibility readers. New writes are strict JSON with explicit non-finite metadata and PSNR meaning; malformed rows fail before a file is created or truncated. Legacy unversioned baselines load as v0, and v0 rows written into a new merge record `source_result_schema_version: 0`. Early reconstructed v0 rows without IDs receive deterministic content-derived IDs marked `legacy_identity_synthesized: true`. |
| H2 | **DONE** | Separate logical-cell identity from execution identity | Canonical payloads and SHA-256 IDs live in `benchkit/identity.py` and beside every new native row. Resume matches exact `execution_id`; merge supersedes by `logical_cell_id` under the documented success/append/source rule and rejects payload collisions; cross-baseline publication pairing refuses unequal logical IDs. Execution identity covers graph/run parameters, normalized pipeline source and resolution inputs, consumed dataset bytes, tool executables, harness/config, and machine/software state. Legacy `cell_key` remains readable but cannot suppress an H2 run. Tests cover transition, sensitivity, collision, resume, supersession, and pairing. |
| H3 | **DONE** | Complete provenance and dataset integrity | Content-addressed session inputs retain the exact experiment and dataset YAML, sanitized site settings, resolved dataset inventory, tracked harness patch, and every rendered pipeline. The manifest records the full harness commit, tracked patch and untracked-file identity, observed/declared consumed-byte dataset SHA-256, executable identity, and declared tool version/source/build/patch fields. Dataset verification is a pre-measurement gate; `--allow-unverified-datasets` is an explicit non-publication-grade escape hatch. Every new success/failure row joins an immutable, ID-addressed, shard-bound `provenance-v1` invocation manifest, so resumes on another node or at another time preserve rather than overwrite earlier provenance. The ID covers the complete persisted payload; H6's fake resume test guards against same-ID/different-byte collisions. Regression tests cover manifest retention, archive immutability, redaction, dirty identity, shard sensitivity, and provenance history. |
| H4 | **DONE** | Mechanical session verification and completion contract | `benchkit verify` validates all raw/canonical schemas, immutable provenance and input checksums, unique IDs, row/provenance/dataset joins, expected matrix coverage, current merge state, H3 eligibility, failures, gating exclusions, and timing reliability. It atomically emits strict `verification.json` with complete failure details and returns nonzero unless every check passes. Accepted failures/exclusions/unreliable timing require archived experiment policy with a non-empty selector and written reason; blanket implicit success is impossible. Regression tests cover clean completion, missing cells, stale merge, corrupted inputs, failure/timing policy, and severe-quality exclusions. |
| H5 | **DONE** | Reproducible publication/AD-AE bundle | `benchkit artifact build/verify` creates and offline-verifies a content-addressed bundle containing configs, manifests, canonical/raw rows, H4 report, table inputs, logs, build/environment recipes, notices/licenses, and a one-command checksum-locked smoke reproduction. Generated inputs and included outputs carry source/generator metadata. A final H100/FZGM bundle was verified offline and its generated smoke independently passed all 12 H4 checks on 2026-08-24. |
| H6 | **DONE** | Test, CI, packaging, and documentation closure | CPU tests cover fake-CLI run/resume/shards/merge/failure, adapter parser fixtures, schemas/identity, dataset integrity, verification, and golden/tamper artifact cases. CI tests Python 3.10/3.12 from an exact development lock; the clean-install script passed all 55 tests in a fresh venv. Machine-built FSZ/lsCOMP ELF files were removed from tracking while their sources/build recipes remain. Reference-tool documentation now states the external-source limitation accurately. |

#### M6 compatibility and release rules

- Curated historical baselines are evidence and are never rewritten in place. Readers
  accept legacy schema v0; new writers emit only the current version.
- Schema changes are additive within a version. A semantic or representational change
  increments the relevant schema version and ships a tested migration/read-compatibility
  path.
- A session is **publication-grade** only after H4 verification succeeds. `status: ok`,
  a zero process exit code, or a complete-looking row count is not a substitute.
- H1 through H4 are the minimum gate before using the new format for an authoritative
  paper baseline. H5 and H6 are the minimum gate before handing it to artifact evaluators.
- Each completed work package updates this table, `docs/RUN_LEDGER.md`, tests, and any
  affected adapter contract. This section is the technical roadmap; the run ledger is
  the operational status record, so no separate planning document is maintained.

---

## 10. Decision log

| # | Decision | Rationale |
|---|---|---|
| D1 | **Python** orchestration + analysis; compressors driven as **subprocesses**. | pandas/matplotlib/pydantic ecosystem; subprocess isolation matches heterogeneous CLIs and keeps the harness language-agnostic about compressors. |
| D2 | **Amended by H6:** reference compressors were planned as pinned submodules, but the implemented system uses external source/SDK builds. | The old text described a target that never landed. H3/H4 now require exact version/source/build/patch and executable identities; H5 ships recipes and notices, while `docs/reference-tools.md` states the remaining acquisition limitation. |
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
| D13 | **Sharding + resume** originally used deterministic `cell_key`; H2 replaces resume with `execution_id` while preserving the old field as a read alias. Each shard writes its own `runs.shard-k-of-N.jsonl`; merge dedupes by logical cell. | SLURM job arrays split a big matrix across tasks with no append contention; jobs resume idempotently only when every execution-relevant input matches. |
| D14 | **Per-shard provenance** (not one shared manifest). | Each array task may land on a different node/GPU — capturing GPU+scheduler+software per shard is correct, and avoids a write race. |
| D15 | **Timing reliability = variance-primary, throttle-reasons-secondary.** `cv` over kept reps decides `timing_reliable`; concurrent GPU sampling is diagnostic. | Clocks can't be locked on shared nodes; cv catches sub-sample-rate clock bounce that a clock query misses, while throttle reasons explain *why* when something is detectably throttling. |
| D16 | **Graph-mode plumbing**: a per-run-entry `graph: true` sets `RunSpec.graph` → adapter passes `--graph` → row records `graph_requested`/`graph_active`/`graph_reason` (parsed from report-json's `"graph"` object). **benchkit does not maintain a stage compatibility matrix** — the FZGM library validates a captured DAG stage-by-stage (`CompressionDAG::setCaptureMode`) and throws a descriptive error naming incompatible stages; the CLI's `--graph` (added 2026-07-03) catches that and falls back to normal execution, reporting the outcome in `report-json` schema 1.1. Confirmed live on BigRed200: `cusz.toml` (Huffman) falls back cleanly; `cuszp2.toml`/`cuszp3.toml` (linear ABS Quantizer → Lorenzo/TiledLorenzo → AdaptiveBitpack forward) capture and replay correctly. | benchkit's plumbing (2026-07-02) predated the CLI flag (2026-07-03) intentionally, so the contract could be specified up front; one mismatch surfaced on first real integration test — the adapter expected flat `graph_active`/`graph_incompatible_reason` keys but the shipped schema nests them under `"graph": {...}` — fixed in the adapter once seen against the real binary. NOA-mode Quantizer pipelines (pfpl, quantizer_lorenzo_bitpack) still need a precomputed value base (`setValueBase`) to avoid a D2H scan; not yet tested. See `docs/adapters/fzgm.md` "Graph mode". |
| D17 | **Per-run-entry dataset scoping** (`only_datasets` / `skip_datasets` on a `RunEntry`, mutually exclusive). | Some FZGM presets are dimensionality-specific (e.g. `TiledLorenzo`-based cuSZp3 presets are 2-D/3-D only; `GInterp`-based cuSZ-Hi presets are structurally 2-D+ only, see `docs/adapters/cuszhi.md`). Forcing one `pipeline:` across the whole dataset x field matrix silently produced nonsense on mismatched fields (E12: cuszp3's 2-D preset fed 1-D HACC data collapsed to degenerate tiles, CR 0.69). Scoping a run entry to the fields its preset actually supports is config, not a runner special-case — consistent with D7 "config over code." |
| D18 | **Aggregate CR report** (`report --aggregate`) computes both **ratio-of-sums** (`sum(original_bytes)/sum(compressed_bytes)` across fields, size-weighted) and **geometric mean of per-field CR** (`(Π CR_i)^(1/n)`, every field weighted equally), grouped by (compressor, variant, pipeline, error_bound). | Both appear in the compression literature as "the" multi-field CR and answer different questions — ratio-of-sums is what SDRBench-style papers usually report as an "overall CR" (a big field dominates); geomean is more robust to one huge/tiny field skewing the number. Reps of a cell are collapsed via the *median* compressed_bytes first (deterministic compressors shouldn't vary CR across reps; median is a defensive statistic against a rep-level fluke, not an assumption that they will). Implemented stdlib-only in `benchkit/analysis.py`, ahead of the pandas-based M4 analysis layer, because the FZGM-vs-native validation matrix needed it now. |
| D19 | **Emulated relative bounds for tools with no native range/maxabs-relative mode** (`read_range_stats` helper in `adapters/base.py`): the adapter reads the input file itself, computes `max-min`/`max\|x\|`, multiplies by the canonical `eb`, and passes the product as the tool's native **absolute** bound. Used by zfp, MGARD, and SPERR (all `abs`-only natively). SPERR additionally subtracts half an output-dtype ulp at field `maxabs` from native `--pwe`, because its final f32/f64 output cast occurs after internal error control. | Nearly every experiment config in this repo sweeps `rel_range`; tools that only accept an absolute tolerance would otherwise be unusable in those same configs. Since the emulation reads the identical file the harness's own `eb_ok` check reads, the two computations of range/maxabs agree, so the emulated bound is cross-tool-comparable. The SPERR guard was required by a real f32 row that exceeded the nominal bound by 1.5% solely after output conversion. The same work surfaced MGARD's `-s 0` issue: it does not bound pointwise max error even in `-em abs` mode (see `docs/adapters/mgard.md`) — use `-s inf`. |
| D20 | **cuSZp2/cuSZp3 source patched** (in `~/compressors/`, outside this repo) to cache their internal scratch-buffer `cudaMalloc`s across calls instead of allocating+freeing on every compress/decompress, plus `sm_90` added to both CMakeLists.txt. | On the JetStream2 H100 (GPU-passthrough cloud VM), native cuSZp2/cuSZp3 throughput measured 20-100x below expectation (~8 GB/s vs. an A100 baseline of ~58 GB/s on the same cell) while every other tool measured correctly. Root-caused with `nsys`: a single `cudaMalloc` call took up to 451 ms on this VM vs. ~145 μs of actual kernel time — cuSZp calls `cudaMalloc`/`cudaFree` for 3 small scratch arrays on *every* compress/decompress call, an allocator-latency tax that's noise on bare-metal hardware but dominates cuSZp's sub-millisecond kernels specifically. Fixed by caching the scratch allocation (grow-only, reused across calls); confirmed 8.0/7.8 → 159.6/207.1 GB/s (CESM) and ~124/131 → 653/1205 GB/s (NYX 3-D) through the real adapter, CR/PSNR unaffected. See `docs/adapters/cuszp.md`. Lesson for any future cloud/passthrough GPU site: a tool whose own self-reported throughput looks implausibly low relative to its known bare-metal numbers is a `cudaMalloc`-in-the-hot-path smell before it's a "this cloud GPU is slow" conclusion — profile with `nsys`/hardware counters, don't trust the tool's own timer at face value. |
| D21 | **Full audit of all native reference compressors for the two D20 failure modes** (missing `sm_90` SASS, `cudaMalloc` inside the timed region), following the cuSZp fix. Findings: **sm_90 missing/silently overridden** in `FZ-GPU` (no `-arch` flag at all in the Makefile → nvcc's ancient sm_52 default, ran only via PTX JIT on this box) and `lsCOMP` (unconditional `set(CMAKE_CUDA_ARCHITECTURES 80 86)` in CMakeLists.txt — the *identical* cache-shadowing bug cuSZp had, silently discarding any `-DCMAKE_CUDA_ARCHITECTURES=90` passed at configure time). Both fixed (`FZ-GPU/Makefile` gets an explicit `-gencode arch=compute_90,code=sm_90`; `lsCOMP/CMakeLists.txt`'s `set()` now includes `90`) and rebuilt; verified via `cuobjdump -lelf`. cuSZ, cuSZ-Hi, PFPL, and MGARD were already correctly targeting `sm_90` (either no hardcoded `set()` at all, or one guarded with `if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)`). **`cudaMalloc`-in-hot-path**: none of the other tools have it — cuSZ and cuSZ-Hi already use CUDA-event device-only timing with an in-process `--repeat` (a *different*, pre-existing bugfix: an aliased-pointer use-after-free, not an allocator issue — see `docs/adapters/cusz.md`); FZ-GPU allocates once per `runFzgpu()` call, outside its `compressionStart`/`compressionEnd` window, with its own in-process `repeat` loop; PFPL allocates once before its `NUM_RUNS` loop and times only the kernel launches inside it. cuSZp2/cuSZp3 were the only tools with allocation *inside* the timed region. | The user asked, after the cuSZp fix, whether the same two problems (missing Hopper codegen, malloc-in-hot-path) were silently present in the other reference tools before trusting any of them for paper-quality numbers on this VM. Worth recording that `set(CMAKE_CUDA_ARCHITECTURES ...)` with no `if(NOT DEFINED ...)` guard is a recurring anti-pattern across this codebase's vendored compressors (cuSZp, lsCOMP) — anyone adding a new reference tool should grep its CMakeLists for a bare `set(CMAKE_CUDA_ARCHITECTURES` and either add `90` directly or gate it behind `if(NOT DEFINED ...)` before trusting `-DCMAKE_CUDA_ARCHITECTURES=90` to reach the compiler. Also worth recording the contrast with FZGPUModules itself (see below) — it never had this class of bug because it was designed against a stream-ordered pool allocator from the start. |
| D22 | **How FZGPUModules avoids the D20 class of bug by construction.** `Pipeline` buffers use `MemoryStrategy::PREALLOCATE` and a `MemoryPool` (`src/mem/mempool.cpp`) built on CUDA's native stream-ordered pool allocator (`cudaMallocFromPoolAsync`, with a synchronous `cudaMalloc` fallback only when the pool is unavailable, e.g. vGPU). The CLI's `-b --runs N` loop (`cli.cpp`) runs one **untimed warmup** `pipeline->compress()` call first (warms the pool, JIT, GPU clocks), then times each subsequent `compress()`/`decompress()` call with `std::chrono` bracketing just the call + `cudaDeviceSynchronize()` — no allocation happens inside that window because the pool-owned buffers persist across calls (freeing pool-owned pointers is explicitly forbidden — see the Memory Ownership table in FZGPUModules' `docs/architecture.md`). Each call also independently records a device-only `dag_elapsed_ms` via internal CUDA events (`pipeline->getLastPerfResult()`), giving two corroborating timing sources per run. | Directly answers "how does FZGPUModules time in relation to cudaMalloc calls" (asked alongside the D21 audit): unlike cuSZp's ad-hoc fix (a hand-rolled grow-once scratch cache retrofitted after the fact), FZGPUModules never allocates inside its timed region in the first place — allocation-vs-timing safety was a design constraint (`PREALLOCATE` + pool allocator + ownership rules), not a bug that had to be found. This is the reason FZGM's own numbers were never in question during the D20 investigation even though they run on the same VM as the broken cuSZp numbers. |
| D23 | **Native cuSZp2 (not v3) corrupted output on this H100 for ~11/24 cells — root-caused and fixed.** Found while building an A100-vs-H100 comparison (CR/PSNR were expected to match exactly; they didn't for native cuSZp2's `plain`/`outlier` modes on HACC/HURR). Empirically ruled out this session's own scratch-buffer-cache and `TIMING_REPEATS` changes (both independently reverted, bug persisted bit-for-bit each time) and int32 quantization overflow (checked HACC/vx's actual worst-case code, ~2547, six orders below `INT32_MAX`). **Root cause, confirmed by direct instrumentation** (a temporary per-block failure counter added to `examples/cuSZp.cpp`): the `excl_sum` `__shared__` variable in the decoupled-look-back GPU-wide prefix-sum scan is only ever assigned for `warp>0` — block/warp 0 (which every grid has) reads it uninitialized at `base_idx = excl_sum + rate_ofs`, undefined behavior whose value depends on physical shared-memory bank residue, plausibly differing by GPU architecture/driver. Instrumentation showed exactly 2 of 8575 blocks corrupted for the worst cell (block 0 itself, plus block 5 — collateral damage from block 0's compressed bytes landing at a garbage offset and clobbering block 5's). **Fix:** one line, `excl_sum = 0;`, added to the pre-existing `if(warp==0)` branch in all 8 kernel functions (compress/decompress × plain/outlier × f32/f64). Confirmed **cuSZp-V3 already has this exact line** in the same branch — a one-line backport from v3, not a novel fix. Verified: all 24 previously-checked native-cuSZp2 cells now pass cuSZp2's own internal error check (0 failures), `cuSZp_test_f32`/`cuSZp_test_f64` self-tests still pass. Patch lives only in `~/compressors/cuSZp-V2.0.1/src/cuSZp_kernels_{f32,f64}.cu` (not upstreamed). See `docs/adapters/cuszp.md`. | The user's instinct to sanity-check "CR/PSNR should be the same between platforms" before trusting a performance-only comparison caught a real, previously-unknown correctness bug — worth verifying that assumption quantitatively rather than assuming it, even (especially) when the numbers being compared are throughput. The user is a coworker of cuSZp's author and explicitly wants bugs like this surfaced upstream, which is why this got a full fix rather than just a documented caveat. |
| D24 | **`results/baselines/` — curated, git-tracked cross-machine result snapshots**, carved out of the otherwise-gitignored `results/` (`.gitignore`: `results/*` + `!results/baselines/`). Each `results/baselines/<id>/` holds `runs.jsonl` + `provenance.json` (harness schema, copied verbatim when possible) plus a `metadata.yaml` (GPU/site/date/known-issues, schema documented in `results/baselines/README.md`). Added `scripts/reconstruct_runs_from_stdout.py` (best-effort `runs.jsonl`/`provenance.json` from a stdout/SLURM log when the harness's own output was never copied off the source machine — explicitly marked lower-fidelity, missing per-rep timing/stages/gpu_sampling) and `scripts/build_comparison_artifact.py` (generalized version of this session's one-off comparison-artifact scripts: takes two baseline directories, cross-checks CR/PSNR agreement before charting throughput, flags disagreements generically rather than assuming which side is at fault). Seeded with the two baselines from the A100-vs-H100 investigation: `a100-bigred200-slurm7562329` (reconstructed from a stdout log, carries its own uninitialized-`excl_sum` residual per D23) and `h100-jetstream2-20260719` (native, full-fidelity, the first baseline with verified-correct native cuSZp2 output). | Building the A100-vs-H100 comparison required manually locating a stray SLURM log on this machine, hand-parsing its summary table, and writing one-off scripts against hardcoded absolute paths — none of which would work from a different machine or survive this session ending. The user wants to keep doing cross-machine comparisons as more GPUs/sites are benchmarked, so the reusable pieces (data format, parsing script, artifact-building script) needed to move from scratchpad one-offs into the repo, not just the data. `metadata.yaml`'s `known_issues` field exists specifically because D23 was found by two baselines *disagreeing* on cells that should have matched — future baselines should record that kind of finding right next to the data, not just in a decision-log entry that's easy to miss when comparing against an old snapshot later. |
| D25 | **`retain_compressed: false` added as a new config option, defaulting off** (mirrors the existing `retain_decompressed`, D11). The compressed artifact (`c.fzm`/`c.cuszp`/`c.cusza`/etc.) is checksummed (`compressed_sha256`, new row field) then deleted after `benchmark()` completes for that cell, unless the experiment config sets `retain_compressed: true`. `compressed_bytes` (the size) was already captured independent of the file persisting, so this loses nothing CR/throughput analysis needs. All checked-in experiment configs now set it explicitly (`false`, next to `retain_decompressed`) for the same clarity-over-implicit-default reason D11 set that pattern. | The root disk filled to 98% (4.2GB free) during this session's repeated full `fzgm_vs_native.yaml` reruns — `retain_decompressed: false` was already doing its job (no `d.bin` files survived), but nothing was cleaning up the *compressed* file, and at up to 1.3GB/cell for HACC that's ~16GB per full 210-cell session with no corresponding benefit once the row's CR/PSNR/throughput/checksum are safely in `runs.jsonl` (and, for cross-machine-worthy runs, in `results/baselines/` per D24). Same reasoning as D11, just the other file. |
| D26 | **No-retention cleanup covers adapter-owned benchmark scratch as well as the canonical artifacts, and aggregate CR gained a per-dataset grouping (`report --aggregate --by-dataset`) plus a spread statistic.** PFPL first exposed that `benchmark()` can write a full-original-size `d_bench.bin` which is not the path returned by `decompress()`. PFPL deletes its scratch locally, and the runner now additionally sweeps every non-diagnostic regular file from each cell directory in a `finally` whenever both retention flags are false. This makes the contract adapter-independent and covers successful, failed, and interrupted cells while preserving logs, JSON reports, and rendered TOML. Separately, `aggregate_cr()` also reports `gsd_cr` (geometric standard deviation), `min_cr`/`max_cr` and their field names, and `--by-dataset` prepends `dataset` to the group keys. | Sizing full-corpus sweeps repeatedly surfaced this class of leak: PFPL once left 4.99 GB across 12 files, and the 2026-08-28 standalone-reference sweep left **48.98 GiB across 367 SZ3 `d_bench.bin` files**, filling the root filesystem despite both retention flags being false. Adapter-local cleanup remains useful for minimizing lifetime, but the user-visible no-retention policy must be enforced centrally because future adapters can introduce differently named timing scratch. The aggregation work is D18 finished for multi-field reality: with 79 CESM fields in one group a bare mean hides the dataset, and pooling datasets is actively misleading because ratio-of-sums is size-weighted. Geometric (not arithmetic) SD reports spread as a factor, and a single-field group reports `-` rather than `0.0`. |
| D27 | **FZGM pipeline presets now render their float `input_type` from the field dtype** (`PipelineToml.render(..., dtype=)`, plus `check_dtype()` for the verbatim `from_toml` path), and **the cuSZ-Hi presets switched from `MINIMAL` to `PREALLOCATE`**. | Two bugs surfaced by adding MIRANDA (f64) and CESMATM-3D (large 3-D) to the corpus; both looked like FZGM library bugs and only one was. (a) Every preset hardcodes `input_type = "float32"` on its raw-consuming first stage, and the renderer substituted error bound/dims/input_size but never dtype. FZGM has always supported f64 (`QuantizerStage<double,uint32_t>` is instantiated in `src/pipeline/config.cpp`), so this was purely benchkit-side. The dangerous half is that it did not fail cleanly: TiledLorenzo/GInterp aborted with "Benchmark size mismatch", but the plain-Lorenzo pipelines *silently returned garbage* — PSNR `nan`, `eb_satisfied` false, CR exactly 64.00/128.00 from degenerate all-zero codes. 4 of 8 f64 cells were the silent kind. After the fix all 8 pass at PSNR 71.39 dB matching native exactly, and fzgm cuszp3_plain reports CR 6.32 vs native 6.32. Only *float* input_type lines are rewritten (audited: all 21 float-typed keys across `configs/pipelines/*.toml` belong to the first stage); a preset with no float input_type now raises rather than silently no-op-ing. (b) Under `MINIMAL`, cuSZ-Hi corrupts memory on large 3-D fields — the compressed size varies slightly between successive `compress()` calls on a reused pipeline (23,079,728 then 23,079,732 bytes on CESMATM-3D/CLDICE), so the inverse buffer sized from the previous run is undersized, and the illegal access surfaces against whichever stage syncs first (reported at `rre_stage.cu:475`, which is NOT the culprit — a reminder that a CUDA IMA names the next synchronization point, not the faulting kernel). This is exactly the failure the E19 note in `ginterp_stage.cu` predicts for reused MINIMAL pipelines. PREALLOCATE fixes it, is byte-identical (CR 185.10x / PSNR 75.75 dB on NYX), and is *faster* (compress 81.1 -> 95.5 GB/s, decompress 74.2 -> 80.1) at ~1.6x peak memory. **Caveat: cuSZ-Hi throughput is therefore not comparable with pre-2026-07-28 baselines**, which ran MINIMAL; CR and PSNR are. Worth testing whether this also explains the unexplained cuSZ-Hi (cr) H100 slowness recorded in the H200 baseline metadata — MINIMAL allocates per call, and D20 established that this GPU-passthrough VM has pathological `cudaMalloc` latency. |
| D28 | **Delta's GPU partitions need a single-node/8-GPU job, not `submit_full_corpus.slurm`'s `--array=0-7 --exclusive` — and the MI100 node is architecturally heterogeneous.** Added `scripts/submit_delta_multigpu.slurm`: one job, one `--exclusive` node, 8 concurrent shards each pinned to its own GPU, with the device list **probed at runtime by `gfx` arch** rather than assumed to be `0..7`. | Two separate findings from the first full-corpus run on Delta. (a) **Geometry:** `MaxTime` on `gpuH200x8`/`gpuMI100x8` is 2-00:00:00, not the 8 h the runbook assumes, so sharding here is for throughput, not to survive a wall limit; and `gpuMI100x8` is a *single* node while `gpuH200x8` is 8 nodes, so an 8-way exclusive array **serialises** on the MI100 and claims the whole partition on the H200. (b) **Heterogeneity — the expensive one:** `gpud01` carries 8 MI100s (gfx908) *and* one MI210 (gfx90a), with the MI210 at **device ordinal 6**. A naive shard-k -> ordinal-k map put shard 6 on hardware the gfx908-only HIP build has no code objects for, and all 607 of its cells died with `invalid device function`. `--gres=gpu:mi100:8` does **not** protect against this, because `--exclusive` exposes the whole node regardless. The failure was near-invisible: the job exited **0** with a full 4860-row count, `ctest` had passed 46/46 on that same node, and `provenance.json` reported `gfx908` for every shard — because `benchkit/gpu.py` enumerates via raw sysfs and ignores `ROCR_VISIBLE_DEVICES` masking, so it cannot attribute a shard to a device at all. It was only findable from the per-cell failure distribution (one shard at 607/607, the rest at 12-15). Lesson: on any multi-GPU node, **verify the arch of the device each shard actually got** rather than trusting the gres request, the exit code, or the row count; and treat a uniform-looking provenance block as unverified until `gpu.py` honours the visibility mask. |
| D29 | **`benchkit merge` now prefers the successful attempt when deduplicating by `cell_key`** (an `ok` row always beats a non-ok row; later wins among equals). Previously it kept the **first** occurrence. | Resume *appends*, so a cell that failed and was later retried has its stale `fail` row ahead of the good retry in the same shard file. First-occurrence-wins therefore resurrected the failure and silently discarded the recovered cell. On `fullcorpus-delta-mi100` this would have emitted **701 failures instead of 108**, throwing away 593 recovered cells — at a correct-looking 4860-row count and exit 0, i.e. indistinguishable from a good merge without inspecting statuses. This is not specific to that run: it corrupts **any** resumed session, which is exactly the requeue-until-done workflow `docs/running-the-full-corpus.md` prescribes for short-wall-limit sites like BigRed200. **Any baseline produced by merging a resumed session before this fix should be re-merged.** Same family as D23 — a silent-wrong-data bug caught only by checking a number that should have been predictable (108 documented LDS failures) against what the tool actually produced. | **Amended same day:** this now composes with D30's `--only-stale` supersede rule, so `merge` applies three ordered rules — an `ok` row always beats a non-ok row; among equals the later row in the SAME file wins (the newest re-measurement); and across files shard files beat `runs.jsonl` (a previous merge's output, hence older). Rule 1 deliberately outranks the other two: recovering a cell matters more than positional recency, and a later failure is never evidence that an earlier success was wrong.
| D30 | **Reported means are computed over a validity-gated population, not over every `ok` row** (`benchkit/validity.py`; `report --aggregate` gates by default and prints the audit above the table; `--exclusions` prints the audit alone; `--no-gate` reproduces ungated numbers). | `status == "ok"` means "both tools exited 0 and metrics were computed" — not "the numbers are usable". The 9,816-cell full-corpus sweep returned 9,416 `ok` rows of which 798 were not: 290 missed their error bound by >1.01x, 108 had infinite PSNR, and 12 had `cr <= 1` (native cuSZ at eb=1e-4 on EXAALT/HACC *doubled* the data size at PSNR -23.6 dB and still exited 0 — more dangerous than the 400 hard failures precisely because it survives an unattended `status == ok` filter). Measured effect on one cell of the paper table (CESM-2D / cuszp2_outlier / eb=1e-4): geomean CR 10.44 -> 9.96 and the reported best field stops being an artifact. Three design choices worth keeping: (a) **degenerate fields are detected from the data, not a name list** — the rule is "every compressor reconstructed this field with exactly zero error", which found CESM-2D/SFCLDICE and SFCLDLIQ (both entirely zero, verified `unique() == 1` over the full file) without hardcoding them; the corpus went from 8 to 13 dataset families in a week and a name list would rot. It also self-validates: all 108 non-finite-PSNR rows are explained by those two fields, leaving `psnr_nonfinite` at 0. (b) **Marginal bound misses (<=1.01x) are retained, not dropped** — 374 of 388 sit at eb=1e-4 and hit every compressor including native cuSZ, so they are f32 round-off near f32 resolution, and gating them away would hide a real uniform property; they are counted and printed instead. (c) **Nothing is deleted or rewritten** — `runs.jsonl` stays the raw record and the gate is applied at read time, so every excluded row can be recovered and every reason code carries a written rationale that a paper can cite. See open question 7 for the unresolved half. |
| D31 | **A `lossless` canonical error mode, plus two validity-gate carve-outs keyed on it** (`benchkit/config.py` `CANONICAL_MODES`/`UNBOUNDED_MODES`; `benchkit/validity.py`; `metrics.compute_quality` basis `"lossless"`). Added to make the nvCOMP comparison possible. | nvCOMP's codecs (Zstd/LZ4/Deflate/GDeflate/ANS) compress a byte stream and take **no error bound**, so they do not fit the abs/rel_range/rel_maxabs model at all — the same wall that left MANS and lsCOMP as stubs. `lossless` says what is actually true: no lossy stage, contract is bit-exactness. `compute_quality` then checks `max_abs_err == 0` with **no `eb_tol` slack** (there is no bound for round-off to hide under) rather than dividing by a zero bound and reporting `err_over_bound = inf`, which is also not valid JSON. The two gate carve-outs are the non-obvious part, and both are silent-wrong-data bugs of the D29/D30 family if omitted: (a) **`degenerate_fields()` ignores lossless rows entirely.** That detector infers "this field is constant" from "no compressor produced any error on it" — sound over lossy runs, meaningless over lossless ones, where every row is exact by construction. Without the carve-out an all-lossless session marks *every* field degenerate and the gate drops 100% of it, at exit 0 and a full row count. (b) **`cr <= 1` becomes `lossless_expansion` and is RETAINED.** For a lossy codec `cr <= 1` has never been seen without catastrophic quality beside it (D30: PSNR -23.6 dB), so it is a corruption signal. For a lossless codec it is an ordinary measurement — nvCOMP LZ4 compresses CESM-2D/CLDHGH to **0.996x**, because raw f32 mantissas are near-incompressible byte-wise and framing overhead is real. Gating those away would bias every lossless CR mean *upward* by deleting exactly the cases where the codec lost. `psnr == inf` is likewise reported as `lossless_exact` (quality aggregates only; CR and throughput retained) rather than as the unexplained-anomaly code `psnr_nonfinite`. Because the gate keys all of this on the `error_mode` field, a mislabeled row corrupts the gate for every *other* row in the session — so the FZGM adapter **refuses** a pipeline that declares an `error_bound` under `lossless` instead of shipping it. See docs/adapters/nvcomp.md. |
| D32 | **The FZGM-vs-nvCOMP comparison runs in two framings, and the end-to-end one is not among them** (`configs/experiments/nvcomp_vs_fzgm_lossless.yaml`, `nvcomp_vs_fzgm_backend.yaml`). | FZGM's `gpu_zstd` preset is an error-bounded **lossy** compressor (LorenzoQuant -> GPULZ split -> Huffman + ANS x3); nvCOMP's Zstd is a **lossless** byte coder. On CESM-2D/CLDHGH the preset reaches 10.87x at rel_range 1e-3 against nvCOMP Zstd's 1.14x — a 9.5x gap that is **the Lorenzo predictor**, not the Zstd implementation, and quoting it as a Zstd-vs-Zstd result would be wrong. So: (1) *lossless head-to-head* strips the predictor (`gpu_zstd_lossless.toml`) so both consume identical raw f32 bytes under identical bit-exactness; (2) *back-end isolation* hands both coders the uint16 Lorenzo quant codes FZGM's own preset feeds its back end, extracted by `scripts/extract_quant_codes.py` via a predictor-only pipeline and `benchkit/fzm.py` (an .fzm **reader** — benchkit never writes that format; a second writer would be a second thing to keep in sync with fzgmod-cli). Framing 2 is the fairer one for FZGM's back end, which was tuned on Lorenzo residuals rather than raw mantissa bytes, and it is decisive: on 3 CESM-2D fields FZGM gets 5.43/9.36/10.63x at ~14-15 GB/s compress against nvCOMP Zstd's 4.46/7.66/8.56x at ~0.8-1.1 GB/s. Rows from framing 2 carry a **back-end** ratio over an already-2x-reduced intermediate with the outlier streams excluded, and `original_bytes` is the codes size — they must never be quoted as a compression ratio for the field, nor pooled with raw-field rows. |
| D33 | **Every row now records host wall time next to device time, plus `*_host_over_device`** (`BenchmarkResult.compress_host_ms_all`, `runner._row`). A cross-tool device-only comparison is only valid if you know what each tool's bracket excludes. | Found while auditing the FZGM-vs-nvCOMP throughput gap, which looked too large to be real. FZGM's `device_ms` is `dag_elapsed_ms`: a CUDA event pair around `dag->execute()` only. On `gpu_zstd_lossless`/CLDHGH that reports **1.26 ms device against 3.60 ms host** — 2.33 ms, 2.9x, of excluded host work. It is **specific to split mode**: the single-stream `gpulz->ans` chain shows 0.016 ms (1.02x), so the cost is assembling GPULZ's four coded ports into one archive on the host. nvCOMP's manager writes one contiguous device buffer and measures **6.14 ms device / 6.14 ms host** (1.00x) — it excludes essentially nothing. So on CLDHGH compress the two framings give very different answers: device-only **20.6 vs 4.2 GB/s (4.9x)**, host wall **7.2 vs 4.2 GB/s (1.7x)** — and against nvCOMP's *best* chunk size, host wall is **7.2 vs 8.8 GB/s, i.e. nvCOMP wins**. None of that is visible from `device_ms` alone. Device-only remains the reported throughput (it is the standard kernel-cost metric and the right denominator for stage attribution), but the host ratio is now in the row so the exclusion can never again be invisible. Whether FZGM's archive assembly is inherent or just not yet moved onto the device is an implementation question, not a measurement one — but until it moves, a paper claim about end-to-end compress throughput must use the host number or state the exclusion. |
| D34 | **nvCOMP is benchmarked at a swept chunk size, not only at the vendor-recommended 65536.** | nvCOMP's own header says "For best performance, a chunk size of 65536 bytes is recommended"; on this H100 that is wrong for Zstd by up to 2.1x. Chunk count *is* the parallelism — 25.9 MB / 64 KB is 396 chunks over 132 SMs, ~3 per SM. Measured compress on CESM-2D/CLDHGH: 4 KB 7.55, 8 KB 8.08, **16 KB 8.84**, 32 KB 6.87, 64 KB 4.22, 256 KB 1.61 GB/s, with only 0.2% CR spread between 16 KB and 64 KB. The effect shrinks where chunks are already plentiful (NYX 536 MB: 8.80 vs 7.70, 1.14x), so it is worst exactly on the small fields that dominate a corpus by count. FZGM's side of this comparison is tuned by measurement (chunk_size, word_size, match_level), so quoting nvCOMP at an untuned default would not be a fair fight. Both settings are run; the better one is what gets quoted. Related: the first `--reps 1` decompress was also reading 48% slow (20.4 vs 13.8 ms) because `nvcomp_cli` pre-warmed only the compress path — benchkit's warmup-rep drop hid it from the median, but that asymmetry was ours, not nvCOMP's, and is fixed. |
| D35 | **A 2.9x host/device gap traced to a concat kernel using one block per segment; `benchkit stale`-visible presets for SZ3 comparison added.** | Following D33's host-time audit to its cause: FZGM's `concatOutputs()` gather kernel launched `<<<n_segs, 256>>>`, one block per output port. GPULZ split mode puts ~99% of the archive in `literals`, so a single 256-thread block was copying ~23 MB — **2.27 ms, 60.7% of the pipeline's entire GPU time, ~20 GB/s on a 3 TB/s H100** — and because it runs *after* the DAG event bracket it appeared in no per-stage table, only as the host/device ratio D33 had just started recording. Fixed upstream in FZGPUModules (grid `(256, n_segs)`, all blocks in a row cooperating): **112x on the kernel, and CLDHGH split-mode compress 7.2 -> 19.5 GB/s host-wall**, output bit-identical, 48/48 tests green. This reverses D33's conclusion — FZGM now leads nvCOMP Zstd ~2.9x on end-to-end compress rather than trailing it. Two lessons the harness keeps: a device-only figure can hide a *device* cost if the bracket does not span the whole operation, and `host_over_device` is what surfaced it. Also fixed there: `.fzm` archives were not byte-reproducible (FZM header structs wrote uninitialized tail padding — 12 bytes differed between runs), which matters because rows record `compressed_sha256` and D24 baselines compare on it. |
| D36 | **CR comparisons against a CPU reference are made at matched PSNR, not matched error bound** (`docs/results/sz3-vs-ginterp-gpu-zstd.md`). | SZ3 vs the GPU analogue (`ginterp_huffman_gpu_zstd.toml`, the same interpolation+Huffman+Zstd structure) reads as GPU = **0.409x** of SZ3's CR at matched bound — but the GPU pipeline also returns **+2.0 dB median PSNR**, i.e. the two are not at the same operating point and the 'loss' is partly quality nobody asked for. On a rate-distortion basis (CR at matched PSNR, log-interpolated in range) it is **0.557**, and **0.799 excluding one field** — 6 of 7 fields land within 11-30% of SZ3's curve, which is the answer to 'is the GPU port faithful'. The exception, NYX/baryon_density, is 0.06x and structural, not polish: at eb=1e-2 the field is **22 distinct quant codes with one covering 99.999%** and a mean run length of 78,952 elements (158 KB), so (a) Huffman's >=1-bit-per-symbol floor alone costs 16.8 MB against SZ3's ~20 KB total, and (b) GPULZ's independent 2048-byte chunks cannot span a 158 KB run. Measured: swapping Huffman for **RLE is worth 4.6x** (855 -> 3,927x, gap 32x -> 6.9x); Huffman after RLE adds nothing; raising `chunk_size` to 4096 changes nothing, as expected for limit (b). Skewed near-constant fields are not rare in this corpus, so a hard-coded Huffman back end pays that floor repeatedly. |
| D37 | **A reference tool's version is pinned, recorded in provenance, and checked against upstream — not assumed current.** | The nvCOMP work started on **5.2.0.10**, taken from an unrelated project's vendored submodule because it happened to be on disk. It was already a minor release behind: **5.3.0.16** shipped 2026-07-14. Checked properly (PyPI `nvidia-nvcomp-cu12` for the version list, then the redist JSON at `developer.download.nvidia.com/compute/nvcomp/redist/` for the C++ SDK tarball — the Python wheel has bindings only). Measured 5.2 vs 5.3 on CESM-2D/CLDHGH, 8 reps x 3 trials: **compression ratios bit-identical in all 8 configurations**, Zstd compress **-2.6%**, and **ANS compress +29-34%** (83.1 -> 111.2 GB/s at 16 KB, non-overlapping trials). So the exposure was real but landed on an algorithm outside the headline comparison, and the direction was favourable-to-nvCOMP, meaning no published FZGM-vs-Zstd conclusion was ever at risk. That is luck, not process. The process fix: `NvcompAdapter.provenance()` now reads the SDK version from `$NVCOMP_ROOT/lib/cmake/nvcomp/nvcomp-config-version.cmake` and records `nvcomp_version` in every session manifest, `~/compressors/nvcomp` is a symlink with 5.2.0.10 retained beside it for A/B, and docs/adapters/nvcomp.md carries the two commands that check for a newer redist. Generalizes: every reference compressor here is a hand-built tree or hand-unpacked tarball with nothing forcing it current, and a 'we benchmarked against X' claim is only as good as the recorded version. |
| D38 | **A type-aware reference coder must be told its element type; nvCOMP's default (`uchar`) is not a neutral choice but a wrong one.** `nvcomp:bitcomp`/`nvcomp:cascaded` REQUIRE `dtype=` and the adapter refuses to guess. | Adding Bitcomp and Cascaded to `tools/nvcomp_cli` exposed the sharpest instance yet of D34. Unlike the seven byte coders, both model the input as an array of a declared element type — Cascaded's delta and bitpack passes operate on elements, Bitcomp's whole scheme is type-directed — and nvCOMP defaults that type to `NVCOMP_TYPE_UCHAR`. Measured on CESM-2D/CLDHGH raw f32: at the default, **Bitcomp 0.996x and Cascaded 0.998x**; at the correct 4-byte width, **1.437x and 1.515x**. Defaulted, both would have been written up as 'compresses nothing'; correctly typed, **Cascaded at 1.539x (its best scheme) beats every byte-oriented coder in either library on raw f32** — nvCOMP Zstd 1.184, Deflate 1.183, FZGM GPU-Zstd 1.131 — because it is the only nvCOMP entry that does any *prediction*. That is FZGM's own thesis arriving from the other direction, so it is the last place a silent bad default could be afforded. Two corollaries. (a) **nvCOMP 5.3 has no f32 element type** — `NVCOMP_TYPE_FLOAT` (enum 8) was removed and only FLOAT16 survives, so f32 can only be described as 4-byte *integer* width; that is a modelling limit of the comparison and must be stated wherever a raw-f32 Bitcomp/Cascaded row is quoted, not hidden behind an alias (hence no `float`/`f32` spelling is accepted). (b) **Cascaded's documented default scheme is also not its best**: sweeping `{rles, deltas, bp}` on raw f32, `bp=0` yields 0.998x in *every* configuration (bitpack is the only pass that actually compresses), one delta pass is worth 1.300 -> 1.539, and RLE hurts monotonically (1.539 / 1.531 / 1.515 / 1.508 for 0/1/2/3 passes) so the default `rles=2` costs 1.6%. Both the default and the swept-best are therefore run, per D34. **Amended after the broadened sweep (`parity-raw-broad`/`parity-codes-broad`, 885 cells): the 'RLE hurts' half of this is WRONG as a general claim and was an artefact of sweeping a single smooth field.** Across 15 raw cells `rles=0` is a geomean **0.917x** of the default, and across 45 codes cells **0.847x** — RLE is load-bearing wherever the data has runs, which CESM-2D/CLDHGH does not: HURR/CLOUD raw drops 5.81 -> 1.30 (4.5x) without it, and NYX-qcodes-1e-4/baryon_density drops 56.75 -> 10.04. The distribution is bimodal, not centred — `rles=0` is within 0.1% on run-free fields and catastrophic on the rest — so a geomean alone understates it. The `bp=0` finding and the dtype finding both DID generalise. Lesson: a knob sweep on one field establishes that a knob matters, never which setting to standardise on; sweep it on a field chosen for the property the knob exploits. Cascaded is also the one nvCOMP entry with a *composable* FZGM counterpart rather than a single-coder pairing — `{num_RLEs, num_deltas, use_bp}` maps onto rle/rre/rze/rare/raze, Lorenzo, and bitpack/adaptive_bitpack — so it is mirrored by a DAG, not paired. |
| D39 | **Bitcomp's `algorithm` must be passed on the DECOMPRESS side too; the NVCOMP_NATIVE header does not carry it.** `prep_algo_args()` re-emits `--level` for bitcomp only. | Deflate and Gdeflate's `algorithm` is an encoder-side choice the bitstream header describes, so decompression needs only `-a`. Bitcomp is not like that: a stream written with `algorithm=1` (sparse) and decoded by a manager constructed with the default `algorithm=0` **decodes to wrong bytes at exit 0**. Measured on EXAALT-qcodes-noa0.001/xx: `status: ok`, a plausible CR of 1.89, and **PSNR 20.85 dB on a codec that is lossless by construction** — the giveaway was only that a lossless row must report `inf`. Nothing in nvCOMP errors, and re-running the same stream with `--level 1` on both sides is bit-exact, so the failure is entirely in what the caller forgets to repeat. This is the D23/D29/D31/E23 family again — silent wrong data behind a successful exit — and it was caught by the harness's own bit-exactness check, which is the third time that check has been the only thing standing between a plausible number and a wrong one. Generalizes: **an option that only affects the encoder in one algorithm may be structural in another**, so 'the header carries it' has to be verified per algorithm rather than assumed from the family. |
| D40 | **Publication hardening uses versioned schemas and separates logical-cell identity from execution identity. Historical baselines are immutable legacy-v0 evidence.** | One `cell_key` cannot safely mean both “the scientific cell to compare/supersede” and “the exact execution safe to resume”: graph mode, resolved configs, datasets, binaries, and harness state affect the latter without necessarily changing the former. H1 introduces explicit schema versions and strict JSON; H2 introduces `logical_cell_id` for comparison/merge and `execution_id` for resume/provenance. Readers retain tested legacy-v0 support, and any migration writes a provenance-linked copy rather than rewriting curated evidence. See M6. |
| D41 | **The reference-compressor trees are now reproducible from a pinned manifest, not just documented.** `compressors/` in this repo holds `manifest.toml` (upstream repo + 40-char commit + submodule flag + patch list + build recipe + CLI path, per tool), `patches/<name>/*.patch` (all local source modifications as `git apply --3way`-able diffs against the pinned commit — the CLI timing harnesses of D16/D20/D21/D23, the FZ-GPU file-I/O patch, build-config bumps, the MANS u32/adm fix), `build/<name>.sh` (the recipes from `scripts/env-*.sh` comments and `MGARD/build_scripts/`, parameterised by `$CUDA_ARCH`/`$CC`/`$CUDA_ROOT`/`$JOBS` — no machine-specific paths), `vendor/` (full source for the 3 cuSZp H100-optimization forks, which have no upstream and diverge too far for a patch series), and `bootstrap.py` (stdlib-only: clone → checkout pin → submodules → apply patches → build; `capture <name>` re-freezes a patch from a locally-edited checkout; `status` diffs disk against the manifest). Checkouts still land at the same `~/compressors/<X>` paths, so `scripts/env-*.sh` is unchanged. Verified: every patch applies cleanly to a fresh clone at its pin, and a patched clone is byte-identical to the current local tree (cuSZp2/3 fully). This does **not** vendor upstream source into the repo or solve the AD/AE third-party-access limitation (`docs/reference-tools.md`) — it makes rebuilding on a new machine a scripted `~30–60 min` step instead of an archaeology exercise across shell-comment build recipes and uncommitted working trees. Supersedes the "external acquisition is an explicit limitation, not a pinned recipe" framing for everything except licensing/access. ROIBIN, which physically lived in `~/compressors/` but is a case study not a compressor, moved to `case_studies/roibin/` with its prior git history preserved as a bundle. | The ~3 GB `~/compressors/` tree was the single hardest thing to move between machines (BigRed200 / JetStream2 / Delta), and the irreplaceable part of it was tiny and unbacked-up: ~600 lines of uncommitted patches across 6 repos plus 3 fork directories with no git at all. Build artifacts don't transfer (arch/CUDA-specific) and upstream source is re-clonable, so the portable representation is the manifest + patches + recipes (~3 MB), not the directory. Same principle as D7 (FZGM consumed as a build, not vendored) and D12 (no hardcoded paths) applied to the reference side. |
| D42 | **FZGM's automatic-fusion feature is now "Pipeline Specialization"; benchkit tracks `FZ_SPECIALIZE` in execution identity and the CLI report's `specialization` block, keeping the legacy `fusion` names.** (2026-09) FZGM renamed the finalize-time optimizer to reflect that it does more than fuse kernels (single-pass decoupled-lookback, NVRTC codegen, a roofline profitability gate, and — new this cycle — *decompress*-side fusion for both the warp-register and chunk-cooperative families, plus specialization-aware buffer coloring that lowers peak pool memory). The public API and env var moved (`FZ_FUSION` → `FZ_SPECIALIZE`, `setFusionPolicy` → `setSpecializationPolicy`, …) with back-compat aliases, and `--report-json` now emits a `specialization` object **and** a byte-identical legacy `fusion` object. Benchkit change is deliberately minimal: the adapter still reads `rep.get("fusion")` (unchanged), and `_FZGM_BEHAVIOR_ENV` gains `FZ_SPECIALIZE` alongside `FZ_FUSION` so an off/auto A/B keyed on either name yields distinct `execution_id`s. New preflight: `specialization_vs_native_smoke.yaml` (native vs FZGM-staged vs FZGM-specialized, each launch `FZ_SPECIALIZE=off` then `auto`) and `specialization_memory_smoke.yaml` (peak-memory ablation: {staged, specialized} × {coloring on, off}, via `*_nocolor.toml` presets carrying `coloring = false`), driver `scripts/run-specialization-smoke.sh`. | The rename is upstream's, not ours, and the `fusion_*` row columns / `fusion` JSON block are load-bearing for existing baselines and parsers — so the guiding principle was "follow the rename in provenance identity and docs, change nothing a stored session or a report reader depends on". Adding `FZ_SPECIALIZE` to the behavior-env tuple is the one non-cosmetic requirement: without it, `FZ_SPECIALIZE=off` and `FZ_SPECIALIZE=auto` sessions would share an `execution_id` and the resume/merge logic would treat them as the same measurement. Same class as D40's logical-vs-execution identity split. |
---

## 11. Status of the original M1 open questions

This list is retained as design history. It is no longer a current M1 checklist:
questions 1, 2, 4, 5, and 6 are resolved; question 3 remains an infrastructure gap;
question 7 remains a scientific/correctness investigation. Current campaign actions
belong in `docs/RUN_LEDGER.md`.

1. **~~`fzgmod-cli` machine-readable output.~~ RESOLVED (2026-06-18).** FZGM now ships
   `--report-json <path>` (schema_version 1.0): a standalone JSON file with `tool`,
   `status`, `config`, `size`, `timing` (device_ms + host_wall_ms, per-rep `all` arrays),
   `throughput`, `memory`, `quality`, and an FZGM-only `stages[]` per-stage device-time
   breakdown. `--report-json` auto-enables profiling, so device timing and stages are
   always populated. Adapter integration rules captured in
   [`docs/adapters/fzgm.md`](adapters/fzgm.md); full schema lives in the FZGM repo at
   `memory/report_json_spec.md`. Other adapters still need stdout scraping; FZGM is the
   one we control and it is now clean.
   **Correction (2026-07-29), since FIXED upstream: "stages are always populated" was
   false for the compress phase.** `stages[]` carried decompress entries only — the
   9,816-cell full-corpus sweep produced 4,860 FZGM rows, every one decompress-only.
   Root cause was in FZGM, not benchkit: `Pipeline::setMemoryStrategy()` replaces
   `dag_` with a fresh `CompressionDAG` and dropped the flag `enableProfiling()` had
   set on the old one, so `collectTimings()` returned `{}`. Since the CLI enables
   profiling *before* `loadConfig()` and every preset carries a `memory_strategy` key,
   this fired on every config-driven run; decompress escaped only because its inverse
   DAG is built later. Fixed in `src/pipeline/compressor.cpp` with regression test
   `Profiling.SurvivesMemoryStrategyChange`. **Baselines captured before 2026-07-29
   still have decompress-only `stages[]`** — fine for decompress-side invalidation,
   but compress-side attribution needs a re-run.
2. **~~Decompressed-output retention.~~ RESOLVED.** Every executable benchmark adapter
   used for quality comparisons emits a raw decompressed array, and the harness computes
   quality from it. The runner deletes it after measurement by default
   (`retain_decompressed: false`) while retaining its checksum and metrics. A tool that
   cannot provide raw reconstruction is not eligible for a publication-quality lossy
   comparison; MANS/lsCOMP limitations are documented separately.
3. **OPEN — uniform peak-memory method.** Decide NVML-poll vs `nsys` as the default;
   NVML-poll is lighter and per-process-attributable, `nsys` is more precise but heavier. Default
   proposal: NVML-poll, nullable, method recorded. The existing GPU sampler records
   clocks, temperature, and throttle state, not process peak allocation, so this is not
   closed by H0-H6.
4. **~~Clock policy.~~ RESOLVED (D15).** Clock locking is optional and recorded. Where
   permissions do not allow it, Benchkit records observed clocks/throttle reasons and
   uses repetition variance as the primary `timing_reliable` gate.
5. **~~Dim-order convention.~~ RESOLVED.** `FieldSpec.dims` and dataset manifests use
   fast-to-slow order. Adapters explicitly translate when a native CLI expects another
   order, resolved inventories retain `dim_order`, and H4 verifies the dataset join.
6. **~~Stage-level result invalidation ("results as a database").~~ RESOLVED
   (2026-07-29).** FZGM reports compress/decompress stages and transitive source
   fingerprints; Benchkit records `stage_versions`, supports `stale --stage` and
   `stale --against-build`, re-runs with `--only-stale`, and append-preserving merge
   supersession. Rows predating fingerprints are explicitly excluded from automatic
   drift claims. See [`docs/stage-level-invalidation.md`](stage-level-invalidation.md).
7. **OPEN — error-bound violations under NOA / `rel_range` (not an infrastructure
   blocker).** The counts below are the original 2026-07-29 full-corpus snapshot; the
   newer tight-bound disposition is maintained in `docs/RUN_LEDGER.md`. That sweep
   found 678 of 9,416 ok cells missing their bound, *all* in
   `rel_range` (FZGM `NOA`), none in `abs`. Two distinct populations, both understood
   well enough to defer but not to dismiss:
   - **Marginal (388 cells, ≤1.0093x).** 374 sit at the tightest bound (eb=1e-4) and it
     hits every compressor including native cuSZ, so it reads as f32 round-off in the
     prediction/reconstruction chain near f32 resolution. Retained by the validity gate
     (D30) and reported as a caveat.
   - **Severe (99 cells, >10x).** PSNR stays nominal and tracks eb at 20 dB/decade while
     max error blows up 16-1025x — the bound holds in RMS and fails at isolated points,
     i.e. missed outlier capture. `cuszp3_outlier` alone is 52 of 99. A separate,
     legitimate sub-case is S3D: `N2` spans 0.736889-0.7369, a relative dynamic range of
     1.5e-5, so NOA yields an absolute bound of 1.1e-9 that no f64 codec can hold — there
     the error *mode* is wrong for the field, not the codec.
   Not purely a native defect: on the 4,518 paired cells native violates alone 270x and
   **FZGM violates alone 146x**. The FZGM-only violations concentrate in `pfpl` (34) and
   `cuszp2` (21 plain + 21 outlier). **Leading hypothesis for the PFPL half: native PFPL
   implements an error-correction/residual pass that the FZGM port does not** — native
   `pfpl` violates the bound in 0 of 558 cells while `fzgm:pfpl` violates in 34, which is
   exactly the asymmetry that hypothesis predicts. Confirming it and porting the pass is
   future work.
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

**Resume.** New rows carry both identities defined in §6.1. On start the runner scans
all run files in the session and skips only `status: ok` rows whose `execution_id`
matches the currently resolved execution. A changed dataset, pipeline source, graph
request, binary/tool declaration, harness/config, timing policy, or machine/software
state reruns rather than silently inheriting a stale result. Legacy `cell_key` rows are
readable and mergeable but cannot suppress a new execution.

**Canonical reads after merge.** Before merge, loading a session directory reads all
`runs.shard-*.jsonl` files. After `merge` creates the deduplicated `runs.jsonl`, reports
read that canonical file only; the retained shards are raw history and must not be read
alongside it or every merged row would be counted twice. Resume scanning remains
separate and intentionally examines every run file.

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
