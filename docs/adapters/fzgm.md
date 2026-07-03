# Adapter notes — FZGM (`fzgmod-cli`)

Integration quirks for the FZGPUModules adapter. Variant: `fzgm`. This is the one
compressor whose output we fully control — it emits structured JSON, so the adapter
parses a file instead of scraping stdout.

Authoritative schema: FZGM repo `memory/report_json_spec.md`; user docs `docs/cli.md`.
Schema version targeted: **1.0**.

## Invocation

Add `--report-json <path>` to any operation; it writes a standalone, pure-JSON file
(logs / warnings / `--profile` table stay on stdout/stderr and never contaminate it).
`--report-json` **auto-enables profiling**, so `device_ms` and `stages[]` are always
populated — do not also pass `--profile`. Works independently of `--report` (the
human-readable text), which the adapter ignores.

```bash
# Benchmark — the mode the adapter uses for throughput (N reps, full timing arrays)
fzgmod-cli -b -i data.f32 -l 3600x1800 -m rel -e 1e-3 --runs 10 --report-json out.json

# Single-shot compress / decompress also supported
fzgmod-cli -z -i data.f32 -l 3600x1800 -m rel -e 1e-3 -o c.fzm --report-json out.json
fzgmod-cli -x -i c.fzm -o d.f32 --compare data.f32 --report-json out.json
```

The pipeline is set via `--stages "s1->s2->..."` (or `-c preset.toml`); the resolved
chain comes back in `config.pipeline`, which the adapter records as the row's `pipeline`.

## Parsing rules (the ones that bite)

1. **Check `status == "ok"` first.** On failure the file is still valid JSON with
   `"status":"error"`, a non-null `error_message`, partial `config`, and a non-zero exit
   code. Record `status: "fail"` from this — never infer failure from a missing file.
2. **Timing source for throughput = `timing.<phase>.device_ms`, never `host_wall_ms`.**
   `device_ms` is true GPU wall time from CUDA events bracketing the DAG
   (`timing_method: "cuda_events_dag"`, excludes PCIe + host setup). `host_wall_ms`
   includes a fresh pipeline build + file I/O and can be 10–100× larger on single-shot
   decompress. The harness records `host_wall_ms` only as the end-to-end lower bound.
3. **Recompute, don't read derived fields.** `size.original_bytes`,
   `size.compressed_bytes`, `config.num_elements` are always present and authoritative.
   `size.ratio`, `size.bitrate_bits_per_elem`, and `throughput.*` are conveniences —
   the harness recomputes CR / bit-rate / throughput in its own unit convention
   (decimal GB/s). FZGM's native throughput unit is **decimal GB/s** (bytes / 1e9 / s);
   record that in the row so it's never confused with cuSZ's GiB/s.
4. **Compute spread from `timing.<phase>.<clock>.all`.** Per-rep arrays are provided so
   the harness derives median/IQR itself. Use FZGM `median` (or `min`) consistently with
   how other tools' single-run numbers are treated (a single warm run ≈ our `min`).
5. **Per-operation shape differs — check key presence, don't assume.**
   - `-z` → `timing.compress` only.
   - `-x` → `timing.decompress` only; single-rep (`n_runs:1`, one-element `all`); no
     `memory` block.
   - `-b` → both phases, `--runs` reps.
   - `quality` present only with `--compare` (or benchmark self-compare).
   - Optional blocks are **omitted, not null** — test for the key.

## `stages[]` — FZGM-only, the debugging payoff

Each entry is `{name, phase, device_ms}`. When an FZGM port is slower than the reference
it's modeled on, this localizes *which* stage regressed — reference compressors give us
nothing comparable. Worth persisting per-row (e.g. as a nested field or a sidecar) even
though it has no cross-tool counterpart.

Note: `sum(stages[].device_ms)` ≈ per-kernel total (≈ cuSZ's "(total)" model) and is
**less than** `timing.compress.device_ms`, because the bracketed DAG time also includes
inter-kernel launch gaps. That gap = launch/scheduling overhead, itself a useful signal.

## Three gotchas confirmed against the real binary

- **Chunk padding — truncate before computing quality.** Decompressed output is padded
  to chunk boundaries: a 25,920,000-byte original came back as 25,935,872 bytes from a
  bare `-x -o`. `metrics.py` must read only the first `num_elements` (from the dataset
  manifest) of *both* arrays. Passing `--compare <orig>` makes the CLI truncate the
  written file itself, but the harness truncates defensively regardless.
- **Timing must come from in-process `-b --runs N`, not runner-driven single-shot reps.**
  A single-shot `-z` reported 31.7 ms compress; the in-process benchmark `min` for the
  same data was 0.93 ms — a ~34× gap from cold GPU clocks + JIT in a fresh process. Each
  separate process starts cold, so looping single-shot calls measures startup, not the
  kernel. The adapter therefore exposes `benchmark()` backed by `-b --runs N` for all
  timing numbers; `compress()` / `decompress()` exist only to produce the
  artifacts (compressed blob, decompressed array) that harness-owned metrics need, and
  their `device_ms` is ignored. Drop the first `warmup_reps` entries of `timing.*.all`
  before taking median/min (FZGM does only one internal warmup).
- **Non-standard stage types must be registered in `createStage()`.** The `.fzm` header
  stores stage types as integer enums and reconstructs stages via `createStage()` in
  `stage_factory.h`. Stages not listed there (`Quantizer` = 14, `Difference` = 15, etc.)
  exit with `Unknown stage type: N` on decompress. Fix: add a `case StageType::QUANTIZER:`
  (and `DIFFERENCE` etc.) to `createStage()` in the FZGPUModules source and rebuild.
  Note: `-c pipeline.toml` is **silently ignored** by `run_decompress()` in `cli.cpp` —
  it calls `Pipeline::decompressFromFile()` unconditionally regardless of `config_path`.
  Passing `-c` during `-x` does nothing in the current binary. The adapter passes it
  anyway as a forward-compat hint in case `-x -c` is wired up in a future version.
  (Diagnosed 2026-07-02 against pfpl/pfpl_minimal pipelines; fixed by binary patch.)

## Pipelines: TOML-first

The adapter drives pipelines via TOML (`-c config.toml`), not the `--stages` text path.
TOML exposes the full DAG — branches, fused stages, per-stage params — and lets a
hand-tuned config be benchmarked as-is and shipped with its results. `configs/pipelines/`
holds repo-owned templates (`cusz`, `fzgpu`, `pfpl`, `quantizer_lorenzo_bitpack`) that
map 1:1 to the reference compressors. A `--stages` chain is still accepted (pipeline
string not ending in `.toml`) for quick linear tests, but cannot express branches and
hits the Huffman symbol-range wall (zigzag isn't exposed) — use the TOML.

`prepare()` resolves a cell: for a sweep it renders the bound into every lossy stage
(`error_bound` + `error_bound_mode`) via text substitution (comments preserved), writes
`work/<run_id>/pipeline.toml`, and records its sha256. For `from_toml` mode it ships the
template verbatim and reads the bound/mode the config already declares. Dims (`-l`) and
type (`-t`) still come from the CLI; the bound is in the TOML, so no `-m`/`-e`.

## Error-mode mapping

Canonical → FZGM native (TOML `error_bound_mode`) → eb basis:

| canonical | TOML mode | basis | note |
|---|---|---|---|
| `rel_range` | `NOA` | range | **the comparable**; exact (`err_over_bound`≈1.000003) |
| `rel_maxabs` | `REL` | maxabs | Lorenzo REL = `eb·max(\|data\|)`; approx per-elem (≈1.0002×) |
| `abs` | `ABS` | abs | |
| `from_toml` | (declared) | from mode | run the config's own bound; no sweep |

**FZGM `REL` ≠ cuSZ `REL`.** FZGM/PFPL use ABS/NOA/REL where `NOA` is the range-relative
bound that cuSZ et al. call `REL`; FZGM's `REL` scales by `max(|data|)` for the Lorenzo
predictor (`modules/predictors/predictor_utils.cuh`: "REL → max(|data|)"), approximate
per-element, and is a *different, exact* thing for QuantizerStage (revisit when
benchmarking quantizer pipelines). `eb_tol` defaults to 1e-3 to absorb the approximation.

> History: the harness first assumed FZGM `rel` was range-relative and flagged *every*
> row as bound-violating — the discrepancy is what led to checking the source. Exactly
> the error-mode-normalization hazard DESIGN.md §5.4 was written to catch.

## Dim order

FZGM takes `-l <fast>x<mid>x<slow>` (fast-to-slow). The dataset manifest must agree, and
the adapter asserts `config.dims` echoed in the JSON matches what was requested — a
silent transpose wrecks quality metrics (open question #5).
