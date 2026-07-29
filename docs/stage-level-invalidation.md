# Scoping: results as a database, with stage-level invalidation

**Status:** proposal, nothing implemented. Written 2026-07-29 after the first
full-corpus sweep (9,816 cells, ~20 h on an H100).

## The problem

A full sweep costs ~20 h per machine. Most FZGM changes touch **one stage**. Today
the only options are "re-run everything" or "re-run nothing and hope", because
nothing records which cells actually depended on the code that changed.

What we want: change `AdaptiveBitpack` → the harness names the affected cells →
re-run only those. A stage used by 3 of 9 pipelines turns a 20 h re-sweep into
roughly 30–90 minutes.

## What already exists (more than expected)

Every result row already carries the makings of this:

| field | what it gives us |
|---|---|
| `cell_key` | content-keyed cell identity; already drives resume + `merge` dedupe |
| `pipeline_ref` | which preset TOML produced the cell |
| `pipeline_sha256` | content hash of the **rendered** TOML — catches config drift |
| `stages[]` | the stage names that actually executed, with per-stage device_ms |
| `provenance.json` | GPU, driver, toolkit, host |

`stages[]` is the important one, and it is real: 14 distinct stage names observed
across the 4,860 FZGM rows of the last sweep (`AdaptiveBitpack`, `BitplaneRZE`,
`Bitshuffle`, `Difference`, `GInterp`, `Huffman`, `Lorenzo`, `LorenzoQuant`,
`Merge`, `Quantizer`, `RRE`, `RZE`, `TiledLorenzo`, `Zigzag`).

Resume is already content-keyed, so "re-run this subset" needs no new execution
machinery — only a way to mark rows stale.

## The two gaps

### ~~Gap 1 — `stages[]` is decompress-only~~ FIXED 2026-07-29

Root cause was neither candidate originally suspected. `Pipeline::setMemoryStrategy()`
(`src/pipeline/compressor.cpp`) **replaces `dag_` with a fresh `CompressionDAG`**,
which dropped the profiling flag `enableProfiling()` had set on the previous one.
`Pipeline::profiling_enabled_` stayed true so `compress()` still asked for timings,
but `CompressionDAG::collectTimings()` returned `{}` because the *new* DAG had
profiling off and none of its nodes ever got a `start_event`.

It fired on every config-driven run: the CLI calls `enableProfiling()` **before**
`loadConfig()`, and every TOML preset carries a `memory_strategy` key, so
`loadConfig()` → `setMemoryStrategy()` reset it every time. Decompress was
unaffected because its inverse DAG is built later, inside `decompress()` — which is
exactly why the loss was one-sided and went unnoticed for so long. Absent data
draws no attention.

Fix: `setMemoryStrategy()` re-applies the flag to the replacement DAG. Regression
test `Profiling.SurvivesMemoryStrategyChange`, verified to fail without it; ctest
46/46. Both phases now appear:

```
$ fzgmod-cli -b ... --report-json b.json && jq -r '.stages[]|"\(.phase) \(.name)"' b.json
compress Quantizer / compress Lorenzo / compress AdaptiveBitpack
decompress AdaptiveBitpack / decompress Lorenzo / decompress Quantizer
```

**Existing baselines captured before this fix have decompress-only `stages[]`.**
They are still valid for invalidation on the decompress side; compress-side
attribution requires a re-run (or simply accepting that any cell whose pipeline
contains the stage is stale, which is the conservative rule anyway).

### Gap 2 — no per-stage version fingerprint

`pipeline_sha256` hashes the rendered TOML. Editing a CUDA kernel changes **no
TOML**, so the hash is identical and the stale cell looks current. This is exactly
the case the feature exists for.

## Proposal

### A. FZGM exposes its stage inventory (and versions it)

FZGM already maintains the master list — `kStageRegistry` in
`src/pipeline/config.cpp`, 25 entries, with an "add a new stage type" procedure
documented directly above it:

```c
static const StageEntry kStageRegistry[] = {
    { "Lorenzo",      StageType::LORENZO,       addLorenzoStage,      saveLorenzoStage      },
    { "LorenzoQuant", StageType::LORENZO_QUANT, addLorenzoQuantStage, saveLorenzoQuantStage },
    ...
```

So the answer to "should FZGM maintain a master list?" is: **it already does, and
this should read from it rather than start a second one.** Any hand-maintained
list in benchkit would drift the first time someone adds a stage — the registry
comment tells them to touch exactly one place, and that place should feed this.

Two additions:

1. ~~`fzgmod-cli --list-stages --json`~~ **DONE 2026-07-29.** `fz::registeredStageTypes()`
   (`include/pipeline/config.h`) plus `fzgmod-cli --list-stages[=json]`, both derived
   from `kStageRegistry`. Reports all 25 stage types. Worth noting the drift this
   was meant to prevent had *already happened*: the prose stage list in `config.h`'s
   file comment was missing GInterp, AdaptiveBitpack, TiledLorenzo, GPULZ, ANS,
   Huffman and ADM. It now points at the accessor instead of restating the set.
2. A per-stage **version fingerprint**, the load-bearing piece. Options, cheapest
   first:
   - **Build-time source hash.** CMake hashes each stage's `.cu`/`.cuh`/`.inl`
     into a generated header. Exact, zero discipline required, but rebuild-coupled
     and noisy under formatting-only edits.
   - **Manual `stage_version` int in the registry.** Trivial to implement, but
     relies on remembering to bump it — the same discipline that makes
     hand-maintained lists rot.
   - **Hybrid:** manual semantic version for reporting, source hash for
     invalidation. Recommended.

   Emit as `stage_versions: {name: fingerprint}` in the report JSON, and record it
   per row.

### B. benchkit gains an index + invalidation query

No new storage engine. `runs.jsonl` stays the record; build the index on read.

```bash
python -m benchkit index   <session>/              # stage -> cells, pipeline -> cells
python -m benchkit stale   <session>/ --stage AdaptiveBitpack
python -m benchkit stale   <session>/ --against-build ~/FZGPUModules/build
python -m benchkit run <exp> --only-stale <session>/
```

- `stale --stage X` — every cell whose `stages[]` contains X.
- `stale --against-build` — compare recorded `stage_versions` to the current
  build's; report drift without being told what changed. This is the real target.
- `run --only-stale` — filter the matrix to those `cell_key`s. Resume already
  handles the rest.

A stale cell should be **superseded, not overwritten**: append the new row and let
`merge` prefer the newest per `cell_key`. That keeps "this stage got 1.8x faster"
answerable from the file, which is most of the value of treating results as a
database.

### C. Cross-machine

Baselines are already partitioned by provenance and never pooled (D15/D24).
Invalidation must respect that: a stage change invalidates cells **per machine**,
and a re-run on the H100 says nothing about the A100's rows. The index key is
therefore `(baseline_id, cell_key)`.

## Order of work

1. ~~**Fix Gap 1**~~ **DONE 2026-07-29** — see above.
2. ~~**`--list-stages`**~~ **DONE 2026-07-29** — see above.
3. ~~**`benchkit index` / `stale --stage`**~~ **DONE 2026-07-29.** Shipped as
   `benchkit stale` (`benchkit/invalidate.py`) — see below.
4. ~~**Stage fingerprints**~~ **DONE 2026-07-29.** See below.
5. ~~**`run --only-stale`** + supersede-on-merge~~ **DONE 2026-07-29.** See below.

All five steps landed on 2026-07-29. The workflow is complete.

## The whole loop (step 5)

```bash
# 1. change a stage, rebuild FZGM
# 2. re-measure only what that invalidated — no need to name the stage
python -m benchkit run <exp> --session-id <S> --only-stale --against-build $FZGMOD_CLI
# 3. collapse to one row per cell, newest wins
python -m benchkit merge $BENCHKIT_RESULTS_ROOT/<S>/
```

`--only-stale` also accepts `--stage NAME` (repeatable) when you would rather name
the stage than diff a build — necessary for pre-2026-07-29 rows, which carry no
fingerprints.

Verified end to end: seeded a 2-cell session, edited `adaptive_bitpack_kernels.cu`,
rebuilt, and the re-run selected **1 of 2** cells, appended the new measurement, and
`merge` reported `1 superseded by a newer re-run` while keeping the new fingerprint.
Running it again with nothing changed exits without touching the session.

### Append, never overwrite

A re-measured cell is **appended**; the superseded row stays in the raw file. That
is deliberate — it keeps "this stage got 1.8x faster" answerable from the data
instead of destroying the evidence, which is most of the point of treating results
as a database. `merge` is what collapses to one row per cell, under two precedence
rules that must not be conflated:

- **Within a file, the last row wins** — that is the newest measurement. Ordering is
  positional because rows carry no timestamp.
- **Across files, shard files beat `runs.jsonl`** — `runs.jsonl` is the *output* of a
  previous merge, so on a re-merge it holds older data than the shards. This was the
  pre-existing rule and is preserved.

### Sharding still applies

`--only-stale` composes with `--shard k/N`: the stale set is filtered, then the
shard filter applies, so a large re-measurement can still be split across an array
job. If a shard ends up with no stale cells it says so and exits cleanly.

## Stage fingerprints (step 4)

```bash
benchkit stale <session>/ --against-build ~/FZGPUModules/build/bin/fzgmod-cli
```

reports which stages changed since those rows were recorded, then lists the cells
they invalidate — no need to know or name what you edited.

**How the fingerprint is built.** `scripts/gen_stage_fingerprints.py` (in the FZGM
repo, run at build time) hashes each stage's own sources **plus the transitive
closure of its repo-local `#include`s**, keyed off a `source_dir` field added to
`kStageRegistry`. The transitive part is the whole point: stages share
infrastructure and include each other, so a per-directory hash would miss a change
to the memory pool or to a transform a stage inlines. Verified behaviour:

| edit | stages whose fingerprint moves |
|---|---|
| `modules/coders/adaptive_bitpack/*.cu` | 1 (AdaptiveBitpack) |
| `modules/transforms/zigzag/zigzag.h` | 4 (Zigzag + LorenzoQuant, Quantizer, Difference) |
| `include/mem/mempool.h` | all 25 |

FZGM emits the fingerprints of the stages a run used as `stage_versions` in
`--report-json`; benchkit records that per row.

**Two deliberate choices.** It is *conservative* — a comment-only edit moves the
fingerprint, because proving an edit semantically inert is not something a hash can
do, and a needless re-run is much cheaper than a wrong cached number. And it is
*automatic* — no manual version integer, so there is nothing to forget to bump.

**Rows without fingerprints are skipped, not assumed clean.** Everything recorded
before 2026-07-29 has no `stage_versions`; `--against-build` says so explicitly and
excludes those rows rather than reporting them unchanged. Absent data is not
evidence of no change. For those baselines, invalidate by name with `--stage`.

## What shipped (steps 1–3)

```bash
benchkit stale <session>/                          # every stage -> how many cells use it
benchkit stale <session>/ --stage AdaptiveBitpack  # what a change to it invalidates
benchkit stale <session>/ --stage GInterp --stage Merge   # repeatable
```

On the 9,816-cell H100 sweep:

| changed stage | stale cells | share of sweep |
|---|---|---|
| `Quantizer` | 2,790 | 28.4% |
| `AdaptiveBitpack` | 2,232 | 22.7% |
| `GInterp` + `Merge` | 1,020 | 10.4% |
| `BitplaneRZE` | 492 | 5.0% |

So an `AdaptiveBitpack` change costs ~4.5 h of re-running instead of ~20 h, and a
`BitplaneRZE` change ~1 h.

### Where the stage lists come from, and why no rows were rewritten

The pipeline→stages map lives in `configs/pipeline_stages.json`, generated by
`scripts/probe_pipeline_stages.py`, which renders each preset and runs it once on a
few KB of synthetic data. Existing result rows are **not** modified: invalidation
only needs *which stages a pipeline contains*, and every row already carries
`pipeline_ref`. Nothing has to be backfilled, and no timing value is invented for a
measurement that was never taken.

The substitution is validated, not assumed. Across the 4,860 FZGM rows of the
2026-07-29 sweep every preset maps to exactly one stage set, and the probed index
reproduces **all 13 of them exactly**. Regenerate the index whenever a preset
changes; `--check` fails on drift.

### Compress vs decompress

The probe confirmed the two phases run the identical stage **set** for all 20
presets, so the phase gap in old baselines does not affect invalidation. Ordering
is *usually* just the reverse — but not always: `cusz_hi_tp` compresses
`GInterp→Zigzag→Bitshuffle→RRE→Merge→Bitshuffle→RRE→RZE` and its decompress order
is not that list reversed, because it is a branching DAG with repeated stages
(8 stage instances, 6 distinct names). Reversal is a safe way to recover the *set*;
it is not a safe way to attribute *per-stage compress time* on a branching DAG.
For that, use a build dated 2026-07-29 or later.

### Two broken presets the probe found

Running every preset for the first time surfaced two that could never have worked:

- `fzgpu_f64.toml` declared `type = "BitplaneRLE"`, which is not a registered stage
  type — so it raised `loadConfig: unknown stage type` on every use, and
  `configs/experiments/sdrbench-miranda.yaml`, which references it, could never
  have run. The stage is `BitplaneRZE`; the rename missed this copy. **Fixed.**
- `quantizer_lorenzo_bitpack.toml` sets `nbits = 12`, but `BitpackStage::setNBits`
  requires a power of two. **Not fixed** — 8 and 16 are both plausible intents and
  the choice changes what the preset means. No experiment references it.

## Limits worth stating up front

- **Native compressors have no `stages[]`** and never will — they are opaque
  third-party binaries. Invalidation is FZGM-only; native rows are invalidated by
  rebuilding a reference, which is coarse and rare.
- **A stage change can alter results for pipelines that do not contain it**, via
  shared buffers or memory-pool sizing. Rare, but it means the index answers "what
  probably changed", not "what provably changed". Periodic full sweeps stay
  necessary; this reduces their frequency, it does not replace them.
- **Timing comparability.** Re-running a subset mixes measurements taken at
  different times on the same GPU. Fine under locked clocks on a single-tenant
  machine (JetStream2); on the clusters, where clock control is admin-only (D15),
  a partially re-run baseline needs a note in its `metadata.yaml`.
