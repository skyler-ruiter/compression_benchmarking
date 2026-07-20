# Baselines

Curated, cross-machine-portable benchmark snapshots — tracked in git, unlike
the rest of `results/` (harness scratch output, gitignored, one dir per ad
hoc local run). Put a result set here when it's worth comparing against from
a *different* machine later, without needing to physically copy files around
first (that's what prompted this directory: comparing an H100 run on
JetStream2 against an A100 run on BigRed200 required manually locating and
parsing a stray SLURM log — see `docs/DESIGN.md` D24).

## Layout

```
results/baselines/<baseline_id>/
  runs.jsonl        # one JSON object per benchmark cell (harness schema)
  provenance.json   # GPU/driver/toolkit/host info (harness schema)
  metadata.yaml      # human-readable summary — see below
  raw_slurm_stdout.log  # optional: only present for reconstructed baselines
```

`<baseline_id>` convention: `<gpu>-<site>-<unique-identifier>`, e.g.
`h100-jetstream2-20260719` or `a100-bigred200-slurm7562329`. Use a date if
you have one you trust; otherwise a SLURM job ID or benchkit session ID is
fine — it just needs to be unique and traceable back to its source.

## Getting a baseline here

**From this machine, right after a real run** (preferred — full fidelity):
```bash
mkdir -p results/baselines/<baseline_id>
cp results/<session_id>/runs.jsonl results/baselines/<baseline_id>/
cp results/<session_id>/provenance.json results/baselines/<baseline_id>/
# write metadata.yaml by hand (see schema below)
```

**From a stdout log only** (e.g. a SLURM job whose own `results/` output
never got copied off the cluster) — lossy, use only when the real
`runs.jsonl` truly isn't recoverable:
```bash
python scripts/reconstruct_runs_from_stdout.py <log> results/baselines/<baseline_id>/
# fill in the provenance.json blanks it leaves, then write metadata.yaml
```
A reconstructed `runs.jsonl` is missing per-rep timing arrays, `stages[]`,
`gpu_sampling`, and a few other fields a native run captures — CR/PSNR/
throughput survive, which is enough for a comparison artifact, not enough
for deep timing-reliability analysis. Say so in `metadata.yaml`
(`source.fidelity`) so nobody mistakes it for the real thing later.

## `metadata.yaml` schema

Not enforced by code — a convention, read by humans and by
`scripts/build_comparison_artifact.py`'s tile/callout text. Fields:

| Field | Meaning |
|---|---|
| `baseline_id` | Matches the directory name. |
| `gpu`, `site`, `driver`, `cuda_toolkit`, `date` | What ran this and when. |
| `experiment_config` | Which `configs/experiments/*.yaml` produced it. |
| `clocks_locked` | Whether GPU clocks were pinned (affects timing variance — see D15). |
| `source.kind` | `benchkit_native_output` or `reconstructed_from_stdout_log`. |
| `source.fidelity` | Free text: what's missing, if anything. |
| `cells` | `total`/`ok`/`failed` counts, plus a note on *why* anything failed. |
| `known_issues` | Anything that would mislead someone comparing this baseline against another — the reason this section exists at all is the cuSZp2 `excl_sum` bug (D23), found by noticing two baselines *disagreed* on cells that should be bit-for-bit deterministic. |
| `related_decisions` | `docs/DESIGN.md` decision-log IDs relevant to this baseline's correctness/methodology. |

## Generating a comparison artifact from two baselines

```bash
python scripts/build_comparison_artifact.py \
  results/baselines/a100-bigred200-slurm7562329 \
  results/baselines/h100-jetstream2-20260719 \
  -o /tmp/compare.html
```

See `scripts/build_comparison_artifact.py --help` for options (metric
labels, output path). The script diffs CR/PSNR between the two baselines as
a correctness sanity check before charting throughput — surfacing exactly
this kind of disagreement is the point, not just a formality.

## When there are many of these

Compress old ones (`gzip runs.jsonl`, or tar the whole baseline dir) once
this directory gets large — `metadata.yaml` and `provenance.json` should
stay uncompressed/readable since they're small and meant to be skimmed.
