# Publication and AD/AE bundles

Build an artifact only after the source session passes H4:

```bash
python -m benchkit artifact build results/<session> artifacts/<bundle-name>
python -m benchkit artifact verify artifacts/<bundle-name>
```

`artifact build` refuses an existing output directory and deletes only the new partial
directory if construction fails. `artifact verify` requires no datasets, compressor,
GPU, network, or original session directory. It validates the content-addressed file
inventory and reruns H4 against the bundled session evidence without rewriting it.

The bundle contains:

- canonical and raw result rows, immutable provenance, H4 report, archived inputs,
  rendered pipelines, logs, and lightweight work diagnostics;
- `publication/rows.csv` plus metadata naming its source session, source-row checksum,
  generator contract, and harness commit;
- Benchkit's license, third-party/data notice, locked Python environment, build recipes,
  and site environment recipes;
- `reproduction/` with a generated one-cell experiment, checksum-locked dataset
  manifest, effective pipeline where applicable, and `run-smoke.sh`;
- `artifact.json`, whose `artifact_id` hashes the complete sorted file inventory.

Additional paper tables or figures can be included with repeated `--include PATH`.
They are copied under `publication/included/` and receive a metadata sidecar containing
their source session, generator contract, harness commit, and checksum:

```bash
python -m benchkit artifact build results/<session> artifacts/paper-ae \
  --include paper/tables/main.csv --include paper/figures/throughput.pdf
```

The reproduction script intentionally writes outside the immutable bundle:

```bash
export BENCHKIT_DATA_ROOT=/path/to/locked/sdrbench_data
export FZGMOD_CLI=/path/to/fzgmod-cli       # or the selected adapter's variable
./artifacts/paper-ae/reproduction/run-smoke.sh
```

Datasets and third-party binaries are not redistributed. Their exact expected bytes and
build identities are recorded, while acquisition and licenses remain governed by their
upstream projects. See `docs/reference-tools.md` and `docs/dataset-integrity.md`.
