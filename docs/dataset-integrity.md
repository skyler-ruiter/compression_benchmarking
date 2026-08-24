# Dataset integrity and registration

Benchkit separates the readable dataset description (`configs/datasets.yaml`) from the
machine-generated checksum lock (`configs/datasets.checksums.yaml`). A locked digest is
equivalent to an inline field-level `sha256`. If both exist, loading fails unless they
agree.

The digest covers exactly `product(dims) * dtype_width` bytes from the start of the
file—the same prefix consumed by adapters and identified by `execution_id`. This avoids
accidentally treating unrelated trailing bytes as scientific input. It also means that
changing dtype or dimensions requires revalidation even when the file is unchanged.

The current lock establishes exact identities for 238 locally available fields across
31 groups. Most are SDRBench extractions; it also includes the explicitly generated
ADM field, QMCPACK orbital splits, and quant-code derivatives. These hashes identify the
local bytes but are not cryptographic signatures from SDRBench. The obsolete
`CESM-CLDHGH-local` alias and seven optional `SCALING-SYNTH` fields were unavailable and
remain unlocked.

## Verify an installation

```bash
python -m benchkit dataset-checksums \
  --data-root /path/to/sdrbench_data \
  --dataset CESM-2D
```

Exit 0 means every available selected field matches its declaration. A mismatch exits
2 and is never rewritten implicitly. Checking the entire manifest also reports optional
or site-local files that are unavailable on that machine.

## Register a new dataset safely

1. Record the upstream name, stable URL, archive filename, license/redistribution terms,
   dtype, endianness, dimensions, and extraction or conversion command. Prefer adding it
   to both downloader implementations when it is an upstream SDRBench family.
2. Verify the downloaded archive before extraction when upstream publishes a checksum.
   Otherwise retain the source URL/archive name and validate archive structure and
   extraction success; a locally generated hash is identity evidence, not publisher
   authentication.
3. Add the dataset and fields to `configs/datasets.yaml`. Resolve every field and confirm
   its declared byte count against the file size.
4. Add only that dataset's missing hashes:

   ```bash
   python -m benchkit dataset-checksums --data-root /path/to/data \
     --dataset NEW-DATASET --write
   ```

5. Review the checksum-lock diff and rerun the command without `--write`. Ideally verify
   a separately downloaded extraction on a second machine before publication.
6. For derived fields, record the generator command/version and source-field hashes in
   the dataset comments or its generation script. Regeneration should reproduce the
   locked bytes.

Never use `--accept-changed` merely to make verification pass. First establish why the
bytes changed (upstream revision, corrected extraction, conversion change, or damage),
preserve the old identity in existing sessions, and document the reason. Only then run:

```bash
python -m benchkit dataset-checksums --data-root /path/to/data \
  --dataset AFFECTED-DATASET --write --accept-changed
```

The lock is written atomically. Publication-capable benchmark runs still independently
rehash selected fields before invoking any compressor and archive both the manifest and
the checksum lock in the session.
