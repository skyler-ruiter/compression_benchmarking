#!/usr/bin/env python3
"""Generate synthetic f32 scaling-test datasets for benchmarking throughput vs. data size.

Two modes:
  --source <file.f32>   Tile an existing f32 binary (e.g. NYX baryon_density.f32)
                        so statistics stay realistic. np.resize cycles the source
                        array, meaning tile boundaries repeat at source-size intervals.
  (no --source)         Generate a smooth Brownian-motion field (cumulative sum of
                        Gaussian noise, seed=42). Compresses similarly to scientific
                        data under a Lorenzo predictor.

Target sizes: 0.5, 1, 2, 4, 8, 12, 16 GB (decimal). Override with --sizes-gb.

Usage:
    python scripts/make_scaling_data.py \\
        --source $BENCHKIT_DATA_ROOT/NYX_512x512x512/baryon_density.f32 \\
        --out-dir $BENCHKIT_DATA_ROOT/SCALING-SYNTH

    python scripts/make_scaling_data.py \\
        --out-dir $BENCHKIT_DATA_ROOT/SCALING-SYNTH   # synthetic fallback

After writing, the script prints the datasets.yaml entries for the generated files.
These are already added to configs/datasets.yaml under SCALING-SYNTH; re-running the
script is idempotent (existing same-size files are skipped).
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np

DEFAULT_SIZES_GB = [0.5, 1.0, 2.0, 4.0, 8.0, 12.0, 16.0]
ELEM_BYTES = 4  # f32

SIZE_LABELS = {
    0.5: "synth_0p5GB",
    1.0: "synth_1GB",
    2.0: "synth_2GB",
    4.0: "synth_4GB",
    8.0: "synth_8GB",
    12.0: "synth_12GB",
    16.0: "synth_16GB",
}


def _sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _make_synthetic_seed(n: int = 1_000_000) -> np.ndarray:
    """Brownian-motion seed: cumulative sum of Gaussian noise, normalized to unit range.

    A cumulative sum is a good Lorenzo-compressor test signal — the inter-element
    differences are i.i.d. Gaussian, so the predictor residuals are well-behaved and
    the quantizer fills its range uniformly. Compresses significantly better than
    pure white noise (which is nearly incompressible).
    """
    rng = np.random.default_rng(42)
    walk = rng.standard_normal(n).cumsum().astype(np.float64)
    lo, hi = walk.min(), walk.max()
    normalized = ((walk - lo) / (hi - lo)).astype(np.float32)
    return normalized


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Generate synthetic f32 scaling-test datasets.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("--source",
                    help="Source f32 binary to tile. If omitted, a synthetic "
                         "Brownian-motion field is generated instead.")
    ap.add_argument("--out-dir", required=True,
                    help="Output directory (created if absent).")
    ap.add_argument("--sizes-gb", nargs="+", type=float, default=DEFAULT_SIZES_GB,
                    help="Target sizes in decimal GB. Default: 0.5 1 2 4 8 12 16")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.source:
        src = Path(args.source)
        if not src.exists():
            sys.exit(f"Error: --source file not found: {src}")
        print(f"Source: {src}  ({src.stat().st_size / 1e9:.3f} GB)", flush=True)
        source_data = np.fromfile(src, dtype=np.float32)
        print(f"  {source_data.size:,} elements  "
              f"min={source_data.min():.4g}  max={source_data.max():.4g}", flush=True)
    else:
        print("No --source provided; generating synthetic Brownian-motion seed.", flush=True)
        source_data = _make_synthetic_seed()
        print(f"  seed size: {source_data.size:,} elements  "
              f"min={source_data.min():.4g}  max={source_data.max():.4g}", flush=True)

    print(f"\nOutput directory: {out_dir}\n", flush=True)

    entries: list[tuple[str, str, int, int]] = []
    for size_gb in sorted(args.sizes_gb):
        n_elems = int(size_gb * 1e9 / ELEM_BYTES)
        actual_bytes = n_elems * ELEM_BYTES
        label = SIZE_LABELS.get(size_gb, f"synth_{size_gb:.3g}GB")
        fname = f"{label}.f32"
        out = out_dir / fname

        if out.exists() and out.stat().st_size == actual_bytes:
            print(f"  {fname:<24}  {actual_bytes/1e9:6.3f} GB  (exists, skip)", flush=True)
        else:
            tiled = np.resize(source_data, n_elems)
            tiled.tofile(out)
            print(f"  {fname:<24}  {actual_bytes/1e9:6.3f} GB  written", flush=True)

        entries.append((label, fname, n_elems, actual_bytes))

    print("\n# ----- datasets.yaml snippet (already in SCALING-SYNTH) ---------------")
    for label, fname, n_elems, actual_bytes in entries:
        print(f"    {label}:  {{dims: [{n_elems}], path: {fname}}}  "
              f"# {actual_bytes/1e9:.3f} GB")
    print("# -----------------------------------------------------------------------\n")
    print("Done. Run the experiment with:")
    print("  python -m benchkit run configs/experiments/fzgm_scaling.yaml")


if __name__ == "__main__":
    main()
