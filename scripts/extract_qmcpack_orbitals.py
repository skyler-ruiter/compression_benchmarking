#!/usr/bin/env python3
"""Split SDRBench QMCPACK's 4-D einspline array into individual 3-D orbital fields.

Why this exists: QMCPACK ships as a single 4-D array (288 orbitals x 115 x 69 x 69),
and no adapter in this repo handles 4-D input -- SPERR and zfp reject it explicitly,
and cuSZ/cuSZ-Hi/cuSZp take at most 3-D. Rather than flatten the orbital axis into a
neighbouring dimension (which would fabricate spatial locality across unrelated
orbitals and quietly distort every predictor's error), this splits the array along
its slowest axis into genuinely independent 3-D fields, which is what an orbital is.

Source layout: results/baselines-adjacent `$BENCHKIT_DATA_ROOT/QMCPACK/288x115x69x69/
einspline_288_115_69_69.pre.f32`. That file is ORBITAL-MAJOR in C order --
[288][115][69][69] with the orbital index slowest -- so orbital i is the contiguous
byte range [i * 115*69*69*4, (i+1) * 115*69*69*4). The sibling 115x69x69x288 file is
the same data with the orbital axis fastest (interleaved); it is NOT usable here,
because no orbital is contiguous in it.

Emitted dims are fast-to-slow [69, 69, 115] to match this repo's dim_order convention
(C-order [115][69][69] means the last axis varies fastest).

Sampling: taking all 288 orbitals would add ~15.5k cells (~17 h) to a full sweep for
one dataset of 0.6 GB, which is out of proportion to every other dataset. The default
takes N orbitals evenly spaced across the full index range rather than the first N,
because orbital character varies systematically with index (roughly, energy level) --
the first N would sample one end of that range and misrepresent the dataset.

Usage:
    python scripts/extract_qmcpack_orbitals.py [--n 8] [--data-root $BENCHKIT_DATA_ROOT]
    python scripts/extract_qmcpack_orbitals.py --emit-yaml    # print the datasets.yaml block
"""
import argparse
import os
import sys
from pathlib import Path

N_ORBITALS = 288
ORB_DIMS_C = (115, 69, 69)          # C order, slow-to-fast, within one orbital
ORB_DIMS_FAST_TO_SLOW = [69, 69, 115]
ELEM = 4                             # f32
ORB_BYTES = ORB_DIMS_C[0] * ORB_DIMS_C[1] * ORB_DIMS_C[2] * ELEM   # 2_190_060
SRC_REL = Path("QMCPACK") / "288x115x69x69" / "einspline_288_115_69_69.pre.f32"
OUT_REL = Path("QMCPACK") / "orbitals_69x69x115"


def chosen_indices(n: int) -> list[int]:
    """n indices evenly spaced across [0, 287], inclusive of both ends."""
    if n <= 1:
        return [0]
    if n >= N_ORBITALS:
        return list(range(N_ORBITALS))
    step = (N_ORBITALS - 1) / (n - 1)
    return sorted({int(round(i * step)) for i in range(n)})


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=8, help="how many orbitals to extract (default 8)")
    ap.add_argument("--data-root", default=os.environ.get("BENCHKIT_DATA_ROOT"),
                    help="defaults to $BENCHKIT_DATA_ROOT")
    ap.add_argument("--emit-yaml", action="store_true",
                    help="print the configs/datasets.yaml block for the extracted files")
    args = ap.parse_args()

    if not args.data_root:
        print("error: set BENCHKIT_DATA_ROOT or pass --data-root", file=sys.stderr)
        return 2
    root = Path(args.data_root)
    src = root / SRC_REL
    out_dir = root / OUT_REL
    idx = chosen_indices(args.n)

    if args.emit_yaml:
        print("QMCPACK-3D:")
        print("  dtype: f32")
        print("  dim_order: fast-to-slow")
        print(f"  root: ${{BENCHKIT_DATA_ROOT}}/{OUT_REL.as_posix()}")
        print("  fields:")
        for i in idx:
            print(f"    orbital_{i:03d}: {{dims: {ORB_DIMS_FAST_TO_SLOW}, "
                  f"path: orbital_{i:03d}.f32}}")
        return 0

    if not src.exists():
        print(f"error: source not found: {src}", file=sys.stderr)
        return 1
    expected = N_ORBITALS * ORB_BYTES
    actual = src.stat().st_size
    if actual != expected:
        print(f"error: {src} is {actual} bytes, expected {expected} "
              f"({N_ORBITALS} x {ORB_BYTES}). Wrong file or truncated download.",
              file=sys.stderr)
        return 1

    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"source : {src}")
    print(f"output : {out_dir}")
    print(f"orbitals: {len(idx)} of {N_ORBITALS} -> {idx}")

    import struct
    written = 0
    with open(src, "rb") as fh:
        for i in idx:
            dest = out_dir / f"orbital_{i:03d}.f32"
            fh.seek(i * ORB_BYTES)
            buf = fh.read(ORB_BYTES)
            if len(buf) != ORB_BYTES:
                print(f"error: short read for orbital {i}", file=sys.stderr)
                return 1
            # Guard against silently emitting a degenerate field: an all-zero or
            # constant orbital has undefined value-range, which makes rel_range error
            # bounds meaningless and would poison the dataset aggregate.
            sample = struct.unpack(f"<{ORB_BYTES // ELEM}f", buf)
            lo, hi = min(sample), max(sample)
            if not (hi > lo):
                print(f"  skip orbital {i}: constant field (value {lo}) — "
                      f"no value range, rel_range eb undefined")
                continue
            dest.write_bytes(buf)
            written += 1
            print(f"  orbital {i:3d} -> {dest.name}  range [{lo:.6g}, {hi:.6g}]")

    print(f"\nwrote {written} fields, {written * ORB_BYTES / 1e6:.1f} MB total")
    print("Add the block from `--emit-yaml` to configs/datasets.yaml.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
