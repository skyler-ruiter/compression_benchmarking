#!/usr/bin/env python3
"""Split an SDRBench S3D timestep file into its 11 individual 3-D variables.

Why this exists: S3D does not ship as 11 separate field files, despite
sdrbench.github.io listing it as "11 fields". Each `stat_planar.*.field.d64` is a
single 11,000,000,000-byte blob holding 11 variables of 500x500x500 f64 stacked
back to back (11 * 500^3 * 8 = 11e9 exactly). The tarball's own template.txt states
the layout: "ARRAY DIMENSION: 4D (Spatially 3D, but 11 variables at each spatial
grid point) / DATAPOINT NUMBER: 11x500x500x500", little-endian, double precision.

Benchkit's FieldSpec is path-based with no byte-offset support, so the variables
have to become real files before they can be registered.

Only ONE timestep is split by default. The tarball ships 5 timesteps (and flist.txt
names 400 in the full dataset); they are the same 11 variables at different
simulation times, so splitting all of them would multiply disk by 5 for very little
extra variety — the fields within one timestep already differ far more from each
other than a variable does from itself one timestep later.

Layout verification: variable-major ordering is asserted, not assumed. Variables
0-5 are mass fractions, which must lie in [0, 1]; variable 6 is normalized
temperature. If the file were interleaved instead, block 0 would contain a mix of
mass fractions, pressures and velocities and the range check would fail.

Usage:
    python scripts/extract_s3d_variables.py [--timestep 1.1000E-03] [--data-root DIR]
    python scripts/extract_s3d_variables.py --emit-yaml
"""
import argparse
import os
import struct
import sys
from pathlib import Path

NX = NY = NZ = 500
ELEM = 8                                   # f64
VAR_ELEMS = NX * NY * NZ
VAR_BYTES = VAR_ELEMS * ELEM               # 1_000_000_000
N_VARS = 11
TOTAL_BYTES = N_VARS * VAR_BYTES           # 11_000_000_000

# Order per the tarball's template.txt.
VARS = [
    ("CH4",  "Mass Fraction of CH4"),
    ("O2",   "Mass Fraction of O2"),
    ("CO",   "Mass Fraction of CO"),
    ("CO2",  "Mass Fraction of CO2"),
    ("H2O",  "Mass Fraction of H2O"),
    ("N2",   "Mass Fraction of N2"),
    ("TEMP", "Normalized Temperature (x120 K)"),
    ("PRES", "Normalized Pressure (x1.41837E+05 Pa)"),
    ("U",    "Normalized U velocity (x347.2 m/s)"),
    ("V",    "Normalized V velocity (x347.2 m/s)"),
    ("W",    "Normalized W velocity (x347.2 m/s)"),
]
MASS_FRACTION_VARS = 6                     # first six must be in [0, 1]

SRC_DIR = Path("S3D_500x500x500")
OUT_DIR = SRC_DIR / "vars_500x500x500"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--timestep", default="1.1000E-03",
                    help="which stat_planar.<TS>.field.d64 to split (default 1.1000E-03)")
    ap.add_argument("--data-root", default=os.environ.get("BENCHKIT_DATA_ROOT"))
    ap.add_argument("--emit-yaml", action="store_true")
    ap.add_argument("--chunk-mb", type=int, default=256)
    args = ap.parse_args()

    if args.emit_yaml:
        print("S3D:")
        print("  dtype: f64")
        print("  dim_order: fast-to-slow")
        print(f"  root: ${{BENCHKIT_DATA_ROOT}}/{OUT_DIR.as_posix()}")
        print("  fields:")
        for short, desc in VARS:
            print(f"    {short + ':':6s} {{dims: [{NX}, {NY}, {NZ}], "
                  f"path: {short}.d64}}   # {desc}")
        return 0

    if not args.data_root:
        print("error: set BENCHKIT_DATA_ROOT or pass --data-root", file=sys.stderr)
        return 2
    root = Path(args.data_root)
    src = root / SRC_DIR / f"stat_planar.{args.timestep}.field.d64"
    out = root / OUT_DIR

    if not src.exists():
        print(f"error: {src} not found. Available:", file=sys.stderr)
        for p in sorted((root / SRC_DIR).glob("stat_planar.*.field.d64")):
            print(f"  {p.name}", file=sys.stderr)
        return 1
    size = src.stat().st_size
    if size != TOTAL_BYTES:
        print(f"error: {src} is {size} bytes, expected {TOTAL_BYTES} "
              f"({N_VARS} x {VAR_BYTES}). Wrong file or truncated download.",
              file=sys.stderr)
        return 1

    free = os.statvfs(out.parent if out.parent.exists() else root)
    free_bytes = free.f_bavail * free.f_frsize
    if free_bytes < TOTAL_BYTES:
        print(f"error: need {TOTAL_BYTES/1e9:.1f} GB free, have {free_bytes/1e9:.1f} GB",
              file=sys.stderr)
        return 1

    out.mkdir(parents=True, exist_ok=True)
    print(f"source: {src}")
    print(f"output: {out}\n")

    chunk = args.chunk_mb * 1024 * 1024
    with open(src, "rb") as fh:
        for i, (short, desc) in enumerate(VARS):
            dest = out / f"{short}.d64"
            fh.seek(i * VAR_BYTES)
            written = 0
            lo, hi = float("inf"), float("-inf")
            with open(dest, "wb") as of:
                while written < VAR_BYTES:
                    buf = fh.read(min(chunk, VAR_BYTES - written))
                    if not buf:
                        print(f"error: short read on {short}", file=sys.stderr)
                        return 1
                    of.write(buf)
                    # Track range from the first chunk only — enough to validate the
                    # layout without a second full pass over 11 GB.
                    if written == 0:
                        vals = struct.unpack(f"<{len(buf)//ELEM}d", buf)
                        lo, hi = min(vals), max(vals)
                    written += len(buf)
            flag = ""
            if i < MASS_FRACTION_VARS:
                ok = -1e-6 <= lo and hi <= 1.0 + 1e-6
                flag = "  [mass fraction in [0,1]: OK]" if ok else \
                       "  [*** NOT in [0,1] — layout may not be variable-major ***]"
                if not ok:
                    print(f"  {short:5s} range [{lo:.6g}, {hi:.6g}]{flag}", file=sys.stderr)
                    print("aborting: refusing to emit fields from an unverified layout",
                          file=sys.stderr)
                    return 1
            print(f"  {short:5s} -> {dest.name:9s} range [{lo:11.6g}, {hi:11.6g}]{flag}")

    print(f"\nwrote {N_VARS} fields, {N_VARS * VAR_BYTES / 1e9:.1f} GB total")
    print("Add the block from `--emit-yaml` to configs/datasets.yaml.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
