#!/usr/bin/env python3
"""Head-to-head A/B table across variants within one session.

Why this exists
---------------
`benchkit report --aggregate` groups by (compressor, variant, pipeline,
error_bound) but reports **CR only** — there is no throughput aggregation and no
ratio anywhere in the harness. An ablation experiment (arm vs arm, same field,
same bound) is exactly the shape it cannot print.

`scripts/build_comparison_artifact.py` compares two *baseline directories* and
cannot be reused here: its `side` rule collapses every FZGM variant onto the
same side, so two FZGM arms in one session become indistinguishable.

What it does
------------
Pivots rows on `variant` within each (dataset, field, error_bound) cell and
divides each arm by `--baseline`, then reports the geometric mean of those
per-cell ratios. Geometric, not arithmetic: these are ratios, and a 2x speedup
and a 0.5x slowdown must cancel to 1.0 rather than average to 1.25.

Ratios are formed **per cell and then averaged**, never as a ratio of two
independently-computed means. The corpus spans 11 MB to 1.1 GB, so a mean of
throughputs is dominated by whichever fields happen to be in the subset; a mean
of per-cell ratios is not.

Gating
------
Rows go through `benchkit.validity` exactly as `report --aggregate` does, and
the exclusion audit prints alongside. Two deliberate departures:

- `--metric cr` gates rows (an `expansion` row's CR is meaningless).
- `--metric ctp/dtp` does NOT gate on `expansion`: a coder that expanded still
  has a valid *throughput* measurement, and for the chunked-RLE experiment those
  are the rows under test. `failed` rows are always dropped.

`timing_reliable is False` cells are counted and flagged (`!` suffix) rather
than dropped — a wide-variance cell is a caveat on that cell, not grounds to
delete a measurement.

Usage
-----
    python scripts/build_variant_ratio.py <session_dir|runs.jsonl> \
        --baseline fzgm-huf-perblock [--metric ctp|dtp|cr] [--stage Huffman] \
        [--by-field] [--phase compress|decompress]

`--stage NAME` switches the metric to that stage's per-stage `device_ms` from
FZGM's `--report-json` (lower is better, so the ratio is inverted to stay
"higher = better" like the throughput metrics). This is how you isolate the
entropy coder from the predictor in front of it — the end-to-end pipeline number
answers a different question.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from benchkit import validity  # noqa: E402


METRICS = {
    "ctp": ("compress_throughput_gbs", "compress GB/s"),
    "dtp": ("decompress_throughput_gbs", "decompress GB/s"),
    "cr":  ("cr", "compression ratio"),
}


def load_rows(target: str) -> list[dict]:
    p = Path(target)
    if p.is_dir():
        p = p / "runs.jsonl"
    if not p.exists():
        raise SystemExit(f"no runs.jsonl at {p}")
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


def stage_ms(row: dict, stage: str, phase: str) -> float | None:
    """Summed device_ms for `stage` in `phase`.

    Summed, not first-match: `stages[]` carries no instance disambiguation, so a
    pipeline with two stages of the same type emits two entries under one name.
    Summing is the only interpretation that is correct for both cases.
    """
    total, seen = 0.0, False
    for s in row.get("stages") or []:
        if s.get("name") == stage and s.get("phase") == phase:
            v = s.get("device_ms")
            if isinstance(v, (int, float)) and math.isfinite(v):
                total += float(v)
                seen = True
    return total if seen else None


def geomean(vals: list[float]) -> float | None:
    vals = [v for v in vals if v and v > 0 and math.isfinite(v)]
    if not vals:
        return None
    return math.exp(sum(math.log(v) for v in vals) / len(vals))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("target", help="session dir or runs.jsonl")
    ap.add_argument("--baseline", required=True, help="variant every arm is divided by")
    ap.add_argument("--metric", default="ctp", choices=sorted(METRICS))
    ap.add_argument("--stage", help="use this stage's device_ms instead of --metric")
    ap.add_argument("--phase", default="compress", choices=("compress", "decompress"))
    ap.add_argument("--by-field", action="store_true", help="also print every cell")
    ap.add_argument("--no-gate", action="store_true", help="skip the validity gate")
    args = ap.parse_args()

    rows = validity.annotate(load_rows(args.target))

    # Throughput survives an `expansion` row; CR does not. See module docstring.
    def usable(r: dict) -> bool:
        if r.get("status") != "ok":
            return False
        if args.no_gate:
            return True
        if args.stage or args.metric in ("ctp", "dtp"):
            reasons = set(r.get("_exclusions") or [])
            return not (reasons - {"expansion", "psnr_nonfinite"})
        return validity.is_valid(r)

    if args.stage:
        label = f"{args.stage} {args.phase} device_ms"
        better = "lower"
        getter = lambda r: stage_ms(r, args.stage, args.phase)  # noqa: E731
    else:
        key, label = METRICS[args.metric]
        better = "higher"
        getter = lambda r: r.get(key)  # noqa: E731

    # (dataset, field, eb) -> variant -> value
    cells: dict[tuple, dict[str, float]] = {}
    unreliable: set[tuple[tuple, str]] = set()
    for r in rows:
        if not usable(r):
            continue
        v = getter(r)
        if v is None or not isinstance(v, (int, float)) or not math.isfinite(v) or v <= 0:
            continue
        ck = (r.get("dataset"), r.get("field"), r.get("error_bound"))
        cells.setdefault(ck, {})[r.get("variant")] = float(v)
        if r.get("timing_reliable") is False:
            unreliable.add((ck, r.get("variant")))

    variants: list[str] = []
    for cv in cells.values():
        for name in cv:
            if name not in variants:
                variants.append(name)
    if args.baseline not in variants:
        raise SystemExit(f"baseline variant {args.baseline!r} not present. "
                         f"Found: {', '.join(sorted(variants))}")
    others = [v for v in variants if v != args.baseline]

    print(f"metric: {label}   ({better} is better)")
    print(f"baseline: {args.baseline}")
    print(f"ratios are per-cell, then geometric mean over cells "
          f"(n = cells where BOTH arms produced a usable row)\n")

    # Missing cells are informative here: PerBlock fails the 27-bit codeword
    # limit on skewed data where Adaptive succeeds, and that shows up as a
    # smaller n rather than as a worse ratio. Print n and the miss count.
    rowsout = []
    for var in others:
        ratios, missing_base, missing_var, flagged = [], 0, 0, 0
        for ck, cv in cells.items():
            b, o = cv.get(args.baseline), cv.get(var)
            if b is None and o is not None:
                missing_base += 1
                continue
            if o is None:
                missing_var += 1
                continue
            ratios.append((b / o) if better == "lower" else (o / b))
            if (ck, var) in unreliable or (ck, args.baseline) in unreliable:
                flagged += 1
        g = geomean(ratios)
        rowsout.append((var, g, len(ratios), missing_base, missing_var, flagged))

    w = max(len(v) for v in others) + 2
    print(f"{'variant':<{w}} {'ratio':>8} {'n':>5} {'base-only-miss':>15} "
          f"{'arm-miss':>9} {'lowconf':>8}")
    print("-" * (w + 50))
    for var, g, n, mb, mv, fl in sorted(rowsout, key=lambda t: -(t[1] or 0)):
        gs = f"{g:.3f}x" if g else "  n/a"
        print(f"{var:<{w}} {gs:>8} {n:>5} {mb:>15} {mv:>9} {fl:>8}")

    print("\n  base-only-miss = cells where the BASELINE produced no usable row but "
          "this arm did\n  arm-miss       = cells where THIS ARM produced none but "
          "the baseline did\n  lowconf        = cells counted but with "
          "timing_reliable=False on either side")

    if args.by_field:
        print("\nper-cell:")
        hdr = f"{'dataset':<13}{'field':<17}{'eb':>8}  " + "".join(
            f"{v[:14]:>16}" for v in [args.baseline] + others)
        print(hdr)
        print("-" * len(hdr))
        for ck in sorted(cells, key=lambda k: (str(k[0]), str(k[1]), k[2] or 0)):
            cv = cells[ck]
            base = cv.get(args.baseline)
            line = f"{str(ck[0]):<13}{str(ck[1]):<17}{ck[2]:>8.0e}  "
            line += f"{(f'{base:.2f}' if base else '-'):>16}"
            for var in others:
                o = cv.get(var)
                if o is None or base is None:
                    line += f"{'-':>16}"
                else:
                    r = (base / o) if better == "lower" else (o / base)
                    mark = "!" if ((ck, var) in unreliable) else ""
                    line += f"{f'{r:.3f}x{mark}':>16}"
            print(line)

    if not args.no_gate:
        print()
        print(validity.exclusion_report(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
