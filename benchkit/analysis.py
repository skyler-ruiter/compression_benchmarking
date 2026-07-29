"""Minimal results summary (stdlib only).

M1 deliverable: turn runs.jsonl into one readable comparison table. The richer
rate-distortion / delta-report analysis (pandas, matplotlib) lands in M4 under the
repo-root analysis/ directory.
"""
from __future__ import annotations

import math

from . import validity


_COLS = [
    ("compressor", "comp", 6),
    ("variant", "variant", 10),
    ("dataset", "dataset", 14),
    ("field", "field", 12),
    ("error_bound", "eb", 8),
    ("cr", "CR", 8),
    ("psnr", "PSNR", 8),
    ("compress_throughput_gbs", "cGB/s", 8),
    ("decompress_throughput_gbs", "dGB/s", 8),
    ("eb_satisfied", "eb_ok", 6),
    ("timing_reliable", "tOK", 5),
    ("graph_active", "graph", 6),
]


def _fmt(key: str, val) -> str:
    if key == "graph_active":
        return "on" if val else ("off" if val is False else "-")
    if val is None:
        return "-"
    if key == "error_bound":
        return f"{val:g}"
    if key in ("cr", "psnr", "compress_throughput_gbs", "decompress_throughput_gbs"):
        try:
            return f"{float(val):.2f}"
        except (TypeError, ValueError):
            return str(val)
    return str(val)


def print_table(rows: list[dict]) -> None:
    ok = [r for r in rows if r.get("status") == "ok"]
    failed = [r for r in rows if r.get("status") != "ok"]
    if not rows:
        print("(no rows)")
        return

    header = "  ".join(f"{lbl:<{w}}" for _, lbl, w in _COLS)
    print(header)
    print("-" * len(header))
    for r in ok:
        print("  ".join(f"{_fmt(key, r.get(key)):<{w}}" for key, _, w in _COLS))

    print(f"\n{len(ok)} ok, {len(failed)} failed")
    for r in failed:
        print(f"  FAIL {r.get('run_id')}: {r.get('error_message')}")


_AGG_GROUP_KEYS = ("compressor", "variant", "pipeline", "error_bound")


def aggregate_cr(rows: list[dict], group_keys: tuple[str, ...] = _AGG_GROUP_KEYS,
                 gate: bool = True) -> list[dict]:
    """Roll a multi-field run matrix up into one aggregate CR per group (typically
    compressor/variant/pipeline/error_bound, i.e. one number per pipeline-under-test).

    Reports two statistics — both used in the compression literature, answering different
    questions (see docs/DESIGN.md "aggregate CR"):
      - ratio_of_sums_cr: sum(original_bytes)/sum(compressed_bytes) over fields. Size-weighted
        (a 512^3 field dominates a 64^3 one); this is what SDRBench-style papers usually mean
        by "overall CR" across a test suite.
      - geomean_cr: geometric mean of each field's own CR — every field counts equally
        regardless of size, so one huge/tiny field can't dominate the number.

    Reps of the same cell (same group + dataset + field) vary only in timing for a
    deterministic compressor, so they're collapsed to one cell via the *median*
    compressed_bytes before aggregating — a defensive statistic, not an assumption that
    reps disagree. Recomputes CR from raw original/compressed bytes per D4 (harness owns
    size metrics), never averages the per-row `cr` field directly for ratio_of_sums.

    `gate=True` (the default) applies benchkit.validity: constant/degenerate fields,
    expansions, and severe error-bound misses are kept out of the mean. `status == ok`
    alone admits cells that are numerically unusable — see benchkit/validity.py for the
    full rationale and `exclusion_report()` for an audit of what was dropped. Pass
    gate=False only to reproduce an ungated legacy number.
    """
    if gate:
        rows = [r for r in validity.annotate(rows) if validity.is_valid(r)]
    ok = [r for r in rows if r.get("status") == "ok" and r.get("compressed_bytes") and r.get("original_bytes")]

    cells: dict[tuple, list[dict]] = {}
    for r in ok:
        gkey = tuple(r.get(k) for k in group_keys)
        ckey = gkey + (r.get("dataset"), r.get("field"))
        cells.setdefault(ckey, []).append(r)

    field_cells: dict[tuple, list[dict]] = {}
    for ckey, reps in cells.items():
        gkey = ckey[:len(group_keys)]
        comp_sizes = sorted(int(r["compressed_bytes"]) for r in reps)
        median_comp = comp_sizes[len(comp_sizes) // 2]
        orig = int(reps[0]["original_bytes"])
        field_cells.setdefault(gkey, []).append({
            "dataset": reps[0].get("dataset"), "field": reps[0].get("field"),
            "original_bytes": orig, "compressed_bytes": median_comp,
            "cr": orig / median_comp,
        })

    out = []
    for gkey, fcells in sorted(field_cells.items(), key=lambda kv: kv[0]):
        total_orig = sum(fc["original_bytes"] for fc in fcells)
        total_comp = sum(fc["compressed_bytes"] for fc in fcells)
        crs = [fc["cr"] for fc in fcells]
        logs = [math.log(c) for c in crs]
        geomean = math.exp(sum(logs) / len(logs))

        # Spread, so a multi-field mean is never quoted bare. CR is a ratio and its
        # per-field distribution is right-skewed and strictly positive, so an arithmetic
        # SD would be the wrong shape (it can reach below zero on a wide group and
        # implies symmetry that is not there). The geometric SD is the multiplicative
        # analogue: it is a dimensionless FACTOR, and the one-sigma band is
        # geomean/gsd .. geomean*gsd. gsd = 1.0 means every field compressed identically.
        # Sample SD (n-1) for n > 1; a single-field group has no spread, reported as None
        # rather than 0.0 so "one field" is never mistaken for "perfectly consistent".
        if len(logs) > 1:
            mean_log = sum(logs) / len(logs)
            var_log = sum((x - mean_log) ** 2 for x in logs) / (len(logs) - 1)
            gsd = math.exp(math.sqrt(var_log))
        else:
            gsd = None
        lo = min(fcells, key=lambda fc: fc["cr"])
        hi = max(fcells, key=lambda fc: fc["cr"])

        out.append({
            **dict(zip(group_keys, gkey)),
            "n_fields": len(fcells),
            "fields": sorted(f"{fc['dataset']}/{fc['field']}" for fc in fcells),
            "ratio_of_sums_cr": total_orig / total_comp,
            "geomean_cr": geomean,
            "gsd_cr": gsd,
            "min_cr": lo["cr"],
            "min_cr_field": f"{lo['dataset']}/{lo['field']}",
            "max_cr": hi["cr"],
            "max_cr_field": f"{hi['dataset']}/{hi['field']}",
            "total_original_bytes": total_orig,
            "total_compressed_bytes": total_comp,
        })
    return out


def print_aggregate_table(rows: list[dict], group_keys: tuple[str, ...] = _AGG_GROUP_KEYS,
                          gate: bool = True) -> None:
    # The audit prints first, so a number is never read without the population it
    # was computed over being visible directly above it.
    if gate:
        print(validity.exclusion_report(rows))
        print()
    groups = aggregate_cr(rows, group_keys, gate=gate)
    if not groups:
        print("(no ok rows to aggregate)")
        return

    header_cols = list(group_keys) + ["n_fields", "ratio_of_sums_cr", "geomean_cr", "gsd_cr"]

    def cell(g, k):
        v = g[k]
        if v is None:
            return "-"
        if isinstance(v, float):
            return f"{v:.3f}" if k == "error_bound" else f"{v:.2f}"
        return str(v)

    # Width from the values as well as the header: `pipeline` holds full TOML paths on
    # fzgm rows, which are far wider than the 10-char header and would otherwise push
    # every later column out of alignment row by row.
    widths = {k: max(len(k), 10, *(len(cell(g, k)) for g in groups)) for k in header_cols}
    print("  ".join(f"{k:<{widths[k]}}" for k in header_cols))
    print("-" * (sum(widths.values()) + 2 * (len(header_cols) - 1)))
    for g in groups:
        print("  ".join(f"{cell(g, k):<{widths[k]}}" for k in header_cols))
        if g["n_fields"] > 1:
            # The range matters more than the field list once there are dozens of
            # fields: it says whether the mean describes the dataset or hides it.
            print(f"    range: {g['min_cr']:.2f} ({g['min_cr_field']})"
                  f"  ..  {g['max_cr']:.2f} ({g['max_cr_field']})")
        if g["n_fields"] <= 8:
            print(f"    fields: {', '.join(g['fields'])}")
