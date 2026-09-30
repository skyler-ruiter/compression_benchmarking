#!/usr/bin/env python3
"""Size-stratified specialization performance, fixed per-invocation cost, and
in-kernel versus outside-kernel device time.

Inputs are the frozen artifacts behind the paper's specialization figure, so every
number here inherits that figure's validity and timing gates:

* ``--native-csv``: specialization_native_performance.csv from
  build_specialization_native_figure.py (one gated row per family/phase/coordinate).
* ``--auto``: the Auto session directory, for per-stage device times.
* ``--crossover-off`` / ``--crossover-auto``: the controlled size-sweep sessions, for
  the fixed per-invocation cost.

Size strata are fixed in advance from the controlled crossover sweep (the cuSZp2-shaped
compression crossover lies between 2^20 and 2^24 elements), not tuned to the corpus.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchkit import validity  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "native_figure", ROOT / "scripts" / "build_specialization_native_figure.py")
_native_figure = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_native_figure)
hierarchical_gmean = _native_figure.hierarchical_gmean
gmean = _native_figure.gmean
sha256_file = _native_figure.sha256_file

# Element-count strata, chosen from the crossover sweep before looking at the corpus.
STRATA = (
    ("< 2^24", 0, 1 << 24),
    ("2^24 - 2^27", 1 << 24, 1 << 27),
    (">= 2^27", 1 << 27, math.inf),
)

# FZGPUModules variants whose Auto rows carry a generated (fused:*) stage.
SPECIALIZED_VARIANTS = (
    "cuszp2_plain_sp", "cuszp2_outlier_sp", "cuszp3_fixed", "cuszp3_plain_sp",
    "cuszp3_outlier_sp", "pfpl",
)


def elements(dims: list[int]) -> int:
    return math.prod(dims)


def stratum(n: int) -> str:
    for label, lo, hi in STRATA:
        if lo <= n < hi:
            return label
    raise ValueError(n)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def size_strata(csv_path: Path) -> list[dict[str, Any]]:
    with csv_path.open() as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        row["dims"] = json.loads(row["dims"])
        row["elements"] = elements(row["dims"])
        row["installed"] = row["auto_specialization_installed"] == "True"
        for key in ("auto_over_native", "auto_over_staged", "staged_over_native"):
            row[key] = float(row[key])
    out = []
    families = sorted({r["family"] for r in rows})
    for family in families:
        for phase in ("compression", "decompression"):
            for label, lo, hi in STRATA:
                sel = [r for r in rows if r["family"] == family and r["phase"] == phase
                       and lo <= r["elements"] < hi]
                if not sel:
                    continue
                installed = [r for r in sel if r["installed"]]
                out.append({
                    "family": family, "phase": phase, "stratum": label,
                    "pairs": len(sel), "installed_pairs": len(installed),
                    "datasets": sorted({r["dataset"] for r in sel}),
                    "auto_over_native_gmean": hierarchical_gmean(sel, "auto_over_native"),
                    "auto_over_staged_gmean": hierarchical_gmean(sel, "auto_over_staged"),
                    "installed_auto_over_native_gmean": hierarchical_gmean(
                        installed, "auto_over_native") if installed else None,
                    "installed_auto_over_staged_gmean": hierarchical_gmean(
                        installed, "auto_over_staged") if installed else None,
                    "auto_at_or_above_native_fraction":
                        sum(r["auto_over_native"] >= 1.0 for r in sel) / len(sel),
                })
    return out


def per_dataset(csv_path: Path) -> list[dict[str, Any]]:
    with csv_path.open() as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        row["dims"] = json.loads(row["dims"])
        for key in ("auto_over_native", "auto_over_staged"):
            row[key] = float(row[key])
    out = []
    for family in sorted({r["family"] for r in rows}):
        for phase in ("compression", "decompression"):
            for dataset in sorted({r["dataset"] for r in rows}):
                sel = [r for r in rows if r["family"] == family and r["phase"] == phase
                       and r["dataset"] == dataset]
                if not sel:
                    continue
                out.append({
                    "family": family, "phase": phase, "dataset": dataset,
                    "elements": elements(sel[0]["dims"]), "dtype": sel[0]["dtype"],
                    "pairs": len(sel),
                    "auto_over_native_gmean": hierarchical_gmean(sel, "auto_over_native"),
                    "auto_over_staged_gmean": hierarchical_gmean(sel, "auto_over_staged"),
                })
    return out


def kernel_share(auto_dir: Path) -> list[dict[str, Any]]:
    """Share of specialized device time spent outside generated (fused:*) kernels."""
    rows = load_jsonl(auto_dir / "runs.jsonl")
    out = []
    for variant in SPECIALIZED_VARIANTS:
        for phase, metric in (("compress", "compress_device_ms_median"),
                              ("decompress", "decompress_device_ms_median")):
            for label, lo, hi in STRATA:
                shares = []
                for r in rows:
                    if (r.get("compressor"), r.get("variant")) != ("fzgm", variant):
                        continue
                    if r.get("status") != "ok" or not validity.is_valid(r):
                        continue
                    if r.get("timing_reliable") is False:
                        continue
                    if not (lo <= elements(r["dims"]) < hi):
                        continue
                    kernel = sum(s["device_ms"] for s in r.get("stages") or []
                                 if s.get("phase") == phase
                                 and s.get("name", "").startswith("fused"))
                    total = r.get(metric)
                    if kernel <= 0 or not total:
                        continue
                    shares.append(max(total - kernel, 0.0) / total)
                if shares:
                    shares.sort()
                    out.append({
                        "variant": variant, "phase": phase, "stratum": label,
                        "rows": len(shares),
                        "outside_kernel_share_mean": sum(shares) / len(shares),
                        "outside_kernel_share_median": shares[len(shares) // 2],
                        "outside_kernel_share_p90": shares[int(0.9 * (len(shares) - 1))],
                    })
    return out


def fixed_cost(off_dir: Path, auto_dir: Path) -> list[dict[str, Any]]:
    """Per-invocation device-time floor from the controlled size sweep.

    Reports the median device time at the smallest swept size (where the per-element
    term is negligible) and an ordinary least-squares fit t = a + b*N over all sizes.
    """
    def collect(path: Path) -> dict[tuple, list[tuple[int, float, float]]]:
        groups: dict[tuple, list[tuple[int, float, float]]] = {}
        for r in load_jsonl(path / "runs.jsonl"):
            if r.get("compressor") != "fzgm" or r.get("status") != "ok":
                continue
            key = (r.get("variant"), r.get("pipeline"))
            groups.setdefault(key, []).append((
                r["num_elements"], r["compress_device_ms_median"],
                r["decompress_device_ms_median"]))
        return groups

    def ols(points: list[tuple[int, float]]) -> tuple[float, float]:
        n = len(points)
        mx = sum(p[0] for p in points) / n
        my = sum(p[1] for p in points) / n
        sxx = sum((p[0] - mx) ** 2 for p in points)
        b = sum((p[0] - mx) * (p[1] - my) for p in points) / sxx
        return my - b * mx, b

    off, auto = collect(off_dir), collect(auto_dir)
    out = []
    for key in sorted(set(off) & set(auto), key=repr):
        for arm, data in (("staged", off[key]), ("auto", auto[key])):
            smallest = min(p[0] for p in data)
            for phase, idx in (("compress", 1), ("decompress", 2)):
                small = sorted(p[idx] for p in data if p[0] == smallest)
                a, b = ols([(p[0], p[idx]) for p in data])
                out.append({
                    "variant": key[0], "pipeline": key[1], "arm": arm, "phase": phase,
                    "smallest_elements": smallest,
                    "device_ms_at_smallest": small[len(small) // 2],
                    "ols_intercept_ms": a, "ols_ns_per_element": b * 1e6,
                    "sizes": sorted({p[0] for p in data}),
                })
    return out


def fmt(x: float | None, spec: str = ".2f") -> str:
    return "--" if x is None else format(x, spec)


def markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Specialization performance by input size",
        "",
        f"Native CSV: `{payload['sources']['native_csv']}`  ",
        f"Auto session: `{payload['sources']['auto']}`  ",
        f"Crossover sessions: `{payload['sources']['crossover_off']}`, "
        f"`{payload['sources']['crossover_auto']}`",
        "",
        "Strata are element counts fixed from the controlled crossover sweep. Means are",
        "hierarchical geometric means (bounds within fields, fields within datasets,",
        "datasets equally weighted) over the figure's gated coordinates.",
        "",
        "## Size strata",
        "",
        "| Family | Phase | Stratum | Pairs | Installed | Auto/native | Auto/staged | "
        "Installed Auto/native | Installed Auto/staged | Auto>=native | Datasets |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for r in payload["size_strata"]:
        lines.append(
            f"| {r['family']} | {r['phase']} | {r['stratum']} | {r['pairs']} | "
            f"{r['installed_pairs']} | {fmt(r['auto_over_native_gmean'])} | "
            f"{fmt(r['auto_over_staged_gmean'])} | "
            f"{fmt(r['installed_auto_over_native_gmean'])} | "
            f"{fmt(r['installed_auto_over_staged_gmean'])} | "
            f"{r['auto_at_or_above_native_fraction']:.0%} | {', '.join(r['datasets'])} |")
    lines += ["", "## Per dataset", "",
              "| Family | Phase | Dataset | Elements | dtype | Pairs | Auto/native | Auto/staged |",
              "|---|---|---|---:|---|---:|---:|---:|"]
    for r in payload["per_dataset"]:
        lines.append(
            f"| {r['family']} | {r['phase']} | {r['dataset']} | {r['elements']:,} | "
            f"{r['dtype']} | {r['pairs']} | {fmt(r['auto_over_native_gmean'])} | "
            f"{fmt(r['auto_over_staged_gmean'])} |")
    lines += ["", "## Device time outside generated kernels", "",
              "| Variant | Phase | Stratum | Rows | Mean share | Median | p90 |",
              "|---|---|---|---:|---:|---:|---:|"]
    for r in payload["kernel_share"]:
        lines.append(
            f"| {r['variant']} | {r['phase']} | {r['stratum']} | {r['rows']} | "
            f"{r['outside_kernel_share_mean']:.1%} | {r['outside_kernel_share_median']:.1%} | "
            f"{r['outside_kernel_share_p90']:.1%} |")
    lines += ["", "## Fixed per-invocation cost (controlled size sweep)", "",
              "| Variant | Pipeline | Arm | Phase | Smallest N | ms at smallest N | "
              "OLS intercept (ms) | OLS ns/element |",
              "|---|---|---|---|---:|---:|---:|---:|"]
    for r in payload["fixed_cost"]:
        lines.append(
            f"| {r['variant']} | {r['pipeline']} | {r['arm']} | {r['phase']} | "
            f"{r['smallest_elements']:,} | {r['device_ms_at_smallest']:.4f} | "
            f"{r['ols_intercept_ms']:.4f} | {r['ols_ns_per_element']:.4f} |")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-csv", type=Path, required=True)
    parser.add_argument("--auto", type=Path, required=True)
    parser.add_argument("--crossover-off", type=Path, required=True)
    parser.add_argument("--crossover-auto", type=Path, required=True)
    parser.add_argument("--output-prefix", type=Path, required=True)
    args = parser.parse_args()

    payload = {
        "sources": {
            "native_csv": str(args.native_csv.resolve()),
            "native_csv_sha256": sha256_file(args.native_csv),
            "auto": args.auto.name,
            "auto_runs_sha256": sha256_file(args.auto / "runs.jsonl"),
            "crossover_off": args.crossover_off.name,
            "crossover_auto": args.crossover_auto.name,
        },
        "generator_sha256": sha256_file(Path(__file__)),
        "strata": [{"label": s[0], "min_elements": s[1],
                    "max_elements": None if s[2] == math.inf else s[2]} for s in STRATA],
        "size_strata": size_strata(args.native_csv),
        "per_dataset": per_dataset(args.native_csv),
        "kernel_share": kernel_share(args.auto),
        "fixed_cost": fixed_cost(args.crossover_off, args.crossover_auto),
    }
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    args.output_prefix.with_suffix(".json").write_text(json.dumps(payload, indent=2) + "\n")
    args.output_prefix.with_suffix(".md").write_text(markdown(payload))


if __name__ == "__main__":
    main()
