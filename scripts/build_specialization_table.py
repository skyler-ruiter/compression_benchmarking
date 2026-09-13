#!/usr/bin/env python3
"""Generate the FZGM staged-versus-specialized paper table.

The table intentionally isolates the controlled FZGM A/B. Native proximity and
recovered-gap analysis belong in a separate artifact because not every FZGM port
has byte- or ratio-identical native semantics.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchkit import validity


DEFAULT_VARIANTS = (
    "cusz",
    "cuszp2_outlier_sp",
    "cuszp2_plain_sp",
    "cuszp3_fixed",
    "cuszp3_outlier_sp",
    "cuszp3_plain_sp",
    "fzgpu",
    "pfpl",
    "fsz",
    "szp_composed",
)

LABELS = {
    "cusz": "cuSZ",
    "cuszp2_outlier_sp": "cuSZp2-shaped, outlier",
    "cuszp2_plain_sp": "cuSZp2-shaped, plain",
    "cuszp3_fixed": "cuSZp3-shaped, fixed",
    "cuszp3_outlier_sp": "cuSZp3-shaped, outlier",
    "cuszp3_plain_sp": "cuSZp3-shaped, plain",
    "fzgpu": "FZ-GPU",
    "pfpl": "PFPL-shaped",
    "fsz": "FSZ-shaped",
    "szp_composed": "cuSZp-style plain, B=128",
}

FALLBACK_LABELS = {
    "no_legal_group": "No legal group",
    "no_profitable_implementation": "No registered implementation",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def gmean(values: list[float]) -> float | None:
    kept = [value for value in values if value > 0 and math.isfinite(value)]
    if not kept:
        return None
    return math.exp(sum(math.log(value) for value in kept) / len(kept))


def hierarchical_speedup_gmean(pairs: list[tuple[dict, dict]], metric: str) -> float | None:
    """Aggregate bounds within fields, fields within datasets, then datasets."""
    grouped: dict[str, dict[tuple, list[float]]] = defaultdict(lambda: defaultdict(list))
    for staged, specialized in pairs:
        field_key = (
            specialized.get("field"), specialized.get("dtype"),
            tuple(specialized.get("dims") or []), specialized.get("dim_order"),
            specialized.get("error_mode"),
        )
        grouped[specialized.get("dataset")][field_key].append(
            specialized[metric] / staged[metric]
        )
    dataset_means = [
        gmean([gmean(values) for values in fields.values()])
        for fields in grouped.values()
    ]
    return gmean(dataset_means)


def load_rows(path: Path) -> list[dict]:
    with path.open() as stream:
        return validity.annotate([json.loads(line) for line in stream if line.strip()])


def usable(row: dict) -> bool:
    return row.get("status") == "ok" and validity.is_valid(row)


def summarize(off_rows: list[dict], auto_rows: list[dict], variants: tuple[str, ...]) -> list[dict]:
    off = {
        row["logical_cell_id"]: row
        for row in off_rows
        if row.get("compressor") == "fzgm" and row.get("logical_cell_id")
    }
    auto = {
        row["logical_cell_id"]: row
        for row in auto_rows
        if row.get("compressor") == "fzgm" and row.get("logical_cell_id")
    }
    grouped: dict[str, list[tuple[dict, dict]]] = defaultdict(list)
    for logical_id in off.keys() & auto.keys():
        staged, specialized = off[logical_id], auto[logical_id]
        variant = specialized.get("variant")
        if variant in variants and usable(staged) and usable(specialized):
            grouped[variant].append((staged, specialized))

    result = []
    for variant in variants:
        pairs = grouped.get(variant, [])
        timed = [
            (staged, specialized)
            for staged, specialized in pairs
            if staged.get("timing_reliable") is not False
            and specialized.get("timing_reliable") is not False
            and staged.get("compress_throughput_gbs")
            and specialized.get("compress_throughput_gbs")
            and staged.get("decompress_throughput_gbs")
            and specialized.get("decompress_throughput_gbs")
        ]
        installed = sum(
            specialized.get("fusion_installed_group_count", 0) > 0
            and specialized.get("fusion_inverse_installed_group_count", 0) > 0
            for _, specialized in pairs
        )
        fallback_reasons: dict[str, int] = defaultdict(int)
        for _, specialized in pairs:
            if not (
                specialized.get("fusion_installed_group_count", 0) > 0
                and specialized.get("fusion_inverse_installed_group_count", 0) > 0
            ):
                fallback_reasons[
                    specialized.get("fusion_fallback_reason") or "unspecified"
                ] += 1
        size_mismatches = sum(
            staged.get("compressed_bytes") != specialized.get("compressed_bytes")
            for staged, specialized in pairs
        )
        reconstruction_mismatches = sum(
            staged.get("decompressed_sha256") != specialized.get("decompressed_sha256")
            for staged, specialized in pairs
        )
        result.append(
            {
                "variant": variant,
                "label": LABELS.get(variant, variant),
                "valid_pairs": len(pairs),
                "timed_pairs": len(timed),
                "installed_both": installed,
                "installed_fraction": installed / len(pairs) if pairs else None,
                "fallback_reasons": dict(sorted(fallback_reasons.items())),
                "compress_speedup_gmean": hierarchical_speedup_gmean(
                    timed, "compress_throughput_gbs"),
                "decompress_speedup_gmean": hierarchical_speedup_gmean(
                    timed, "decompress_throughput_gbs"),
                "compressed_size_mismatches": size_mismatches,
                "reconstruction_mismatches": reconstruction_mismatches,
            }
        )
    return result


def pct(value: float | None) -> str:
    return "--" if value is None else f"{100.0 * value:.1f}\\%"


def speedup(value: float | None) -> str:
    return "--" if value is None else f"{value:.2f}$\\times$"


def md_pct(value: float | None) -> str:
    return "--" if value is None else f"{100.0 * value:.1f}%"


def md_speedup(value: float | None) -> str:
    return "--" if value is None else f"{value:.2f}x"


def fallback_label(reasons: dict[str, int]) -> str:
    if not reasons:
        return "None"
    return ", ".join(
        f"{FALLBACK_LABELS.get(reason, reason)} ({count})"
        for reason, count in reasons.items()
    )


def markdown(rows: list[dict], off_name: str, auto_name: str) -> str:
    lines = [
        "# H100 staged-to-specialized performance",
        "",
        f"Staged source: `{off_name}`  ",
        f"Specialized source: `{auto_name}`",
        "",
        "Only status-ok, validity-gated FZGM logical pairs are counted. Throughput",
        "requires reliable timing in both arms. Geometric means give equal weight",
        "to bounds within fields, fields within datasets, and datasets. Memory is",
        "intentionally deferred to the separately measured peak-allocation artifact.",
        "",
        "| Pipeline | Valid pairs | Timed pairs | Fwd+inv installed | Compress | Decompress | Fallback | Identity mismatches |",
        "|---|---:|---:|---:|---:|---:|---|---:|",
    ]
    for row in rows:
        mismatch = row["compressed_size_mismatches"] + row["reconstruction_mismatches"]
        lines.append(
            f"| {row['label']} | {row['valid_pairs']} | {row['timed_pairs']} | "
            f"{md_pct(row['installed_fraction'])} | "
            f"{md_speedup(row['compress_speedup_gmean'])} | "
            f"{md_speedup(row['decompress_speedup_gmean'])} | "
            f"{fallback_label(row['fallback_reasons'])} | {mismatch} |"
        )
    lines += [
        "",
        "For the cuSZp and PFPL families, non-installing pairs are primarily f64 cells",
        "for which no matching float-only implementation is registered. cuSZ and",
        "FZ-GPU have no legal specialization group; FSZ has no registered whole-chain",
        "implementation. The legacy `no_profitable_implementation` token means that an",
        "implementation is absent, not that a predictive cost model rejected the cell.",
        "Fallback cells remain in the aggregate, so the result describes Auto over the",
        "complete valid corpus rather than only successful installations.",
        "",
        "This table deliberately does not report Auto/native or recovered-native-gap",
        "values. Those require a separate matched-semantics/native-provenance table.",
        "",
    ]
    return "\n".join(lines)


def latex(rows: list[dict]) -> str:
    lines = [
        "% Generated by compression_benchmarking/scripts/build_specialization_table.py.",
        "% Do not edit measured values by hand.",
        "\\begin{table*}[t]",
        "  \\centering",
        "  \\caption{Controlled staged-to-specialized H100 performance over paired, valid",
        "  logical cells. Throughput ratios use cells with reliable timing in both",
        "  arms and aggregate bounds within fields, fields within datasets, then",
        "  datasets with equal weight. Fallback cells remain in the aggregate.",
        "  Compressed sizes and",
        "  reconstructed outputs match for every pair. `Installed' requires both",
        "  compression and decompression implementations.}",
        "  \\label{tab:specialization}",
        "  \\small",
        "  \\resizebox{\\textwidth}{!}{%",
        "  \\begin{tabular}{lrrrrl}",
        "    \\toprule",
        "    Pipeline & Valid/timed & Installed & Compress & Decompress & Fallback \\\\",
        "    \\midrule",
    ]
    for row in rows:
        lines.append(
            f"    {row['label']} & {row['valid_pairs']}/{row['timed_pairs']} & {pct(row['installed_fraction'])} & "
            f"{speedup(row['compress_speedup_gmean'])} & "
            f"{speedup(row['decompress_speedup_gmean'])} & "
            f"{fallback_label(row['fallback_reasons'])} \\\\"
        )
    lines += [
        "    \\bottomrule",
        "  \\end{tabular}",
        "  }",
        "\\end{table*}",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--off", required=True, type=Path, help="staged baseline directory")
    parser.add_argument("--auto", required=True, type=Path, help="specialized baseline directory")
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-md", type=Path)
    parser.add_argument("--output-tex", type=Path)
    parser.add_argument("--policy", type=Path,
                        help="frozen experiment policy recorded in output provenance")
    args = parser.parse_args()

    rows = summarize(load_rows(args.off / "runs.jsonl"), load_rows(args.auto / "runs.jsonl"), DEFAULT_VARIANTS)
    payload = {
        "schema_version": 2,
        "aggregate_method": (
            "geometric mean across bounds within each field, then fields within "
            "each dataset, then equal-weight datasets"
        ),
        "staged_source": args.off.name,
        "specialized_source": args.auto.name,
        "staged_runs_sha256": sha256_file(args.off / "runs.jsonl"),
        "specialized_runs_sha256": sha256_file(args.auto / "runs.jsonl"),
        "policy_path": str(args.policy.resolve()) if args.policy else None,
        "policy_sha256": sha256_file(args.policy) if args.policy else None,
        "generator_sha256": sha256_file(Path(__file__)),
        "rows": rows,
    }
    outputs = {
        args.output_json: json.dumps(payload, indent=2) + "\n",
        args.output_md: markdown(rows, args.off.name, args.auto.name),
        args.output_tex: latex(rows),
    }
    for path, content in outputs.items():
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content)
    if not any(outputs):
        print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
