#!/usr/bin/env python3
"""Compare two independent native-only B1 validity recheck sessions."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from benchkit.schema import load_result_file
from benchkit.validity import annotate


KEYS = ("compressor", "variant", "pipeline", "dataset", "field", "dtype", "error_mode", "error_bound")


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def key(row: dict) -> tuple:
    return tuple(row.get(k) for k in KEYS)


def fmt(value: object) -> str:
    return f"{value:.6g}" if isinstance(value, (int, float)) else "---"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--session", type=Path, action="append", required=True)
    p.add_argument("--output-prefix", type=Path, required=True)
    a = p.parse_args()
    if len(a.session) != 2:
        raise ValueError("exactly two --session arguments are required")

    sources = []
    indexed = []
    for session in a.session:
        runs = session.resolve() / "runs.jsonl"
        rows = annotate(load_result_file(runs))
        idx = {key(r): r for r in rows}
        if len(idx) != len(rows):
            raise ValueError(f"{session}: duplicate logical comparison keys")
        indexed.append(idx)
        sources.append({"session": session.name, "runs_sha256": digest(runs), "rows": len(rows)})

    if indexed[0].keys() != indexed[1].keys():
        raise ValueError("the two sessions do not contain identical coordinates")

    details = []
    for k in sorted(indexed[0], key=repr):
        x, y = indexed[0][k], indexed[1][k]
        severe_x = "eb_violated_severe" in x["_exclusions"]
        severe_y = "eb_violated_severe" in y["_exclusions"]
        details.append({
            **dict(zip(KEYS, k)),
            "status_a": x.get("status"), "status_b": y.get("status"),
            "err_over_bound_a": x.get("err_over_bound"),
            "err_over_bound_b": y.get("err_over_bound"),
            "severe_a": severe_x, "severe_b": severe_y,
            "decompressed_sha256_equal": (
                x.get("decompressed_sha256") is not None
                and x.get("decompressed_sha256") == y.get("decompressed_sha256")
            ),
        })

    payload = {
        "analysis_schema_version": 1,
        "sources": sources,
        "coordinates": len(details),
        "severe_in_both": sum(d["severe_a"] and d["severe_b"] for d in details),
        "validity_class_equal": sum(d["severe_a"] == d["severe_b"] for d in details),
        "reconstruction_hash_equal": sum(d["decompressed_sha256_equal"] for d in details),
        "rows": details,
    }
    prefix = a.output_prefix.resolve()
    prefix.parent.mkdir(parents=True, exist_ok=True)
    prefix.with_suffix(".json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    lines = [
        "# Native error-bound recheck",
        "",
        "Two independent Benchkit sessions reran each coordinate in a fresh process.",
        "",
        f"- Coordinates: **{payload['coordinates']}**",
        f"- Same severe/non-severe classification: **{payload['validity_class_equal']}/{payload['coordinates']}**",
        f"- Severe in both sessions: **{payload['severe_in_both']}**",
        f"- Identical reconstructed-data hash: **{payload['reconstruction_hash_equal']}/{payload['coordinates']}**",
        "",
        "| Compressor | Dataset/field | Bound | Error/bound A | Error/bound B | Severe in both | Hash equal |",
        "|---|---|---:|---:|---:|:---:|:---:|",
    ]
    for d in details:
        lines.append(
            f"| {d['compressor']} ({d['variant']}) | {d['dataset']}/{d['field']} | "
            f"{d['error_bound']:.0e} | {fmt(d['err_over_bound_a'])} | "
            f"{fmt(d['err_over_bound_b'])} | "
            f"{'yes' if d['severe_a'] and d['severe_b'] else 'no'} | "
            f"{'yes' if d['decompressed_sha256_equal'] else 'no'} |"
        )
    lines.extend(["", "The JSON companion records source checksums and all row-level outcomes.", ""])
    prefix.with_suffix(".md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
