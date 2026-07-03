"""Minimal results summary (stdlib only).

M1 deliverable: turn runs.jsonl into one readable comparison table. The richer
rate-distortion / delta-report analysis (pandas, matplotlib) lands in M4 under the
repo-root analysis/ directory.
"""
from __future__ import annotations


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
