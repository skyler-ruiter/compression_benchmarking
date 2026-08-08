"""Validity gating for benchmark rows — which cells may enter a reported mean.

Why this module exists
----------------------
`status == "ok"` means "both tools exited 0 and the harness computed metrics".
It does NOT mean the numbers are usable. The 2026-07-29 full-corpus sweep
(9,816 cells) produced 9,416 `ok` rows of which ~790 were numerically unusable:
compressors that expanded the data to 2x its original size while destroying it
and still exited 0, fields whose PSNR is legitimately infinite, and cells that
missed their error bound by three orders of magnitude. Aggregating those into a
mean silently corrupts every number downstream.

So the gate is explicit, data-driven, and auditable: every excluded row carries
a reason code, every reason code carries a written rationale, and
`exclusion_report()` prints the counts so a paper table can cite exactly what
was dropped and why. Nothing here deletes or rewrites rows — `runs.jsonl` stays
the raw record, and the gate is applied at read time.

Design notes
------------
- Degenerate (constant) fields are detected from the DATA, not from a hardcoded
  name list. CESM-2D's SFCLDICE/SFCLDLIQ are entirely zero today; the corpus has
  grown from 8 dataset families to 13 in one week and will keep growing, so a
  name list would silently rot. The detector asks "did EVERY compressor achieve
  exactly zero error on this field?", which is true only for constant (or
  otherwise trivially-representable) data.
- Marginal error-bound misses are RETAINED by default. 374 of the 388 marginal
  cases sit at the tightest bound (eb=1e-4) and exceed it by <1%, which is f32
  round-off in the prediction/reconstruction chain rather than a broken codec.
  Excluding them would hide a real, uniform property of every compressor tested.
  They are counted and reported instead. `MARGINAL_EB_RATIO` is the one knob.
- Severe misses ARE excluded: PSNR stays nominal while max error blows up by
  10-1000x, i.e. the bound is held in an RMS sense and violated at isolated
  points. Those cells describe a different thing than the mean they'd land in.

See docs/DESIGN.md D30 and results/baselines/*/metadata.yaml `validity_gate`.
"""
from __future__ import annotations

import math

# An error-bound overshoot at or below this factor is treated as boundary
# round-off, not a violation: retained in aggregates, reported as a caveat.
# Raise it only with evidence; lower it to 1.0 to gate strictly.
MARGINAL_EB_RATIO = 1.01

# code -> (short label, why this disqualifies a row from a reported mean)
REASONS: dict[str, tuple[str, str]] = {
    "failed": (
        "cell failed",
        "status != ok: the tool crashed or the harness could not compute metrics. "
        "No numbers exist to aggregate.",
    ),
    "degenerate_field": (
        "constant/degenerate field",
        "Every compressor reconstructed this field with exactly zero error, which "
        "identifies it as constant (or trivially representable). PSNR is then "
        "legitimately infinite and CR reflects the codec's constant-data path, not "
        "its behaviour on real data. Including it poisons any PSNR mean and inflates "
        "CR. Reported separately rather than silently averaged.",
    ),
    "expansion": (
        "output larger than input",
        "cr <= 1: the compressor expanded the data. Always paired with catastrophic "
        "quality in practice (observed at PSNR -23.6 dB). A 'compression ratio' below "
        "1 is not a compression result.",
    ),
    "eb_violated_severe": (
        f"error bound missed by > {MARGINAL_EB_RATIO}x",
        "max_abs_err exceeds the requested bound by more than the marginal factor. "
        "PSNR typically stays nominal, so the bound holds in RMS and fails at isolated "
        "points -- the cell is not describing the same quantity as the cells beside it "
        "in a mean. Kept out of aggregates; counted and reported.",
    ),
    "psnr_nonfinite": (
        "PSNR not finite",
        "PSNR is None, inf or nan for a field that is NOT degenerate, so the cause is "
        "unexplained rather than benign. Excluded from quality means; the row is still "
        "valid for CR and throughput, so this reason alone does not gate those.",
    ),
    "lossless_expansion": (
        "lossless output larger than input",
        "cr <= 1 on a lossless run. Unlike the lossy `expansion` case this is a real "
        "measurement, not a failure: some fields are incompressible byte-wise and the "
        "codec's framing overhead shows up as a ratio below 1. RETAINED in aggregates "
        "-- excluding it would bias every lossless CR mean upward by deleting the "
        "cases where the codec lost. Counted here so the count can be quoted.",
    ),
    "lossless_exact": (
        "lossless run, no quality to report",
        "error_mode == lossless: the codec reconstructed the input bit-exactly, so MSE "
        "is 0 and PSNR is legitimately infinite. This is the run succeeding, not an "
        "anomaly -- but an infinite PSNR cannot enter a quality mean, so the row is "
        "gated out of quality aggregates only. Its CR and throughput are fully valid "
        "and are the whole point of the run.",
    ),
}

# Reasons that disqualify a row from EVERY aggregate. `psnr_nonfinite` and
# `lossless_exact` are deliberately absent: they gate quality only (see
# quality_valid()).
_HARD_REASONS = ("failed", "degenerate_field", "expansion", "eb_violated_severe")


def _finite(x) -> bool:
    return x is not None and isinstance(x, (int, float)) and math.isfinite(x)


def is_lossless(row: dict) -> bool:
    """True for a run that declared no error bound and must be bit-exact."""
    return row.get("error_mode") == "lossless"


def degenerate_fields(rows: list[dict]) -> set[tuple]:
    """(dataset, field) pairs where every ok row reconstructed with zero error.

    Data-driven on purpose -- see the module docstring. A field is only called
    degenerate if it has at least one ok row and *all* of them are exact, so a
    single lossless cell among lossy ones never triggers it.

    `lossless` rows are excluded from the vote entirely, not merely outvoted. The
    detector infers "this field is constant" from "no compressor could produce any
    error on it", and that inference is only sound over LOSSY runs -- a lossless
    codec reconstructs every field exactly, constant or not. Without this, an
    nvCOMP-vs-FZGM-lossless session (where every row is exact by construction)
    would mark all 186 fields degenerate and the gate would drop the entire
    session. See DESIGN.md D31.
    """
    seen: dict[tuple, list[dict]] = {}
    for r in rows:
        if r.get("status") != "ok" or is_lossless(r):
            continue
        seen.setdefault((r.get("dataset"), r.get("field")), []).append(r)
    out = set()
    for key, rs in seen.items():
        if rs and all(_finite(r.get("max_abs_err")) and float(r["max_abs_err"]) == 0.0
                      for r in rs):
            out.add(key)
    return out


def row_reasons(row: dict, degenerate: set[tuple]) -> list[str]:
    """Exclusion reason codes for one row; empty list means fully usable."""
    out: list[str] = []
    if row.get("status") != "ok":
        return ["failed"]

    if (row.get("dataset"), row.get("field")) in degenerate:
        out.append("degenerate_field")

    cr = row.get("cr")
    if _finite(cr) and float(cr) <= 1.0:
        # For a LOSSY codec, cr <= 1 is a corruption signal: it has never been seen
        # without catastrophic quality beside it (PSNR -23.6 dB), so the row is
        # describing a failure, not a ratio. For a LOSSLESS codec it is an ordinary
        # measurement -- nvCOMP LZ4 compresses CESM-2D/CLDHGH to 0.996x, because raw
        # f32 mantissas are close to incompressible byte-wise and the framing
        # overhead is real. Dropping those rows would bias every lossless CR mean
        # upward by silently deleting exactly the cases where the codec lost.
        # Counted and reported separately, but RETAINED.
        out.append("lossless_expansion" if is_lossless(row) else "expansion")

    if row.get("eb_satisfied") is False:
        ratio = row.get("err_over_bound")
        if not _finite(ratio) or float(ratio) > MARGINAL_EB_RATIO:
            # A lossless codec that returned anything but the input is broken, not
            # merely off-bound -- there is no MARGINAL_EB_RATIO slack to fall under
            # (metrics.compute_quality sets err_over_bound = 0 for these, so the
            # `not _finite` arm never carries them here).
            out.append("eb_violated_severe")

    # An infinite PSNR means "exact" either way; what differs is whether that was
    # the point. For a lossless run it is the contract being met, so it gets its own
    # reason code rather than being filed under "unexplained".
    if not _finite(row.get("psnr")) and "degenerate_field" not in out:
        out.append("lossless_exact" if is_lossless(row) else "psnr_nonfinite")

    return out


def annotate(rows: list[dict]) -> list[dict]:
    """Attach `_exclusions` (list of codes) to a copy of each row.

    Does not mutate the input rows, so callers that also write results are never
    at risk of persisting harness-side annotations into runs.jsonl.
    """
    degen = degenerate_fields(rows)
    return [{**r, "_exclusions": row_reasons(r, degen)} for r in rows]


def is_valid(row: dict) -> bool:
    """Usable in CR / throughput aggregates."""
    return not any(c in _HARD_REASONS for c in row.get("_exclusions", []))


def quality_valid(row: dict) -> bool:
    """Usable in PSNR / quality aggregates (stricter than is_valid)."""
    return not row.get("_exclusions")


def marginal_eb_count(rows: list[dict]) -> int:
    """Rows retained despite missing the bound within MARGINAL_EB_RATIO."""
    n = 0
    for r in rows:
        if r.get("status") != "ok" or r.get("eb_satisfied") is not False:
            continue
        ratio = r.get("err_over_bound")
        if _finite(ratio) and float(ratio) <= MARGINAL_EB_RATIO:
            n += 1
    return n


def exclusion_report(rows: list[dict]) -> str:
    """Human-readable audit of what the gate drops and why.

    Printed alongside any aggregate table so a quoted mean can always be traced
    back to the population it was computed over.
    """
    ann = annotate(rows)
    total = len(ann)
    valid = sum(1 for r in ann if is_valid(r))
    qvalid = sum(1 for r in ann if quality_valid(r))

    counts: dict[str, int] = {k: 0 for k in REASONS}
    for r in ann:
        for c in r["_exclusions"]:
            counts[c] += 1

    lines = ["Validity gate", "=" * 13,
             f"{total} rows -> {valid} usable for CR/throughput, "
             f"{qvalid} usable for quality (PSNR)", ""]
    for code, n in sorted(counts.items(), key=lambda kv: -kv[1]):
        if not n:
            continue
        label, why = REASONS[code]
        gates = "all aggregates" if code in _HARD_REASONS else "quality aggregates only"
        lines.append(f"  {code}  n={n}  [{label}] -- excluded from {gates}")
        for chunk in _wrap(why, 88):
            lines.append(f"      {chunk}")
        lines.append("")

    degen = degenerate_fields(rows)
    if degen:
        lines.append(f"  degenerate fields ({len(degen)}): "
                     + ", ".join(f"{d}/{f}" for d, f in sorted(degen)))
        lines.append("")

    marg = marginal_eb_count(rows)
    if marg:
        lines.append(f"  RETAINED: {marg} rows miss the error bound by <= "
                     f"{MARGINAL_EB_RATIO}x and are kept (boundary round-off, not a")
        lines.append("      broken codec -- see benchkit/validity.py). Quote this count "
                     "with any eb-satisfaction claim.")
    return "\n".join(lines)


def _wrap(text: str, width: int) -> list[str]:
    words, lines, cur = text.split(), [], ""
    for w in words:
        if cur and len(cur) + 1 + len(w) > width:
            lines.append(cur)
            cur = w
        else:
            cur = f"{cur} {w}" if cur else w
    if cur:
        lines.append(cur)
    return lines
