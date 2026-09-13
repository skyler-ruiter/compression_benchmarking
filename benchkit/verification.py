"""Mechanical publication-completion checks for a Benchkit session."""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import re
from typing import Any

import yaml

from .identity import assert_no_id_collisions, logical_merge_key, normalize_pipeline_ref
from .provenance import assign_provenance_id
from .schema import load_result_file, load_session_text
from .store import sha256_file
from . import validity


VERIFICATION_SCHEMA_VERSION = 1
REQUIRED_CHECKS = (
    "provenance_schema", "result_schema", "merge_state", "unique_ids",
    "provenance_joins", "artifact_checksums", "matrix_coverage", "dataset_joins", "failures",
    "validity_exclusions", "timing_reliability", "h3_eligibility",
)
_POLICY_SELECTOR_KEYS = {
    "compressor", "variant", "pipeline", "dataset", "field", "error_mode",
    "error_bound", "error_type", "fail_phase",
}
_GATING_EXCLUSIONS = {"degenerate_field", "eb_violated_severe", "psnr_nonfinite"}


def _check(name: str, passed: bool, summary: str, details=None) -> dict:
    out = {"name": name, "status": "pass" if passed else "fail", "summary": summary}
    if details:
        out["details"] = details
    return out


def _safe_file(session: Path, relative: str) -> Path:
    candidate = (session / relative).resolve()
    try:
        candidate.relative_to(session.resolve())
    except ValueError as exc:
        raise ValueError(f"artifact path escapes session: {relative}") from exc
    return candidate


def _load_provenance(session: Path) -> tuple[list[dict], list[str]]:
    addressed = sorted(session.glob("provenance.provenance-v1-*.json"))
    files = addressed or sorted(session.glob("provenance*.json"))
    manifests, errors, seen = [], [], set()
    for path in files:
        try:
            manifest = load_session_text(path.read_text())
            pid = manifest.get("provenance_id")
            if addressed and path.name != f"provenance.{pid}.json":
                errors.append(f"{path.name}: filename does not match provenance_id")
            if pid in seen:
                continue
            seen.add(pid)
            manifests.append(manifest)
        except Exception as exc:  # malformed evidence belongs in the report
            errors.append(f"{path.name}: {exc}")
    return manifests, errors


def _merge_precedence(files: list[Path]) -> list[dict]:
    best: dict[str, tuple[int, dict]] = {}
    order: list[str] = []
    for rank, path in enumerate(files):
        for row in load_result_file(path):
            key = logical_merge_key(row)
            current = best.get(key)
            if current is None:
                best[key] = (rank, row)
                order.append(key)
                continue
            current_rank, current_row = current
            current_ok, new_ok = current_row.get("status") == "ok", row.get("status") == "ok"
            if current_ok != new_ok:
                if new_ok:
                    best[key] = (rank, row)
            elif rank == current_rank:
                best[key] = (rank, row)
    return [best[key][1] for key in order]


def _artifact_checks(session: Path, manifests: list[dict], rows: list[dict]) -> list[str]:
    errors: list[str] = []
    seen: set[tuple[str, str]] = set()
    artifacts = []
    for manifest in manifests:
        artifacts.extend(manifest.get("input_artifacts", {}).values())
    artifacts.extend(r.get("rendered_pipeline_artifact") for r in rows
                     if r.get("rendered_pipeline_artifact"))
    for artifact in artifacts:
        if not isinstance(artifact, dict):
            errors.append(f"invalid artifact descriptor: {artifact!r}")
            continue
        rel, expected = artifact.get("path"), artifact.get("sha256")
        if not isinstance(rel, str) or not isinstance(expected, str):
            errors.append(f"incomplete artifact descriptor: {artifact!r}")
            continue
        if (rel, expected) in seen:
            continue
        seen.add((rel, expected))
        try:
            path = _safe_file(session, rel)
            if not path.is_file():
                errors.append(f"missing artifact: {rel}")
            elif sha256_file(path) != expected:
                errors.append(f"checksum mismatch: {rel}")
        except Exception as exc:
            errors.append(f"{rel}: {exc}")
    return errors


def _input_yaml(session: Path, manifests: list[dict], name: str) -> dict:
    candidates = []
    for manifest in manifests:
        artifact = manifest.get("input_artifacts", {}).get(name)
        if artifact:
            candidates.append(artifact)
    digests = {a.get("sha256") for a in candidates}
    if not candidates:
        raise ValueError(f"no archived {name} input")
    if len(digests) != 1:
        raise ValueError(f"shards disagree on archived {name} input")
    return yaml.safe_load(_safe_file(session, candidates[0]["path"]).read_text()) or {}


def _matrix_key(compressor, variant, pipeline, dataset, field, mode, bound) -> tuple:
    slot = None if mode in {"lossless", "from_toml"} else float(bound)
    return (compressor, variant, normalize_pipeline_ref(pipeline), dataset, field, mode, slot)


def _expected_matrix(experiment: dict, datasets: dict) -> Counter:
    mode = experiment.get("error", {}).get("mode", "rel_range")
    bounds = [None] if mode in {"lossless", "from_toml"} else experiment["error"]["bounds"]
    selected = experiment.get("datasets", [])
    field_selection = experiment.get("fields", "all")
    expected: Counter = Counter()
    for run in experiment.get("runs", []):
        only, skip = run.get("only_datasets"), run.get("skip_datasets")
        for dataset in selected:
            if only is not None and dataset not in only:
                continue
            if skip is not None and dataset in skip:
                continue
            if field_selection in ("all", None):
                fields = list(datasets[dataset].get("fields", {}))
            elif isinstance(field_selection, list):
                fields = field_selection
            else:
                fields = field_selection.get(dataset, "all")
                if fields == "all" or fields is None:
                    fields = list(datasets[dataset].get("fields", {}))
            for field in fields:
                for bound in bounds:
                    expected[_matrix_key(run["compressor"],
                                         run.get("variant", run["compressor"]),
                                         run.get("pipeline", "default"), dataset, field,
                                         mode, bound)] += 1
    return expected


def _actual_matrix(rows: list[dict]) -> Counter:
    return Counter(_matrix_key(r.get("compressor"), r.get("variant"), r.get("pipeline"),
                               r.get("dataset"), r.get("field"), r.get("error_mode"),
                               r.get("error_bound")) for r in rows)


def _matches(row: dict, selector: dict) -> bool:
    return all(row.get(key) == value for key, value in selector.items())


def _policy(experiment: dict, kind: str) -> list[dict]:
    entries = experiment.get("verification", {}).get(kind, [])
    if not isinstance(entries, list):
        raise ValueError(f"verification.{kind} must be a list")
    for entry in entries:
        if not isinstance(entry, dict) or not isinstance(entry.get("reason"), str) or not entry["reason"].strip():
            raise ValueError(f"verification.{kind} entries need a non-empty reason")
        match = entry.get("match", {})
        if not isinstance(match, dict) or set(match) - _POLICY_SELECTOR_KEYS:
            raise ValueError(f"verification.{kind} has invalid match selector")
        codes = entry.get("codes")
        if codes is not None and (not isinstance(codes, list) or not codes or
                                  any(code not in _GATING_EXCLUSIONS for code in codes)):
            raise ValueError(f"verification.{kind} has invalid exclusion codes")
        if not match and not (kind == "accepted_exclusions" and codes):
            raise ValueError(f"verification.{kind} entries need a non-empty match selector")
    return entries


def _accepted(row: dict, entries: list[dict], code: str | None = None) -> bool:
    for entry in entries:
        codes = entry.get("codes")
        if code is not None and codes is not None and code not in codes:
            continue
        if _matches(row, entry.get("match", {})):
            return True
    return False


def verify_session(session_dir: str | Path, *, write_report: bool = True,
                   expected_session_id: str | None = None) -> dict:
    session = Path(session_dir).resolve()
    checks: list[dict] = []
    session_id = expected_session_id or session.name

    shard_files = sorted(session.glob("runs.shard-*.jsonl"))
    manifests, provenance_errors = _load_provenance(session)
    for manifest in manifests:
        pid = manifest.get("provenance_id")
        if pid != assign_provenance_id(manifest):
            provenance_errors.append(f"provenance ID does not match payload: {pid}")
        if manifest.get("session_id") != session_id:
            provenance_errors.append(f"provenance session_id mismatch: {pid}")
        shard = manifest.get("shard")
        if shard is not None and (not isinstance(shard, list) or len(shard) != 2 or
                                  not all(isinstance(v, int) for v in shard) or
                                  not 0 <= shard[0] < shard[1]):
            provenance_errors.append(f"invalid shard identity: {pid}: {shard!r}")
    manifest_shards = {tuple(m["shard"]) for m in manifests if m.get("shard") is not None}
    for path in shard_files:
        match = re.search(r"\.shard-(\d+)-of-(\d+)\.jsonl$", path.name)
        if not match or (int(match.group(1)), int(match.group(2))) not in manifest_shards:
            provenance_errors.append(f"{path.name}: no matching shard provenance manifest")
    checks.append(_check("provenance_schema", bool(manifests) and not provenance_errors,
                         f"{len(manifests)} unique provenance manifest(s)", provenance_errors))

    merged_file = session / "runs.jsonl"
    canonical_file = merged_file if merged_file.exists() else None
    schema_errors: list[str] = []
    rows: list[dict] = []
    attempt_rows: list[dict] = []
    try:
        if canonical_file:
            rows = load_result_file(canonical_file)
        elif not shard_files:
            schema_errors.append("no canonical runs.jsonl or shard result files")
    except Exception as exc:
        schema_errors.append(f"{canonical_file.name}: {exc}")
    attempt_files = shard_files or ([merged_file] if merged_file.exists() else [])
    for path in attempt_files:
        try:
            attempt_rows.extend(load_result_file(path))
        except Exception as exc:
            schema_errors.append(f"{path.name}: {exc}")
    if shard_files:
        # runs.jsonl may preserve an earlier successful attempt when a repair shard
        # contains only a later failure. Include such canonical-only history once.
        raw_ids = {row.get("run_id") for row in attempt_rows}
        attempt_rows.extend(row for row in rows if row.get("run_id") not in raw_ids)
    checks.append(_check("result_schema", bool(rows) and bool(attempt_rows) and not schema_errors,
                         f"{len(rows)} canonical row(s), {len(attempt_rows)} raw attempt(s)",
                         schema_errors))

    merge_errors: list[str] = []
    if shard_files:
        if not merged_file.exists():
            merge_errors.append("sharded session has not been merged")
        else:
            try:
                # Mirror cmd_merge exactly: the existing canonical file is the lowest-
                # priority source, but an earlier success may legitimately survive a
                # later failed repair attempt (success outranks source recency).
                expected_merge = _merge_precedence(shard_files + [merged_file])
                actual_signature = [(logical_merge_key(r), r.get("run_id")) for r in rows]
                expected_signature = [(logical_merge_key(r), r.get("run_id"))
                                      for r in expected_merge]
                if actual_signature != expected_signature:
                    merge_errors.append("runs.jsonl is stale or differs from shard precedence")
            except Exception as exc:
                merge_errors.append(str(exc))
    checks.append(_check("merge_state", not merge_errors,
                         "canonical merge is current" if not merge_errors else "merge incomplete",
                         merge_errors))

    id_errors: list[str] = []
    try:
        assert_no_id_collisions(attempt_rows)
        run_ids = [r.get("run_id") for r in attempt_rows]
        logical_ids = [r.get("logical_cell_id") for r in rows]
        if len(run_ids) != len(set(run_ids)):
            id_errors.append("duplicate run_id in raw attempt history")
        if None in logical_ids or len(logical_ids) != len(set(logical_ids)):
            id_errors.append("missing or duplicate logical_cell_id in canonical rows")
    except Exception as exc:
        id_errors.append(str(exc))
    checks.append(_check("unique_ids", not id_errors, "canonical IDs are unique", id_errors))

    known_provenance = {m.get("provenance_id") for m in manifests}
    join_errors = [f"{r.get('run_id')}: unknown provenance_id {r.get('provenance_id')}"
                   for r in attempt_rows if r.get("provenance_id") not in known_provenance]
    checks.append(_check("provenance_joins", not join_errors,
                         f"{len(attempt_rows) - len(join_errors)}/{len(attempt_rows)} rows joined",
                         join_errors))

    artifact_errors = _artifact_checks(session, manifests, attempt_rows)
    checks.append(_check("artifact_checksums", not artifact_errors,
                         "all archived inputs match SHA-256", artifact_errors))

    # A session can retain provenance manifests from an invocation that produced no
    # result rows (for example, an interrupted preflight followed by a resume).  Keep
    # validating those manifests and their archived artifacts above, but reconstruct
    # the measured matrix from the manifests actually referenced by the attempt
    # history.  If attempts genuinely span different experiment inputs, _input_yaml
    # remains strict and reports the disagreement.
    referenced_provenance = {r.get("provenance_id") for r in attempt_rows}
    measured_manifests = [m for m in manifests
                          if m.get("provenance_id") in referenced_provenance]

    experiment: dict[str, Any] = {}
    datasets: dict[str, Any] = {}
    matrix_errors: list[str] = []
    try:
        experiment = _input_yaml(session, measured_manifests, "experiment")
        datasets = _input_yaml(session, measured_manifests, "datasets")
        expected, actual = _expected_matrix(experiment, datasets), _actual_matrix(rows)
        missing, extra = expected - actual, actual - expected
        if missing:
            matrix_errors.append(f"{sum(missing.values())} expected cell(s) missing")
            matrix_errors.extend("missing: " + "|".join(map(str, key))
                                 for key in missing.elements())
        if extra:
            matrix_errors.append(f"{sum(extra.values())} unexpected/duplicate cell(s)")
            matrix_errors.extend("unexpected: " + "|".join(map(str, key))
                                 for key in extra.elements())
        matrix_summary = f"{sum(actual.values())}/{sum(expected.values())} expected cells"
    except Exception as exc:
        matrix_errors.append(str(exc))
        matrix_summary = "expected matrix unavailable"
    checks.append(_check("matrix_coverage", not matrix_errors, matrix_summary, matrix_errors))

    dataset_join_errors: list[str] = []
    widths = {"f32": 4, "f64": 8, "i32": 4, "i64": 8,
              "u8": 1, "u16": 2, "u32": 4}
    for row in attempt_rows:
        try:
            dataset = datasets[row["dataset"]]
            field = dataset["fields"][row["field"]]
            dtype = dataset.get("dtype", "f32")
            dims = [int(value) for value in field["dims"]]
            dim_order = dataset.get("dim_order", "fast-to-slow")
            elements = 1
            for dim in dims:
                elements *= dim
            byte_count = elements * widths.get(dtype, 4)
            resolved = row.get("execution_context", {}).get("resolved_config", {})
            consumed = row.get("execution_context", {}).get("dataset", {}).get("bytes")
            if (resolved.get("dtype") != dtype or resolved.get("dims") != dims or
                    resolved.get("dim_order") != dim_order or consumed != byte_count):
                raise ValueError("execution context disagrees with archived dataset manifest")
            if row.get("status") == "ok" and (
                    row.get("dtype") != dtype or row.get("dims") != dims or
                    row.get("num_elements") != elements or row.get("original_bytes") != byte_count):
                raise ValueError("measurement metadata disagrees with archived dataset manifest")
        except Exception as exc:
            dataset_join_errors.append(f"{row.get('run_id')}: {exc}")
    checks.append(_check("dataset_joins", not dataset_join_errors,
                         f"{len(attempt_rows) - len(dataset_join_errors)}/{len(attempt_rows)} rows joined",
                         dataset_join_errors))

    try:
        failure_policy = _policy(experiment, "accepted_failures")
        exclusion_policy = _policy(experiment, "accepted_exclusions")
        timing_policy = _policy(experiment, "accepted_unreliable_timing")
        policy_errors = []
    except Exception as exc:
        failure_policy = exclusion_policy = timing_policy = []
        policy_errors = [str(exc)]
    failures = [r for r in rows if r.get("status") != "ok"]
    unaccepted_failures = [r.get("run_id") for r in failures
                           if not _accepted(r, failure_policy)]
    policy_errors += [f"unaccepted failure: {run_id}" for run_id in unaccepted_failures]
    checks.append(_check("failures", not policy_errors,
                         f"{len(failures) - len(unaccepted_failures)}/{len(failures)} failures accepted",
                         policy_errors))

    annotated = validity.annotate(rows)
    gated = [(r, code) for r in annotated for code in r["_exclusions"]
             if code in _GATING_EXCLUSIONS]
    unaccepted_exclusions = [(r.get("run_id"), code) for r, code in gated
                             if not _accepted(r, exclusion_policy, code)]
    checks.append(_check("validity_exclusions", not unaccepted_exclusions,
                         f"{len(gated) - len(unaccepted_exclusions)}/{len(gated)} gating exclusions accepted",
                         [f"{run_id}: {code}" for run_id, code in unaccepted_exclusions]))

    unreliable = [r for r in rows if r.get("status") == "ok" and
                  r.get("timing_reliable") is False]
    unaccepted_timing = [r.get("run_id") for r in unreliable
                         if not _accepted(r, timing_policy)]
    checks.append(_check("timing_reliability", not unaccepted_timing,
                         f"{len(unreliable) - len(unaccepted_timing)}/{len(unreliable)} unreliable rows accepted",
                         unaccepted_timing))

    eligibility_errors = []
    required_inputs = {"experiment", "datasets", "dataset_checksums", "site",
                       "resolved_datasets", "harness_patch"}
    for manifest in manifests:
        integrity = manifest.get("dataset_integrity", {})
        if not integrity.get("all_verified"):
            eligibility_errors.append(f"{manifest.get('provenance_id')}: datasets unverified")
        for dataset in integrity.get("datasets", []):
            expected, observed = dataset.get("expected_sha256"), dataset.get("observed_sha256")
            if (not dataset.get("verified") or expected != observed or
                    not isinstance(expected, str) or len(expected) != 64):
                eligibility_errors.append(
                    f"{manifest.get('provenance_id')}: invalid dataset integrity "
                    f"{dataset.get('dataset')}/{dataset.get('field')}")
        integrity_index = {(d.get("dataset"), d.get("field")): d.get("observed_sha256")
                           for d in integrity.get("datasets", [])}
        for row in attempt_rows:
            if row.get("provenance_id") != manifest.get("provenance_id"):
                continue
            if integrity_index.get((row.get("dataset"), row.get("field"))) != row.get("dataset_sha256"):
                eligibility_errors.append(
                    f"{manifest.get('provenance_id')}: row {row.get('run_id')} does not "
                    "join resolved dataset integrity")
        if not integrity.get("publication_grade"):
            eligibility_errors.append(f"{manifest.get('provenance_id')}: H3 provenance incomplete")
        missing_inputs = required_inputs - set(manifest.get("input_artifacts", {}))
        if missing_inputs:
            eligibility_errors.append(
                f"{manifest.get('provenance_id')}: missing inputs {sorted(missing_inputs)}")
        for label, tool in manifest.get("compressors", {}).items():
            if not tool.get("build_provenance_complete"):
                eligibility_errors.append(f"{manifest.get('provenance_id')}: {label} build incomplete")
        patch = manifest.get("input_artifacts", {}).get("harness_patch", {})
        recorded_patch = manifest.get("harness", {}).get("git", {}).get("patch_sha256")
        if recorded_patch is not None and patch.get("sha256") != recorded_patch:
            eligibility_errors.append(
                f"{manifest.get('provenance_id')}: harness patch digest does not join archive")
    checks.append(_check("h3_eligibility", bool(manifests) and not eligibility_errors,
                         "datasets and build provenance are publication-eligible",
                         eligibility_errors))

    passed = all(c["status"] == "pass" for c in checks)
    report = {
        "verification_schema_version": VERIFICATION_SCHEMA_VERSION,
        "session_id": session_id,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "publication_grade": passed,
        "summary": {"checks_passed": sum(c["status"] == "pass" for c in checks),
                    "checks_failed": sum(c["status"] == "fail" for c in checks),
                    "canonical_rows": len(rows)},
        "policy": experiment.get("verification", {}) if isinstance(experiment, dict) else {},
        "checks": checks,
    }
    validate_verification_report(report)
    encoded = json.dumps(report, indent=2, allow_nan=False) + "\n"
    if write_report:
        target = session / "verification.json"
        temporary = session / ".verification.json.tmp"
        temporary.write_text(encoded)
        temporary.replace(target)
    return report


def validate_verification_report(report: dict) -> None:
    if not isinstance(report, dict) or report.get("verification_schema_version") != 1:
        raise ValueError("unsupported verification report schema")
    if not isinstance(report.get("session_id"), str):
        raise ValueError("verification report needs session_id")
    if not isinstance(report.get("publication_grade"), bool):
        raise ValueError("verification report needs boolean publication_grade")
    checks = report.get("checks")
    if not isinstance(checks, list) or [c.get("name") for c in checks] != list(REQUIRED_CHECKS):
        raise ValueError("verification report has missing, duplicate, or reordered checks")
    if any(c.get("status") not in {"pass", "fail"} for c in checks):
        raise ValueError("verification report check has invalid status")
    # This is also the strict-JSON gate for arbitrary nested policy/details.
    json.dumps(report, allow_nan=False)


def load_verification_report(path: str | Path) -> dict:
    report = json.loads(Path(path).read_text())
    validate_verification_report(report)
    return report
