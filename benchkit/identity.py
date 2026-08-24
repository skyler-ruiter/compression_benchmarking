"""Canonical scientific-cell and execution identities.

``logical_cell_id`` answers "which scientific measurement target is this?" and is
therefore the key used to supersede attempts.  ``execution_id`` answers "is this exact
resolved execution already complete?" and is therefore the resume key.  Keeping the
canonical payloads beside their hashes makes collisions and accidental omissions
mechanically detectable instead of trusting an opaque digest.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any


IDENTITY_SCHEMA_VERSION = 1
LOGICAL_ID_PREFIX = "logical-v1-"
EXECUTION_ID_PREFIX = "execution-v1-"


class IdentityError(ValueError):
    """An identity payload is incomplete, non-canonical, or contradicts its ID."""


def canonical_json(payload: dict) -> str:
    """Return the sole byte representation hashed by the identity contracts."""
    try:
        return json.dumps(payload, sort_keys=True, separators=(",", ":"),
                          ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise IdentityError(f"identity payload is not strict JSON: {exc}") from exc


def _identifier(prefix: str, payload: dict) -> str:
    digest = hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()
    return prefix + digest


def logical_cell_id(payload: dict) -> str:
    return _identifier(LOGICAL_ID_PREFIX, payload)


def execution_id(context: dict) -> str:
    return _identifier(EXECUTION_ID_PREFIX, context)


def normalize_pipeline_ref(pipeline: str, repo_root: Path | None = None) -> str:
    """Normalize repo-owned paths without erasing meaningful pipeline parameters."""
    text = str(pipeline).strip()
    path = Path(text)
    if repo_root is not None and path.is_absolute():
        try:
            return path.resolve().relative_to(repo_root.resolve()).as_posix()
        except ValueError:
            pass
    if path.is_absolute() and "configs" in path.parts:
        # Historical rows sometimes captured a machine-specific absolute path to a
        # repo-owned preset. Recover the portable suffix without rewriting arbitrary
        # external TOMLs that happen to live outside the repository.
        index = path.parts.index("configs")
        return Path(*path.parts[index:]).as_posix()
    return path.as_posix() if path.suffix == ".toml" else text


def make_logical_cell(*, compressor: str, variant: str, pipeline: str,
                      dataset: str, field: str, error_mode: str,
                      error_bound: float | None) -> dict:
    """Canonical payload for one named experimental arm and scientific cell.

    Resolved files, binaries, graph mode, and machine state intentionally do not live
    here: changing those creates a new execution of the same cell, which merge may
    explicitly supersede.  An ablation arm must have its own ``variant``.
    """
    eb = None if error_bound is None else float(error_bound)
    if eb is not None and not math.isfinite(eb):
        raise IdentityError("logical cell error_bound must be finite or null")
    return {
        "identity_schema_version": IDENTITY_SCHEMA_VERSION,
        "compressor": str(compressor),
        "variant": str(variant),
        "pipeline": str(pipeline),
        "dataset": str(dataset),
        "field": str(field),
        "error_mode": str(error_mode),
        "error_bound": eb,
    }


def sha256_prefix(path: Path, size: int, chunk: int = 1 << 20) -> str:
    """Hash exactly the bytes the harness treats as the input field."""
    if size < 0:
        raise ValueError("size must be non-negative")
    digest = hashlib.sha256()
    remaining = size
    with open(path, "rb") as fh:
        while remaining:
            block = fh.read(min(chunk, remaining))
            if not block:
                raise ValueError(f"{path}: ended before the declared {size} input bytes")
            digest.update(block)
            remaining -= len(block)
    return digest.hexdigest()


def _sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def tool_identity(provenance: dict) -> dict:
    """Reduce adapter provenance to stable declarations plus executable digests.

    H3 will make source/build provenance exhaustive.  H2 already makes resume safe
    against the executable paths exposed by adapters: files are content-hashed and
    executable files below adapter bin directories are hashed as a sorted manifest.
    Human explanatory ``*_note`` text is excluded because editing documentation must
    not invalidate measurements.
    """
    declared = {k: v for k, v in provenance.items() if not k.endswith("_note")}
    artifacts: dict[str, Any] = {}
    for key, value in sorted(provenance.items()):
        if not (key == "cli_path" or key.endswith("_cli_path")):
            continue
        if not isinstance(value, str) or not value:
            continue
        path = Path(value).expanduser()
        if not path.exists():
            artifacts[key] = {"path": str(path), "available": False}
        elif path.is_file():
            resolved = path.resolve()
            artifacts[key] = {
                "path": str(resolved), "size": resolved.stat().st_size,
                "sha256": _sha256_file(resolved),
            }
        elif path.is_dir():
            files = []
            for candidate in sorted(path.rglob("*")):
                if candidate.is_file() and os.access(candidate, os.X_OK):
                    files.append({
                        "path": candidate.relative_to(path).as_posix(),
                        "size": candidate.stat().st_size,
                        "sha256": _sha256_file(candidate),
                    })
            artifacts[key] = {"path": str(path.resolve()), "executables": files}
    return {"declared": declared, "artifacts": artifacts}


def make_execution_context(*, logical_id: str, run_parameters: dict,
                           resolved_config: dict, dataset: dict,
                           tool: dict, harness: dict, environment: dict) -> dict:
    return {
        "identity_schema_version": IDENTITY_SCHEMA_VERSION,
        "logical_cell_id": logical_id,
        "run_parameters": run_parameters,
        "resolved_config": resolved_config,
        "dataset": dataset,
        "tool": tool,
        "harness": harness,
        "environment": environment,
    }


def validate_identity(logical_payload: dict, logical_id_value: str,
                      execution_context: dict, execution_id_value: str) -> None:
    expected_logical = logical_cell_id(logical_payload)
    if logical_id_value != expected_logical:
        raise IdentityError(
            f"logical_cell_id does not match payload: {logical_id_value} != {expected_logical}")
    if execution_context.get("logical_cell_id") != logical_id_value:
        raise IdentityError("execution_context.logical_cell_id does not join logical_cell_id")
    expected_execution = execution_id(execution_context)
    if execution_id_value != expected_execution:
        raise IdentityError(
            f"execution_id does not match context: {execution_id_value} != {expected_execution}")


def logical_cell_from_row(row: dict) -> dict | None:
    """Return an explicit payload or reconstruct one from a full legacy row."""
    explicit = row.get("logical_cell")
    if isinstance(explicit, dict):
        return explicit
    required = ("compressor", "variant", "pipeline", "dataset", "field", "error_mode")
    if all(row.get(key) is not None for key in required):
        return make_logical_cell(
            compressor=row["compressor"], variant=row["variant"],
            pipeline=normalize_pipeline_ref(row["pipeline"]),
            dataset=row["dataset"], field=row["field"], error_mode=row["error_mode"],
            error_bound=row.get("error_bound"),
        )
    # Early native failure rows recorded only the old seven-part cell_key plus a
    # subset of top-level fields. Parse it only at the exact historical arity; a
    # malformed/foreign key falls back to run_id rather than being guessed.
    legacy = row.get("cell_key")
    if isinstance(legacy, str):
        parts = legacy.split("|")
        if len(parts) == 7 and all(parts[:6]):
            compressor, variant, pipeline, dataset, field, mode, eb_text = parts
            try:
                eb = None if eb_text == "toml" else float(eb_text)
            except ValueError:
                return None
            return make_logical_cell(
                compressor=compressor, variant=variant,
                pipeline=normalize_pipeline_ref(pipeline), dataset=dataset, field=field,
                error_mode=mode, error_bound=eb)
    return None


def logical_cell_id_from_row(row: dict) -> str | None:
    explicit = row.get("logical_cell_id")
    if isinstance(explicit, str):
        return explicit
    payload = logical_cell_from_row(row)
    return logical_cell_id(payload) if payload is not None else None


def logical_merge_key(row: dict) -> str:
    """Transition key: canonical logical ID, then legacy aliases as a last resort."""
    return (logical_cell_id_from_row(row) or row.get("cell_key") or
            f"run:{row.get('run_id')}")


def assert_no_id_collisions(rows: list[dict]) -> None:
    """Reject one ID being associated with multiple canonical payloads."""
    seen: dict[tuple[str, str], str] = {}
    for row in rows:
        for id_key, payload_key in (("logical_cell_id", "logical_cell"),
                                    ("execution_id", "execution_context")):
            identity = row.get(id_key)
            payload = row.get(payload_key)
            if not isinstance(identity, str) or not isinstance(payload, dict):
                continue
            canonical = canonical_json(payload)
            marker = (id_key, identity)
            previous = seen.get(marker)
            if previous is not None and previous != canonical:
                raise IdentityError(f"{id_key} collision for {identity}")
            seen[marker] = canonical
