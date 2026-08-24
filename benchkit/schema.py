"""Versioned on-disk schemas for Benchkit sessions and result rows.

Python's ``json`` module accepts and emits NaN/Infinity by default even though those
tokens are not JSON.  Benchkit uses infinities internally (most importantly PSNR for
an exact reconstruction), so v1 stores non-finite values as JSON ``null`` plus an
explicit JSON-pointer -> kind map.  Readers restore the Python float for existing
analysis code; the bytes on disk remain strict, portable JSON.

Unversioned historical files are legacy schema v0.  They remain readable and are never
rewritten merely by loading them.  Passing a loaded v0 row to a writer produces a new
v1 document carrying ``source_result_schema_version: 0``.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from .identity import (IDENTITY_SCHEMA_VERSION, IdentityError, normalize_pipeline_ref,
                       validate_identity)


SESSION_SCHEMA_VERSION = 1
RESULT_SCHEMA_VERSION = 1
LEGACY_SCHEMA_VERSION = 0

NONFINITE_KINDS = {
    "nan": math.nan,
    "positive_infinity": math.inf,
    "negative_infinity": -math.inf,
}
PSNR_KINDS = {"finite", "exact", "undefined"}
RESULT_KINDS = {"measurement", "failure", "reconstructed_measurement"}
SESSION_KINDS = {"native", "reconstructed"}


class SchemaError(ValueError):
    """A versioned Benchkit document violates its on-disk contract."""


# Core required/optional contracts.  Additive tool-specific nested content is allowed;
# these rules protect the fields on which storage, merge, reporting, and validity rely.
SESSION_REQUIRED = {
    "session_schema_version": (int, False),
    "session_kind": (str, False),
    "session_id": (str, False),
}
SESSION_NATIVE_REQUIRED = {
    "timestamp": (str, False),
    "shard": (list, True),
    "gpu": (dict, False),
    "host": (dict, False),
    "scheduler": (dict, False),
    "software": (dict, False),
    "harness": (dict, False),
    "compressors": (dict, False),
    "nvidia_smi": (str, True),
}
SESSION_OPTIONAL = {
    "nonfinite_values": (dict, False),
    "reconstructed": (bool, False),
    "reconstruction_source": (str, True),
    "reconstruction_note": (str, True),
    "experiment_config": (str, True),
    "original_results_path": (str, True),
    "node_job_label": (str, True),
    "provenance_schema_version": (int, False),
    "provenance_id": (str, False),
    "input_artifacts": (dict, False),
    "dataset_integrity": (dict, False),
}

RESULT_REQUIRED = {
    "result_schema_version": (int, False),
    "record_kind": (str, False),
    "run_id": (str, False),
    "session_id": (str, False),
    "compressor": (str, False),
    "variant": (str, False),
    "dataset": (str, False),
    "field": (str, False),
    "status": (str, False),
}
RESULT_NATIVE_ID_REQUIRED = {
    "pipeline": (str, False),
}
RESULT_MEASUREMENT_REQUIRED = {
    "timestamp": (str, False),
    "dtype": (str, False),
    "dims": (list, False),
    "num_elements": (int, False),
    "original_bytes": (int, False),
    "error_mode": (str, False),
    "error_bound": ((int, float), True),
    "compressed_bytes": (int, False),
    "cr": ((int, float), True),
    "psnr": ((int, float), True),
    "psnr_kind": (str, False),
    "max_abs_err": ((int, float), False),
    "eb_satisfied": (bool, False),
    "timing_reliable": (bool, False),
}
RESULT_FAILURE_REQUIRED = {
    "error_message": (str, False),
    "error_type": (str, False),
    "fail_phase": (str, False),
}
RESULT_RECONSTRUCTED_REQUIRED = {
    "error_bound": ((int, float), True),
    "cr": ((int, float), True),
    "psnr": ((int, float), True),
    "psnr_kind": (str, False),
    "eb_satisfied": (bool, False),
    "timing_reliable": (bool, False),
    "reconstructed": (bool, False),
    "reconstruction_source": (str, False),
}
RESULT_OPTIONAL = {
    "source_result_schema_version": (int, False),
    "nonfinite_values": (dict, False),
    "error_message": (str, True),
    "error_bound": ((int, float), True),
    "pipeline_ref": (str, True),
    "pipeline_sha256": (str, True),
    "native_mode": (str, True),
    "rel_basis": (str, True),
    "native_psnr": ((int, float), True),
    "graph_active": (bool, True),
    "graph_reason": (str, True),
    "legacy_identity_synthesized": (bool, False),
    "identity_schema_version": (int, False),
    "logical_cell_id": (str, False),
    "execution_id": (str, False),
    "logical_cell": (dict, False),
    "execution_context": (dict, False),
    "dataset_sha256": (str, False),
    "cell_key": (str, False),
    "provenance_id": (str, False),
    "rendered_pipeline_artifact": (dict, True),
}

RESULT_IDENTITY_FIELDS = {
    "identity_schema_version": (int, False),
    "logical_cell_id": (str, False),
    "execution_id": (str, False),
    "logical_cell": (dict, False),
    "execution_context": (dict, False),
    "dataset_sha256": (str, False),
}


def _type_name(expected) -> str:
    if isinstance(expected, tuple):
        return " or ".join(t.__name__ for t in expected)
    return expected.__name__


def _check_fields(document: dict, rules: dict, label: str) -> None:
    for key, (expected, nullable) in rules.items():
        if key not in document:
            raise SchemaError(f"{label}: missing required field '{key}'")
        value = document[key]
        if value is None and nullable:
            continue
        # bool is an int subclass, but a boolean is never a valid byte count/version.
        numeric_types = expected if isinstance(expected, tuple) else (expected,)
        if isinstance(value, bool) and any(t in (int, float) for t in numeric_types):
            ok = False
        else:
            ok = isinstance(value, expected)
        if not ok:
            null_note = " or null" if nullable else ""
            raise SchemaError(
                f"{label}.{key}: expected {_type_name(expected)}{null_note}, "
                f"got {type(value).__name__}")


def _check_optional_fields(document: dict, rules: dict, label: str) -> None:
    _check_fields({k: document[k] for k in rules if k in document},
                  {k: rule for k, rule in rules.items() if k in document}, label)


def _valid_prefixed_sha(value: Any, prefix: str) -> bool:
    if not isinstance(value, str) or not value.startswith(prefix):
        return False
    digest = value[len(prefix):]
    return len(digest) == 64 and all(c in "0123456789abcdef" for c in digest)


def _pointer_escape(token: str) -> str:
    return token.replace("~", "~0").replace("/", "~1")


def _pointer_unescape(token: str) -> str:
    return token.replace("~1", "/").replace("~0", "~")


def _encode_nonfinite(value: Any, path: str, found: dict[str, str]) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        kind = ("nan" if math.isnan(value) else
                "positive_infinity" if value > 0 else "negative_infinity")
        found[path or "/"] = kind
        return None
    if isinstance(value, dict):
        return {k: _encode_nonfinite(v, f"{path}/{_pointer_escape(str(k))}", found)
                for k, v in value.items()}
    if isinstance(value, list):
        return [_encode_nonfinite(v, f"{path}/{i}", found)
                for i, v in enumerate(value)]
    return value


def _restore_pointer(document: Any, pointer: str, value: float) -> None:
    tokens = [_pointer_unescape(t) for t in pointer.split("/")[1:]]
    if not tokens:
        raise SchemaError("nonfinite_values cannot target the document root")
    target = document
    for token in tokens[:-1]:
        try:
            target = target[int(token)] if isinstance(target, list) else target[token]
        except (KeyError, IndexError, ValueError, TypeError) as exc:
            raise SchemaError(f"nonfinite_values: invalid pointer '{pointer}'") from exc
    final = tokens[-1]
    try:
        current = target[int(final)] if isinstance(target, list) else target[final]
    except (KeyError, IndexError, ValueError, TypeError) as exc:
        raise SchemaError(f"nonfinite_values: invalid pointer '{pointer}'") from exc
    if current is not None:
        raise SchemaError(f"nonfinite_values: target '{pointer}' must be null on disk")
    if isinstance(target, list):
        target[int(final)] = value
    else:
        target[final] = value


def _assert_strict_json_value(value: Any, path: str = "$") -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise SchemaError(f"{path}: non-finite float was not encoded")
    if isinstance(value, dict):
        for key, child in value.items():
            if not isinstance(key, str):
                raise SchemaError(f"{path}: JSON object keys must be strings")
            _assert_strict_json_value(child, f"{path}.{key}")
    elif isinstance(value, list):
        for i, child in enumerate(value):
            _assert_strict_json_value(child, f"{path}[{i}]")
    elif value is not None and not isinstance(value, (str, int, float, bool)):
        raise SchemaError(f"{path}: {type(value).__name__} is not a JSON value")


def _decode_nonfinite(document: dict) -> dict:
    decoded = copy.deepcopy(document)
    mapping = decoded.pop("nonfinite_values", {})
    if not isinstance(mapping, dict):
        raise SchemaError("nonfinite_values must be an object")
    for pointer, kind in mapping.items():
        if kind not in NONFINITE_KINDS:
            raise SchemaError(f"nonfinite_values.{pointer}: unknown kind '{kind}'")
        _restore_pointer(decoded, pointer, NONFINITE_KINDS[kind])
    return decoded


def _psnr_kind(row: dict) -> str:
    value = row.get("psnr")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        value = float(value)
        if math.isfinite(value):
            return "finite"
        if value > 0 and row.get("max_abs_err") == 0:
            return "exact"
    return "undefined"


def validate_session_document(document: dict, allow_legacy: bool = True) -> int:
    if not isinstance(document, dict):
        raise SchemaError("session: expected an object")
    version = document.get("session_schema_version")
    if version is None:
        if allow_legacy:
            return LEGACY_SCHEMA_VERSION
        raise SchemaError("session: missing session_schema_version")
    if version != SESSION_SCHEMA_VERSION:
        raise SchemaError(f"session: unsupported schema version {version}")
    _check_fields(document, SESSION_REQUIRED, "session")
    if document["session_kind"] not in SESSION_KINDS:
        raise SchemaError(f"session.session_kind: unknown value {document['session_kind']!r}")
    if document["session_kind"] == "native":
        _check_fields(document, SESSION_NATIVE_REQUIRED, "session")
    _check_optional_fields(document, SESSION_OPTIONAL, "session")
    if "provenance_id" in document:
        value = document["provenance_id"]
        if document.get("provenance_schema_version") != 1 or not _valid_prefixed_sha(
                value, "provenance-v1-"):
            raise SchemaError("session.provenance_id is not a valid provenance-v1 ID")
    _assert_strict_json_value(document)
    return version


def validate_result_document(document: dict, allow_legacy: bool = True) -> int:
    if not isinstance(document, dict):
        raise SchemaError("result: expected an object")
    version = document.get("result_schema_version")
    if version is None:
        if allow_legacy:
            return LEGACY_SCHEMA_VERSION
        raise SchemaError("result: missing result_schema_version")
    if version != RESULT_SCHEMA_VERSION:
        raise SchemaError(f"result: unsupported schema version {version}")
    _check_fields(document, RESULT_REQUIRED, "result")
    kind = document["record_kind"]
    if kind not in RESULT_KINDS:
        raise SchemaError(f"result.record_kind: unknown value {kind!r}")
    status = document["status"]
    if kind == "failure":
        if status == "ok":
            raise SchemaError("result: failure record cannot have status 'ok'")
        _check_fields(document, RESULT_NATIVE_ID_REQUIRED, "result")
        _check_fields(document, RESULT_FAILURE_REQUIRED, "result")
    elif kind == "measurement":
        if status != "ok":
            raise SchemaError("result: measurement record must have status 'ok'")
        _check_fields(document, RESULT_NATIVE_ID_REQUIRED, "result")
        _check_fields(document, RESULT_MEASUREMENT_REQUIRED, "result")
    else:
        if status != "ok":
            raise SchemaError("result: reconstructed measurement must have status 'ok'")
        _check_fields(document, RESULT_RECONSTRUCTED_REQUIRED, "result")
    _check_optional_fields(document, RESULT_OPTIONAL, "result")
    if "provenance_id" in document and not _valid_prefixed_sha(
            document["provenance_id"], "provenance-v1-"):
        raise SchemaError("result.provenance_id is not a valid provenance-v1 ID")
    identity_present = any(key in document for key in RESULT_IDENTITY_FIELDS)
    if identity_present:
        _check_fields(document, RESULT_IDENTITY_FIELDS, "result")
        if document["identity_schema_version"] != IDENTITY_SCHEMA_VERSION:
            raise SchemaError(
                f"result: unsupported identity schema version "
                f"{document['identity_schema_version']}")
        dataset = document["execution_context"].get("dataset")
        if not isinstance(dataset, dict) or dataset.get("sha256") != document["dataset_sha256"]:
            raise SchemaError("result.dataset_sha256 does not join execution_context.dataset")
        digest = document["dataset_sha256"]
        if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise SchemaError("result.dataset_sha256 must be a lowercase SHA-256 digest")
        try:
            validate_identity(document["logical_cell"], document["logical_cell_id"],
                              document["execution_context"], document["execution_id"])
        except IdentityError as exc:
            raise SchemaError(f"result identity: {exc}") from exc
        logical = document["logical_cell"]
        for key in ("compressor", "variant", "dataset", "field", "error_bound"):
            if logical.get(key) != document.get(key):
                raise SchemaError(f"result.{key} does not join logical_cell.{key}")
        if logical.get("pipeline") != normalize_pipeline_ref(document.get("pipeline", "")):
            raise SchemaError("result.pipeline does not join logical_cell.pipeline")
        if document.get("error_mode") is not None and \
                logical.get("error_mode") != document["error_mode"]:
            raise SchemaError("result.error_mode does not join logical_cell.error_mode")
        if kind == "measurement" and \
                document["execution_context"]["dataset"].get("bytes") != document["original_bytes"]:
            raise SchemaError(
                "result.original_bytes does not join execution_context.dataset.bytes")
    elif kind in {"measurement", "failure"} and not document.get("cell_key"):
        raise SchemaError(
            "result: native record needs H2 identity fields or legacy cell_key")
    if kind != "failure":
        psnr_kind = document["psnr_kind"]
        if psnr_kind not in PSNR_KINDS:
            raise SchemaError(f"result.psnr_kind: unknown value {psnr_kind!r}")
        if psnr_kind == "finite" and document["psnr"] is None:
            raise SchemaError("result.psnr: finite PSNR cannot be null")
        if psnr_kind != "finite" and document["psnr"] is not None:
            raise SchemaError("result.psnr: non-finite PSNR must be null on disk")
        nonfinite = document.get("nonfinite_values", {})
        if psnr_kind == "exact" and nonfinite.get("/psnr") != "positive_infinity":
            raise SchemaError(
                "result.psnr: exact PSNR requires positive_infinity metadata")
    _assert_strict_json_value(document)
    return version


def prepare_session_for_write(manifest: dict) -> dict:
    out = copy.deepcopy(manifest)
    source = out.get("session_schema_version")
    if source not in (None, SESSION_SCHEMA_VERSION, LEGACY_SCHEMA_VERSION):
        raise SchemaError(f"session: cannot write unsupported source version {source}")
    out["session_schema_version"] = SESSION_SCHEMA_VERSION
    out.setdefault("session_kind", "reconstructed" if out.get("reconstructed") else "native")
    found: dict[str, str] = {}
    out.pop("nonfinite_values", None)
    out = _encode_nonfinite(out, "", found)
    if found:
        out["nonfinite_values"] = found
    validate_session_document(out, allow_legacy=False)
    return out


def prepare_result_for_write(row: dict) -> dict:
    out = copy.deepcopy(row)
    source = out.get("result_schema_version")
    if source not in (None, RESULT_SCHEMA_VERSION, LEGACY_SCHEMA_VERSION):
        raise SchemaError(f"result: cannot write unsupported source version {source}")
    if source == LEGACY_SCHEMA_VERSION:
        out["source_result_schema_version"] = LEGACY_SCHEMA_VERSION
    out["result_schema_version"] = RESULT_SCHEMA_VERSION
    if out.get("reconstructed"):
        out.setdefault("record_kind", "reconstructed_measurement")
    else:
        out.setdefault("record_kind", "measurement" if out.get("status") == "ok" else "failure")
    if out["record_kind"] != "failure":
        out["psnr_kind"] = _psnr_kind(out)
    found: dict[str, str] = {}
    out.pop("nonfinite_values", None)
    out = _encode_nonfinite(out, "", found)
    if found:
        out["nonfinite_values"] = found
    validate_result_document(out, allow_legacy=False)
    return out


def dumps_session(manifest: dict, *, indent: int | None = None) -> str:
    return json.dumps(prepare_session_for_write(manifest), indent=indent,
                      allow_nan=False)


def dumps_result(row: dict) -> str:
    return json.dumps(prepare_result_for_write(row), allow_nan=False)


def load_session_text(text: str) -> dict:
    raw = json.loads(text)
    version = validate_session_document(raw)
    if version == LEGACY_SCHEMA_VERSION:
        return {**raw, "session_schema_version": LEGACY_SCHEMA_VERSION}
    return _decode_nonfinite(raw)


def load_result_line(line: str) -> dict:
    raw = json.loads(line)
    version = validate_result_document(raw)
    if version == LEGACY_SCHEMA_VERSION:
        return {**raw, "result_schema_version": LEGACY_SCHEMA_VERSION}
    return _decode_nonfinite(raw)


def load_result_file(path: str | Path) -> list[dict]:
    """Load results and supply deterministic IDs for early legacy exports.

    Some reconstructed v0 baselines predate ``session_id`` and ``run_id``.  A
    content digest supplies a stable session namespace; the source line then
    distinguishes duplicate rows.  A marker keeps these compatibility IDs from
    being mistaken for identities captured during the original execution.
    """
    text = Path(path).read_text()
    legacy_session_id = f"legacy-file-{hashlib.sha256(text.encode()).hexdigest()[:16]}"
    rows: list[dict] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        row = load_result_line(line)
        if row["result_schema_version"] == LEGACY_SCHEMA_VERSION:
            synthesized = False
            if not row.get("session_id"):
                row["session_id"] = legacy_session_id
                synthesized = True
            if not row.get("run_id"):
                material = f"{legacy_session_id}\0{line_number}\0{line}".encode()
                row["run_id"] = f"legacy-row-{hashlib.sha256(material).hexdigest()[:16]}"
                synthesized = True
            if synthesized:
                row["legacy_identity_synthesized"] = True
        rows.append(row)
    return rows
