"""Build and verify offline Benchkit publication/AD-AE bundles."""
from __future__ import annotations

import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import stat
import subprocess
from typing import Iterable

import yaml

from .schema import load_result_file, load_session_text
from .store import sha256_file
from .verification import load_verification_report, verify_session


ARTIFACT_SCHEMA_VERSION = 1
ARTIFACT_PREFIX = "artifact-v1-"
REPO_ROOT = Path(__file__).resolve().parent.parent
GENERATOR_MODULE_SHA256 = sha256_file(Path(__file__))


def _git_commit() -> str | None:
    proc = subprocess.run(["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
                          capture_output=True, text=True)
    return proc.stdout.strip() if proc.returncode == 0 else None


def _copy(source: Path, root: Path, relative: Path) -> Path:
    if source.is_symlink():
        raise ValueError(f"refusing symlink in artifact input: {source}")
    target = root / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    return target


def _source_files(session: Path) -> Iterable[tuple[Path, Path, str]]:
    for pattern, role in (("runs*.jsonl", "results"),
                          ("provenance*.json", "provenance")):
        for path in sorted(session.glob(pattern)):
            yield path, Path(path.name), role
    verification = session / "verification.json"
    if verification.is_file():
        yield verification, Path("verification.json"), "verification"
    for directory, role in (("inputs", "input"), ("logs", "log")):
        base = session / directory
        if base.is_dir():
            for path in sorted(p for p in base.rglob("*") if p.is_file()):
                yield path, path.relative_to(session), role
    work = session / "work"
    if work.is_dir():
        for path in sorted(p for p in work.rglob("*") if p.is_file() and
                           p.suffix in {".log", ".json", ".toml", ".txt"}):
            yield path, path.relative_to(session), "diagnostic"


def _input_from_provenance(bundle: Path, name: str) -> Path:
    files = sorted(bundle.glob("provenance.provenance-v1-*.json"))
    if not files:
        files = sorted(bundle.glob("provenance*.json"))
    artifacts = []
    for path in files:
        manifest = load_session_text(path.read_text())
        if name in manifest.get("input_artifacts", {}):
            artifacts.append(manifest["input_artifacts"][name])
    hashes = {artifact["sha256"] for artifact in artifacts}
    if not artifacts or len(hashes) != 1:
        raise ValueError(f"cannot select unique archived {name}")
    return bundle / artifacts[0]["path"]


def _write_publication_inputs(bundle: Path, session_id: str) -> None:
    rows = load_result_file(bundle / "runs.jsonl")
    columns = [
        "logical_cell_id", "execution_id", "provenance_id", "compressor", "variant",
        "pipeline", "dataset", "field", "dtype", "dims", "error_mode", "error_bound",
        "status", "compressed_bytes", "original_bytes", "cr", "psnr", "max_abs_err",
        "eb_satisfied", "compress_throughput_gbs", "decompress_throughput_gbs",
        "timing_reliable",
    ]
    output = bundle / "publication" / "rows.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: (json.dumps(row.get(key), allow_nan=False)
                                   if isinstance(row.get(key), (dict, list))
                                   else row.get(key)) for key in columns})
    metadata = {
        "metadata_schema_version": 1,
        "generator": "benchkit artifact-v1",
        "generator_commit": _git_commit(),
        "generator_module_sha256": GENERATOR_MODULE_SHA256,
        "source_session_id": session_id,
        "source_runs_sha256": sha256_file(bundle / "runs.jsonl"),
        "rows": len(rows),
        "columns": columns,
    }
    (bundle / "publication" / "rows.metadata.json").write_text(
        json.dumps(metadata, indent=2, allow_nan=False) + "\n")


def _write_reproduction(bundle: Path, session_id: str) -> None:
    experiment_path = _input_from_provenance(bundle, "experiment")
    datasets_path = _input_from_provenance(bundle, "datasets")
    checksums_path = _input_from_provenance(bundle, "dataset_checksums")
    experiment = yaml.safe_load(experiment_path.read_text()) or {}
    rows = load_result_file(bundle / "runs.jsonl")
    row = next((candidate for candidate in rows if candidate.get("status") == "ok"), rows[0])
    source_run = next(
        (run for run in experiment.get("runs", [])
         if run.get("compressor") == row.get("compressor") and
         run.get("variant", run.get("compressor")) == row.get("variant")), None)
    if source_run is None:
        raise ValueError("canonical row does not join an archived run entry")
    run = dict(source_run)
    pipeline_artifact = row.get("rendered_pipeline_artifact")
    if pipeline_artifact:
        source_pipeline = bundle / pipeline_artifact["path"]
        _copy(source_pipeline, bundle, Path("reproduction/pipeline.toml"))
        run["pipeline"] = "reproduction/pipeline.toml"
    mode = row.get("error_mode")
    error = {"mode": mode}
    if mode not in {"lossless", "from_toml"}:
        error["bounds"] = [row.get("error_bound")]
    smoke = {
        "name": f"{session_id}-artifact-smoke",
        "description": "One-cell reproduction generated from an H4-verified session.",
        "datasets": [row["dataset"]],
        "fields": {row["dataset"]: [row["field"]]},
        "error": error,
        "repetitions": 2,
        "warmup_reps": 1,
        "lock_clocks": False,
        "retain_decompressed": False,
        "retain_compressed": False,
        "runs": [run],
    }
    reproduction = bundle / "reproduction"
    reproduction.mkdir(exist_ok=True)
    (reproduction / "smoke.yaml").write_text(yaml.safe_dump(smoke, sort_keys=False))
    _copy(datasets_path, bundle, Path("reproduction/datasets.yaml"))
    _copy(checksums_path, bundle, Path("reproduction/datasets.checksums.yaml"))
    script = reproduction / "run-smoke.sh"
    script.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        "ROOT=$(cd \"$(dirname \"${BASH_SOURCE[0]}\")/..\" && pwd)\n"
        ": \"${BENCHKIT_DATA_ROOT:?set BENCHKIT_DATA_ROOT to the SDRBench corpus}\"\n"
        "PYTHON=${PYTHON:-python3}\n"
        "OUT=${BENCHKIT_REPRO_RESULTS_ROOT:-/tmp/benchkit-artifact-reproduction}\n"
        "cd \"$ROOT\"\n"
        "\"$PYTHON\" -m benchkit run reproduction/smoke.yaml \\\n"
        "  --datasets reproduction/datasets.yaml --results-root \"$OUT\" \\\n"
        f"  --session-id {session_id}-artifact-smoke\n"
        f"\"$PYTHON\" -m benchkit verify \"$OUT/{session_id}-artifact-smoke\"\n")
    script.chmod(script.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    (reproduction / "README.md").write_text(
        "# Reproduction smoke\n\n"
        "Install Benchkit and the compressor declared in `smoke.yaml`, set "
        "`BENCHKIT_DATA_ROOT` to the locked SDRBench corpus and any adapter-specific "
        "environment variables (for example `FZGMOD_CLI`), then run:\n\n"
        "```bash\n./reproduction/run-smoke.sh\n```\n\n"
        "The command writes outside this bundle and finishes by applying the H4 gate.\n")


def _write_software(bundle: Path) -> None:
    for source, relative in ((REPO_ROOT / "LICENSE", Path("licenses/BENCHKIT-LICENSE")),
                             (REPO_ROOT / "pyproject.toml", Path("software/pyproject.toml")),
                             (REPO_ROOT / "requirements-dev.lock",
                              Path("software/requirements-dev.lock"))):
        if source.is_file():
            _copy(source, bundle, relative)
    for pattern in ("build*.sh", "env-*.sh"):
        for source in sorted((REPO_ROOT / "scripts").glob(pattern)):
            _copy(source, bundle, Path("software/scripts") / source.name)
    notice = bundle / "licenses" / "NOTICE.md"
    notice.parent.mkdir(parents=True, exist_ok=True)
    notice.write_text(
        "# External-data and tool notice\n\n"
        "This bundle contains no SDRBench arrays or third-party compressor binaries. "
        "Those inputs must be obtained under their upstream terms. The archived dataset "
        "manifest and checksum lock identify the exact expected bytes. Third-party tool "
        "versions, source commits, build flags and patch identities are recorded in the "
        "provenance manifests; their upstream licenses apply.\n")


def _inventory(root: Path) -> list[dict]:
    files = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"refusing symlink in artifact: {path}")
        if not path.is_file() or path.name == "artifact.json":
            continue
        relative = path.relative_to(root).as_posix()
        role = relative.split("/", 1)[0] if "/" in relative else "session"
        files.append({"path": relative, "role": role, "size": path.stat().st_size,
                      "sha256": sha256_file(path)})
    return files


def build_artifact(session_dir: str | Path, output_dir: str | Path,
                   includes: Iterable[str | Path] = ()) -> dict:
    session, output = Path(session_dir).resolve(), Path(output_dir).resolve()
    if not session.is_dir():
        raise ValueError(f"no such session: {session}")
    if output.exists():
        raise ValueError(f"artifact output already exists: {output}")
    report = verify_session(session)
    if not report["publication_grade"]:
        raise ValueError("source session is not H4 publication-grade")
    output.mkdir(parents=True)
    try:
        for source, relative, _role in _source_files(session):
            _copy(source, output, relative)
        _write_publication_inputs(output, session.name)
        _write_reproduction(output, session.name)
        _write_software(output)
        for include in includes:
            source = Path(include).resolve()
            if not source.is_file():
                raise ValueError(f"included publication file is not a file: {source}")
            target = _copy(source, output, Path("publication/included") / source.name)
            metadata = {"metadata_schema_version": 1, "generator": "benchkit artifact-v1",
                        "generator_commit": _git_commit(), "source_session_id": session.name,
                        "generator_module_sha256": GENERATOR_MODULE_SHA256,
                        "included_source_name": source.name, "sha256": sha256_file(target)}
            target.with_name(target.name + ".metadata.json").write_text(
                json.dumps(metadata, indent=2, allow_nan=False) + "\n")
        files = _inventory(output)
        created_at = datetime.now(timezone.utc).isoformat()
        core = {"artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
                "source_session_id": session.name, "generator": "benchkit artifact-v1",
                "generator_commit": _git_commit(),
                "generator_module_sha256": GENERATOR_MODULE_SHA256,
                "created_at": created_at, "files": files}
        encoded_core = json.dumps(core, sort_keys=True, separators=(",", ":"),
                                  allow_nan=False).encode()
        manifest = {**core, "artifact_id": ARTIFACT_PREFIX +
                    hashlib.sha256(encoded_core).hexdigest()}
        (output / "artifact.json").write_text(json.dumps(manifest, indent=2,
                                                          allow_nan=False) + "\n")
        verify_artifact(output)
        return manifest
    except Exception:
        # The output did not exist before this operation, so cleanup cannot erase user data.
        shutil.rmtree(output)
        raise


def verify_artifact(bundle_dir: str | Path) -> dict:
    root = Path(bundle_dir).resolve()
    manifest_path = root / "artifact.json"
    if not manifest_path.is_file():
        raise ValueError("artifact.json is missing")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("artifact_schema_version") != ARTIFACT_SCHEMA_VERSION:
        raise ValueError("unsupported artifact schema")
    core_keys = ("artifact_schema_version", "source_session_id", "generator",
                 "generator_commit", "generator_module_sha256", "created_at", "files")
    try:
        core = {key: manifest[key] for key in core_keys}
    except KeyError as exc:
        raise ValueError(f"artifact manifest is missing {exc.args[0]}") from exc
    expected_id = ARTIFACT_PREFIX + hashlib.sha256(
        json.dumps(core, sort_keys=True, separators=(",", ":"),
                   allow_nan=False).encode()).hexdigest()
    if manifest.get("artifact_id") != expected_id:
        raise ValueError("artifact_id does not match manifest payload")
    listed = set()
    for entry in manifest.get("files", []):
        if not isinstance(entry, dict) or set(entry) != {"path", "role", "size", "sha256"}:
            raise ValueError("invalid artifact inventory entry")
        if not isinstance(entry["path"], str) or not isinstance(entry["role"], str) or \
                not isinstance(entry["size"], int) or not isinstance(entry["sha256"], str):
            raise ValueError("invalid artifact inventory entry types")
        relative = Path(entry["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"unsafe artifact path: {entry['path']}")
        path = root / relative
        if path.is_symlink() or not path.is_file() or path.stat().st_size != entry["size"] or \
                sha256_file(path) != entry["sha256"]:
            raise ValueError(f"artifact file mismatch: {entry['path']}")
        if entry["path"] in listed:
            raise ValueError(f"duplicate artifact path: {entry['path']}")
        listed.add(entry["path"])
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ValueError(f"artifact contains symlink: {path.relative_to(root)}")
    actual = {path.relative_to(root).as_posix()
              for path in root.rglob("*") if path.is_file() and path.name != "artifact.json"}
    if actual != listed:
        raise ValueError(f"artifact inventory mismatch: missing={sorted(listed-actual)}, "
                         f"unexpected={sorted(actual-listed)}")
    stored_verification = load_verification_report(root / "verification.json")
    if not stored_verification["publication_grade"]:
        raise ValueError("bundled H4 report is not publication-grade")
    live = verify_session(root, write_report=False,
                          expected_session_id=manifest["source_session_id"])
    if not live["publication_grade"]:
        failed = [c["name"] for c in live["checks"] if c["status"] == "fail"]
        raise ValueError(f"bundled session fails offline H4 verification: {failed}")
    return manifest
