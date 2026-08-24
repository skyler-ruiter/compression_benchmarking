"""Append-only results store, shard-aware for HPC job arrays.

One session = one directory under the results root. Within it:
  - runs.jsonl                       single-process runs
  - runs.shard-{k}-of-{N}.jsonl      one file per job-array task (no append contention)
  - provenance[.shard-{k}-of-{N}].json   per-task env manifest (each task may be a
                                          different node/GPU, so capture per shard)
  - logs/ , work/                    raw tool output + intermediate artifacts

Rows are append-only so re-runs never clobber; resumability is by scanning all run files
for completed ``execution_id`` values.  Legacy ``cell_key`` remains readable but is not
safe enough to suppress a new H2 execution.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .schema import dumps_result, dumps_session, load_result_file


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def _shard_suffix(shard: tuple[int, int] | None) -> str:
    return "" if shard is None else f".shard-{shard[0]}-of-{shard[1]}"


class ResultStore:
    def __init__(self, results_root: Path, session_id: str,
                 shard: tuple[int, int] | None = None):
        self.dir = Path(results_root) / session_id
        self.dir.mkdir(parents=True, exist_ok=True)
        (self.dir / "logs").mkdir(exist_ok=True)
        (self.dir / "work").mkdir(exist_ok=True)
        (self.dir / "inputs" / "rendered-pipelines").mkdir(parents=True, exist_ok=True)
        self.session_id = session_id
        self.shard = shard
        self.runs_path = self.dir / f"runs{_shard_suffix(shard)}.jsonl"

    def write_provenance(self, manifest: dict) -> None:
        path = self.dir / f"provenance{_shard_suffix(self.shard)}.json"
        encoded = dumps_session(manifest, indent=2)
        provenance_id = manifest.get("provenance_id")
        if provenance_id:
            immutable = self.dir / f"provenance.{provenance_id}.json"
            content = encoded + "\n"
            if immutable.exists() and immutable.read_text() != content:
                raise RuntimeError(f"provenance ID collision: {immutable}")
            if not immutable.exists():
                immutable.write_text(content)
        with open(path, "w") as fh:
            fh.write(encoded + "\n")

    def append(self, row: dict) -> None:
        encoded = dumps_result(row)
        with open(self.runs_path, "a") as fh:
            fh.write(encoded + "\n")

    def archive_bytes(self, category: str, content: bytes, suffix: str) -> dict:
        digest = hashlib.sha256(content).hexdigest()
        path = self.dir / "inputs" / f"{category}.{digest}{suffix}"
        if path.exists() and path.read_bytes() != content:
            raise RuntimeError(f"immutable input archive collision: {path}")
        if not path.exists():
            path.write_bytes(content)
        return {"path": str(path.relative_to(self.dir)), "sha256": digest}

    def archive_json(self, category: str, document: dict) -> dict:
        content = (json.dumps(document, sort_keys=True, indent=2,
                              allow_nan=False) + "\n").encode()
        return self.archive_bytes(category, content, ".json")

    def archive_rendered_pipeline(self, path: Path | None) -> dict | None:
        if path is None:
            return None
        content = path.read_bytes()
        digest = hashlib.sha256(content).hexdigest()
        target = self.dir / "inputs" / "rendered-pipelines" / f"{digest}{path.suffix}"
        if not target.exists():
            target.write_bytes(content)
        elif target.read_bytes() != content:
            raise RuntimeError(f"immutable rendered-pipeline collision: {target}")
        return {"path": str(target.relative_to(self.dir)), "sha256": digest}

    def workdir(self, run_id: str) -> Path:
        wd = self.dir / "work" / run_id
        wd.mkdir(parents=True, exist_ok=True)
        return wd

    def completed_execution_ids(self) -> set[str]:
        """Exact executions already done OK across every raw/canonical run file."""
        done: set[str] = set()
        for f in sorted(self.dir.glob("runs*.jsonl")):
            for row in load_result_file(f):
                if row.get("status") == "ok" and row.get("execution_id"):
                    done.add(row["execution_id"])
        return done

    def completed_keys(self) -> set[str]:
        """Compatibility view of completed legacy cell keys (not used for resume)."""
        done: set[str] = set()
        for f in sorted(self.dir.glob("runs*.jsonl")):
            for row in load_result_file(f):
                if row.get("status") == "ok" and row.get("cell_key"):
                    done.add(row["cell_key"])
        return done

    def load_rows(self, canonical: bool = True) -> list[dict]:
        """Load canonical session rows, or only this store's current run file.

        Before merge, the shard files collectively are canonical.  After
        ``benchkit merge`` creates ``runs.jsonl``, that deduplicated file is
        canonical and the retained shards are raw history; reading both would
        duplicate every merged row.

        ``canonical=False`` reads only ``self.runs_path``.  The runner uses that
        view for its per-task summary.  Resume bookkeeping remains separate in
        :meth:`completed_execution_ids` and intentionally scans every run file.
        """
        if canonical:
            merged = self.dir / "runs.jsonl"
            files = ([merged] if merged.exists()
                     else sorted(self.dir.glob("runs.shard-*.jsonl")))
        else:
            files = [self.runs_path]
        rows = []
        for f in files:
            if f.exists():
                rows += load_result_file(f)
        return rows
