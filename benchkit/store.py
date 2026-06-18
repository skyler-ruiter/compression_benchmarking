"""Append-only results store.

One session = one directory under results/. Rows are appended to runs.jsonl (one JSON
object per line); provenance.json holds the session manifest; logs/ and work/ hold raw
tool output and intermediate artifacts. Append-only so re-runs never clobber
(DESIGN.md principle #4).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


class ResultStore:
    def __init__(self, results_root: Path, session_id: str):
        self.dir = results_root / session_id
        self.dir.mkdir(parents=True, exist_ok=True)
        (self.dir / "logs").mkdir(exist_ok=True)
        (self.dir / "work").mkdir(exist_ok=True)
        self.runs_path = self.dir / "runs.jsonl"
        self.session_id = session_id

    def write_provenance(self, manifest: dict) -> None:
        with open(self.dir / "provenance.json", "w") as fh:
            json.dump(manifest, fh, indent=2, default=str)

    def append(self, row: dict) -> None:
        with open(self.runs_path, "a") as fh:
            fh.write(json.dumps(row, default=str) + "\n")

    def workdir(self, run_id: str) -> Path:
        wd = self.dir / "work" / run_id
        wd.mkdir(parents=True, exist_ok=True)
        return wd

    def load_rows(self) -> list[dict]:
        if not self.runs_path.exists():
            return []
        with open(self.runs_path) as fh:
            return [json.loads(line) for line in fh if line.strip()]
