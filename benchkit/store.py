"""Append-only results store, shard-aware for HPC job arrays.

One session = one directory under the results root. Within it:
  - runs.jsonl                       single-process runs
  - runs.shard-{k}-of-{N}.jsonl      one file per job-array task (no append contention)
  - provenance[.shard-{k}-of-{N}].json   per-task env manifest (each task may be a
                                          different node/GPU, so capture per shard)
  - logs/ , work/                    raw tool output + intermediate artifacts

Rows are append-only so re-runs never clobber; resumability is by scanning all run files
for completed `cell_key`s (DESIGN.md principle #4, §9 M2).
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


def _shard_suffix(shard: tuple[int, int] | None) -> str:
    return "" if shard is None else f".shard-{shard[0]}-of-{shard[1]}"


class ResultStore:
    def __init__(self, results_root: Path, session_id: str,
                 shard: tuple[int, int] | None = None):
        self.dir = Path(results_root) / session_id
        self.dir.mkdir(parents=True, exist_ok=True)
        (self.dir / "logs").mkdir(exist_ok=True)
        (self.dir / "work").mkdir(exist_ok=True)
        self.session_id = session_id
        self.shard = shard
        self.runs_path = self.dir / f"runs{_shard_suffix(shard)}.jsonl"

    def write_provenance(self, manifest: dict) -> None:
        path = self.dir / f"provenance{_shard_suffix(self.shard)}.json"
        with open(path, "w") as fh:
            json.dump(manifest, fh, indent=2, default=str)

    def append(self, row: dict) -> None:
        with open(self.runs_path, "a") as fh:
            fh.write(json.dumps(row, default=str) + "\n")

    def workdir(self, run_id: str) -> Path:
        wd = self.dir / "work" / run_id
        wd.mkdir(parents=True, exist_ok=True)
        return wd

    def completed_keys(self) -> set[str]:
        """cell_keys already done OK across *all* run files in this session (resume)."""
        done: set[str] = set()
        for f in sorted(self.dir.glob("runs*.jsonl")):
            for line in f.read_text().splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                if row.get("status") == "ok" and row.get("cell_key"):
                    done.add(row["cell_key"])
        return done

    def load_rows(self, all_shards: bool = True) -> list[dict]:
        files = sorted(self.dir.glob("runs*.jsonl")) if all_shards else [self.runs_path]
        rows = []
        for f in files:
            if f.exists():
                rows += [json.loads(l) for l in f.read_text().splitlines() if l.strip()]
        return rows
