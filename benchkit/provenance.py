"""Session provenance capture.

M1: lightweight but real — GPU, driver, host, harness git SHA, config hash. M2 expands
this (locked clocks, ECC, per-compressor commit+build flags). Every result row carries
the session_id as a foreign key (DESIGN.md principle #3).
"""
from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def _sh(argv: list[str]) -> str | None:
    try:
        out = subprocess.run(argv, capture_output=True, text=True, timeout=10)
        return out.stdout.strip() if out.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None


def _gpu_info() -> dict:
    q = _sh(["nvidia-smi",
             "--query-gpu=name,driver_version,memory.total,clocks.sm,clocks.mem,ecc.mode.current",
             "--format=csv,noheader"])
    if not q:
        return {"available": False}
    name, driver, mem, sm, memclk, ecc = (x.strip() for x in q.split(",", 5))
    return {"available": True, "name": name, "driver": driver,
            "memory_total": mem, "sm_clock": sm, "mem_clock": memclk, "ecc": ecc}


def _git_sha(repo: Path) -> str | None:
    return _sh(["git", "-C", str(repo), "rev-parse", "--short", "HEAD"])


def capture_session(config_raw: dict, repo_root: Path,
                    adapter_provenance: dict | None = None) -> dict:
    now = datetime.now(timezone.utc)
    cfg_hash = hashlib.sha256(
        json.dumps(config_raw, sort_keys=True, default=str).encode()).hexdigest()
    host = platform.uname()
    return {
        "session_id": now.strftime("%Y%m%d-%H%M%S") + "-" + host.node,
        "timestamp": now.isoformat(),
        "gpu": _gpu_info(),
        "host": {"node": host.node, "system": host.system, "release": host.release,
                 "machine": host.machine, "processor": platform.processor()},
        "harness": {"git_sha": _git_sha(repo_root), "config_sha256": cfg_hash,
                    "python": platform.python_version()},
        "compressors": adapter_provenance or {},
        "nvidia_smi": _sh(["nvidia-smi"]),
    }
