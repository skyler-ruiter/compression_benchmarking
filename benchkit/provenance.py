"""Session provenance capture.

M1: lightweight but real — GPU, driver, host, harness git SHA, config hash. M2 expands
this (locked clocks, ECC, per-compressor commit+build flags). Every result row carries
the session_id as a foreign key (DESIGN.md principle #3).
"""
from __future__ import annotations

import hashlib
import os
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
             "--query-gpu=name,driver_version,memory.total,clocks.sm,clocks.max.sm,clocks.mem,ecc.mode.current,persistence_mode",
             "--format=csv,noheader"])
    if not q:
        return {"available": False}
    name, driver, mem, sm, smmax, memclk, ecc, persist = (x.strip() for x in q.split(",", 7))
    return {"available": True, "name": name, "driver": driver, "memory_total": mem,
            "sm_clock": sm, "sm_clock_max": smmax, "mem_clock": memclk,
            "ecc": ecc, "persistence": persist}


def _git_sha(repo: Path) -> str | None:
    return _sh(["git", "-C", str(repo), "rev-parse", "--short", "HEAD"])


# Scheduler / environment vars worth recording on HPC (job, node, GPU pinning, array).
_SCHED_ENV = [
    "SLURM_JOB_ID", "SLURM_ARRAY_JOB_ID", "SLURM_ARRAY_TASK_ID", "SLURM_NODELIST",
    "SLURM_JOB_PARTITION", "SLURM_NTASKS", "SLURM_GPUS", "SLURM_JOB_GPUS",
    "PBS_JOBID", "PBS_NODEFILE", "LSB_JOBID",
    "CUDA_VISIBLE_DEVICES", "GPU_DEVICE_ORDINAL",
]


def _scheduler_env() -> dict:
    present = {k: os.environ[k] for k in _SCHED_ENV if k in os.environ}
    if "SLURM_JOB_ID" in present:
        present["_scheduler"] = "slurm"
    elif "PBS_JOBID" in present:
        present["_scheduler"] = "pbs"
    elif "LSB_JOBID" in present:
        present["_scheduler"] = "lsf"
    return present


def _software_env() -> dict:
    """Loaded modules / Spack — the reproducibility-critical software stack on HPC."""
    return {
        "modules": os.environ.get("LOADEDMODULES"),       # `module list` content
        "module_list": _sh(["bash", "-lc", "module list 2>&1"]),
        "spack_env": os.environ.get("SPACK_ENV"),
        "nvcc": _sh(["nvcc", "--version"]),
    }


def capture_session(config_raw: dict, repo_root: Path,
                    adapter_provenance: dict | None = None,
                    session_id: str | None = None,
                    shard: tuple[int, int] | None = None) -> dict:
    now = datetime.now(timezone.utc)
    cfg_hash = hashlib.sha256(
        json.dumps(config_raw, sort_keys=True, default=str).encode()).hexdigest()
    host = platform.uname()
    sid = session_id or (now.strftime("%Y%m%d-%H%M%S") + "-" + host.node)
    return {
        "session_id": sid,
        "timestamp": now.isoformat(),
        "shard": list(shard) if shard else None,
        "gpu": _gpu_info(),
        "host": {"node": host.node, "system": host.system, "release": host.release,
                 "machine": host.machine, "processor": platform.processor()},
        "scheduler": _scheduler_env(),
        "software": _software_env(),
        "harness": {"git_sha": _git_sha(repo_root), "config_sha256": cfg_hash,
                    "python": platform.python_version()},
        "compressors": adapter_provenance or {},
        "nvidia_smi": _sh(["nvidia-smi"]),
    }
