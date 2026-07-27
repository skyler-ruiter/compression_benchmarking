"""Session provenance capture.

M1: lightweight but real — GPU, driver, host, harness git SHA, config hash. M2 expands
this (locked clocks, ECC, per-compressor commit+build flags). Every result row carries
the session_id as a foreign key (DESIGN.md principle #3).
"""
from __future__ import annotations

import csv
import hashlib
import io
import os
import json
import platform
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def _sh(argv: list[str]) -> str | None:
    try:
        out = subprocess.run(argv, capture_output=True, text=True, timeout=10)
        return out.stdout.strip() if out.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None


def _gpu_info_nvidia() -> dict | None:
    q = _sh(["nvidia-smi",
             "--query-gpu=name,driver_version,memory.total,clocks.sm,clocks.max.sm,clocks.mem,ecc.mode.current,persistence_mode",
             "--format=csv,noheader"])
    if not q:
        return None
    # One CSV line per *visible* GPU. Under `sbatch --exclusive` the whole node is
    # allocated, so this is 8 lines on Delta's gpuH200x8/gpuMI100x8 — splitting the
    # whole blob on "," made the 8th field swallow every subsequent line (observed:
    # `persistence` = "Enabled\nNVIDIA H200, 570.148.08, ...x7"). Take the first row
    # only; the benchmark itself pins one device.
    first = q.strip().splitlines()[0]
    name, driver, mem, sm, smmax, memclk, ecc, persist = (x.strip() for x in first.split(",", 7))
    return {"available": True, "vendor": "nvidia",
            "name": name, "driver": driver, "memory_total": mem,
            "sm_clock": sm, "sm_clock_max": smmax, "mem_clock": memclk,
            "ecc": ecc, "persistence": persist}


def _gpu_info_amd() -> dict | None:
    """ROCm equivalent. Field names mirror the NVIDIA dict so downstream analysis and
    the baseline schema stay vendor-independent; anything ROCm doesn't expose is simply
    absent rather than guessed (notably ECC mode and persistence, which have no direct
    rocm-smi analogue)."""
    # `rocm-smi --showproductname --csv` is a proper named-column table:
    #   device,Card Series,Card Model,Card Vendor,Card SKU,...,GFX Version
    # Parse it by header name. Positional/heuristic parsing gets this wrong — "Card
    # Vendor" ("Advanced Micro Devices Inc. [AMD/ATI]") is the longest cell in the row
    # but says nothing about *which* GPU, and the name is what partitions results by
    # device (DESIGN.md §12), so MI100-vs-MI250 has to survive it.
    name = gfx = None
    prod = _sh(["rocm-smi", "--showproductname", "--csv"])
    if prod:
        try:
            rows = list(csv.DictReader(io.StringIO(prod.strip())))
        except csv.Error:
            rows = []
        for row in rows:
            clean = {(k or "").strip(): (v or "").strip() for k, v in row.items()}
            gfx = clean.get("GFX Version") or None
            for key in ("Card Series", "Card Model", "Card SKU"):
                val = clean.get(key)
                if val and not val.startswith("0x"):
                    name = val
                    break
            if name or gfx:
                break
    if name is None and gfx is None:
        return None
    # "AMD Instinct MI100 (gfx908)" — the gfx target is what the build was compiled for
    # (CMAKE_HIP_ARCHITECTURES), so keeping both makes a row self-describing.
    label = " ".join(x for x in (name, f"({gfx})" if gfx else None) if x) or "AMD GPU"

    out: dict = {"available": True, "vendor": "amd", "name": label}
    if gfx:
        out["gfx_arch"] = gfx
    drv = _sh(["rocm-smi", "--showdriverversion", "--csv"])
    if drv:
        m = re.search(r"(\d+\.\d[\d.\-]*)", drv)
        if m:
            out["driver"] = m.group(1)
    vram = _sh(["rocm-smi", "--showmeminfo", "vram", "--csv"])
    if vram:
        m = re.search(r"(\d{6,})", vram)          # total VRAM in bytes
        if m:
            out["memory_total"] = f"{int(m.group(1)) // (1024*1024)} MiB"
    clocks = _sh(["rocm-smi", "--showclocks", "--csv"])
    if clocks:
        m = re.search(r"sclk clock speed:?,?\s*\((\d+)Mhz\)", clocks, re.IGNORECASE)
        if not m:
            m = re.search(r"\((\d+)Mhz\)", clocks)
        if m:
            out["sm_clock"] = f"{m.group(1)} MHz"
    return out


def _gpu_info() -> dict:
    """GPU description for the session manifest, whichever vendor is present.

    Results from different GPUs are partitioned by provenance and never pooled
    (DESIGN.md §12), so `vendor`/`name` here is what keeps an MI100 run from being
    silently compared against an H200 one.
    """
    return _gpu_info_nvidia() or _gpu_info_amd() or {"available": False}


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
