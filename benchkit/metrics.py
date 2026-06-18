"""Harness-owned metrics.

Per DESIGN.md principle #1, the harness computes all size and quality metrics itself
from raw artifacts; it never trusts a tool's self-reported CR/PSNR/throughput. The only
value taken from a tool is device kernel time (handled in the adapter). Throughput is
always recomputed here in one unit convention (decimal GB/s) from raw bytes + a chosen
device time, so cross-tool unit differences (GiB/s, MiB/ms) cannot leak in.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np

_NP_DTYPE = {"f32": np.float32, "f64": np.float64,
             "i32": np.int32, "i64": np.int64}


def _read(path: str | Path, dtype: str, count: int) -> np.ndarray:
    """Read exactly `count` elements, ignoring any chunk padding past that."""
    arr = np.fromfile(path, dtype=_NP_DTYPE[dtype], count=count)
    if arr.size < count:
        raise ValueError(f"{path}: read {arr.size} elements, expected {count}")
    return arr


@dataclass
class QualityMetrics:
    val_min: float
    val_max: float
    val_range: float
    val_maxabs: float
    mse: float
    psnr: float
    nrmse: float
    max_abs_err: float
    max_rel_err: float
    rel_basis: str
    eb_abs_effective: float
    err_over_bound: float       # realized max error / requested abs bound
    eb_satisfied: bool


def compute_quality(original_path: str | Path, decompressed_path: str | Path,
                    dtype: str, num_elements: int,
                    error_bound: float, basis: str,
                    eb_tol: float = 1e-3) -> QualityMetrics:
    """Compare original vs decompressed (both truncated to num_elements).

    `basis` is the eb basis the adapter translated the canonical mode into — one of
    "abs" (eb), "range" (eb x [max-min]), "maxabs" (eb x max|data|) (DESIGN.md §5.4).
    `err_over_bound` exposes the realized overshoot so a near-miss is visible, not hidden
    behind the boolean.
    """
    orig = _read(original_path, dtype, num_elements).astype(np.float64)
    dec = _read(decompressed_path, dtype, num_elements).astype(np.float64)

    vmin, vmax = float(orig.min()), float(orig.max())
    vrange = vmax - vmin
    vmaxabs = float(np.abs(orig).max())
    diff = np.abs(orig - dec)
    mse = float(np.mean((orig - dec) ** 2))
    max_abs = float(diff.max())
    max_rel = max_abs / vrange if vrange > 0 else 0.0

    if mse == 0.0:
        psnr = math.inf
    elif vrange > 0:
        psnr = 20.0 * math.log10(vrange) - 10.0 * math.log10(mse)
    else:
        psnr = 0.0
    nrmse = math.sqrt(mse) / vrange if vrange > 0 else 0.0

    basis_val = {"abs": 1.0, "range": vrange, "maxabs": vmaxabs}.get(basis)
    if basis_val is None:
        raise ValueError(f"unknown eb basis '{basis}'")
    eb_abs = error_bound * basis_val
    over = max_abs / eb_abs if eb_abs > 0 else math.inf
    eb_satisfied = max_abs <= eb_abs * (1.0 + eb_tol)

    return QualityMetrics(
        val_min=vmin, val_max=vmax, val_range=vrange, val_maxabs=vmaxabs, mse=mse,
        psnr=psnr, nrmse=nrmse, max_abs_err=max_abs, max_rel_err=max_rel,
        rel_basis=basis, eb_abs_effective=eb_abs, err_over_bound=over,
        eb_satisfied=eb_satisfied,
    )


@dataclass
class SizeMetrics:
    original_bytes: int
    compressed_bytes: int
    cr: float
    bitrate_bits_per_elem: float


def compute_size(original_bytes: int, compressed_bytes: int, num_elements: int) -> SizeMetrics:
    cr = original_bytes / compressed_bytes if compressed_bytes else math.inf
    bitrate = (compressed_bytes * 8.0) / num_elements if num_elements else math.inf
    return SizeMetrics(original_bytes, compressed_bytes, cr, bitrate)


@dataclass
class TimingStat:
    median_ms: float
    min_ms: float
    max_ms: float
    n: int
    throughput_gbs: float   # original_bytes / median_ms, decimal GB/s


def _gbs(original_bytes: int, ms: float) -> float:
    return original_bytes / (ms * 1e-3) / 1e9 if ms > 0 else 0.0


def summarize_timing(device_ms_all: list[float], original_bytes: int,
                     warmup_reps: int) -> TimingStat:
    """Drop warmup reps, summarize the rest. Throughput uses the median device time."""
    kept = list(device_ms_all[warmup_reps:]) or list(device_ms_all)
    arr = np.asarray(kept, dtype=np.float64)
    median = float(np.median(arr))
    return TimingStat(
        median_ms=median, min_ms=float(arr.min()), max_ms=float(arr.max()),
        n=int(arr.size), throughput_gbs=_gbs(original_bytes, median),
    )


def as_dict(obj) -> dict:
    return asdict(obj)
