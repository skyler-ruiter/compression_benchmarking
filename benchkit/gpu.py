"""Concurrent GPU state sampling for throttle detection.

On shared clusters we can't lock clocks (`nvidia-smi -lgc` is admin-only), so we detect
*during* the benchmark whether the GPU was throttling. A post-hoc query is useless — by
the time the subprocess exits the GPU is idle — so a background thread polls clocks +
throttle reasons while the timed work runs. Pairs with the timing-variance flag in
metrics.py: variance says "this number is unreliable", throttle reasons say "why".

Dependency-free (nvidia-smi subprocess); degrades to {available: False} with no GPU.
"""
from __future__ import annotations

import os
import subprocess
import threading
import time

# NVML clocksThrottleReasons bits. We ignore GpuIdle(0x1), ApplicationsClocksSetting(0x2,
# = clocks deliberately locked, which is good), SyncBoost, DisplayClocks — none are
# performance-robbing under load. Power cap is common on consumer cards and recorded
# separately from the serious thermal/hw slowdowns.
_IDLE = 0x1
_THROTTLE_BITS = {
    0x04: "sw_power_cap",
    0x08: "hw_slowdown",
    0x20: "sw_thermal",
    0x40: "hw_thermal",
    0x80: "hw_power_brake",
}
_THERMAL_HW = {"hw_slowdown", "sw_thermal", "hw_thermal", "hw_power_brake"}


def _device_index() -> str:
    v = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0].strip()
    return v if v.isdigit() else "0"


def max_sm_clock_mhz(device: str | None = None) -> int | None:
    dev = device or _device_index()
    try:
        out = subprocess.run(
            ["nvidia-smi", "-i", dev, "--query-gpu=clocks.max.sm",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5)
        if out.returncode == 0:
            return int(out.stdout.strip().splitlines()[0])
    except (OSError, subprocess.SubprocessError, ValueError):
        pass
    return None


def _sample(dev: str):
    out = subprocess.run(
        ["nvidia-smi", "-i", dev,
         "--query-gpu=clocks.current.sm,clocks_throttle_reasons.active,temperature.gpu",
         "--format=csv,noheader,nounits"],
        capture_output=True, text=True, timeout=5)
    if out.returncode != 0:
        return None
    sm, reasons, temp = out.stdout.strip().splitlines()[0].split(",")
    return int(sm), int(reasons.strip(), 16), int(temp)


class GpuSampler:
    """Context manager: polls GPU clocks/throttle reasons in a thread while active."""

    def __init__(self, interval: float = 0.1, device: str | None = None):
        self.interval = interval
        self.dev = device or _device_index()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._samples: list[tuple[int, int, int]] = []

    def __enter__(self) -> "GpuSampler":
        try:
            if _sample(self.dev) is None:
                return self
        except (OSError, subprocess.SubprocessError, ValueError):
            return self
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                s = _sample(self.dev)
                if s is not None:
                    self._samples.append(s)
            except (OSError, subprocess.SubprocessError, ValueError):
                pass
            self._stop.wait(self.interval)

    def summary(self) -> dict:
        # Only "active" (non-idle) samples reflect the timed kernels; startup is idle.
        active = [(sm, r, t) for (sm, r, t) in self._samples if not (r & _IDLE)]
        if not active:
            return {"available": False, "n_samples": len(self._samples)}
        clocks = [sm for sm, _, _ in active]
        reasons: set[str] = set()
        for _, r, _ in active:
            for bit, name in _THROTTLE_BITS.items():
                if r & bit:
                    reasons.add(name)
        return {
            "available": True,
            "n_samples": len(active),
            "sm_clock_min": min(clocks),
            "sm_clock_mean": round(sum(clocks) / len(clocks), 1),
            "sm_clock_max": max(clocks),
            "temp_max": max(t for _, _, t in active),
            "throttle_reasons": sorted(reasons),
            "throttled_thermal": bool(reasons & _THERMAL_HW),
        }
