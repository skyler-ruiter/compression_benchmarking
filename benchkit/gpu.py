"""Concurrent GPU state sampling for throttle detection.

On shared clusters we can't lock clocks (`nvidia-smi -lgc` is admin-only), so we detect
*during* the benchmark whether the GPU was throttling. A post-hoc query is useless — by
the time the subprocess exits the GPU is idle — so a background thread polls clocks +
throttle reasons while the timed work runs. Pairs with the timing-variance flag in
metrics.py: variance says "this number is unreliable", throttle reasons say "why".

Dependency-free; degrades to {available: False} with no GPU.

**Two vendors, two fidelities.** NVIDIA is sampled via `nvidia-smi` and yields NVML
throttle *reasons* — a positive statement that the GPU was or wasn't being held back.
AMD (ROCm) has no NVML-equivalent reason bitmask, so its samples carry clocks and
temperature only. That difference is reported explicitly rather than papered over:
`throttle_detection` is `"nvml"` or `"unavailable"`, and on AMD `throttled_thermal` is
never set, so a missing reason can't masquerade as a checked-and-clean result. The
*primary* reliability signal — timing variance (metrics.py) — is vendor-independent and
works identically on both.

AMD sampling reads sysfs rather than shelling out to `rocm-smi`: `rocm-smi` is a Python
program costing ~320 ms per invocation (measured on MI100), which at this module's poll
interval would spawn processes continuously alongside the very benchmark being timed.
sysfs reads are syscall-cheap and perturb nothing. `rocm-smi` is still used once, up
front (outside the timed region), to resolve *which* card the job was allocated — sysfs
enumerates every GPU on the node regardless of cgroup/ROCR_VISIBLE_DEVICES masking, so
picking card0 blindly would sample a neighbouring job's GPU on a shared 8-GPU node.
"""
from __future__ import annotations

import os
import re
import subprocess
import threading
import time
from pathlib import Path

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

_DRM = Path("/sys/class/drm")


# ── vendor detection ────────────────────────────────────────────────────────────

def _have(cmd: str) -> bool:
    try:
        return subprocess.run([cmd], capture_output=True, timeout=5).returncode is not None
    except (OSError, subprocess.SubprocessError):
        return False


def detect_vendor() -> str:
    """'nvidia' | 'amd' | 'none'. Cheap enough to call once per sampler."""
    try:
        if subprocess.run(["nvidia-smi", "-L"], capture_output=True,
                          timeout=10).returncode == 0:
            return "nvidia"
    except (OSError, subprocess.SubprocessError):
        pass
    try:
        if subprocess.run(["rocm-smi", "--showid"], capture_output=True,
                          timeout=20).returncode == 0:
            return "amd"
    except (OSError, subprocess.SubprocessError):
        pass
    return "none"


# ── NVIDIA ──────────────────────────────────────────────────────────────────────

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


def _sample_nvidia(dev: str):
    out = subprocess.run(
        ["nvidia-smi", "-i", dev,
         "--query-gpu=clocks.current.sm,clocks_throttle_reasons.active,temperature.gpu",
         "--format=csv,noheader,nounits"],
        capture_output=True, text=True, timeout=5)
    if out.returncode != 0:
        return None
    sm, reasons, temp = out.stdout.strip().splitlines()[0].split(",")
    return int(sm), int(reasons.strip(), 16), int(temp)


# ── AMD / ROCm ──────────────────────────────────────────────────────────────────

def _amd_pci_of_allocated() -> str | None:
    """PCI address of the GPU this job was actually given, via one rocm-smi call.

    rocm-smi honours the cgroup/ROCR_VISIBLE_DEVICES masking SLURM applies; raw sysfs
    does not, so this is what keeps us off a co-tenant's card.
    """
    try:
        out = subprocess.run(["rocm-smi", "--showbus", "--csv"],
                             capture_output=True, text=True, timeout=20)
        if out.returncode != 0:
            return None
        for line in out.stdout.strip().splitlines()[1:]:      # skip csv header
            m = re.search(r"([0-9a-fA-F]{4}:[0-9a-fA-F]{2}:[0-9a-fA-F]{2}\.\d)", line)
            if m:
                return m.group(1).lower()
    except (OSError, subprocess.SubprocessError):
        pass
    return None


def _amd_card_dir(pci: str | None) -> Path | None:
    """Map a PCI address to its /sys/class/drm/cardN/device directory."""
    try:
        cards = sorted(p for p in _DRM.glob("card[0-9]*")
                       if (p / "device" / "uevent").exists())
    except OSError:
        return None
    for card in cards:
        dev = card / "device"
        if not (dev / "gpu_busy_percent").exists():
            continue                                   # not an AMD GPU (or no telemetry)
        if pci is None:
            return dev                                 # no mapping available: first GPU
        try:
            uevent = (dev / "uevent").read_text()
        except OSError:
            continue
        m = re.search(r"PCI_SLOT_NAME=(\S+)", uevent)
        if m and m.group(1).lower() == pci:
            return dev
    return None


def _amd_hwmon_temp_files(dev: Path) -> dict[str, Path]:
    """Map temperature sensor label ('edge'/'junction'/'mem') -> its *_input file."""
    out: dict[str, Path] = {}
    try:
        for hw in (dev / "hwmon").glob("hwmon*"):
            for lbl in hw.glob("temp*_label"):
                try:
                    name = lbl.read_text().strip().lower()
                except OSError:
                    continue
                inp = lbl.with_name(lbl.name.replace("_label", "_input"))
                if inp.exists():
                    out[name] = inp
    except OSError:
        pass
    return out


def _read_int(p: Path) -> int | None:
    try:
        return int(p.read_text().strip())
    except (OSError, ValueError):
        return None


def _parse_dpm_sclk(text: str) -> int | None:
    """pp_dpm_sclk lists levels, current marked '*':  '15: 1502Mhz *'."""
    for line in text.splitlines():
        if "*" in line:
            m = re.search(r"(\d+)\s*Mhz", line, re.IGNORECASE)
            if m:
                return int(m.group(1))
    return None


class _AmdProbe:
    """Resolved sysfs handles for one AMD GPU. All reads are plain file reads."""

    def __init__(self, dev: Path):
        self.dev = dev
        self.busy = dev / "gpu_busy_percent"
        self.sclk = dev / "pp_dpm_sclk"
        temps = _amd_hwmon_temp_files(dev)
        # Junction (hotspot) is the sensor that actually governs thermal throttling;
        # fall back to edge, which is what rocm-smi shows first.
        self.temp = temps.get("junction") or temps.get("edge") or temps.get("mem")

    def sample(self):
        """-> (sclk_mhz, busy_pct, temp_c) or None."""
        try:
            sclk = _parse_dpm_sclk(self.sclk.read_text())
        except OSError:
            return None
        if sclk is None:
            return None
        busy = _read_int(self.busy)
        traw = _read_int(self.temp) if self.temp else None
        temp = round(traw / 1000.0) if traw is not None else None   # millidegrees C
        return sclk, (busy if busy is not None else 0), temp


# ── sampler ─────────────────────────────────────────────────────────────────────

class GpuSampler:
    """Context manager: polls GPU clocks/throttle reasons in a thread while active.

    Public shape is vendor-independent — runner.py just does `with GpuSampler() as s`
    and reads `s.summary()`.
    """

    def __init__(self, interval: float = 0.1, device: str | None = None,
                 vendor: str | None = None):
        self.interval = interval
        self.dev = device or _device_index()
        self.vendor = vendor or detect_vendor()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._samples: list[tuple] = []
        self._amd: _AmdProbe | None = None

    def __enter__(self) -> "GpuSampler":
        try:
            if self.vendor == "nvidia":
                if _sample_nvidia(self.dev) is None:
                    return self
            elif self.vendor == "amd":
                # Resolve the card once, here — outside the timed region.
                dev = _amd_card_dir(_amd_pci_of_allocated())
                if dev is None:
                    return self
                probe = _AmdProbe(dev)
                if probe.sample() is None:
                    return self
                self._amd = probe
            else:
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
                s = (_sample_nvidia(self.dev) if self.vendor == "nvidia"
                     else self._amd.sample() if self._amd else None)
                if s is not None:
                    self._samples.append(s)
            except (OSError, subprocess.SubprocessError, ValueError):
                pass
            self._stop.wait(self.interval)

    def summary(self) -> dict:
        if self.vendor == "amd":
            return self._summary_amd()
        return self._summary_nvidia()

    def _summary_nvidia(self) -> dict:
        # Only "active" (non-idle) samples reflect the timed kernels; startup is idle.
        active = [(sm, r, t) for (sm, r, t) in self._samples if not (r & _IDLE)]
        if not active:
            return {"available": False, "vendor": "nvidia",
                    "n_samples": len(self._samples)}
        clocks = [sm for sm, _, _ in active]
        reasons: set[str] = set()
        for _, r, _ in active:
            for bit, name in _THROTTLE_BITS.items():
                if r & bit:
                    reasons.add(name)
        return {
            "available": True,
            "vendor": "nvidia",
            "throttle_detection": "nvml",
            "n_samples": len(active),
            "sm_clock_min": min(clocks),
            "sm_clock_mean": round(sum(clocks) / len(clocks), 1),
            "sm_clock_max": max(clocks),
            "temp_max": max(t for _, _, t in active),
            "throttle_reasons": sorted(reasons),
            "throttled_thermal": bool(reasons & _THERMAL_HW),
        }

    def _summary_amd(self) -> dict:
        # AMD's analogue of NVML's GpuIdle bit is the busy percentage.
        active = [(sclk, busy, t) for (sclk, busy, t) in self._samples if busy > 0]
        if not active:
            return {"available": False, "vendor": "amd",
                    "throttle_detection": "unavailable",
                    "n_samples": len(self._samples)}
        clocks = [c for c, _, _ in active]
        temps = [t for _, _, t in active if t is not None]
        out = {
            "available": True,
            "vendor": "amd",
            # ROCm exposes no NVML-style throttle-reason bitmask. Say so explicitly:
            # `throttled_thermal` is deliberately absent rather than False, so nothing
            # downstream can read "not throttled" out of "never checked".
            "throttle_detection": "unavailable",
            "n_samples": len(active),
            "sm_clock_min": min(clocks),
            "sm_clock_mean": round(sum(clocks) / len(clocks), 1),
            "sm_clock_max": max(clocks),
            "busy_pct_mean": round(sum(b for _, b, _ in active) / len(active), 1),
        }
        if temps:
            out["temp_max"] = max(temps)
        return out
