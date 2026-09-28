"""Per-process peak device memory via NVML, sampled from a background thread.

The primary instrument of the peak-memory study (docs/peak_memory_methodology.md).
NVML's per-process `usedGpuMemory` is the driver's own resident-memory accounting
for a process: CUDA context, loaded modules, every cudaMalloc-family allocation,
driver VMM, and allocator-internal churn that no LD_PRELOAD probe can see.

Bound through ctypes against libnvidia-ml.so.1 rather than pynvml so the repo keeps
its numpy+pyyaml dependency contract. `nvmlDeviceGetComputeRunningProcesses_v3`
returns `nvmlProcessInfo_v2_t` {uint pid; ull usedGpuMemory; uint gpuInstanceId;
uint computeInstanceId} — the same layout pynvml's c_nvmlProcessInfo_v2_t uses.

Sampling a subprocess.run()-spawned tool: the sampler watches a set of PIDs (the
Popen child and any descendants it finds via /proc) and keeps the running max of
their summed usedGpuMemory. It also records any *foreign* compute process seen on
the device during the window, so a polluted measurement is visible in the row.
"""
from __future__ import annotations

import ctypes
import os
import threading
import time
from pathlib import Path

NVML_SUCCESS = 0
NVML_ERROR_INSUFFICIENT_SIZE = 7
NVML_VALUE_NOT_AVAILABLE = 0xFFFFFFFFFFFFFFFF


class _ProcInfo(ctypes.Structure):
    _fields_ = [("pid", ctypes.c_uint),
                ("usedGpuMemory", ctypes.c_ulonglong),
                ("gpuInstanceId", ctypes.c_uint),
                ("computeInstanceId", ctypes.c_uint)]


class _Memory(ctypes.Structure):
    _fields_ = [("total", ctypes.c_ulonglong),
                ("free", ctypes.c_ulonglong),
                ("used", ctypes.c_ulonglong)]


class Nvml:
    """Minimal NVML binding: init once, query one device's compute processes."""

    def __init__(self, device_index: int = 0):
        self.lib = ctypes.CDLL("libnvidia-ml.so.1")
        self._check(self.lib.nvmlInit_v2(), "nvmlInit_v2")
        self.handle = ctypes.c_void_p()
        self._check(self.lib.nvmlDeviceGetHandleByIndex_v2(
            ctypes.c_uint(device_index), ctypes.byref(self.handle)),
            "nvmlDeviceGetHandleByIndex_v2")
        self._fn = self.lib.nvmlDeviceGetComputeRunningProcesses_v3
        self._cap = 64
        self._buf = (_ProcInfo * self._cap)()

    @staticmethod
    def _check(rc: int, what: str) -> None:
        if rc != NVML_SUCCESS:
            raise RuntimeError(f"{what} failed: NVML return code {rc}")

    def processes(self) -> dict[int, int]:
        """{pid: usedGpuMemory bytes} for every compute process on the device."""
        while True:
            n = ctypes.c_uint(self._cap)
            rc = self._fn(self.handle, ctypes.byref(n), self._buf)
            if rc == NVML_ERROR_INSUFFICIENT_SIZE:
                self._cap *= 2
                self._buf = (_ProcInfo * self._cap)()
                continue
            self._check(rc, "nvmlDeviceGetComputeRunningProcesses_v3")
            out = {}
            for i in range(n.value):
                used = self._buf[i].usedGpuMemory
                out[int(self._buf[i].pid)] = 0 if used == NVML_VALUE_NOT_AVAILABLE else int(used)
            return out

    def device_used_bytes(self) -> int:
        m = _Memory()
        self._check(self.lib.nvmlDeviceGetMemoryInfo(self.handle, ctypes.byref(m)),
                    "nvmlDeviceGetMemoryInfo")
        return int(m.used)

    def string(self, fn_name: str, *handle_arg) -> str:
        buf = ctypes.create_string_buffer(256)
        fn = getattr(self.lib, fn_name)
        args = [*handle_arg, buf, ctypes.c_uint(256)]
        self._check(fn(*args), fn_name)
        return buf.value.decode()

    def driver_version(self) -> str:
        return self.string("nvmlSystemGetDriverVersion")

    def device_name(self) -> str:
        return self.string("nvmlDeviceGetName", self.handle)

    def device_uuid(self) -> str:
        return self.string("nvmlDeviceGetUUID", self.handle)


def _descendants(root: int) -> set[int]:
    """root plus every live descendant, from /proc/<pid>/task/*/children."""
    seen, stack = {root}, [root]
    while stack:
        pid = stack.pop()
        for task in Path(f"/proc/{pid}/task").glob("*"):
            try:
                kids = (task / "children").read_text().split()
            except OSError:
                continue
            for k in kids:
                k = int(k)
                if k not in seen:
                    seen.add(k)
                    stack.append(k)
    return seen


class ProcessPeakSampler:
    """Background max-of-sum over a process tree's NVML usedGpuMemory.

    Start before (or immediately after) the child is spawned, call watch(pid) once
    the PID is known, stop() after it exits. period_s=0.5e-3 matches the protocol.
    """

    def __init__(self, nvml: Nvml, period_s: float = 0.5e-3, preexisting: set[int] | None = None):
        self.nvml = nvml
        self.period_s = period_s
        self.preexisting = set(preexisting or ())
        self.root: int | None = None
        self.peak_bytes = 0
        self.samples = 0
        self.nonzero_samples = 0
        self.foreign: dict[int, int] = {}   # pid -> max usedGpuMemory seen
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._tree: set[int] = set()
        self._tree_refresh = 0.0

    def watch(self, pid: int) -> None:
        self.root = pid

    def start(self) -> "ProcessPeakSampler":
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        self._thread.join()

    def _run(self) -> None:
        while not self._stop.is_set():
            procs = self.nvml.processes()
            now = time.monotonic()
            if self.root is not None and now - self._tree_refresh > 0.01:
                self._tree = _descendants(self.root)
                self._tree_refresh = now
            mine = 0
            for pid, used in procs.items():
                if pid in self._tree or pid == self.root:
                    mine += used
                elif pid != os.getpid():
                    self.foreign[pid] = max(self.foreign.get(pid, 0), used)
            self.samples += 1
            if mine:
                self.nonzero_samples += 1
            if mine > self.peak_bytes:
                self.peak_bytes = mine
            time.sleep(self.period_s)
