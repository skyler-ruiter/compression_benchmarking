# cuda_mem_probe

A tiny `LD_PRELOAD` shared library that measures the **peak live device-memory
allocation** of any CUDA process, black-box, without touching its source. Built
by this repo (like `tools/nvcomp_cli/`, `tools/fsz_hosttime/`) so benchkit can
wrap native compressors uniformly and compare their peak device memory to FZGM's.

## Why it exists

FZGM self-reports peak memory via `Pipeline::getPeakMemoryUsage()` (its
`MemoryPool` high-water). Native compressors report nothing, and even if they did
the boundary would differ. To compare fairly we measure **both** sides the same
way: the peak sum of concurrently-live `cudaMalloc`-family bytes, excluding the
CUDA context. Running FZGM under the probe and diffing against its self-report
also validates the completeness of FZGM's own tracker.

## How it works

Symbol interposition. `LD_PRELOAD` makes our `cudaMalloc`/`cudaFree`/... resolve
before libcudart's. Each wrapper resolves the real symbol once with
`dlsym(RTLD_NEXT, name)`, calls it, then updates a global live-bytes counter and a
`ptr -> size` map under a lock; `peak = max(peak, live)`. The CUDA context is not
a `cudaMalloc`, so it is excluded automatically. We track **live** bytes (not
reserved), matching FZGM's pool semantics.

## Symbols to wrap (coverage contract)

Allocations (add size, record ptr): `cudaMalloc`, `cudaMallocAsync`,
`cudaMallocManaged`, `cudaMallocPitch` (bytes = pitch*height), and the driver API
`cuMemAlloc`, `cuMemAllocPitch`, `cuMemAllocManaged`.
Frees (subtract, erase): `cudaFree`, `cudaFreeAsync`, `cuMemFree`.
- Unknown ptr on free -> increment `untracked_frees`, do not go negative.
- `cudaMallocAsync`/`cudaFreeAsync` are stream-ordered; tracking at call time is a
  documented approximation (counts the free when requested, slightly early).
- VMM (`cuMemCreate`+`cuMemMap`) is OUT OF SCOPE for v1 — document it; the NVML
  cross-check covers processes that use it.
- Thread-safe: guard the map + counters (mutex or atomics). Guard against dlsym
  re-entrancy during initialization.

## Output contract (benchkit parses this — do not change keys)

At process exit (destructor/atexit) write one JSON object to the path in
`$CUDA_MEM_PROBE_OUT`, else to stderr each line prefixed `[cuda_mem_probe] `:

```json
{"peak_device_bytes": 123456789, "live_at_exit_bytes": 0, "n_alloc": 42, "n_free": 42, "untracked_frees": 0, "device_count_seen": 1, "vmm_calls_seen": 0}
```

`vmm_calls_seen > 0` warns that VMM allocations occurred and the peak may
undercount. Aggregate across devices (single-GPU here); optionally note per-device.

## Signal control (phase bracketing)

- `SIGUSR1`: reset `peak` to current `live` (start a fresh high-water window).
- `SIGUSR2`: dump the current stats line immediately (does not reset).

Lets the harness bracket a phase (send SIGUSR1 before, read peak after) for tools
that expose a wall-clock phase boundary. Default whole-run peak needs no signals.

## Usage

```bash
make                                   # builds libcudamemprobe.so
CUDA_MEM_PROBE_OUT=/tmp/peak.json \
  LD_PRELOAD=$PWD/libcudamemprobe.so   <compressor CLI ...>
cat /tmp/peak.json
```

## Validation (tests/)

`tests/probe_selftest.cu` performs a **known** allocation pattern with a
computable peak (e.g. alloc 100MB, alloc 50MB, free 100MB, alloc 200MB -> peak
250MB) and `tests/run_selftest.sh` runs it under the probe and asserts the
reported `peak_device_bytes` equals the analytic value (within one page). Cover
plain, async, managed, and pitched allocations, and an unmatched free.

## Known limitations

- **Statically-linked cudart defeats interposition — the probe silently reports 0.**
  `LD_PRELOAD` only overrides *dynamically* resolved symbols; `nvcc` statically links
  `libcudart` by default. Any target must dynamically link cudart (`ldd <bin> | grep libcudart`
  should show it). For nvcc-built tools, compile with `-cudart shared`. Both the smoke test
  and the self-test hit this. FZGM's `fzgmod-cli` and cuSZ/cuSZp2/cuSZp3/FSZ are dynamic;
  FZ-GPU's binary is static and must be rebuilt to be measurable.
- VMM/`cuMemCreate` reservations not tracked (v1) — cross-check with NVML.
- Async free timing is call-ordered, not stream-ordered (small transient error).
- Excludes CUDA context and host (`cudaHostAlloc`) memory by design.
- Counts live, not reserved — a caching sub-allocator's spare capacity is not counted
  (this is the intended, FZGM-comparable semantics).
