#!/usr/bin/env bash
# run_selftest.sh — build libcudamemprobe.so, compile probe_selftest.cu, run it
# under the probe, and assert the reported peak_device_bytes / untracked_frees
# match the analytically-known values the test program itself computes.
#
# Invoked as `make test` from tools/cuda_mem_probe/Makefile, or directly.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TOOL_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# nvcc lives here on this host; harmless to prepend if it's already on PATH.
NVCC_BIN_DIR="/software/u24/nvhpc/25.7/Linux_x86_64/25.7/cuda/12.9/bin"
if [ -d "$NVCC_BIN_DIR" ]; then
    export PATH="$NVCC_BIN_DIR:$PATH"
fi

PAGE_BYTES=4096
SO_PATH="$TOOL_DIR/libcudamemprobe.so"
BIN_PATH="$SCRIPT_DIR/probe_selftest"

echo "== cuda_mem_probe selftest =="

# ---- Availability gate: skip cleanly (exit 0) with no GPU/CUDA toolchain ----
# so this doesn't hard-fail CPU-only CI. Still attempt the library build,
# best-effort, since that part needs no GPU.
have_cuda=1
if ! command -v nvcc >/dev/null 2>&1; then
    have_cuda=0
fi
if [ "$have_cuda" -eq 1 ] && { ! command -v nvidia-smi >/dev/null 2>&1 || ! nvidia-smi >/dev/null 2>&1; }; then
    have_cuda=0
fi

if [ "$have_cuda" -eq 0 ]; then
    echo "SKIP: no CUDA/GPU available on this host"
    echo "-- attempting best-effort build of libcudamemprobe.so anyway --"
    (cd "$TOOL_DIR" && make) || echo "note: libcudamemprobe.so build also failed/incomplete (non-fatal for SKIP)"
    exit 0
fi

echo "-- building libcudamemprobe.so --"
if ! (cd "$TOOL_DIR" && make); then
    echo "FAIL: libcudamemprobe.so build failed (see compiler errors above)."
    echo "      Note: this library is under active implementation elsewhere in the"
    echo "      repo; if cuda_mem_probe.c is still a scaffold, this failure is expected"
    echo "      until that work lands."
    exit 1
fi

if [ ! -f "$SO_PATH" ]; then
    echo "FAIL: make reported success but $SO_PATH does not exist."
    exit 1
fi

echo "-- compiling probe_selftest.cu --"
# -cudart shared: nvcc links the CUDA runtime STATICALLY by default, which
# resolves cudaMalloc/cudaFree at link time and makes them immune to
# LD_PRELOAD interposition. The probe only works against a dynamically
# linked libcudart, so force that here.
nvcc -O2 -cudart shared -o "$BIN_PATH" "$SCRIPT_DIR/probe_selftest.cu"

OUT_JSON="$(mktemp /tmp/cuda_mem_probe_selftest.XXXXXX.json)"
STDOUT_LOG="$(mktemp /tmp/cuda_mem_probe_selftest_stdout.XXXXXX.log)"
cleanup() { rm -f "$OUT_JSON" "$STDOUT_LOG"; }
trap cleanup EXIT

echo "-- running probe_selftest under libcudamemprobe.so --"
CUDA_MEM_PROBE_OUT="$OUT_JSON" LD_PRELOAD="$SO_PATH" "$BIN_PATH" | tee "$STDOUT_LOG"

EXPECTED_PEAK="$(grep -o 'EXPECTED_PEAK=[0-9]*' "$STDOUT_LOG" | head -1 | cut -d= -f2 || true)"
EXPECTED_UNTRACKED="$(grep -o 'EXPECTED_UNTRACKED_FREES=[0-9]*' "$STDOUT_LOG" | head -1 | cut -d= -f2 || true)"

if [ -z "${EXPECTED_PEAK:-}" ] || [ -z "${EXPECTED_UNTRACKED:-}" ]; then
    echo "FAIL: probe_selftest did not print EXPECTED_PEAK=/EXPECTED_UNTRACKED_FREES= to stdout."
    exit 1
fi

if [ ! -s "$OUT_JSON" ]; then
    echo "FAIL: no (or empty) probe output at \$CUDA_MEM_PROBE_OUT ($OUT_JSON)."
    echo "      The probe should write its JSON report there at process exit."
    exit 1
fi

echo "-- probe report ($OUT_JSON) --"
cat "$OUT_JSON"
echo

PEAK_DEVICE_BYTES="$(python3 -c "
import json
with open('$OUT_JSON') as f:
    d = json.load(f)
print(d['peak_device_bytes'])
" 2>/dev/null || true)"
UNTRACKED_FREES="$(python3 -c "
import json
with open('$OUT_JSON') as f:
    d = json.load(f)
print(d['untracked_frees'])
" 2>/dev/null || true)"

if [ -z "${PEAK_DEVICE_BYTES:-}" ] || [ -z "${UNTRACKED_FREES:-}" ]; then
    echo "FAIL: could not parse peak_device_bytes/untracked_frees as JSON from $OUT_JSON"
    exit 1
fi

echo "Expected peak_device_bytes:   $EXPECTED_PEAK"
echo "Reported peak_device_bytes:   $PEAK_DEVICE_BYTES"
echo "Expected untracked_frees:     $EXPECTED_UNTRACKED"
echo "Reported untracked_frees:     $UNTRACKED_FREES"

fail=0

diff=$(( PEAK_DEVICE_BYTES > EXPECTED_PEAK ? PEAK_DEVICE_BYTES - EXPECTED_PEAK : EXPECTED_PEAK - PEAK_DEVICE_BYTES ))
if [ "$diff" -gt "$PAGE_BYTES" ]; then
    echo "FAIL: peak_device_bytes differs from expected by more than one page ($PAGE_BYTES bytes): diff=$diff"
    fail=1
fi

if [ "$UNTRACKED_FREES" -ne "$EXPECTED_UNTRACKED" ]; then
    echo "FAIL: untracked_frees=$UNTRACKED_FREES, expected exactly $EXPECTED_UNTRACKED (the one deliberate bogus free; matched frees must not be counted as untracked)."
    fail=1
fi

if [ "$fail" -ne 0 ]; then
    echo "== RESULT: FAIL =="
    exit 1
fi

echo "== RESULT: PASS =="
exit 0
