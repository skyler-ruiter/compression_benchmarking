#!/usr/bin/env bash
# Build tools/fsz_hosttime — the host-wall-clock harness the FSZ adapter prefers.
#
#   ./scripts/build-fsz-hosttime.sh                    # uses $FSZ_ROOT, sm_90
#   FSZ_ROOT=/path/to/FSZ CUDA_ARCH=80 ./scripts/build-fsz-hosttime.sh
#
# WHY THIS EXISTS: the stock `fsz` CLI reports device time only (a cudaEvent pair
# around the kernel). Per DESIGN D33, a device-only figure is half an answer —
# comparing FZGM's device_ms against a native tool's device_ms has already been
# badly misleading once. This links against libfsz and times std::chrono around
# fsz::compress + cudaStreamSynchronize, the same bracket FZGM's "Host elapsed"
# uses, so both sides of an fsz_vs_native row are measured the same way.
#
# Without it the adapter silently falls back to the stock CLI and FSZ rows carry
# no host figure; provenance.timing_method records which path ran.
#
# FSZ must be built first (it is a source tree, not a prebuilt SDK):
#   cd $FSZ_ROOT && cmake -S . -B build -DCMAKE_CUDA_ARCHITECTURES=90 \
#                 && cmake --build build -j
#
# TOOLCHAIN: same pin as build-nvcomp-cli.sh, for the same reason — on the
# JetStream2 node a bare `nvcc` resolves through nvhpc to a CUDA 11.8 wrapper,
# which cannot target sm_90. Source scripts/env-jetstream2.sh (or ~/load-env)
# first, or set CUDA_HOME.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC="${REPO}/tools/fsz_hosttime"

FSZ_ROOT="${FSZ_ROOT:-${HOME}/compressors/FSZ}"
CUDA_ARCH="${CUDA_ARCH:-90}"
NVCC="${NVCC:-nvcc}"

if [[ ! -d "${FSZ_ROOT}/include/fsz" ]]; then
  echo "error: no FSZ headers at ${FSZ_ROOT}/include/fsz" >&2
  echo "       set FSZ_ROOT, or clone https://github.com/JiajunHuang1999/FSZ" >&2
  exit 1
fi
if [[ ! -f "${FSZ_ROOT}/build/libfsz.so" && ! -f "${FSZ_ROOT}/build/libfsz.a" ]]; then
  echo "error: no libfsz in ${FSZ_ROOT}/build — build FSZ first:" >&2
  echo "       cmake -S ${FSZ_ROOT} -B ${FSZ_ROOT}/build -DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCH}" >&2
  echo "       cmake --build ${FSZ_ROOT}/build -j" >&2
  exit 1
fi

set -x
"${NVCC}" -O3 -std=c++17 -arch="sm_${CUDA_ARCH}" \
  "${SRC}/fsz_hosttime.cu" -o "${SRC}/fsz_hosttime" \
  -I"${FSZ_ROOT}/include" -L"${FSZ_ROOT}/build" -lfsz \
  -Xlinker -rpath -Xlinker "${FSZ_ROOT}/build"
set +x

echo
echo "built: ${SRC}/fsz_hosttime"
echo "the FSZ adapter finds it here automatically; override with FSZ_HOSTTIME_CLI."
