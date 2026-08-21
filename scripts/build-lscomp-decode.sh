#!/usr/bin/env bash
# Build the standalone decoder required by benchkit's lsCOMP adapter.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LSCOMP_ROOT="${LSCOMP_ROOT:-${HOME}/compressors/lsCOMP}"
CUDA_ARCH="${CUDA_ARCH:-90}"
NVCC="${NVCC:-nvcc}"

"${NVCC}" -O3 -std=c++17 -arch="sm_${CUDA_ARCH}" \
  "${REPO}/tools/lscomp_decode/lscomp_decode.cu" \
  -o "${REPO}/tools/lscomp_decode/lscomp_decode" \
  -I"${LSCOMP_ROOT}/include" -L"${LSCOMP_ROOT}/build" -llsCOMP \
  -Xlinker -rpath -Xlinker "${LSCOMP_ROOT}/build"

echo "built: ${REPO}/tools/lscomp_decode/lscomp_decode"
