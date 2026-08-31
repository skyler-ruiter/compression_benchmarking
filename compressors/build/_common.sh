# Sourced by every build/<name>.sh. Resolves the toolchain from the environment
# (set by scripts/env-<machine>.sh) with desktop-friendly defaults, so a bare
# shell still builds — it just targets $DEFAULT_ARCH.
#
# Contract (all optional):
#   CUDA_ARCH   GPU arch digits: 80 / 86 / 90 / 90a. Default 90.
#   CC / CXX    host compiler. Default: gcc / g++ on PATH.
#   CUDA_ROOT   toolkit prefix with bin/nvcc. Default: derived from `which nvcc`.
#   JOBS        build parallelism. Default: nproc.
#   COMPRESSORS_ROOT   where checkouts live. Default: ~/compressors.

set -euo pipefail

CUDA_ARCH="${CUDA_ARCH:-90}"
CC="${CC:-$(command -v gcc || true)}"
CXX="${CXX:-$(command -v g++ || true)}"
JOBS="${JOBS:-$(nproc)}"
COMPRESSORS_ROOT="${COMPRESSORS_ROOT:-$HOME/compressors}"

if [ -z "${CUDA_ROOT:-}" ]; then
  _nvcc="$(command -v nvcc || true)"
  [ -n "$_nvcc" ] && CUDA_ROOT="$(cd "$(dirname "$_nvcc")/.." && pwd)"
fi

# Common CMake args for the CUDA compressors.
cuda_cmake_args() {
  local a=(
    -DCMAKE_BUILD_TYPE=Release
    -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCH}"
  )
  [ -n "${CC:-}" ]        && a+=(-DCMAKE_C_COMPILER="${CC}")
  [ -n "${CXX:-}" ]       && a+=(-DCMAKE_CXX_COMPILER="${CXX}")
  [ -n "${CUDA_ROOT:-}" ] && a+=(-DCUDAToolkit_ROOT="${CUDA_ROOT}"
                                 -DCMAKE_CUDA_COMPILER="${CUDA_ROOT}/bin/nvcc")
  printf '%s\n' "${a[@]}"
}

# gencode string for the Makefile-based builds (FZ-GPU).
gencode_flags() {
  echo "-gencode arch=compute_${CUDA_ARCH%a},code=sm_${CUDA_ARCH} -gencode arch=compute_${CUDA_ARCH%a},code=compute_${CUDA_ARCH%a}"
}

banner() { printf '\n\033[1m── %s ──\033[0m\n' "$*"; }
