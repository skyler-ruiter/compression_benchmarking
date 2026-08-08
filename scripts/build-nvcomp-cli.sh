#!/usr/bin/env bash
# Build tools/nvcomp_cli — the CLI the nvcomp adapter drives.
#
#   ./scripts/build-nvcomp-cli.sh                 # uses $NVCOMP_ROOT, sm_90
#   NVCOMP_ROOT=/path/to/nvcomp CUDA_ARCH=80 ./scripts/build-nvcomp-cli.sh
#
# nvCOMP is a prebuilt SDK download (developer.nvidia.com/nvcomp), not a source
# tree: unpack it somewhere and point NVCOMP_ROOT at the directory containing
# include/ and lib/. Zstd needs nvCOMP >= 3.0; this is developed against 5.2.0.10.
#
# TOOLCHAIN: this deliberately pins nvcc and g++ rather than taking whatever is
# first on PATH. On the JetStream2 node, `nvcc` resolves through the nvhpc bundle
# to a CUDA *11.8* wrapper, and nvhpc's default C++ is nvc++ — which both fails to
# compile nvCOMP's headers and, separately, has a stack-alignment codegen bug in
# large CUDA translation units. Picking the CUDA 12.x nvcc with g++ as host
# compiler avoids both.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC="${REPO}/tools/nvcomp_cli"
BUILD="${SRC}/build"

NVCOMP_ROOT="${NVCOMP_ROOT:-${HOME}/compressors/nvcomp}"
CUDA_ARCH="${CUDA_ARCH:-90}"

if [[ ! -f "${NVCOMP_ROOT}/include/nvcomp/zstd.hpp" ]]; then
    echo "error: no nvCOMP SDK at NVCOMP_ROOT=${NVCOMP_ROOT}" >&2
    echo "       (expected include/nvcomp/zstd.hpp). Set NVCOMP_ROOT." >&2
    exit 1
fi

# Prefer an explicitly-set CUDACXX; otherwise find a CUDA >= 12 nvcc, skipping the
# nvhpc compilers/bin wrapper that redirects to the bundled CUDA 11.8.
if [[ -z "${CUDACXX:-}" ]]; then
    for cand in $(compgen -c nvcc 2>/dev/null | sort -u) $(command -v nvcc || true); do
        p="$(command -v "${cand}" 2>/dev/null || true)"
        [[ -n "${p}" && "${p}" != *"/compilers/bin/nvcc" ]] || continue
        ver="$("${p}" --version 2>/dev/null | sed -n 's/.*release \([0-9]*\)\..*/\1/p')"
        if [[ -n "${ver}" && "${ver}" -ge 12 ]]; then CUDACXX="${p}"; break; fi
    done
fi
if [[ -z "${CUDACXX:-}" ]]; then
    echo "error: no CUDA >= 12 nvcc found. Source your site env script " >&2
    echo "       (e.g. scripts/env-jetstream2.sh) or set CUDACXX explicitly." >&2
    exit 1
fi

echo "nvCOMP:  ${NVCOMP_ROOT}"
echo "nvcc:    ${CUDACXX}  ($("${CUDACXX}" --version | tail -2 | head -1))"
echo "host cc: $(command -v g++)"
echo "arch:    sm_${CUDA_ARCH}"

rm -rf "${BUILD}"
cmake -S "${SRC}" -B "${BUILD}" \
      -DNVCOMP_ROOT="${NVCOMP_ROOT}" \
      -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCH}" \
      -DCMAKE_CUDA_COMPILER="${CUDACXX}" \
      -DCMAKE_CXX_COMPILER="$(command -v g++)"
cmake --build "${BUILD}" -j"$(nproc)"

echo
echo "built: ${BUILD}/nvcomp_cli"
echo "export NVCOMP_CLI=\"${BUILD}/nvcomp_cli\""
echo "export NVCOMP_ROOT=\"${NVCOMP_ROOT}\""
