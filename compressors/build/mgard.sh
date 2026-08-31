#!/usr/bin/env bash
# MGARD-X (CUDA). Adapted from the upstream build_scripts/ helper — it vendors
# its own nvcomp / zstd / protobuf into a private install prefix, then builds
# and installs MGARD against them.
#
#   install prefix:  <src>/install-cuda-<arch-name>
#   runtime:         that prefix's lib/ must be on LD_LIBRARY_PATH
#                    (scripts/env-<machine>.sh already does this).
. "$(dirname "$0")/_run.sh"

case "$CUDA_ARCH" in
  90|90a) arch_name=hopper ;;
  80)     arch_name=ampere ;;
  86)     arch_name=ampere86 ;;
  *)      arch_name="sm${CUDA_ARCH}" ;;
esac
build_dir="./build-cuda-${arch_name}"
install_dir="$(pwd)/install-cuda-${arch_name}"

export CC="${CC:-gcc}" CXX="${CXX:-g++}" CUDACXX="${CUDA_ROOT:+${CUDA_ROOT}/bin/}nvcc"
export LD_LIBRARY_PATH="${install_dir}/lib:${install_dir}/lib64:${LD_LIBRARY_PATH:-}"

clone_dep() { # url tag dest
  [ -d "$3" ] || git clone --depth 1 -b "$2" --recurse-submodules "$1" "$3"
}

banner "MGARD deps: nvcomp 2.2.0 / zstd 1.5.0 / protobuf 3.19.4"
clone_dep https://github.com/NVIDIA/nvcomp.git v2.2.0 "${build_dir}/nvcomp/src"
cmake -S "${build_dir}/nvcomp/src" -B "${build_dir}/nvcomp/build" -DCMAKE_INSTALL_PREFIX="${install_dir}"
cmake --build "${build_dir}/nvcomp/build" -j"$JOBS" && cmake --install "${build_dir}/nvcomp/build"

clone_dep https://github.com/facebook/zstd.git v1.5.0 "${build_dir}/zstd/src"
cmake -S "${build_dir}/zstd/src/build/cmake" -B "${build_dir}/zstd/build" \
  -DZSTD_MULTITHREAD_SUPPORT=ON -DCMAKE_INSTALL_LIBDIR=lib -DCMAKE_INSTALL_PREFIX="${install_dir}"
cmake --build "${build_dir}/zstd/build" -j"$JOBS" && cmake --install "${build_dir}/zstd/build"

clone_dep https://github.com/protocolbuffers/protobuf.git v3.19.4 "${build_dir}/protobuf/src"
cmake -S "${build_dir}/protobuf/src/cmake" -B "${build_dir}/protobuf/build" \
  -Dprotobuf_BUILD_SHARED_LIBS=ON -Dprotobuf_BUILD_TESTS=OFF -DCMAKE_INSTALL_PREFIX="${install_dir}"
cmake --build "${build_dir}/protobuf/build" -j"$JOBS" && cmake --install "${build_dir}/protobuf/build"

banner "MGARD-X (arch ${CUDA_ARCH})"
cmake -S . -B "${build_dir}/mgard" \
  -DCMAKE_PREFIX_PATH="${install_dir};${install_dir}/lib/cmake/zstd" \
  -DMGARD_ENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCH}" \
  -DMGARD_ENABLE_DOCS=OFF -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="${install_dir}"
cmake --build "${build_dir}/mgard" -j"$JOBS"
cmake --install "${build_dir}/mgard"

# Stable path for the manifest's `cli` field regardless of arch name.
ln -sfn "install-cuda-${arch_name}" "install-cuda-hopper"
