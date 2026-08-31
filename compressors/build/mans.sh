#!/usr/bin/env bash
# MANS. cpu_nv target -> CPU CLIs at build/bin/cpu/. HDF5 plugin off.
. "$(dirname "$0")/_run.sh"
rm -rf build && cmake -S . -B build $(cuda_cmake_args) -DTARGET_PLATFORM=cpu_nv -DBUILD_HDF5_PLUGIN=OFF
cmake --build build -j"$JOBS" --target cpu_mans_compress cpu_mans_decompress
