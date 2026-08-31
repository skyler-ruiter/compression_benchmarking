#!/usr/bin/env bash
# cuSZ-Hi. Release, examples OFF (googletest submodule not needed).
. "$(dirname "$0")/_run.sh"
rm -rf build && cmake -S . -B build $(cuda_cmake_args) -DPSZ_BACKEND=cuda -DPSZ_BUILD_EXAMPLES=off
cmake --build build -j"$JOBS"
