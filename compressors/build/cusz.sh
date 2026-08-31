#!/usr/bin/env bash
# cuSZ (reference). Release, examples on (builds the `cusz` CLI).
. "$(dirname "$0")/_run.sh"
rm -rf build && cmake -S . -B build $(cuda_cmake_args) -DPSZ_BACKEND=cuda -DPSZ_BUILD_EXAMPLES=on
cmake --build build -j"$JOBS"
