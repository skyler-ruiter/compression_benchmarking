#!/usr/bin/env bash
# FSZ. Release; tools on (the `fsz` CLI). Tests/fortran off for portability.
. "$(dirname "$0")/_run.sh"
rm -rf build && cmake -S . -B build $(cuda_cmake_args) -DFSZ_BUILD_TOOLS=ON -DFSZ_BUILD_TESTS=OFF -DFSZ_BUILD_FORTRAN=OFF -DFSZ_BUILD_EXAMPLES=OFF
cmake --build build -j"$JOBS"
