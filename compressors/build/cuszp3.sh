#!/usr/bin/env bash
# cuSZp v3 (also the _optimized vendored variant).
. "$(dirname "$0")/_run.sh"
rm -rf build && cmake -S . -B build $(cuda_cmake_args)
cmake --build build -j"$JOBS"
