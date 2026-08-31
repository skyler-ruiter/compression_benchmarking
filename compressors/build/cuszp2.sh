#!/usr/bin/env bash
# cuSZp v2 (also the _optimized / _split vendored variants — same recipe).
. "$(dirname "$0")/_run.sh"
rm -rf build && cmake -S . -B build $(cuda_cmake_args)
cmake --build build -j"$JOBS" --target cuSZp
