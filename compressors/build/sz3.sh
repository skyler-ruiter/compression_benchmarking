#!/usr/bin/env bash
# SZ3. CPU-only, Release.
. "$(dirname "$0")/_run.sh"
rm -rf build && cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j"$JOBS"
