#!/usr/bin/env bash
# SPERR. CPU (OpenMP), Release, CLI utilities on.
. "$(dirname "$0")/_run.sh"
rm -rf build && cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DUSE_OMP=ON -DBUILD_CLI_UTILITIES=ON -DBUILD_UNIT_TESTS=OFF
cmake --build build -j"$JOBS"
