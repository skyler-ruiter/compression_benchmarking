#!/usr/bin/env bash
# lsCOMP. Release; builds the uint32 + uint16 integer CLIs.
. "$(dirname "$0")/_run.sh"
rm -rf build && cmake -S . -B build $(cuda_cmake_args)
cmake --build build -j"$JOBS" --target lsCOMP_uint32 lsCOMP_uint16
