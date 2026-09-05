#!/usr/bin/env bash
# lsCOMP. Release; builds the uint32 + uint16 integer CLIs, plus liblsCOMP.so.
# The `lsCOMP` library target is required: benchkit's standalone decoder
# (scripts/build-lscomp-decode.sh -> tools/lscomp_decode) links -llsCOMP against
# build/. Without it that script fails with "cannot find -llsCOMP" and the
# lsCOMP adapter has no decode path.
. "$(dirname "$0")/_run.sh"
rm -rf build && cmake -S . -B build $(cuda_cmake_args)
cmake --build build -j"$JOBS" --target lsCOMP_uint32 lsCOMP_uint16 lsCOMP
