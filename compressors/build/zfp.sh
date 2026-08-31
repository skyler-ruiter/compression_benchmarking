#!/usr/bin/env bash
# zfp. Release, utilities on (the `zfp` CLI), CUDA backend on.
. "$(dirname "$0")/_run.sh"
rm -rf build && cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_UTILITIES=ON -DZFP_WITH_CUDA=ON -DBUILD_TESTING=OFF
cmake --build build -j"$JOBS"
