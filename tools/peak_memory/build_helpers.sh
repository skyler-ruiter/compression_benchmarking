#!/usr/bin/env bash
# Build the peak-memory study's helper binaries (all gitignored build outputs).
#   ctx_baseline            empty-context NVML floor
#   lmem_probe/lmem_<N>     local-memory reservation microbenchmark (N-byte stack frame)
#   lmem_probe/libstacklimit.so  LD_PRELOAD shim reporting the max cudaLimitStackSize
# Requires nvcc on PATH (source scripts/env-<machine>.sh). CUDA_ARCH defaults to 90.
set -euo pipefail
cd "$(dirname "$0")"
ARCH="${CUDA_ARCH:-90}"; ARCH="${ARCH%a}"
nvcc -O2 -cudart shared -arch=sm_"$ARCH" ctx_baseline.cu -o ctx_baseline
for s in 0 1024 4352 8480; do
    nvcc -O3 -cudart shared -arch=sm_"$ARCH" -DSTACK_BYTES=$s lmem_probe/lmem_probe.cu -o lmem_probe/lmem_$s
done
gcc -O2 -shared -fPIC lmem_probe/stacklimit_shim.c -o lmem_probe/libstacklimit.so -ldl
echo "built: ctx_baseline lmem_probe/lmem_{0,1024,4352,8480} lmem_probe/libstacklimit.so"
