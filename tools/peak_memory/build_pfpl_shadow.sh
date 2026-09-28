#!/usr/bin/env bash
# Shadow rebuild of the pinned PFPL tree with a *dynamically* linked cudart, so the
# LD_PRELOAD probe (tools/cuda_mem_probe) can see its allocations. PFPL's makefile
# links cudart statically (nvcc's default), which makes interposition a silent no-op.
#
# Same pinned commit + same local makefile patch as compressors/build/pfpl.sh; the
# ONLY change is `-cudart shared` added to each nvcc line of the `gpu:` target. No
# source change. NVML (the primary metric) does not need this rebuild; it is used so
# NVML and the probe can observe the same process.
#
#   usage: build_pfpl_shadow.sh [PFPL_SRC] [OUT_DIR]
#   default PFPL_SRC = ~/compressors/PFPL (must be at 36f5aaef)
#   default OUT_DIR  = ~/compressors-shadow/pfpl-cudart-shared
#   then:  export PEAKMEM_PFPL_SHADOW_BIN_DIR=$OUT_DIR/bin
set -euo pipefail
SRC="${1:-$HOME/compressors/PFPL}"
OUT="${2:-$HOME/compressors-shadow/pfpl-cudart-shared}"
PIN=36f5aaef42744ed78b4d1525b03a7fb2168b363c
ARCH="${CUDA_ARCH:-90}"; ARCH="${ARCH%a}"

head=$(git -C "$SRC" rev-parse HEAD)
[[ "$head" == "$PIN" ]] || { echo "PFPL at $head, expected $PIN" >&2; exit 1; }

rm -rf "$OUT"; mkdir -p "$OUT"
cp -r "$SRC"/src "$SRC"/makefile "$OUT"/
mkdir -p "$OUT"/bin/f32/gpu "$OUT"/bin/f64/gpu
# Add -cudart shared to nvcc invocations inside the gpu: target only.
awk '/^gpu:/{g=1} /^[A-Za-z_]+:/{if($0!~/^gpu:/)g=0} {if(g && $1=="nvcc") sub(/nvcc /,"nvcc -cudart shared "); print}' \
    "$SRC"/makefile > "$OUT"/makefile
diff "$SRC"/makefile "$OUT"/makefile > "$OUT"/makefile.shadow.diff || true
(cd "$OUT" && make gpu NV_SM="$ARCH")

for b in "$OUT"/bin/f*/gpu/*_cuda; do
    ldd "$b" | grep -q libcudart || { echo "not dynamic: $b" >&2; exit 1; }
done
echo "shadow PFPL built at $OUT/bin ($(ls "$OUT"/bin/f*/gpu | grep -c _cuda) binaries, cudart shared)"
