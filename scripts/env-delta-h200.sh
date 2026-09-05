#!/usr/bin/env bash
# NCSA Delta gpuH200x8 (NVIDIA H200, sm_90) environment for compression benchmarking.
# Source this file — do not execute it:
#   source scripts/env-delta-h200.sh    (from repo root)
#   .      scripts/env-delta-h200.sh    (POSIX)
#
# Idempotent: module swap is a no-op if already applied; venv activation is skipped
# if already active. Safe to re-source from a running session.
#
# Get an interactive node first:
#   salloc --account=bdqz-delta-gpu --partition=gpuH200x8-interactive \
#     --nodes=1 --gpus-per-node=1 --time=00:30:00
#
# Adding a new compressor:
#   1. Uncomment / add its PATH and CLI export below.
#   2. Add a matching entry to benchkit/adapters/__init__.py.
#   3. Re-source and run the smoke test.

# ── core modules ─────────────────────────────────────────────────────────────
# Delta's default toolchain moved to cudatoolkit/26.5_13.2 (nvcc 13.2) and
# gcc-native/14 (it was 25.3_12.8 / gcc-native 13 when the FZGM cuda-h200 build
# and the 2026-07 baselines were made). Both support sm_90.
# craype-accel-nvidia80 is the default Cray accel target (only matters if
# something links via the Cray CC/cc wrapper — fzgmod-cli does not); swap to
# nvidia90 anyway for consistency with what the FZGM cuda-h200 build used.
#
# Guarded, not a bare `module swap`: Lmod's swap requires the FROM module to be
# loaded, so re-sourcing this script in a shell that already swapped once (e.g.
# a driver script that sources env-<site>.sh internally, invoked from a job
# that already sourced it) makes the bare swap fail with "craype-accel-nvidia80
# is not loaded" and, under `set -e`, kill the caller. Confirmed 2026-09-02:
# run-specialization-smoke.sh dies on this exact line when double-sourced.
if ! module is-loaded craype-accel-nvidia90 2>/dev/null; then
    module swap craype-accel-nvidia80 craype-accel-nvidia90 2>&1
fi

# CUDA 12.9, not Delta's 13.2 default. CUDA 13 moved Thrust/CUB under
# include/cccl/, so any host .cc that does `#include <thrust/...>` no longer
# resolves — cuSZ-Hi fails to build outright (utils/analyzer.hh). Rather than
# patch include paths into every recipe, pin the toolkit the manifest recipes
# were validated against; 12.9 is also what the JetStream2 reference builds
# used, which keeps this machine's numbers comparable to those baselines.
# NCSA labels the module "-unsupported"; that is about NCSA support, not
# correctness. Revisit only when the recipes are ported to CCCL.
# Same double-source guard as above.
if ! module is-loaded cudatoolkit/26.5_12.9-unsupported 2>/dev/null; then
    module swap cudatoolkit/26.5_13.2 cudatoolkit/26.5_12.9-unsupported 2>&1
fi
hash -r 2>/dev/null || true

# ── reference-compressor toolchain (compressors/bootstrap.py) ───────────────
# Consumed by compressors/build/_common.sh. CUDA_ARCH is the one that must be
# set explicitly — without it everything targets manifest.toml's default (90,
# which happens to be right here, but do not rely on that).
export CUDA_ARCH=90
export COMPRESSORS_ROOT="${HOME}/compressors"
export CUDA_ROOT="${CUDA_HOME:-/opt/nvidia/hpc_sdk/Linux_x86_64/26.5/cuda/12.9}"

# ── data & results paths ─────────────────────────────────────────────────────
# No $SCRATCH env var on Delta; project scratch is /scratch/bdqz/$USER. Shared
# with the MI100 env script — same filesystem, same account.
export BENCHKIT_DATA_ROOT="/scratch/bdqz/${USER}/sdrbench_data"
export BENCHKIT_RESULTS_ROOT="/scratch/bdqz/${USER}/benchkit-results"
export BENCHKIT_GPU_ARCH="sm_90"
export FZGMOD_BACKEND="CUDA"
# CUDA_ARCH is exported above (compressors/bootstrap.py toolchain block).

# ── FZGM ─────────────────────────────────────────────────────────────────────
# CUDA (sm_90) Release build via the FZGPUModules 'cuda-h200' CMake preset,
# verified on hardware: ctest 58/58 on gpuH200x8 (2026-08-31, CUDA 12.9).
#
# To re-verify, BUILD_TESTING must be passed explicitly — it defaults to OFF
# (FZGPUModules/CMakeLists.txt) and the cuda-h200 preset does not set it, so a
# plain `ctest --test-dir build/cuda-h200` prints "No tests were found!!!" and
# still exits 0. That looks like a pass and verifies nothing:
#   cmake --preset cuda-h200 -DBUILD_TESTING=ON && cmake --build build/cuda-h200 -j16
#   ctest --test-dir build/cuda-h200 --output-on-failure   # on a GPU node
export FZGMOD_CLI="${HOME}/FZGPUModules/build/cuda-h200/bin/fzgmod-cli"

# ── reference compressors ────────────────────────────────────────────────────
# All built by `python compressors/bootstrap.py` into ${COMPRESSORS_ROOT} at the
# paths pinned in compressors/manifest.toml. Unlike the MI100 (where the CUDA-only
# references cannot exist at all), the H200 is a full FZGM-vs-native machine.
# Rebuild any one of them with `python compressors/bootstrap.py build <name>`.

# error-bounded lossy, GPU
export CUSZ_CLI="${COMPRESSORS_ROOT}/cuSZ/build/cusz"
export CUSZHI_CLI="${COMPRESSORS_ROOT}/cuSZ-Hi/build/cuszhi"
export CUSZP2_CLI="${COMPRESSORS_ROOT}/cuSZp-V2.0.1/build/examples/bin/cuSZp"
export CUSZP3_CLI="${COMPRESSORS_ROOT}/cuSZp-V3.0.0/build/examples/bin/cuSZp"
export FZGPU_CLI="${COMPRESSORS_ROOT}/FZ-GPU/fz-gpu"
export PFPL_BIN_DIR="${COMPRESSORS_ROOT}/PFPL/bin"          # dir, not a binary
export FSZ_CLI="${COMPRESSORS_ROOT}/FSZ/build/fsz"

# error-bounded lossy, CPU (CR/quality baselines, not throughput peers)
export SZ3_CLI="${COMPRESSORS_ROOT}/SZ3/build/tools/sz3/sz3"
export ZFP_CLI="${COMPRESSORS_ROOT}/zfp/build/bin/zfp"
export SPERR_BIN_DIR="${COMPRESSORS_ROOT}/SPERR/build/bin"  # sperr2d, sperr3d

# MGARD needs its own bundled nvcomp/protobuf/zstd on LD_LIBRARY_PATH at runtime.
# BOTH lib and lib64 are required here: on this RHEL9 box MGARD's own
# libmgard.so.1 installs to lib64/, while the vendored deps land in lib/.
# (On the Ubuntu JetStream2 node everything is in lib/, so env-jetstream2.sh
# adds only that — copying its single path here yields exit 127,
# "libmgard.so.1: cannot open shared object file".)
export MGARD_CLI="${COMPRESSORS_ROOT}/MGARD/install-cuda-hopper/bin/mgard-x"
export LD_LIBRARY_PATH="${COMPRESSORS_ROOT}/MGARD/install-cuda-hopper/lib64:${COMPRESSORS_ROOT}/MGARD/install-cuda-hopper/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

# quantized-integer backends (float wrappers — see docs/adapters/{mans,lscomp}.md)
export MANS_CLI="${COMPRESSORS_ROOT}/MANS/build/bin/cpu/cpu_mans_compress"
export MANS_DECOMPRESS_CLI="${COMPRESSORS_ROOT}/MANS/build/bin/cpu/cpu_mans_decompress"
export LSCOMP_CLI="${COMPRESSORS_ROOT}/lsCOMP/build/lsCOMP_uint32"
export LSCOMP_UINT16_CLI="${COMPRESSORS_ROOT}/lsCOMP/build/lsCOMP_uint16"

# ── repo-built CLIs ──────────────────────────────────────────────────────────
# These three live in this repo, not in ${COMPRESSORS_ROOT}: the vendor binaries
# either don't exist (nvCOMP ships a library only) or don't report device time.
#   ./scripts/build-nvcomp-cli.sh ./scripts/build-fsz-hosttime.sh ./scripts/build-lscomp-decode.sh
_DELTA_H200_REPO="${HOME}/compression_benchmarking"
export NVCOMP_ROOT="${COMPRESSORS_ROOT}/nvcomp"             # symlink -> pinned 5.3.0.16
export NVCOMP_CLI="${_DELTA_H200_REPO}/tools/nvcomp_cli/build/nvcomp_cli"
export FSZ_HOSTTIME_CLI="${_DELTA_H200_REPO}/tools/fsz_hosttime/fsz_hosttime"
export LSCOMP_DECODE_CLI="${_DELTA_H200_REPO}/tools/lscomp_decode/lscomp_decode"
unset _DELTA_H200_REPO

# ── Python venv ──────────────────────────────────────────────────────────────
# Resolve relative to this script's own location so it works regardless of cwd
# (login shell, srun/salloc interactive session, sbatch job).
_DELTA_H200_SCRIPTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -z "${VIRTUAL_ENV:-}" ]]; then
    module load python/3.13.5-gcc13.3.1 2>&1
    # shellcheck source=/dev/null
    source "${_DELTA_H200_SCRIPTS_DIR}/../.venv/bin/activate"
fi
unset _DELTA_H200_SCRIPTS_DIR
