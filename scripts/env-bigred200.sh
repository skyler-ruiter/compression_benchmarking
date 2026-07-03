#!/usr/bin/env bash
# BigRed200 environment for compression benchmarking.
# Source this file — do not execute it:
#   source scripts/env-bigred200.sh    (from repo root)
#   .      scripts/env-bigred200.sh    (POSIX)
#
# Idempotent: module loads are no-ops if already loaded; venv activation is
# skipped if already active.  Safe to re-source from a running session.
#
# Adding a new compressor:
#   1. Uncomment / add its PATH and CLI export below.
#   2. Add a matching entry to benchkit/adapters/__init__.py.
#   3. Re-source and run the smoke test.

# ── core modules ─────────────────────────────────────────────────────────────
module load python/3.12.11
module load cudatoolkit                # CUDA 12.x — required by all GPU compressors
module load gcc-native/12.3   # 13.2 has ICE on cuSZ's complex template code (hf_hl.cc)
# module load cray-hdf5/1.14.3.5 # MANS


# ── data & results paths ─────────────────────────────────────────────────────
# $SCRATCH is set by SLURM inside jobs; fall back to the explicit path for
# login-shell / interactive use (BigRed200: $SCRATCH == /N/scratch/$USER).
_BR_SCRATCH="${SCRATCH:-/N/scratch/sruiter}"
export BENCHKIT_DATA_ROOT="${_BR_SCRATCH}/sdrbench_data"
export BENCHKIT_RESULTS_ROOT="${_BR_SCRATCH}/benchkit-results"
unset _BR_SCRATCH

# ── FZGM ─────────────────────────────────────────────────────────────────────
export FZGMOD_CLI=/N/u/sruiter/BigRed200/research/FZGPUModules/build_bench/bin/fzgmod-cli

# ── cuSZ (reference) ─────────────────────────────────────────────────────────
# BUILD (login node is fine; gcc-native/12.3 required — 12/13 ICE on hf_hl.cc
#        fixed by patching: sed -i 's/const RMerge rm/const auto rm/g;
#        s/const SMerge sm/const auto sm/g' codec/hf/src/hf_hl.cc):
#   cd ~/research/compressors/cuSZ
#   git submodule update --init --recursive
#   # apply the auto patch if not already done (idempotent)
#   sed -i 's/const RMerge rm = opts\.rm;/const auto rm = opts.rm;/g
#            s/const SMerge sm = opts\.sm;/const auto sm = opts.sm;/g' \
#     codec/hf/src/hf_hl.cc
#   rm -rf build && mkdir build
#   cmake -S . -B build \
#     -DPSZ_BACKEND=cuda -DPSZ_BUILD_EXAMPLES=on \
#     -DCMAKE_CUDA_ARCHITECTURES="80;86" -DCMAKE_BUILD_TYPE=Release \
#     -DCMAKE_COLOR_DIAGNOSTICS=on \
#     -DCUDAToolkit_ROOT=/N/soft/sles15sp6/cuda/gnu/12.6 \
#     -DCMAKE_CUDA_COMPILER=/N/soft/sles15sp6/cuda/gnu/12.6/bin/nvcc \
#     -DCMAKE_CUDA_FLAGS="-I/N/soft/sles15sp6/cuda/gnu/12.6/extras/CUPTI/include" \
#     -DCMAKE_CXX_FLAGS="-I/N/soft/sles15sp6/cuda/gnu/12.6/extras/CUPTI/include"
#   cmake --build build -j8
#
export CUSZ_CLI=/N/u/sruiter/BigRed200/research/compressors/cuSZ/build/cusz
export PATH="/N/u/sruiter/BigRed200/research/compressors/cuSZ/build${PATH:+:$PATH}"

# ── cuSZp2 ────────────────────────────────────────────────────────────────────
# BUILD:
#   cmake -S . -B build -DCMAKE_CUDA_ARCHITECTURES="80" -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=$(which g++) -DCMAKE_C_COMPILER=$(which gcc)
#
export CUSZP2_CLI=/N/u/sruiter/BigRed200/research/compressors/cuSZp-V2.0.1/build/examples/bin/cuSZp

# ── cuSZp3 ────────────────────────────────────────────────────────────────────
# BUILD:
#   cmake -S . -B build -DCMAKE_CUDA_ARCHITECTURES="80" -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=$(which g++) -DCMAKE_C_COMPILER=$(which gcc)
#
export CUSZP3_CLI=/N/u/sruiter/BigRed200/research/compressors/cuSZp-V3.0.0/build/examples/bin/cuSZp

# ── cuSZ-Hi ──────────────────────────────────────────────────────────────────
# BUILD:
#   cd ~/research/compressors/cuSZ-Hi && mkdir -p build
#   cmake -S . -B build -DPSZ_BACKEND=cuda -DPSZ_BUILD_EXAMPLES=off \
#     -DCMAKE_CUDA_ARCHITECTURES="80" -DCMAKE_BUILD_TYPE=Release \
#     -DCMAKE_CXX_COMPILER=$(which g++) -DCMAKE_C_COMPILER=$(which gcc)
#   cmake --build build -j8
#
export CUSZHI_CLI=/N/u/sruiter/BigRed200/research/compressors/cuSZ-Hi/build/cuszhi

# ── SZ3 ──────────────────────────────────────────────────────────────────────
# BUILD:
#   cd ~/research/SZ3 && mkdir -p build && cd build
#   cmake .. -DCMAKE_BUILD_TYPE=Release
#   make -j8
#
# export SZ3_CLI=/N/u/sruiter/BigRed200/research/SZ3/build/bin/sz3
# export PATH="/N/u/sruiter/BigRed200/research/SZ3/build/bin${PATH:+:$PATH}"

# ── MANS ─────────────────────────────────────────────────────────────────────
# BUILD:
#   cmake -S . -B build -DTARGET_PLATFORM=cpu_nv \
#     -DCMAKE_CUDA_COMPILER="/N/soft/sles15sp6/cuda/gnu/12.6/bin/nvcc" \
#     -DBUILD_HDF5_PLUGIN=OFF -DCMAKE_CUDA_ARCHITECTURES=80 \
#     -DCMAKE_CXX_COMPILER=$(which g++) -DCMAKE_C_COMPILER=$(which gcc)
# NOTE: stub adapter — lossless int compressor; quantization wrapper needed.
#
export MANS_CLI=/N/u/sruiter/BigRed200/research/compressors/MANS/build/bin/nv/nv_mans_compress

# ── FZ-GPU ───────────────────────────────────────────────────────────────────
# NOTE: stub adapter — no file I/O in current binary; source changes required.
#       See docs/adapters/fzgpu.md.
#
export FZGPU_CLI=/N/u/sruiter/BigRed200/research/compressors/FZ-GPU/fz-gpu

# ── PFPL ─────────────────────────────────────────────────────────────────────
# Update SM in Makefile to match GPU architecture (80 for A100, 86 for H100)
# BUILD: make all
# PFPL_BIN_DIR points to the bin/ directory containing f32/gpu/, f64/gpu/, etc.
#
export PFPL_BIN_DIR=/N/u/sruiter/BigRed200/research/compressors/PFPL/bin


# ── Python venv ──────────────────────────────────────────────────────────────
# Resolve relative to this script's own location so it works regardless of cwd
# (login shell, srun interactive session, sbatch job).
_BR_SCRIPTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -z "${VIRTUAL_ENV:-}" ]]; then
    # shellcheck source=/dev/null
    source "${_BR_SCRIPTS_DIR}/../.venv/bin/activate"
fi
unset _BR_SCRIPTS_DIR
