#!/usr/bin/env bash
# NCSA Delta gpuMI100x8 (AMD Instinct MI100, gfx908) environment for compression
# benchmarking. Source this file — do not execute it:
#   source scripts/env-delta-mi100.sh    (from repo root)
#   .      scripts/env-delta-mi100.sh    (POSIX)
#
# Idempotent: module swaps are no-ops if already applied; venv activation is skipped
# if already active. Safe to re-source from a running session.
#
# Get an interactive node first:
#   salloc --account=bdqz-delta-gpu --partition=gpuMI100x8-interactive \
#     --nodes=1 --gpus-per-node=1 --time=00:30:00
#
# Adding a new compressor:
#   1. Uncomment / add its PATH and CLI export below.
#   2. Add a matching entry to benchkit/adapters/__init__.py.
#   3. Re-source and run the smoke test.

# ── GPU backend toolchain (ROCm / HIP) ───────────────────────────────────────
# Cray's CC wrapper auto-links cray-mpich's GPU transport layer, chosen from
# whichever craype-accel-* module is loaded at each invocation -- the default
# nvidia80 module pulls -lmpi_gtl_cuda into every C++ binary regardless of backend,
# which fails to load on this node with "libcuda.so.1: cannot open shared object
# file". The cudatoolkit module separately, unconditionally appends
# -lcupti/-lcudart/-lcuda to every link line via CRAY_CUDATOOLKIT_POST_LINK_OPTS.
# Both must go for a clean HIP link/run; module state doesn't persist across
# shells, so this must be sourced in every new shell (see ~/setup-amd.sh).
module swap craype-accel-nvidia80 craype-accel-amd-gfx908 2>&1
module unload cudatoolkit 2>&1

export PATH="/opt/rocm/bin${PATH:+:$PATH}"
export ROCM_PATH=/opt/rocm
export CMAKE_PREFIX_PATH="/opt/rocm${CMAKE_PREFIX_PATH:+:$CMAKE_PREFIX_PATH}"
# CUDA toolkit headers left on CPATH by module load history can shadow HIP's own
# headers/hipCUB's cub facade -- strip them.
export CPATH="$(echo "$CPATH" | tr ':' '\n' | grep -v "nvidia/hpc_sdk" | paste -sd: -)"

# ── data & results paths ─────────────────────────────────────────────────────
# No $SCRATCH env var on Delta; project scratch is /scratch/bdqz/$USER.
export BENCHKIT_DATA_ROOT="/scratch/bdqz/${USER}/sdrbench_data"
export BENCHKIT_RESULTS_ROOT="/scratch/bdqz/${USER}/benchkit-results"

# ── FZGM ─────────────────────────────────────────────────────────────────────
# HIP (gfx908) Release build, verified on hardware: full compress/decompress
# round-trip via fzgmod-cli on gpuMI100x8 (2026-07-24).
export FZGMOD_CLI="${HOME}/FZGPUModules/build/hip/bin/fzgmod-cli"

# ── Python venv ──────────────────────────────────────────────────────────────
# Resolve relative to this script's own location so it works regardless of cwd
# (login shell, srun/salloc interactive session, sbatch job).
_DELTA_MI100_SCRIPTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -z "${VIRTUAL_ENV:-}" ]]; then
    module load python/3.13.5-gcc13.3.1 2>&1
    # shellcheck source=/dev/null
    source "${_DELTA_MI100_SCRIPTS_DIR}/../.venv/bin/activate"
fi
unset _DELTA_MI100_SCRIPTS_DIR
