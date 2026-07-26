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
# cudatoolkit/25.3_12.8 (nvcc 12.8) is already loaded by default on Delta and
# supports sm_90. craype-accel-nvidia80 is the default Cray accel target
# (only matters if something links via the Cray CC/cc wrapper — fzgmod-cli
# does not); swap to nvidia90 anyway for consistency with what the FZGM
# cuda-h200 build used.
module swap craype-accel-nvidia80 craype-accel-nvidia90 2>&1

# ── data & results paths ─────────────────────────────────────────────────────
# No $SCRATCH env var on Delta; project scratch is /scratch/bdqz/$USER. Shared
# with the MI100 env script — same filesystem, same account.
export BENCHKIT_DATA_ROOT="/scratch/bdqz/${USER}/sdrbench_data"
export BENCHKIT_RESULTS_ROOT="/scratch/bdqz/${USER}/benchkit-results"

# ── FZGM ─────────────────────────────────────────────────────────────────────
# CUDA (sm_90) Release build via the FZGPUModules 'cuda-h200' CMake preset,
# verified on hardware: ctest 40/40 on gpuH200x8 (2026-07-24).
export FZGMOD_CLI="${HOME}/FZGPUModules/build/cuda-h200/bin/fzgmod-cli"

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
