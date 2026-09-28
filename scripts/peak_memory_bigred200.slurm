#!/bin/bash
# Bare-metal spot check for the RQ3 peak-memory study (docs/peak_memory_methodology.md).
#
# Question it answers: are the H100 study's fixed per-process costs (empty CUDA
# context, per-thread local-memory reservation) and hence the native/FZGM crossover a
# JetStream2 pass-through-VM artifact? BigRed200 A100 nodes are bare metal.
#
# Cells: cuSZp2-outlier and PFPL, on CESM-2D/CLDHGH (26 MB) and HACC/vx (1.12 GB);
# native vs FZGM planned (staged + Auto), plus the empty-context baseline. Same config
# and driver as the H100 run, filtered. 3 reps. Expected wall time: < 10 minutes.
#
# Prerequisites on BigRed200 (one-time, login node):
#   1. This repo at the commit that adds tools/peak_memory/ (git pull).
#   2. FZGM pinned build, a dedicated worktree (the job verifies FZGMOD_GIT_SHA):
#        git -C ~/research/FZGPUModules worktree add --detach \
#            ~/research/FZGPUModules-memory-d511ebc d511ebc
#        (source scripts/env-bigred200.sh; cd ~/research/FZGPUModules-memory-d511ebc &&
#         cmake -S . -B build_memory -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=80 \
#           -DBUILD_CLI=ON -DBUILD_EXAMPLES=OFF -DBUILD_PROFILING=OFF -DBUILD_TESTING=OFF \
#           -DBUILD_SHARED_LIBS=ON && cmake --build build_memory -j 16)
#   3. make -C tools/cuda_mem_probe   and   CUDA_ARCH=80 tools/peak_memory/build_helpers.sh
#   PFPL: the pinned static binaries are used as-is (NVML needs no shadow rebuild; the
#   probe/stack shim are blind to static cudart, so PFPL rows there carry NVML only).
#
#   sbatch scripts/peak_memory_bigred200.slurm
#
#SBATCH --job-name=peakmem-spot
#SBATCH --gpus=1
#SBATCH --time=00:30:00
#SBATCH --output=slurm-peakmem-%j.out
#SBATCH --partition=gpu
#SBATCH --exclusive
#SBATCH --account=r01156
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source scripts/env-bigred200.sh

FZGM_CLI="${PEAKMEM_FZGM_CLI:-/N/u/sruiter/BigRed200/research/FZGPUModules-memory-d511ebc/build_memory/bin/fzgmod-cli}"
export PEAKMEM_PFPL_SHADOW_BIN_DIR="$PFPL_BIN_DIR"   # static binaries; see header
export BENCHKIT_MACHINE_ID=br200-a100
export COMPRESSORS_ROOT=/N/u/sruiter/BigRed200/research/compressors   # native pin checks
OUT="$BENCHKIT_RESULTS_ROOT/peak_memory/peakmem-rq3-br200-a100-${SLURM_JOB_ID}"

nvidia-smi -q | grep -E "Product Name|Driver Version|Virtualization Mode|Persistence" || true

python tools/peak_memory/run_peak_memory.py run configs/peak_memory/rq3_h100.yaml \
    --out "$OUT" --fzgm-cli "$FZGM_CLI" \
    --only-fields CESM-2D/CLDHGH HACC/vx \
    --only-families cuszp2_outlier pfpl \
    --only-arms native baseline fzgm:planned:off fzgm:planned:auto \
    --max-wait-idle-s 60
echo "results: $OUT/cells.jsonl"
