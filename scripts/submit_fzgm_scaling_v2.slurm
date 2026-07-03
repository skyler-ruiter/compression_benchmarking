#!/bin/bash
# sbatch script for fzgm_scaling_v2: fzgpu_minimal + PFPL (PREALLOCATE + MINIMAL)
#
# Continues from fzgm_scaling session (job 7519560, 2026-06-30).
# Results land in $SCRATCH/benchkit-results/$SLURM_JOB_ID/
#
# Submit:
#   sbatch scripts/submit_fzgm_scaling_v2.slurm
#
# After job completes, merge shards (if any) and print report:
#   python -m benchkit report $SCRATCH/benchkit-results/<JOB_ID>
#
#SBATCH --job-name=bkit-scaling-v2
#SBATCH --gpus=1
#SBATCH --time=04:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --partition=gpu
#SBATCH --exclusive
#SBATCH --account=r01156

set -euo pipefail

EXPERIMENT=configs/experiments/fzgm_scaling_v2.yaml
export BENCHKIT_RESULTS_ROOT="${SCRATCH:-/N/scratch/sruiter}/benchkit-results"

cd "$SLURM_SUBMIT_DIR"

source scripts/env-bigred200.sh

# Sanity: confirm we're on a GPU node and print what we got
nvidia-smi --query-gpu=name,memory.total,driver_version \
           --format=csv,noheader | sed 's/^/[gpu] /'

echo "[benchkit] session = $SLURM_JOB_ID"
echo "[benchkit] results = $BENCHKIT_RESULTS_ROOT/$SLURM_JOB_ID"
echo "[benchkit] experiment = $EXPERIMENT"
echo "[benchkit] fzgmod-cli = $FZGMOD_CLI"

python -m benchkit run "$EXPERIMENT" --session-id "$SLURM_JOB_ID"

echo "[benchkit] run complete — printing report"
python -m benchkit report "$BENCHKIT_RESULTS_ROOT/$SLURM_JOB_ID"
