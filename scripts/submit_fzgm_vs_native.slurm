#!/bin/bash
# sbatch script for fzgm_vs_native — the FZGM-vs-original validation matrix.
#
# --exclusive is the point of this script: prior sessions of this experiment ran on a
# node shared with other users' jobs, which corrupted native FZ-GPU's wall-clock
# compress timing (its adapter isn't CUDA-event-isolated, so it picks up contention —
# see docs/experiments/fzgpumodules_bugreport_2026-07-02.md and the artifact's "env"
# ledger entry). CUDA-event-timed adapters (cusz/cuszhi/cuszp2/cuszp3/pfpl, all fzgm
# cells) were largely unaffected either way, but exclusive scheduling removes the
# caveat entirely instead of footnoting around it.
#
# Single task, no sharding: 210 cells (70 pipeline x dataset-field combos x 3 error
# bounds) x 5 reps is a few dozen minutes of GPU time, well within one task. Results
# land in $SCRATCH/benchkit-results/$SLURM_JOB_ID/.
#
# Submit:
#   sbatch scripts/submit_fzgm_vs_native.slurm
#
# After the job completes:
#   python -m benchkit report           $SCRATCH/benchkit-results/<JOB_ID>
#   python -m benchkit report --aggregate $SCRATCH/benchkit-results/<JOB_ID>
#
#SBATCH --job-name=bkit-fzgm-vs-native
#SBATCH --gpus=1
#SBATCH --time=04:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --partition=gpu
#SBATCH --exclusive
#SBATCH --account=r01156

set -euo pipefail

EXPERIMENT=configs/experiments/fzgm_vs_native.yaml
export BENCHKIT_RESULTS_ROOT="${SCRATCH:-/N/scratch/sruiter}/benchkit-results"

cd "$SLURM_SUBMIT_DIR"

source scripts/env-bigred200.sh

# Sanity: confirm we're on an idle, exclusively-held GPU node.
nvidia-smi --query-gpu=name,memory.total,driver_version \
           --format=csv,noheader | sed 's/^/[gpu] /'
squeue --node="$SLURMD_NODENAME" -h -o '%j %u' | sed 's/^/[node-jobs] /' || true

echo "[benchkit] session = $SLURM_JOB_ID"
echo "[benchkit] results = $BENCHKIT_RESULTS_ROOT/$SLURM_JOB_ID"
echo "[benchkit] experiment = $EXPERIMENT"
echo "[benchkit] fzgmod-cli = $FZGMOD_CLI"

python -m benchkit run "$EXPERIMENT" --session-id "$SLURM_JOB_ID"

echo "[benchkit] run complete — printing reports"
python -m benchkit report "$BENCHKIT_RESULTS_ROOT/$SLURM_JOB_ID"
echo
python -m benchkit report --aggregate "$BENCHKIT_RESULTS_ROOT/$SLURM_JOB_ID"
