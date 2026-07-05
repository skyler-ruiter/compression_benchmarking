#!/bin/bash
# One-off resume of session 7543911 (fzgm_vs_native) if the interactive
# allocation it started under is walltime-killed mid-run. Resume is idempotent
# (cell_key scan, DESIGN.md D13) so this only runs the remaining cells.
#SBATCH --job-name=bkit-fzgm-resume
#SBATCH --gpus=1
#SBATCH --time=01:30:00
#SBATCH --output=slurm-resume-%j.out
#SBATCH --partition=gpu
#SBATCH --exclusive
#SBATCH --account=r01156

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source scripts/env-bigred200.sh
export BENCHKIT_RESULTS_ROOT="${SCRATCH:-/N/scratch/sruiter}/benchkit-results"

python -m benchkit run configs/experiments/fzgm_vs_native.yaml --session-id 7543911

echo "[benchkit] run complete — printing reports"
python -m benchkit report "$BENCHKIT_RESULTS_ROOT/7543911"
echo
python -m benchkit report --aggregate "$BENCHKIT_RESULTS_ROOT/7543911"
