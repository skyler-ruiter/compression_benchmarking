#!/bin/bash
# SLURM job-array template for a benchkit experiment matrix.
#
# Each array task runs one shard of the matrix and writes its own
# runs.shard-k-of-N.jsonl + provenance.shard-k-of-N.json into a shared session dir
# (session id = the array job id). A dependent job merges the shards at the end.
#
# Edit the marked values, then:   sbatch scripts/submit.slurm
# Re-submitting the same array resumes — completed cells are skipped.
#
#SBATCH --job-name=benchkit
#SBATCH --array=0-7                  # <-- N shards: array indices 0..N-1
#SBATCH --gpus=1                     # one GPU per task
#SBATCH --time=02:00:00              # walltime per task (resume covers overruns)
#SBATCH --output=slurm-%A_%a.out     # %A=array job id, %a=task id
#SBATCH --partition=gpu            # <-- set your partition / account
#SBATCH --exclusive                # exclusive node => steadier GPU clocks
#SBATCH --account=r01156

set -euo pipefail

N=8                                  # <-- must equal the --array size above
EXPERIMENT=configs/experiments/sdrbench.yaml   # <-- your experiment
export BENCHKIT_RESULTS_ROOT="$SCRATCH/benchkit-results"   # <-- scratch, not the repo

# --- environment: make fzgmod-cli + CUDA available (edit for your site) ----------
module load python/3.12.11
module load cudatoolkit              # loads cudatoolkit/12.6 (default)
# Path to fzgmod-cli; set FZGMOD_CLI or point at your build directly:
export FZGMOD_CLI=/N/u/sruiter/BigRed200/research/FZGPUModules/build_bench/bin/fzgmod-cli
# Where the SDRBench data lives (downloaded by scripts/download-sdrbench.sh):
export BENCHKIT_DATA_ROOT="${SCRATCH}/sdrbench_data"

cd "$SLURM_SUBMIT_DIR"
source .venv/bin/activate
python -m benchkit run "$EXPERIMENT" \
    --session-id "$SLURM_ARRAY_JOB_ID" \
    --shard "${SLURM_ARRAY_TASK_ID}/${N}"

# --- merge the shards once the whole array finishes ------------------------------
# Submit this with a dependency so it runs after all tasks complete:
#   ARRAY_ID=$(sbatch --parsable scripts/submit.slurm)
#   sbatch --dependency=afterok:$ARRAY_ID --wrap \
#     "cd $SLURM_SUBMIT_DIR && source .venv/bin/activate && python -m benchkit merge $SCRATCH/benchkit-results/$ARRAY_ID"
