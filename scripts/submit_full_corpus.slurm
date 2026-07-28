#!/bin/bash
# SLURM job ARRAY template for the full-corpus sweep.
#
# Unlike scripts/submit_fzgm_vs_native.slurm (single task, 210 cells, ~1 h) this runs
# ~9.8k cells and must be sharded: each array task takes cells where
# index % N == SLURM_ARRAY_TASK_ID, writes its own runs.shard-k-of-N.jsonl (so tasks
# never contend on a single file), and captures its own provenance (array tasks may
# land on different nodes).
#
# RESUME is content-keyed on cell_key, so a task killed by the wall limit loses only
# the cell in flight: resubmit the SAME --session-id and completed cells are skipped.
# That is what makes a short queue limit workable — requeue until `merge` reports the
# expected total. You may even change N between submissions.
#
# ---------------------------------------------------------------------------
# Pick ONE site block below, then:
#   sbatch --array=0-7 scripts/submit_full_corpus.slurm
#   # ... repeat the same sbatch until no task reports remaining cells ...
#   python -m benchkit merge  $BENCHKIT_RESULTS_ROOT/<session>/
#   python -m benchkit report $BENCHKIT_RESULTS_ROOT/<session>/ --aggregate --by-dataset
#
# SESSION_ID must be stable across resubmissions — do NOT use $SLURM_ARRAY_JOB_ID,
# which changes every sbatch. Set it explicitly (below) or pass it in the environment.
# ---------------------------------------------------------------------------

#SBATCH --job-name=bkit-fullcorpus
#SBATCH --gpus=1
#SBATCH --time=08:00:00
#SBATCH --output=slurm-%A_%a.out
#SBATCH --exclusive

# --- BigRed200 (A100, natives ARE built here -> full vs-native matrix) -------
#SBATCH --partition=gpu
#SBATCH --account=r01156
SITE_ENV=scripts/env-bigred200.sh
EXPERIMENT=configs/experiments/fzgm_vs_native_full.yaml

# --- NCSA Delta H200 (natives NOT built -> FZGM-only) ------------------------
# #SBATCH --partition=gpuH200x8
# #SBATCH --account=bdqz-delta-gpu
# SITE_ENV=scripts/env-delta-h200.sh
# EXPERIMENT=configs/experiments/fzgm_only_full.yaml

# --- NCSA Delta MI100 (AMD/HIP, natives IMPOSSIBLE -> FZGM-only) -------------
# #SBATCH --partition=gpuMI100x8
# #SBATCH --account=bdqz-delta-gpu
# SITE_ENV=scripts/env-delta-mi100.sh
# EXPERIMENT=configs/experiments/fzgm_only_full.yaml

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
# shellcheck source=/dev/null
source "$SITE_ENV"

# Stable across resubmissions. Change the tag per machine, not per submission.
SESSION_ID="${SESSION_ID:-fullcorpus-${SLURM_JOB_PARTITION:-site}}"

N=${SLURM_ARRAY_TASK_COUNT:-1}
K=${SLURM_ARRAY_TASK_ID:-0}

echo "[submit] session=$SESSION_ID shard=$K/$N experiment=$EXPERIMENT"
echo "[submit] node=$(hostname) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || rocm-smi --showproductname 2>/dev/null | head -1)"

# --exclusive is load-bearing, not cosmetic: clock control is admin-only on both
# clusters (D15), so single-tenancy is the only lever on timing variance. Trust the
# per-row timing_reliable flag and prefer *_device_ms_min when it is false.
python -m benchkit run "$EXPERIMENT" \
    --session-id "$SESSION_ID" \
    --shard "$K/$N"
