#!/bin/bash
# Single-node, 8-GPU full-corpus sweep for NCSA Delta (gpuH200x8 / gpuMI100x8).
#
# WHY THIS EXISTS instead of scripts/submit_full_corpus.slurm's `--array=0-7`:
# Delta's geometry breaks that script's two premises.
#
#   1. MaxTime on both GPU partitions is 2-00:00:00, not 8 h. Sharding is not forced
#      by the wall clock here, so the requeue-until-done loop is unnecessary.
#   2. gpuMI100x8 is a SINGLE node (8 MI100s) and gpuH200x8 is only 8 nodes. An
#      `--array=0-7 --exclusive` submission therefore *serialises* on MI100 (each task
#      wants the whole node) and claims the entire partition on H200.
#
# So: one job, one node, `--exclusive`, and 8 concurrent benchkit processes each
# pinned to its own GPU. That still honours the rule in docs/running-the-full-corpus.md
# — sharding is across separate GPUs, never several processes onto one — while holding
# one node instead of eight. The node is exclusive, so the only shared resource is host
# CPU (96 cores on H200 / 128 on MI100, i.e. 12+ per shard) and the timed region is
# device kernel time regardless.
#
# Each shard writes its own runs.shard-k-of-8.jsonl, so they never contend on a file.
# RESUME is content-keyed on cell_key: if the job hits the wall limit, resubmit with the
# SAME SESSION_ID and finished cells are skipped.
#
# ---------------------------------------------------------------------------
# Usage — pick a site with SITE, everything else is derived:
#
#   sbatch --export=ALL,SITE=h200  scripts/submit_delta_multigpu.slurm
#   sbatch --export=ALL,SITE=mi100 scripts/submit_delta_multigpu.slurm
#
#   python -m benchkit merge  $BENCHKIT_RESULTS_ROOT/<session>/
#   python -m benchkit report $BENCHKIT_RESULTS_ROOT/<session>/ --aggregate --by-dataset
#
# SESSION_ID is derived from SITE and is stable across resubmissions by construction —
# do NOT substitute $SLURM_JOB_ID, which changes every sbatch and would defeat resume.
# ---------------------------------------------------------------------------

#SBATCH --job-name=bkit-delta-fullcorpus
#SBATCH --account=bdqz-delta-gpu
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --time=24:00:00
#SBATCH --output=slurm-%j.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

SITE="${SITE:?set SITE=h200 or SITE=mi100 via --export=ALL,SITE=...}"
NSHARDS="${NSHARDS:-8}"

case "$SITE" in
  h200)
    SITE_ENV=scripts/env-delta-h200.sh
    GPU_ENV_VAR=CUDA_VISIBLE_DEVICES
    ;;
  mi100)
    SITE_ENV=scripts/env-delta-mi100.sh
    # ROCR_VISIBLE_DEVICES is the mask the ROCm runtime and rocm-smi both honour
    # (benchkit/gpu.py relies on that for the throttle sampler); HIP_VISIBLE_DEVICES
    # alone would not be seen by the sampler.
    GPU_ENV_VAR=ROCR_VISIBLE_DEVICES
    ;;
  *) echo "unknown SITE=$SITE (expected h200 or mi100)" >&2; exit 2 ;;
esac

# FZGM-only on both Delta GPUs: the native references are not built on Delta, and on
# MI100 cannot be (cuSZ / cuSZ-Hi / cuSZp / FZ-GPU / PFPL are CUDA-only). Shared,
# unmodified between the two so rows line up 1:1.
EXPERIMENT="${EXPERIMENT:-configs/experiments/fzgm_only_full.yaml}"
SESSION_ID="${SESSION_ID:-fullcorpus-delta-${SITE}}"

# shellcheck source=/dev/null
source "$SITE_ENV"

echo "[submit] site=$SITE session=$SESSION_ID shards=$NSHARDS experiment=$EXPERIMENT"
echo "[submit] node=$(hostname) job=$SLURM_JOB_ID"
echo "[submit] fzgmod-cli=$FZGMOD_CLI"
echo "[submit] results=$BENCHKIT_RESULTS_ROOT"

# Build the list of device ordinals to pin shards to. On MI100 this is NOT simply
# 0..7: gpud01 carries 8 MI100s (gfx908) *and* one MI210 (gfx90a), and the MI210 sits
# at ordinal 6 — so a naive shard->ordinal mapping puts one shard on hardware the
# gfx908-only build has no code objects for, and every one of its cells dies with
# `invalid device function`. (This cost a full shard, 607 cells, on job 20559888.)
# Probe the arch and keep only devices matching the arch we built for.
DEVICES=()
if [[ "$SITE" == "mi100" ]]; then
    for i in $(seq 0 15); do
        arch=$(ROCR_VISIBLE_DEVICES=$i rocminfo 2>/dev/null | grep -m1 -oE 'gfx[0-9a-z]+' || true)
        [[ -z "$arch" ]] && continue
        if [[ "$arch" == "gfx908" ]]; then
            DEVICES+=("$i")
        else
            echo "[submit] skipping device $i (arch $arch != gfx908)"
        fi
    done
else
    for ((i = 0; i < NSHARDS; i++)); do DEVICES+=("$i"); done
fi

if (( ${#DEVICES[@]} < NSHARDS )); then
    echo "[submit] only ${#DEVICES[@]} usable GPUs for $NSHARDS shards; reducing" >&2
    NSHARDS=${#DEVICES[@]}
fi
echo "[submit] device map: shards 0..$((NSHARDS-1)) -> ordinals ${DEVICES[*]:0:$NSHARDS}"

pids=()
for ((k = 0; k < NSHARDS; k++)); do
    env "$GPU_ENV_VAR=${DEVICES[$k]}" \
        python -m benchkit run "$EXPERIMENT" \
            --session-id "$SESSION_ID" \
            --shard "$k/$NSHARDS" \
        > "shard-${SITE}-${k}-of-${NSHARDS}.${SLURM_JOB_ID}.log" 2>&1 &
    pids+=($!)
    echo "[submit] shard $k -> gpu ordinal ${DEVICES[$k]} (pid ${pids[-1]})"
done

# Wait on every shard and fail the job if any shard failed, rather than letting a
# silent partial sweep look like a clean one.
rc=0
for i in "${!pids[@]}"; do
    if ! wait "${pids[$i]}"; then
        echo "[submit] shard $i FAILED (see shard-${SITE}-${i}-of-${NSHARDS}.${SLURM_JOB_ID}.log)" >&2
        rc=1
    fi
done

echo "[submit] all shards done (rc=$rc)"
exit $rc
