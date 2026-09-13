#!/usr/bin/env bash
# Pipeline Specialization vs. native — full-corpus campaign, 4 GPUs on ONE node.
#
# Runs 4 concurrent shards of specialization_vs_native_full.yaml, one per physical
# GPU (pinned via CUDA_VISIBLE_DEVICES — never two shards sharing a GPU, D15), all
# writing into the SAME session id so they merge cleanly afterward. Each shard's
# own invocation still loops off -> auto sequentially (matching
# run-specialization-full.sh); the 4 shards run those two phases in parallel with
# each other, not necessarily in lockstep (fine — shard identity is independent).
#
#   scripts/run-specialization-full-4gpu.sh <host-tag> [<site-env-script>]
#
# Requires the allocation to already expose 4 GPUs (salloc/sbatch --gpus=4 or
# equivalent on a single node) — this script does not request the allocation.
#
# Env overrides:
#   SPEC_NGPU=<n>       number of GPUs to use from this allocation (default 4)
#   SPEC_DATE_TAG=...   pin the session date tag explicitly (needed for a resume
#                       that crosses midnight, or to match a specific job id)
set -euo pipefail

host_tag="${1:?usage: run-specialization-full-4gpu.sh <host-tag> [<site-env-script>]}"
site_env="${2:-scripts/env-jetstream2.sh}"
n_gpu="${SPEC_NGPU:-4}"
date_tag="${SPEC_DATE_TAG:-$(date +%Y%m%d)}"

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repo_dir}"

# Clock locking is managed HERE, once, node-wide -- NOT by each shard's own
# run-specialization-full.sh (SPEC_SKIP_CLOCK_MGMT tells it to skip its internal
# lock/EXIT-trap-unlock). nvidia-smi -lgc/-rgc are not scoped by
# CUDA_VISIBLE_DEVICES; 4 independent lock+unlock-on-exit shards would race
# (whichever finishes first unlocks the other three mid-sweep).
export SPEC_SKIP_CLOCK_MGMT=1
if command -v nvidia-smi >/dev/null && [ -w /dev/nvidia0 ] 2>/dev/null; then
    bash scripts/lock_clocks.sh || true
    trap 'bash scripts/unlock_clocks.sh || true' EXIT
else
    echo "  (no writable /dev/nvidia0 -- skipping clock lock; rely on --exclusive for steady clocks, D15)"
fi

echo "=== launching ${n_gpu} concurrent shards on ${host_tag}, session date tag ${date_tag} ==="
pids=()
for ((k=0; k<n_gpu; k++)); do
  logfile="/tmp/spec-full-4gpu-${host_tag}-shard${k}-${date_tag}.log"
  echo "  shard ${k}/${n_gpu} -> GPU ${k} -> ${logfile}"
  (
    export CUDA_VISIBLE_DEVICES="${k}"
    export SPEC_SHARD="${k}/${n_gpu}"
    export SPEC_DATE_TAG="${date_tag}"
    bash scripts/run-specialization-full.sh "${host_tag}" "${site_env}"
  ) > "${logfile}" 2>&1 &
  pids+=($!)
done

echo "=== waiting on ${#pids[@]} shard processes (PIDs: ${pids[*]}) ==="
fail=0
for pid in "${pids[@]}"; do
  wait "${pid}" || { echo "shard PID ${pid} exited non-zero"; fail=1; }
done
[ "${fail}" -eq 0 ] || { echo "one or more shards failed — inspect /tmp/spec-full-4gpu-${host_tag}-shard*-${date_tag}.log before merging"; exit 1; }

echo "=== all shards finished — merging ==="
source "${site_env}"
base="specialization-vs-native-full-${host_tag}"
for pol in off auto; do
  sid="${base}-${pol}-${date_tag}"
  python -m benchkit merge "${BENCHKIT_RESULTS_ROOT}/${sid}"
  python -m benchkit verify "${BENCHKIT_RESULTS_ROOT}/${sid}" || \
    echo "WARN: verify non-zero for ${sid} — inspect the audit (native eb misses + HPC cv expected, CR drift off<->auto is not)"
  python -m benchkit report "${BENCHKIT_RESULTS_ROOT}/${sid}" --aggregate --by-dataset
done

echo "Done. Compare the pair:"
echo "  python -m benchkit report \$BENCHKIT_RESULTS_ROOT/${base}-auto-${date_tag} --aggregate --by-dataset"
