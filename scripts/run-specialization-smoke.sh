#!/usr/bin/env bash
# Pipeline Specialization vs. native — cross-machine smoke driver.
#
# Runs both smoke experiments as an off/auto A/B (FZ_SPECIALIZE), verifies each
# session, and prints a combined report. This is the preflight before any
# full-corpus specialization campaign on a machine.
#
#   scripts/run-specialization-smoke.sh <host-tag> [<site-env-script>]
#
#   host-tag         short machine id that goes in the session id (e.g. skyler-h100,
#                    bigred200-a100, delta-h200, delta-mi100, lair-a6000)
#   site-env-script  scripts/env-<site>.sh to source (default: scripts/env-jetstream2.sh)
#
# On a machine with NO native CUDA compressors (Delta), pass the fzgm-only flag:
#   SPEC_FZGM_ONLY=1 scripts/run-specialization-smoke.sh delta-h200 scripts/env-delta-h200.sh
# which skips specialization_vs_native_smoke.yaml (its native rows can't run) and
# runs only the memory ablation, whose rows are all fzgm.
set -euo pipefail

host_tag="${1:?usage: run-specialization-smoke.sh <host-tag> [<site-env-script>]}"
site_env="${2:-scripts/env-jetstream2.sh}"
# Session ids embed this tag. It MUST stay fixed across a resume, so override it
# (SPEC_DATE_TAG=20260901 ...) when resuming a run that may cross midnight —
# otherwise the new day's tag starts a fresh, empty session instead of resuming.
date_tag="${SPEC_DATE_TAG:-$(date +%Y%m%d)}"

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repo_dir}"
# shellcheck source=/dev/null
source "${site_env}"

# Lock clocks only where we are allowed to (single-tenant VMs). On SLURM sites the
# script is a no-op / errors harmlessly; rely on --exclusive in the job there.
if command -v nvidia-smi >/dev/null && [ -w /dev/nvidia0 ] 2>/dev/null; then
    bash scripts/lock_clocks.sh || true
    trap 'bash scripts/unlock_clocks.sh || true' EXIT
fi

run_pair () {
    local exp="$1" tag="$2"
    local base="specialization-${tag}-${host_tag}"
    for pol in off auto; do
        local sid="${base}-${pol}-${date_tag}"
        echo "=== ${exp}  FZ_SPECIALIZE=${pol}  ->  ${sid} ==="
        FZ_SPECIALIZE="${pol}" python -m benchkit run \
            "configs/experiments/${exp}" --session-id "${sid}"
        python -m benchkit verify "${BENCHKIT_RESULTS_ROOT}/${sid}" || \
            echo "WARN: verify non-zero for ${sid} (smoke: inspect, don't gate)"
        python -m benchkit report "${BENCHKIT_RESULTS_ROOT}/${sid}" --aggregate
    done
}

if [ "${SPEC_FZGM_ONLY:-0}" != "1" ]; then
    run_pair specialization_vs_native_smoke.yaml vs-native-smoke
fi
run_pair specialization_memory_smoke.yaml memory-smoke

cat <<EOF

Done. Compare the off vs auto sessions:

  # throughput + CR + peak memory, specialized vs staged
  python -m benchkit report \\
    ${BENCHKIT_RESULTS_ROOT}/specialization-vs-native-smoke-${host_tag}-auto-${date_tag} --aggregate

  # per-row: fusion_installed_group_count / fusion_inverse_installed_group_count
  # must be > 0 for every *_sp / pfpl / szp_composed row in the AUTO session,
  # and 0 in the OFF session and for every *_hp row.

Snapshot both sessions to results/baselines/<id>/ per results/baselines/README.md.
EOF
