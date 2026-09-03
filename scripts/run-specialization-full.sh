#!/usr/bin/env bash
# Pipeline Specialization vs. native — full-corpus campaign driver.
#
# Runs configs/experiments/specialization_vs_native_full.yaml as an off/auto A/B
# (FZ_SPECIALIZE), verifies each session, prints the aggregated report. Run the
# smoke (scripts/run-specialization-smoke.sh) FIRST and confirm it is clean on
# this machine — this is ~52 h of GPU time on an H100 for the pair.
#
#   scripts/run-specialization-full.sh <host-tag> [<site-env-script>]
#
#   host-tag         short machine id for the session id (skyler-h100, bigred200-a100, ...)
#   site-env-script  scripts/env-<site>.sh to source (default: scripts/env-jetstream2.sh)
#
# Cross-midnight resume: the session id embeds a date tag. Override it on a resume
#   SPEC_DATE_TAG=20260903 scripts/run-specialization-full.sh skyler-h100
#
# Sharding (SLURM array / multiple GPUs — NEVER two shards on one GPU, D15):
#   SPEC_SHARD="$SLURM_ARRAY_TASK_ID/8" SPEC_DATE_TAG="$SLURM_ARRAY_JOB_ID" \
#     scripts/run-specialization-full.sh bigred200-a100 scripts/env-bigred200.sh
#   ...then `benchkit merge` each session dir once all shards finish.
#
# FZGM-only machines (Delta): the native rows cannot run. There is no FZGM-only
# full specialization experiment yet — use fzgm_only_full.yaml under off/auto if
# you need corpus-scale FZGM-only specialization numbers there, or just rely on
# the smoke + the H100/A100 vs-native pairs.
set -euo pipefail

host_tag="${1:?usage: run-specialization-full.sh <host-tag> [<site-env-script>]}"
site_env="${2:-scripts/env-jetstream2.sh}"
date_tag="${SPEC_DATE_TAG:-$(date +%Y%m%d)}"
shard_arg=()
[ -n "${SPEC_SHARD:-}" ] && shard_arg=(--shard "${SPEC_SHARD}")

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repo_dir}"
# shellcheck source=/dev/null
source "${site_env}"

if command -v nvidia-smi >/dev/null && [ -w /dev/nvidia0 ] 2>/dev/null; then
    bash scripts/lock_clocks.sh || true
    trap 'bash scripts/unlock_clocks.sh || true' EXIT
fi

exp=specialization_vs_native_full.yaml
base="specialization-vs-native-full-${host_tag}"
for pol in off auto; do
    sid="${base}-${pol}-${date_tag}"
    echo "=== ${exp}  FZ_SPECIALIZE=${pol}  ->  ${sid}  ${shard_arg[*]:-} ==="
    FZ_SPECIALIZE="${pol}" python -m benchkit run \
        "configs/experiments/${exp}" --session-id "${sid}" "${shard_arg[@]}"
    if [ -z "${SPEC_SHARD:-}" ]; then
        python -m benchkit verify "${BENCHKIT_RESULTS_ROOT}/${sid}" || \
            echo "WARN: verify non-zero for ${sid} — inspect the audit; native eb misses + HPC cv are expected, a CR drift off<->auto is not"
        python -m benchkit report "${BENCHKIT_RESULTS_ROOT}/${sid}" --aggregate --by-dataset
    fi
done

cat <<EOF

Done${SPEC_SHARD:+ (shard ${SPEC_SHARD} — run \`benchkit merge\` on each session dir once all shards finish, then verify)}.

Compare the pair:
  python -m benchkit report ${BENCHKIT_RESULTS_ROOT}/${base}-auto-${date_tag} --aggregate --by-dataset

Gate checks (must hold):
  - per-cell CR identical off vs auto for every fzgm row (byte-identical payload)
  - fusion_installed_group_count / fusion_inverse_installed_group_count > 0 for
    every *_sp / pfpl / szp_composed row in AUTO, 0 in OFF
  - cusz / fzgpu / cuszp3_fixed rows: auto == off exactly
    (controls — they never specialize)
  - f64 _sp rows (MIRANDA/S3D/NWCHEM/BROWN): confirm AUTO actually fused; a silent
    staged fallback is a finding to report, not a failure

Snapshot both sessions to results/baselines/<id>/ per results/baselines/README.md.
EOF
