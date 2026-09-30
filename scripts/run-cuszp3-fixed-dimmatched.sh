#!/usr/bin/env bash
# Dimension-matched cuSZp3 fixed rerun (off, then auto) at FZGM d511ebc.
#
# Replaces the cuSZp3 fixed rows of specialization_vs_native_full, whose native side ran
# 1-D on every dataset. FZGMOD_CLI must be a clean d511ebc build whose NVRTC include
# paths point at a d511ebc tree (the primary checkout's headers have moved on):
#
#   FZGMOD_CLI_OVERRIDE=~/FZGPUModules-d511ebc-cuszp3fixed/build/bin/fzgmod-cli \
#     scripts/run-cuszp3-fixed-dimmatched.sh skyler-h100
set -euo pipefail

host_tag="${1:?usage: run-cuszp3-fixed-dimmatched.sh <host-tag> [<site-env-script>]}"
site_env="${2:-scripts/env-jetstream2.sh}"
date_tag="${SPEC_DATE_TAG:-$(date +%Y%m%d)}"

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repo_dir}"
# shellcheck source=/dev/null
source "${site_env}"
export FZGMOD_CLI="${FZGMOD_CLI_OVERRIDE:?set FZGMOD_CLI_OVERRIDE to the clean d511ebc build}"

if command -v nvidia-smi >/dev/null && [ -w /dev/nvidia0 ] 2>/dev/null; then
    bash scripts/lock_clocks.sh || true
    trap 'bash scripts/unlock_clocks.sh || true' EXIT
fi

exp=cuszp3_fixed_dimmatched_full.yaml
base="cuszp3-fixed-dimmatched-${host_tag}"
for pol in off auto; do
    sid="${base}-${pol}-${date_tag}"
    echo "=== ${exp}  FZ_SPECIALIZE=${pol}  ->  ${sid} ==="
    FZ_SPECIALIZE="${pol}" python -m benchkit run "configs/experiments/${exp}" --session-id "${sid}"
    python -m benchkit verify "${BENCHKIT_RESULTS_ROOT}/${sid}" || \
        echo "WARN: verify non-zero for ${sid} -- inspect the audit"
done
