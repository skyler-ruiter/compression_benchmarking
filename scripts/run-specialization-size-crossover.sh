#!/usr/bin/env bash
# Run the fixed-graph size-crossover matrix as staged and Auto sessions.
set -euo pipefail

host_tag="${1:?usage: run-specialization-size-crossover.sh <host-tag> [<site-env-script>]}"
site_env="${2:-scripts/env-jetstream2.sh}"
date_tag="${SPEC_DATE_TAG:-$(date +%Y%m%d)}"

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repo_dir}"
# shellcheck source=/dev/null
source "${site_env}"

if [ -n "${FZGMOD_CLI_OVERRIDE:-}" ]; then
    export FZGMOD_CLI="${FZGMOD_CLI_OVERRIDE}"
fi

python_bin="${BENCHKIT_PYTHON:-${repo_dir}/.venv/bin/python}"
if [ ! -x "${python_bin}" ]; then
    python_bin="$(command -v python3)"
fi

if command -v nvidia-smi >/dev/null && [ -w /dev/nvidia0 ] 2>/dev/null; then
    bash scripts/lock_clocks.sh || true
    trap 'bash scripts/unlock_clocks.sh || true' EXIT
fi

exp="${SPEC_EXPERIMENT:-configs/experiments/specialization_size_crossover.yaml}"
datasets="${SPEC_DATASETS:-configs/datasets_specialization_sizes.yaml}"
base="${SPEC_SESSION_BASE:-specialization-size-crossover-${host_tag}}"
for pol in off auto; do
    sid="${base}-${pol}-${date_tag}"
    echo "=== ${exp} FZ_SPECIALIZE=${pol} -> ${sid} ==="
    FZ_SPECIALIZE="${pol}" "${python_bin}" -m benchkit run "${exp}" \
        --datasets "${datasets}" --session-id "${sid}"
    "${python_bin}" -m benchkit verify "${BENCHKIT_RESULTS_ROOT}/${sid}" || \
        echo "WARN: verify non-zero for ${sid}; inspect before comparing"
done

cat <<EOF

Done. Pair Off/Auto rows by dataset, field, pipeline, mode, and bound. Confirm
archive/reconstruction identity and inspect specialization.groups[].execution_path
before fitting or describing a crossover.
EOF
