#!/usr/bin/env bash
# Run the controlled 1-D Lorenzo + AdaptiveBitpack block-size sweep as a staged
# versus specialized pair. Use a clean FZGPUModules binary via
# FZGMOD_CLI_OVERRIDE for provenance-sensitive runs.
set -euo pipefail

host_tag="${1:?usage: run-specialization-blocksize-sweep.sh <host-tag> [<site-env-script>]}"
site_env="${2:-scripts/env-jetstream2.sh}"
date_tag="${SPEC_DATE_TAG:-$(date +%Y%m%d)}"

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repo_dir}"
# shellcheck source=/dev/null
source "${site_env}"

if [ -n "${FZGMOD_CLI_OVERRIDE:-}" ]; then
    export FZGMOD_CLI="${FZGMOD_CLI_OVERRIDE}"
fi

# Prefer the repository environment explicitly. Some long-lived shells retain
# VIRTUAL_ENV after its bin directory has fallen out of PATH, causing the site
# script's idempotent activation guard to skip reactivation.
python_bin="${BENCHKIT_PYTHON:-${repo_dir}/.venv/bin/python}"
if [ ! -x "${python_bin}" ]; then
    python_bin="$(command -v python3)"
fi

if command -v nvidia-smi >/dev/null && [ -w /dev/nvidia0 ] 2>/dev/null; then
    bash scripts/lock_clocks.sh || true
    trap 'bash scripts/unlock_clocks.sh || true' EXIT
fi

exp=specialization_blocksize_sweep.yaml
base="specialization-blocksize-sweep-${host_tag}"
for pol in off auto; do
    sid="${base}-${pol}-${date_tag}"
    echo "=== ${exp} FZ_SPECIALIZE=${pol} -> ${sid} ==="
    FZ_SPECIALIZE="${pol}" "${python_bin}" -m benchkit run \
        "configs/experiments/${exp}" --session-id "${sid}"
    "${python_bin}" -m benchkit verify "${BENCHKIT_RESULTS_ROOT}/${sid}" || \
        echo "WARN: verify non-zero for ${sid}; inspect before comparing"
    "${python_bin}" -m benchkit report "${BENCHKIT_RESULTS_ROOT}/${sid}" --aggregate --by-dataset
done

cat <<EOF

Done. Pair rows by logical_cell_id and compare off versus auto by block size,
outlier mode, dataset/field size, and requested bound. Confirm compressed bytes
and reconstruction hashes match before interpreting throughput.
EOF
