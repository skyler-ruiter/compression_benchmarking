#!/usr/bin/env bash
# Run the H3/H4 preflight then the authoritative final RQ3 X2 matrix.
# The caller must ensure the H100 is idle; this script refuses to overlap a
# foreign GPU compute process rather than producing contaminated timing data.
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd "${script_dir}/.." && pwd)"
cd "${repo_dir}"

source scripts/env-jetstream2.sh
export FZGMOD_CLI=/home/exouser/FZGPUModules-benchmark-b90fdfe/build_benchmarking/bin/fzgmod-cli
export FZ_SPECIALIZE=off

require_idle_gpu() {
    local apps
    apps="$(nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>/dev/null | sed '/^[[:space:]]*$/d')"
    if [[ -n "${apps}" ]]; then
        echo "Refusing to run the paired matrix while GPU compute work is active:" >&2
        echo "${apps}" >&2
        exit 2
    fi
}

smoke_session=fzgm-rq3-x2-paired-final-smoke-r2-h100-20260908
final_session=fzgm-rq3-x2-paired-final-h100-20260908

require_idle_gpu
bash scripts/lock_clocks.sh

python -m benchkit run configs/experiments/fzgm_rq3_x2_paired_final_smoke.yaml --session-id "${smoke_session}"
python -m benchkit merge "${BENCHKIT_RESULTS_ROOT}/${smoke_session}"
python -m benchkit verify "${BENCHKIT_RESULTS_ROOT}/${smoke_session}"

require_idle_gpu
python -m benchkit run configs/experiments/fzgm_rq3_x2_paired_final.yaml --session-id "${final_session}"
python -m benchkit merge "${BENCHKIT_RESULTS_ROOT}/${final_session}"
python -m benchkit verify "${BENCHKIT_RESULTS_ROOT}/${final_session}"
