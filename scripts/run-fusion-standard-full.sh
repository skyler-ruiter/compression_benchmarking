#!/usr/bin/env bash
# Sequential publication run for the FZGM automatic-fusion study.
# The staged arm contains only semantics-preserving A/B controls. The Auto arm
# additionally measures the fused plain-coder modes for native comparison.
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repo_dir}"
source scripts/env-jetstream2.sh

bash scripts/lock_clocks.sh
trap 'bash scripts/unlock_clocks.sh' EXIT

export FZ_FUSION=off
.venv/bin/python -m benchkit run \
  configs/experiments/fusion_standard_full_staged.yaml \
  --session-id fusion-standard-full-staged-policy-20260831
.venv/bin/python -m benchkit verify \
  "${BENCHKIT_RESULTS_ROOT}/fusion-standard-full-staged-policy-20260831"

export FZ_FUSION=auto
.venv/bin/python -m benchkit run \
  configs/experiments/fusion_standard_full_auto.yaml \
  --session-id fusion-standard-full-auto-policy-20260831
.venv/bin/python -m benchkit verify \
  "${BENCHKIT_RESULTS_ROOT}/fusion-standard-full-auto-policy-20260831"
