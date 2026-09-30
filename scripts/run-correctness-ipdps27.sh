#!/usr/bin/env bash
# Correctness-only rerun at the ipdps27-submission tag, queued behind any running
# benchkit timing campaign so it never perturbs another session's timing.
# Output: sessions correctness-ipdps27-{off,auto}-<date>, cuszhi TP off, and
# results/correctness_ipdps27_report.md (payload/recon/bound vs published sessions).
set -u
cd "$(dirname "$0")/.."
source scripts/env-jetstream2.sh >/dev/null 2>&1
export FZGMOD_CLI=/home/exouser/FZGPUModules-roibin-a062d88/build_rel/bin/fzgmod-cli
DATE=${DATE_TAG:-$(date -u +%Y%m%d)}
log() { echo "[$(date -u +%FT%TZ)] $*"; }
log "waiting for other benchkit campaigns and an idle GPU"
while pgrep -f 'run-range-precomputed.sh' >/dev/null || pgrep -f 'benchkit run configs/experiments/(range|specialization|fzgm|cuszhi|b1|peak)' >/dev/null \
      || [ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]; do sleep 300; done
sleep 120   # let a chained next step (if any) claim the GPU first
if pgrep -f 'benchkit run' >/dev/null; then log "another campaign started; waiting again"; exec "$0"; fi
log "starting correctness rerun at $(git -C ~/FZGPUModules rev-parse --short ipdps27-submission^{commit})"
FZ_SPECIALIZE=off  python -m benchkit run configs/experiments/correctness_ipdps27_specialization.yaml --session-id correctness-ipdps27-off-$DATE  > results/correctness-ipdps27-off-$DATE.log 2>&1 &
FZ_SPECIALIZE=auto python -m benchkit run configs/experiments/correctness_ipdps27_specialization.yaml --session-id correctness-ipdps27-auto-$DATE > results/correctness-ipdps27-auto-$DATE.log 2>&1 &
wait
FZ_SPECIALIZE=off python -m benchkit run configs/experiments/correctness_ipdps27_cuszhi_tp.yaml --session-id correctness-ipdps27-cuszhi-tp-off-$DATE > results/correctness-ipdps27-cuszhi-tp-$DATE.log 2>&1
R=${BENCHKIT_RESULTS_ROOT:-$HOME/benchkit-results}
python -m benchkit merge $R/correctness-ipdps27-off-$DATE/  >/dev/null 2>&1
python -m benchkit merge $R/correctness-ipdps27-auto-$DATE/ >/dev/null 2>&1
python scripts/compare_correctness_rerun.py \
  --published-off  $R/specialization-vs-native-full-skyler-h100-off-20260909 \
  --published-auto $R/specialization-vs-native-full-skyler-h100-auto-20260909 \
  --new-off $R/correctness-ipdps27-off-$DATE --new-auto $R/correctness-ipdps27-auto-$DATE \
  --out results/correctness_ipdps27_report.md > /dev/null
log "done; report: results/correctness_ipdps27_report.md"
