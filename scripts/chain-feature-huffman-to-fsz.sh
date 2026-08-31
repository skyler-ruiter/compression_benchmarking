#!/usr/bin/env bash
# Wait for the 2026-08-28 FEAT-HUFF campaign, gate FSZ on f32/f64 smoke cells,
# then launch the full standard-bound paired FSZ campaign on the same H100.
set -euo pipefail

repo=/home/exouser/compression_benchmarking
results=/home/exouser/benchkit-results
predecessor=feature-huffman-full-f32-20260828
predecessor_rows=1896
smoke=fsz-standard-smoke-20260828
full=fsz-standard-full-20260828
chain_log="${results}/_run_logs/chain-feature-huffman-to-fsz-20260828.log"

mkdir -p "${results}/_run_logs"
exec >>"${chain_log}" 2>&1
echo "[$(date -u +%FT%TZ)] waiting for ${predecessor} (${predecessor_rows} rows)"

while true; do
  runs="${results}/${predecessor}/runs.jsonl"
  rows=0
  if [[ -f "${runs}" ]]; then
    rows=$(wc -l <"${runs}")
  fi
  if (( rows >= predecessor_rows )); then
    break
  fi
  if ! tmux has-session -t "${predecessor}" 2>/dev/null; then
    echo "[$(date -u +%FT%TZ)] predecessor stopped incomplete at ${rows}/${predecessor_rows}; refusing to start FSZ"
    exit 1
  fi
  echo "[$(date -u +%FT%TZ)] predecessor ${rows}/${predecessor_rows}"
  sleep 30
done

# The last result row is appended just before the producer exits. Wait for the
# tmux session itself to disappear so no timed GPU work can overlap.
while tmux has-session -t "${predecessor}" 2>/dev/null; do
  sleep 5
done

cd "${repo}"
# shellcheck source=/dev/null
source scripts/env-jetstream2.sh
if [[ ! -x "${FSZ_CLI}" || ! -x "${FSZ_HOSTTIME_CLI}" || ! -x "${FZGMOD_CLI}" ]]; then
  echo "[$(date -u +%FT%TZ)] required FSZ/FZGM executable missing; refusing full run"
  exit 1
fi

echo "[$(date -u +%FT%TZ)] starting paired FSZ smoke"
.venv/bin/python -m benchkit.cli run \
  configs/experiments/fsz_vs_native_standard_smoke.yaml \
  --session-id "${smoke}"

smoke_runs="${results}/${smoke}/runs.jsonl"
smoke_rows=$(wc -l <"${smoke_runs}")
smoke_ok=$(.venv/bin/python - "${smoke_runs}" <<'PY'
import json, sys
rows = [json.loads(line) for line in open(sys.argv[1]) if line.strip()]
print(sum(r.get("status") == "ok" and r.get("eb_satisfied") is True for r in rows))
PY
)
if [[ "${smoke_rows}" != 4 || "${smoke_ok}" != 4 ]]; then
  echo "[$(date -u +%FT%TZ)] FSZ smoke failed gate: rows=${smoke_rows}, valid_ok=${smoke_ok}; refusing full run"
  exit 1
fi

echo "[$(date -u +%FT%TZ)] smoke passed; starting ${full}"
.venv/bin/python -m benchkit.cli run \
  configs/experiments/fsz_vs_native_full_standard.yaml \
  --session-id "${full}"
echo "[$(date -u +%FT%TZ)] ${full} finished"
