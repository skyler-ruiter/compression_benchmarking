#!/usr/bin/env bash
# Pin GPU SM clock to its max boost value for stable timing.
#
# Unlike BigRed200 (shared cluster, nvidia-smi -lgc is admin-only — see
# docs/DESIGN.md D15, docs/hpc-setup.md), this JetStream2 VM has passwordless
# sudo and an exclusive, single-tenant GPU (confirmed: no other compute
# processes, persistence mode already enabled). Locking clocks here is safe
# and directly addresses the timing-variance ("!UNSTABLE", tOK=False) flags
# seen in earlier smoke runs on unlocked clocks.
#
# Usage:
#   bash scripts/lock_clocks.sh          # lock to max boost (1980 MHz here)
#   bash scripts/lock_clocks.sh <mhz>    # lock to a specific clock
#
# Pair with scripts/unlock_clocks.sh when done. Set lock_clocks: true in the
# experiment config to record that clocks were actually locked for this run
# (DESIGN.md's lock_clocks field is documentary only — the harness does not
# lock clocks itself).

set -euo pipefail

CLOCK_MHZ="${1:-$(nvidia-smi --query-gpu=clocks.max.sm --format=csv,noheader,nounits)}"

echo "[lock_clocks] locking SM clock to ${CLOCK_MHZ} MHz"
sudo nvidia-smi -lgc "${CLOCK_MHZ},${CLOCK_MHZ}"
sudo nvidia-smi -pm 1   # persistence mode (idempotent; usually already enabled)

echo "[lock_clocks] current state:"
nvidia-smi --query-gpu=clocks.current.sm,clocks.max.sm,persistence_mode,power.limit --format=csv
