#!/usr/bin/env bash
# Undo scripts/lock_clocks.sh — restore automatic clock management.

set -euo pipefail

echo "[unlock_clocks] resetting SM clock to automatic"
sudo nvidia-smi -rgc

echo "[unlock_clocks] current state:"
nvidia-smi --query-gpu=clocks.current.sm,clocks.max.sm,persistence_mode --format=csv
