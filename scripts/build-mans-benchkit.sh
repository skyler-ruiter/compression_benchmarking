#!/usr/bin/env bash
# Apply benchkit's u32 capacity fix to MANS and rebuild its CPU CLIs.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MANS_ROOT="${MANS_ROOT:-${HOME}/compressors/MANS}"
PATCH="${REPO}/patches/mans-u32-adm-capacity.patch"

if grep -q 'max_bytes_signal_per_ele_32b = 3' "${MANS_ROOT}/cpu/adm/adm.h"; then
  git -C "${MANS_ROOT}" apply "${PATCH}"
elif ! grep -q 'max_bytes_signal_per_ele_32b = 4' "${MANS_ROOT}/cpu/adm/adm.h"; then
  echo "error: unrecognized MANS u32 capacity declaration" >&2
  exit 1
fi

cmake --build "${MANS_ROOT}/build" -j"$(nproc)" \
  --target cpu_mans_compress cpu_mans_decompress

echo "built MANS CPU adapter CLIs with four-byte u32 ADM capacity"
