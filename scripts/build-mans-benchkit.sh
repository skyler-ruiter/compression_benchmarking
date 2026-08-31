#!/usr/bin/env bash
# Apply benchkit's MANS fixes and rebuild its CPU CLIs.
#
# Thin wrapper around the portable bootstrap. The canonical patch is
# compressors/patches/mans/0001-u32-adm-capacity-and-cuda-compiler.patch
# (u32 ADM capacity 3->4 bytes, plus a guarded CMAKE_CUDA_COMPILER default).
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec python3 "${REPO}/compressors/bootstrap.py" all mans
