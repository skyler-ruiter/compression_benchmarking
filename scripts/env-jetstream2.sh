#!/usr/bin/env bash
# JetStream2 H100 environment for compression benchmarking (host: skyler-h100).
# Source this file — do not execute it:
#   source scripts/env-jetstream2.sh    (from repo root)
#   .      scripts/env-jetstream2.sh    (POSIX)
#
# Idempotent: PATH/LD_LIBRARY_PATH exports are harmless to repeat; venv activation is
# skipped if already active.  Safe to re-source from a running session.
#
# Unlike BigRed200 this is a single persistent VM (no module system, no SLURM, no
# scratch filesystem) — CUDA comes from ~/load-env (nvhpc 25.7 bundling CUDA 12.9),
# and data/results live on the one local disk (150GB, see `df -h /`).
#
# Adding a new compressor:
#   1. Uncomment / add its PATH and CLI export below.
#   2. Add a matching entry to benchkit/adapters/__init__.py.
#   3. Re-source and run the smoke test.

# ── CUDA toolkit ─────────────────────────────────────────────────────────────
# shellcheck source=/dev/null
source "${HOME}/load-env"   # nvhpc 25.7 / CUDA 12.9; driver reports CUDA 13.2 (fine, newer)

# ── data & results paths ─────────────────────────────────────────────────────
# No scratch filesystem here — everything lives on the local disk under $HOME.
export BENCHKIT_DATA_ROOT="${HOME}/data"
export BENCHKIT_RESULTS_ROOT="${HOME}/benchkit-results"

# ── FZGM ─────────────────────────────────────────────────────────────────────
# build_benchmarking/ is the Release build used for timing (build/ is Debug, kept
# for tests; build_profiling/ is a separate Release build with profiling instrumentation).
# All sm_90 (H100) — see ~/FZGPUModules/CMakeLists.txt CMAKE_CUDA_ARCHITECTURES.
export FZGMOD_CLI="${HOME}/FZGPUModules/build_benchmarking/bin/fzgmod-cli"

# ── cuSZ (reference) ─────────────────────────────────────────────────────────
# Built at ~/compressors/cuSZ/build (Release, sm_90) — the gcc-13 ICE on hf_hl.cc
# that BigRed200 hit (needed gcc-native/12.3) apparently didn't recur here, or was
# already patched in this checkout.
export CUSZ_CLI="${HOME}/compressors/cuSZ/build/cusz"
export PATH="${HOME}/compressors/cuSZ/build${PATH:+:$PATH}"

# ── cuSZp2 ────────────────────────────────────────────────────────────────────
export CUSZP2_CLI="${HOME}/compressors/cuSZp-V2.0.1/build/examples/bin/cuSZp"

# ── cuSZp3 ────────────────────────────────────────────────────────────────────
export CUSZP3_CLI="${HOME}/compressors/cuSZp-V3.0.0/build/examples/bin/cuSZp"

# ── cuSZ-Hi ──────────────────────────────────────────────────────────────────
export CUSZHI_CLI="${HOME}/compressors/cuSZ-Hi/build/cuszhi"

# ── SZ3 ──────────────────────────────────────────────────────────────────────
# CPU-only, Release. Native ABS + REL(range) — no emulation needed.
export SZ3_CLI="${HOME}/compressors/SZ3/build/tools/sz3/sz3"
export PATH="${HOME}/compressors/SZ3/build/tools/sz3${PATH:+:$PATH}"

# ── MANS ─────────────────────────────────────────────────────────────────────
# NOTE: stub adapter — lossless int compressor; quantization wrapper needed.
export MANS_CLI="${HOME}/compressors/MANS/build/bin/nv/nv_mans_compress"

# ── FZ-GPU ───────────────────────────────────────────────────────────────────
# Built at ~/compressors/FZ-GPU/fz-gpu — src/fz.cu already carries the
# compress_out/decompress_out/repeat patch docs/adapters/fzgpu.md describes
# (confirmed by inspection: the patched runFzgpu() signature and main() are
# already in this checkout). Verified working on the H100 with real data.
export FZGPU_CLI="${HOME}/compressors/FZ-GPU/fz-gpu"

# ── PFPL ─────────────────────────────────────────────────────────────────────
# Built at ~/compressors/PFPL/bin — makefile already has NV_SM := 90 (H100).
# Verified working on the H100 with real data (f32 NOA GPU binary).
export PFPL_BIN_DIR="${HOME}/compressors/PFPL/bin"

# ── zfp ──────────────────────────────────────────────────────────────────────
# CPU-only for error-bounded modes (its CUDA backend only supports fixed-rate —
# see benchkit/adapters/zfp.py). CR/quality baseline, not a throughput peer.
export ZFP_CLI="${HOME}/compressors/zfp/build/bin/zfp"

# ── MGARD ────────────────────────────────────────────────────────────────────
# GPU (cuda device), Release, built at ~/compressors/MGARD/install-cuda-hopper.
# Needs its own lib/ on LD_LIBRARY_PATH (libmgard.so, bundled nvcomp/protobuf/zstd).
export MGARD_CLI="${HOME}/compressors/MGARD/install-cuda-hopper/bin/mgard-x"
export LD_LIBRARY_PATH="${HOME}/compressors/MGARD/install-cuda-hopper/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

# ── SPERR ────────────────────────────────────────────────────────────────────
# CPU-only (OpenMP), Release. 2D/3D only (no 1D/4D — see benchkit/adapters/sperr.py).
export SPERR_BIN_DIR="${HOME}/compressors/SPERR/build/bin"

# ── lsCOMP ───────────────────────────────────────────────────────────────────
# NOTE: stub adapter — quantized-integer (uint32/16) compressor, no direct
# float+error-bound CLI mode. See docs/adapters/lscomp.md.
export LSCOMP_CLI="${HOME}/compressors/lsCOMP/build/lsCOMP_uint32"


# ── Python venv ──────────────────────────────────────────────────────────────
# Resolve relative to this script's own location so it works regardless of cwd.
_JS2_SCRIPTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -z "${VIRTUAL_ENV:-}" ]]; then
    # shellcheck source=/dev/null
    source "${_JS2_SCRIPTS_DIR}/../.venv/bin/activate"
fi
unset _JS2_SCRIPTS_DIR
