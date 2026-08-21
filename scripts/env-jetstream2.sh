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
# Datasets live on the attached 200 GB volume (/dev/sdb), NOT under $HOME: the root
# disk is 145 GB and was already 59% full with 38 GB of SDRBench data on it, which
# left no room for the rest of the corpus. Moved 2026-07-28.
# Results stay on the root disk — they are small (JSONL + logs) and the work/ dir is
# cleaned per cell (D11/D25/D26).
export BENCHKIT_DATA_ROOT="/media/volume/Compression_Data/sdrbench_data"
export BENCHKIT_RESULTS_ROOT="${HOME}/benchkit-results"

# If the volume is not mounted, fail loudly here rather than 20 minutes into a sweep
# with a confusing "field not found" from the dataset catalog.
if [ ! -d "${BENCHKIT_DATA_ROOT}" ]; then
    echo "env-jetstream2.sh: BENCHKIT_DATA_ROOT=${BENCHKIT_DATA_ROOT} is missing." >&2
    echo "  Is the volume mounted?  findmnt /media/volume/Compression_Data" >&2
fi

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
# Float wrapper uses the CPU backend consistently: the installed NVIDIA backend
# crashes on the u32 codes required by tight bounds. See docs/adapters/mans.md.
export MANS_CLI="${HOME}/compressors/MANS/build/bin/cpu/cpu_mans_compress"
export MANS_DECOMPRESS_CLI="${HOME}/compressors/MANS/build/bin/cpu/cpu_mans_decompress"

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

# ── nvCOMP ───────────────────────────────────────────────────────────────────
# GPU LOSSLESS (zstd/lz4/deflate/gdeflate/ans) — no error bound, so it only runs
# under error.mode: lossless. nvCOMP is a prebuilt SDK; nvcomp_cli is built from
# THIS repo against it (nvCOMP ships no CLI that reports device time):
#
#   ./scripts/build-nvcomp-cli.sh
#
# Pinned to the latest redist: 5.3.0.16 (released 2026-07-14). Check for newer with
#   curl -s https://developer.download.nvidia.com/compute/nvcomp/redist/ | grep redistrib_
# ~/compressors/nvcomp is a symlink to the version in use; 5.2.0.10 is kept beside
# it for A/B. 5.2 -> 5.3 is API-identical for everything nvcomp_cli uses and produces
# byte-identical bitstreams, but ANS compress is +34% and Zstd compress -2.6%
# (docs/adapters/nvcomp.md). Zstd needs >= 3.0.
export NVCOMP_ROOT="${HOME}/compressors/nvcomp"
export NVCOMP_CLI="${HOME}/compression_benchmarking/tools/nvcomp_cli/build/nvcomp_cli"

# Native FSZ 1.0.0 plus the repo-owned host/device timing harness. FSZ is the
# fused reference for FZGM's Quantizer -> AdaptiveLorenzo -> AdaptiveBitpack
# reconstruction; see docs/adapters/fsz.md.
export FSZ_CLI="${HOME}/compressors/FSZ/build/fsz"
export FSZ_HOSTTIME_CLI="${HOME}/compression_benchmarking/tools/fsz_hosttime/fsz_hosttime"

# ── lsCOMP ───────────────────────────────────────────────────────────────────
# Float wrapper uses both integer CLIs and the repo-owned standalone decoder.
export LSCOMP_CLI="${HOME}/compressors/lsCOMP/build/lsCOMP_uint32"
export LSCOMP_UINT16_CLI="${HOME}/compressors/lsCOMP/build/lsCOMP_uint16"
export LSCOMP_DECODE_CLI="${HOME}/compression_benchmarking/tools/lscomp_decode/lscomp_decode"

# ── Python venv ──────────────────────────────────────────────────────────────
# Resolve relative to this script's own location so it works regardless of cwd.
_JS2_SCRIPTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -z "${VIRTUAL_ENV:-}" ]]; then
    # shellcheck source=/dev/null
    source "${_JS2_SCRIPTS_DIR}/../.venv/bin/activate"
fi
unset _JS2_SCRIPTS_DIR
