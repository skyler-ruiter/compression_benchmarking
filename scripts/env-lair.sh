#!/usr/bin/env bash
# LAIR (IU Luddy) environment for compression benchmarking.
# Source this file — do not execute it:
#   source scripts/env-lair.sh    (from repo root)
#   .      scripts/env-lair.sh    (POSIX)
#
# Idempotent: PATH/LD_LIBRARY_PATH exports are harmless to repeat; venv activation is
# skipped if already active. Safe to re-source from a running session.
#
# LAIR has no CUDA environment module — nvcc/CUDA 12.8 is baked into the GPU compute
# node images (Lambda Stack, apt-packaged) and is invisible on the login node. GPU
# work (builds, benchmarking) must run inside a Slurm allocation (srun/sbatch) on an
# L40S or H100 node. See docs/machines/lair.md.
#
# All third-party compressors below were built for L40S (sm_89) only; see
# /data/user/sruiter/compression/build_all.sh for the build recipe and
# /data/user/sruiter/compression/logs/ for build logs.
#
# Adding a new compressor:
#   1. Uncomment / add its PATH and CLI export below.
#   2. Add a matching entry to benchkit/adapters/__init__.py.
#   3. Re-source and run the smoke test.

# ── cmake (system cmake 3.22.1 is too old for FZGPUModules, which needs >=3.24) ──
module load cmake/3.29.2 2>/dev/null || true

# ── data & results paths ─────────────────────────────────────────────────────
# Per docs/machines/lair.md storage guidance: /data/user/$USER is for private
# datasets, builds, and results (1TB quota, snapshotted, not backed up off-site).
export BENCHKIT_DATA_ROOT="/data/user/sruiter/compression/sdrbench_data"
export BENCHKIT_RESULTS_ROOT="/data/user/sruiter/compression/results"

if [ ! -d "${BENCHKIT_DATA_ROOT}" ]; then
    echo "env-lair.sh: BENCHKIT_DATA_ROOT=${BENCHKIT_DATA_ROOT} is missing." >&2
    echo "  Run scripts/download-sdrbench.sh first (see docs/hpc-setup.md step 5)." >&2
fi

_CSET="/data/user/sruiter/compression/compressors_set"

# ── FZGM ─────────────────────────────────────────────────────────────────────
export FZGMOD_CLI="${HOME}/FZGPUModules/build/release/bin/fzgmod-cli"

# ── cuSZ (reference) ─────────────────────────────────────────────────────────
export CUSZ_CLI="${_CSET}/cuSZ/build/cusz"
export PATH="${_CSET}/cuSZ/build${PATH:+:$PATH}"

# ── cuSZ-Hi ──────────────────────────────────────────────────────────────────
export CUSZHI_CLI="${_CSET}/cuSZ-Hi/build/cuszhi"

# ── cuSZp2 ────────────────────────────────────────────────────────────────────
export CUSZP2_CLI="${_CSET}/cuSZp-V2.0.1/build/examples/bin/cuSZp"

# ── cuSZp3 ────────────────────────────────────────────────────────────────────
export CUSZP3_CLI="${_CSET}/cuSZp-V3.0.0/build/examples/bin/cuSZp"

# ── FZ-GPU ───────────────────────────────────────────────────────────────────
export FZGPU_CLI="${_CSET}/FZ-GPU/fz-gpu"

# ── PFPL ─────────────────────────────────────────────────────────────────────
# makefile built with NV_SM=89 (L40S). bin/f32 and bin/f64 both present.
export PFPL_BIN_DIR="${_CSET}/PFPL/bin"

# ── FSZ ──────────────────────────────────────────────────────────────────────
export FSZ_CLI="${_CSET}/FSZ/build/fsz"
export FSZ_HOSTTIME_CLI="${HOME}/compression_benchmarking/tools/fsz_hosttime/fsz_hosttime"

# ── MANS ─────────────────────────────────────────────────────────────────────
# CPU backend only — the GPU/NVIDIA backend crashes on u32 codes (docs/adapters/mans.md).
export MANS_CLI="${_CSET}/MANS/build/bin/cpu/cpu_mans_compress"
export MANS_DECOMPRESS_CLI="${_CSET}/MANS/build/bin/cpu/cpu_mans_decompress"

# ── lsCOMP ───────────────────────────────────────────────────────────────────
export LSCOMP_CLI="${_CSET}/lsCOMP/build/lsCOMP_uint32"
export LSCOMP_UINT16_CLI="${_CSET}/lsCOMP/build/lsCOMP_uint16"
export LSCOMP_DECODE_CLI="${HOME}/compression_benchmarking/tools/lscomp_decode/lscomp_decode"

# ── SZ3 ──────────────────────────────────────────────────────────────────────
export SZ3_CLI="${_CSET}/SZ3/build/tools/sz3/sz3"
export PATH="${_CSET}/SZ3/build/tools/sz3${PATH:+:$PATH}"

# ── zfp ──────────────────────────────────────────────────────────────────────
export ZFP_CLI="${_CSET}/zfp/build/bin/zfp"

# ── SPERR ────────────────────────────────────────────────────────────────────
export SPERR_BIN_DIR="${_CSET}/SPERR/build/bin"

# ── MGARD ────────────────────────────────────────────────────────────────────
# NOT BUILT: MGARD's CMakeLists requires a `protoc` binary; this machine has no
# apt-installed protobuf-compiler (no sudo) and the only substitute reachable via
# pip (grpcio-tools' bundled protoc, v35.x) expects an external protoc-gen-cpp
# plugin that doesn't exist here, so C++ codegen fails. Revisit by building
# protobuf's protoc from source (matching libprotobuf23 3.12.4) if MGARD is needed.
# export MGARD_CLI="${_CSET}/MGARD/install-cuda/bin/mgard-x"
# export LD_LIBRARY_PATH="${_CSET}/MGARD/install-cuda/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

# ── nvCOMP ───────────────────────────────────────────────────────────────────
# GPU LOSSLESS (zstd/lz4/deflate/gdeflate/ans) — no error bound, runs under
# error.mode: lossless only. Pinned to the latest redist, 5.3.0.16; 5.2.0.10 kept
# alongside for A/B (see docs/adapters/nvcomp.md, D37). Both nvcomp_cli builds are
# kept as separately-named binaries under tools/nvcomp_cli/by-version/.
export NVCOMP_ROOT="${_CSET}/nvcomp-5.3.0.16"
export NVCOMP_CLI="${HOME}/compression_benchmarking/tools/nvcomp_cli/by-version/nvcomp_cli-5.3.0.16"
# 5.2.0.10 A/B build: NVCOMP_CLI_5_2="${HOME}/compression_benchmarking/tools/nvcomp_cli/by-version/nvcomp_cli-5.2.0.10"

# ── Python venv ──────────────────────────────────────────────────────────────
_LAIR_SCRIPTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -z "${VIRTUAL_ENV:-}" ]]; then
    # shellcheck source=/dev/null
    source "${_LAIR_SCRIPTS_DIR}/../.venv/bin/activate"
fi
unset _LAIR_SCRIPTS_DIR
