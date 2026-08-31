#!/usr/bin/env bash
# PFPL. Makefile; NV_SM passed on the command line (overrides the := default).
. "$(dirname "$0")/_run.sh"
make clean || true
make all NV_SM="${CUDA_ARCH%a}"
