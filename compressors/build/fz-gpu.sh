#!/usr/bin/env bash
# FZ-GPU. Plain Makefile; arch injected via GENCODE (patch adds the hook).
. "$(dirname "$0")/_run.sh"
make clean || true
make GENCODE="$(gencode_flags)"
