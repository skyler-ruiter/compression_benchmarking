#!/usr/bin/env bash
# nvCOMP is a prebuilt SDK, not a source build. This unpacks the pinned redist
# tarball next to the other checkouts and points `nvcomp` at it.
#
#   arg 1: destination dir (bootstrap passes <COMPRESSORS_ROOT>/nvcomp-<version>)
#   env:   NVCOMP_URL  (from manifest)   NVCOMP_VERSION
#
# nvcomp_cli itself is built inside benchkit (tools/nvcomp_cli), not here —
# see scripts/build-nvcomp-cli.sh.
set -euo pipefail
DEST="${1:?usage: $0 <dest-dir>}"
URL="${NVCOMP_URL:?NVCOMP_URL not set (manifest [compressor.nvcomp].url)}"
VER="${NVCOMP_VERSION:-unknown}"

if [ -e "${DEST}/include/nvcomp.h" ] || [ -e "${DEST}/include/nvcomp/nvcomp.h" ]; then
  echo "nvcomp ${VER} already present at ${DEST}"
else
  mkdir -p "${DEST}"
  tmp="$(mktemp -d)"; trap 'rm -rf "$tmp"' EXIT
  echo "downloading nvcomp ${VER}"
  curl -fL "$URL" -o "$tmp/nvcomp.tar.xz"
  # redist archives have a single top-level dir; strip it.
  tar -xf "$tmp/nvcomp.tar.xz" -C "$tmp"
  inner="$(find "$tmp" -maxdepth 1 -mindepth 1 -type d)"
  cp -a "$inner"/. "${DEST}/"
fi

# `nvcomp` -> active version (matches the local convention).
ln -sfn "$(basename "${DEST}")" "$(dirname "${DEST}")/nvcomp"
echo "nvcomp -> $(basename "${DEST}")"
