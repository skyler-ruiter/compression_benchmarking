#!/usr/bin/env bash
# Reproduce CI's CPU-only install/test path in a fresh virtual environment.
set -euo pipefail

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
TMP=$(mktemp -d /tmp/benchkit-clean-install.XXXXXX)
trap 'rm -rf -- "$TMP"' EXIT

python3 -m venv "$TMP/venv"
"$TMP/venv/bin/python" -m pip install -r "$REPO/requirements-dev.lock"
"$TMP/venv/bin/python" -m pip install --no-build-isolation --no-deps -e "$REPO"
cd "$REPO"
"$TMP/venv/bin/python" -m compileall -q benchkit tests
"$TMP/venv/bin/python" -m unittest discover -s tests -v
"$TMP/venv/bin/benchkit" --help >/dev/null
