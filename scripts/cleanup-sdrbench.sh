#!/bin/bash
# One-time cleanup of the sdrbench_data directory after the first (broken) download run.
# Renames dimension-only dirs to descriptive names, removes old empty placeholder dirs,
# and then re-runs the download script to write metadata.yaml into each dataset dir.
#
# Run AFTER the download job finishes:
#   bash scripts/cleanup-sdrbench.sh [DATA_DIR]

set -euo pipefail

DATA_DIR="${1:-/N/scratch/sruiter/sdrbench_data}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "[cleanup] DATA_DIR = $DATA_DIR"

move_if_needed() {
    local src="$DATA_DIR/$1"
    local dst="$DATA_DIR/$2"
    if [ -d "$src" ] && [ -n "$(ls -A "$src" 2>/dev/null)" ]; then
        echo "[rename]  $1 -> $2"
        mkdir -p "$dst"
        mv "$src"/* "$dst"/
        rmdir "$src"
    elif [ -d "$src" ]; then
        echo "[remove]  $1 (empty)"
        rmdir "$src"
    fi
}

remove_if_empty() {
    local dir="$DATA_DIR/$1"
    if [ -d "$dir" ] && [ -z "$(ls -A "$dir" 2>/dev/null)" ]; then
        echo "[remove]  $1 (empty placeholder)"
        rmdir "$dir"
    fi
}

# Rename dimension-only dirs the broken run created.
move_if_needed "1800x3600"                    "CESM_1800x3600"
move_if_needed "100x500x500"                  "HURR_100x500x500"
move_if_needed "280953867"                    "HACCM_280953867"
move_if_needed "2869440"                      "EXAALT_2869440"
move_if_needed "SDRBENCH-CESM-ATM-26x1800x3600" "CESMATM_26x1800x3600"
move_if_needed "SDRBENCH-EXASKY-NYX-512x512x512" "NYX_512x512x512"
move_if_needed "SDRBENCH-Miranda-256x384x384" "MIRANDA_256x384x384"

# Remove old empty placeholder dirs that existed before the download.
remove_if_empty "CESM_1800x3600"      # only if still empty after the move above
remove_if_empty "CESMATM_26x1800x3600"
remove_if_empty "EXAALT_2869440"
remove_if_empty "HACCM_280953867"
remove_if_empty "HURR_100x500x500"
remove_if_empty "MIRANDA_256x384x384"
remove_if_empty "NYX_512x512x512"

echo ""
echo "[state] $DATA_DIR after cleanup:"
for d in "$DATA_DIR"/*/; do
    [ -d "$d" ] || continue
    count=$(find "$d" -type f 2>/dev/null | wc -l)
    printf "  %-35s %d file(s)\n" "$(basename "$d")" "$count"
done

# Re-run the download script — it will skip populated dirs and just write
# the missing metadata.yaml files into each dataset directory.
echo ""
echo "[metadata] writing metadata.yaml into each dataset dir..."
SDRBENCH_DATASETS="CESM CESMATM EXAALT HURR HACC NYX MIRANDA QMCPACK" \
    bash "$SCRIPT_DIR/download-sdrbench.sh" "$DATA_DIR"
