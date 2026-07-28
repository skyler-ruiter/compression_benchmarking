#!/bin/bash
# Download SDRBench datasets from the Globus mirror.
#
# Usage:
#   bash download-sdrbench.sh [DATA_DIR [TARBALL_DIR]]
#
#   DATA_DIR     where extracted data goes  (default: $SDRBENCH_DATA_ROOT or ./sdrbench_data)
#   TARBALL_DIR  where .tar.gz files cache  (default: DATA_DIR/../sdrbench_tarballs)
#
# Environment:
#   SDRBENCH_DATA_ROOT   sets DATA_DIR default — set this in your job script
#   SDRBENCH_DATASETS    space-separated subset of keys to download, e.g. "CESM HURR NYX"
#                        (omit to download everything listed in ENABLED_DATASETS below)
#
# SLURM example (recommended — avoids login-node time limits):
#   sbatch --account=<acct> --partition=general --time=04:00:00 \
#          --wrap="bash /path/to/download-sdrbench.sh /path/to/scratch/sdrbench_data"
#
# Features:
#   - Tarballs are cached in TARBALL_DIR; re-running only re-extracts missing datasets.
#   - wget -c makes downloads resumable after interruption.
#   - Extraction uses a temp directory so the tarball's internal naming never pollutes DATA_DIR.
#   - A metadata.yaml is written into each dataset directory for self-documentation.
#   - Datasets are skipped if their directory already contains a metadata.yaml.
#
# Dependencies: bash, wget, tar, mktemp (all standard on Linux/HPC)

set -euo pipefail

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_DIR="${1:-${SDRBENCH_DATA_ROOT:-$(pwd)/sdrbench_data}}"
TARBALL_DIR="${2:-$(dirname "$DATA_DIR")/sdrbench_tarballs}"
BASE_URL="https://g-8d6b0.fd635.8443.data.globus.org/ds131.2/Data-Reduction-Repo/raw-data"

mkdir -p "$DATA_DIR" "$TARBALL_DIR"
echo "[config] DATA_DIR    = $DATA_DIR"
echo "[config] TARBALL_DIR = $TARBALL_DIR"
echo ""

# ---------------------------------------------------------------------------
# Dataset selection
# Edit ENABLED_DATASETS to choose a default subset, or set SDRBENCH_DATASETS
# in the environment at call time (e.g. SDRBENCH_DATASETS="CESM NYX" bash ...).
# Keys must match the first column of the DATASETS table below.
# ---------------------------------------------------------------------------
# The default set is every dataset this harness has a working reader for. The
# 2026-07-28 additions (SCALE, NWCHEM, S3D, BROWN, EXAFEL) roughly double the corpus
# on disk — S3D alone is a 46 GB tarball — so on a space-constrained machine pass
# SDRBENCH_DATASETS explicitly instead of taking the default.
#
# Deliberately NOT listed (see results/baselines/README.md and docs/adapters/):
#   NSTX_GPI  — ships as an ADIOS2 .bp container, not a raw binary; needs an ADIOS2
#               reader before benchkit can load it.
#   XGC       — unstructured-mesh output; the .zip "binary" variant needs its layout
#               confirmed before it can be registered as a plain 2-D field.
#   MIRANDA (3072^3, 106 GB) and HACC (full 19 GB) — larger variants of datasets
#               already present at a usable size.
#   EXAFEL 986x32x185x388 — the 4-D build; no adapter here handles 4-D (see the
#               QMCPACK-3D split in configs/datasets.yaml for the pattern if wanted).
ENABLED_DATASETS="${SDRBENCH_DATASETS:-
    CESM
    CESMATM
    EXAALT
    HURR
    HACC
    NYX
    MIRANDA
    QMCPACK
    SCALE
    NWCHEM
    S3D
    BROWN
    EXAFEL
}"

# ---------------------------------------------------------------------------
# Dataset table  (stride 4): key | tarball filename | url suffix | dest dir name
#
# dest dir is the final name under DATA_DIR. The tarball's own internal
# directory name is ignored — we always extract to a temp dir and rename.
# ---------------------------------------------------------------------------
DATASETS=(
  "CESM"    "SDRBENCH-CESM-ATM-1800x3600.tar.gz"           "CESM-ATM/SDRBENCH-CESM-ATM-1800x3600.tar.gz"                   "CESM_1800x3600"
  "CESMATM" "SDRBENCH-CESM-ATM-26x1800x3600.tar.gz"        "CESM-ATM/SDRBENCH-CESM-ATM-26x1800x3600.tar.gz"                "CESMATM_26x1800x3600"
  "EXAALT"  "SDRBENCH-EXAALT-2869440.tar.gz"               "EXAALT/SDRBENCH-EXAALT-2869440.tar.gz"                          "EXAALT_2869440"
  "HURR"    "SDRBENCH-Hurricane-ISABEL-100x500x500.tar.gz"  "Hurricane-ISABEL/SDRBENCH-Hurricane-ISABEL-100x500x500.tar.gz"  "HURR_100x500x500"
  "HACC"    "EXASKY-HACC-data-medium-size.tar.gz"          "EXASKY/HACC/EXASKY-HACC-data-medium-size.tar.gz"                "HACCM_280953867"
  "NYX"     "SDRBENCH-EXASKY-NYX-512x512x512.tar.gz"       "EXASKY/NYX/SDRBENCH-EXASKY-NYX-512x512x512.tar.gz"             "NYX_512x512x512"
  "MIRANDA" "SDRBENCH-Miranda-256x384x384.tar.gz"          "Miranda/SDRBENCH-Miranda-256x384x384.tar.gz"                    "MIRANDA_256x384x384"
  "QMCPACK" "SDRBENCH-QMCPack.tar.gz"                      "QMCPack/SDRBENCH-QMCPack.tar.gz"                               "QMCPACK"
  # ── added 2026-07-28: the rest of SDRBench that this harness can actually read ──
  "SCALE"   "SDRBENCH-SCALE-98x1200x1200.tar.gz"           "SCALE_LETKF/SDRBENCH-SCALE-98x1200x1200.tar.gz"                "SCALE_98x1200x1200"
  "NWCHEM"  "SDRBENCH-NWChem-dataset.tar.gz"               "NWChem/SDRBENCH-NWChem-dataset.tar.gz"                          "NWCHEM_102953248"
  "S3D"     "SDRBENCH-S3D.tar.gz"                          "S3D/SDRBENCH-S3D.tar.gz"                                       "S3D_500x500x500"
  "BROWN"   "SDRBENCH-BROWN-33554433.tar.gz"               "brown_synth/SDRBENCH-BROWN-33554433.tar.gz"                     "BROWN_33554433"
  "EXAFEL"  "SDRBENCH-EXAFEL-130x1480x1552.tar.gz"         "EXAFEL/SDRBENCH-EXAFEL-130x1480x1552.tar.gz"                    "EXAFEL_130x1480x1552"
)

# Metadata written into each dataset directory after extraction.  (stride 6)
# Columns: key | display name | dims | dtype | num_fields | description
METADATA=(
  "CESM"    "CESM-ATM 2D"      "1800x3600"        "f32" "79"  "CESM atmosphere model, 79 2-D surface fields"
  "CESMATM" "CESM-ATM 3D"      "26x1800x3600"     "f32" "33"  "CESM atmosphere model, 33 3-D fields (26 pressure levels)"
  "EXAALT"  "EXAALT"           "2869440"           "f32" "6"   "Molecular dynamics particle positions/velocities (vx/vy/vz/xx/yy/zz)"
  "HURR"    "Hurricane Isabel" "100x500x500"       "f32" "13"  "Hurricane Isabel simulation (NCAR WRF), 13 atmospheric fields"
  "HACC"    "HACC medium"      "280953867"         "f32" "6"   "N-body cosmology (HACC), 280M particles, positions/velocities"
  "NYX"     "NYX"              "512x512x512"       "f32" "6"   "AMR cosmology simulation (NYX), 6 fields"
  "MIRANDA" "Miranda"          "256x384x384"       "f64" "7"   "Turbulence simulation (Miranda), 7 fields, double precision"
  "QMCPACK" "QMCPACK"          "69x69x115x288"     "f32" "288" "Quantum Monte Carlo orbitals, 288 orbitals"
  "SCALE"   "SCALE-LETKF"      "98x1200x1200"      "f32" "13"  "SCALE-LETKF weather/climate ensemble, 13 3-D fields"
  "NWCHEM"  "NWChem"           "102953248"         "f64" "3"   "Quantum chemistry (NWChem), 1-D f64 arrays"
  "S3D"     "S3D"              "500x500x500"       "f64" "11"  "Combustion DNS (S3D), 11 3-D f64 fields"
  "BROWN"   "Brown samples"    "33554433"          "f64" "3"   "Synthetic Brownian-motion 1-D f64 samples"
  "EXAFEL"  "EXAFEL 3-D"       "130x1480x1552"     "f32" "1"   "LCLS detector images, 3-D variant (the 4-D 986x32x185x388 build is NOT used — no 4-D adapter)"
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
is_enabled() {
    local key="$1"
    for k in $ENABLED_DATASETS; do [ "$k" = "$key" ] && return 0; done
    return 1
}

get_meta() {
    local key="$1" col="$2"    # col: 1=name 2=dims 3=dtype 4=num_fields 5=description
    for (( i=0; i<${#METADATA[@]}; i+=6 )); do
        if [ "${METADATA[i]}" = "$key" ]; then
            echo "${METADATA[i+col]}"
            return
        fi
    done
}

write_metadata() {
    local key="$1" dest="$2" url="$3"
    cat > "$dest/metadata.yaml" <<EOF
dataset:     $(get_meta "$key" 1)
key:         $key
dims:        $(get_meta "$key" 2)
dtype:       $(get_meta "$key" 3)
num_fields:  $(get_meta "$key" 4)
description: $(get_meta "$key" 5)
source:      SDRBench (https://sdrbench.github.io)
url:         $BASE_URL/$url
EOF
}

download_and_extract() {
    local key="$1" tarball="$2" url_suffix="$3" dest_name="$4"
    local dest_dir="$DATA_DIR/$dest_name"

    if ! is_enabled "$key"; then
        echo "[skip]     $key (not in ENABLED_DATASETS)"
        return 0
    fi

    # Skip if the directory already exists, is non-empty, and has a metadata file.
    if [ -f "$dest_dir/metadata.yaml" ] && [ -n "$(ls -A "$dest_dir" 2>/dev/null)" ]; then
        echo "[skip]     $dest_name — already populated"
        return 0
    fi

    local tarpath="$TARBALL_DIR/$tarball"
    if [ ! -f "$tarpath" ]; then
        echo "[download] $tarball"
        wget -c -P "$TARBALL_DIR" "$BASE_URL/$url_suffix"
    else
        echo "[cached]   $tarball"
    fi

    # Extract into a temp dir so the tarball's internal naming doesn't leak into DATA_DIR.
    local tmpdir
    tmpdir=$(mktemp -d -p "$DATA_DIR" ".tmp_XXXXXX")
    echo "[extract]  $tarball -> $dest_name/"
    tar -C "$tmpdir" -xf "$tarpath"

    # If the tarball has exactly one top-level directory, hoist its contents
    # so dest_dir holds the files directly rather than a nested subdirectory.
    mkdir -p "$dest_dir"
    local all=("$tmpdir"/*)
    if [ ${#all[@]} -eq 1 ] && [ -d "${all[0]}" ]; then
        mv "${all[0]}"/* "$dest_dir"/
    else
        mv "$tmpdir"/* "$dest_dir"/
    fi
    rm -rf "$tmpdir"

    write_metadata "$key" "$dest_dir" "$url_suffix"
    echo "[done]     $dest_name/ ($(find "$dest_dir" -type f ! -name metadata.yaml | wc -l) data files)"
}

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
for (( i=0; i<${#DATASETS[@]}; i+=4 )); do
    download_and_extract \
        "${DATASETS[i]}"   \
        "${DATASETS[i+1]}" \
        "${DATASETS[i+2]}" \
        "${DATASETS[i+3]}"
done

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
echo ""
echo "[summary] $DATA_DIR:"
for d in "$DATA_DIR"/*/; do
    [ -d "$d" ] || continue
    count=$(find "$d" -type f ! -name metadata.yaml 2>/dev/null | wc -l)
    meta=""
    [ -f "$d/metadata.yaml" ] && meta="  ($(grep '^dataset:' "$d/metadata.yaml" | cut -d: -f2- | xargs))"
    printf "  %-35s %d file(s)%s\n" "$(basename "$d")" "$count" "$meta"
done
echo ""
echo "Tarballs cached in: $TARBALL_DIR"
echo "Delete once verified: rm -rf \"$TARBALL_DIR\""
