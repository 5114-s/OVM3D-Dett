#!/usr/bin/env bash
set -euo pipefail

# E0 paired control for train_sun_projection_selection_only.sh.
PL_FOLDER="${PL_FOLDER:-Omni3D_pl-projection-paired-original}" \
OUTPUT_DIR="${OUTPUT_DIR:-output/training/SUN_projection_paired_original}" \
DIST_URL="${DIST_URL:-tcp://0.0.0.0:12615}" \
bash scripts/train_sun_projection_selection_only.sh "$@"

