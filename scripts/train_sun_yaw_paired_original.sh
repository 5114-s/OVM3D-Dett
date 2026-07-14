#!/usr/bin/env bash
set -euo pipefail

# E0 control paired with train_sun_prior_yaw_only.sh.
PL_FOLDER="${PL_FOLDER:-Omni3D_pl-yaw-paired-original}" \
OUTPUT_DIR="${OUTPUT_DIR:-output/training/SUN_yaw_paired_original}" \
DIST_URL="${DIST_URL:-tcp://0.0.0.0:12617}" \
bash scripts/train_sun_prior_yaw_only.sh "$@"

