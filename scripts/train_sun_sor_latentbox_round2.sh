#!/usr/bin/env bash
set -euo pipefail

PL_FOLDER="${PL_FOLDER:-Omni3D_pl-sor-latent-r2}" \
OUTPUT_DIR="${OUTPUT_DIR:-output/training/SUN_sor_latentbox_round2}" \
WEIGHTS="${WEIGHTS:-output/training/SUN_sor_latentbox_stage2/model_final.pth}" \
DIST_URL="${DIST_URL:-tcp://0.0.0.0:12370}" \
bash scripts/train_sun_sor_latentbox_stage2.sh "$@"
