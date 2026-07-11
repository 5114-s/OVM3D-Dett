#!/usr/bin/env bash
set -euo pipefail

PL_FOLDER="${PL_FOLDER:-Omni3D_pl-udv2-only}"
OUTPUT_DIR="${OUTPUT_DIR:-output/training/SUN_udv2_only_original_head}"
DIST_URL="${DIST_URL:-tcp://0.0.0.0:12491}"
NUM_GPUS="${NUM_GPUS:-2}"

# Original OVM3D-Det training head. The only experimental change should be the
# pseudo-label folder generated with UniDepthV2-only depth.
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}" python tools/train_net.py \
  --config-file configs/Base_Omni3D_SUN.yaml \
  --dist-url "${DIST_URL}" \
  --num-gpus "${NUM_GPUS}" \
  DATASETS.FOLDER_NAME "${PL_FOLDER}" \
  OUTPUT_DIR "${OUTPUT_DIR}" \
  "$@"
