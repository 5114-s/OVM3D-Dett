#!/usr/bin/env bash
set -euo pipefail

# Control-variable training:
#   pseudo labels: Omni3D_pl-moca-yaw-pointcloud
#   training head/loss: original OVM3D-Det SUN config only
#
# No depth consistency, no factorized pseudo weights, no RSH/ZEM/CoP/GS.

PL_FOLDER="${PL_FOLDER:-Omni3D_pl-moca-yaw-pointcloud}"
OUTPUT_DIR="${OUTPUT_DIR:-output/training/SUN_moca_yaw_pointcloud_original_head}"
DIST_URL="${DIST_URL:-tcp://0.0.0.0:12506}"
NUM_GPUS="${NUM_GPUS:-2}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}" python tools/train_net.py \
  --config-file configs/Base_Omni3D_SUN.yaml \
  --dist-url "${DIST_URL}" \
  --num-gpus "${NUM_GPUS}" \
  DATASETS.FOLDER_NAME "${PL_FOLDER}" \
  OUTPUT_DIR "${OUTPUT_DIR}" \
  "$@"
