#!/usr/bin/env bash
set -euo pipefail

PL_FOLDER="${PL_FOLDER:-Omni3D_pl-gsam2-udv2-ng}"
OUTPUT_DIR="${OUTPUT_DIR:-output/training/SUN_gsam2_udv2_ng_depthreg}"
DEPTH_ROOT="${DEPTH_ROOT:-pseudo_label_gsam2_udv2/SUNRGBD}"
DIST_URL="${DIST_URL:-tcp://0.0.0.0:12394}"
NUM_GPUS="${NUM_GPUS:-2}"

# Stable comparison route:
# - upgraded pseudo labels from GroundingSAM2 + UniDepthV2
# - cached UniDepthV2 depth used for front-surface depth consistency
# - factorized pseudo weights supervise reliable attributes more strongly
# - no RSH / CoP-GS / ZEM / latentbox
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}" python tools/train_net.py \
  --config-file configs/Base_Omni3D_SUN.yaml \
  --dist-url "${DIST_URL}" \
  --num-gpus "${NUM_GPUS}" \
  DATASETS.FOLDER_NAME "${PL_FOLDER}" \
  INPUT.USE_DEPTH True \
  INPUT.DEPTH_ROOT "${DEPTH_ROOT}" \
  INPUT.DEPTH_ALLOW_SENSOR_FALLBACK False \
  INPUT.USE_PSEUDO_MASK False \
  MODEL.ROI_CUBE_HEAD.USE_PSEUDO_WEIGHT False \
  MODEL.ROI_CUBE_HEAD.USE_FACTORIZED_PSEUDO_WEIGHT True \
  MODEL.ROI_CUBE_HEAD.USE_DEPTH_ROI False \
  MODEL.ROI_CUBE_HEAD.USE_ZERO_INIT_RESIDUAL False \
  MODEL.ROI_CUBE_HEAD.USE_REGION_SEGMENTATION_HEAD False \
  MODEL.ROI_CUBE_HEAD.USE_COP_GS False \
  MODEL.ROI_CUBE_HEAD.USE_ZEM_ADAPTER False \
  MODEL.ROI_CUBE_HEAD.USE_GEOMETRY_INTERPRETER False \
  MODEL.ROI_CUBE_HEAD.USE_DEPTH_CONSISTENCY_LOSS True \
  MODEL.ROI_CUBE_HEAD.LOSS_W_DEPTH_CONSISTENCY 0.05 \
  MODEL.ROI_CUBE_HEAD.DEPTH_CONSISTENCY_MIN_PIXELS 16 \
  MODEL.ROI_CUBE_HEAD.DEPTH_CONSISTENCY_CENTER_CROP 1.0 \
  MODEL.ROI_CUBE_HEAD.DEPTH_CONSISTENCY_MODE front_surface \
  MODEL.ROI_CUBE_HEAD.DEPTH_CONSISTENCY_PERCENTILE 0.35 \
  MODEL.EMA_TEACHER.ENABLED False \
  OUTPUT_DIR "${OUTPUT_DIR}" \
  "$@"
