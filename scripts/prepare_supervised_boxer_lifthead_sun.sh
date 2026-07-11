#!/usr/bin/env bash
set -euo pipefail

OUT_DIR="${OUT_DIR:-outputs/supervised_boxer_lifthead_sun}"
SOURCE_DIR="${SOURCE_DIR:-datasets/Omni3D_pl-ng-weighted}"
TARGET_DIR="${TARGET_DIR:-datasets/Omni3D}"
IMAGE_ROOT="${IMAGE_ROOT:-datasets}"
DEPTH_ROOT="${DEPTH_ROOT:-pseudo_label/SUNRGBD}"
MIN_MATCH_IOU="${MIN_MATCH_IOU:-0.30}"
ROI_FEATURE_MODE="${ROI_FEATURE_MODE:-color_grid}"
ROI_GRID_SIZE="${ROI_GRID_SIZE:-4}"
DEPTH_FEATURE_MODE="${DEPTH_FEATURE_MODE:-box_stats}"

mkdir -p "${OUT_DIR}"

python tools/prepare_lifthead_data.py \
  --source_json "${SOURCE_DIR}/SUNRGBD_train.json" \
  --target_json "${TARGET_DIR}/SUNRGBD_train.json" \
  --image_root "${IMAGE_ROOT}" \
  --output "${OUT_DIR}/train_pairs.pth" \
  --min_match_iou "${MIN_MATCH_IOU}" \
  --roi_feature_mode "${ROI_FEATURE_MODE}" \
  --roi_grid_size "${ROI_GRID_SIZE}" \
  --depth_root "${DEPTH_ROOT}" \
  --depth_split train \
  --depth_feature_mode "${DEPTH_FEATURE_MODE}"

python tools/prepare_lifthead_data.py \
  --source_json "${SOURCE_DIR}/SUNRGBD_val.json" \
  --target_json "${TARGET_DIR}/SUNRGBD_val.json" \
  --image_root "${IMAGE_ROOT}" \
  --output "${OUT_DIR}/val_pairs.pth" \
  --min_match_iou "${MIN_MATCH_IOU}" \
  --roi_feature_mode "${ROI_FEATURE_MODE}" \
  --roi_grid_size "${ROI_GRID_SIZE}" \
  --depth_root "${DEPTH_ROOT}" \
  --depth_split val \
  --depth_feature_mode "${DEPTH_FEATURE_MODE}"

