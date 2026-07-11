#!/usr/bin/env bash
set -euo pipefail

LIFT_DIR="${LIFT_DIR:-outputs/supervised_boxer_lifthead_sun}"
SOURCE_DIR="${SOURCE_DIR:-datasets/Omni3D_pl-ng-weighted}"
OUT_DIR="${OUT_DIR:-datasets/Omni3D_pl-supervised-boxer-lift}"
IMAGE_ROOT="${IMAGE_ROOT:-datasets}"
DEPTH_ROOT="${DEPTH_ROOT:-pseudo_label/SUNRGBD}"
CHECKPOINT="${CHECKPOINT:-${LIFT_DIR}/best.pth}"
GPU="${GPU:-0}"
BLEND="${BLEND:-0.85}"
MIN_AFTER_PROJ_IOU="${MIN_AFTER_PROJ_IOU:-0.02}"

mkdir -p "${OUT_DIR}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" python tools/infer_lifthead_omni3d.py \
  --source_json "${SOURCE_DIR}/SUNRGBD_train.json" \
  --image_root "${IMAGE_ROOT}" \
  --checkpoint "${CHECKPOINT}" \
  --output_json "${OUT_DIR}/SUNRGBD_train.json" \
  --depth_root "${DEPTH_ROOT}" \
  --depth_split train \
  --gpu "${GPU}" \
  --blend "${BLEND}" \
  --min_after_proj_iou "${MIN_AFTER_PROJ_IOU}" \
  --update_yaw

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" python tools/infer_lifthead_omni3d.py \
  --source_json "${SOURCE_DIR}/SUNRGBD_val.json" \
  --image_root "${IMAGE_ROOT}" \
  --checkpoint "${CHECKPOINT}" \
  --output_json "${OUT_DIR}/SUNRGBD_val.json" \
  --depth_root "${DEPTH_ROOT}" \
  --depth_split val \
  --gpu "${GPU}" \
  --blend "${BLEND}" \
  --min_after_proj_iou "${MIN_AFTER_PROJ_IOU}" \
  --update_yaw

