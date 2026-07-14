#!/usr/bin/env bash
set -euo pipefail

LIFT_DIR="${LIFT_DIR:-outputs/supervised_boxer_lifthead_sun_base_novel}"
SOURCE_TRAIN_JSON="${SOURCE_TRAIN_JSON:-outputs/boxer_original_gsam_source_sun/SUNRGBD_train_boxer_original_gsam.json}"
SOURCE_VAL_JSON="${SOURCE_VAL_JSON:-outputs/boxer_original_gsam_source_sun/SUNRGBD_val_boxer_original_gsam.json}"
OUT_DIR="${OUT_DIR:-datasets/Omni3D_pl-supervised-boxer-lift-base-novel}"
IMAGE_ROOT="${IMAGE_ROOT:-datasets}"
DEPTH_ROOT="${DEPTH_ROOT:-pseudo_label/SUNRGBD}"
CHECKPOINT="${CHECKPOINT:-${LIFT_DIR}/best.pth}"
GPU="${GPU:-0}"
BLEND="${BLEND:-0.85}"
MIN_AFTER_PROJ_IOU="${MIN_AFTER_PROJ_IOU:-0.02}"

# Leave empty to correct every category at inference.  For strict novel-only
# diagnostics set ONLY_CATEGORIES to NOVEL_CATEGORIES below.
ONLY_CATEGORIES="${ONLY_CATEGORIES:-}"
SKIP_CATEGORIES="${SKIP_CATEGORIES:-}"

mkdir -p "${OUT_DIR}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" python tools/infer_lifthead_omni3d.py \
  --source_json "${SOURCE_TRAIN_JSON}" \
  --image_root "${IMAGE_ROOT}" \
  --checkpoint "${CHECKPOINT}" \
  --output_json "${OUT_DIR}/SUNRGBD_train.json" \
  --depth_root "${DEPTH_ROOT}" \
  --depth_split train \
  --gpu "${GPU}" \
  --blend "${BLEND}" \
  --min_after_proj_iou "${MIN_AFTER_PROJ_IOU}" \
  --update_yaw \
  --only_categories "${ONLY_CATEGORIES}" \
  --skip_categories "${SKIP_CATEGORIES}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" python tools/infer_lifthead_omni3d.py \
  --source_json "${SOURCE_VAL_JSON}" \
  --image_root "${IMAGE_ROOT}" \
  --checkpoint "${CHECKPOINT}" \
  --output_json "${OUT_DIR}/SUNRGBD_val.json" \
  --depth_root "${DEPTH_ROOT}" \
  --depth_split val \
  --gpu "${GPU}" \
  --blend "${BLEND}" \
  --min_after_proj_iou "${MIN_AFTER_PROJ_IOU}" \
  --update_yaw \
  --only_categories "${ONLY_CATEGORIES}" \
  --skip_categories "${SKIP_CATEGORIES}"
