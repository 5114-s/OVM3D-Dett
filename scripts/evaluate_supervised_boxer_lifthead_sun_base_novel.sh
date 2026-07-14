#!/usr/bin/env bash
set -euo pipefail

SOURCE_VAL_JSON="${SOURCE_VAL_JSON:-outputs/boxer_original_gsam_source_sun/SUNRGBD_val_boxer_original_gsam.json}"
LIFTED_VAL_JSON="${LIFTED_VAL_JSON:-datasets/Omni3D_pl-supervised-boxer-lift-base-novel/SUNRGBD_val.json}"
GT_VAL_JSON="${GT_VAL_JSON:-datasets/Omni3D/SUNRGBD_val.json}"
OUT_DIR="${OUT_DIR:-outputs/supervised_boxer_lifthead_sun_base_novel/eval}"
MIN_MATCH_IOU="${MIN_MATCH_IOU:-0.30}"
NOVEL_CATEGORIES="${NOVEL_CATEGORIES:-monitor,bag,dresser,board,printer,keyboard,painting,drawers,microwave,computer,kitchen pan,potted plant,tissues,rack,tray,toys,phone,podium,cart,soundsystem,fire place,tram}"

mkdir -p "${OUT_DIR}"

python tools/evaluate_lifthead_split.py \
  --pred_json "${SOURCE_VAL_JSON}" \
  --gt_json "${GT_VAL_JSON}" \
  --categories "${NOVEL_CATEGORIES}" \
  --min_match_iou "${MIN_MATCH_IOU}" \
  --output_json "${OUT_DIR}/source_boxer_novel_quality.json"

python tools/evaluate_lifthead_split.py \
  --pred_json "${LIFTED_VAL_JSON}" \
  --gt_json "${GT_VAL_JSON}" \
  --categories "${NOVEL_CATEGORIES}" \
  --min_match_iou "${MIN_MATCH_IOU}" \
  --output_json "${OUT_DIR}/lifthead_novel_quality.json"

python tools/evaluate_lifthead_split.py \
  --pred_json "${LIFTED_VAL_JSON}" \
  --gt_json "${GT_VAL_JSON}" \
  --exclude_categories "${NOVEL_CATEGORIES}" \
  --min_match_iou "${MIN_MATCH_IOU}" \
  --output_json "${OUT_DIR}/lifthead_base_quality.json"
