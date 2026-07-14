#!/usr/bin/env bash
set -euo pipefail

# Route A:
#   Train LiftHead only on base categories with true Omni3D/SUNRGBD 3D boxes.
#   Novel categories are excluded from 3D supervision and used only for testing.

OUT_DIR="${OUT_DIR:-outputs/supervised_boxer_lifthead_sun_base_novel}"
SOURCE_TRAIN_JSON="${SOURCE_TRAIN_JSON:-outputs/boxer_original_gsam_source_sun/SUNRGBD_train_boxer_original_gsam.json}"
SOURCE_VAL_JSON="${SOURCE_VAL_JSON:-outputs/boxer_original_gsam_source_sun/SUNRGBD_val_boxer_original_gsam.json}"
TARGET_TRAIN_JSON="${TARGET_TRAIN_JSON:-datasets/Omni3D/SUNRGBD_train.json}"
TARGET_VAL_JSON="${TARGET_VAL_JSON:-datasets/Omni3D/SUNRGBD_val.json}"
IMAGE_ROOT="${IMAGE_ROOT:-datasets}"
DEPTH_ROOT="${DEPTH_ROOT:-pseudo_label/SUNRGBD}"
MIN_MATCH_IOU="${MIN_MATCH_IOU:-0.30}"
ROI_FEATURE_MODE="${ROI_FEATURE_MODE:-color_grid}"
ROI_GRID_SIZE="${ROI_GRID_SIZE:-4}"
DEPTH_FEATURE_MODE="${DEPTH_FEATURE_MODE:-box_stats}"

NOVEL_CATEGORIES="${NOVEL_CATEGORIES:-monitor,bag,dresser,board,printer,keyboard,painting,drawers,microwave,computer,kitchen pan,potted plant,tissues,rack,tray,toys,phone,podium,cart,soundsystem,fire place,tram}"
BASE_EXCLUDE_CATEGORIES="${BASE_EXCLUDE_CATEGORIES:-${NOVEL_CATEGORIES}}"

mkdir -p "${OUT_DIR}"

python tools/prepare_lifthead_data.py \
  --source_json "${SOURCE_TRAIN_JSON}" \
  --target_json "${TARGET_TRAIN_JSON}" \
  --image_root "${IMAGE_ROOT}" \
  --output "${OUT_DIR}/train_pairs.pth" \
  --min_match_iou "${MIN_MATCH_IOU}" \
  --roi_feature_mode "${ROI_FEATURE_MODE}" \
  --roi_grid_size "${ROI_GRID_SIZE}" \
  --depth_root "${DEPTH_ROOT}" \
  --depth_split train \
  --depth_feature_mode "${DEPTH_FEATURE_MODE}" \
  --exclude_categories "${BASE_EXCLUDE_CATEGORIES}"

python tools/prepare_lifthead_data.py \
  --source_json "${SOURCE_VAL_JSON}" \
  --target_json "${TARGET_VAL_JSON}" \
  --image_root "${IMAGE_ROOT}" \
  --output "${OUT_DIR}/val_pairs.pth" \
  --min_match_iou "${MIN_MATCH_IOU}" \
  --roi_feature_mode "${ROI_FEATURE_MODE}" \
  --roi_grid_size "${ROI_GRID_SIZE}" \
  --depth_root "${DEPTH_ROOT}" \
  --depth_split val \
  --depth_feature_mode "${DEPTH_FEATURE_MODE}" \
  --exclude_categories "${BASE_EXCLUDE_CATEGORIES}"

cat > "${OUT_DIR}/split.txt" <<EOF
base = all categories except:
${BASE_EXCLUDE_CATEGORIES}

novel =
${NOVEL_CATEGORIES}
EOF
