#!/usr/bin/env bash
set -euo pipefail

# Detic proposal route:
#   Detic 2D boxes -> SAM2 masks -> fuse into original GroundingSAM cache.
# Downstream SOR/PCA scripts can then run with:
#   PSEUDO_ROOT=pseudo_label_detic_fusion bash scripts/build_sor_refine_sun_fast.sh

OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/detic_gsam_fusion}"
FUSED_PSEUDO_ROOT="${FUSED_PSEUDO_ROOT:-pseudo_label_detic_fusion}"
BASE_PSEUDO_ROOT="${BASE_PSEUDO_ROOT:-pseudo_label}"
IMAGE_ROOT="${IMAGE_ROOT:-datasets}"
DETIC_ROOT="${DETIC_ROOT:-third_party/Detic}"
DETIC_CONFIG="${DETIC_CONFIG:-configs/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.yaml}"
DETIC_WEIGHTS="${DETIC_WEIGHTS:-models/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth}"
DETIC_SCORE="${DETIC_SCORE:-0.35}"
FUSION_SCORE="${FUSION_SCORE:-0.25}"
FUSION_IOU="${FUSION_IOU:-0.65}"
CLASS_AGNOSTIC_IOU="${CLASS_AGNOSTIC_IOU:-0.90}"
DETIC_MAX_DETECTIONS="${DETIC_MAX_DETECTIONS:-100}"

MAX_IMAGES_ARGS=()
if [[ -n "${MAX_IMAGES:-}" ]]; then
  MAX_IMAGES_ARGS=(--max_images "${MAX_IMAGES}")
fi

mkdir -p "${OUTPUT_ROOT}"

for SPLIT in train val; do
  OMNI_JSON="datasets/Omni3D/SUNRGBD_${SPLIT}.json"
  DETIC_JSON="${OUTPUT_ROOT}/SUNRGBD_${SPLIT}_detic_2d.json"
  DETIC_SAM2_JSON="${OUTPUT_ROOT}/SUNRGBD_${SPLIT}_detic_sam2.json"
  MASK_DIR="${OUTPUT_ROOT}/masks_${SPLIT}"

  python tools/run_detic_omni3d.py \
    --json_file "${OMNI_JSON}" \
    --image_root "${IMAGE_ROOT}" \
    --output_json "${DETIC_JSON}" \
    --detic_root "${DETIC_ROOT}" \
    --config_file "${DETIC_CONFIG}" \
    --weights "${DETIC_WEIGHTS}" \
    --score_threshold "${DETIC_SCORE}" \
    --max_detections_per_image "${DETIC_MAX_DETECTIONS}" \
    --gpu "${GPU:-0}" \
    "${MAX_IMAGES_ARGS[@]}"

  python tools/add_sam2_masks_to_detany3d.py \
    --detany3d_json "${DETIC_JSON}" \
    --omni3d_json "${OMNI_JSON}" \
    --image_root "${IMAGE_ROOT}" \
    --output_json "${DETIC_SAM2_JSON}" \
    --mask_dir "${MASK_DIR}" \
    --input_bbox_format xyxy \
    --output_bbox_format xyxy \
    --mask_path_relative_to absolute \
    --score_threshold "${FUSION_SCORE}" \
    --gpu "${GPU:-0}" \
    --resume \
    "${MAX_IMAGES_ARGS[@]}"

  python tools/fuse_external_2d_into_gsam_cache.py \
    --omni3d_json "${OMNI_JSON}" \
    --external_2d_json "${DETIC_SAM2_JSON}" \
    --base_pseudo_root "${BASE_PSEUDO_ROOT}" \
    --output_pseudo_root "${FUSED_PSEUDO_ROOT}" \
    --dataset SUNRGBD \
    --split "${SPLIT}" \
    --external_bbox_format xyxy \
    --score_threshold "${FUSION_SCORE}" \
    --fusion_iou_threshold "${FUSION_IOU}" \
    --class_agnostic_iou_threshold "${CLASS_AGNOSTIC_IOU}"
done
