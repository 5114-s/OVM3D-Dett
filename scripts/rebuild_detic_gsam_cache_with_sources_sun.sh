#!/usr/bin/env bash
set -euo pipefail

# Re-fuse already generated Detic+SAM2 proposal JSONs into the GroundingSAM
# cache, adding proposal source metadata. This avoids rerunning Detic/SAM2.

OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/detic_gsam_fusion}"
FUSED_PSEUDO_ROOT="${FUSED_PSEUDO_ROOT:-pseudo_label_detic_fusion_strict}"
BASE_PSEUDO_ROOT="${BASE_PSEUDO_ROOT:-pseudo_label}"
FUSION_SCORE="${FUSION_SCORE:-0.25}"
FUSION_IOU="${FUSION_IOU:-0.65}"
CLASS_AGNOSTIC_IOU="${CLASS_AGNOSTIC_IOU:-0.90}"

for SPLIT in train val; do
  OMNI_JSON="datasets/Omni3D/SUNRGBD_${SPLIT}.json"
  DETIC_SAM2_JSON="${OUTPUT_ROOT}/SUNRGBD_${SPLIT}_detic_sam2.json"

  if [[ ! -f "${DETIC_SAM2_JSON}" ]]; then
    echo "Missing ${DETIC_SAM2_JSON}; run scripts/build_detic_gsam_cache_sun.sh first." >&2
    exit 1
  fi

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
