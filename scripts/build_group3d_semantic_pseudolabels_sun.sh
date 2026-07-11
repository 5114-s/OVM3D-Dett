#!/usr/bin/env bash
set -euo pipefail

OUT_DIR="${OUT_DIR:-datasets/Omni3D_pl-group3d-semantic}"
STATS_DIR="${STATS_DIR:-outputs/group3d_semantic_stats}"
PSEUDO_ROOT="${PSEUDO_ROOT:-pseudo_label}"
SOURCE_DIR="${SOURCE_DIR:-datasets/Omni3D_pl-1}"
MAX_IMAGES_ARG=()
if [[ -n "${MAX_IMAGES:-}" ]]; then
  MAX_IMAGES_ARG=(--max_images "${MAX_IMAGES}")
fi

mkdir -p "${OUT_DIR}" "${STATS_DIR}"

# Minimal Group3D-style verification route:
# original cached GroundingSAM + UniDepth + adaptive erosion + original PCA/raytrace/prior/ground
# plus only semantic compatibility grouping/filtering on the 2D proposals.
for SPLIT in train val; do
  MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mplconfig}" python tools/build_dfu_robust_pca_pseudolabels.py \
    --source_json "${SOURCE_DIR}/SUNRGBD_${SPLIT}.json" \
    --pseudo_root "${PSEUDO_ROOT}" \
    --dataset SUNRGBD \
    --split "${SPLIT}" \
    --output_json "${OUT_DIR}/SUNRGBD_${SPLIT}.json" \
    --use_source_geometry_anchor \
    --use_semantic_compatibility_filter \
    --semantic_duplicate_iou "${SEMANTIC_DUPLICATE_IOU:-0.70}" \
    --semantic_duplicate_mask_iou "${SEMANTIC_DUPLICATE_MASK_IOU:-0.55}" \
    --semantic_incompatible_iou "${SEMANTIC_INCOMPATIBLE_IOU:-0.90}" \
    --semantic_incompatible_mask_iou "${SEMANTIC_INCOMPATIBLE_MASK_IOU:-0.80}" \
    --semantic_min_score_ratio "${SEMANTIC_MIN_SCORE_RATIO:-0.55}" \
    --semantic_min_area_ratio "${SEMANTIC_MIN_AREA_RATIO:-0.35}" \
    --stats_json "${STATS_DIR}/SUNRGBD_${SPLIT}_stats.json" \
    "${MAX_IMAGES_ARG[@]}"
done
