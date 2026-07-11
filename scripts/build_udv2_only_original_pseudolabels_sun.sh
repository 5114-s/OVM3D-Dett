#!/usr/bin/env bash
set -euo pipefail

# Build Omni3D pseudo-label JSON with the original OVM3D-Det Step 3/4:
#   cached mask + cached depth -> pseudo point cloud -> PCA/raytrace box
#   info_3d.pth -> Omni3D JSON
#
# The only intended ablation is that PSEUDO_ROOT contains UniDepthV2 depth
# instead of the original UniDepth depth.

PSEUDO_ROOT="${PSEUDO_ROOT:-pseudo_label_original_gsam_udv2}"
OUT_DIR="${OUT_DIR:-datasets/Omni3D_pl-udv2-only}"
DATASET="${DATASET:-SUN}"
DATASET_NAME="${DATASET_NAME:-SUNRGBD}"
GEN_OUTPUT_DIR="${GEN_OUTPUT_DIR:-output/generate_pseudo_label_udv2_only/${DATASET}}"

mkdir -p "${OUT_DIR}" "${GEN_OUTPUT_DIR}"

echo "[1/2] Original Step 3 PCA pseudo boxes from cache: ${PSEUDO_ROOT}"
PSEUDO_LABEL_ROOT="${PSEUDO_ROOT}" python tools/generate_pseudo_bbox.py \
  --config-file "configs/Base_Omni3D_${DATASET}.yaml" \
  OUTPUT_DIR "${GEN_OUTPUT_DIR}"

echo "[2/2] Convert info_3d.pth to Omni3D JSON -> ${OUT_DIR}"
python tools/transform_to_coco.py \
  --dataset_name "${DATASET_NAME}" \
  --input_root "${PSEUDO_ROOT}" \
  --output_dir "${OUT_DIR}"

echo "Done. JSON folder: ${OUT_DIR}"
