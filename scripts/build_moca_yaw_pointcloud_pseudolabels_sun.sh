#!/usr/bin/env bash
set -euo pipefail

# Route C:
#   GroundingSAM mask/box + UniDepth point cloud
#   + MoCA3D-Cube yaw/projected-corner evidence
#   + LLM class prior
#   -> yaw-frame robust metric cuboid pseudo labels
#
# This does not use original OVM3D 3D boxes as anchors.

SOURCE_DIR="${SOURCE_DIR:-datasets/Omni3D}"
MOCA_DIR="${MOCA_DIR:-datasets/Omni3D_pl-moca3d-cube}"
OUT_DIR="${OUT_DIR:-datasets/Omni3D_pl-moca-yaw-pointcloud}"
STATS_DIR="${STATS_DIR:-outputs/moca_yaw_pointcloud_stats}"
PSEUDO_ROOT="${PSEUDO_ROOT:-pseudo_label}"
SPLITS="${SPLITS:-train val}"
MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mplconfig}"
export MPLCONFIGDIR

MAX_IMAGES_ARG=()
if [[ -n "${MAX_IMAGES:-}" ]]; then
  MAX_IMAGES_ARG=(--max_images "${MAX_IMAGES}")
fi
FALLBACK_ARG=()
if [[ "${FALLBACK_PCA_YAW:-1}" == "0" ]]; then
  FALLBACK_ARG=(--no_fallback_pca_yaw)
fi
BEV_NMS_ARG=()
if [[ "${BEV_NMS:-1}" == "0" ]]; then
  BEV_NMS_ARG=(--no_bev_nms)
fi

mkdir -p "${OUT_DIR}" "${STATS_DIR}"

for SPLIT in ${SPLITS}; do
  python tools/build_moca_yaw_pointcloud_pseudolabels.py \
    --source_json "${SOURCE_DIR}/SUNRGBD_${SPLIT}.json" \
    --moca_json "${MOCA_DIR}/SUNRGBD_${SPLIT}.json" \
    --pseudo_root "${PSEUDO_ROOT}" \
    --dataset SUNRGBD \
    --split "${SPLIT}" \
    --output_json "${OUT_DIR}/SUNRGBD_${SPLIT}.json" \
    --stats_json "${STATS_DIR}/SUNRGBD_${SPLIT}_stats.json" \
    "${FALLBACK_ARG[@]}" \
    "${BEV_NMS_ARG[@]}" \
    "${MAX_IMAGES_ARG[@]}"
done

echo "Done. MoCA-yaw point-cloud pseudo labels: ${OUT_DIR}"
