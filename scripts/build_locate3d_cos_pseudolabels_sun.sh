#!/usr/bin/env bash
set -euo pipefail

# LocateAnything3D-style pseudo-label route.
#
# This is a standalone post-processing route:
#   existing Omni3D pseudo labels
#     -> 2D evidence first
#     -> near-to-far curriculum
#     -> center/depth/size/yaw factorized weights
#     -> new Omni3D JSON
#
# It does not modify or rerun MoCA3D-style, Detic-fusion, ZEM, SOR, Boxer, or
# GroundingSAM/UniDepth generation.

SOURCE_DIR="${SOURCE_DIR:-datasets/Omni3D_pl-ng-weighted}"
OUT_DIR="${OUT_DIR:-datasets/Omni3D_pl-locate3d-cos}"
STATS_DIR="${STATS_DIR:-outputs/locate3d_cos_pseudolabel_stats}"
COMBINE_MODE="${COMBINE_MODE:-multiply}"

MAX_IMAGES_ARGS=()
if [[ -n "${MAX_IMAGES:-}" ]]; then
  MAX_IMAGES_ARGS=(--max_images "${MAX_IMAGES}")
fi

mkdir -p "${OUT_DIR}" "${STATS_DIR}"

for SPLIT in train val; do
  python tools/build_locate3d_cos_pseudolabels.py \
    --source_json "${SOURCE_DIR}/SUNRGBD_${SPLIT}.json" \
    --output_json "${OUT_DIR}/SUNRGBD_${SPLIT}.json" \
    --stats_json "${STATS_DIR}/SUNRGBD_${SPLIT}_stats.json" \
    --combine_mode "${COMBINE_MODE}" \
    --near_z "${LOCATE3D_NEAR_Z:-1.0}" \
    --far_z "${LOCATE3D_FAR_Z:-6.0}" \
    --min_near_weight "${LOCATE3D_MIN_NEAR_WEIGHT:-0.55}" \
    --xy_floor "${LOCATE3D_XY_FLOOR:-0.80}" \
    --z_floor "${LOCATE3D_Z_FLOOR:-0.75}" \
    --dims_floor "${LOCATE3D_DIMS_FLOOR:-0.45}" \
    --pose_floor "${LOCATE3D_POSE_FLOOR:-0.25}" \
    --joint_floor "${LOCATE3D_JOINT_FLOOR:-0.40}" \
    "${MAX_IMAGES_ARGS[@]}"
done
