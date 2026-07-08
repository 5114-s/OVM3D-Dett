#!/usr/bin/env bash
set -euo pipefail

# Detic-as-2D-recall route:
#   ng-weighted GroundingSAM pseudo labels remain the main 3D teacher
#   Detic/SAM2 fused cache adds extra foreground 2D boxes
#   external Detic boxes keep 2D supervision but get zero 3D dims/yaw loss
#   only very high-quality external boxes may keep weak center/depth loss.

OUT_DIR="${OUT_DIR:-datasets/Omni3D_pl-detic-2d-recall}"
STATS_DIR="${STATS_DIR:-outputs/detic_2d_recall_stats}"
SOURCE_DIR="${SOURCE_DIR:-datasets/Omni3D_pl-ng-weighted}"
REFERENCE_DIR="${REFERENCE_DIR:-datasets/Omni3D_pl-ng-weighted}"
PSEUDO_ROOT="${PSEUDO_ROOT:-pseudo_label_detic_fusion_strict}"

MAX_IMAGES_ARGS=()
if [[ -n "${MAX_IMAGES:-}" ]]; then
  MAX_IMAGES_ARGS=(--max_images "${MAX_IMAGES}")
fi

mkdir -p "${OUT_DIR}" "${STATS_DIR}"

COMMON_ARGS=(
  --pseudo_root "${PSEUDO_ROOT}"
  --dataset SUNRGBD
  --use_depth_edge_filter
  --depth_edge_rel_threshold 0.025
  --depth_edge_dilate 1
  --depth_edge_min_keep_ratio 0.25
  --use_depth_aware_mask_selector
  --mask_selector_mad_scale 2.5
  --mask_selector_min_depth_window 0.05
  --mask_selector_min_keep_ratio 0.18
  --mask_selector_min_score_gain 0.03
  --use_source_geometry_anchor
  --use_surface_box_optimization
  --surface_center_mode conservative
  --surface_depth_percentile 35
  --surface_max_shift_ratio 0.06
  --surface_yaw_delta_deg 0
  --surface_scale_delta 0
  --surface_height_scale_delta 0
  --surface_projection_weight 1.5
  --surface_silhouette_weight 2.0
  --surface_depth_weight 2.5
  --surface_support_weight 2.0
  --surface_prior_weight 0.15
  --surface_min_point_support 0.20
  --surface_min_support_ratio 0.80
  --surface_require_improvement
  --surface_min_loss_gain 0.04
  --surface_max_bbox_iou_drop 0.02
  --surface_max_depth_worsen_ratio 1.05
  --surface_max_support_drop 0.03
  --surface_include_right_angle_yaws
  --surface_enable_dims_swap
  --use_bev_nms
  --bev_nms_iou_threshold "${BEV_NMS_IOU_THRESHOLD:-0.65}"
  --bev_nms_score_weight "${BEV_NMS_SCORE_WEIGHT:-0.35}"
  --use_external_strict_3d
  --external_strict_sources detic,external_2d,detany3d
  --external_strict_min_score "${EXTERNAL_STRICT_MIN_SCORE:-0.35}"
  --external_strict_accept_quality "${EXTERNAL_STRICT_ACCEPT_QUALITY:-0.62}"
  --external_strict_low_quality "${EXTERNAL_STRICT_LOW_QUALITY:-0.35}"
  --external_strict_high_cap "${EXTERNAL_STRICT_HIGH_CAP:-0.72}"
  --external_strict_mid_cap "${EXTERNAL_STRICT_MID_CAP:-0.45}"
  --external_strict_low_cap "${EXTERNAL_STRICT_LOW_CAP:-0.18}"
  --external_strict_xy_floor "${EXTERNAL_STRICT_XY_FLOOR:-0.35}"
  --external_strict_z_floor "${EXTERNAL_STRICT_Z_FLOOR:-0.35}"
  --external_strict_dims_floor "${EXTERNAL_STRICT_DIMS_FLOOR:-0.12}"
  --external_strict_pose_floor "${EXTERNAL_STRICT_POSE_FLOOR:-0.08}"
  --use_external_2d_recall_only
  --external_2d_recall_promote_quality "${EXTERNAL_2D_RECALL_PROMOTE_QUALITY:-0.82}"
  --external_2d_recall_promote_xy_weight "${EXTERNAL_2D_RECALL_XY_WEIGHT:-0.20}"
  --external_2d_recall_promote_z_weight "${EXTERNAL_2D_RECALL_Z_WEIGHT:-0.20}"
  --external_2d_recall_promote_dims_weight "${EXTERNAL_2D_RECALL_DIMS_WEIGHT:-0.0}"
  --external_2d_recall_promote_pose_weight "${EXTERNAL_2D_RECALL_POSE_WEIGHT:-0.0}"
)

if [[ "${EXTERNAL_2D_RECALL_REQUIRE_ANCHOR:-0}" == "1" ]]; then
  COMMON_ARGS+=(--external_2d_recall_require_source_anchor)
fi

for SPLIT in train val; do
  REF_ARGS=()
  if [[ -f "${REFERENCE_DIR}/SUNRGBD_${SPLIT}.json" ]]; then
    REF_ARGS=(--reference_json "${REFERENCE_DIR}/SUNRGBD_${SPLIT}.json")
  fi

  MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mplconfig}" python \
    tools/build_dfu_robust_pca_pseudolabels.py \
    --source_json "${SOURCE_DIR}/SUNRGBD_${SPLIT}.json" \
    --split "${SPLIT}" \
    --output_json "${OUT_DIR}/SUNRGBD_${SPLIT}.json" \
    --stats_json "${STATS_DIR}/SUNRGBD_${SPLIT}_stats.json" \
    "${COMMON_ARGS[@]}" \
    "${REF_ARGS[@]}" \
    "${MAX_IMAGES_ARGS[@]}"
done
