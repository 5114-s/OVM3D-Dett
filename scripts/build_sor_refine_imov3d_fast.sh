#!/usr/bin/env bash
set -euo pipefail

OUT_DIR="${OUT_DIR:-datasets/Omni3D_pl-sor-imov3d-fast}"
STATS_DIR="${STATS_DIR:-outputs/sor_imov3d_fast_stats}"
SOURCE_DIR="${SOURCE_DIR:-datasets/Omni3D_pl-1}"
REFERENCE_DIR="${REFERENCE_DIR:-datasets/Omni3D_pl-ng-weighted}"
IMOV3D_QUALITY_MIN="${IMOV3D_QUALITY_MIN:-0.65}"
IMOV3D_PRIOR_QUALITY_STRENGTH="${IMOV3D_PRIOR_QUALITY_STRENGTH:-0.45}"
BEV_NMS_IOU_THRESHOLD="${BEV_NMS_IOU_THRESHOLD:-0.65}"
BEV_NMS_SCORE_WEIGHT="${BEV_NMS_SCORE_WEIGHT:-0.35}"
NORMAL_FUSION_WEIGHT="${NORMAL_FUSION_WEIGHT:-0.25}"
MAX_IMAGES_ARGS=()
if [[ -n "${MAX_IMAGES:-}" ]]; then
  MAX_IMAGES_ARGS=(--max_images "${MAX_IMAGES}")
fi

mkdir -p "${OUT_DIR}" "${STATS_DIR}"

# ImOV3D-style SOR refine:
#   1) depth/normal gravity revision,
#   2) density clustering to remove background points,
#   3) class-prior quality weighting,
# while keeping the fast yaw90/dims-swap surface optimization profile.
COMMON_ARGS=(
  --pseudo_root pseudo_label
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
  --use_frustum_dbscan
  --dbscan_eps_ratio 0.020
  --dbscan_eps_min 0.040
  --dbscan_min_samples 6
  --dbscan_max_fit_points 1200
  --dbscan_min_cluster_points 10
  --dbscan_min_keep_ratio 0.06
  --use_normal_ground_fusion
  --normal_stride 4
  --normal_min_count 80
  --normal_min_vertical_dot 0.65
  --normal_fusion_weight "${NORMAL_FUSION_WEIGHT}"
  --use_source_geometry_anchor
  --use_surface_box_optimization
  --surface_center_mode conservative
  --surface_depth_percentile 35
  --surface_max_shift_ratio 0.08
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
  --use_imov3d_quality_weight
  --imov3d_quality_min "${IMOV3D_QUALITY_MIN}"
  --imov3d_prior_quality_strength "${IMOV3D_PRIOR_QUALITY_STRENGTH}"
  --imov3d_cluster_fallback_weight 0.88
  --imov3d_normal_missing_weight 0.95
  --use_bev_nms
  --bev_nms_iou_threshold "${BEV_NMS_IOU_THRESHOLD}"
  --bev_nms_score_weight "${BEV_NMS_SCORE_WEIGHT}"
)

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
