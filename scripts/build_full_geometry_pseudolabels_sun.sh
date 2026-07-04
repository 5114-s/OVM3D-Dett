#!/usr/bin/env bash
set -euo pipefail

OUT_DIR="${OUT_DIR:-datasets/Omni3D_pl-full-geometry}"
STATS_DIR="${STATS_DIR:-outputs/full_geometry_pseudolabel_stats}"
SOURCE_DIR="${SOURCE_DIR:-datasets/Omni3D_pl-1}"
REFERENCE_DIR="${REFERENCE_DIR:-datasets/Omni3D_pl-ng-weighted}"
MAX_IMAGES_ARGS=()
if [[ -n "${MAX_IMAGES:-}" ]]; then
  MAX_IMAGES_ARGS=(--max_images "${MAX_IMAGES}")
fi

mkdir -p "${OUT_DIR}" "${STATS_DIR}"

# Full pseudo-label revision route:
#   OVM3D-Det pseudo labels
#   + ImOV3D-style soft point revision (gravity/normal, clustering, prior weight)
#   + MoCA3D-style projected corner/front-depth PAG score
#   + LocateAnything3D-style factorized curriculum pseudo weights
#
# The route is intentionally recall-preserving: labels are not hard-deleted
# except by a loose BEV NMS duplicate filter.
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
  --normal_fusion_weight 0.15
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
  --imov3d_quality_min 0.85
  --imov3d_prior_quality_strength 0.20
  --imov3d_cluster_fallback_weight 0.92
  --imov3d_normal_missing_weight 0.97
  --use_projected_corner_depth_score
  --pag_front_depth_percentile 35
  --pag_corner_depth_radius 2
  --pag_min_corner_depth_samples 1
  --pag_min_score 0.75
  --pag_apply_to_weight
  --pag_weight_strength 0.25
  --pag_store_projection
  --use_locate3d_factorized_curriculum
  --curriculum_xy_floor 0.80
  --curriculum_z_floor 0.80
  --curriculum_dims_floor 0.55
  --curriculum_pose_floor 0.35
  --curriculum_joint_floor 0.50
  --use_bev_nms
  --bev_nms_iou_threshold 0.85
  --bev_nms_score_weight 0.20
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
