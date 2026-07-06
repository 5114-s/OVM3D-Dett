#!/usr/bin/env bash
set -euo pipefail

# Independent pseudo-label route that does not overwrite the existing flows.
#
# Base:
#   Omni3D_pl-ng-weighted stable pseudo labels
# Added:
#   DFU3D-style point cleaning
#   ImOV3D-style normal/gravity + class-prior soft revision
#   MoCA3D-style projected-corner/front-depth PAG score
#
# Output:
#   datasets/Omni3D_pl-dfu-imov3d-moca

OUT_DIR="${OUT_DIR:-datasets/Omni3D_pl-dfu-imov3d-moca}"
STATS_DIR="${STATS_DIR:-outputs/dfu_imov3d_moca_stats}"
SOURCE_DIR="${SOURCE_DIR:-datasets/Omni3D_pl-ng-weighted}"
REFERENCE_DIR="${REFERENCE_DIR:-datasets/Omni3D_pl-ng-weighted}"
PSEUDO_ROOT="${PSEUDO_ROOT:-pseudo_label}"
IMOV3D_QUALITY_MIN="${IMOV3D_QUALITY_MIN:-0.82}"
IMOV3D_PRIOR_QUALITY_STRENGTH="${IMOV3D_PRIOR_QUALITY_STRENGTH:-0.25}"
NORMAL_FUSION_WEIGHT="${NORMAL_FUSION_WEIGHT:-0.20}"
PAG_MIN_SCORE="${PAG_MIN_SCORE:-0.70}"
PAG_WEIGHT_STRENGTH="${PAG_WEIGHT_STRENGTH:-0.25}"
MAX_IMAGES_ARGS=()
if [[ -n "${MAX_IMAGES:-}" ]]; then
  MAX_IMAGES_ARGS=(--max_images "${MAX_IMAGES}")
fi

mkdir -p "${OUT_DIR}" "${STATS_DIR}"

COMMON_ARGS=(
  --pseudo_root "${PSEUDO_ROOT}"
  --dataset SUNRGBD

  # DFU3D-style point cleaning: remove depth edges, select a depth-coherent
  # foreground mask, cluster away background/flying points, then robust-filter
  # the remaining mask point cloud before PCA/raytrace fitting.
  --use_depth_edge_filter
  --depth_edge_rel_threshold 0.025
  --depth_edge_dilate 1
  --depth_edge_min_keep_ratio 0.25
  --use_depth_aware_mask_selector
  --mask_selector_mad_scale 2.5
  --mask_selector_min_depth_window 0.05
  --mask_selector_min_keep_ratio 0.18
  --mask_selector_min_score_gain 0.03
  --dfu_use_radius_filter
  --dfu_radius 0.45
  --dfu_radius_nb_points 8
  --use_frustum_dbscan
  --dbscan_eps_ratio 0.020
  --dbscan_eps_min 0.040
  --dbscan_min_samples 6
  --dbscan_max_fit_points 1200
  --dbscan_min_cluster_points 10
  --dbscan_min_keep_ratio 0.06

  # ImOV3D-style geometry/priors: fuse cached SUNRGBD ground with depth-normal
  # gravity evidence and turn class-prior/cluster/normal reliability into soft
  # pseudo weights instead of hard-deleting labels.
  --use_normal_ground_fusion
  --normal_stride 4
  --normal_min_count 80
  --normal_min_vertical_dot 0.65
  --normal_fusion_weight "${NORMAL_FUSION_WEIGHT}"
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
  --use_imov3d_quality_weight
  --imov3d_quality_min "${IMOV3D_QUALITY_MIN}"
  --imov3d_prior_quality_strength "${IMOV3D_PRIOR_QUALITY_STRENGTH}"
  --imov3d_cluster_fallback_weight 0.92
  --imov3d_normal_missing_weight 0.97

  # MoCA3D-style projected-corner/front-depth consistency. This writes numeric
  # pag_score and moca3d_projected_corner_depth_score for gated corner aux loss.
  --use_projected_corner_depth_score
  --pag_front_depth_percentile 35
  --pag_corner_depth_radius 2
  --pag_min_corner_depth_samples 1
  --pag_min_score "${PAG_MIN_SCORE}"
  --pag_apply_to_weight
  --pag_weight_strength "${PAG_WEIGHT_STRENGTH}"
  --pag_store_projection

  # Convert the above geometry evidence into attribute-wise pseudo weights:
  # center/depth stay stronger, dims/yaw become conservative.
  --use_locate3d_factorized_curriculum
  --curriculum_xy_floor 0.80
  --curriculum_z_floor 0.80
  --curriculum_dims_floor 0.55
  --curriculum_pose_floor 0.35
  --curriculum_joint_floor 0.50
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
