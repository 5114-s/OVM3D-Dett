#!/usr/bin/env bash
set -euo pipefail

# Full replacement pseudo-label route:
#   GroundingSAM2 + UniDepthV2 cache
#   -> calibrated 2D proposal gate
#   -> less aggressive mask erosion for SAM2 masks
#   -> DFU/PCA anchor
#   -> conservative silhouette/depth surface optimization
#   -> factorized pseudo weights
#
# This does NOT copy old 3D boxes with --use_source_geometry_anchor. It is the
# fair "replace GroundingSAM + UniDepth" experiment, but with thresholds tuned
# for the new front-end instead of the original pipeline's mask/depth statistics.

PSEUDO_ROOT="${PSEUDO_ROOT:-pseudo_label_gsam2_udv2_calibrated}"
OUT_DIR="${OUT_DIR:-datasets/Omni3D_pl-gsam2-udv2-calibrated}"
STATS_DIR="${STATS_DIR:-outputs/gsam2_udv2_calibrated_stats}"
SOURCE_DIR="${SOURCE_DIR:-datasets/Omni3D}"
REFERENCE_DIR="${REFERENCE_DIR:-}"

MIN_2D_SCORE="${MIN_2D_SCORE:-0.25}"
MIN_2D_AREA_RATIO="${MIN_2D_AREA_RATIO:-0.0001}"
MAX_2D_AREA_RATIO="${MAX_2D_AREA_RATIO:-0.72}"
MIN_MASK_FILL_RATIO="${MIN_MASK_FILL_RATIO:-0.00}"
MAX_MASK_FILL_RATIO="${MAX_MASK_FILL_RATIO:-100.00}"

MAX_IMAGES_ARGS=()
if [[ -n "${MAX_IMAGES:-}" ]]; then
  MAX_IMAGES_ARGS=(--max_images "${MAX_IMAGES}")
fi

mkdir -p "${OUT_DIR}" "${STATS_DIR}"

COMMON_ARGS=(
  --pseudo_root "${PSEUDO_ROOT}"
  --dataset SUNRGBD
  --min_2d_score "${MIN_2D_SCORE}"
  --min_2d_area_ratio "${MIN_2D_AREA_RATIO}"
  --max_2d_area_ratio "${MAX_2D_AREA_RATIO}"
  --min_mask_fill_ratio "${MIN_MASK_FILL_RATIO}"
  --max_mask_fill_ratio "${MAX_MASK_FILL_RATIO}"
  --mask_erode_vertical 6
  --mask_erode_vertical_min 1
  --mask_erode_horizontal 3
  --mask_erode_horizontal_min 1
  --min_mask_pixels 16
  --min_points 16
  --max_pca_points 1200
  --dfu_depth_percentile_low 3.0
  --dfu_depth_percentile_high 97.0
  --dfu_thin_depth_percentile_low 0.5
  --dfu_thin_depth_percentile_high 99.5
  --dfu_mad_scale 4.0
  --dfu_thin_mad_scale 6.0
  --dfu_min_depth_window 0.08
  --use_depth_edge_filter
  --depth_edge_rel_threshold 0.04
  --depth_edge_dilate 1
  --depth_edge_min_keep_ratio 0.45
  --use_depth_aware_mask_selector
  --mask_selector_mad_scale 3.5
  --mask_selector_min_depth_window 0.08
  --mask_selector_min_keep_ratio 0.35
  --mask_selector_min_score_gain 0.06
  --use_normal_ground_fusion
  --normal_stride 4
  --normal_min_count 80
  --normal_min_vertical_dot 0.65
  --normal_fusion_weight 0.20
  --use_surface_box_optimization
  --surface_center_mode conservative
  --surface_depth_percentile 35
  --surface_max_shift_ratio 0.06
  --surface_yaw_delta_deg 7.5
  --surface_scale_delta 0.05
  --surface_height_scale_delta 0.03
  --surface_projection_weight 1.8
  --surface_silhouette_weight 2.0
  --surface_depth_weight 3.0
  --surface_support_weight 2.2
  --surface_prior_weight 0.20
  --surface_min_point_support 0.25
  --surface_min_support_ratio 0.85
  --surface_require_improvement
  --surface_min_loss_gain 0.05
  --surface_max_bbox_iou_drop 0.015
  --surface_max_depth_worsen_ratio 1.03
  --surface_max_support_drop 0.02
  --surface_include_right_angle_yaws
  --surface_enable_dims_swap
  --use_imov3d_quality_weight
  --imov3d_quality_min 0.70
  --imov3d_prior_quality_strength 0.25
  --imov3d_cluster_fallback_weight 0.90
  --imov3d_normal_missing_weight 0.96
  --use_projected_corner_depth_score
  --pag_front_depth_percentile 35
  --pag_corner_depth_radius 2
  --pag_min_corner_depth_samples 1
  --pag_min_score 0.65
  --pag_apply_to_weight
  --pag_weight_strength 0.20
  --use_locate3d_factorized_curriculum
  --curriculum_xy_floor 0.75
  --curriculum_z_floor 0.75
  --curriculum_dims_floor 0.50
  --curriculum_pose_floor 0.30
  --curriculum_joint_floor 0.45
  --use_bev_nms
  --bev_nms_iou_threshold 0.70
  --bev_nms_score_weight 0.25
)

for SPLIT in train val; do
  REF_ARGS=()
  if [[ -n "${REFERENCE_DIR}" && -f "${REFERENCE_DIR}/SUNRGBD_${SPLIT}.json" ]]; then
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

echo "Done. Calibrated pseudo labels: ${OUT_DIR}"
