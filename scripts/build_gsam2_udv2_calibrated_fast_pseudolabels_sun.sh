#!/usr/bin/env bash
set -euo pipefail

# Fast amodal calibrated replacement route:
#   GroundingSAM2 + UniDepthV2 cache
#   -> calibrated 2D proposal gate
#   -> DFU/PCA anchor
#   -> cheap yaw90/dims-swap surface validation
#   -> factorized pseudo weights
#
# Compared with build_gsam2_udv2_calibrated_pseudolabels_sun.sh, this avoids
# expensive per-object scale/depth/xy grid search. It is the practical default
# when generating all SUNRGBD train/val pseudo labels.

PSEUDO_ROOT="${PSEUDO_ROOT:-pseudo_label_gsam2_udv2}"
OUT_DIR="${OUT_DIR:-datasets/Omni3D_pl-gsam2-udv2-amodal-fast}"
STATS_DIR="${STATS_DIR:-outputs/gsam2_udv2_amodal_fast_stats}"
SOURCE_DIR="${SOURCE_DIR:-datasets/Omni3D}"
REFERENCE_DIR="${REFERENCE_DIR:-}"

MIN_2D_SCORE="${MIN_2D_SCORE:-0.20}"
MIN_2D_AREA_RATIO="${MIN_2D_AREA_RATIO:-0.00005}"
MAX_2D_AREA_RATIO="${MAX_2D_AREA_RATIO:-0.90}"
MIN_MASK_FILL_RATIO="${MIN_MASK_FILL_RATIO:-0.0}"
MAX_MASK_FILL_RATIO="${MAX_MASK_FILL_RATIO:-100.0}"

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
  --drop_invalid_annotations
  --mask_erode_vertical 6
  --mask_erode_vertical_min 1
  --mask_erode_horizontal 3
  --mask_erode_horizontal_min 1
  --min_mask_pixels 8
  --min_points 10
  --max_pca_points 800
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
  --mask_selector_min_keep_ratio 0.20
  --mask_selector_min_score_gain 0.06
  --use_core_extent_masks
  --extent_mask_dilate 7
  --extent_bbox_pad_ratio 0.05
  --core_center_blend_xy 0.50
  --core_center_blend_z 0.85
  --use_normal_ground_fusion
  --normal_stride 6
  --normal_min_count 60
  --normal_min_vertical_dot 0.65
  --normal_fusion_weight 0.15
  --use_surface_box_optimization
  --surface_center_mode locked
  --surface_depth_percentile 35
  --surface_max_shift_ratio 0.00
  --surface_yaw_delta_deg 0
  --surface_scale_delta 0
  --surface_height_scale_delta 0
  --surface_projection_weight 1.8
  --surface_silhouette_weight 2.0
  --surface_depth_weight 2.5
  --surface_support_weight 2.0
  --surface_prior_weight 0.20
  --surface_min_point_support 0.20
  --surface_min_support_ratio 0.80
  --surface_require_improvement
  --surface_min_loss_gain 0.03
  --surface_max_bbox_iou_drop 0.02
  --surface_max_depth_worsen_ratio 1.05
  --surface_max_support_drop 0.03
  --surface_include_right_angle_yaws
  --surface_enable_dims_swap
  --use_imov3d_quality_weight
  --min_weight 0.20
  --imov3d_quality_min 0.70
  --imov3d_prior_quality_strength 0.25
  --imov3d_cluster_fallback_weight 0.90
  --imov3d_normal_missing_weight 0.96
  --use_projected_corner_depth_score
  --pag_front_depth_percentile 35
  --pag_corner_depth_radius 2
  --pag_min_corner_depth_samples 1
  --pag_min_score 0.50
  --pag_apply_to_weight
  --pag_weight_strength 0.30
  --use_locate3d_factorized_curriculum
  --curriculum_xy_floor 0.75
  --curriculum_z_floor 0.75
  --curriculum_dims_floor 0.45
  --curriculum_pose_floor 0.25
  --curriculum_joint_floor 0.35
  --use_classwise_attribute_weight
  --classwise_xy_weight_floor 0.75
  --classwise_z_weight_floor 0.75
  --thin_dims_weight_cap 0.40
  --thin_pose_weight_cap 0.20
  --small_object_area_ratio 0.002
  --small_dims_weight_cap 0.45
  --small_pose_weight_cap 0.25
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

echo "Done. Fast amodal pseudo labels: ${OUT_DIR}"
