#!/usr/bin/env bash
set -euo pipefail

# Step3 replacement route:
#   original cached GroundingSAM + UniDepth proposals/depth
#   -> DFU/depth-aware point cleaning
#   -> PCA anchor only
#   -> Geometry-Verified Cuboid Optimizer (GVO)
#   -> projection/depth/ground/prior/corner-depth factorized weights
#
# This deliberately does NOT use --use_source_geometry_anchor, so the original
# PCA/raytrace box is not copied as the final answer. It is only the initial
# anchor for the new cuboid optimizer.

OUT_DIR="${OUT_DIR:-datasets/Omni3D_pl-gvo-step3}"
STATS_DIR="${STATS_DIR:-outputs/gvo_step3_stats}"
SOURCE_DIR="${SOURCE_DIR:-datasets/Omni3D_pl-1}"
REFERENCE_DIR="${REFERENCE_DIR:-datasets/Omni3D_pl-ng-weighted}"
PSEUDO_ROOT="${PSEUDO_ROOT:-pseudo_label}"
GVO_YAW_DELTA="${GVO_YAW_DELTA:-0}"
GVO_SCALE_DELTA="${GVO_SCALE_DELTA:-0.0}"
GVO_HEIGHT_SCALE_DELTA="${GVO_HEIGHT_SCALE_DELTA:-0.0}"
GVO_XY_BLENDS="${GVO_XY_BLENDS:-0.0,0.25}"
GVO_DEPTH_BLENDS="${GVO_DEPTH_BLENDS:-0.0,0.5,1.0}"
GVO_PRIOR_BLENDS="${GVO_PRIOR_BLENDS:-0.0,0.20}"
MAX_IMAGES_ARGS=()
if [[ -n "${MAX_IMAGES:-}" ]]; then
  MAX_IMAGES_ARGS=(--max_images "${MAX_IMAGES}")
fi

mkdir -p "${OUT_DIR}" "${STATS_DIR}"

COMMON_ARGS=(
  --pseudo_root "${PSEUDO_ROOT}"
  --dataset SUNRGBD
  --drop_invalid_annotations
  --mask_erode_vertical 8
  --mask_erode_vertical_min 1
  --mask_erode_horizontal 4
  --mask_erode_horizontal_min 1
  --min_mask_pixels 8
  --min_points 10
  --max_pca_points 900
  --dfu_depth_percentile_low 3.0
  --dfu_depth_percentile_high 97.0
  --dfu_thin_depth_percentile_low 0.5
  --dfu_thin_depth_percentile_high 99.5
  --dfu_mad_scale 4.0
  --dfu_thin_mad_scale 6.0
  --dfu_min_depth_window 0.08
  --use_depth_edge_filter
  --depth_edge_rel_threshold 0.03
  --depth_edge_dilate 1
  --depth_edge_min_keep_ratio 0.35
  --use_depth_aware_mask_selector
  --mask_selector_mad_scale 3.0
  --mask_selector_min_depth_window 0.06
  --mask_selector_min_keep_ratio 0.18
  --mask_selector_min_score_gain 0.04
  --use_core_extent_masks
  --extent_mask_dilate 5
  --extent_bbox_pad_ratio 0.04
  --core_center_blend_xy 0.35
  --core_center_blend_z 0.70
  --use_normal_ground_fusion
  --normal_stride 6
  --normal_min_count 60
  --normal_min_vertical_dot 0.65
  --normal_fusion_weight 0.12
  --use_geometry_verified_cuboid_optimizer
  --gvo_yaw_delta_deg "${GVO_YAW_DELTA}"
  --gvo_scale_delta "${GVO_SCALE_DELTA}"
  --gvo_height_scale_delta "${GVO_HEIGHT_SCALE_DELTA}"
  --gvo_xy_blends "${GVO_XY_BLENDS}"
  --gvo_depth_blends "${GVO_DEPTH_BLENDS}"
  --gvo_prior_blends "${GVO_PRIOR_BLENDS}"
  --gvo_require_improvement
  --gvo_min_loss_gain 0.015
  --surface_center_mode free
  --surface_depth_percentile 35
  --surface_max_shift_ratio 0.10
  --surface_yaw_delta_deg 0
  --surface_scale_delta 0
  --surface_height_scale_delta 0
  --surface_projection_weight 1.8
  --surface_silhouette_weight 2.0
  --surface_depth_weight 2.6
  --surface_support_weight 2.0
  --surface_prior_weight 0.20
  --surface_min_point_support 0.18
  --surface_min_support_ratio 0.75
  --surface_max_bbox_iou_drop 0.02
  --surface_max_depth_worsen_ratio 1.05
  --surface_max_support_drop 0.03
  --use_imov3d_quality_weight
  --min_weight 0.25
  --imov3d_quality_min 0.75
  --imov3d_prior_quality_strength 0.25
  --imov3d_cluster_fallback_weight 0.92
  --imov3d_normal_missing_weight 0.97
  --use_projected_corner_depth_score
  --pag_front_depth_percentile 35
  --pag_corner_depth_radius 2
  --pag_min_corner_depth_samples 1
  --pag_min_score 0.65
  --pag_apply_to_weight
  --pag_weight_strength 0.25
  --use_locate3d_factorized_curriculum
  --curriculum_xy_floor 0.75
  --curriculum_z_floor 0.75
  --curriculum_dims_floor 0.50
  --curriculum_pose_floor 0.30
  --curriculum_joint_floor 0.45
  --use_classwise_attribute_weight
  --classwise_xy_weight_floor 0.75
  --classwise_z_weight_floor 0.75
  --thin_dims_weight_cap 0.42
  --thin_pose_weight_cap 0.22
  --small_object_area_ratio 0.002
  --small_dims_weight_cap 0.45
  --small_pose_weight_cap 0.25
  --use_bev_nms
  --bev_nms_iou_threshold 0.85
  --bev_nms_score_weight 0.25
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

echo "Done. GVO Step3 pseudo labels: ${OUT_DIR}"
