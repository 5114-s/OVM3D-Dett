#!/usr/bin/env bash
set -euo pipefail

# Distributional latent-box pseudo labels for SUNRGBD.
#
# Main idea:
#   stable ng-weighted/source pseudo box = main JSON target
#   geometry search candidates = latent distribution stored per annotation
#   training uses soft-min distributional supervision instead of hard replacing
#   the source 3D box.

OUT_DIR="${OUT_DIR:-datasets/Omni3D_pl-distributional-anchor}"
STATS_DIR="${STATS_DIR:-outputs/distributional_anchor_stats}"
SOURCE_DIR="${SOURCE_DIR:-datasets/Omni3D_pl-ng-weighted}"
REFERENCE_DIR="${REFERENCE_DIR:-datasets/Omni3D_pl-ng-weighted}"
PSEUDO_ROOT="${PSEUDO_ROOT:-pseudo_label}"

MAX_IMAGES_ARGS=()
if [[ -n "${MAX_IMAGES:-}" ]]; then
  MAX_IMAGES_ARGS=(--max_images "${MAX_IMAGES}")
fi

mkdir -p "${OUT_DIR}" "${STATS_DIR}"

COMMON_ARGS=(
  --pseudo_root "${PSEUDO_ROOT}"
  --dataset SUNRGBD
  --drop_invalid_annotations
  --min_2d_score "${MIN_2D_SCORE:-0.0}"
  --min_2d_area_ratio "${MIN_2D_AREA_RATIO:-0.0}"
  --max_2d_area_ratio "${MAX_2D_AREA_RATIO:-0.85}"
  --min_mask_fill_ratio "${MIN_MASK_FILL_RATIO:-0.02}"
  --max_mask_fill_ratio "${MAX_MASK_FILL_RATIO:-1.35}"

  --mask_erode_vertical 8
  --mask_erode_vertical_min 1
  --mask_erode_horizontal 4
  --mask_erode_horizontal_min 1
  --min_mask_pixels 8
  --min_points 10
  --max_pca_points "${MAX_PCA_POINTS:-700}"

  --use_depth_edge_filter
  --depth_edge_rel_threshold 0.03
  --depth_edge_dilate 1
  --depth_edge_min_keep_ratio 0.35
  --use_depth_aware_mask_selector
  --mask_selector_mad_scale 3.0
  --mask_selector_min_depth_window 0.06
  --mask_selector_min_keep_ratio 0.18
  --mask_selector_min_score_gain 0.04

  --dfu_depth_percentile_low 3.0
  --dfu_depth_percentile_high 97.0
  --dfu_thin_depth_percentile_low 0.5
  --dfu_thin_depth_percentile_high 99.5
  --dfu_mad_scale 4.0
  --dfu_thin_mad_scale 6.0
  --dfu_min_depth_window 0.08
  --dfu_axis_percentile_low 1.0
  --dfu_axis_percentile_high 99.0
  --use_frustum_dbscan
  --dbscan_eps_ratio 0.018
  --dbscan_eps_min 0.035
  --dbscan_min_samples 8
  --dbscan_min_cluster_points 12
  --dbscan_min_keep_ratio 0.08
  --use_normal_ground_fusion
  --normal_stride 8
  --normal_min_count 50
  --normal_min_vertical_dot 0.65
  --normal_fusion_weight 0.12

  --use_source_geometry_anchor
  --use_surface_box_optimization
  --use_geometry_verified_cuboid_optimizer
  --use_latent_box_closure
  --latent_store_candidates
  --latent_keep_anchor_as_main
  --latent_topk "${LATENT_TOPK:-8}"
  --latent_temperature "${LATENT_TEMPERATURE:-0.25}"
  --surface_center_mode "${SURFACE_CENTER_MODE:-locked}"
  --surface_depth_percentile 35
  --surface_max_shift_ratio 0.12
  --surface_yaw_delta_deg "${SURFACE_YAW_DELTA_DEG:-0}"
  --surface_scale_delta "${SURFACE_SCALE_DELTA:-0.0}"
  --surface_height_scale_delta "${SURFACE_HEIGHT_SCALE_DELTA:-0.0}"
  --surface_include_right_angle_yaws
  --surface_enable_dims_swap
  --gvo_yaw_delta_deg "${GVO_YAW_DELTA_DEG:-0}"
  --gvo_scale_delta "${GVO_SCALE_DELTA:-0.0}"
  --gvo_height_scale_delta "${GVO_HEIGHT_SCALE_DELTA:-0.0}"
  --gvo_xy_blends "${GVO_XY_BLENDS:-0.0}"
  --gvo_depth_blends "${GVO_DEPTH_BLENDS:-0.0}"
  --gvo_prior_blends "${GVO_PRIOR_BLENDS:-0.0,0.25}"
  --gvo_include_prior_dims

  --reference_min_iou 0.10
  --min_weight "${MIN_WEIGHT:-0.42}"
  --unmatched_weight "${UNMATCHED_WEIGHT:-0.62}"
  --use_imov3d_quality_weight
  --imov3d_quality_min 0.76
  --imov3d_prior_quality_strength 0.22
  --imov3d_cluster_fallback_weight 0.92
  --imov3d_normal_missing_weight 0.97
  --use_projected_corner_depth_score
  --pag_front_depth_percentile 35
  --pag_corner_depth_radius 2
  --pag_min_corner_depth_samples 1
  --pag_min_score 0.64
  --pag_apply_to_weight
  --pag_weight_strength 0.25
  --use_locate3d_factorized_curriculum
  --curriculum_xy_floor 0.76
  --curriculum_z_floor 0.76
  --curriculum_dims_floor 0.56
  --curriculum_pose_floor 0.38
  --curriculum_joint_floor 0.48
  --use_classwise_attribute_weight
  --classwise_xy_weight_floor 0.76
  --classwise_z_weight_floor 0.76
  --thin_dims_weight_cap 0.52
  --thin_pose_weight_cap 0.32
  --small_object_area_ratio 0.002
  --small_dims_weight_cap 0.52
  --small_pose_weight_cap 0.32
  --use_bev_nms
  --bev_nms_iou_threshold "${BEV_NMS_IOU_THRESHOLD:-0.80}"
  --bev_nms_score_weight "${BEV_NMS_SCORE_WEIGHT:-0.30}"
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
