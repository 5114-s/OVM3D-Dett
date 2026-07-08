#!/usr/bin/env bash
set -euo pipefail

# Complete pseudo-label route for a PCA-free Step3 trial:
#
#   cached GroundingSAM / optional fused 2D proposals
#   + cached UniDepth metric depth
#   -> depth-aware mask selector
#   -> depth-edge + DFU3D-style point filtering
#   -> ImOV3D-style normal / gravity fusion
#   -> PCA-free amodal cuboid candidate optimizer
#   -> MoCA3D-style projected corner/front-depth score
#   -> LocateAnything3D-style factorized pseudo weights
#   -> BEV NMS
#
# PCA is not used as the main estimator.  By default it is only a rare fallback
# when the amodal optimizer cannot produce any valid cuboid.

OUT_DIR="${OUT_DIR:-datasets/Omni3D_pl-amodal-cuboid}"
STATS_DIR="${STATS_DIR:-outputs/amodal_cuboid_stats}"
SOURCE_DIR="${SOURCE_DIR:-datasets/Omni3D_pl-1}"
REFERENCE_DIR="${REFERENCE_DIR:-datasets/Omni3D_pl-ng-weighted}"
PSEUDO_ROOT="${PSEUDO_ROOT:-pseudo_label}"

AMODAL_FALLBACK_TO_PCA="${AMODAL_FALLBACK_TO_PCA:-1}"
MAX_IMAGES_ARGS=()
if [[ -n "${MAX_IMAGES:-}" ]]; then
  MAX_IMAGES_ARGS=(--max_images "${MAX_IMAGES}")
fi

mkdir -p "${OUT_DIR}" "${STATS_DIR}"

COMMON_ARGS=(
  --pseudo_root "${PSEUDO_ROOT}"
  --dataset SUNRGBD
  --drop_invalid_annotations

  # Step1/2: keep original cached proposals, but make mask/depth evidence cleaner.
  --min_2d_score "${MIN_2D_SCORE:-0.0}"
  --min_2d_area_ratio "${MIN_2D_AREA_RATIO:-0.0}"
  --max_2d_area_ratio "${MAX_2D_AREA_RATIO:-1.0}"
  --min_mask_fill_ratio "${MIN_MASK_FILL_RATIO:-0.0}"
  --max_mask_fill_ratio "${MAX_MASK_FILL_RATIO:-2.0}"
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
  --dfu_axis_percentile_low 1.0
  --dfu_axis_percentile_high 99.0
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
  --core_center_blend_xy 0.15
  --core_center_blend_z 0.35

  # ImOV3D-style scene geometry correction.
  --use_frustum_dbscan
  --dbscan_eps_ratio 0.018
  --dbscan_eps_min 0.035
  --dbscan_min_samples 8
  --dbscan_min_cluster_points 12
  --dbscan_min_keep_ratio 0.08
  --use_normal_ground_fusion
  --normal_stride 6
  --normal_min_count 60
  --normal_min_vertical_dot 0.65
  --normal_fusion_weight 0.12

  # Step3 replacement: PCA-free amodal cuboid optimizer.
  --use_amodal_cuboid_optimizer
  "--amodal_yaw_candidates_deg=${AMODAL_YAW_CANDIDATES_DEG:--90,-45,0,45,90}"
  --amodal_prior_blends "${AMODAL_PRIOR_BLENDS:-0.35,0.65,1.0}"
  --amodal_scale_delta "${AMODAL_SCALE_DELTA:-0.10}"
  --amodal_height_scale_delta "${AMODAL_HEIGHT_SCALE_DELTA:-0.08}"
  --amodal_extent_scale "${AMODAL_EXTENT_SCALE:-1.10}"
  --amodal_min_visible_extent_ratio "${AMODAL_MIN_VISIBLE_EXTENT_RATIO:-0.80}"
  --amodal_xy_blends "${AMODAL_XY_BLENDS:-0.0,0.20}"
  --amodal_depth_blends "${AMODAL_DEPTH_BLENDS:-0.5,1.0}"
  --amodal_depth_percentile "${AMODAL_DEPTH_PERCENTILE:-35}"
  --amodal_projection_weight "${AMODAL_PROJECTION_WEIGHT:-2.4}"
  --amodal_silhouette_weight "${AMODAL_SILHOUETTE_WEIGHT:-1.5}"
  --amodal_depth_weight "${AMODAL_DEPTH_WEIGHT:-2.2}"
  --amodal_support_weight "${AMODAL_SUPPORT_WEIGHT:-0.9}"
  --amodal_prior_weight "${AMODAL_PRIOR_WEIGHT:-0.75}"
  --amodal_ground_weight "${AMODAL_GROUND_WEIGHT:-0.15}"
  --amodal_min_point_support "${AMODAL_MIN_POINT_SUPPORT:-0.18}"

  # Step4: quality scoring and factorized supervision.
  --reference_min_iou 0.10
  --min_weight "${MIN_WEIGHT:-0.45}"
  --unmatched_weight "${UNMATCHED_WEIGHT:-0.62}"
  --use_imov3d_quality_weight
  --imov3d_quality_min 0.78
  --imov3d_prior_quality_strength 0.22
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
  --curriculum_xy_floor 0.78
  --curriculum_z_floor 0.78
  --curriculum_dims_floor 0.58
  --curriculum_pose_floor 0.40
  --curriculum_joint_floor 0.50
  --use_classwise_attribute_weight
  --classwise_xy_weight_floor 0.78
  --classwise_z_weight_floor 0.78
  --thin_dims_weight_cap 0.55
  --thin_pose_weight_cap 0.35
  --small_object_area_ratio 0.002
  --small_dims_weight_cap 0.55
  --small_pose_weight_cap 0.35
  --use_bev_nms
  --bev_nms_iou_threshold "${BEV_NMS_IOU_THRESHOLD:-0.85}"
  --bev_nms_score_weight "${BEV_NMS_SCORE_WEIGHT:-0.25}"
)

if [[ "${AMODAL_FALLBACK_TO_PCA}" == "0" ]]; then
  COMMON_ARGS+=(--no_amodal_fallback_to_pca)
fi
if [[ "${AMODAL_STORE_CANDIDATES:-0}" == "1" ]]; then
  COMMON_ARGS+=(--amodal_store_candidates)
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
