#!/usr/bin/env bash
set -euo pipefail

# Anchor-preserved multi-hypothesis pseudo labels for SUNRGBD.
#
# Main idea:
#   stable PCA/source pseudo box = anchor
#   DFU/OpenBox/amodal candidates = challengers
#   unified projection-depth-support verifier decides whether a challenger can
#   edit the anchor.  By default only dims+yaw are edited, while center/depth
#   stay anchored to the original pseudo label.

OUT_DIR="${OUT_DIR:-datasets/Omni3D_pl-anchor-multihyp}"
STATS_DIR="${STATS_DIR:-outputs/anchor_multihyp_stats}"
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

  --use_openbox_context_refinement
  --openbox_context_bbox_pad_ratio "${OPENBOX_CONTEXT_BBOX_PAD_RATIO:-0.05}"
  --openbox_context_mask_dilate "${OPENBOX_CONTEXT_MASK_DILATE:-3}"
  --openbox_context_eps_ratio "${OPENBOX_CONTEXT_EPS_RATIO:-0.022}"
  --openbox_context_eps_min "${OPENBOX_CONTEXT_EPS_MIN:-0.045}"
  --openbox_context_min_cluster_points "${OPENBOX_CONTEXT_MIN_CLUSTER_POINTS:-16}"
  --openbox_context_max_points "${OPENBOX_CONTEXT_MAX_POINTS:-1000}"
  --openbox_context_core_radius "${OPENBOX_CONTEXT_CORE_RADIUS:-0.08}"
  --openbox_context_min_gain_ratio "${OPENBOX_CONTEXT_MIN_GAIN_RATIO:-0.14}"
  --openbox_context_max_expand_ratio "${OPENBOX_CONTEXT_MAX_EXPAND_RATIO:-1.6}"
  --openbox_context_depth_window_scale "${OPENBOX_CONTEXT_DEPTH_WINDOW_SCALE:-2.5}"
  --openbox_context_min_area_ratio "${OPENBOX_CONTEXT_MIN_AREA_RATIO:-0.0015}"
  --openbox_context_max_area_ratio "${OPENBOX_CONTEXT_MAX_AREA_RATIO:-0.20}"

  --use_amodal_cuboid_optimizer
  --use_physical_type_adaptive_box
  --use_openbox_adaptive_completion
  --openbox_completion_prior_ratio "${OPENBOX_COMPLETION_PRIOR_RATIO:-0.80}"
  --openbox_completion_side_shift "${OPENBOX_COMPLETION_SIDE_SHIFT:-0.45}"
  --openbox_completion_ray_margin "${OPENBOX_COMPLETION_RAY_MARGIN:-0.05}"
  --openbox_completion_max_center_shift_ratio "${OPENBOX_COMPLETION_MAX_CENTER_SHIFT_RATIO:-0.20}"
  --openbox_deformable_prior_ratio "${OPENBOX_DEFORMABLE_PRIOR_RATIO:-0.65}"
  "--amodal_yaw_candidates_deg=${AMODAL_YAW_CANDIDATES_DEG:--90,0,90}"
  --amodal_prior_blends "${AMODAL_PRIOR_BLENDS:-0.65,1.0}"
  --amodal_scale_delta "${AMODAL_SCALE_DELTA:-0.0}"
  --amodal_height_scale_delta "${AMODAL_HEIGHT_SCALE_DELTA:-0.0}"
  --amodal_extent_scale "${AMODAL_EXTENT_SCALE:-1.06}"
  --amodal_min_visible_extent_ratio "${AMODAL_MIN_VISIBLE_EXTENT_RATIO:-0.74}"
  --amodal_xy_blends "${AMODAL_XY_BLENDS:-0.0}"
  --amodal_depth_blends "${AMODAL_DEPTH_BLENDS:-1.0}"
  --amodal_depth_percentile "${AMODAL_DEPTH_PERCENTILE:-35}"
  --amodal_projection_weight "${AMODAL_PROJECTION_WEIGHT:-2.4}"
  --amodal_silhouette_weight "${AMODAL_SILHOUETTE_WEIGHT:-1.6}"
  --amodal_depth_weight "${AMODAL_DEPTH_WEIGHT:-2.2}"
  --amodal_support_weight "${AMODAL_SUPPORT_WEIGHT:-0.9}"
  --amodal_prior_weight "${AMODAL_PRIOR_WEIGHT:-0.75}"
  --amodal_ground_weight "${AMODAL_GROUND_WEIGHT:-0.15}"
  --amodal_min_point_support "${AMODAL_MIN_POINT_SUPPORT:-0.15}"

  --use_anchor_preserving_selector
  --anchor_selector_replace_mode "${ANCHOR_SELECTOR_REPLACE_MODE:-dims_yaw}"
  --anchor_selector_min_score_gain "${ANCHOR_SELECTOR_MIN_SCORE_GAIN:-0.04}"
  --anchor_selector_min_loss_gain "${ANCHOR_SELECTOR_MIN_LOSS_GAIN:-0.05}"
  --anchor_selector_min_candidate_score "${ANCHOR_SELECTOR_MIN_CANDIDATE_SCORE:-0.56}"
  --anchor_selector_max_center_shift_ratio "${ANCHOR_SELECTOR_MAX_CENTER_SHIFT_RATIO:-0.15}"
  --anchor_selector_max_depth_shift_ratio "${ANCHOR_SELECTOR_MAX_DEPTH_SHIFT_RATIO:-0.12}"
  --anchor_selector_max_bbox_iou_drop "${ANCHOR_SELECTOR_MAX_BBOX_IOU_DROP:-0.02}"
  --anchor_selector_max_support_drop "${ANCHOR_SELECTOR_MAX_SUPPORT_DROP:-0.04}"
  --anchor_selector_max_depth_worsen_ratio "${ANCHOR_SELECTOR_MAX_DEPTH_WORSEN_RATIO:-1.03}"
  --anchor_selector_preserve_source_weights
  --anchor_selector_source_weight_floor_ratio "${ANCHOR_SELECTOR_SOURCE_WEIGHT_FLOOR_RATIO:-0.95}"

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
  --use_ovscan_pair_filter
  --ovscan_main_min_quality "${OVSCAN_MAIN_MIN_QUALITY:-0.40}"
  --ovscan_external_min_quality "${OVSCAN_EXTERNAL_MIN_QUALITY:-0.58}"
  --ovscan_external_reject_quality "${OVSCAN_EXTERNAL_REJECT_QUALITY:-0.28}"
  --ovscan_weight_strength "${OVSCAN_WEIGHT_STRENGTH:-0.55}"
  --ovscan_low_quality_joint_cap "${OVSCAN_LOW_QUALITY_JOINT_CAP:-0.45}"
  --ovscan_low_quality_dims_cap "${OVSCAN_LOW_QUALITY_DIMS_CAP:-0.30}"
  --ovscan_low_quality_pose_cap "${OVSCAN_LOW_QUALITY_POSE_CAP:-0.20}"
  --use_bev_nms
  --bev_nms_iou_threshold "${BEV_NMS_IOU_THRESHOLD:-0.80}"
  --bev_nms_score_weight "${BEV_NMS_SCORE_WEIGHT:-0.30}"
)

if [[ "${OPENBOX_CONTEXT_EXTERNAL_ONLY:-1}" == "1" ]]; then
  COMMON_ARGS+=(--openbox_context_external_only)
fi
if [[ "${ANCHOR_SELECTOR_STORE_CANDIDATES:-0}" == "1" ]]; then
  COMMON_ARGS+=(--anchor_selector_store_candidates)
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
