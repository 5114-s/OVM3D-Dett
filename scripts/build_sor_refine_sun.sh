#!/usr/bin/env bash
set -euo pipefail

OUT_DIR="${OUT_DIR:-datasets/Omni3D_pl-sor-refine}"
STATS_DIR="${STATS_DIR:-outputs/sor_refine_stats}"
SOURCE_DIR="${SOURCE_DIR:-datasets/Omni3D_pl-1}"
REFERENCE_DIR="${REFERENCE_DIR:-datasets/Omni3D_pl-ng-weighted}"
MAX_IMAGES_ARGS=()
if [[ -n "${MAX_IMAGES:-}" ]]; then
  MAX_IMAGES_ARGS=(--max_images "${MAX_IMAGES}")
fi

mkdir -p "${OUT_DIR}" "${STATS_DIR}"

# SOR-refine keeps the original OVM3D pseudo label as a geometry anchor.
# The local surface/depth/yaw search can only replace it when an explicit
# projection-depth-support gate says the candidate is better.
COMMON_ARGS=(
  --pseudo_root pseudo_label
  --dataset SUNRGBD
  --use_depth_edge_filter
  --depth_edge_rel_threshold 0.025
  --depth_edge_dilate 1
  --depth_edge_min_keep_ratio 0.25
  --use_frustum_dbscan
  --dbscan_eps_ratio 0.018
  --dbscan_eps_min 0.035
  --dbscan_min_samples 8
  --dbscan_min_cluster_points 12
  --dbscan_min_keep_ratio 0.08
  --use_source_geometry_anchor
  --use_surface_box_optimization
  --surface_center_mode conservative
  --surface_depth_percentile 35
  --surface_max_shift_ratio 0.08
  --surface_yaw_delta_deg 8
  --surface_scale_delta 0.08
  --surface_height_scale_delta 0.05
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
  --use_latent_box_closure
  --latent_topk 8
  --latent_temperature 0.20
  --latent_min_attribute_weight 0.20
  --latent_max_attribute_weight 1.0
  --use_bev_nms
  --bev_nms_iou_threshold 0.65
  --bev_nms_score_weight 0.35
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
