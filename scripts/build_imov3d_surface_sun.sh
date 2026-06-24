#!/usr/bin/env bash
set -euo pipefail

OUT_DIR="${OUT_DIR:-datasets/Omni3D_pl-imov3d-surface}"
STATS_DIR="${STATS_DIR:-outputs/imov3d_surface_stats}"
MAX_IMAGES_ARGS=()
if [[ -n "${MAX_IMAGES:-}" ]]; then
  MAX_IMAGES_ARGS=(--max_images "${MAX_IMAGES}")
fi

mkdir -p "${OUT_DIR}" "${STATS_DIR}"

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
  --use_normal_ground_fusion
  --normal_stride 4
  --normal_min_count 80
  --normal_min_vertical_dot 0.65
  --normal_fusion_weight 0.35
  --use_surface_box_optimization
  --use_source_geometry_anchor
  --surface_center_mode locked
  --surface_depth_percentile 35
  --surface_max_shift_ratio 0.20
  --surface_yaw_delta_deg 10
  --surface_scale_delta 0.10
  --surface_height_scale_delta 0.05
  --surface_projection_weight 1.5
  --surface_silhouette_weight 2.0
  --surface_depth_weight 2.0
  --surface_support_weight 2.0
  --surface_prior_weight 0.15
  --surface_min_point_support 0.25
  --surface_min_support_ratio 0.75
)

MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mplconfig}" python tools/build_dfu_robust_pca_pseudolabels.py \
  --source_json datasets/Omni3D_pl-1/SUNRGBD_train.json \
  --split train \
  --output_json "${OUT_DIR}/SUNRGBD_train.json" \
  --stats_json "${STATS_DIR}/SUNRGBD_train_stats.json" \
  "${COMMON_ARGS[@]}" \
  "${MAX_IMAGES_ARGS[@]}"

MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mplconfig}" python tools/build_dfu_robust_pca_pseudolabels.py \
  --source_json datasets/Omni3D_pl-1/SUNRGBD_val.json \
  --split val \
  --output_json "${OUT_DIR}/SUNRGBD_val.json" \
  --stats_json "${STATS_DIR}/SUNRGBD_val_stats.json" \
  "${COMMON_ARGS[@]}" \
  "${MAX_IMAGES_ARGS[@]}"
