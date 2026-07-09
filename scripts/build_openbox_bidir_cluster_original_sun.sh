#!/usr/bin/env bash
set -euo pipefail

OUT_DIR="${OUT_DIR:-datasets/Omni3D_pl-openbox-bidir-cluster}"
STATS_DIR="${STATS_DIR:-outputs/openbox_bidir_cluster_stats}"
PSEUDO_ROOT="${PSEUDO_ROOT:-pseudo_label}"
SOURCE_DIR="${SOURCE_DIR:-datasets/Omni3D_pl-1}"
MAX_IMAGES_ARG=()
if [[ -n "${MAX_IMAGES:-}" ]]; then
  MAX_IMAGES_ARG=(--max_images "${MAX_IMAGES}")
fi

mkdir -p "${OUT_DIR}" "${STATS_DIR}"

# Minimal verification route:
# original cached GroundingSAM + UniDepth + adaptive erosion + original PCA/raytrace/prior/ground
# plus only OpenBox-style bidirectional cluster refinement before PCA.
MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mplconfig}" python tools/build_dfu_robust_pca_pseudolabels.py \
  --source_json "${SOURCE_DIR}/SUNRGBD_train.json" \
  --pseudo_root "${PSEUDO_ROOT}" \
  --dataset SUNRGBD \
  --split train \
  --output_json "${OUT_DIR}/SUNRGBD_train.json" \
  --use_openbox_bidirectional_cluster_refine \
  --openbox_bidir_bbox_pad_ratio "${OPENBOX_BIDIR_BBOX_PAD_RATIO:-0.03}" \
  --openbox_bidir_context_dilate "${OPENBOX_BIDIR_CONTEXT_DILATE:-2}" \
  --openbox_bidir_alpha "${OPENBOX_BIDIR_ALPHA:-0.42}" \
  --openbox_bidir_beta "${OPENBOX_BIDIR_BETA:-0.35}" \
  --openbox_bidir_min_keep_ratio "${OPENBOX_BIDIR_MIN_KEEP_RATIO:-0.18}" \
  --stats_json "${STATS_DIR}/SUNRGBD_train_stats.json" \
  "${MAX_IMAGES_ARG[@]}"

MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mplconfig}" python tools/build_dfu_robust_pca_pseudolabels.py \
  --source_json "${SOURCE_DIR}/SUNRGBD_val.json" \
  --pseudo_root "${PSEUDO_ROOT}" \
  --dataset SUNRGBD \
  --split val \
  --output_json "${OUT_DIR}/SUNRGBD_val.json" \
  --use_openbox_bidirectional_cluster_refine \
  --openbox_bidir_bbox_pad_ratio "${OPENBOX_BIDIR_BBOX_PAD_RATIO:-0.03}" \
  --openbox_bidir_context_dilate "${OPENBOX_BIDIR_CONTEXT_DILATE:-2}" \
  --openbox_bidir_alpha "${OPENBOX_BIDIR_ALPHA:-0.42}" \
  --openbox_bidir_beta "${OPENBOX_BIDIR_BETA:-0.35}" \
  --openbox_bidir_min_keep_ratio "${OPENBOX_BIDIR_MIN_KEEP_RATIO:-0.18}" \
  --stats_json "${STATS_DIR}/SUNRGBD_val_stats.json" \
  "${MAX_IMAGES_ARG[@]}"
