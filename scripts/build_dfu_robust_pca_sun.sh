#!/usr/bin/env bash
set -euo pipefail

OUT_DIR="datasets/Omni3D_pl-dfu-robust-pca"
STATS_DIR="outputs/dfu_robust_pca_stats"
mkdir -p "${OUT_DIR}" "${STATS_DIR}"

MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mplconfig}" python tools/build_dfu_robust_pca_pseudolabels.py \
  --source_json datasets/Omni3D_pl-1/SUNRGBD_train.json \
  --pseudo_root pseudo_label \
  --dataset SUNRGBD \
  --split train \
  --output_json "${OUT_DIR}/SUNRGBD_train.json" \
  --reference_json outputs/boxer_cached_gsam_unidepth_dfu_recall_cuda_train/SUNRGBD_train_boxer_cached_dfu_recall.json \
  --stats_json "${STATS_DIR}/SUNRGBD_train_stats.json"

MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mplconfig}" python tools/build_dfu_robust_pca_pseudolabels.py \
  --source_json datasets/Omni3D_pl-1/SUNRGBD_val.json \
  --pseudo_root pseudo_label \
  --dataset SUNRGBD \
  --split val \
  --output_json "${OUT_DIR}/SUNRGBD_val.json" \
  --reference_json outputs/boxer_cached_gsam_unidepth_dfu_recall_cuda_val/SUNRGBD_val_boxer_cached_dfu_recall.json \
  --stats_json "${STATS_DIR}/SUNRGBD_val_stats.json"
