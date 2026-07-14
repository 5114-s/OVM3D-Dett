#!/usr/bin/env bash
set -euo pipefail

# Build a Boxer source from the original OVM3D cached GroundingSAM results.
# This intentionally uses --box_source original_gsam, so it reads:
#   pseudo_label/SUNRGBD/{train,val}/info.pth
# and does not run the newer online Grounding-SAM2 branch.

OUT_ROOT="${OUT_ROOT:-outputs/boxer_original_gsam_source_sun}"
IMAGE_ROOT="${IMAGE_ROOT:-datasets}"
DATASET="${DATASET:-SUNRGBD}"
ORIGINAL_PSEUDO_ROOT="${ORIGINAL_PSEUDO_ROOT:-pseudo_label}"
GPU="${GPU:-0}"
FORCE_PRECISION="${FORCE_PRECISION:-float32}"

mkdir -p "${OUT_ROOT}/train" "${OUT_ROOT}/val"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" python tools/run_boxer_omni3d.py \
  --json_file datasets/Omni3D/SUNRGBD_train.json \
  --image_root "${IMAGE_ROOT}" \
  --output_dir "${OUT_ROOT}/train" \
  --output_json "${OUT_ROOT}/SUNRGBD_train_boxer_original_gsam.json" \
  --box_source original_gsam \
  --depth_source original_unidepth \
  --dataset "${DATASET}" \
  --split train \
  --original_pseudo_root "${ORIGINAL_PSEUDO_ROOT}" \
  --gpu "${GPU}" \
  --force_precision "${FORCE_PRECISION}" \
  --thresh3d 0.01 \
  --prior_min_ratio 0.05 \
  --prior_max_ratio 12.0 \
  --prior_adjust_min_ratio 0.60 \
  --min_proj_iou 0.00 \
  --ground_max_distance 2.0 \
  --min_depth_pixels 1 \
  --min_depth_support 0.0 \
  --max_rel_depth_error 2.0 \
  --no_ground_gate \
  --stats_json "${OUT_ROOT}/train/boxer_stats.json"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" python tools/run_boxer_omni3d.py \
  --json_file datasets/Omni3D/SUNRGBD_val.json \
  --image_root "${IMAGE_ROOT}" \
  --output_dir "${OUT_ROOT}/val" \
  --output_json "${OUT_ROOT}/SUNRGBD_val_boxer_original_gsam.json" \
  --box_source original_gsam \
  --depth_source original_unidepth \
  --dataset "${DATASET}" \
  --split val \
  --original_pseudo_root "${ORIGINAL_PSEUDO_ROOT}" \
  --gpu "${GPU}" \
  --force_precision "${FORCE_PRECISION}" \
  --thresh3d 0.01 \
  --prior_min_ratio 0.05 \
  --prior_max_ratio 12.0 \
  --prior_adjust_min_ratio 0.60 \
  --min_proj_iou 0.00 \
  --ground_max_distance 2.0 \
  --min_depth_pixels 1 \
  --min_depth_support 0.0 \
  --max_rel_depth_error 2.0 \
  --no_ground_gate \
  --stats_json "${OUT_ROOT}/val/boxer_stats.json"
