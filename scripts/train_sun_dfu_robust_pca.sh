#!/usr/bin/env bash
set -euo pipefail

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}" python tools/train_net.py \
  --config-file configs/Base_Omni3D_SUN.yaml \
  --dist-url tcp://0.0.0.0:12348 \
  --num-gpus 2 \
  DATASETS.FOLDER_NAME "Omni3D_pl-dfu-robust-pca" \
  OUTPUT_DIR output/training/SUN_dfu_robust_pca
