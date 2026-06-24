#!/usr/bin/env bash
set -euo pipefail

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}" python tools/train_net.py \
  --config-file configs/Base_Omni3D_SUN.yaml \
  --dist-url "${DIST_URL:-tcp://0.0.0.0:12352}" \
  --num-gpus "${NUM_GPUS:-2}" \
  DATASETS.FOLDER_NAME "Omni3D_pl-imov3d-surface" \
  OUTPUT_DIR output/training/SUN_imov3d_surface
