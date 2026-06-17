#!/usr/bin/env bash
set -euo pipefail

CUDA_VISIBLE_DEVICES=0,1 python tools/train_net.py \
  --config-file configs/Base_Omni3D_SUN.yaml \
  --dist-url tcp://0.0.0.0:12347 \
  --num-gpus 2 \
  DATASETS.FOLDER_NAME "Omni3D_pl-ng-weighted" \
  OUTPUT_DIR output/training/SUN_ng_weighted
