#!/usr/bin/env bash
set -euo pipefail

PL_FOLDER="${PL_FOLDER:-Omni3D_pl-candidate-consistency-only}"
OUTPUT_DIR="${OUTPUT_DIR:-output/training/SUN_candidate_consistency_only}"
DIST_URL="${DIST_URL:-tcp://0.0.0.0:12612}"
NUM_GPUS="${NUM_GPUS:-2}"
SEED_VALUE="${SEED_VALUE:-42}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}" python tools/train_net.py \
  --config-file configs/Base_Omni3D_SUN.yaml \
  --dist-url "${DIST_URL}" \
  --num-gpus "${NUM_GPUS}" \
  DATASETS.FOLDER_NAME "${PL_FOLDER}" \
  OUTPUT_DIR "${OUTPUT_DIR}" \
  SEED "${SEED_VALUE}" \
  "$@"
