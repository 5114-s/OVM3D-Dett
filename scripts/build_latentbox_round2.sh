#!/usr/bin/env bash
set -euo pipefail

WEIGHTS="${WEIGHTS:-output/training/SUN_latentbox/model_final.pth}"
ROUND1_DIR="${ROUND1_DIR:-datasets/Omni3D_pl-latentbox}"
TEACHER_OUTPUT="${TEACHER_OUTPUT:-outputs/latentbox_teacher}"
ANCHOR_DIR="${ANCHOR_DIR:-outputs/latentbox_teacher_anchors}"
ROUND2_DIR="${ROUND2_DIR:-datasets/Omni3D_pl-latentbox-r2}"
ROUND2_STATS="${ROUND2_STATS:-outputs/latentbox_round2_stats}"
NUM_GPUS="${NUM_GPUS:-2}"

mkdir -p "${TEACHER_OUTPUT}" "${ANCHOR_DIR}" "${ROUND2_DIR}" "${ROUND2_STATS}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}" python tools/train_net.py \
  --eval-only \
  --config-file configs/Base_Omni3D_SUN.yaml \
  --num-gpus "${NUM_GPUS}" \
  --dist-url "${DIST_URL:-tcp://0.0.0.0:12355}" \
  DATASETS.FOLDER_NAME "$(basename "${ROUND1_DIR}")" \
  DATASETS.TEST "('SUNRGBD_train','SUNRGBD_val')" \
  MODEL.WEIGHTS "${WEIGHTS}" \
  OUTPUT_DIR "${TEACHER_OUTPUT}"

for SPLIT in train val; do
  python tools/build_teacher_geometry_anchors.py \
    --source_json "${ROUND1_DIR}/SUNRGBD_${SPLIT}.json" \
    --teacher_json "${TEACHER_OUTPUT}/inference/iter_final/SUNRGBD_${SPLIT}/omni_instances_results.json" \
    --output_json "${ANCHOR_DIR}/SUNRGBD_${SPLIT}.json" \
    --min_score 0.25 \
    --min_iou 0.30 \
    --max_blend 0.35
done

SOURCE_DIR="${ANCHOR_DIR}" \
OUT_DIR="${ROUND2_DIR}" \
STATS_DIR="${ROUND2_STATS}" \
bash scripts/build_latentbox_sun.sh
