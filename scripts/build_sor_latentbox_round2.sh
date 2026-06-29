#!/usr/bin/env bash
set -euo pipefail

WEIGHTS="${WEIGHTS:-output/training/SUN_sor_latentbox_stage2/model_final.pth}"
ROUND1_DIR="${ROUND1_DIR:-datasets/Omni3D_pl-sor-refine}"
TEACHER_OUTPUT="${TEACHER_OUTPUT:-outputs/sor_latentbox_teacher}"
ANCHOR_DIR="${ANCHOR_DIR:-outputs/sor_latentbox_teacher_anchors}"
ROUND2_DIR="${ROUND2_DIR:-datasets/Omni3D_pl-sor-latent-r2}"
ROUND2_STATS="${ROUND2_STATS:-outputs/sor_latentbox_round2_stats}"
REFERENCE_DIR="${REFERENCE_DIR:-datasets/Omni3D_pl-ng-weighted}"
NUM_GPUS="${NUM_GPUS:-2}"

mkdir -p "${TEACHER_OUTPUT}" "${ANCHOR_DIR}" "${ROUND2_DIR}" "${ROUND2_STATS}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}" python tools/train_net.py \
  --eval-only \
  --config-file configs/Base_Omni3D_SUN.yaml \
  --num-gpus "${NUM_GPUS}" \
  --dist-url "${DIST_URL:-tcp://0.0.0.0:12369}" \
  DATASETS.FOLDER_NAME "$(basename "${ROUND1_DIR}")" \
  DATASETS.TEST "('SUNRGBD_train','SUNRGBD_val')" \
  INPUT.USE_DEPTH True \
  INPUT.DEPTH_ROOT pseudo_label/SUNRGBD \
  INPUT.DEPTH_ALLOW_SENSOR_FALLBACK False \
  INPUT.USE_PSEUDO_MASK True \
  INPUT.PSEUDO_MASK_ROOT pseudo_label/SUNRGBD \
  INPUT.USE_GROUND_MASK True \
  INPUT.GROUND_MASK_ROOT pseudo_label/SUNRGBD \
  MODEL.WEIGHTS "${WEIGHTS}" \
  MODEL.ROI_CUBE_HEAD.USE_GEOMETRY_INTERPRETER True \
  MODEL.ROI_CUBE_HEAD.GEOMETRY_APPLY_IN_INFERENCE True \
  MODEL.ROI_CUBE_HEAD.USE_DIFFERENTIABLE_RENDERER True \
  OUTPUT_DIR "${TEACHER_OUTPUT}"

for SPLIT in train val; do
  python tools/build_teacher_geometry_anchors.py \
    --source_json "${ROUND1_DIR}/SUNRGBD_${SPLIT}.json" \
    --teacher_json "${TEACHER_OUTPUT}/inference/iter_final/SUNRGBD_${SPLIT}/omni_instances_results.json" \
    --output_json "${ANCHOR_DIR}/SUNRGBD_${SPLIT}.json" \
    --min_score 0.35 \
    --min_iou 0.35 \
    --max_blend 0.20
done

SOURCE_DIR="${ANCHOR_DIR}" \
OUT_DIR="${ROUND2_DIR}" \
STATS_DIR="${ROUND2_STATS}" \
REFERENCE_DIR="${REFERENCE_DIR}" \
bash scripts/build_sor_refine_sun.sh
