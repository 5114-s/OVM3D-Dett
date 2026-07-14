#!/usr/bin/env bash
set -euo pipefail

# Strict paired single-module ablation from the same cached masks/depth/order.
# E0: released OVM3D-Det Step 3
# E3: E0 + prior-conditioned PCA-vs-orthogonal yaw disambiguation only

DATASET="${DATASET:-SUNRGBD}"
SRC_ROOT="${SRC_ROOT:-pseudo_label}"
BASE_WORK_ROOT="${BASE_WORK_ROOT:-pseudo_label_yaw_paired_original}"
YAW_WORK_ROOT="${YAW_WORK_ROOT:-pseudo_label_prior_dual_axis_yaw}"
BASE_OUT_DIR="${BASE_OUT_DIR:-datasets/Omni3D_pl-yaw-paired-original}"
YAW_OUT_DIR="${YAW_OUT_DIR:-datasets/Omni3D_pl-prior-dual-axis-yaw-only}"
BASE_GEN_OUTPUT_DIR="${BASE_GEN_OUTPUT_DIR:-output/generate_pseudo_label/SUN_yaw_paired_original}"
YAW_GEN_OUTPUT_DIR="${YAW_GEN_OUTPUT_DIR:-output/generate_pseudo_label/SUN_prior_dual_axis_yaw_only}"
CONFIG_FILE="${CONFIG_FILE:-configs/Base_Omni3D_SUN.yaml}"
CONDA_ENV="${CONDA_ENV:-ovm3d-1}"
SEED_VALUE="${SEED_VALUE:-42}"

if [[ -n "${CONDA_ENV}" ]]; then
  PYTHON_CMD=(conda run --no-capture-output -n "${CONDA_ENV}" python)
else
  PYTHON_CMD=(python)
fi

prepare_split() {
  local work_root="$1"
  local split="$2"
  local src_dir="${SRC_ROOT}/${DATASET}/${split}"
  local dst_dir="${work_root}/${DATASET}/${split}"
  mkdir -p "${dst_dir}"
  cp "${src_dir}/info.pth" "${dst_dir}/info.pth"
  cp "${src_dir}/info_ground.pth" "${dst_dir}/info_ground.pth"
  ln -sfn "$(realpath "${src_dir}/depth")" "${dst_dir}/depth"
  ln -sfn "$(realpath "${src_dir}/mask")" "${dst_dir}/mask"
  ln -sfn "$(realpath "${src_dir}/ground_mask")" "${dst_dir}/ground_mask"
}

for split in train val; do
  prepare_split "${BASE_WORK_ROOT}" "${split}"
  prepare_split "${YAW_WORK_ROOT}" "${split}"
done

echo "[E0] Generating paired released-original yaw control"
OVM3D_USE_BIDIR_CLUSTER=0 \
OVM3D_FIX_CANDIDATE_CONSISTENCY=0 \
OVM3D_USE_PROJECTION_SELECTION=0 \
OVM3D_USE_PRIOR_YAW=0 \
OVM3D_RELEASED_SPARSE_RAY_ZERO=1 \
PSEUDO_LABEL_ROOT="${BASE_WORK_ROOT}" \
"${PYTHON_CMD[@]}" tools/generate_pseudo_bbox.py \
  --config-file "${CONFIG_FILE}" \
  OUTPUT_DIR "${BASE_GEN_OUTPUT_DIR}" \
  SEED "${SEED_VALUE}"

"${PYTHON_CMD[@]}" tools/transform_to_coco.py \
  --dataset_name "${DATASET}" \
  --input_root "${BASE_WORK_ROOT}" \
  --output_dir "${BASE_OUT_DIR}"

echo "[E3] Generating prior-dual-axis-yaw-only pseudo labels"
OVM3D_USE_BIDIR_CLUSTER=0 \
OVM3D_FIX_CANDIDATE_CONSISTENCY=0 \
OVM3D_USE_PROJECTION_SELECTION=0 \
OVM3D_USE_PRIOR_YAW=1 \
OVM3D_RELEASED_SPARSE_RAY_ZERO=1 \
PSEUDO_LABEL_ROOT="${YAW_WORK_ROOT}" \
"${PYTHON_CMD[@]}" tools/generate_pseudo_bbox.py \
  --config-file "${CONFIG_FILE}" \
  OUTPUT_DIR "${YAW_GEN_OUTPUT_DIR}" \
  SEED "${SEED_VALUE}"

"${PYTHON_CMD[@]}" tools/transform_to_coco.py \
  --dataset_name "${DATASET}" \
  --input_root "${YAW_WORK_ROOT}" \
  --output_dir "${YAW_OUT_DIR}"

"${PYTHON_CMD[@]}" tools/summarize_prior_yaw.py \
  --work-root "${YAW_WORK_ROOT}" \
  --dataset "${DATASET}" \
  --output "${YAW_OUT_DIR}/prior_yaw_report.json"

"${PYTHON_CMD[@]}" tools/audit_prior_yaw_pair.py \
  --base-dir "${BASE_OUT_DIR}" \
  --yaw-dir "${YAW_OUT_DIR}" \
  --dataset "${DATASET}" \
  --output "${YAW_OUT_DIR}/prior_yaw_audit.json"

"${PYTHON_CMD[@]}" tools/evaluate_prior_yaw_decisions_sun.py \
  --base-info "${BASE_WORK_ROOT}/${DATASET}/val/info_3d.pth" \
  --yaw-info "${YAW_WORK_ROOT}/${DATASET}/val/info_3d.pth" \
  --gt-json "datasets/Omni3D/${DATASET}_val.json" \
  --output "${YAW_OUT_DIR}/prior_yaw_gt_diagnostic_val.json"

echo "Paired datasets written:"
echo "  E0: ${BASE_OUT_DIR}"
echo "  E3: ${YAW_OUT_DIR}"
