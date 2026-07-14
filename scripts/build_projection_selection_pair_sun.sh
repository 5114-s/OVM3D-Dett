#!/usr/bin/env bash
set -euo pipefail

# Paired single-module ablation built from the exact same cached 2D detections,
# masks, depth maps, image order, and RNG seed.
#
# E0: released OVM3D-Det indoor box generation
# E2: E0 + depth-aware projective candidate selection only

DATASET="${DATASET:-SUNRGBD}"
SRC_ROOT="${SRC_ROOT:-pseudo_label}"
BASE_WORK_ROOT="${BASE_WORK_ROOT:-pseudo_label_projection_paired_original}"
PROJ_WORK_ROOT="${PROJ_WORK_ROOT:-pseudo_label_projection_selection}"
BASE_OUT_DIR="${BASE_OUT_DIR:-datasets/Omni3D_pl-projection-paired-original}"
PROJ_OUT_DIR="${PROJ_OUT_DIR:-datasets/Omni3D_pl-projection-selection-only}"
BASE_GEN_OUTPUT_DIR="${BASE_GEN_OUTPUT_DIR:-output/generate_pseudo_label/SUN_projection_paired_original}"
PROJ_GEN_OUTPUT_DIR="${PROJ_GEN_OUTPUT_DIR:-output/generate_pseudo_label/SUN_projection_selection_only}"
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
  prepare_split "${PROJ_WORK_ROOT}" "${split}"
done

echo "[E0] Generating paired released-original pseudo labels"
OVM3D_USE_BIDIR_CLUSTER=0 \
OVM3D_FIX_CANDIDATE_CONSISTENCY=0 \
OVM3D_USE_PROJECTION_SELECTION=0 \
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

echo "[E2] Generating projection-selection-only pseudo labels"
OVM3D_USE_BIDIR_CLUSTER=0 \
OVM3D_FIX_CANDIDATE_CONSISTENCY=0 \
OVM3D_USE_PROJECTION_SELECTION=1 \
OVM3D_RELEASED_SPARSE_RAY_ZERO=1 \
PSEUDO_LABEL_ROOT="${PROJ_WORK_ROOT}" \
"${PYTHON_CMD[@]}" tools/generate_pseudo_bbox.py \
  --config-file "${CONFIG_FILE}" \
  OUTPUT_DIR "${PROJ_GEN_OUTPUT_DIR}" \
  SEED "${SEED_VALUE}"

"${PYTHON_CMD[@]}" tools/transform_to_coco.py \
  --dataset_name "${DATASET}" \
  --input_root "${PROJ_WORK_ROOT}" \
  --output_dir "${PROJ_OUT_DIR}"

"${PYTHON_CMD[@]}" tools/summarize_projection_selection.py \
  --work-root "${PROJ_WORK_ROOT}" \
  --dataset "${DATASET}" \
  --output "${PROJ_OUT_DIR}/projection_selection_report.json"

"${PYTHON_CMD[@]}" tools/audit_projection_selection_pair.py \
  --base-dir "${BASE_OUT_DIR}" \
  --projection-dir "${PROJ_OUT_DIR}" \
  --dataset "${DATASET}" \
  --output "${PROJ_OUT_DIR}/projection_selection_audit.json"

echo "Paired datasets written:"
echo "  E0: ${BASE_OUT_DIR}"
echo "  E2: ${PROJ_OUT_DIR}"

