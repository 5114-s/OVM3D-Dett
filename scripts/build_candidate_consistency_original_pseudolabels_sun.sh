#!/usr/bin/env bash
set -euo pipefail

# Single-module ablation on the original OVM3D-Det pseudo-label route:
#
#   original GroundingSAM mask + UniDepth
#   -> original adaptive erosion
#   -> original PCA orientation
#   -> original candidate generation and ray/inside objective
#   -> candidate-consistent export (the only enabled ablation)
#
# The source pseudo-label cache is never modified.  Depth and mask directories
# are linked read-only into a separate work root; only the copied info files and
# the newly generated info_3d.pth live there.

DATASET="${DATASET:-SUNRGBD}"
SRC_ROOT="${SRC_ROOT:-pseudo_label}"
WORK_ROOT="${WORK_ROOT:-pseudo_label_candidate_consistency}"
OUT_DIR="${OUT_DIR:-datasets/Omni3D_pl-candidate-consistency-original}"
GEN_OUTPUT_DIR="${GEN_OUTPUT_DIR:-output/generate_pseudo_label/SUN_candidate_consistency_original}"
CONFIG_FILE="${CONFIG_FILE:-configs/Base_Omni3D_SUN.yaml}"
CONDA_ENV="${CONDA_ENV:-ovm3d-1}"

if [[ -n "${CONDA_ENV}" ]]; then
  PYTHON_CMD=(conda run --no-capture-output -n "${CONDA_ENV}" python)
else
  PYTHON_CMD=(python)
fi

prepare_split() {
  local split="$1"
  local src_dir="${SRC_ROOT}/${DATASET}/${split}"
  local dst_dir="${WORK_ROOT}/${DATASET}/${split}"
  mkdir -p "${dst_dir}"

  cp "${src_dir}/info.pth" "${dst_dir}/info.pth"
  cp "${src_dir}/info_ground.pth" "${dst_dir}/info_ground.pth"

  ln -sfn "$(realpath "${src_dir}/depth")" "${dst_dir}/depth"
  ln -sfn "$(realpath "${src_dir}/mask")" "${dst_dir}/mask"
  ln -sfn "$(realpath "${src_dir}/ground_mask")" "${dst_dir}/ground_mask"
}

prepare_split train
prepare_split val

# Explicitly disable the separate, currently experimental point-cluster module
# so this run differs from the corrected original route in exactly one place.
OVM3D_USE_BIDIR_CLUSTER=0 \
OVM3D_FIX_CANDIDATE_CONSISTENCY=1 \
PSEUDO_LABEL_ROOT="${WORK_ROOT}" \
"${PYTHON_CMD[@]}" tools/generate_pseudo_bbox.py \
  --config-file "${CONFIG_FILE}" \
  OUTPUT_DIR "${GEN_OUTPUT_DIR}"

"${PYTHON_CMD[@]}" tools/transform_to_coco.py \
  --dataset_name "${DATASET}" \
  --input_root "${WORK_ROOT}" \
  --output_dir "${OUT_DIR}"

echo "Wrote candidate-consistency pseudo labels to ${OUT_DIR}"
