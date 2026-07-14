#!/usr/bin/env bash
set -euo pipefail

SOURCE_DIR="${SOURCE_DIR:-datasets/Omni3D_pl-1}"
OUTPUT_DIR="${OUTPUT_DIR:-datasets/Omni3D_pl-candidate-consistency-only}"
CONDA_ENV="${CONDA_ENV:-ovm3d-1}"

if [[ -n "${CONDA_ENV}" ]]; then
  PYTHON_CMD=(conda run --no-capture-output -n "${CONDA_ENV}" python)
else
  PYTHON_CMD=(python)
fi

"${PYTHON_CMD[@]}" tools/fix_candidate_consistency_dimensions.py \
  --source_dir "${SOURCE_DIR}" \
  --output_dir "${OUTPUT_DIR}" \
  --dataset SUNRGBD \
  --splits train val \
  --tolerance 0.01

echo "Wrote dimensions-only pseudo labels to ${OUTPUT_DIR}"
