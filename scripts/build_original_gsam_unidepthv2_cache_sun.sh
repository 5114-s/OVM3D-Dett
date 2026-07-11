#!/usr/bin/env bash
set -euo pipefail

# Clean ablation cache:
#   original OVM3D GroundingSAM masks / boxes / ground masks
#   + official UniDepthV2 metric depth
#
# This script does not run GroundingSAM2, Detic, Boxer, DFU, SOR, or any other
# pseudo-label refinement module.

OUTPUT_ROOT="${OUTPUT_ROOT:-pseudo_label_original_gsam_udv2}"
BASE_ROOT="${BASE_ROOT:-pseudo_label}"
DATASET_NAME="${DATASET_NAME:-SUNRGBD}"
SPLITS="${SPLITS:-train val}"
IMAGE_ROOT="${IMAGE_ROOT:-datasets}"
JSON_ROOT="${JSON_ROOT:-datasets/Omni3D}"
MODEL_NAME="${MODEL_NAME:-unidepth-v2-vitl14}"
UNIDEPTH_LOCAL_FILES_ONLY="${UNIDEPTH_LOCAL_FILES_ONLY:-1}"

MAX_IMAGES_ARGS=()
if [[ -n "${MAX_IMAGES:-}" ]]; then
  MAX_IMAGES_ARGS=(--max_images "${MAX_IMAGES}")
fi

UNIDEPTH_CACHE_ARGS=()
if [[ "${UNIDEPTH_LOCAL_FILES_ONLY}" == "1" ]]; then
  UNIDEPTH_CACHE_ARGS=(--local_files_only)
fi

link_if_missing() {
  local src="$1"
  local dst="$2"
  if [[ ! -e "${src}" ]]; then
    echo "Missing required original cache item: ${src}" >&2
    exit 1
  fi
  if [[ -e "${dst}" || -L "${dst}" ]]; then
    echo "Keep existing: ${dst}"
    return
  fi
  ln -s "$(realpath "${src}")" "${dst}"
}

echo "[1/2] Link original GroundingSAM cache -> ${OUTPUT_ROOT}"
for SPLIT in ${SPLITS}; do
  BASE_DIR="${BASE_ROOT}/${DATASET_NAME}/${SPLIT}"
  OUT_DIR="${OUTPUT_ROOT}/${DATASET_NAME}/${SPLIT}"
  mkdir -p "${OUT_DIR}"

  link_if_missing "${BASE_DIR}/info.pth" "${OUT_DIR}/info.pth"
  link_if_missing "${BASE_DIR}/info_ground.pth" "${OUT_DIR}/info_ground.pth"
  link_if_missing "${BASE_DIR}/mask" "${OUT_DIR}/mask"
  link_if_missing "${BASE_DIR}/ground_mask" "${OUT_DIR}/ground_mask"
done

echo "[2/2] UniDepthV2 depth cache -> ${OUTPUT_ROOT}"
MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mplconfig}" \
PYTHONPATH="/data/ZhaoX/OVM3D-Dett/third_party/UniDepth_official_v2:${PYTHONPATH:-}" \
python third_party/UniDepth_official_v2/run_unidepth_ovm3d.py \
  --dataset "${DATASET_NAME}" \
  --splits ${SPLITS} \
  --image_root "${IMAGE_ROOT}" \
  --json_root "${JSON_ROOT}" \
  --output_root "${OUTPUT_ROOT}" \
  --model_name "${MODEL_NAME}" \
  --resume \
  "${UNIDEPTH_CACHE_ARGS[@]}" \
  "${MAX_IMAGES_ARGS[@]}"

echo "Done. UniDepthV2-only cache root: ${OUTPUT_ROOT}/${DATASET_NAME}"
