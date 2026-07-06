#!/usr/bin/env bash
set -euo pipefail

# Build a clean front-end cache:
#   GroundingDINO + SAM2 object masks
#   UniDepthV2 metric depth
#
# Ground masks are copied from the original OVM3D cache by default so this
# ablation isolates object mask/depth changes from ground-plane changes.

OUTPUT_ROOT="${OUTPUT_ROOT:-pseudo_label_gsam2_udv2}"
BASE_GROUND_ROOT="${BASE_GROUND_ROOT:-pseudo_label}"
SPLITS="${SPLITS:-train val}"
USE_LARGE_GDINO="${USE_LARGE_GDINO:-1}"
BOX_THRESHOLD="${BOX_THRESHOLD:-0.20}"
TEXT_THRESHOLD="${TEXT_THRESHOLD:-0.20}"
MAX_IMAGES="${MAX_IMAGES:-}"
UNIDEPTH_LOCAL_FILES_ONLY="${UNIDEPTH_LOCAL_FILES_ONLY:-1}"

MAX_IMAGES_ARGS=()
if [[ -n "${MAX_IMAGES}" ]]; then
  MAX_IMAGES_ARGS=(--max_images "${MAX_IMAGES}")
fi

GDINO_ARGS=()
if [[ "${USE_LARGE_GDINO}" == "1" ]]; then
  GDINO_ARGS=(--use_large_gdino)
fi

UNIDEPTH_CACHE_ARGS=()
if [[ "${UNIDEPTH_LOCAL_FILES_ONLY}" == "1" ]]; then
  UNIDEPTH_CACHE_ARGS=(--local_files_only)
fi

echo "[1/2] UniDepthV2 depth cache -> ${OUTPUT_ROOT}"
MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mplconfig}" \
PYTHONPATH="/data/ZhaoX/OVM3D-Dett/third_party/UniDepth_official_v2:${PYTHONPATH:-}" \
python third_party/UniDepth_official_v2/run_unidepth_ovm3d.py \
  --dataset SUNRGBD \
  --splits ${SPLITS} \
  --output_root "${OUTPUT_ROOT}" \
  --model_name unidepth-v2-vitl14 \
  --resume \
  "${UNIDEPTH_CACHE_ARGS[@]}"

echo "[2/2] GroundingSAM2 object mask cache -> ${OUTPUT_ROOT}"
MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mplconfig}" python tools/grounded_sam2_cache.py \
  --dataset SUNRGBD \
  --splits ${SPLITS} \
  --output_root "${OUTPUT_ROOT}" \
  --copy_ground_from "${BASE_GROUND_ROOT}" \
  --box_threshold "${BOX_THRESHOLD}" \
  --text_threshold "${TEXT_THRESHOLD}" \
  --resume \
  "${GDINO_ARGS[@]}" \
  "${MAX_IMAGES_ARGS[@]}"

echo "Done. Cache root: ${OUTPUT_ROOT}/SUNRGBD"
