#!/usr/bin/env bash
set -euo pipefail

# Original OVM3D-Det pseudo-label route with only one inserted module:
#   GroundingSAM mask + UniDepth
#   -> original adaptive erosion
#   -> bidirectional instance/context point-cloud cluster filtering
#   -> original PCA/raytrace box estimation
#
# The original pseudo_label cache is not modified.  This script creates a
# lightweight cache root with symlinks to depth/mask/ground_mask and writes the
# new info_3d.pth there.

DATASET="${DATASET:-SUNRGBD}"
SRC_ROOT="${SRC_ROOT:-pseudo_label}"
WORK_ROOT="${WORK_ROOT:-pseudo_label_bidir_cluster}"
OUT_DIR="${OUT_DIR:-datasets/Omni3D_pl-bidir-cluster-original}"
GEN_OUTPUT_DIR="${GEN_OUTPUT_DIR:-output/generate_pseudo_label/SUN_bidir_cluster_original}"
CONFIG_FILE="${CONFIG_FILE:-configs/Base_Omni3D_SUN.yaml}"

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

OVM3D_USE_BIDIR_CLUSTER=1 \
OVM3D_CLUSTER_EPS="${OVM3D_CLUSTER_EPS:-0.18}" \
OVM3D_CLUSTER_MIN_SAMPLES="${OVM3D_CLUSTER_MIN_SAMPLES:-6}" \
OVM3D_CLUSTER_DELTA="${OVM3D_CLUSTER_DELTA:-0.14}" \
OVM3D_CLUSTER_ALPHA="${OVM3D_CLUSTER_ALPHA:-0.20}" \
OVM3D_CLUSTER_BETA="${OVM3D_CLUSTER_BETA:-0.25}" \
OVM3D_CLUSTER_MAX_POINTS="${OVM3D_CLUSTER_MAX_POINTS:-1200}" \
OVM3D_CLUSTER_MIN_KEEP_POINTS="${OVM3D_CLUSTER_MIN_KEEP_POINTS:-12}" \
OVM3D_CLUSTER_TOPK="${OVM3D_CLUSTER_TOPK:-2}" \
OVM3D_CLUSTER_BBOX_PADDING="${OVM3D_CLUSTER_BBOX_PADDING:-0.03}" \
PSEUDO_LABEL_ROOT="${WORK_ROOT}" \
python tools/generate_pseudo_bbox.py \
  --config-file "${CONFIG_FILE}" \
  OUTPUT_DIR "${GEN_OUTPUT_DIR}"

python tools/transform_to_coco.py \
  --dataset_name "${DATASET}" \
  --input_root "${WORK_ROOT}" \
  --output_dir "${OUT_DIR}"

echo "Wrote bidirectional-cluster pseudo labels to ${OUT_DIR}"
