#!/usr/bin/env bash
set -euo pipefail

# Build a calibrated MoCA3D-Cube pseudo-label branch:
#
#   MoCA3D projected corners/yaw
#   + cached UniDepth depth anchor
#   + original OVM3D class-size priors
#   -> calibrated center/dims/R_cam + factorized pseudo weights
#
# This script does not rerun MoCA inference. It expects
# datasets/Omni3D_pl-moca3d-cube/SUNRGBD_{train,val}.json to already exist.

MOCA_DIR="${MOCA_DIR:-datasets/Omni3D_pl-moca3d-cube}"
REFERENCE_DIR="${REFERENCE_DIR:-datasets/Omni3D_pl-1}"
OUT_DIR="${OUT_DIR:-datasets/Omni3D_pl-moca3d-calibrated}"
STATS_DIR="${STATS_DIR:-outputs/moca3d_calibrated_sun}"
DEPTH_ROOT="${DEPTH_ROOT:-pseudo_label/SUNRGBD}"
SPLITS="${SPLITS:-train val}"

mkdir -p "${OUT_DIR}" "${STATS_DIR}"

for SPLIT in ${SPLITS}; do
  python tools/build_moca3d_calibrated_pseudolabels.py \
    --moca_json "${MOCA_DIR}/SUNRGBD_${SPLIT}.json" \
    --reference_json "${REFERENCE_DIR}/SUNRGBD_${SPLIT}.json" \
    --output_json "${OUT_DIR}/SUNRGBD_${SPLIT}.json" \
    --stats_json "${STATS_DIR}/SUNRGBD_${SPLIT}_stats.json" \
    --depth_root "${DEPTH_ROOT}" \
    --dataset SUNRGBD \
    --split "${SPLIT}"
done

echo "Done. MoCA3D calibrated pseudo labels: ${OUT_DIR}"
