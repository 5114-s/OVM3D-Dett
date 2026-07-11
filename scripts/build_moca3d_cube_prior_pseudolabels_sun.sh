#!/usr/bin/env bash
set -euo pipefail

# Constrain MoCA3D-Cube pseudo labels with original OVM3D-Det geometry priors.
#
# Inputs by default:
#   datasets/Omni3D_pl-moca3d-cube/SUNRGBD_{train,val}.json
#   datasets/Omni3D_pl-1/SUNRGBD_{train,val}.json
#
# Output:
#   datasets/Omni3D_pl-moca3d-cube-prior/SUNRGBD_{train,val}.json

MOCA_DIR="${MOCA_DIR:-datasets/Omni3D_pl-moca3d-cube}"
REFERENCE_DIR="${REFERENCE_DIR:-datasets/Omni3D_pl-1}"
OUT_DIR="${OUT_DIR:-datasets/Omni3D_pl-moca3d-cube-prior}"
STATS_DIR="${STATS_DIR:-outputs/moca3d_cube_prior_sun}"
SPLITS="${SPLITS:-train val}"

mkdir -p "${OUT_DIR}" "${STATS_DIR}"

for SPLIT in ${SPLITS}; do
  python tools/apply_moca3d_original_priors.py \
    --moca_json "${MOCA_DIR}/SUNRGBD_${SPLIT}.json" \
    --reference_json "${REFERENCE_DIR}/SUNRGBD_${SPLIT}.json" \
    --output_json "${OUT_DIR}/SUNRGBD_${SPLIT}.json" \
    --stats_json "${STATS_DIR}/SUNRGBD_${SPLIT}_stats.json" \
    --moca_scale_to_ref
done

echo "Done. Prior-constrained MoCA3D-Cube pseudo labels: ${OUT_DIR}"
