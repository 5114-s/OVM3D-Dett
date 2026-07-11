#!/usr/bin/env bash
set -euo pipefail

# Clean route:
#   original GroundedSAM cache -> MoCA3D-Cube -> Omni3D JSON
#
# Required checkpoints by default:
#   third_party/MoCA3D/checkpoints/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth
#   third_party/MoCA3D/checkpoints/MoCA3D_Cube/best_iou_inv_joint.pt
#
# The Python runner mirrors MoCA3D tools/evaluate_cube.py:
#   prefer joint checkpoint -> otherwise fall back to separate MoCA/Cube ckpts.

OUT_DIR="${OUT_DIR:-datasets/Omni3D_pl-moca3d-cube}"
STATS_DIR="${STATS_DIR:-outputs/moca3d_cube_sun}"
JSON_ROOT="${JSON_ROOT:-datasets/Omni3D}"
IMAGE_ROOT="${IMAGE_ROOT:-datasets}"
PSEUDO_ROOT="${PSEUDO_ROOT:-pseudo_label}"
MOCA_ROOT="${MOCA_ROOT:-third_party/MoCA3D}"
GPU="${GPU:-0}"
BATCH_SIZE="${BATCH_SIZE:-8}"
PRECISION="${PRECISION:-float32}"
SPLITS="${SPLITS:-train val}"
JOINT_CHECKPOINT="${JOINT_CHECKPOINT:-${MOCA_ROOT}/checkpoints/MoCA3D_Cube/best_iou_inv_joint.pt}"

mkdir -p "${OUT_DIR}" "${STATS_DIR}"

COMMON_ARGS=(
  --image_root "${IMAGE_ROOT}"
  --dataset SUNRGBD
  --original_pseudo_root "${PSEUDO_ROOT}"
  --moca_root "${MOCA_ROOT}"
  --gpu "${GPU}"
  --batch_size "${BATCH_SIZE}"
  --precision "${PRECISION}"
)

if [[ -n "${MOCA_CHECKPOINT:-}" ]]; then
  COMMON_ARGS+=(--moca_checkpoint "${MOCA_CHECKPOINT}")
fi
if [[ -n "${CUBE_CHECKPOINT:-}" ]]; then
  COMMON_ARGS+=(--cube_checkpoint "${CUBE_CHECKPOINT}")
fi
if [[ -f "${JOINT_CHECKPOINT}" ]]; then
  COMMON_ARGS+=(--joint_checkpoint "${JOINT_CHECKPOINT}")
fi
if [[ "${PREFER_EMA:-0}" == "1" ]]; then
  COMMON_ARGS+=(--prefer_ema)
fi
if [[ -n "${MAX_IMAGES:-}" ]]; then
  COMMON_ARGS+=(--max_images "${MAX_IMAGES}")
fi
if [[ -n "${START_INDEX:-}" ]]; then
  COMMON_ARGS+=(--start_index "${START_INDEX}")
fi
if [[ -n "${SKIP_IMAGES:-}" ]]; then
  COMMON_ARGS+=(--skip_images "${SKIP_IMAGES}")
fi

for SPLIT in ${SPLITS}; do
  python tools/run_moca3d_omni3d.py \
    --json_file "${JSON_ROOT}/SUNRGBD_${SPLIT}.json" \
    --split "${SPLIT}" \
    --output_json "${OUT_DIR}/SUNRGBD_${SPLIT}.json" \
    --output_dir "${STATS_DIR}/${SPLIT}" \
    "${COMMON_ARGS[@]}"
done

echo "Done. MoCA3D-Cube pseudo labels: ${OUT_DIR}"
