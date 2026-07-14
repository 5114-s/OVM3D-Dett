#!/usr/bin/env bash
set -euo pipefail

OUT_DIR="${OUT_DIR:-outputs/supervised_boxer_lifthead_sun_base_novel}"
GPU="${GPU:-0}"
EPOCHS="${EPOCHS:-120}"
BATCH_SIZE="${BATCH_SIZE:-1024}"
LR="${LR:-0.001}"
HIDDEN_DIM="${HIDDEN_DIM:-384}"
NUM_LAYERS="${NUM_LAYERS:-4}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" python tools/train_lifthead.py \
  --train_pth "${OUT_DIR}/train_pairs.pth" \
  --val_pth "${OUT_DIR}/val_pairs.pth" \
  --output_dir "${OUT_DIR}" \
  --epochs "${EPOCHS}" \
  --batch_size "${BATCH_SIZE}" \
  --lr "${LR}" \
  --hidden_dim "${HIDDEN_DIM}" \
  --num_layers "${NUM_LAYERS}" \
  --loss_center 2.5 \
  --loss_dims 1.2 \
  --loss_yaw 0.45 \
  --gpu "${GPU}"

