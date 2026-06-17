#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

mkdir -p \
  outputs/boxer_cached_gsam_unidepth_dfu_full_cuda_train_shard0 \
  outputs/boxer_cached_gsam_unidepth_dfu_full_cuda_train_shard1

COMMON_ARGS=(
  --json_file datasets/Omni3D/SUNRGBD_train.json
  --image_root datasets
  --box_source original_gsam
  --depth_source original_unidepth
  --dataset SUNRGBD
  --split train
  --original_pseudo_root pseudo_label
  --gpu 0
  --force_precision float32
  --thresh3d 0.05
  --use_mask_depth_gate
  --boxer_refine_with_depth
  --use_dfu_point_filter
  --dfu_use_radius_outlier
  --dfu_radius_backend torch_cuda
  --dfu_radius_chunk_size 1024
  --dfu_min_points 16
  --dfu_radius 0.45
  --dfu_radius_nb_points 10
  --dfu_min_box_support 0.05
  --min_proj_iou 0.05
  --min_depth_pixels 8
  --min_depth_support 0.02
  --max_rel_depth_error 1.0
  --prior_min_ratio 0.15
  --prior_max_ratio 6.0
  --ground_max_distance 0.9
  --classwise_quality_gate
  --boxer_nms
)

nohup env \
  CUDA_VISIBLE_DEVICES=0 \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  conda run -n ovm3d-1 python tools/run_boxer_omni3d.py \
  "${COMMON_ARGS[@]}" \
  --start_index 0 \
  --skip_images 2 \
  --output_dir outputs/boxer_cached_gsam_unidepth_dfu_full_cuda_train_shard0 \
  --output_json outputs/boxer_cached_gsam_unidepth_dfu_full_cuda_train_shard0/SUNRGBD_train_boxer_cached_dfu_full_shard0.json \
  --stats_json outputs/boxer_cached_gsam_unidepth_dfu_full_cuda_train_shard0/boxer_stats.json \
  > boxer_cached_dfu_full_cuda_train_g0.log 2>&1 &
pid0=$!

nohup env \
  CUDA_VISIBLE_DEVICES=1 \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  conda run -n ovm3d-1 python tools/run_boxer_omni3d.py \
  "${COMMON_ARGS[@]}" \
  --start_index 1 \
  --skip_images 2 \
  --output_dir outputs/boxer_cached_gsam_unidepth_dfu_full_cuda_train_shard1 \
  --output_json outputs/boxer_cached_gsam_unidepth_dfu_full_cuda_train_shard1/SUNRGBD_train_boxer_cached_dfu_full_shard1.json \
  --stats_json outputs/boxer_cached_gsam_unidepth_dfu_full_cuda_train_shard1/boxer_stats.json \
  > boxer_cached_dfu_full_cuda_train_g1.log 2>&1 &
pid1=$!

echo "Started shard0 on GPU0: PID ${pid0}, log boxer_cached_dfu_full_cuda_train_g0.log"
echo "Started shard1 on GPU1: PID ${pid1}, log boxer_cached_dfu_full_cuda_train_g1.log"
echo "Note: each process sees one GPU, so --gpu 0 is correct for both shards."
