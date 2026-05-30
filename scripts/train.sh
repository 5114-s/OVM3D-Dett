DATASET=$1
NUM_GPUS=${2:-1}

CUDA_VISIBLE_DEVICES=0,1 python tools/train_net.py \
  --config-file configs/Base_Omni3D_$DATASET.yaml --dist-url tcp://0.0.0.0:12345 --num-gpus $NUM_GPUS \
    DATASETS.FOLDER_NAME "Omni3D_pl" \
    OUTPUT_DIR output/training/$DATASET \
    SOLVER.IMS_PER_BATCH $((16 * NUM_GPUS))

