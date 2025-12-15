#!/bin/bash

# train both tiny and small models for 3d medical segmentation
# runs them sequentially to avoid gpu memory conflicts

set -e  # exit on error

# default args
DATASET="btcv"
DATA_PATH="./data/btcv"
IMAGE_SIZE=1024
VAL_FREQ=1
PROMPT="bbox"
PROMPT_FREQ=2
BATCH_SIZE_TINY=1
BATCH_SIZE_SMALL=1
HIGH_MEMORY=false
PARALLEL=false
GPU_DEVICE_TINY=0
GPU_DEVICE_SMALL=0

# detect high-end gpu and suggest better defaults
if command -v nvidia-smi &> /dev/null; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)
    if [[ "$GPU_NAME" == *"H200"* ]] || [[ "$GPU_NAME" == *"H100"* ]] || [[ "$GPU_NAME" == *"A100"* ]]; then
        echo "detected high-end gpu: $GPU_NAME"
        echo "suggest using --high-memory flag for optimal batch sizes"
        echo ""
    fi
fi

# parse command line args if provided
while [[ $# -gt 0 ]]; do
    case $1 in
        --dataset)
            DATASET="$2"
            shift 2
            ;;
        --data-path)
            DATA_PATH="$2"
            shift 2
            ;;
        --image-size)
            IMAGE_SIZE="$2"
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE_TINY="$2"
            BATCH_SIZE_SMALL="$2"
            shift 2
            ;;
        --batch-size-tiny)
            BATCH_SIZE_TINY="$2"
            shift 2
            ;;
        --batch-size-small)
            BATCH_SIZE_SMALL="$2"
            shift 2
            ;;
        --high-memory)
            HIGH_MEMORY=true
            BATCH_SIZE_TINY=16
            BATCH_SIZE_SMALL=12
            shift
            ;;
        --prompt)
            PROMPT="$2"
            shift 2
            ;;
        --prompt-freq)
            PROMPT_FREQ="$2"
            shift 2
            ;;
        --val-freq)
            VAL_FREQ="$2"
            shift 2
            ;;
        --parallel)
            PARALLEL=true
            shift
            ;;
        --gpu-device-tiny)
            GPU_DEVICE_TINY="$2"
            shift 2
            ;;
        --gpu-device-small)
            GPU_DEVICE_SMALL="$2"
            shift 2
            ;;
        *)
            echo "unknown option: $1"
            echo "usage: $0 [options]"
            echo ""
            echo "options:"
            echo "  --dataset DATASET           dataset name (default: btcv)"
            echo "  --data-path PATH            path to data (default: ./data/btcv)"
            echo "  --image-size SIZE           image size (default: 1024)"
            echo "  --batch-size SIZE           batch size for both models (default: 1)"
            echo "  --batch-size-tiny SIZE       batch size for tiny model only"
            echo "  --batch-size-small SIZE      batch size for small model only"
            echo "  --high-memory               use aggressive batch sizes for h200/h100/a100 (tiny: 16, small: 12)"
            echo "  --parallel                  run both models simultaneously"
            echo "  --gpu-device-tiny ID        gpu device id for tiny model (default: 0)"
            echo "  --gpu-device-small ID       gpu device id for small model (default: 0)"
            echo "  --prompt TYPE                prompt type: bbox or click (default: bbox)"
            echo "  --prompt-freq FREQ          prompt frequency (default: 2)"
            echo "  --val-freq FREQ             validation frequency (default: 1)"
            echo ""
            echo "examples:"
            echo "  $0 --high-memory"
            echo "  $0 --batch-size 8"
            echo "  $0 --batch-size-tiny 16 --batch-size-small 8"
            echo "  $0 --parallel --gpu-device-tiny 0 --gpu-device-small 0"
            exit 1
            ;;
    esac
done

echo "=========================================="
echo "training both tiny and small models"
echo "=========================================="
echo "dataset: $DATASET"
echo "data path: $DATA_PATH"
echo "image size: $IMAGE_SIZE"
echo "batch size (tiny): $BATCH_SIZE_TINY"
echo "batch size (small): $BATCH_SIZE_SMALL"
echo "prompt: $PROMPT"
echo "prompt freq: $PROMPT_FREQ"
echo "val freq: $VAL_FREQ"
if [ "$HIGH_MEMORY" = true ]; then
    echo "high-memory mode: enabled"
fi
if [ "$PARALLEL" = true ]; then
    echo "execution mode: parallel"
    echo "gpu device (tiny): $GPU_DEVICE_TINY"
    echo "gpu device (small): $GPU_DEVICE_SMALL"
else
    echo "execution mode: sequential"
fi
echo "=========================================="
echo ""

# train tiny model
run_tiny() {
    echo "starting training for hiera-tiny (batch size: $BATCH_SIZE_TINY) on gpu $GPU_DEVICE_TINY..."
    CUDA_VISIBLE_DEVICES=$GPU_DEVICE_TINY python train_3d.py \
        -net sam2 \
        -exp_name ${DATASET}_MedSAM2_Tiny \
        -sam_ckpt ./checkpoints/sam2_hiera_tiny.pt \
        -sam_config sam2_hiera_t \
        -image_size $IMAGE_SIZE \
        -val_freq $VAL_FREQ \
        -prompt $PROMPT \
        -prompt_freq $PROMPT_FREQ \
        -dataset $DATASET \
        -data_path $DATA_PATH \
        -b $BATCH_SIZE_TINY
}

# train small model
run_small() {
    echo "starting training for hiera-small (batch size: $BATCH_SIZE_SMALL) on gpu $GPU_DEVICE_SMALL..."
    CUDA_VISIBLE_DEVICES=$GPU_DEVICE_SMALL python train_3d.py \
        -net sam2 \
        -exp_name ${DATASET}_MedSAM2_Small \
        -sam_ckpt ./checkpoints/sam2_hiera_small.pt \
        -sam_config sam2_hiera_s \
        -image_size $IMAGE_SIZE \
        -val_freq $VAL_FREQ \
        -prompt $PROMPT \
        -prompt_freq $PROMPT_FREQ \
        -dataset $DATASET \
        -data_path $DATA_PATH \
        -b $BATCH_SIZE_SMALL
}

if [ "$PARALLEL" = true ]; then
    echo "launching both trainings in parallel..."
    set +e
    run_tiny &
    PID_TINY=$!
    run_small &
    PID_SMALL=$!
    wait $PID_TINY
    RC_TINY=$?
    wait $PID_SMALL
    RC_SMALL=$?
    set -e

    if [ $RC_TINY -ne 0 ]; then
        echo ""
        echo "tiny model training failed with exit code $RC_TINY"
        exit 1
    fi
    if [ $RC_SMALL -ne 0 ]; then
        echo ""
        echo "small model training failed with exit code $RC_SMALL"
        exit 1
    fi
else
    run_tiny
    echo ""
    echo "tiny model training completed successfully!"
    echo ""

    run_small
fi

echo ""
echo "small model training completed successfully!"
echo ""
echo "=========================================="
echo "both models trained successfully!"
echo "checkpoints saved in logs/${DATASET}_MedSAM2_*/Model/"
echo "=========================================="

