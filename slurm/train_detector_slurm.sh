#!/bin/bash
# ============================================================================
# SLURM job script — RTMDet golfer + club detector training
# ============================================================================
#SBATCH --job-name=rtmdet_golfer
#SBATCH --account=g164
#SBATCH --time=03:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --partition=normal
#SBATCH --constraint=gpu
#
# Trains one of three RTMDet detector configs on 4 GPUs (single node).
#
# Arguments:
#   $1  CONFIG   : nano | tiny | s  (default: tiny)
#   $@  extra    : forwarded to mmdet.train  (e.g. --amp, --resume)
#
# Examples:
#   sbatch slurm/train_detector_slurm.sh             # RTMDet-tiny
#   sbatch slurm/train_detector_slurm.sh nano        # RTMDet-nano  (fastest)
#   sbatch slurm/train_detector_slurm.sh s           # RTMDet-s     (best mAP)
#   sbatch slurm/train_detector_slurm.sh tiny --amp  # tiny + mixed precision
#
# WandB logging is enabled by default. Disable with WANDB_MODE=disabled.
# ============================================================================

set -euo pipefail

MODEL_SIZE="${1:-tiny}"
if [ $# -gt 0 ]; then shift; fi
EXTRA_ARGS=("$@")

case "$MODEL_SIZE" in
    nano) CONFIG_FILE="configs/mmdet/golfpose_detector_rtmdet_nano.py" ;;
    tiny) CONFIG_FILE="configs/mmdet/golfpose_detector_rtmdet_tiny.py" ;;
    s)    CONFIG_FILE="configs/mmdet/golfpose_detector_rtmdet_s.py"    ;;
    *)
        echo "ERROR: Unknown model size '$MODEL_SIZE'. Choose: nano | tiny | s"
        exit 1
        ;;
esac

export WANDB_API_KEY="${WANDB_API_KEY:-}"
export WANDB_MODE="${WANDB_MODE:-online}"
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export CUDNN_BENCHMARK=1
export PYTHONUNBUFFERED=1
export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME=hsn0
export NCCL_IB_DISABLE=0
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export GLOO_SOCKET_IFNAME=hsn0

PROJECT_DIR="${PROJECT_DIR:-/users/ckuya/golfpose}"
if [ ! -d "$PROJECT_DIR" ]; then
    if [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -d "$SLURM_SUBMIT_DIR/configs" ]; then
        PROJECT_DIR="$SLURM_SUBMIT_DIR"
    else
        PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
    fi
fi
export PYTHONPATH="$PROJECT_DIR:${PYTHONPATH:-}"

CONFIG_PATH="$PROJECT_DIR/$CONFIG_FILE"
if [ ! -f "$CONFIG_PATH" ]; then
    echo "ERROR: Config not found: $CONFIG_PATH"
    exit 1
fi

SCRATCH="${SCRATCH:-/capstor/scratch/cscs/${USER:-ckuya}}"
CONTAINER_IMAGE="${CONTAINER_IMAGE:-$SCRATCH/ce-images/new_experiments_v2.sqsh}"
RUN_NAME="rtmdet_${MODEL_SIZE}_golfer"
WORK_DIR="${WORK_DIR:-$SCRATCH/golfpose_det/$RUN_NAME}"
CONTAINER_MOUNTS="${CONTAINER_MOUNTS:-/capstor:/capstor,$SCRATCH:$SCRATCH,$HOME:$HOME}"
mkdir -p "$WORK_DIR"

exec > >(tee -a "$WORK_DIR/slurm_${SLURM_JOB_ID:-train}.log") 2>&1

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
export MASTER_PORT=29501     
echo "Node: $MASTER_ADDR  Port: $MASTER_PORT"

DATA_DIR="${DATA_DIR:-/capstor/scratch/cscs/${USER:-ckuya}/golfpose_dataset/golfswing}"
if [ -d "$DATA_DIR" ]; then
    mkdir -p "$PROJECT_DIR/golfswing/coco"

    for json_file in "$DATA_DIR/coco"/*.json; do
        if [ -f "$json_file" ]; then
            ln -sfn "$json_file" "$PROJECT_DIR/golfswing/coco/$(basename "$json_file")"
        fi
    done
    echo "Linked annotations from $DATA_DIR/coco -> $PROJECT_DIR/golfswing/coco"

    if [ ! -e "$PROJECT_DIR/golfswing/images" ] || [ -L "$PROJECT_DIR/golfswing/images" ]; then
        ln -sfn "$DATA_DIR/images" "$PROJECT_DIR/golfswing/images"
        echo "Linked images: $DATA_DIR/images -> $PROJECT_DIR/golfswing/images"
    fi
fi
export DATA_DIR

mkdir -p "$PROJECT_DIR/models"

echo ""
echo "=============================================="
echo "  RTMDet Golfer Detector Training"
echo "  Model size : $MODEL_SIZE"
echo "  Config     : $CONFIG_FILE"
echo "  Work dir   : $WORK_DIR"
echo "  GPUs       : $SLURM_GPUS_ON_NODE x $SLURM_NNODES node(s)"
echo "  WandB      : $WANDB_MODE"
echo "  Extra args : ${EXTRA_ARGS[*]:-none}"
echo "=============================================="
echo ""

cd "$PROJECT_DIR"


srun --unbuffered \
     --container-image="$CONTAINER_IMAGE" \
     --container-mounts="$CONTAINER_MOUNTS" \
     --container-workdir="$PROJECT_DIR" \
     python -m mmdet.train \
         "$CONFIG_PATH" \
         --launcher slurm \
         --work-dir "$WORK_DIR" \
         --amp \
         "${EXTRA_ARGS[@]}"


echo ""
echo "Training complete! Results saved to: $WORK_DIR"
