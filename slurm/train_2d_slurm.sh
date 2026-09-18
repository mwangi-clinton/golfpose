#!/bin/bash
#SBATCH --job-name=golfpose_train_2d
#SBATCH --account=g164
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --partition=normal
#SBATCH --constraint=gpu

# MMPose 2D keypoint training (mmengine Runner, --launcher slurm; mmengine
# reads the SLURM env vars for distributed setup).
#
# Arguments:
#   $1  CONFIG     : path to an MMPose config
#                    (default: configs/mmpose/lightweight/rtmpose_tiny_golfer_256x192.py)
#   $@  extra args : passed through to tools/train_2d.py (e.g. --amp)
#
# Examples:
#   sbatch slurm/train_2d_slurm.sh
#   sbatch slurm/train_2d_slurm.sh configs/mmpose/lightweight/rtmpose_tiny_golfer_256x192.py --amp
#
# WandB logging is configured in the config; WANDB_MODE=disabled (default)
# turns wandb.init into a no-op. Enable with WANDB_MODE=online.

set -euo pipefail

CONFIG="${1:-configs/mmpose/lightweight/rtmpose_tiny_golfer_256x192.py}"
if [ $# -gt 0 ]; then shift; fi
EXTRA_ARGS=("$@")

export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export WANDB_MODE=${WANDB_MODE:-disabled}
export CUDNN_BENCHMARK=1
export PYTHONUNBUFFERED=1
export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME=hsn0
export NCCL_IB_DISABLE=0
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export GLOO_SOCKET_IFNAME=hsn0

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$PROJECT_DIR:${PYTHONPATH:-}"

if [[ "$CONFIG" != /* ]]; then
    CONFIG="$PROJECT_DIR/$CONFIG"
fi
if [ ! -f "$CONFIG" ]; then
    echo "ERROR: Config file does not exist: $CONFIG"
    exit 1
fi

CONTAINER_IMAGE="${CONTAINER_IMAGE:-$SCRATCH/ce-images/golfpose.sqsh}"
RUN_NAME="$(basename "$CONFIG" .py)"
WORK_DIR="${WORK_DIR:-$SCRATCH/golfpose_2d/$RUN_NAME}"
mkdir -p "$WORK_DIR"
exec > >(tee -a "$WORK_DIR/slurm_${SLURM_JOB_ID:-train}.log") 2>&1

export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n1)
export MASTER_PORT=29500
echo "Nodes: $(scontrol show hostnames $SLURM_JOB_NODELIST | tr '\n' ' ')  Master: $MASTER_ADDR"

echo "WandB mode  : $WANDB_MODE"
echo "Starting: $SLURM_NNODES node(s) x 4 GPUs | Config: $CONFIG | work_dir=$WORK_DIR"

srun --unbuffered \
     --container-image="$CONTAINER_IMAGE" \
     --container-mounts="$SCRATCH,$HOME" \
     --container-workdir="$PROJECT_DIR" \
     python "$PROJECT_DIR/tools/train_2d.py" \
         "$CONFIG" \
         --launcher slurm \
         --work-dir "$WORK_DIR" \
         "${EXTRA_ARGS[@]}"

# Fallback for non-CSCS clusters without pyxis containers — replace the srun
# block above with something like:
#   source ~/miniconda3/etc/profile.d/conda.sh
#   conda activate golfpose
#   python tools/train_2d.py "$CONFIG" --launcher pytorch --work-dir "$WORK_DIR" ...

echo "Training complete!"
