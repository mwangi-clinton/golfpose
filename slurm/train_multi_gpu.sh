#!/bin/bash
#SBATCH --job-name=golfpose_train_4g
#SBATCH --account=g164
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --partition=normal
#SBATCH --constraint=gpu

# Single-node multi-GPU (4 GPUs) training for GolfPose 3D lifting models.
#
# Arguments:
#   $1  MODEL      : mixste | mixste_lite | linformer | rela | temporal_conv (default: mixste_lite)
#   $@  extra args : passed through to train_ddp.py
#
# Examples:
#   sbatch slurm/train_multi_gpu.sh
#   sbatch slurm/train_multi_gpu.sh mixste --model-preset base
#   WANDB_MODE=online sbatch slurm/train_multi_gpu.sh linformer
#
# WandB logging is off by default; enable with WANDB_MODE=online.

set -euo pipefail

MODEL="${1:-mixste_lite}"
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

CONTAINER_IMAGE="${CONTAINER_IMAGE:-$SCRATCH/ce-images/golfpose.sqsh}"
WORK_DIR="${WORK_DIR:-$SCRATCH/golfpose_runs/${MODEL}_1n4g}"
mkdir -p "$WORK_DIR"
exec > >(tee -a "$WORK_DIR/slurm_${SLURM_JOB_ID:-train}.log") 2>&1

export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n1)
export MASTER_PORT=29500
echo "Node: $MASTER_ADDR  Port: $MASTER_PORT"

WANDB_FLAGS=(--wandb --wandb-project golfpose --wandb-run-name "${MODEL}_1n4g_${SLURM_JOB_ID:-local}")
if [ "$WANDB_MODE" = "disabled" ]; then
    WANDB_FLAGS=()
fi
echo "WandB mode  : $WANDB_MODE (project=golfpose)"

echo "Starting: 1 node x 4 GPUs | Model: $MODEL | work_dir=$WORK_DIR"

srun --unbuffered \
     --container-image="$CONTAINER_IMAGE" \
     --container-mounts="$SCRATCH,$HOME" \
     --container-workdir="$PROJECT_DIR" \
     torchrun \
         --nproc_per_node=4 \
         --rdzv_backend=c10d \
         --rdzv_endpoint=localhost:$MASTER_PORT \
         "$PROJECT_DIR/train_ddp.py" \
             --model "$MODEL" \
             -d golf -k gt \
             -str G1,G2,G3,G4 -ste G5,G6 --val-subjects G4 \
             -f 243 -s 243 -b 64 --club 5 \
             --epochs 200 --patience 20 \
             --checkpoint "$WORK_DIR" \
             "${WANDB_FLAGS[@]}" \
             "${EXTRA_ARGS[@]}"

# Fallback for non-CSCS clusters without pyxis containers — replace the srun
# block above with something like:
#   source ~/miniconda3/etc/profile.d/conda.sh
#   conda activate golfpose
#   torchrun --nproc_per_node=4 --rdzv_backend=c10d \
#       --rdzv_endpoint=localhost:$MASTER_PORT train_ddp.py ...

echo "Training complete!"
