#!/bin/bash
#SBATCH --job-name=golfpose_eval
#SBATCH --account=g164
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --partition=normal
#SBATCH --constraint=gpu

# Single-GPU comprehensive evaluation of a GolfPose 3D lifting checkpoint.
#
# Arguments:
#   $1  CHECKPOINT : path to checkpoint file (REQUIRED)
#   $2  MODEL      : mixste | mixste_lite | linformer | rela | temporal_conv
#                    (default: inferred from the checkpoint path, else mixste)
#   $@  extra args : passed through to tools/eval_comprehensive.py
#
# Examples:
#   sbatch slurm/eval_slurm.sh $SCRATCH/golfpose_runs/mixste/best.pth
#   sbatch slurm/eval_slurm.sh $SCRATCH/golfpose_runs/linformer/best.pth linformer
#   WANDB_MODE=online sbatch slurm/eval_slurm.sh ckpt.pth mixste --wandb
#
# Results are written to $WORK_DIR/metrics_${SLURM_JOB_ID}.json.

set -euo pipefail

CKPT="${1:-}"
if [ $# -gt 0 ]; then shift; fi

if [ -z "$CKPT" ]; then
    echo "ERROR: checkpoint path required"
    echo "Usage: sbatch slurm/eval_slurm.sh <checkpoint> [model] [extra args]"
    exit 1
fi

MODEL="${1:-}"
if [ $# -gt 0 ]; then shift; fi
EXTRA_ARGS=("$@")

# Infer model name from the checkpoint path if not given.
if [ -z "$MODEL" ]; then
    for m in mixste_lite mixste linformer rela temporal_conv; do
        if [[ "$CKPT" == *"$m"* ]]; then
            MODEL="$m"
            break
        fi
    done
    MODEL="${MODEL:-mixste}"
fi
echo "Checkpoint: $CKPT  Model: $MODEL"

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

if [[ "$CKPT" != /* ]]; then
    CKPT="$PROJECT_DIR/$CKPT"
fi
if [ ! -f "$CKPT" ]; then
    echo "ERROR: checkpoint not found: $CKPT"
    exit 1
fi

SCRATCH="${SCRATCH:-/capstor/scratch/cscs/${USER:-ckuya}}"
CONTAINER_IMAGE="${CONTAINER_IMAGE:-$SCRATCH/ce-images/golfpose.sqsh}"
WORK_DIR="${WORK_DIR:-$SCRATCH/golfpose_eval/${MODEL}}"
CONTAINER_MOUNTS="${CONTAINER_MOUNTS:-/capstor:/capstor,$SCRATCH:$SCRATCH,$HOME:$HOME}"
mkdir -p "$WORK_DIR"
exec > >(tee -a "$WORK_DIR/slurm_${SLURM_JOB_ID:-eval}.log") 2>&1

export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n1)
export MASTER_PORT=29500

echo "WandB mode  : $WANDB_MODE"
echo "Starting eval: 1 node x 1 GPU | Model: $MODEL | work_dir=$WORK_DIR"

srun --unbuffered \
     --container-image="$CONTAINER_IMAGE" \
     --container-mounts="$CONTAINER_MOUNTS" \
     --container-workdir="$PROJECT_DIR" \
     python "$PROJECT_DIR/tools/eval_comprehensive.py" \
         --checkpoint "$CKPT" \
         --model "$MODEL" \
         -d golf -k gt -ste G5,G6 \
         -f 243 --club 5 \
         --out "$WORK_DIR/metrics_${SLURM_JOB_ID:-eval}.json" \
         "${EXTRA_ARGS[@]}"

# Fallback for non-CSCS clusters without pyxis containers — replace the srun
# block above with something like:
#   source ~/miniconda3/etc/profile.d/conda.sh
#   conda activate golfpose
#   python tools/eval_comprehensive.py --checkpoint "$CKPT" --model "$MODEL" ...

echo "Eval complete!"
