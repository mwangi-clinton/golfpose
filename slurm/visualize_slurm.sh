#!/bin/bash
#SBATCH --job-name=golfpose_vis
#SBATCH --account=g164
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=normal
#SBATCH --constraint=gpu

# Single-GPU visualisation of GolfPose 3D lifting results.
#
# Arguments:
#   $1  SUBCOMMAND : pose | heatmap | compare | curves (default: heatmap)
#   $@  extra args : passed through to tools/visualize.py
#
# Examples:
#   # Per-joint heatmap from an evaluation JSON
#   sbatch slurm/visualize_slurm.sh heatmap \
#       --metrics $SCRATCH/golfpose_eval/mixste/metrics_comprehensive.json
#
#   # Render 3D pose animation
#   sbatch slurm/visualize_slurm.sh pose \
#       --checkpoint $SCRATCH/golfpose_runs/mixste/best.pth \
#       --model mixste_lite --club 5 --subject G5 --action Swing01
#
#   # Compare multiple models
#   sbatch slurm/visualize_slurm.sh compare \
#       --metrics m1.json m2.json m3.json
#
#   # Training curves from TensorBoard logs
#   sbatch slurm/visualize_slurm.sh curves \
#       --tb-logdir $SCRATCH/golfpose_runs/mixste/tb_logs

set -euo pipefail

SUBCMD="${1:-heatmap}"
if [ $# -gt 0 ]; then shift; fi
EXTRA_ARGS=("$@")

export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export PYTHONUNBUFFERED=1

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$PROJECT_DIR:${PYTHONPATH:-}"

SCRATCH="${SCRATCH:-/capstor/scratch/cscs/${USER:-ckuya}}"
CONTAINER_IMAGE="${CONTAINER_IMAGE:-$SCRATCH/ce-images/golfpose.sqsh}"
WORK_DIR="${WORK_DIR:-$SCRATCH/golfpose_vis}"
CONTAINER_MOUNTS="${CONTAINER_MOUNTS:-/capstor:/capstor,$SCRATCH:$SCRATCH,$HOME:$HOME}"
mkdir -p "$WORK_DIR"
exec > >(tee -a "$WORK_DIR/slurm_${SLURM_JOB_ID:-vis}.log") 2>&1

echo "Starting visualisation: $SUBCMD | work_dir=$WORK_DIR"

srun --unbuffered \
     --container-image="$CONTAINER_IMAGE" \
     --container-mounts="$CONTAINER_MOUNTS" \
     --container-workdir="$PROJECT_DIR" \
     python "$PROJECT_DIR/tools/visualize.py" \
         "$SUBCMD" \
         --output-dir "$WORK_DIR" \
         "${EXTRA_ARGS[@]}"

# Fallback for non-CSCS clusters without pyxis containers — replace the srun
# block above with something like:
#   source ~/miniconda3/etc/profile.d/conda.sh
#   conda activate golfpose
#   python tools/visualize.py "$SUBCMD" --output-dir "$WORK_DIR" "${EXTRA_ARGS[@]}"

echo "Visualisation complete! Outputs in $WORK_DIR"
