# SLURM launch scripts (CSCS Alps)

Scripts for training and evaluating GolfPose models on a SLURM cluster. They are
modeled on CSCS Alps (pyxis container runtime, `hsn0` Slingshot fabric) but can
be adapted to other clusters — see "Non-CSCS clusters" below.

## Usage

```bash
# 3D lifting model training (train_ddp.py)
sbatch slurm/train_single_gpu.sh mixste_lite                      # 1 node x 1 GPU
sbatch slurm/train_multi_gpu.sh mixste --model-preset base        # 1 node x 4 GPUs (torchrun)
sbatch slurm/train_multi_node.sh linformer                        # 4 nodes x 4 GPUs (torchrun)

# 2D MMPose keypoint training (tools/train_2d.py, mmengine --launcher slurm)
sbatch slurm/train_2d_slurm.sh configs/mmpose/lightweight/rtmpose_tiny_golfer_256x192.py --amp

# Comprehensive evaluation (tools/eval_comprehensive.py), 1 GPU
sbatch slurm/eval_slurm.sh $SCRATCH/golfpose_runs/mixste/best.pth mixste
```

All extra arguments after the positional ones are passed through to the
underlying training/eval script (e.g. `--epochs 100`, `--amp`, `--wandb`).

## Environment variable overrides

Set these before `sbatch` (SLURM propagates the submission environment):

| Variable | Default | Purpose |
|---|---|---|
| `CONTAINER_IMAGE` | `$SCRATCH/ce-images/golfpose.sqsh` | SquashFS container image |
| `WORK_DIR` | `$SCRATCH/golfpose_runs/...` (train) or `$SCRATCH/golfpose_2d/...`, `$SCRATCH/golfpose_eval/...` | Output/checkpoint dir; logs are tee'd to `$WORK_DIR/slurm_<jobid>.log` |
| `WANDB_MODE` | `disabled` | Set to `online` to actually log to WandB (project `golfpose`); training scripts only pass `--wandb` flags when enabled |

Example:

```bash
WANDB_MODE=online CONTAINER_IMAGE=$SCRATCH/ce-images/golfpose_v2.sqsh \
    sbatch slurm/train_multi_gpu.sh mixste --wandb-run-name mixste_v2
```

## Non-CSCS clusters

`--account=g164`, `--partition=normal`, and `--constraint=gpu` are CSCS
Alps-specific — edit the `#SBATCH` headers for your site. If your cluster has
no pyxis/`--container-*` support, each script ends with a commented fallback
block: activate a conda env (e.g. `conda activate golfpose`) and invoke the
same command without the container flags. For the 2D script, also switch
`--launcher slurm` to `--launcher pytorch` on clusters where mmengine cannot
read SLURM env vars.

The NCCL/GLOO settings (`NCCL_SOCKET_IFNAME=hsn0`, `NCCL_IB_DISABLE=0`) target
the Slingshot interconnect on Alps; on other fabrics adjust or drop them.
