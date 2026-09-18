# GolfPose: Efficient 2D & 3D Golf Swing Pose Estimation

<p align="center"> <img src="./images/framework_v13.svg" width="80%"> </p>

An end-to-end training, evaluation, and visualization pipeline for full-body and golf club pose estimation, featuring lightweight 2D backbones (RTMPose, MobileNetV2, ShuffleNetV2, EfficientNet, SqueezeNet), multi-GPU Distributed Data Parallel (DDP) 3D lifting, comprehensive evaluation metrics, and SLURM cluster support.

---

## Environment Setup

Create and activate the conda environment:
```bash
conda env create -f environment.yml
conda activate golfpose
```

---

## Dataset Preparation

Organize the GolfPose dataset directory under the repository root as follows:
```
GolfPose/
`-- golfswing/
    |-- coco/
    |-- data_2d_golf_gt.npz
    |-- data_3d_golf_gt.npz
    `-- images/
```

> **Note**: The raw dataset files in `golfswing/` are excluded from version control via `.gitignore`.

---

## 2D Keypoint Training (MMPose)

We provide top-down and bottom-up lightweight MMPose configurations optimized for golfer and club keypoint detection at both standard (`256x192`) and ultra-compact (`128x96`) resolutions:

### Single GPU
```bash
# RTMPose-Tiny (Fastest top-down)
python tools/train_2d.py configs/mmpose/lightweight/rtmpose_tiny_golfer_256x192.py

# Ultra-compact 128x96 variant
python tools/train_2d.py configs/mmpose/lightweight/rtmpose_tiny_golfer_128x96.py

# MobileNetV2
python tools/train_2d.py configs/mmpose/lightweight/mobilenetv2_golfer_256x192.py

# ShuffleNetV2 (128x96)
python tools/train_2d.py configs/mmpose/lightweight/shufflenetv2_golfer_128x96.py
```

### Multi-GPU (Distributed)
```bash
torchrun --nproc_per_node=4 tools/train_2d.py \
    configs/mmpose/lightweight/rtmpose_tiny_golfer_256x192.py \
    --launcher pytorch --amp
```

Available configs in `configs/mmpose/lightweight/`:
- `rtmpose_tiny_golfer_256x192.py` / `rtmpose_tiny_golfer_128x96.py`
- `rtmpose_s_golfer_256x192.py`
- `mobilenetv2_golfer_256x192.py` / `mobilenetv2_golfer_128x96.py`
- `shufflenetv2_golfer_256x192.py` / `shufflenetv2_golfer_128x96.py`
- `efficientnet_golfer_256x192.py` / `efficientnet_golfer_128x96.py`
- `squeezenet_golfer_256x192.py` / `squeezenet_golfer_128x96.py`
- `dekr_mobilenetv2_golfer_512x512.py` (Bottom-up)

---

## 3D Lifter Training (Distributed DDP)

Train 3D temporal lifter models with PyTorch Distributed Data Parallel (DDP), Automatic Mixed Precision (AMP), early stopping, and WandB / TensorBoard logging:

### Single GPU
```bash
python train_ddp.py \
    -k gt -d golf \
    -str G1,G2,G3,G4 -ste G5,G6 \
    -f 243 -s 243 -club 5 \
    -c checkpoint/golfpose_3d \
    --lr 0.0002 --epochs 80 --batch-size 64 \
    --use-amp --early-stopping-patience 10
```

### Multi-GPU (torchrun)
```bash
torchrun --nproc_per_node=4 train_ddp.py \
    -k gt -d golf \
    -str G1,G2,G3,G4 -ste G5,G6 \
    -f 243 -s 243 -club 5 \
    -c checkpoint/golfpose_3d \
    --lr 0.0004 --epochs 80 --batch-size 64 \
    --use-amp --wandb-project golfpose-3d
```

---

## Evaluation

Run comprehensive multi-metric evaluation (MPJPE, P-MPJPE, per-joint errors, club head error, and phase breakdowns) with CSV / JSON export:

```bash
python tools/eval_comprehensive.py \
    --checkpoint checkpoint/golfpose_3d/best_epoch.bin \
    -k gt -d golf \
    -str G1,G2,G3,G4 -ste G5,G6 \
    -f 243 -s 243 -club 5 \
    --output-dir results/eval \
    --save-csv --save-json
```

---

## Visualization

Generate side-by-side 2D and 3D animated pose predictions:

```bash
python tools/visualize.py \
    --checkpoint checkpoint/golfpose_3d/best_epoch.bin \
    --data-dir golfswing \
    --split test \
    --sample-idx 0 \
    --save-dir results/vis \
    --fps 30
```

---

## SLURM Cluster Scripts

Ready-to-submit batch scripts for high-performance compute clusters are located in `slurm/`:

```bash
# 3D Single GPU Training
sbatch slurm/train_single_gpu.sh

# 3D Multi-GPU DDP Training (4x GPUs)
sbatch slurm/train_multi_gpu.sh

# 3D Multi-Node Distributed Training (2 nodes x 4 GPUs)
sbatch slurm/train_multi_node.sh

# 2D MMPose Training
sbatch slurm/train_2d_slurm.sh

# Comprehensive Evaluation & Visualization
sbatch slurm/eval_slurm.sh
sbatch slurm/visualize_slurm.sh
```

---

## Acknowledgements

This repository builds upon and extends the following works:

- **[GolfPose](https://github.com/MingHanLee/GolfPose)**: The original implementation and dataset by Ming-Han Lee, Yu-Chen Zhang, Kun-Ru Wu, and Yu-Chee Tseng.
- **[OpenMMLab MMPose](https://github.com/open-mmlab/mmpose)**: Open-source 2D pose estimation toolbox.
- **[OpenMMLab MMDetection](https://github.com/open-mmlab/mmdetection)**: Open-source object detection toolbox.
- **[MixSTE](https://github.com/JinluZhang1126/MixSTE)**: Mixed Spatio-Temporal Encoder for 3D human pose estimation.

---

## Citation

If you use this work or the GolfPose dataset, please cite the original paper:

```bibtex
@inproceedings{lee2025golfpose,
  title={GolfPose: From Regular Posture to Golf Swing Posture},
  author={Lee, Ming-Han and Zhang, Yu-Chen and Wu, Kun-Ru and Tseng, Yu-Chee},
  booktitle={International Conference on Pattern Recognition},
  pages={387--402},
  year={2025},
  organization={Springer}
}
```