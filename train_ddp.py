#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""GolfPose 3D lifter — DDP training with WandB, TensorBoard, early stopping.

Usage (single-GPU, local):
    python train_ddp.py --model mixste_lite -d golf -k gt \
        -str G1,G2,G3,G4 -ste G5,G6 --val-subjects G4 \
        -f 243 -b 64 --club 5 --epochs 200 --patience 20

Usage (multi-GPU via torchrun):
    torchrun --nproc_per_node=4 train_ddp.py --model mixste_lite ...

SLURM launch: see slurm/train_single_gpu.sh, slurm/train_multi_gpu.sh,
              slurm/train_multi_node.sh
"""

import argparse
import errno
import math
import os
import sys
import json
import time
from copy import deepcopy
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from einops import rearrange
from torch.utils.tensorboard import SummaryWriter

# GolfPose imports
from common.camera import (
    normalize_screen_coordinates,
    image_coordinates,
    vicon_to_world_golf,
    world_to_camera_golf,
    camera_to_world_golf,
    world_to_vicon_golf,
)
from common.loss import (
    mpjpe,
    weighted_mpjpe,
    n_mpjpe,
    p_mpjpe,
    mean_velocity_error,
    mean_velocity_error_train,
    bonelen_consistency_loss,
)
from common.generators import ChunkedGenerator_Seq, UnchunkedGenerator_Seq
from common.utils import deterministic_random
from common.model_lightweight import create_lifter
from common.dist_utils import (
    setup_distributed,
    cleanup_distributed,
    get_rank,
    get_world_size,
    is_main_process,
    reduce_tensor,
    barrier,
)
from common.early_stopping import EarlyStopping
from common.logging import setup_logger


torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False



def parse_args():
    p = argparse.ArgumentParser(description="GolfPose 3D lifter — DDP training")

    # ---- data ----
    p.add_argument("-d", "--dataset", default="golf", type=str)
    p.add_argument("-k", "--keypoints", default="gt", type=str)
    p.add_argument("-str", "--subjects-train", default="G1,G2,G3,G4", type=str)
    p.add_argument("-ste", "--subjects-test", default="G5,G6", type=str)
    p.add_argument("--val-subjects", default="G4", type=str,
                   help="Validation subjects (for early stopping). Comma-separated.")
    p.add_argument("-a", "--actions", default="*", type=str)
    p.add_argument("--club", "--club-num", dest="club_num", default=0, type=int)
    p.add_argument("--subset", default=1.0, type=float)
    p.add_argument("--downsample", default=1, type=int)

    # ---- model ----
    p.add_argument("--model", default="mixste_lite", type=str,
                   choices=["mixste", "mixste_lite", "linformer", "rela", "temporal_conv"],
                   help="Lifter architecture name")
    p.add_argument("--model-preset", default="base", type=str, choices=["small", "base"])
    p.add_argument("-f", "--number-of-frames", default=243, type=int)
    p.add_argument("-cs", default=512, type=int, help="embed_dim_ratio override")
    p.add_argument("-dep", default=8, type=int, help="depth override")

    # ---- training ----
    p.add_argument("-e", "--epochs", default=200, type=int)
    p.add_argument("-b", "--batch-size", default=64, type=int)
    p.add_argument("-s", "--stride", default=1, type=int)
    p.add_argument("-lr", "--learning-rate", default=4e-5, type=float)
    p.add_argument("-lrd", "--lr-decay", default=0.99, type=float)
    p.add_argument("--warmup-epochs", default=5, type=int)
    p.add_argument("--clip-grad", default=1.0, type=float, help="gradient clipping norm")
    p.add_argument("--amp", action="store_true", help="mixed precision training")
    p.add_argument("--weight-decay", default=0.1, type=float)
    p.add_argument("--patience", default=20, type=int,
                   help="early stopping patience (0 = disabled)")
    p.add_argument("--no-data-augmentation", dest="data_augmentation",
                   action="store_false")
    p.set_defaults(data_augmentation=True)

    # ---- checkpoint / resume ----
    p.add_argument("-c", "--checkpoint", default="", type=str)
    p.add_argument("--resume", default="", type=str)
    p.add_argument("--checkpoint-frequency", default=20, type=int)

    # ---- logging ----
    p.add_argument("--wandb", action="store_true", help="enable WandB logging")
    p.add_argument("--wandb-project", default="golfpose", type=str)
    p.add_argument("--wandb-run-name", default="", type=str)
    p.add_argument("--wandb-group", default="3d_lifter", type=str)
    p.add_argument("--nolog", action="store_true")
    p.add_argument("--json-log", action="store_true",
                   help="write JSON-formatted log file")
    p.add_argument("--no-eval", action="store_true")
    p.add_argument("--export-training-curves", action="store_true")

    args = p.parse_args()
    return args


def load_dataset(args, total_num):
    """Load 3D + 2D data and return (dataset, keypoints, keypoints_metadata)."""
    dataset_name = "data_3d_" + args.dataset + "_" + args.keypoints + ".npz"
    dataset_path = "golfswing/" + dataset_name
    
    # Fallback to HPC scratch directory if not found locally
    if not os.path.exists(dataset_path):
        fallback_path = f"/capstor/scratch/cscs/ckuya/golfpose_dataset/golfswing/{dataset_name}"
        if os.path.exists(fallback_path):
            dataset_path = fallback_path

    if args.dataset == "h36m":
        from common.h36m_dataset import Human36mDataset
        dataset = Human36mDataset(dataset_path)
    elif args.dataset == "golf":
        from common.golf_dataset import GolfDataset
        dataset = GolfDataset(dataset_path, args.keypoints)
    elif args.dataset.startswith("humaneva"):
        from common.humaneva_dataset import HumanEvaDataset
        dataset = HumanEvaDataset(dataset_path)
    elif args.dataset.startswith("custom"):
        from common.custom_dataset import CustomDataset
        dataset = CustomDataset(
            "data/data_2d_" + args.dataset + "_" + args.keypoints + ".npz"
        )
    else:
        raise KeyError("Invalid dataset: " + args.dataset)

    # Prepare 3D positions
    for subject in sorted(dataset.subjects()):
        for action in sorted(dataset[subject].keys()):
            anim = dataset[subject][action]
            if "positions" in anim:
                anim["positions"] = anim["positions"][:, :total_num, :]
                positions_3d = []
                for cam in anim["cameras"]:
                    pos_3d_mm = anim["positions"] * 1000
                    pos_3d_world = vicon_to_world_golf(
                        pos_3d_mm, cam["vicon_to_world_basis_dots"], cam["square_size"]
                    )
                    pos_3d_cam = world_to_camera_golf(
                        pos_3d_world, cam["orientation"], cam["translation_mm"]
                    )
                    pos_3d = pos_3d_cam / 1000
                    pos_3d[:, 1:] -= pos_3d[:, :1]
                    positions_3d.append(pos_3d)
                anim["positions_3d"] = positions_3d

    # Load 2D keypoints
    kp_name = "data_2d_" + args.dataset + "_" + args.keypoints + ".npz"
    kp_path = "golfswing/" + kp_name
    
    if not os.path.exists(kp_path):
        fallback_kp = f"/capstor/scratch/cscs/ckuya/golfpose_dataset/golfswing/{kp_name}"
        if os.path.exists(fallback_kp):
            kp_path = fallback_kp
            
    keypoints = np.load(kp_path, allow_pickle=True)
    keypoints_metadata = keypoints["metadata"].item()
    keypoints = keypoints["positions_2d"].item()

    # Trim to total_num joints
    for subject in sorted(keypoints.keys()):
        for action in sorted(keypoints[subject]):
            for cam_idx, kps in enumerate(keypoints[subject][action]):
                keypoints[subject][action][cam_idx] = kps[..., :total_num, :]
    keypoints_metadata["num_joints"] = total_num

    # Align lengths
    for subject in sorted(dataset.subjects()):
        assert subject in keypoints
        for action in sorted(dataset[subject].keys()):
            assert action in keypoints[subject]
            if "positions_3d" not in dataset[subject][action]:
                continue
            for cam_idx in range(len(keypoints[subject][action])):
                mocap_length = dataset[subject][action]["positions_3d"][cam_idx].shape[0]
                assert keypoints[subject][action][cam_idx].shape[0] >= mocap_length
                if keypoints[subject][action][cam_idx].shape[0] > mocap_length:
                    keypoints[subject][action][cam_idx] = keypoints[subject][action][cam_idx][:mocap_length]
            assert len(keypoints[subject][action]) == len(dataset[subject][action]["positions_3d"])

    # Normalise 2D
    for subject in sorted(keypoints.keys()):
        for action in sorted(keypoints[subject]):
            for cam_idx, kps in enumerate(keypoints[subject][action]):
                cam = dataset.cameras()[subject][cam_idx]
                kps[..., :2] = normalize_screen_coordinates(
                    kps[..., :2], w=cam["res_w"], h=cam["res_h"]
                )
                keypoints[subject][action][cam_idx] = kps

    return dataset, keypoints, keypoints_metadata


def fetch(dataset, keypoints, subjects, args, action_filter=None, subset=1, parse_3d_poses=True):
    """Fetch pose sequences for given subjects."""
    out_poses_3d, out_poses_2d, out_camera_params = [], [], []
    stride = args.downsample

    for subject in sorted(subjects):
        for action in sorted(keypoints[subject].keys()):
            if action_filter is not None:
                found = any(action.startswith(a) for a in action_filter)
                if not found:
                    continue

            poses_2d = keypoints[subject][action]
            for i in range(len(poses_2d)):
                out_poses_2d.append(poses_2d[i])

            if subject in dataset.cameras():
                cams = dataset.cameras()[subject]
                assert len(cams) == len(poses_2d)
                for cam in cams:
                    if "intrinsic" in cam:
                        out_camera_params.append(cam["intrinsic"])

            if parse_3d_poses and "positions_3d" in dataset[subject][action]:
                poses_3d = dataset[subject][action]["positions_3d"]
                assert len(poses_3d) == len(poses_2d)
                for i in range(len(poses_3d)):
                    out_poses_3d.append(poses_3d[i])

    if not out_camera_params:
        out_camera_params = None
    if not out_poses_3d:
        out_poses_3d = None

    if subset < 1:
        for i in range(len(out_poses_2d)):
            n_frames = int(round(len(out_poses_2d[i]) // stride * subset) * stride)
            start = deterministic_random(0, len(out_poses_2d[i]) - n_frames + 1, str(len(out_poses_2d[i])))
            out_poses_2d[i] = out_poses_2d[i][start:start + n_frames:stride]
            if out_poses_3d is not None:
                out_poses_3d[i] = out_poses_3d[i][start:start + n_frames:stride]
    elif stride > 1:
        for i in range(len(out_poses_2d)):
            out_poses_2d[i] = out_poses_2d[i][::stride]
            if out_poses_3d is not None:
                out_poses_3d[i] = out_poses_3d[i][::stride]

    return out_camera_params, out_poses_3d, out_poses_2d



def eval_data_prepare(receptive_field, inputs_2d, inputs_3d):
    """Split long sequences into receptive_field-sized chunks for evaluation."""
    assert inputs_2d.shape[:-1] == inputs_3d.shape[:-1]
    inputs_2d_p = torch.squeeze(inputs_2d)
    inputs_3d_p = torch.squeeze(inputs_3d)

    if inputs_2d_p.shape[0] / receptive_field > inputs_2d_p.shape[0] // receptive_field:
        out_num = inputs_2d_p.shape[0] // receptive_field + 1
    else:
        out_num = inputs_2d_p.shape[0] // receptive_field

    eval_input_2d = torch.empty(out_num, receptive_field, inputs_2d_p.shape[1], inputs_2d_p.shape[2])
    eval_input_3d = torch.empty(out_num, receptive_field, inputs_3d_p.shape[1], inputs_3d_p.shape[2])

    for i in range(out_num - 1):
        eval_input_2d[i] = inputs_2d_p[i * receptive_field:i * receptive_field + receptive_field]
        eval_input_3d[i] = inputs_3d_p[i * receptive_field:i * receptive_field + receptive_field]

    if inputs_2d_p.shape[0] < receptive_field:
        pad_right = receptive_field - inputs_2d_p.shape[0]
        inputs_2d_p = rearrange(inputs_2d_p, "b f c -> f c b")
        inputs_2d_p = F.pad(inputs_2d_p, (0, pad_right), mode="replicate")
        inputs_2d_p = rearrange(inputs_2d_p, "f c b -> b f c")
    if inputs_3d_p.shape[0] < receptive_field:
        pad_right = receptive_field - inputs_3d_p.shape[0]
        inputs_3d_p = rearrange(inputs_3d_p, "b f c -> f c b")
        inputs_3d_p = F.pad(inputs_3d_p, (0, pad_right), mode="replicate")
        inputs_3d_p = rearrange(inputs_3d_p, "f c b -> b f c")

    eval_input_2d[-1] = inputs_2d_p[-receptive_field:]
    eval_input_3d[-1] = inputs_3d_p[-receptive_field:]
    return eval_input_2d, eval_input_3d




def get_loss_weights(dataset_name, total_num):
    """Return per-joint MPJPE weight tensor."""
    if dataset_name == "h36m":
        w = [1, 1, 2.5, 2.5, 1, 2.5, 2.5, 1, 1, 1, 1.5, 1.5, 4, 4, 1.5, 4, 4]
    elif dataset_name == "golf":
        w = [1, 1, 2.5, 2.5, 1, 2.5, 2.5, 1, 1, 1, 1.5, 1.5, 4, 4, 1.5, 4, 4, 4, 4, 4, 4, 4]
    elif dataset_name == "humaneva15":
        w = [1, 1, 2.5, 2.5, 1, 2.5, 2.5, 1, 1.5, 1.5, 4, 4, 1.5, 4, 4]
    else:
        return None
    return torch.tensor(w[:total_num], dtype=torch.float32).cuda()




@torch.no_grad()
def evaluate(model, test_generator, receptive_field, kps_left, kps_right,
             joints_left, joints_right, human_num, club_num):
    """Run full evaluation; returns (e1, e1_human, e1_club, e2, e3, ev)."""
    model.eval()
    epoch_loss_3d_pos = 0
    epoch_loss_3d_pos_procrustes = 0
    epoch_loss_3d_pos_scale = 0
    epoch_loss_3d_vel = 0
    epoch_loss_3d_pos_human = 0
    epoch_loss_3d_pos_club = 0
    N = 0

    for _, batch, batch_2d in test_generator.next_epoch():
        inputs_2d = torch.from_numpy(batch_2d.astype("float32"))
        inputs_3d = torch.from_numpy(batch.astype("float32"))

        # TTA
        inputs_2d_flip = inputs_2d.clone()
        inputs_2d_flip[:, :, :, 0] *= -1
        inputs_2d_flip[:, :, kps_left + kps_right, :] = inputs_2d_flip[:, :, kps_right + kps_left, :]

        inputs_3d_p = inputs_3d
        inputs_2d, inputs_3d = eval_data_prepare(receptive_field, inputs_2d, inputs_3d_p)
        inputs_2d_flip, _ = eval_data_prepare(receptive_field, inputs_2d_flip, inputs_3d_p)

        if torch.cuda.is_available():
            inputs_3d = inputs_3d.cuda()
            inputs_2d = inputs_2d.cuda()
            inputs_2d_flip = inputs_2d_flip.cuda()

        inputs_3d[:, :, 0] = 0

        predicted_3d_pos = model(inputs_2d)
        predicted_3d_pos_flip = model(inputs_2d_flip)
        predicted_3d_pos_flip[:, :, :, 0] *= -1
        predicted_3d_pos_flip[:, :, joints_left + joints_right] = \
            predicted_3d_pos_flip[:, :, joints_right + joints_left]

        for i in range(predicted_3d_pos.shape[0]):
            predicted_3d_pos[i] = (predicted_3d_pos[i] + predicted_3d_pos_flip[i]) / 2

        error = mpjpe(predicted_3d_pos, inputs_3d)
        epoch_loss_3d_pos += inputs_3d.shape[0] * inputs_3d.shape[1] * error.item()
        epoch_loss_3d_pos_scale += inputs_3d.shape[0] * inputs_3d.shape[1] * n_mpjpe(predicted_3d_pos, inputs_3d).item()
        N += inputs_3d.shape[0] * inputs_3d.shape[1]

        if club_num != 0:
            error_h = mpjpe(predicted_3d_pos[..., :human_num, :], inputs_3d[..., :human_num, :])
            error_c = mpjpe(predicted_3d_pos[..., human_num:, :], inputs_3d[..., human_num:, :])
            epoch_loss_3d_pos_human += inputs_3d.shape[0] * inputs_3d.shape[1] * error_h.item()
            epoch_loss_3d_pos_club += inputs_3d.shape[0] * inputs_3d.shape[1] * error_c.item()

        pred_np = predicted_3d_pos.cpu().numpy().reshape(-1, inputs_3d.shape[-2], inputs_3d.shape[-1])
        tgt_np = inputs_3d.cpu().numpy().reshape(-1, inputs_3d.shape[-2], inputs_3d.shape[-1])
        epoch_loss_3d_pos_procrustes += inputs_3d.shape[0] * inputs_3d.shape[1] * p_mpjpe(pred_np, tgt_np)
        epoch_loss_3d_vel += inputs_3d.shape[0] * inputs_3d.shape[1] * mean_velocity_error(pred_np, tgt_np)

    if N == 0:
        return 0, 0, 0, 0, 0, 0

    e1 = (epoch_loss_3d_pos / N) * 1000
    e2 = (epoch_loss_3d_pos_procrustes / N) * 1000
    e3 = (epoch_loss_3d_pos_scale / N) * 1000
    ev = (epoch_loss_3d_vel / N) * 1000
    e1_human = (epoch_loss_3d_pos_human / N) * 1000
    e1_club = (epoch_loss_3d_pos_club / N) * 1000
    return e1, e1_human, e1_club, e2, e3, ev


def main():
    args = parse_args()

    # ---- Distributed setup ----
    distributed = int(os.environ.get("WORLD_SIZE", 1)) > 1
    if distributed:
        setup_distributed()
    rank = get_rank()
    world_size = get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    # ---- Checkpoint dir ----
    timestamp = datetime.now().strftime("%Y%m%dT%H-%M-%S")
    if not args.checkpoint:
        args.checkpoint = f"checkpoint/{args.model}_{timestamp}"
    if is_main_process():
        os.makedirs(args.checkpoint, exist_ok=True)
    barrier()

    # ---- Logger ----
    logger = setup_logger(
        name="golfpose",
        log_file=os.path.join(args.checkpoint, "train.log") if is_main_process() else None,
        rank=rank,
        json_format=args.json_log,
    )
    logger.info("Args: %s", args)
    logger.info("Distributed: %s  world_size=%d  rank=%d", distributed, world_size, rank)

    # ---- TensorBoard ----
    writer = None
    if is_main_process() and not args.nolog:
        writer = SummaryWriter(os.path.join(args.checkpoint, "tb_logs"))

    # ---- WandB ----
    wandb_run = None
    if args.wandb and is_main_process():
        try:
            import wandb
            wandb_run = wandb.init(
                project=args.wandb_project,
                name=args.wandb_run_name or f"{args.model}_{timestamp}",
                group=args.wandb_group,
                config=vars(args),
                dir=args.checkpoint,
                resume="allow",
            )
            logger.info("WandB run: %s", wandb_run.url)
        except Exception as e:
            logger.warning("WandB init failed: %s", e)

    # ---- Data ----
    human_num = 17
    club_num = args.club_num
    total_num = human_num + club_num

    logger.info("Loading dataset...")
    dataset, keypoints, keypoints_metadata = load_dataset(args, total_num)

    keypoints_symmetry = keypoints_metadata["keypoints_symmetry"]
    kps_left, kps_right = list(keypoints_symmetry[0]), list(keypoints_symmetry[1])
    joints_left = list(dataset.skeleton().joints_left())
    joints_right = list(dataset.skeleton().joints_right())

    subjects_train = args.subjects_train.split(",")
    subjects_test = args.subjects_test.split(",")
    val_subjects = args.val_subjects.split(",") if args.val_subjects else subjects_test

    action_filter = None if args.actions == "*" else args.actions.split(",")

    cameras_test, poses_test, poses_test_2d = fetch(
        dataset, keypoints, subjects_test, args, action_filter
    )
    cameras_val, poses_val, poses_val_2d = fetch(
        dataset, keypoints, val_subjects, args, action_filter
    )

    receptive_field = args.number_of_frames
    logger.info("Receptive field: %d frames", receptive_field)

    pad = (receptive_field - 1) // 2
    causal_shift = 0
    num_joints = keypoints_metadata["num_joints"]

    # ---- Model ----
    logger.info("Creating model: %s (preset=%s)", args.model, args.model_preset)
    overrides = {}
    if args.model in ("mixste", "mixste_lite", "linformer", "rela"):
        overrides["embed_dim_ratio"] = args.cs
        overrides["depth"] = args.dep

    model_pos_train = create_lifter(
        args.model, num_joints, receptive_field,
        preset=args.model_preset, drop_path=0.1, **overrides
    )
    model_pos = create_lifter(
        args.model, num_joints, receptive_field,
        preset=args.model_preset, drop_path=0.0, **overrides
    )

    model_params = sum(p.numel() for p in model_pos.parameters())
    logger.info("Trainable parameters: %.2f M", model_params / 1e6)

    if torch.cuda.is_available():
        model_pos_train = model_pos_train.cuda()
        model_pos = model_pos.cuda()

    if distributed:
        model_pos_train = nn.parallel.DistributedDataParallel(
            model_pos_train, device_ids=[local_rank], output_device=local_rank,
            find_unused_parameters=False,
        )
    elif torch.cuda.device_count() > 1:
        model_pos_train = nn.DataParallel(model_pos_train)
        model_pos = nn.DataParallel(model_pos)

    # ---- Optimizer ----
    optimizer = optim.AdamW(
        model_pos_train.parameters(), lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    # ---- AMP scaler ----
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    # ---- Early stopping ----
    early_stopper = EarlyStopping(patience=args.patience, mode="min") if args.patience > 0 else None

    # ---- Resume ----
    start_epoch = 0
    min_loss = float("inf")
    if args.resume:
        chk_path = os.path.join(args.checkpoint, args.resume) if not os.path.isabs(args.resume) else args.resume
        logger.info("Resuming from %s", chk_path)
        checkpoint = torch.load(chk_path, map_location="cpu")
        start_epoch = checkpoint.get("epoch", 0)
        min_loss = checkpoint.get("min_loss", float("inf"))

        state_dict = checkpoint["model_pos"]
        model_pos_train_dict = model_pos_train.state_dict()
        model_pos_train_dict.update({k: v for k, v in state_dict.items() if k in model_pos_train_dict})
        model_pos_train.load_state_dict(model_pos_train_dict, strict=False)

        if "optimizer" in checkpoint and checkpoint["optimizer"] is not None:
            optimizer.load_state_dict(checkpoint["optimizer"])

        if early_stopper and "early_stopping" in checkpoint:
            early_stopper.load_state_dict(checkpoint["early_stopping"])

        logger.info("Resumed from epoch %d (best_loss=%.4f)", start_epoch, min_loss)

    # ---- Training data ----
    cameras_train, poses_train, poses_train_2d = fetch(
        dataset, keypoints, subjects_train, args, action_filter, subset=args.subset
    )
    local_batch = max(1, args.batch_size // world_size)
    train_generator = ChunkedGenerator_Seq(
        local_batch, cameras_train, poses_train, poses_train_2d,
        args.number_of_frames, pad=pad, causal_shift=causal_shift,
        shuffle=True, augment=args.data_augmentation,
        kps_left=kps_left, kps_right=kps_right,
        joints_left=joints_left, joints_right=joints_right,
        rank=rank, world_size=world_size
    )

    val_generator = UnchunkedGenerator_Seq(
        cameras_val, poses_val, poses_val_2d,
        pad=pad, causal_shift=causal_shift, augment=False,
        kps_left=kps_left, kps_right=kps_right,
        joints_left=joints_left, joints_right=joints_right,
        rank=rank, world_size=world_size
    )

    test_generator = UnchunkedGenerator_Seq(
        cameras_test, poses_test, poses_test_2d,
        pad=pad, causal_shift=causal_shift, augment=False,
        kps_left=kps_left, kps_right=kps_right,
        joints_left=joints_left, joints_right=joints_right,
        rank=rank, world_size=world_size
    )

    logger.info("Training frames: %d", train_generator.num_frames() if hasattr(train_generator, 'num_frames') else -1)
    logger.info("Validation frames: %d", val_generator.num_frames())
    logger.info("Test frames: %d", test_generator.num_frames())

    w_mpjpe = get_loss_weights(args.dataset, total_num)

    if writer:
        writer.add_text("config/model", args.model)
        writer.add_text("config/params_M", f"{model_params / 1e6:.2f}")
        writer.add_text("config/command", "python " + " ".join(sys.argv))

    lr = args.learning_rate

    for epoch in range(start_epoch, args.epochs):
        t_start = time.time()
        model_pos_train.train()

        epoch_loss = 0
        N_train = 0

        for cameras_batch, batch_3d, batch_2d in train_generator.next_epoch():
            inputs_3d = torch.from_numpy(batch_3d.astype("float32"))
            inputs_2d = torch.from_numpy(batch_2d.astype("float32"))

            if torch.cuda.is_available():
                inputs_3d = inputs_3d.cuda()
                inputs_2d = inputs_2d.cuda()

            inputs_3d[:, :, 0] = 0
            optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=args.amp):
                predicted_3d = model_pos_train(inputs_2d)

                loss_3d = weighted_mpjpe(predicted_3d, inputs_3d, w_mpjpe) if w_mpjpe is not None else mpjpe(predicted_3d, inputs_3d)

                # Temporal consistency
                dif_seq = predicted_3d[:, 1:] - predicted_3d[:, :-1]
                wj = torch.ones_like(dif_seq).cuda()
                if w_mpjpe is not None:
                    wj = torch.mul(wj.permute(0, 1, 3, 2), w_mpjpe).permute(0, 1, 3, 2)
                dif_loss = torch.mean(torch.multiply(wj, torch.square(dif_seq)))
                vel_loss = mean_velocity_error_train(predicted_3d, inputs_3d, axis=1)
                loss_diff = 0.5 * dif_loss + 2.0 * vel_loss

                # Bone-length consistency
                loss_bone = bonelen_consistency_loss(args.dataset, args.dataset, predicted_3d)

                loss_total = loss_3d + loss_diff + loss_bone

            scaler.scale(loss_total).backward(loss_total.clone().detach())

            if args.clip_grad > 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model_pos_train.parameters(), args.clip_grad)

            scaler.step(optimizer)
            scaler.update()

            loss_total_scalar = torch.mean(loss_total).item()
            epoch_loss += inputs_3d.shape[0] * inputs_3d.shape[1] * loss_total_scalar
            N_train += inputs_3d.shape[0] * inputs_3d.shape[1]

        train_loss = epoch_loss / max(N_train, 1)

        lr *= args.lr_decay
        for pg in optimizer.param_groups:
            pg["lr"] *= args.lr_decay

        val_mpjpe = 0
        if not args.no_eval:
            # Sync model weights for eval
            model_pos.load_state_dict(
                model_pos_train.module.state_dict() if hasattr(model_pos_train, "module")
                else model_pos_train.state_dict(),
                strict=False,
            )
            e1, e1h, e1c, e2, e3, ev = evaluate(
                model_pos, val_generator, receptive_field,
                kps_left, kps_right, joints_left, joints_right,
                human_num, club_num,
            )
            val_mpjpe = e1

        elapsed = (time.time() - t_start) / 60

        # ---- Logging ----
        if is_main_process():
            if args.no_eval:
                logger.info(
                    "[%d/%d] time=%.1fm lr=%.6f train_loss=%.4f",
                    epoch + 1, args.epochs, elapsed, lr, train_loss * 1000,
                )
            else:
                logger.info(
                    "[%d/%d] time=%.1fm lr=%.6f train=%.2f val_mpjpe=%.2f "
                    "p-mpjpe=%.2f n-mpjpe=%.2f vel=%.2f human=%.2f club=%.2f",
                    epoch + 1, args.epochs, elapsed, lr,
                    train_loss * 1000, e1, e2, e3, ev, e1h, e1c,
                )

            if writer:
                writer.add_scalar("Loss/train", train_loss * 1000, epoch + 1)
                writer.add_scalar("LR/learning_rate", lr, epoch + 1)
                writer.add_scalar("Time/epoch_minutes", elapsed, epoch + 1)
                if not args.no_eval:
                    writer.add_scalar("Metric/val_MPJPE", e1, epoch + 1)
                    writer.add_scalar("Metric/val_P-MPJPE", e2, epoch + 1)
                    writer.add_scalar("Metric/val_N-MPJPE", e3, epoch + 1)
                    writer.add_scalar("Metric/val_MPJVE", ev, epoch + 1)

            if wandb_run:
                log_dict = {
                    "epoch": epoch + 1,
                    "train_loss": train_loss * 1000,
                    "lr": lr,
                    "epoch_time_min": elapsed,
                }
                if not args.no_eval:
                    log_dict.update({
                        "val_MPJPE": e1, "val_P-MPJPE": e2,
                        "val_N-MPJPE": e3, "val_MPJVE": ev,
                        "val_MPJPE_human": e1h, "val_MPJPE_club": e1c,
                    })
                wandb_run.log(log_dict, step=epoch + 1)

        # ---- Checkpointing (rank 0 only) ----
        if is_main_process():
            ckpt_data = {
                "epoch": epoch + 1,
                "lr": lr,
                "min_loss": min_loss,
                "model_pos": (model_pos_train.module.state_dict()
                              if hasattr(model_pos_train, "module")
                              else model_pos_train.state_dict()),
                "optimizer": optimizer.state_dict(),
                "args": vars(args),
            }
            if early_stopper:
                ckpt_data["early_stopping"] = early_stopper.state_dict()

            # Latest
            torch.save(ckpt_data, os.path.join(args.checkpoint, "latest.pth"))

            # Periodic
            if (epoch + 1) % args.checkpoint_frequency == 0:
                torch.save(ckpt_data, os.path.join(args.checkpoint, f"epoch_{epoch + 1}.pth"))

            # Best
            if not args.no_eval and val_mpjpe < min_loss:
                min_loss = val_mpjpe
                ckpt_data["min_loss"] = min_loss
                torch.save(ckpt_data, os.path.join(args.checkpoint, "best.pth"))
                logger.info("New best MPJPE: %.2f mm — saved best.pth", min_loss)

        # ---- Early stopping ----
        if early_stopper and not args.no_eval:
            should_stop = early_stopper.step(val_mpjpe, epoch + 1)
            if distributed:
                stop_signal = torch.tensor([1.0 if should_stop else 0.0]).cuda()
                dist.broadcast(stop_signal, src=0)
                should_stop = stop_signal.item() > 0.5
            if should_stop:
                logger.warning("Early stopping triggered at epoch %d", epoch + 1)
                break

        if is_main_process() and args.export_training_curves and epoch > 3:
            try:
                import matplotlib
                matplotlib.use("Agg")
                import matplotlib.pyplot as plt
                # Simple loss curve
                fig, ax = plt.subplots()
                ax.set_ylabel("MPJPE (mm)")
                ax.set_xlabel("Epoch")
                ax.set_title("Training Progress")
                plt.savefig(os.path.join(args.checkpoint, "loss_curve.png"))
                plt.close("all")
            except Exception:
                pass

    if is_main_process() and not args.no_eval:
        logger.info("=" * 60)
        logger.info("Final evaluation on test set")
        logger.info("=" * 60)

        # Load best checkpoint
        best_path = os.path.join(args.checkpoint, "best.pth")
        if os.path.exists(best_path):
            best_ckpt = torch.load(best_path, map_location="cpu")
            model_pos.load_state_dict(best_ckpt["model_pos"], strict=False)
            logger.info("Loaded best checkpoint (epoch %d)", best_ckpt["epoch"])

        e1, e1h, e1c, e2, e3, ev = evaluate(
            model_pos, test_generator, receptive_field,
            kps_left, kps_right, joints_left, joints_right,
            human_num, club_num,
        )

        logger.info("MPJPE:   %.2f mm", e1)
        logger.info("P-MPJPE: %.2f mm", e2)
        logger.info("N-MPJPE: %.2f mm", e3)
        logger.info("MPJVE:   %.2f mm", ev)
        if club_num:
            logger.info("Human MPJPE: %.2f mm", e1h)
            logger.info("Club MPJPE:  %.2f mm", e1c)

        # Save final metrics
        metrics = {
            "model": args.model, "preset": args.model_preset,
            "params_M": model_params / 1e6,
            "MPJPE": round(e1, 2), "P-MPJPE": round(e2, 2),
            "N-MPJPE": round(e3, 2), "MPJVE": round(ev, 2),
            "MPJPE_human": round(e1h, 2), "MPJPE_club": round(e1c, 2),
        }
        with open(os.path.join(args.checkpoint, "final_metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)

        if wandb_run:
            wandb_run.summary.update(metrics)

    if writer:
        writer.close()
    if wandb_run:
        wandb_run.finish()
    if distributed:
        cleanup_distributed()

    logger.info("Training complete!")


if __name__ == "__main__":
    main()
