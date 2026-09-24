#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Comprehensive evaluation for GolfPose 3D lifting models.

Computes per-action and overall metrics:
  - Protocol #1: MPJPE (mm)
  - Protocol #2: P-MPJPE (Procrustes-aligned)
  - Protocol #3: N-MPJPE (scale-normalized)
  - Velocity:    MPJVE (mm)
  - Per-joint MPJPE breakdown (human + club)
  - PCK@50mm / PCK@100mm / PCK@150mm
  - AUC (area under PCK curve)
  - Model complexity: parameter count, FLOPs, inference latency

Usage:
    python tools/eval_comprehensive.py \\
        --checkpoint best.pth --model mixste_lite \\
        -d golf -k gt -ste G5,G6 -f 243 --club 5 \\
        --out results/metrics.json

    # With WandB logging:
    python tools/eval_comprehensive.py \\
        --checkpoint best.pth --model mixste_lite \\
        --wandb --wandb-project golfpose

    # Per-subject breakdown:
    python tools/eval_comprehensive.py \\
        --checkpoint best.pth --model mixste_lite \\
        --by-subject
"""

import argparse
import json
import logging
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

# Ensure project root is on path
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from common.camera import (
    normalize_screen_coordinates,
    vicon_to_world_golf,
    world_to_camera_golf,
)
from common.loss import mpjpe, p_mpjpe, n_mpjpe, mean_velocity_error
from common.generators import UnchunkedGenerator_Seq
from common.utils import deterministic_random
from common.model_lightweight import create_lifter
from common.logging import setup_logger

logger = logging.getLogger("golfpose.eval")


# ---------------------------------------------------------------------------
# PCK & AUC metrics
# ---------------------------------------------------------------------------

def compute_pck(predicted, target, threshold_mm=150.0):
    """Percentage of Correct Keypoints at a given threshold (mm).

    Parameters
    ----------
    predicted, target : np.ndarray  shape (N, J, 3)
    threshold_mm : float

    Returns
    -------
    pck : float  (0-100 percentage)
    per_joint_pck : np.ndarray  shape (J,)
    """
    assert predicted.shape == target.shape
    # Convert to mm if in metres
    distances = np.linalg.norm(predicted - target, axis=-1)  # (N, J)

    # Assume data is in metres — multiply by 1000 for mm comparison
    distances_mm = distances * 1000

    per_joint_pck = np.mean(distances_mm < threshold_mm, axis=0) * 100  # (J,)
    pck = np.mean(distances_mm < threshold_mm) * 100
    return pck, per_joint_pck


def compute_auc(predicted, target, thresholds=None):
    """Area Under the PCK Curve.

    Parameters
    ----------
    predicted, target : np.ndarray  shape (N, J, 3)
    thresholds : array-like  in mm; defaults to 0-150mm in steps of 5

    Returns
    -------
    auc : float  (0-100)
    """
    if thresholds is None:
        thresholds = np.arange(0, 155, 5)

    pck_values = []
    for t in thresholds:
        pck, _ = compute_pck(predicted, target, threshold_mm=t)
        pck_values.append(pck)

    auc = np.trapz(pck_values, thresholds) / (thresholds[-1] - thresholds[0])
    return auc


# ---------------------------------------------------------------------------
# Model complexity
# ---------------------------------------------------------------------------

def compute_model_complexity(model, num_joints, num_frames, device="cuda"):
    """Compute parameter count and inference latency.

    Returns dict with params_M, latency_ms.
    Optionally computes FLOPs if fvcore is available.
    """
    params_M = sum(p.numel() for p in model.parameters()) / 1e6

    # Inference latency
    model.eval()
    dummy_input = torch.randn(1, num_frames, num_joints, 2).to(device)

    # Warmup
    with torch.no_grad():
        for _ in range(10):
            _ = model(dummy_input)

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    t0 = time.time()
    n_runs = 50
    with torch.no_grad():
        for _ in range(n_runs):
            _ = model(dummy_input)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    latency_ms = (time.time() - t0) / n_runs * 1000

    result = {"params_M": round(params_M, 3), "latency_ms": round(latency_ms, 2)}

    # FLOPs via fvcore (optional)
    try:
        from fvcore.nn import FlopCountAnalysis
        flops = FlopCountAnalysis(model, dummy_input)
        result["flops_G"] = round(flops.total() / 1e9, 3)
    except ImportError:
        logger.info("fvcore not installed — skipping FLOPs computation")
    except Exception as e:
        logger.warning("FLOPs computation failed: %s", e)

    return result


# ---------------------------------------------------------------------------
# Dataset loading (reuse logic from train_ddp)
# ---------------------------------------------------------------------------

def load_dataset(args, total_num):
    dataset_name = "data_3d_" + args.dataset + "_" + args.keypoints + ".npz"
    dataset_path = "golfswing/" + dataset_name
    
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
    else:
        raise KeyError("Invalid dataset: " + args.dataset)

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

    kp_name = "data_2d_" + args.dataset + "_" + args.keypoints + ".npz"
    kp_path = "golfswing/" + kp_name
    
    if not os.path.exists(kp_path):
        fallback_kp = f"/capstor/scratch/cscs/ckuya/golfpose_dataset/golfswing/{kp_name}"
        if os.path.exists(fallback_kp):
            kp_path = fallback_kp
            
    keypoints = np.load(kp_path, allow_pickle=True)
    keypoints_metadata = keypoints["metadata"].item()
    keypoints = keypoints["positions_2d"].item()

    for subject in sorted(keypoints.keys()):
        for action in sorted(keypoints[subject]):
            for cam_idx, kps in enumerate(keypoints[subject][action]):
                keypoints[subject][action][cam_idx] = kps[..., :total_num, :]
    keypoints_metadata["num_joints"] = total_num

    for subject in sorted(dataset.subjects()):
        for action in sorted(dataset[subject].keys()):
            if "positions_3d" not in dataset[subject][action]:
                continue
            for cam_idx in range(len(keypoints[subject][action])):
                mocap_len = dataset[subject][action]["positions_3d"][cam_idx].shape[0]
                if keypoints[subject][action][cam_idx].shape[0] > mocap_len:
                    keypoints[subject][action][cam_idx] = keypoints[subject][action][cam_idx][:mocap_len]

    for subject in sorted(keypoints.keys()):
        for action in sorted(keypoints[subject]):
            for cam_idx, kps in enumerate(keypoints[subject][action]):
                cam = dataset.cameras()[subject][cam_idx]
                kps[..., :2] = normalize_screen_coordinates(kps[..., :2], w=cam["res_w"], h=cam["res_h"])
                keypoints[subject][action][cam_idx] = kps

    return dataset, keypoints, keypoints_metadata


def eval_data_prepare(receptive_field, inputs_2d, inputs_3d):
    """Split sequences into receptive-field chunks."""
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


# ---------------------------------------------------------------------------
# Main evaluation routine
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_action(model, gen, receptive_field,
                    kps_left, kps_right, joints_left, joints_right,
                    human_num, club_num, action_name=None):
    """Evaluate on a single action/generator.

    Returns a dict with all metrics + per-joint arrays.
    """
    model.eval()
    all_predicted = []
    all_target = []

    epoch_loss_3d_pos = 0
    epoch_loss_3d_pos_procrustes = 0
    epoch_loss_3d_pos_scale = 0
    epoch_loss_3d_vel = 0
    epoch_loss_3d_pos_human = 0
    epoch_loss_3d_pos_club = 0
    N = 0

    for _, batch, batch_2d in gen.next_epoch():
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

        pred = model(inputs_2d)
        pred_flip = model(inputs_2d_flip)
        pred_flip[:, :, :, 0] *= -1
        pred_flip[:, :, joints_left + joints_right] = pred_flip[:, :, joints_right + joints_left]
        for i in range(pred.shape[0]):
            pred[i] = (pred[i] + pred_flip[i]) / 2

        error = mpjpe(pred, inputs_3d)
        epoch_loss_3d_pos += inputs_3d.shape[0] * inputs_3d.shape[1] * error.item()
        epoch_loss_3d_pos_scale += inputs_3d.shape[0] * inputs_3d.shape[1] * n_mpjpe(pred, inputs_3d).item()
        N += inputs_3d.shape[0] * inputs_3d.shape[1]

        if club_num != 0:
            err_h = mpjpe(pred[..., :human_num, :], inputs_3d[..., :human_num, :])
            err_c = mpjpe(pred[..., human_num:, :], inputs_3d[..., human_num:, :])
            epoch_loss_3d_pos_human += inputs_3d.shape[0] * inputs_3d.shape[1] * err_h.item()
            epoch_loss_3d_pos_club += inputs_3d.shape[0] * inputs_3d.shape[1] * err_c.item()

        pred_np = pred.cpu().numpy().reshape(-1, inputs_3d.shape[-2], inputs_3d.shape[-1])
        tgt_np = inputs_3d.cpu().numpy().reshape(-1, inputs_3d.shape[-2], inputs_3d.shape[-1])

        epoch_loss_3d_pos_procrustes += inputs_3d.shape[0] * inputs_3d.shape[1] * p_mpjpe(pred_np, tgt_np)
        epoch_loss_3d_vel += inputs_3d.shape[0] * inputs_3d.shape[1] * mean_velocity_error(pred_np, tgt_np)

        all_predicted.append(pred_np)
        all_target.append(tgt_np)

    if N == 0:
        return {}

    all_predicted = np.concatenate(all_predicted, axis=0)
    all_target = np.concatenate(all_target, axis=0)

    # Standard metrics
    e1 = (epoch_loss_3d_pos / N) * 1000
    e2 = (epoch_loss_3d_pos_procrustes / N) * 1000
    e3 = (epoch_loss_3d_pos_scale / N) * 1000
    ev = (epoch_loss_3d_vel / N) * 1000
    e1h = (epoch_loss_3d_pos_human / N) * 1000
    e1c = (epoch_loss_3d_pos_club / N) * 1000

    # Per-joint MPJPE
    per_joint_error = np.mean(np.linalg.norm(all_predicted - all_target, axis=-1), axis=0) * 1000  # (J,)

    # PCK
    pck_50, pck_50_pj = compute_pck(all_predicted, all_target, threshold_mm=50)
    pck_100, pck_100_pj = compute_pck(all_predicted, all_target, threshold_mm=100)
    pck_150, pck_150_pj = compute_pck(all_predicted, all_target, threshold_mm=150)

    # AUC
    auc = compute_auc(all_predicted, all_target)

    result = {
        "action": action_name or "Overall",
        "n_frames": int(N),
        "MPJPE": round(float(e1), 2),
        "P-MPJPE": round(float(e2), 2),
        "N-MPJPE": round(float(e3), 2),
        "MPJVE": round(float(ev), 2),
        "MPJPE_human": round(float(e1h), 2),
        "MPJPE_club": round(float(e1c), 2),
        "PCK@50mm": round(float(pck_50), 2),
        "PCK@100mm": round(float(pck_100), 2),
        "PCK@150mm": round(float(pck_150), 2),
        "AUC": round(float(auc), 2),
        "per_joint_MPJPE": [round(float(x), 2) for x in per_joint_error],
        "per_joint_PCK@50mm": [round(float(x), 2) for x in pck_50_pj],
        "per_joint_PCK@100mm": [round(float(x), 2) for x in pck_100_pj],
        "per_joint_PCK@150mm": [round(float(x), 2) for x in pck_150_pj],
    }
    return result


# ---------------------------------------------------------------------------
# Print formatted results
# ---------------------------------------------------------------------------

def print_results(results, club_num=0):
    """Print a formatted summary table."""
    try:
        from tabulate import tabulate
    except ImportError:
        tabulate = None

    headers = ["Action", "MPJPE↓", "P-MPJPE↓", "N-MPJPE↓", "MPJVE↓",
               "PCK@50↑", "PCK@100↑", "PCK@150↑", "AUC↑"]
    if club_num:
        headers.extend(["Human↓", "Club↓"])

    rows = []
    for r in results:
        row = [
            r["action"], r["MPJPE"], r["P-MPJPE"], r["N-MPJPE"], r["MPJVE"],
            r["PCK@50mm"], r["PCK@100mm"], r["PCK@150mm"], r["AUC"],
        ]
        if club_num:
            row.extend([r["MPJPE_human"], r["MPJPE_club"]])
        rows.append(row)

    if tabulate:
        print("\n" + tabulate(rows, headers=headers, tablefmt="grid", floatfmt=".2f"))
    else:
        print("\n" + " | ".join(headers))
        print("-" * (15 * len(headers)))
        for row in rows:
            print(" | ".join(f"{x:>10}" if isinstance(x, str) else f"{x:>10.2f}" for x in row))

    # Per-joint MPJPE for overall
    overall = [r for r in results if r["action"] == "Overall"]
    if overall:
        pj = overall[0]["per_joint_MPJPE"]
        joint_names = [
            "Hip", "RHip", "RKnee", "RAnkle", "LHip", "LKnee", "LAnkle",
            "Spine", "Thorax", "Neck/Nose", "Head", "LShoulder", "LElbow",
            "LWrist", "RShoulder", "RElbow", "RWrist",
        ]
        if len(pj) > 17:
            joint_names += [f"Club_{i}" for i in range(len(pj) - 17)]

        print("\nPer-joint MPJPE (mm):")
        if tabulate:
            print(tabulate(
                [[n, v] for n, v in zip(joint_names[:len(pj)], pj)],
                headers=["Joint", "MPJPE (mm)"],
                tablefmt="simple",
                floatfmt=".2f",
            ))
        else:
            for n, v in zip(joint_names[:len(pj)], pj):
                print(f"  {n:15s} {v:.2f}")


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="GolfPose comprehensive evaluation")
    p.add_argument("--checkpoint", required=True, type=str, help="Path to .pth checkpoint")
    p.add_argument("--model", default="mixste_lite", type=str,
                   choices=["mixste", "mixste_lite", "linformer", "rela", "temporal_conv"])
    p.add_argument("--model-preset", default="base", type=str, choices=["small", "base"])
    p.add_argument("-d", "--dataset", default="golf", type=str)
    p.add_argument("-k", "--keypoints", default="gt", type=str)
    p.add_argument("-ste", "--subjects-test", default="G5,G6", type=str)
    p.add_argument("-f", "--number-of-frames", default=243, type=int)
    p.add_argument("-cs", default=512, type=int)
    p.add_argument("-dep", default=8, type=int)
    p.add_argument("--club", "--club-num", dest="club_num", default=0, type=int)
    p.add_argument("-a", "--actions", default="*", type=str)
    p.add_argument("--downsample", default=1, type=int)
    p.add_argument("--by-subject", action="store_true")
    p.add_argument("--out", default="", type=str, help="Output JSON file")
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--wandb-project", default="golfpose", type=str)
    p.add_argument("--wandb-run-name", default="", type=str)
    p.add_argument("--no-complexity", action="store_true",
                   help="Skip model complexity analysis")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    setup_logger("golfpose.eval")

    human_num = 17
    club_num = args.club_num
    total_num = human_num + club_num
    receptive_field = args.number_of_frames

    # ---- Load dataset ----
    logger.info("Loading dataset...")
    dataset, keypoints, keypoints_metadata = load_dataset(args, total_num)

    keypoints_symmetry = keypoints_metadata["keypoints_symmetry"]
    kps_left, kps_right = list(keypoints_symmetry[0]), list(keypoints_symmetry[1])
    joints_left = list(dataset.skeleton().joints_left())
    joints_right = list(dataset.skeleton().joints_right())

    num_joints = keypoints_metadata["num_joints"]
    pad = (receptive_field - 1) // 2
    causal_shift = 0

    # ---- Create model ----
    logger.info("Creating model: %s (preset=%s)", args.model, args.model_preset)
    overrides = {}
    if args.model in ("mixste", "mixste_lite", "linformer", "rela"):
        overrides["embed_dim_ratio"] = args.cs
        overrides["depth"] = args.dep

    model = create_lifter(
        args.model, num_joints, receptive_field,
        preset=args.model_preset, drop_path=0.0, **overrides,
    )

    if torch.cuda.is_available():
        model = nn.DataParallel(model)
        model = model.cuda()

    # ---- Load checkpoint ----
    logger.info("Loading checkpoint: %s", args.checkpoint)
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    logger.info("Checkpoint trained for %d epochs", ckpt.get("epoch", -1))
    model.load_state_dict(ckpt["model_pos"], strict=False)
    model.eval()

    # ---- Model complexity ----
    complexity = {}
    if not args.no_complexity:
        logger.info("Computing model complexity...")
        # Use unwrapped model for complexity
        base_model = model.module if hasattr(model, "module") else model
        complexity = compute_model_complexity(base_model, num_joints, receptive_field)
        logger.info("Params: %.3f M | Latency: %.2f ms | FLOPs: %s G",
                     complexity["params_M"], complexity["latency_ms"],
                     complexity.get("flops_G", "N/A"))

    # ---- Organise actions ----
    subjects_test = args.subjects_test.split(",")
    action_filter = None if args.actions == "*" else args.actions.split(",")

    all_actions = {}
    all_actions_by_subject = {}

    for subject in sorted(subjects_test):
        if subject not in all_actions_by_subject:
            all_actions_by_subject[subject] = {}
        for action in sorted(dataset[subject].keys()):
            action_name = action.split(" ")[0]
            if action_filter and not any(action_name.startswith(a) for a in action_filter):
                continue
            all_actions.setdefault(action_name, []).append((subject, action))
            all_actions_by_subject[subject].setdefault(action_name, []).append((subject, action))

    # ---- Evaluate per-action ----
    def run_evaluation(actions_dict, label=""):
        results = []
        for action_key in sorted(actions_dict.keys()):
            poses_3d, poses_2d = [], []
            for subject, action in actions_dict[action_key]:
                p2d = keypoints[subject][action]
                for i in range(len(p2d)):
                    poses_2d.append(p2d[i])
                p3d = dataset[subject][action]["positions_3d"]
                for i in range(len(p3d)):
                    poses_3d.append(p3d[i])

            gen = UnchunkedGenerator_Seq(
                None, poses_3d, poses_2d,
                pad=pad, causal_shift=causal_shift, augment=True,
                kps_left=kps_left, kps_right=kps_right,
                joints_left=joints_left, joints_right=joints_right,
            )

            r = evaluate_action(
                model, gen, receptive_field,
                kps_left, kps_right, joints_left, joints_right,
                human_num, club_num, action_name=action_key,
            )
            if r:
                results.append(r)

        # Overall average
        if results:
            overall = {
                "action": "Overall",
                "n_frames": sum(r["n_frames"] for r in results),
                "MPJPE": round(np.mean([r["MPJPE"] for r in results]), 2),
                "P-MPJPE": round(np.mean([r["P-MPJPE"] for r in results]), 2),
                "N-MPJPE": round(np.mean([r["N-MPJPE"] for r in results]), 2),
                "MPJVE": round(np.mean([r["MPJVE"] for r in results]), 2),
                "MPJPE_human": round(np.mean([r["MPJPE_human"] for r in results]), 2),
                "MPJPE_club": round(np.mean([r["MPJPE_club"] for r in results]), 2),
                "PCK@50mm": round(np.mean([r["PCK@50mm"] for r in results]), 2),
                "PCK@100mm": round(np.mean([r["PCK@100mm"] for r in results]), 2),
                "PCK@150mm": round(np.mean([r["PCK@150mm"] for r in results]), 2),
                "AUC": round(np.mean([r["AUC"] for r in results]), 2),
                "per_joint_MPJPE": [
                    round(np.mean([r["per_joint_MPJPE"][j] for r in results]), 2)
                    for j in range(len(results[0]["per_joint_MPJPE"]))
                ],
                "per_joint_PCK@50mm": [
                    round(np.mean([r["per_joint_PCK@50mm"][j] for r in results]), 2)
                    for j in range(len(results[0]["per_joint_PCK@50mm"]))
                ],
                "per_joint_PCK@100mm": [
                    round(np.mean([r["per_joint_PCK@100mm"][j] for r in results]), 2)
                    for j in range(len(results[0]["per_joint_PCK@100mm"]))
                ],
                "per_joint_PCK@150mm": [
                    round(np.mean([r["per_joint_PCK@150mm"][j] for r in results]), 2)
                    for j in range(len(results[0]["per_joint_PCK@150mm"]))
                ],
            }
            results.append(overall)

        if label:
            print(f"\n{'='*60}")
            print(f"  {label}")
            print(f"{'='*60}")
        print_results(results, club_num)
        return results

    logger.info("Running evaluation...")
    all_results = run_evaluation(all_actions)

    if args.by_subject:
        for subject in sorted(all_actions_by_subject.keys()):
            run_evaluation(all_actions_by_subject[subject], label=f"Subject: {subject}")

    # ---- Assemble output ----
    output = {
        "model": args.model,
        "model_preset": args.model_preset,
        "checkpoint": args.checkpoint,
        "dataset": args.dataset,
        "subjects_test": subjects_test,
        "num_frames": receptive_field,
        "club_num": club_num,
        "complexity": complexity,
        "results": all_results,
    }

    # ---- Save JSON ----
    out_path = args.out
    if not out_path:
        out_dir = os.path.dirname(args.checkpoint) or "."
        out_path = os.path.join(out_dir, "metrics_comprehensive.json")

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    logger.info("Results saved to %s", out_path)

    # ---- WandB ----
    if args.wandb:
        try:
            import wandb
            run = wandb.init(
                project=args.wandb_project,
                name=args.wandb_run_name or f"eval_{args.model}",
                config=vars(args),
            )
            # Log overall metrics
            overall = [r for r in all_results if r["action"] == "Overall"]
            if overall:
                run.summary.update(overall[0])
            run.summary.update(complexity)
            run.finish()
        except Exception as e:
            logger.warning("WandB logging failed: %s", e)

    logger.info("Evaluation complete!")


if __name__ == "__main__":
    main()
