#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""GolfPose post-training visualisation tool.

Generates:
  - 3D pose animations (predicted vs ground truth) as MP4/GIF
  - Per-joint error heatmaps
  - Model comparison bar charts
  - Per-action MPJPE breakdown
  - Training curve plots (from TensorBoard event files)

Usage:
    # Render 3D pose animation
    python tools/visualize.py pose \\
        --checkpoint best.pth --model mixste_lite \\
        -d golf -k gt --club 5 \\
        --subject G5 --action Swing01 --camera 0 \\
        --output-dir vis/

    # Per-joint error heatmap from evaluation JSON
    python tools/visualize.py heatmap \\
        --metrics results/metrics_comprehensive.json \\
        --output-dir vis/

    # Compare models from multiple JSON results
    python tools/visualize.py compare \\
        --metrics results/mixste.json results/linformer.json results/temporal_conv.json \\
        --output-dir vis/

    # Plot training curves from TensorBoard logs
    python tools/visualize.py curves \\
        --tb-logdir checkpoint/mixste_lite/tb_logs \\
        --output-dir vis/
"""

import argparse
import json
import logging
import os
import sys

import numpy as np

# Ensure project root is on path
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from common.logging import setup_logger

logger = logging.getLogger("golfpose.vis")

# Use non-interactive backend for headless rendering
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patches as mpatches


# ============================================================================
# Colour palette
# ============================================================================

COLORS = [
    "#2196F3", "#FF5722", "#4CAF50", "#FFC107", "#9C27B0",
    "#00BCD4", "#E91E63", "#8BC34A", "#FF9800", "#3F51B5",
]

JOINT_NAMES_17 = [
    "Hip", "RHip", "RKnee", "RAnkle", "LHip", "LKnee", "LAnkle",
    "Spine", "Thorax", "Neck", "Head", "LShoulder", "LElbow",
    "LWrist", "RShoulder", "RElbow", "RWrist",
]


# ============================================================================
# Sub-command: pose — 3D animation rendering
# ============================================================================

def cmd_pose(args):
    """Render a 3D pose animation from a checkpoint."""
    import torch
    import torch.nn as nn
    from einops import rearrange
    import torch.nn.functional as F

    from common.camera import (
        normalize_screen_coordinates, image_coordinates,
        vicon_to_world_golf, world_to_camera_golf,
        camera_to_world_golf, world_to_vicon_golf,
    )
    from common.model_lightweight import create_lifter
    from common.generators import UnchunkedGenerator_Seq
    from common.loss import mpjpe

    os.makedirs(args.output_dir, exist_ok=True)

    human_num = 17
    club_num = args.club_num
    total_num = human_num + club_num
    receptive_field = args.number_of_frames

    # Load dataset
    from tools.eval_comprehensive import load_dataset, eval_data_prepare
    dataset, keypoints, keypoints_metadata = load_dataset(args, total_num)

    kps_sym = keypoints_metadata["keypoints_symmetry"]
    kps_left, kps_right = list(kps_sym[0]), list(kps_sym[1])
    joints_left = list(dataset.skeleton().joints_left())
    joints_right = list(dataset.skeleton().joints_right())
    num_joints = keypoints_metadata["num_joints"]
    pad = (receptive_field - 1) // 2

    # Create & load model
    overrides = {}
    if args.model in ("mixste", "mixste_lite", "linformer", "rela"):
        overrides["embed_dim_ratio"] = args.cs
        overrides["depth"] = args.dep

    model = create_lifter(
        args.model, num_joints, receptive_field,
        preset=args.model_preset, drop_path=0.0, **overrides,
    )
    if torch.cuda.is_available():
        model = nn.DataParallel(model).cuda()

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(ckpt["model_pos"], strict=False)
    model.eval()

    # Get input keypoints
    subject = args.subject
    action = args.action
    camera = args.camera

    input_keypoints = keypoints[subject][action][camera].copy()
    cam = dataset.cameras()[subject][camera]
    input_keypoints[..., :2] = normalize_screen_coordinates(input_keypoints[..., :2], w=cam["res_w"], h=cam["res_h"])
    ground_truth = None
    if subject in dataset.subjects() and action in dataset[subject]:
        if "positions_3d" in dataset[subject][action]:
            ground_truth = dataset[subject][action]["positions_3d"][camera].copy()

    gen = UnchunkedGenerator_Seq(
        None, [ground_truth], [input_keypoints],
        pad=pad, causal_shift=0, augment=True,
        kps_left=kps_left, kps_right=kps_right,
        joints_left=joints_left, joints_right=joints_right,
    )

    # Run inference
    with torch.no_grad():
        for _, batch, batch_2d in gen.next_epoch():
            inputs_2d = torch.from_numpy(batch_2d.astype("float32"))
            inputs_3d = torch.from_numpy(batch.astype("float32"))

            inputs_3d_p = inputs_3d
            inputs_2d_eval, inputs_3d_eval = eval_data_prepare(receptive_field, inputs_2d, inputs_3d_p)

            if torch.cuda.is_available():
                inputs_2d_eval = inputs_2d_eval.cuda()

            prediction = model(inputs_2d_eval).squeeze().cpu().numpy()

    # Reshape prediction to match ground truth
    if ground_truth is not None:
        if ground_truth.shape[0] / receptive_field > ground_truth.shape[0] // receptive_field:
            batch_num = ground_truth.shape[0] // receptive_field + 1
            pred2 = np.empty_like(ground_truth)
            for i in range(batch_num - 1):
                pred2[i * receptive_field:(i + 1) * receptive_field] = prediction[i]
            left_frames = ground_truth.shape[0] - (batch_num - 1) * receptive_field
            pred2[-left_frames:] = prediction[-1, -left_frames:]
            prediction = pred2

    # Transform to Vicon space
    cam = dataset.cameras()[subject][camera]
    if ground_truth is not None:
        trajectory = ground_truth[:, :1]
        ground_truth[:, 1:] += trajectory
        prediction += trajectory

        prediction = prediction * 1000
        ground_truth = ground_truth * 1000
        prediction = camera_to_world_golf(prediction, cam["orientation"], cam["translation_mm"])
        ground_truth = camera_to_world_golf(ground_truth, cam["orientation"], cam["translation_mm"])
        prediction = world_to_vicon_golf(prediction, cam["world_to_vicon_basis_dots"])
        ground_truth = world_to_vicon_golf(ground_truth, cam["world_to_vicon_basis_dots"])
        prediction /= 1000
        ground_truth /= 1000

    # Render using existing visualization module
    from common.visualization import render_animation_overlap
    input_kps = image_coordinates(
        input_keypoints[..., :2], w=cam["res_w"], h=cam["res_h"]
    )

    anim_output = {"Reconstruction": prediction}
    if ground_truth is not None:
        anim_output["Ground truth"] = ground_truth

    output_file = os.path.join(args.output_dir, f"{subject}_{action}_cam{camera}.mp4")
    render_animation_overlap(
        input_kps, keypoints_metadata, anim_output,
        dataset.skeleton(), dataset.fps(), 3000, cam.get("azimuth", 70),
        output_file, viewport=(cam["res_w"], cam["res_h"]),
        with_club=club_num != 0,
    )
    logger.info("Saved pose animation: %s", output_file)


# ============================================================================
# Sub-command: heatmap — per-joint error heatmap
# ============================================================================

def cmd_heatmap(args):
    """Generate per-joint error heatmaps from evaluation JSON."""
    os.makedirs(args.output_dir, exist_ok=True)

    with open(args.metrics[0]) as f:
        data = json.load(f)

    results = data.get("results", [])
    overall = [r for r in results if r["action"] == "Overall"]
    if not overall:
        logger.error("No 'Overall' entry in metrics JSON")
        return
    overall = overall[0]

    per_joint = np.array(overall["per_joint_MPJPE"])
    n_joints = len(per_joint)
    joint_names = JOINT_NAMES_17[:min(n_joints, 17)]
    if n_joints > 17:
        joint_names += [f"Club_{i}" for i in range(n_joints - 17)]

    # ---- Bar chart ----
    fig, ax = plt.subplots(figsize=(max(12, n_joints * 0.6), 6))
    bars = ax.bar(range(n_joints), per_joint, color=COLORS[:n_joints], edgecolor="white", linewidth=0.5)
    ax.set_xticks(range(n_joints))
    ax.set_xticklabels(joint_names, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("MPJPE (mm)", fontsize=12)
    ax.set_title(f"Per-joint MPJPE — {data.get('model', 'model')}", fontsize=14)
    ax.grid(axis="y", alpha=0.3)

    # Add value labels on bars
    for bar, val in zip(bars, per_joint):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{val:.1f}", ha="center", va="bottom", fontsize=8)

    fig.tight_layout()
    out_path = os.path.join(args.output_dir, "per_joint_mpjpe.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("Saved per-joint MPJPE chart: %s", out_path)

    # ---- PCK comparison across thresholds ----
    fig2, ax2 = plt.subplots(figsize=(max(12, n_joints * 0.6), 6))
    x = np.arange(n_joints)
    width = 0.25

    pck50 = np.array(overall.get("per_joint_PCK@50mm", [0] * n_joints))
    pck100 = np.array(overall.get("per_joint_PCK@100mm", [0] * n_joints))
    pck150 = np.array(overall.get("per_joint_PCK@150mm", [0] * n_joints))

    ax2.bar(x - width, pck50, width, label="PCK@50mm", color="#2196F3", alpha=0.85)
    ax2.bar(x, pck100, width, label="PCK@100mm", color="#4CAF50", alpha=0.85)
    ax2.bar(x + width, pck150, width, label="PCK@150mm", color="#FF9800", alpha=0.85)

    ax2.set_xticks(x)
    ax2.set_xticklabels(joint_names, rotation=45, ha="right", fontsize=9)
    ax2.set_ylabel("PCK (%)", fontsize=12)
    ax2.set_title(f"Per-joint PCK — {data.get('model', 'model')}", fontsize=14)
    ax2.legend()
    ax2.grid(axis="y", alpha=0.3)
    fig2.tight_layout()
    out_path2 = os.path.join(args.output_dir, "per_joint_pck.png")
    fig2.savefig(out_path2, dpi=150)
    plt.close(fig2)
    logger.info("Saved per-joint PCK chart: %s", out_path2)

    # ---- Error heatmap matrix (action × joint) ----
    action_results = [r for r in results if r["action"] != "Overall"]
    if action_results:
        action_names = [r["action"] for r in action_results]
        error_matrix = np.array([r["per_joint_MPJPE"] for r in action_results])

        fig3, ax3 = plt.subplots(figsize=(max(10, n_joints * 0.5), max(6, len(action_names) * 0.4)))
        im = ax3.imshow(error_matrix, cmap="YlOrRd", aspect="auto")
        ax3.set_xticks(range(n_joints))
        ax3.set_xticklabels(joint_names, rotation=45, ha="right", fontsize=8)
        ax3.set_yticks(range(len(action_names)))
        ax3.set_yticklabels(action_names, fontsize=9)
        ax3.set_title("MPJPE Heatmap (Action × Joint)", fontsize=14)
        plt.colorbar(im, ax=ax3, label="MPJPE (mm)")
        fig3.tight_layout()
        out_path3 = os.path.join(args.output_dir, "error_heatmap.png")
        fig3.savefig(out_path3, dpi=150)
        plt.close(fig3)
        logger.info("Saved error heatmap: %s", out_path3)


# ============================================================================
# Sub-command: compare — multi-model comparison
# ============================================================================

def cmd_compare(args):
    """Compare multiple models from their evaluation JSONs."""
    os.makedirs(args.output_dir, exist_ok=True)

    models_data = []
    for path in args.metrics:
        with open(path) as f:
            data = json.load(f)
        overall = [r for r in data.get("results", []) if r["action"] == "Overall"]
        if overall:
            entry = {
                "name": data.get("model", os.path.basename(path).replace(".json", "")),
                **overall[0],
                **data.get("complexity", {}),
            }
            models_data.append(entry)

    if not models_data:
        logger.error("No valid results found in provided JSON files")
        return

    names = [m["name"] for m in models_data]

    # ---- MPJPE / P-MPJPE / N-MPJPE comparison ----
    fig, axes = plt.subplots(1, 4, figsize=(20, 6))

    for ax, metric, label in zip(
        axes,
        ["MPJPE", "P-MPJPE", "N-MPJPE", "MPJVE"],
        ["MPJPE (mm) ↓", "P-MPJPE (mm) ↓", "N-MPJPE (mm) ↓", "MPJVE (mm) ↓"],
    ):
        values = [m.get(metric, 0) for m in models_data]
        bars = ax.barh(names, values, color=COLORS[:len(names)], edgecolor="white")
        ax.set_xlabel(label, fontsize=11)
        ax.invert_yaxis()
        ax.grid(axis="x", alpha=0.3)
        for bar, val in zip(bars, values):
            ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                    f"{val:.1f}", va="center", fontsize=9)

    fig.suptitle("Model Comparison — GolfPose 3D Lifting", fontsize=15, y=1.02)
    fig.tight_layout()
    out_path = os.path.join(args.output_dir, "model_comparison.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved model comparison: %s", out_path)

    # ---- PCK & AUC comparison ----
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    x = np.arange(len(names))
    width = 0.2

    pck50 = [m.get("PCK@50mm", 0) for m in models_data]
    pck100 = [m.get("PCK@100mm", 0) for m in models_data]
    pck150 = [m.get("PCK@150mm", 0) for m in models_data]
    aucs = [m.get("AUC", 0) for m in models_data]

    ax2.bar(x - 1.5 * width, pck50, width, label="PCK@50mm", color="#2196F3")
    ax2.bar(x - 0.5 * width, pck100, width, label="PCK@100mm", color="#4CAF50")
    ax2.bar(x + 0.5 * width, pck150, width, label="PCK@150mm", color="#FF9800")
    ax2.bar(x + 1.5 * width, aucs, width, label="AUC", color="#9C27B0")

    ax2.set_xticks(x)
    ax2.set_xticklabels(names, fontsize=10)
    ax2.set_ylabel("% / Score", fontsize=12)
    ax2.set_title("PCK & AUC Comparison", fontsize=14)
    ax2.legend()
    ax2.grid(axis="y", alpha=0.3)
    fig2.tight_layout()
    out_path2 = os.path.join(args.output_dir, "pck_auc_comparison.png")
    fig2.savefig(out_path2, dpi=150)
    plt.close(fig2)
    logger.info("Saved PCK/AUC comparison: %s", out_path2)

    # ---- Params vs MPJPE scatter ----
    fig3, ax3 = plt.subplots(figsize=(8, 6))
    for i, m in enumerate(models_data):
        params = m.get("params_M", 0)
        mpjpe_val = m.get("MPJPE", 0)
        ax3.scatter(params, mpjpe_val, s=150, c=COLORS[i], edgecolors="black", zorder=5)
        ax3.annotate(m["name"], (params, mpjpe_val),
                     textcoords="offset points", xytext=(8, 5), fontsize=9)

    ax3.set_xlabel("Parameters (M)", fontsize=12)
    ax3.set_ylabel("MPJPE (mm) ↓", fontsize=12)
    ax3.set_title("Model Efficiency: Params vs MPJPE", fontsize=14)
    ax3.grid(alpha=0.3)
    fig3.tight_layout()
    out_path3 = os.path.join(args.output_dir, "params_vs_mpjpe.png")
    fig3.savefig(out_path3, dpi=150)
    plt.close(fig3)
    logger.info("Saved params vs MPJPE: %s", out_path3)


# ============================================================================
# Sub-command: curves — training curve plots
# ============================================================================

def cmd_curves(args):
    """Plot training curves from TensorBoard event files."""
    os.makedirs(args.output_dir, exist_ok=True)

    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except ImportError:
        logger.error("tensorboard is required for curve plotting: pip install tensorboard")
        return

    ea = EventAccumulator(args.tb_logdir)
    ea.Reload()

    available_tags = ea.Tags().get("scalars", [])
    logger.info("Available scalar tags: %s", available_tags)

    # Group related metrics
    metric_groups = {
        "Loss": [t for t in available_tags if "loss" in t.lower() or "Loss" in t],
        "MPJPE": [t for t in available_tags if "mpjpe" in t.lower() or "MPJPE" in t],
        "Learning Rate": [t for t in available_tags if "lr" in t.lower() or "LR" in t or "learning" in t.lower()],
        "Velocity": [t for t in available_tags if "vel" in t.lower() or "MPJVE" in t],
    }

    for group_name, tags in metric_groups.items():
        if not tags:
            continue

        fig, ax = plt.subplots(figsize=(12, 6))
        for i, tag in enumerate(tags):
            events = ea.Scalars(tag)
            steps = [e.step for e in events]
            values = [e.value for e in events]
            label = tag.split("/")[-1] if "/" in tag else tag
            ax.plot(steps, values, label=label, color=COLORS[i % len(COLORS)], linewidth=1.5)

        ax.set_xlabel("Epoch", fontsize=12)
        ax.set_ylabel(group_name, fontsize=12)
        ax.set_title(f"Training Curves — {group_name}", fontsize=14)
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)
        fig.tight_layout()

        fname = group_name.lower().replace(" ", "_")
        out_path = os.path.join(args.output_dir, f"curves_{fname}.png")
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        logger.info("Saved training curves (%s): %s", group_name, out_path)


# ============================================================================
# Argument parsing
# ============================================================================

def build_parser():
    parser = argparse.ArgumentParser(description="GolfPose visualisation tool")
    subparsers = parser.add_subparsers(dest="command", help="Visualisation sub-command")

    # ---- pose ----
    sp_pose = subparsers.add_parser("pose", help="Render 3D pose animation")
    sp_pose.add_argument("--checkpoint", required=True, type=str)
    sp_pose.add_argument("--model", default="mixste_lite", type=str)
    sp_pose.add_argument("--model-preset", default="base", type=str)
    sp_pose.add_argument("-d", "--dataset", default="golf", type=str)
    sp_pose.add_argument("-k", "--keypoints", default="gt", type=str)
    sp_pose.add_argument("-f", "--number-of-frames", default=243, type=int)
    sp_pose.add_argument("-cs", default=512, type=int)
    sp_pose.add_argument("-dep", default=8, type=int)
    sp_pose.add_argument("--club", "--club-num", dest="club_num", default=0, type=int)
    sp_pose.add_argument("--subject", required=True, type=str)
    sp_pose.add_argument("--action", required=True, type=str)
    sp_pose.add_argument("--camera", default=0, type=int)
    sp_pose.add_argument("--output-dir", default="vis/", type=str)
    sp_pose.add_argument("--downsample", default=1, type=int)

    # ---- heatmap ----
    sp_heat = subparsers.add_parser("heatmap", help="Per-joint error heatmap")
    sp_heat.add_argument("--metrics", nargs="+", required=True, type=str,
                         help="Path(s) to evaluation JSON")
    sp_heat.add_argument("--output-dir", default="vis/", type=str)

    # ---- compare ----
    sp_cmp = subparsers.add_parser("compare", help="Compare models")
    sp_cmp.add_argument("--metrics", nargs="+", required=True, type=str,
                        help="Paths to evaluation JSONs from different models")
    sp_cmp.add_argument("--output-dir", default="vis/", type=str)

    # ---- curves ----
    sp_curves = subparsers.add_parser("curves", help="Plot training curves")
    sp_curves.add_argument("--tb-logdir", required=True, type=str,
                           help="TensorBoard log directory")
    sp_curves.add_argument("--output-dir", default="vis/", type=str)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    setup_logger("golfpose.vis")

    if args.command is None:
        parser.print_help()
        return

    dispatch = {
        "pose": cmd_pose,
        "heatmap": cmd_heatmap,
        "compare": cmd_compare,
        "curves": cmd_curves,
    }
    dispatch[args.command](args)


if __name__ == "__main__":
    main()
