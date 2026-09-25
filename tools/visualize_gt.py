#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""GolfPose ground truth 3D animation visualisation tool.

Usage:
    python tools/visualize_gt.py \\
        -d golf -k gt --club 5 \\
        --subject G5 --action Swing01 --camera 0 \\
        --output-dir vis/
"""

import argparse
import logging
import os
import sys

import numpy as np

# Ensure project root is on path
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from common.logging import setup_logger

logger = logging.getLogger("golfpose.vis_gt")

# Use non-interactive backend for headless rendering
import matplotlib
matplotlib.use("Agg")

def cmd_pose_gt(args):
    """Render a 3D pose animation from ground truth data only."""
    from common.camera import (
        normalize_screen_coordinates, image_coordinates,
        camera_to_world_golf, world_to_vicon_golf,
    )

    os.makedirs(args.output_dir, exist_ok=True)

    human_num = 17
    club_num = args.club_num
    total_num = human_num + club_num

    # Load dataset
    from tools.eval_comprehensive import load_dataset
    dataset, keypoints, keypoints_metadata = load_dataset(args, total_num)

    # Get input keypoints and ground truth
    subject = args.subject
    action = args.action
    camera = args.camera

    try:
        input_keypoints = keypoints[subject][action][camera].copy()
    except (KeyError, IndexError):
        logger.error(f"Data not found for subject {subject}, action {action}, camera {camera}")
        return
    cam = dataset.cameras()[subject][camera]
    
    ground_truth = None
    if subject in dataset.subjects() and action in dataset[subject]:
        if "positions_3d" in dataset[subject][action]:
            ground_truth = dataset[subject][action]["positions_3d"][camera].copy()

    if ground_truth is None:
        logger.error(f"Ground truth 3D positions not found for subject {subject}, action {action}")
        return

    # Transform to Vicon space
    trajectory = ground_truth[:, :1]
    ground_truth[:, 1:] += trajectory

    ground_truth = ground_truth * 1000
    ground_truth = camera_to_world_golf(ground_truth, cam["orientation"], cam["translation_mm"])
    ground_truth = world_to_vicon_golf(ground_truth, cam["world_to_vicon_basis_dots"])
    ground_truth /= 1000

    # Render using existing visualization module
    from common.visualization import render_animation_overlap
    input_kps = image_coordinates(
        normalize_screen_coordinates(input_keypoints[..., :2], w=cam["res_w"], h=cam["res_h"]), 
        w=cam["res_w"], h=cam["res_h"]
    )

    anim_output = {"Ground truth": ground_truth}

    output_file = os.path.join(args.output_dir, f"{subject}_{action}_cam{camera}_gt_only.{args.format}")
    render_animation_overlap(
        input_kps, keypoints_metadata, anim_output,
        dataset.skeleton(), dataset.fps(), 3000, cam.get("azimuth", 70),
        output_file, viewport=(cam["res_w"], cam["res_h"]),
        with_club=club_num != 0,
    )
    logger.info("Saved ground truth pose animation: %s", output_file)

def build_parser():
    parser = argparse.ArgumentParser(description="GolfPose GT visualisation tool")
    parser.add_argument("-d", "--dataset", default="golf", type=str)
    parser.add_argument("-k", "--keypoints", default="gt", type=str)
    parser.add_argument("--club", "--club-num", dest="club_num", default=0, type=int)
    parser.add_argument("--subject", required=True, type=str)
    parser.add_argument("--action", required=True, type=str)
    parser.add_argument("--camera", default=0, type=int)
    parser.add_argument("--output-dir", default="vis/", type=str)
    parser.add_argument("--format", default="mp4", choices=["mp4", "gif"], type=str, help="Output format (mp4 or gif)")
    parser.add_argument("--downsample", default=1, type=int)
    return parser

def main():
    parser = build_parser()
    args = parser.parse_args()

    setup_logger("golfpose.vis_gt")
    cmd_pose_gt(args)

if __name__ == "__main__":
    main()
