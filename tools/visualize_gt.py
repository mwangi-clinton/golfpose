#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""GolfPose ground truth 3D animation visualisation tool.

Usage:
    python tools/visualize_gt.py \
        -d golf -k gt --club 5 \
        --subject G5 --action Swing01 --camera 0 \
        --output-dir vis/ \
        --format gif
"""

import argparse
import logging
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, writers

# Ensure project root is on path
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from common.logging import setup_logger

logger = logging.getLogger("golfpose.vis_gt")

def render_single_3d(ground_truth, skeleton, fps, output, with_club=False, azim=70):
    plt.ioff()
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    ax.view_init(elev=15., azim=azim)
    
    radius = 2
    ax.set_xlim3d([-radius/2, radius/2])
    ax.set_zlim3d([0, radius])
    ax.set_ylim3d([-radius/2, radius/2])
    try:
        ax.set_aspect('equal')
    except NotImplementedError:
        ax.set_aspect('auto')
        
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])
    ax.dist = 7.5
    
    lines_3d = []
    parents = skeleton.parents()
    
    # swap y and z so the character is upright in matplotlib
    data = np.copy(ground_truth)
    temp = np.copy(data[:, :, 1])
    data[:, :, 1] = data[:, :, 0]
    data[:, :, 0] = temp
    trajectory = data[:, 0, [0, 1]]
    
    limit = len(data)
    initialized = False
    
    def update_video(i):
        nonlocal initialized, lines_3d
        
        ax.set_xlim3d([-radius/2 + trajectory[i, 0], radius/2 + trajectory[i, 0]])
        ax.set_ylim3d([-radius/2 + trajectory[i, 1], radius/2 + trajectory[i, 1]])
        ax.set_title(f"Ground Truth: {i}/{limit}")
        
        if not initialized:
            for j, j_parent in enumerate(parents):
                if j_parent == -1:
                    continue
                
                col = 'orange' if j in skeleton.joints_right() else 'green'
                
                # special handling if we have club
                if with_club and j >= 17:
                    col = 'blue'
                elif not with_club and j >= 17:
                    lines_3d.append(None)
                    continue
                
                pos = data[i]
                line = ax.plot([pos[j, 0], pos[j_parent, 0]],
                               [pos[j, 1], pos[j_parent, 1]],
                               [pos[j, 2], pos[j_parent, 2]], zdir='z', c=col)
                lines_3d.append(line)
            initialized = True
        else:
            line_idx = 0
            for j, j_parent in enumerate(parents):
                if j_parent == -1:
                    continue
                if not with_club and j >= 17:
                    continue
                
                pos = data[i]
                lines_3d[line_idx][0].set_xdata(np.array([pos[j, 0], pos[j_parent, 0]]))
                lines_3d[line_idx][0].set_ydata(np.array([pos[j, 1], pos[j_parent, 1]]))
                lines_3d[line_idx][0].set_3d_properties(np.array([pos[j, 2], pos[j_parent, 2]]), zdir='z')
                line_idx += 1
                
        print(f'{i}/{limit}      ', end='\r')

    fig.tight_layout()

    anim = FuncAnimation(fig, update_video, frames=np.arange(0, limit), interval=1000/fps, repeat=False)
    if output.endswith('.mp4'):
        Writer = writers['ffmpeg']
        writer = Writer(fps=fps, metadata={}, bitrate=3000)
        anim.save(output, writer=writer)
    elif output.endswith('.gif'):
        anim.save(output, dpi=80, writer='imagemagick')
    else:
        raise ValueError('Unsupported output format')
    plt.close()

def cmd_pose_gt(args):
    """Render a 3D pose animation from ground truth data only."""
    from common.camera import camera_to_world_golf, world_to_vicon_golf

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

    output_file = os.path.join(args.output_dir, f"{subject}_{action}_cam{camera}_gt_only.{args.format}")
    
    render_single_3d(
        ground_truth, dataset.skeleton(), dataset.fps(), 
        output_file, with_club=club_num != 0, azim=cam.get("azimuth", 70)
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
