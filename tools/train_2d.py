#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Thin wrapper around mmengine Runner for MMPose 2D keypoint training.

Supports ``--launcher pytorch`` (torchrun) and ``--launcher slurm``
(mmengine reads SLURM env vars for distributed setup).

Usage:
    # Single GPU
    python tools/train_2d.py configs/mmpose/lightweight/rtmpose_tiny_golfer_256x192.py

    # Multi-GPU via torchrun
    torchrun --nproc_per_node=4 tools/train_2d.py \\
        configs/mmpose/lightweight/rtmpose_tiny_golfer_256x192.py \\
        --launcher pytorch

    # SLURM distributed (via slurm/train_2d_slurm.sh)
    python tools/train_2d.py config.py --launcher slurm --work-dir $WORK_DIR
"""

import argparse
import os
import sys

# Ensure project root is on path
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)


def parse_args():
    parser = argparse.ArgumentParser(description="MMPose 2D training launcher")
    parser.add_argument("config", help="MMPose config file path")
    parser.add_argument("--work-dir", help="directory to save logs and models")
    parser.add_argument("--launcher", choices=["none", "pytorch", "slurm", "mpi"],
                        default="none", help="distributed launcher")
    parser.add_argument("--amp", action="store_true", help="enable mixed precision")
    parser.add_argument("--resume", nargs="?", const="auto", default=None,
                        help="resume from the latest checkpoint (auto or path)")
    parser.add_argument("--cfg-options", nargs="+", default=[],
                        help="override config settings (key=value)")
    args, unknown = parser.parse_known_args()
    return args


def main():
    args = parse_args()

    try:
        from mmengine.config import Config, DictAction
        from mmengine.runner import Runner
    except ImportError:
        print("ERROR: mmengine is required.  Install with: pip install mmengine")
        sys.exit(1)

    # Load config
    cfg = Config.fromfile(args.config)

    # Apply CLI overrides
    if args.cfg_options:
        for opt in args.cfg_options:
            key, val = opt.split("=", 1)
            # Simple type inference
            try:
                val = eval(val)
            except Exception:
                pass
            keys = key.split(".")
            d = cfg
            for k in keys[:-1]:
                d = d[k]
            d[keys[-1]] = val

    # Work dir
    if args.work_dir is not None:
        cfg.work_dir = args.work_dir
    elif not hasattr(cfg, "work_dir") or cfg.work_dir is None:
        cfg.work_dir = os.path.join(
            "work_dirs", os.path.splitext(os.path.basename(args.config))[0]
        )

    # Launcher
    cfg.launcher = args.launcher

    # AMP
    if args.amp:
        from mmengine.optim import AmpOptimWrapper
        optim_wrapper = cfg.get("optim_wrapper", {})
        if optim_wrapper.get("type", "OptimWrapper") == "OptimWrapper":
            optim_wrapper["type"] = "AmpOptimWrapper"
            optim_wrapper.setdefault("loss_scale", "dynamic")
            cfg.optim_wrapper = optim_wrapper

    # Resume
    if args.resume is not None:
        cfg.resume = True
        if args.resume != "auto":
            cfg.load_from = args.resume

    # Build runner
    runner = Runner.from_cfg(cfg)
    runner.train()


if __name__ == "__main__":
    main()
