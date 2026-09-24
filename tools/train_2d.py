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

# Ensure project root is on path and working directory
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)
try:
    os.chdir(_project_root)
    print(f"[train_2d.py] Working directory set to {_project_root}")
except Exception as e:
    print(f"[train_2d.py] Warning: could not chdir to {_project_root}: {e}")


def _setup_dataset_link(root_dir):
    data_dir = os.environ.get("DATA_DIR", "/capstor/scratch/cscs/ckuya/golfpose_dataset/golfswing")
    if not os.path.isdir(data_dir):
        alt = "/capstor/scratch/cscs/ckuya/golfpose_dataset"
        if os.path.isdir(os.path.join(alt, "golfswing")):
            data_dir = os.path.join(alt, "golfswing")

    if os.path.isdir(data_dir):
        target = os.path.join(root_dir, "golfswing")
        if not os.path.exists(target) and not os.path.islink(target):
            try:
                os.symlink(data_dir, target)
                print(f"[train_2d.py] Symlinked dataset {target} -> {data_dir}")
                return
            except Exception as e:
                print(f"[train_2d.py] Notice: could not symlink {target}: {e}")

        # If target directory already exists, link items inside it and clean broken symlinks
        try:
            os.makedirs(target, exist_ok=True)
            for entry in os.listdir(data_dir):
                sub_src = os.path.join(data_dir, entry)
                sub_target = os.path.join(target, entry)

                if entry == "coco" and os.path.isdir(sub_src):
                    os.makedirs(sub_target, exist_ok=True)
                    for json_file in os.listdir(sub_src):
                        src_json = os.path.join(sub_src, json_file)
                        tgt_json = os.path.join(sub_target, json_file)
                        if os.path.islink(tgt_json) and not os.path.exists(tgt_json):
                            try:
                                os.unlink(tgt_json)
                            except OSError:
                                pass
                        if not os.path.exists(tgt_json) and not os.path.islink(tgt_json):
                            try:
                                os.symlink(src_json, tgt_json)
                            except Exception:
                                pass
                else:
                    if os.path.islink(sub_target) and not os.path.exists(sub_target):
                        try:
                            os.unlink(sub_target)
                        except OSError:
                            pass
                    if not os.path.exists(sub_target) and not os.path.islink(sub_target):
                        try:
                            os.symlink(sub_src, sub_target)
                            print(f"[train_2d.py] Linked dataset item: {entry}")
                        except Exception:
                            pass
        except Exception as e:
            print(f"[train_2d.py] Warning while linking dataset: {e}")


_setup_dataset_link(_project_root)


def _normalize_config_paths(cfg, root_dir, data_root=None):
    """Ensure relative dataset metainfo and data paths in config resolve to absolute paths."""
    data_dir_env = os.environ.get("DATA_DIR") or os.environ.get("GOLFPOSE_DATA_ROOT")
    scratch_dir = os.environ.get("SCRATCH", "/capstor/scratch/cscs/ckuya")
    candidates = [
        data_root,
        data_dir_env,
        "/capstor/scratch/cscs/ckuya/golfpose_dataset/golfswing",
        os.path.join(scratch_dir, "golfpose_dataset", "golfswing") if scratch_dir else None,
        os.path.join(root_dir, "golfswing"),
    ]
    effective_data_root = None
    for c in candidates:
        if c and os.path.isdir(c):
            # Prefer directory containing coco annotation json
            coco_check = os.path.join(c, "coco", "hscc_golf_person_2d_train.json")
            if os.path.isfile(coco_check) or os.path.isdir(os.path.join(c, "coco")):
                effective_data_root = os.path.abspath(c)
                break
            elif not effective_data_root:
                effective_data_root = os.path.abspath(c)

    print(f"[train_2d.py] Effective dataset root: {effective_data_root}")
    if not effective_data_root:
        print("[train_2d.py] WARNING: Could not find valid dataset directory among candidates:")
        for c in candidates:
            if c:
                print(f"  - {c} (exists={os.path.exists(c)})")

    if effective_data_root and hasattr(cfg, "data_root"):
        cfg.data_root = os.path.join(effective_data_root, "")

    def _walk(obj):
        if isinstance(obj, dict):
            # Normalize metainfo from_file
            if "from_file" in obj and isinstance(obj["from_file"], str):
                if not os.path.isabs(obj["from_file"]):
                    candidate = os.path.join(root_dir, obj["from_file"])
                    if os.path.exists(candidate):
                        obj["from_file"] = candidate

            # Normalize data_root if found
            if effective_data_root:
                if "data_root" in obj and isinstance(obj["data_root"], str):
                    if obj["data_root"].rstrip("/").endswith("golfswing") or not os.path.isabs(obj["data_root"]):
                        obj["data_root"] = os.path.join(effective_data_root, "")

                if "ann_file" in obj and isinstance(obj["ann_file"], str):
                    ann = obj["ann_file"]
                    if ann.startswith("golfswing/"):
                        obj["ann_file"] = os.path.join(effective_data_root, ann[len("golfswing/"):])
                    elif not os.path.isabs(ann):
                        cand = os.path.join(effective_data_root, ann)
                        if os.path.exists(cand):
                            obj["ann_file"] = cand

            for v in obj.values():
                _walk(v)
        elif isinstance(obj, list):
            for item in obj:
                _walk(item)

    _walk(cfg)


def _patch_torch_load():
    try:
        import torch
        import functools
        _orig_load = torch.load

        @functools.wraps(_orig_load)
        def _safe_load(f, map_location=None, pickle_module=None, *, weights_only=False, **kwargs):
            try:
                import numpy as np
                import torch.serialization as _ts
                if hasattr(_ts, "safe_globals"):
                    _globals = [
                        np.core.multiarray._reconstruct,
                        np.core.multiarray.scalar,
                        np.ndarray,
                        np.dtype,
                    ]
                    with _ts.safe_globals(_globals):
                        return _orig_load(f, map_location=map_location, weights_only=False, **kwargs)
                return _orig_load(f, map_location=map_location, weights_only=False, **kwargs)
            except Exception:
                return _orig_load(f, map_location=map_location, weights_only=False, **kwargs)

        torch.load = _safe_load
        print("[train_2d.py] torch.load patched for PyTorch checkpoint compatibility")
    except Exception as e:
        print(f"[train_2d.py] torch.load patch failed: {e}")


def _set_dist_env_from_slurm():
    for var, slurm_var in [('RANK', 'SLURM_PROCID'), ('LOCAL_RANK', 'SLURM_LOCALID'), ('WORLD_SIZE', 'SLURM_NTASKS')]:
        val = os.environ.get(slurm_var)
        if val is not None and var not in os.environ:
            os.environ[var] = val
            print(f"[train_2d.py] Set {var}={val} from SLURM env")


def _patch_mmengine_registry():
    try:
        from mmengine.registry import Registry
        _orig = Registry._register_module

        def _tolerant(self, module, module_name=None, force=False):
            try:
                _orig(self, module=module, module_name=module_name, force=force)
            except KeyError as e:
                if "already registered" not in str(e):
                    raise

        Registry._register_module = _tolerant
        print("[train_2d.py] mmengine Registry patched to tolerate duplicate optimizer registrations")
    except Exception as e:
        print(f"[train_2d.py] registry patch failed: {e}")


def _patch_base_pose_estimator():
    try:
        import functools
        from mmpose.models.pose_estimators.base import BasePoseEstimator
        _orig = BasePoseEstimator.forward
        _warned = [False]

        @functools.wraps(_orig)
        def _forward(self, inputs, data_samples=None, mode='tensor', **kwargs):
            if kwargs and not _warned[0]:
                print(f"[train_2d.py] dropping metainfo kwargs {list(kwargs.keys())} (logged once)", flush=True)
                _warned[0] = True
            return _orig(self, inputs, data_samples=data_samples, mode=mode)

        BasePoseEstimator.forward = _forward
    except Exception:
        pass


# Apply patches immediately before importing mmengine runner/optim
_patch_torch_load()
_set_dist_env_from_slurm()
_patch_mmengine_registry()
_patch_base_pose_estimator()


def parse_args():
    parser = argparse.ArgumentParser(description="MMPose 2D training launcher")
    parser.add_argument("config", help="MMPose config file path")
    parser.add_argument("--work-dir", help="directory to save logs and models")
    parser.add_argument("--data-root", "--data-dir", default=None,
                        help="dataset root directory (e.g. /capstor/scratch/cscs/ckuya/golfpose_dataset/golfswing)")
    parser.add_argument("--launcher", choices=["none", "pytorch", "slurm", "mpi"],
                        default="none", help="distributed launcher")
    parser.add_argument("--amp", action="store_true", help="enable mixed precision")
    parser.add_argument("--resume", nargs="?", const="auto", default=None,
                        help="resume from the latest checkpoint (auto or path)")
    parser.add_argument("--resume-from", type=str, default=None,
                        help="resume from a specific checkpoint file path")
    parser.add_argument("--auto-scale-lr", action="store_true",
                        help="enable automatic learning rate scaling")
    parser.add_argument("--local_rank", "--local-rank", type=int, default=0)
    parser.add_argument("--cfg-options", nargs="+", default=[],
                        help="override config settings (key=value)")
    args, unknown = parser.parse_known_args()
    if unknown:
        print(f"[train_2d.py] ignoring unknown args: {unknown}")
    return args


def main():
    args = parse_args()

    if "LOCAL_RANK" not in os.environ:
        os.environ["LOCAL_RANK"] = str(args.local_rank)

    print(f'[train_2d.py] RANK={os.environ.get("RANK","?")} '
          f'LOCAL_RANK={os.environ.get("LOCAL_RANK","?")} '
          f'WORLD_SIZE={os.environ.get("WORLD_SIZE","?")} '
          f'MASTER={os.environ.get("MASTER_ADDR","?")}:{os.environ.get("MASTER_PORT","?")}')

    try:
        from mmengine.config import Config, DictAction
        from mmengine.runner import Runner
    except ImportError:
        print("ERROR: mmengine is required.  Install with: pip install mmengine")
        sys.exit(1)

    _patch_base_pose_estimator()

    # Load config
    cfg = Config.fromfile(args.config)
    _normalize_config_paths(cfg, _project_root, data_root=args.data_root)

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

    # Auto scale lr
    if args.auto_scale_lr:
        if hasattr(cfg, "auto_scale_lr"):
            cfg.auto_scale_lr.enable = True
        else:
            cfg.auto_scale_lr = dict(enable=True, base_batch_size=4096)

    # Resume
    if args.resume_from:
        cfg.resume = True
        cfg.load_from = args.resume_from
        print(f"[train_2d.py] Resuming from: {args.resume_from}")
    elif args.resume is not None:
        cfg.resume = True
        if args.resume != "auto":
            cfg.load_from = args.resume

    # Build runner
    runner = Runner.from_cfg(cfg)
    runner.train()


if __name__ == "__main__":
    main()
