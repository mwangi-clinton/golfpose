#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Thin wrapper around mmdet's training entrypoint.

MMDet 3.x installs its train script at:
  <mmdet_package>/.mim/tools/train.py

This wrapper locates it automatically so SLURM can call:
  python tools/train_det.py <config> --launcher slurm --work-dir <dir>

This avoids the /usr/bin/python vs python3 conflict inside HPC containers
where `python -m mmdet.train` fails but `python tools/train_det.py` works.
"""
import os
import runpy
import sys

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
        print("[train_det.py] mmengine Registry patched to tolerate duplicate optimizer registrations")
    except Exception as e:
        print(f"[train_det.py] registry patch failed: {e}")

_patch_mmengine_registry()

def _patch_mmcv_nms():
    try:
        import torch
        import torchvision
        import mmcv.ops.nms
        
        def _tv_nms(boxes, scores, iou_threshold, offset=0, score_threshold=0, max_num=-1):
            is_numpy = False
            if isinstance(boxes, torch.Tensor) and not boxes.is_cuda:
                pass
            if not isinstance(boxes, torch.Tensor):
                is_numpy = True
                boxes = torch.from_numpy(boxes)
                scores = torch.from_numpy(scores)

            if boxes.numel() == 0:
                empty_dets = torch.empty((0, 5), device=boxes.device)
                empty_inds = torch.empty((0,), dtype=torch.long, device=boxes.device)
                return (empty_dets.numpy(), empty_inds.numpy()) if is_numpy else (empty_dets, empty_inds)

            # Optional mmcv score filtering
            if score_threshold > 0:
                valid_mask = scores > score_threshold
                inds = valid_mask.nonzero(as_tuple=False).squeeze(1)
                valid_boxes, valid_scores = boxes[inds], scores[inds]
            else:
                inds = torch.arange(boxes.size(0), device=boxes.device)
                valid_boxes, valid_scores = boxes, scores

            keep = torchvision.ops.nms(valid_boxes, valid_scores, iou_threshold)
            if max_num > 0 and keep.size(0) > max_num:
                keep = keep[:max_num]
            
            keep = inds[keep]
            dets = torch.cat((boxes[keep], scores[keep].reshape(-1, 1)), dim=1)
            
            if is_numpy:
                return dets.cpu().numpy(), keep.cpu().numpy()
            return dets, keep

        mmcv.ops.nms.nms = _tv_nms
        mmcv.ops.nms = _tv_nms
        print("[train_det.py] mmcv NMS patched to use torchvision (bypassing missing CUDA extensions)")
    except Exception as e:
        print(f"[train_det.py] NMS patch failed: {e}")

_patch_mmcv_nms()

import mmdet

# Locate mmdet's real train.py (installed alongside the package)
_mmdet_train = os.path.join(
    os.path.dirname(mmdet.__file__), '.mim', 'tools', 'train.py'
)

if not os.path.exists(_mmdet_train):
    raise FileNotFoundError(
        f"Could not find mmdet train script at: {_mmdet_train}\n"
        "Please check your mmdet installation."
    )

# Replace argv[0] so argparse inside train.py sees the right script name
sys.argv[0] = _mmdet_train

runpy.run_path(_mmdet_train, run_name='__main__')
