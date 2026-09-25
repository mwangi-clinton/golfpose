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
