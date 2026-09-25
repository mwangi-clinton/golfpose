#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Extract ground truth bounding boxes from GolfPose 2D keypoint annotations.

The existing COCO JSON files already have 'bbox' fields computed from motion capture
data. This script:
  1. Reads the existing COCO-format keypoint annotations (person + club)
  2. Re-derives bboxes from raw keypoints with optional padding (more robust)
  3. Exports a clean COCO detection-only JSON ready to train a bbox detector

Usage:
    python tools/extract_gt_bboxes.py \
        --input  golfswing/coco/hscc_golf_2cls_2d_train.json \
        --output golfswing/coco/hscc_golf_det_train.json \
        --padding 0.10

    python tools/extract_gt_bboxes.py \
        --input  golfswing/coco/hscc_golf_2cls_2d_test.json \
        --output golfswing/coco/hscc_golf_det_test.json \
        --padding 0.10
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np


# ─── helpers ──────────────────────────────────────────────────────────────────

def keypoints_to_bbox(kps_flat: list, padding: float = 0.10, img_w: int = 9999, img_h: int = 9999):
    """
    Convert a flat COCO keypoints list [x, y, v, x, y, v, …] to a bbox [x, y, w, h].

    Args:
        kps_flat : Flat list of length 3*N (x, y, visibility per joint).
        padding  : Fractional padding applied to each side of the tight bbox.
        img_w, img_h: Image dimensions used to clamp the padded box.

    Returns:
        [x_min, y_min, w, h]  (COCO format) or None if no visible keypoints.
    """
    arr = np.array(kps_flat, dtype=np.float32).reshape(-1, 3)
    visible = arr[arr[:, 2] > 0]  # v=0 means not annotated
    if len(visible) == 0:
        return None

    x_min, y_min = visible[:, 0].min(), visible[:, 1].min()
    x_max, y_max = visible[:, 0].max(), visible[:, 1].max()

    # Add proportional padding
    bw = x_max - x_min
    bh = y_max - y_min
    x_min = max(0.0, x_min - padding * bw)
    y_min = max(0.0, y_min - padding * bh)
    x_max = min(img_w, x_max + padding * bw)
    y_max = min(img_h, y_max + padding * bh)

    w = x_max - x_min
    h = y_max - y_min
    if w <= 0 or h <= 0:
        return None

    return [float(round(x_min, 2)), float(round(y_min, 2)), float(round(w, 2)), float(round(h, 2))]


# ─── main ─────────────────────────────────────────────────────────────────────

def convert(input_path: str, output_path: str, padding: float, use_raw_bbox: bool):
    """Convert a COCO keypoints JSON into a detection-only JSON."""

    print(f"[INFO] Reading {input_path}")
    with open(input_path) as f:
        src = json.load(f)

    # Build fast lookup: image_id -> image meta
    img_meta = {img["id"]: img for img in src["images"]}

    det_annotations = []
    skipped = 0
    ann_id = 1

    for ann in src["annotations"]:
        img = img_meta[ann["image_id"]]
        img_w, img_h = img["width"], img["height"]

        if use_raw_bbox and ann.get("bbox"):
            # Use the pre-computed bbox as-is
            bbox = ann["bbox"]
        else:
            # Re-derive from raw keypoints with padding
            bbox = keypoints_to_bbox(ann["keypoints"], padding, img_w, img_h)

        if bbox is None:
            skipped += 1
            continue

        x, y, w, h = bbox
        det_ann = {
            "id": ann_id,
            "image_id": ann["image_id"],
            "category_id": ann["category_id"],
            "bbox": [x, y, w, h],
            "area": round(w * h, 2),
            "iscrowd": 0,
            "segmentation": [],
        }
        det_annotations.append(det_ann)
        ann_id += 1

    # Build detection categories (strip keypoint skeleton info)
    det_categories = []
    for cat in src["categories"]:
        det_categories.append({
            "id": cat["id"],
            "name": cat["name"],
            "supercategory": cat.get("supercategory", cat["name"]),
        })

    output_json = {
        "info": src.get("info", {}),
        "licenses": src.get("licenses", []),
        "images": src["images"],
        "annotations": det_annotations,
        "categories": det_categories,
    }

    os.makedirs(Path(output_path).parent, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output_json, f, indent=2)

    print(f"[INFO] Written {len(det_annotations)} annotations ({skipped} skipped) -> {output_path}")
    print(f"[INFO] Categories: {[c['name'] for c in det_categories]}")


# ─── stats ────────────────────────────────────────────────────────────────────

def print_stats(json_path: str):
    """Print quick stats of a detection JSON."""
    with open(json_path) as f:
        d = json.load(f)

    from collections import Counter
    cat_map = {c["id"]: c["name"] for c in d["categories"]}
    counts = Counter(a["category_id"] for a in d["annotations"])
    areas = {cat_map[k]: [] for k in counts}
    for a in d["annotations"]:
        areas[cat_map[a["category_id"]]].append(a["area"])

    print("\n-- Dataset Statistics --")
    print(f"  Images      : {len(d['images'])}")
    print(f"  Annotations : {len(d['annotations'])}")
    for name, area_list in areas.items():
        arr = np.array(area_list)
        print(f"\n  Class: {name}  ({len(arr)} boxes)")
        print(f"    bbox area  min:{arr.min():.0f}  mean:{arr.mean():.0f}  max:{arr.max():.0f}")
    print("------------------------\n")


# ─── CLI ──────────────────────────────────────────────────────────────────────

def build_parser():
    p = argparse.ArgumentParser(description="Extract GT detection bboxes from GolfPose keypoints")
    p.add_argument("--input",  required=True, help="Input COCO keypoints JSON (2-class)")
    p.add_argument("--output", required=True, help="Output COCO detection JSON")
    p.add_argument("--padding", type=float, default=0.10,
                   help="Fractional padding around keypoint bounding box (default: 0.10 = 10%%)")
    p.add_argument("--use-raw-bbox", action="store_true",
                   help="Use the bbox already stored in the annotation instead of re-deriving from keypoints")
    p.add_argument("--stats", action="store_true",
                   help="Print dataset statistics after conversion")
    return p


def main():
    args = build_parser().parse_args()
    convert(args.input, args.output, args.padding, args.use_raw_bbox)
    if args.stats:
        print_stats(args.output)


if __name__ == "__main__":
    main()
