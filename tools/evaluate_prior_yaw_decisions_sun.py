#!/usr/bin/env python3
"""Evaluate switched SUNRGBD-val yaw decisions against available 3D GT."""

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-info", required=True)
    parser.add_argument("--yaw-info", required=True)
    parser.add_argument("--gt-json", default="datasets/Omni3D/SUNRGBD_val.json")
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--min-prior-aspect", type=float, default=0.0)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def iou_xyxy(box_a, box_b):
    a = np.asarray(box_a, dtype=np.float64)
    b = np.asarray(box_b, dtype=np.float64)
    x1, y1 = np.maximum(a[:2], b[:2])
    x2, y2 = np.minimum(a[2:4], b[2:4])
    intersection = max(x2 - x1, 0.0) * max(y2 - y1, 0.0)
    area_a = max(a[2] - a[0], 0.0) * max(a[3] - a[1], 0.0)
    area_b = max(b[2] - b[0], 0.0) * max(b[3] - b[1], 0.0)
    return float(intersection / max(area_a + area_b - intersection, 1e-12))


def axial_yaw(rotation):
    rotation = np.asarray(rotation, dtype=np.float64)
    if rotation.shape != (3, 3) or np.any(~np.isfinite(rotation)):
        return None
    return float(math.atan2(rotation[2, 0], rotation[0, 0]))


def axial_error_deg(first, second):
    delta = (float(first) - float(second) + math.pi / 2.0) % math.pi
    delta -= math.pi / 2.0
    return abs(math.degrees(delta))


def summarize(records):
    if not records:
        return {"matched": 0}
    base_errors = np.array([item["base_error"] for item in records])
    yaw_errors = np.array([item["yaw_error"] for item in records])
    delta = base_errors - yaw_errors
    return {
        "matched": len(records),
        "base_mean_error_deg": float(np.mean(base_errors)),
        "yaw_mean_error_deg": float(np.mean(yaw_errors)),
        "base_median_error_deg": float(np.median(base_errors)),
        "yaw_median_error_deg": float(np.median(yaw_errors)),
        "mean_error_reduction_deg": float(np.mean(delta)),
        "improved": int(np.sum(delta > 1e-6)),
        "worsened": int(np.sum(delta < -1e-6)),
        "unchanged": int(np.sum(np.abs(delta) <= 1e-6)),
        "base_near_90_rate": float(np.mean(base_errors >= 75.0)),
        "yaw_near_90_rate": float(np.mean(yaw_errors >= 75.0)),
    }


def main():
    args = parse_args()
    base_info = torch.load(args.base_info, map_location="cpu")
    yaw_info = torch.load(args.yaw_info, map_location="cpu")
    with Path(args.gt_json).open("r", encoding="utf-8") as handle:
        gt_data = json.load(handle)

    gt_by_image = defaultdict(list)
    for ann in gt_data.get("annotations", []):
        if not ann.get("valid3D", False):
            continue
        yaw = axial_yaw(ann.get("R_cam"))
        box = ann.get("bbox2D_tight")
        if yaw is None or box is None or min(box) < 0:
            continue
        gt_by_image[ann["image_id"]].append(
            {
                "category": ann.get("category_name"),
                "box": box,
                "yaw": yaw,
            }
        )

    records = []
    unmatched_reasons = Counter()
    switched_total = 0
    for image_id, yaw_image in yaw_info.items():
        if isinstance(image_id, str) and image_id.startswith("_"):
            continue
        if not isinstance(yaw_image, dict) or image_id not in base_info:
            continue
        base_image = base_info[image_id]
        metrics = yaw_image.get("prior_yaw_stats", [])
        for index, metric in enumerate(metrics):
            if not metric.get("switched", False):
                continue
            if float(metric.get("prior_aspect", 0.0)) < float(
                args.min_prior_aspect
            ):
                continue
            switched_total += 1
            if index >= len(yaw_image.get("phrases", [])):
                unmatched_reasons["missing_detection"] += 1
                continue
            category = yaw_image["phrases"][index]
            candidates = [
                gt
                for gt in gt_by_image.get(image_id, [])
                if gt["category"] == category
            ]
            if not candidates:
                unmatched_reasons["no_class_gt"] += 1
                continue
            detection_box = yaw_image["boxes"][index]
            overlaps = [iou_xyxy(detection_box, gt["box"]) for gt in candidates]
            best_index = int(np.argmax(overlaps))
            if overlaps[best_index] < float(args.iou_threshold):
                unmatched_reasons["low_2d_iou"] += 1
                continue
            if (
                index >= len(base_image.get("R_cam", []))
                or index >= len(yaw_image.get("R_cam", []))
            ):
                unmatched_reasons["missing_rotation"] += 1
                continue
            base_yaw = axial_yaw(base_image["R_cam"][index])
            selected_yaw = axial_yaw(yaw_image["R_cam"][index])
            if base_yaw is None or selected_yaw is None:
                unmatched_reasons["invalid_rotation"] += 1
                continue
            gt = candidates[best_index]
            records.append(
                {
                    "category": category,
                    "iou2d": overlaps[best_index],
                    "base_error": axial_error_deg(base_yaw, gt["yaw"]),
                    "yaw_error": axial_error_deg(selected_yaw, gt["yaw"]),
                }
            )

    by_category = {}
    categories = sorted({item["category"] for item in records})
    for category in categories:
        category_records = [
            item for item in records if item["category"] == category
        ]
        if len(category_records) >= 5:
            by_category[category] = summarize(category_records)

    report = {
        "switched_total": switched_total,
        "match_iou_threshold": float(args.iou_threshold),
        "unmatched_reasons": dict(sorted(unmatched_reasons.items())),
        "matched_switched": summarize(records),
        "categories_with_at_least_5_matches": by_category,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
