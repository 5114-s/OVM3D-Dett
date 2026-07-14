#!/usr/bin/env python3
"""Evaluate LiftHead-style Omni3D predictions on a category split.

This is a lightweight diagnostic for base/novel experiments.  It greedily
matches predicted annotations to GT annotations by image/category/2D IoU and
reports geometric errors for the selected categories.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import defaultdict
from typing import Dict, List, Mapping, Set, Tuple

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from tools.lifthead_common import box_iou_xyxy, finite_float, xyxy_from_ann, yaw_from_R  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Omni3D box quality for a category split.")
    parser.add_argument("--pred_json", required=True)
    parser.add_argument("--gt_json", required=True)
    parser.add_argument("--output_json", default=None)
    parser.add_argument("--categories", default="", help="Comma-separated category names/ids to evaluate.")
    parser.add_argument("--exclude_categories", default="", help="Comma-separated category names/ids to skip.")
    parser.add_argument("--min_match_iou", type=float, default=0.30)
    return parser.parse_args()


def load_json(path: str) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def normalize_token(value: object) -> str:
    return str(value).strip().lower().replace("_", " ")


def parse_filter(raw: str) -> Set[str]:
    if not raw:
        return set()
    return {normalize_token(tok) for tok in raw.split(",") if tok.strip()}


def names_by_id(*category_lists: List[Mapping]) -> Dict[int, str]:
    out: Dict[int, str] = {}
    for categories in category_lists:
        for cat in categories or []:
            if "id" not in cat:
                continue
            out[int(cat["id"])] = str(cat.get("name", cat.get("name_readable", cat["id"])))
    return out


def category_allowed(ann: Mapping, cat_names: Mapping[int, str], include: Set[str], exclude: Set[str]) -> bool:
    cat_id = int(ann.get("category_id", -1))
    tokens = {normalize_token(cat_id), normalize_token(cat_names.get(cat_id, cat_id))}
    if include and tokens.isdisjoint(include):
        return False
    if exclude and not tokens.isdisjoint(exclude):
        return False
    return True


def valid_3d(ann: Mapping) -> bool:
    if not bool(ann.get("valid3D", False)):
        return False
    center = np.asarray(ann.get("center_cam", []), dtype=np.float32).reshape(-1)
    dims = np.asarray(ann.get("dimensions", []), dtype=np.float32).reshape(-1)
    return (
        center.shape[0] == 3
        and dims.shape[0] == 3
        and np.all(np.isfinite(center))
        and np.all(np.isfinite(dims))
        and float(center[2]) > 0.05
        and np.all(dims > 0)
    )


def group_annotations(
    anns: List[Mapping],
    cat_names: Mapping[int, str],
    include: Set[str],
    exclude: Set[str],
) -> Dict[Tuple[int, int], List[Mapping]]:
    grouped: Dict[Tuple[int, int], List[Mapping]] = defaultdict(list)
    for ann in anns:
        if not valid_3d(ann):
            continue
        if not category_allowed(ann, cat_names, include, exclude):
            continue
        box = xyxy_from_ann(ann)
        if np.any(box < 0):
            continue
        grouped[(int(ann["image_id"]), int(ann["category_id"]))].append(ann)
    return grouped


def greedy_match(preds: List[Mapping], gts: List[Mapping], min_iou: float) -> List[Tuple[Mapping, Mapping, float]]:
    pairs = []
    for pi, pred in enumerate(preds):
        pred_box = xyxy_from_ann(pred)
        for gi, gt in enumerate(gts):
            iou = box_iou_xyxy(pred_box, xyxy_from_ann(gt))
            if iou >= min_iou:
                pairs.append((iou, pi, gi))
    pairs.sort(reverse=True)
    used_p = set()
    used_g = set()
    matched = []
    for iou, pi, gi in pairs:
        if pi in used_p or gi in used_g:
            continue
        used_p.add(pi)
        used_g.add(gi)
        matched.append((preds[pi], gts[gi], float(iou)))
    return matched


def angle_abs_diff(a: float, b: float) -> float:
    return abs(float((a - b + math.pi) % (2.0 * math.pi) - math.pi))


def summarize(values: List[float]) -> Dict[str, float]:
    arr = np.asarray(values, dtype=np.float32)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"mean": float("nan"), "median": float("nan"), "p90": float("nan")}
    return {
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "p90": float(np.percentile(arr, 90.0)),
    }


def main() -> None:
    args = parse_args()
    pred = load_json(args.pred_json)
    gt = load_json(args.gt_json)
    include = parse_filter(args.categories)
    exclude = parse_filter(args.exclude_categories)
    cat_names = names_by_id(pred.get("categories", []), gt.get("categories", []))

    pred_grouped = group_annotations(pred.get("annotations", []), cat_names, include, exclude)
    gt_grouped = group_annotations(gt.get("annotations", []), cat_names, include, exclude)

    matched_all = []
    for key, gts in gt_grouped.items():
        preds = pred_grouped.get(key, [])
        matched_all.extend(greedy_match(preds, gts, args.min_match_iou))

    center_l2 = []
    xy_l2 = []
    z_abs = []
    log_dim_mae = []
    yaw_deg = []
    match_ious = []
    proj_ious = []
    per_category = defaultdict(lambda: {"gt": 0, "pred": 0, "matched": 0})

    for (image_id, category_id), anns in gt_grouped.items():
        per_category[cat_names.get(category_id, str(category_id))]["gt"] += len(anns)
    for (image_id, category_id), anns in pred_grouped.items():
        per_category[cat_names.get(category_id, str(category_id))]["pred"] += len(anns)

    for pred_ann, gt_ann, iou in matched_all:
        category_name = cat_names.get(int(gt_ann["category_id"]), str(gt_ann["category_id"]))
        per_category[category_name]["matched"] += 1
        pc = np.asarray(pred_ann["center_cam"], dtype=np.float32)
        gc = np.asarray(gt_ann["center_cam"], dtype=np.float32)
        pd = np.maximum(np.asarray(pred_ann["dimensions"], dtype=np.float32), 1e-6)
        gd = np.maximum(np.asarray(gt_ann["dimensions"], dtype=np.float32), 1e-6)
        center_l2.append(float(np.linalg.norm(pc - gc)))
        xy_l2.append(float(np.linalg.norm(pc[:2] - gc[:2])))
        z_abs.append(abs(float(pc[2] - gc[2])))
        log_dim_mae.append(float(np.mean(np.abs(np.log(pd) - np.log(gd)))))
        yaw_deg.append(math.degrees(angle_abs_diff(yaw_from_R(pred_ann.get("R_cam", np.eye(3))), yaw_from_R(gt_ann.get("R_cam", np.eye(3))))))
        match_ious.append(iou)
        proj_box = pred_ann.get("bbox2D_proj")
        if proj_box is not None:
            proj_ious.append(box_iou_xyxy(proj_box, xyxy_from_ann(gt_ann)))

    gt_count = sum(len(v) for v in gt_grouped.values())
    pred_count = sum(len(v) for v in pred_grouped.values())
    stats = {
        "pred_json": os.path.abspath(args.pred_json),
        "gt_json": os.path.abspath(args.gt_json),
        "categories": sorted(include),
        "exclude_categories": sorted(exclude),
        "min_match_iou": float(args.min_match_iou),
        "gt_valid3d": int(gt_count),
        "pred_valid3d": int(pred_count),
        "matched": int(len(matched_all)),
        "gt_match_recall": float(len(matched_all) / max(gt_count, 1)),
        "pred_match_precision": float(len(matched_all) / max(pred_count, 1)),
        "match_2d_iou": summarize(match_ious),
        "proj_iou_to_gt2d": summarize(proj_ious),
        "center_l2_m": summarize(center_l2),
        "xy_l2_m": summarize(xy_l2),
        "z_abs_m": summarize(z_abs),
        "log_dim_mae": summarize(log_dim_mae),
        "yaw_abs_deg": summarize(yaw_deg),
        "per_category": dict(sorted(per_category.items())),
    }

    text = json.dumps(stats, indent=2, ensure_ascii=False)
    print(text)
    if args.output_json:
        os.makedirs(os.path.dirname(os.path.abspath(args.output_json)), exist_ok=True)
        with open(args.output_json, "w") as f:
            f.write(text + "\n")


if __name__ == "__main__":
    main()
