#!/usr/bin/env python3
import argparse
import copy
import json
import math
import os
from collections import defaultdict

import numpy as np


def valid_box_xyxy(box):
    if box is None or len(box) != 4:
        return False
    arr = np.asarray(box, dtype=np.float64)
    return bool(np.all(np.isfinite(arr)) and arr[2] > arr[0] and arr[3] > arr[1])


def ann_box_xyxy(ann):
    for key in ("bbox2D_tight", "bbox2D_proj", "bbox2D_trunc"):
        box = ann.get(key)
        if valid_box_xyxy(box):
            return [float(x) for x in box]
    box = ann.get("bbox")
    if box is not None and len(box) == 4:
        x, y, w, h = [float(v) for v in box]
        if w > 0 and h > 0:
            return [x, y, x + w, y + h]
    return None


def bbox_iou(a, b):
    if a is None or b is None:
        return 0.0
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    den = area_a + area_b - inter
    return float(inter / den) if den > 0 else 0.0


def as_vec(value, n):
    try:
        arr = np.asarray(value, dtype=np.float64).reshape(-1)
    except Exception:
        return None
    if arr.size < n or not np.all(np.isfinite(arr[:n])):
        return None
    return arr[:n]


def safe_exp_score(error, scale):
    if not math.isfinite(error):
        return 0.0
    return float(math.exp(-max(0.0, error) / max(scale, 1e-6)))


def consistency(base_ann, ref_ann, match_iou, min_weight):
    base_center = as_vec(base_ann.get("center_cam"), 3)
    ref_center = as_vec(ref_ann.get("center_cam"), 3)
    base_dims = as_vec(base_ann.get("dimensions"), 3)
    ref_dims = as_vec(ref_ann.get("dimensions"), 3)

    if base_center is not None and ref_center is not None:
        z_den = max(abs(float(base_center[2])), 0.25)
        depth_rel = abs(float(base_center[2] - ref_center[2])) / z_den
        center_shift = float(np.linalg.norm(base_center - ref_center) / z_den)
    else:
        depth_rel = float("inf")
        center_shift = float("inf")

    if (
        base_dims is not None
        and ref_dims is not None
        and np.all(base_dims > 1e-4)
        and np.all(ref_dims > 1e-4)
    ):
        dim_log_error = float(np.mean(np.abs(np.log(base_dims / ref_dims))))
    else:
        dim_log_error = float("inf")

    boxer_quality = float(ref_ann.get("boxer_quality", ref_ann.get("score", 0.5)))
    if not math.isfinite(boxer_quality):
        boxer_quality = 0.5
    boxer_quality = float(np.clip(boxer_quality, 0.0, 1.0))

    dfu_support = float(ref_ann.get("boxer_dfu_box_support", 0.5))
    if not math.isfinite(dfu_support) or dfu_support < 0:
        dfu_support = 0.5
    dfu_support = float(np.clip(dfu_support, 0.0, 1.0))

    depth_score = safe_exp_score(depth_rel, 0.35)
    center_score = safe_exp_score(center_shift, 0.45)
    dim_score = safe_exp_score(dim_log_error, 0.55)
    iou_score = float(np.clip(match_iou, 0.0, 1.0))

    raw = (
        0.30 * iou_score
        + 0.20 * depth_score
        + 0.20 * center_score
        + 0.15 * dim_score
        + 0.10 * boxer_quality
        + 0.05 * dfu_support
    )
    weight = float(min_weight + (1.0 - min_weight) * np.clip(raw, 0.0, 1.0))
    return {
        "pseudo_weight": weight,
        "ng_consistency_score": float(raw),
        "ng_match_iou_2d": iou_score,
        "ng_depth_rel_error": float(depth_rel) if math.isfinite(depth_rel) else -1.0,
        "ng_center_shift_norm": float(center_shift) if math.isfinite(center_shift) else -1.0,
        "ng_dim_log_error": float(dim_log_error) if math.isfinite(dim_log_error) else -1.0,
        "ng_boxer_quality": boxer_quality,
        "ng_dfu_box_support": dfu_support,
        "ng_match_found": True,
    }


def build_index(ref_data):
    index = defaultdict(list)
    for ann in ref_data.get("annotations", []):
        if not bool(ann.get("valid3D", True)):
            continue
        box = ann_box_xyxy(ann)
        if box is None:
            continue
        key = (int(ann["image_id"]), int(ann["category_id"]))
        index[key].append((ann, box))
    return index


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_json", required=True)
    parser.add_argument("--reference_json", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--min_iou", type=float, default=0.10)
    parser.add_argument("--min_weight", type=float, default=0.35)
    parser.add_argument("--unmatched_weight", type=float, default=0.55)
    args = parser.parse_args()

    with open(args.base_json, "r") as f:
        base = json.load(f)
    with open(args.reference_json, "r") as f:
        ref = json.load(f)

    out = copy.deepcopy(base)
    index = build_index(ref)
    stats = defaultdict(int)

    for ann in out.get("annotations", []):
        if not bool(ann.get("valid3D", True)):
            ann["pseudo_weight"] = float(args.min_weight)
            ann["ng_match_found"] = False
            stats["invalid3d"] += 1
            continue

        box = ann_box_xyxy(ann)
        key = (int(ann["image_id"]), int(ann["category_id"]))
        best = None
        for ref_ann, ref_box in index.get(key, []):
            iou = bbox_iou(box, ref_box)
            if best is None or iou > best[0]:
                best = (iou, ref_ann)

        if best is None or best[0] < float(args.min_iou):
            ann["pseudo_weight"] = float(args.unmatched_weight)
            ann["ng_consistency_score"] = 0.0
            ann["ng_match_iou_2d"] = float(best[0]) if best is not None else 0.0
            ann["ng_match_found"] = False
            stats["unmatched"] += 1
            continue

        ann.update(consistency(ann, best[1], best[0], float(args.min_weight)))
        stats["matched"] += 1

    weights = [
        float(a.get("pseudo_weight", 1.0))
        for a in out.get("annotations", [])
        if bool(a.get("valid3D", True))
    ]
    out.setdefault("info", {})
    if isinstance(out["info"], list):
        out["info"] = out["info"][0] if out["info"] else {}
    out["info"]["ng_consistency_reference"] = os.path.abspath(args.reference_json)
    out["info"]["ng_consistency_base"] = os.path.abspath(args.base_json)
    out["info"]["ng_min_weight"] = float(args.min_weight)
    out["info"]["ng_unmatched_weight"] = float(args.unmatched_weight)
    out["info"]["ng_mean_pseudo_weight"] = float(np.mean(weights)) if weights else 1.0

    os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump(out, f)

    print(f"Wrote {args.output_json}")
    print(dict(stats))
    if weights:
        print(
            "weights:",
            {
                "count": len(weights),
                "mean": float(np.mean(weights)),
                "min": float(np.min(weights)),
                "p10": float(np.percentile(weights, 10)),
                "p50": float(np.percentile(weights, 50)),
                "p90": float(np.percentile(weights, 90)),
                "max": float(np.max(weights)),
            },
        )


if __name__ == "__main__":
    main()
