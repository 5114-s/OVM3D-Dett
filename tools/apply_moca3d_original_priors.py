#!/usr/bin/env python3
"""Constrain MoCA3D-Cube pseudo labels with original OVM3D-Det geometry priors.

The clean MoCA route is useful as a learned 2D-box -> 3D proposal, but direct
replacement drifts on SUNRGBD physical size and depth. This post-processor keeps
the original OVM3D pseudo labels as the anchor and lets MoCA replace geometry
only when it is consistent with:

  * the same-image/same-class 2D proposal,
  * original pseudo-label depth,
  * original/category size priors,
  * MoCA's own projection quality.

Output stays in Omni3D JSON format and can be trained with the normal SUN config.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Apply original geometry priors to MoCA3D-Cube labels")
    parser.add_argument("--moca_json", required=True)
    parser.add_argument("--reference_json", required=True, help="Usually datasets/Omni3D_pl-1/SUNRGBD_*.json")
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--stats_json", default=None)
    parser.add_argument("--match_iou", type=float, default=0.35)
    parser.add_argument("--use_moca_min_proj_iou", type=float, default=0.55)
    parser.add_argument("--use_moca_min_corner_iou", type=float, default=0.25)
    parser.add_argument("--min_proj_area_ratio", type=float, default=0.35)
    parser.add_argument("--max_proj_area_ratio", type=float, default=3.50)
    parser.add_argument("--min_dim_ratio_to_ref", type=float, default=0.45)
    parser.add_argument("--max_dim_ratio_to_ref", type=float, default=1.80)
    parser.add_argument("--min_dim_ratio_to_class", type=float, default=0.40)
    parser.add_argument("--max_dim_ratio_to_class", type=float, default=2.20)
    parser.add_argument("--min_z_ratio_to_ref", type=float, default=0.55)
    parser.add_argument("--max_z_ratio_to_ref", type=float, default=1.80)
    parser.add_argument("--max_center_xy_shift_ratio", type=float, default=0.60)
    parser.add_argument(
        "--moca_scale_to_ref",
        action="store_true",
        help="Isotropically scale accepted MoCA dimensions toward the matched reference dimensions.",
    )
    parser.add_argument("--min_scale", type=float, default=0.75)
    parser.add_argument("--max_scale", type=float, default=1.50)
    parser.add_argument(
        "--center_mode",
        choices=["reference", "moca", "blend"],
        default="reference",
        help="Which center to use when MoCA geometry is accepted.",
    )
    parser.add_argument("--center_blend", type=float, default=0.50)
    parser.add_argument(
        "--keep_moca_unmatched",
        action="store_true",
        help="Append unmatched high-quality MoCA labels. Disabled by default for stability.",
    )
    return parser.parse_args()


def load_json(path: str) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def save_json(data: dict, path: str) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        json.dump(data, f)


def valid_xyxy(box: Optional[Sequence[float]]) -> bool:
    if box is None or len(box) < 4:
        return False
    x1, y1, x2, y2 = [float(v) for v in box[:4]]
    return math.isfinite(x1 + y1 + x2 + y2) and x2 > x1 and y2 > y1 and x1 >= 0 and y1 >= 0


def ann_box_xyxy(ann: dict) -> Optional[List[float]]:
    for key in ("bbox2D_tight", "bbox2D_proj", "bbox2D_trunc"):
        box = ann.get(key)
        if valid_xyxy(box):
            return [float(v) for v in box[:4]]
    bbox = ann.get("bbox")
    if bbox is not None and len(bbox) >= 4:
        x, y, w, h = [float(v) for v in bbox[:4]]
        if w > 0 and h > 0:
            return [x, y, x + w, y + h]
    return None


def bbox_area(box: Optional[Sequence[float]]) -> float:
    if not valid_xyxy(box):
        return 0.0
    x1, y1, x2, y2 = [float(v) for v in box[:4]]
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def bbox_iou(a: Optional[Sequence[float]], b: Optional[Sequence[float]]) -> float:
    if not valid_xyxy(a) or not valid_xyxy(b):
        return 0.0
    ax1, ay1, ax2, ay2 = [float(v) for v in a[:4]]
    bx1, by1, bx2, by2 = [float(v) for v in b[:4]]
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = bbox_area(a) + bbox_area(b) - inter
    return float(inter / union) if union > 0 else 0.0


def as_float_array(value: Sequence[float], shape: Optional[Tuple[int, ...]] = None) -> Optional[np.ndarray]:
    try:
        arr = np.asarray(value, dtype=np.float32)
    except Exception:
        return None
    if not np.isfinite(arr).all():
        return None
    if shape is not None and arr.shape != shape:
        return None
    return arr


def valid_3d_ann(ann: dict) -> bool:
    if ann.get("valid3D", True) is False:
        return False
    center = as_float_array(ann.get("center_cam", []), (3,))
    dims = as_float_array(ann.get("dimensions", []), (3,))
    rot = as_float_array(ann.get("R_cam", []), (3, 3))
    if center is None or dims is None or rot is None:
        return False
    return center[2] > 0 and np.all(dims > 0)


def sorted_dims(ann: dict) -> Optional[np.ndarray]:
    dims = as_float_array(ann.get("dimensions", []), (3,))
    if dims is None or np.any(dims <= 0):
        return None
    return np.sort(dims.astype(np.float32))


def build_class_priors(reference_anns: Iterable[dict]) -> Dict[int, dict]:
    by_cat: Dict[int, List[np.ndarray]] = defaultdict(list)
    z_by_cat: Dict[int, List[float]] = defaultdict(list)
    for ann in reference_anns:
        if not valid_3d_ann(ann):
            continue
        dims = sorted_dims(ann)
        center = as_float_array(ann.get("center_cam", []), (3,))
        if dims is None or center is None:
            continue
        cid = int(ann.get("category_id", -1))
        by_cat[cid].append(dims)
        z_by_cat[cid].append(float(center[2]))

    priors = {}
    for cid, values in by_cat.items():
        arr = np.stack(values, axis=0)
        priors[cid] = {
            "dims_p10": np.percentile(arr, 10, axis=0).astype(np.float32),
            "dims_p50": np.percentile(arr, 50, axis=0).astype(np.float32),
            "dims_p90": np.percentile(arr, 90, axis=0).astype(np.float32),
            "z_p50": float(np.percentile(z_by_cat[cid], 50)) if z_by_cat[cid] else None,
            "n": int(arr.shape[0]),
        }
    return priors


def make_corners(center: Sequence[float], dims: Sequence[float], R: Sequence[Sequence[float]]) -> List[List[float]]:
    center_arr = np.asarray(center, dtype=np.float32)
    dims_arr = np.asarray(dims, dtype=np.float32)
    R_arr = np.asarray(R, dtype=np.float32)
    signs = np.array(
        [
            [-1, -1, -1],
            [1, -1, -1],
            [1, 1, -1],
            [-1, 1, -1],
            [-1, -1, 1],
            [1, -1, 1],
            [1, 1, 1],
            [-1, 1, 1],
        ],
        dtype=np.float32,
    )
    local = signs * (dims_arr.reshape(1, 3) * 0.5)
    corners = center_arr.reshape(1, 3) + local @ R_arr.T
    return corners.astype(float).tolist()


def ratios_in_range(values: np.ndarray, low: float, high: float) -> bool:
    return bool(np.all(values >= low) and np.all(values <= high))


def choose_best_moca(
    ref_ann: dict,
    candidates: Sequence[dict],
    used_ids: set,
) -> Tuple[Optional[dict], float]:
    ref_box = ann_box_xyxy(ref_ann)
    best_ann, best_iou = None, 0.0
    for ann in candidates:
        mid = int(ann.get("id", -1))
        if mid in used_ids:
            continue
        iou = bbox_iou(ref_box, ann_box_xyxy(ann))
        if iou > best_iou:
            best_ann, best_iou = ann, iou
    return best_ann, best_iou


def passes_prior_gates(ref_ann: dict, moca_ann: dict, match_iou: float, priors: Dict[int, dict], args: argparse.Namespace) -> Tuple[bool, dict]:
    ref_dims = sorted_dims(ref_ann)
    moca_dims = sorted_dims(moca_ann)
    ref_center = as_float_array(ref_ann.get("center_cam", []), (3,))
    moca_center = as_float_array(moca_ann.get("center_cam", []), (3,))
    if ref_dims is None or moca_dims is None or ref_center is None or moca_center is None:
        return False, {"reason": "invalid_geometry"}

    proj_iou = float(moca_ann.get("moca3d_proj_iou", 0.0) or 0.0)
    corner_iou = float(moca_ann.get("moca3d_corner_bbox_iou", 0.0) or 0.0)
    area_ratio = float(moca_ann.get("moca3d_proj_area_ratio", 0.0) or 0.0)
    dim_ratio_ref = moca_dims / np.maximum(ref_dims, 1e-6)
    z_ratio = float(moca_center[2] / max(float(ref_center[2]), 1e-6))
    xy_shift = float(np.linalg.norm(moca_center[:2] - ref_center[:2]) / max(float(ref_center[2]), 1e-6))

    cid = int(ref_ann.get("category_id", -1))
    class_prior = priors.get(cid)
    class_ratio_ok = True
    if class_prior is not None:
        p50 = np.maximum(class_prior["dims_p50"], 1e-6)
        dim_ratio_class = moca_dims / p50
        class_ratio_ok = ratios_in_range(dim_ratio_class, args.min_dim_ratio_to_class, args.max_dim_ratio_to_class)
    else:
        dim_ratio_class = np.ones(3, dtype=np.float32)

    checks = {
        "match_iou": match_iou,
        "proj_iou": proj_iou,
        "corner_iou": corner_iou,
        "proj_area_ratio": area_ratio,
        "dim_ratio_ref": dim_ratio_ref.astype(float).tolist(),
        "dim_ratio_class": dim_ratio_class.astype(float).tolist(),
        "z_ratio": z_ratio,
        "xy_shift_norm": xy_shift,
    }

    ok = (
        match_iou >= args.match_iou
        and proj_iou >= args.use_moca_min_proj_iou
        and corner_iou >= args.use_moca_min_corner_iou
        and args.min_proj_area_ratio <= area_ratio <= args.max_proj_area_ratio
        and ratios_in_range(dim_ratio_ref, args.min_dim_ratio_to_ref, args.max_dim_ratio_to_ref)
        and class_ratio_ok
        and args.min_z_ratio_to_ref <= z_ratio <= args.max_z_ratio_to_ref
        and xy_shift <= args.max_center_xy_shift_ratio
    )
    if not ok:
        failed = []
        if match_iou < args.match_iou:
            failed.append("match")
        if proj_iou < args.use_moca_min_proj_iou:
            failed.append("proj")
        if corner_iou < args.use_moca_min_corner_iou:
            failed.append("corner")
        if not (args.min_proj_area_ratio <= area_ratio <= args.max_proj_area_ratio):
            failed.append("area")
        if not ratios_in_range(dim_ratio_ref, args.min_dim_ratio_to_ref, args.max_dim_ratio_to_ref):
            failed.append("dim_ref")
        if not class_ratio_ok:
            failed.append("dim_class")
        if not (args.min_z_ratio_to_ref <= z_ratio <= args.max_z_ratio_to_ref):
            failed.append("z")
        if xy_shift > args.max_center_xy_shift_ratio:
            failed.append("xy")
        checks["reason"] = ",".join(failed) if failed else "unknown"
    return ok, checks


def constrained_moca_geometry(ref_ann: dict, moca_ann: dict, args: argparse.Namespace) -> Tuple[List[float], List[float], List[List[float]]]:
    ref_center = as_float_array(ref_ann["center_cam"], (3,))
    ref_dims = as_float_array(ref_ann["dimensions"], (3,))
    moca_center = as_float_array(moca_ann["center_cam"], (3,))
    moca_dims = as_float_array(moca_ann["dimensions"], (3,))
    moca_R = as_float_array(moca_ann["R_cam"], (3, 3))
    assert ref_center is not None and ref_dims is not None
    assert moca_center is not None and moca_dims is not None and moca_R is not None

    dims = moca_dims.copy()
    if args.moca_scale_to_ref:
        scale = float(np.median(np.sort(ref_dims) / np.maximum(np.sort(moca_dims), 1e-6)))
        scale = max(args.min_scale, min(args.max_scale, scale))
        dims = dims * scale

    if args.center_mode == "reference":
        center = ref_center.copy()
    elif args.center_mode == "blend":
        alpha = max(0.0, min(1.0, float(args.center_blend)))
        center = (1.0 - alpha) * ref_center + alpha * moca_center
    else:
        center = moca_center.copy()

    return center.astype(float).tolist(), dims.astype(float).tolist(), moca_R.astype(float).tolist()


def quality_weight(info: dict) -> float:
    proj = float(info.get("proj_iou", 0.0))
    corner = float(info.get("corner_iou", 0.0))
    area = float(info.get("proj_area_ratio", 1.0))
    area_score = math.exp(-abs(math.log(max(area, 1e-6))))
    score = 0.45 * proj + 0.25 * corner + 0.30 * area_score
    return float(max(0.05, min(1.0, score)))


def process_split(moca_data: dict, ref_data: dict, args: argparse.Namespace) -> Tuple[dict, dict]:
    out = copy.deepcopy(ref_data)
    ref_anns = [a for a in ref_data.get("annotations", []) if valid_3d_ann(a)]
    priors = build_class_priors(ref_anns)

    moca_by_key: Dict[Tuple[int, int], List[dict]] = defaultdict(list)
    for ann in moca_data.get("annotations", []):
        if valid_3d_ann(ann):
            moca_by_key[(int(ann.get("image_id", -1)), int(ann.get("category_id", -1)))].append(ann)

    new_anns = []
    used_moca_ids = set()
    stats = defaultdict(int)
    numeric = defaultdict(list)

    for ref_ann in ref_data.get("annotations", []):
        out_ann = copy.deepcopy(ref_ann)
        if not valid_3d_ann(ref_ann):
            out_ann["moca_prior_action"] = "reference_invalid"
            new_anns.append(out_ann)
            stats["reference_invalid"] += 1
            continue

        key = (int(ref_ann.get("image_id", -1)), int(ref_ann.get("category_id", -1)))
        moca_ann, match_iou = choose_best_moca(ref_ann, moca_by_key.get(key, []), used_moca_ids)
        if moca_ann is None:
            out_ann["moca_prior_action"] = "fallback_no_moca"
            out_ann["pseudo_weight"] = 0.75
            new_anns.append(out_ann)
            stats["fallback_no_moca"] += 1
            continue

        used_moca_ids.add(int(moca_ann.get("id", -1)))
        ok, info = passes_prior_gates(ref_ann, moca_ann, match_iou, priors, args)
        for k in ("match_iou", "proj_iou", "corner_iou", "proj_area_ratio", "z_ratio", "xy_shift_norm"):
            if k in info:
                numeric[k].append(float(info[k]))

        out_ann["moca_prior_match_iou"] = float(match_iou)
        out_ann["moca_prior_proj_iou"] = float(info.get("proj_iou", 0.0))
        out_ann["moca_prior_corner_iou"] = float(info.get("corner_iou", 0.0))
        out_ann["moca_prior_area_ratio"] = float(info.get("proj_area_ratio", 0.0))
        out_ann["moca_prior_z_ratio"] = float(info.get("z_ratio", 0.0))
        out_ann["moca_prior_source_id"] = int(moca_ann.get("id", -1))

        if ok:
            center, dims, R = constrained_moca_geometry(ref_ann, moca_ann, args)
            out_ann["center_cam"] = center
            out_ann["dimensions"] = dims
            out_ann["R_cam"] = R
            out_ann["bbox3D_cam"] = make_corners(center, dims, R)
            out_ann["moca_prior_action"] = "use_moca_constrained"
            out_ann["pseudo_weight"] = quality_weight(info)
            out_ann["pseudo_weight_xy"] = max(0.20, min(1.0, out_ann["pseudo_weight"] + 0.10))
            out_ann["pseudo_weight_z"] = max(0.20, min(1.0, out_ann["pseudo_weight"] + 0.05))
            out_ann["pseudo_weight_dims"] = out_ann["pseudo_weight"]
            out_ann["pseudo_weight_pose"] = max(0.10, min(1.0, out_ann["pseudo_weight"] - 0.10))
            out_ann["pseudo_weight_joint"] = out_ann["pseudo_weight"]
            stats["use_moca_constrained"] += 1
        else:
            out_ann["moca_prior_action"] = "fallback_original"
            out_ann["moca_prior_reject_reason"] = str(info.get("reason", "unknown"))
            out_ann["pseudo_weight"] = 0.85
            out_ann["pseudo_weight_xy"] = 0.95
            out_ann["pseudo_weight_z"] = 0.90
            out_ann["pseudo_weight_dims"] = 0.80
            out_ann["pseudo_weight_pose"] = 0.70
            out_ann["pseudo_weight_joint"] = 0.85
            stats["fallback_original"] += 1
            for reason in str(info.get("reason", "")).split(","):
                if reason:
                    stats[f"reject_{reason}"] += 1

        new_anns.append(out_ann)

    if args.keep_moca_unmatched:
        next_id = max([int(a.get("id", 0)) for a in new_anns] or [0]) + 1
        used_ref_keys = {(int(a.get("image_id", -1)), int(a.get("category_id", -1))) for a in ref_data.get("annotations", [])}
        for ann in moca_data.get("annotations", []):
            mid = int(ann.get("id", -1))
            if mid in used_moca_ids or not valid_3d_ann(ann):
                continue
            if float(ann.get("moca3d_proj_iou", 0.0) or 0.0) < args.use_moca_min_proj_iou:
                continue
            out_ann = copy.deepcopy(ann)
            out_ann["id"] = next_id
            next_id += 1
            out_ann["moca_prior_action"] = "append_unmatched_moca"
            out_ann["pseudo_weight"] = 0.35
            new_anns.append(out_ann)
            stats["append_unmatched_moca"] += 1
            _ = used_ref_keys

    out["annotations"] = new_anns
    stats["images"] = len(out.get("images", []))
    stats["annotations"] = len(new_anns)
    stats["moca_annotations"] = len(moca_data.get("annotations", []))
    stats["reference_annotations"] = len(ref_data.get("annotations", []))
    stats["class_priors"] = len(priors)
    for key, vals in numeric.items():
        if vals:
            arr = np.asarray(vals, dtype=np.float32)
            stats[f"{key}_mean_x1000"] = int(round(float(np.mean(arr)) * 1000))
            stats[f"{key}_p50_x1000"] = int(round(float(np.percentile(arr, 50)) * 1000))
            stats[f"{key}_p10_x1000"] = int(round(float(np.percentile(arr, 10)) * 1000))
    return out, dict(stats)


def main() -> None:
    args = parse_args()
    moca = load_json(args.moca_json)
    ref = load_json(args.reference_json)
    out, stats = process_split(moca, ref, args)
    save_json(out, args.output_json)
    print(f"Wrote {args.output_json}")
    print(stats)
    if args.stats_json:
        save_json(stats, args.stats_json)
        print(f"Wrote stats {args.stats_json}")


if __name__ == "__main__":
    main()
