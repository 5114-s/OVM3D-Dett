#!/usr/bin/env python3
"""Calibrate MoCA3D-Cube pseudo labels with UniDepth anchors and class priors.

This branch is intentionally different from pure MoCA replacement:

  original OVM3D pseudo labels are kept as the stable recall/center anchor
  MoCA3D-Cube provides image-plane corners and yaw/rotation evidence
  cached UniDepth provides a metric depth anchor
  original class-level size statistics softly correct MoCA's size bias

The output is a normal Omni3D JSON with factorized pseudo weights, so it can be
trained by the existing OVM3D-Det SUN config plus depth-consistency loss.
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
    parser = argparse.ArgumentParser("Build MoCA3D calibrated Omni3D pseudo labels")
    parser.add_argument("--moca_json", required=True)
    parser.add_argument("--reference_json", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--stats_json", default=None)
    parser.add_argument("--depth_root", default="pseudo_label/SUNRGBD")
    parser.add_argument("--dataset", default="SUNRGBD")
    parser.add_argument("--split", choices=["train", "val"], default="train")
    parser.add_argument("--match_iou", type=float, default=0.30)
    parser.add_argument("--min_moca_proj_iou", type=float, default=0.35)
    parser.add_argument("--min_moca_corner_iou", type=float, default=0.20)
    parser.add_argument("--min_after_proj_iou", type=float, default=0.20)
    parser.add_argument("--min_proj_area_ratio", type=float, default=0.20)
    parser.add_argument("--max_proj_area_ratio", type=float, default=5.00)
    parser.add_argument("--depth_crop_ratio", type=float, default=0.65)
    parser.add_argument("--min_depth_pixels", type=int, default=12)
    parser.add_argument("--depth_scale_min", type=float, default=0.70)
    parser.add_argument("--depth_scale_max", type=float, default=1.40)
    parser.add_argument("--prior_scale_min", type=float, default=0.75)
    parser.add_argument("--prior_scale_max", type=float, default=1.60)
    parser.add_argument("--prior_blend", type=float, default=0.70)
    parser.add_argument(
        "--anisotropic_prior_calibration",
        action="store_true",
        default=True,
        help=(
            "Calibrate the sorted small/middle/large dimensions separately with "
            "class priors, then map them back to MoCA's dimension axes."
        ),
    )
    parser.add_argument(
        "--scalar_prior_calibration",
        dest="anisotropic_prior_calibration",
        action="store_false",
        help="Use one isotropic prior scale instead of sorted-axis prior calibration.",
    )
    parser.add_argument("--center_depth_blend", type=float, default=0.45)
    parser.add_argument("--center_xy_moca_blend", type=float, default=0.35)
    parser.add_argument("--max_center_xy_shift_ratio", type=float, default=0.45)
    parser.add_argument("--min_dim_ratio_to_class_p10", type=float, default=0.55)
    parser.add_argument("--max_dim_ratio_to_class_p90", type=float, default=1.75)
    parser.add_argument("--fallback_reference_weight", type=float, default=0.82)
    parser.add_argument("--append_unmatched_moca", action="store_true")
    parser.add_argument("--unmatched_moca_weight", type=float, default=0.25)
    parser.add_argument("--upright_yaw_only", action="store_true", default=True)
    parser.add_argument("--keep_full_moca_rotation", dest="upright_yaw_only", action="store_false")
    return parser.parse_args()


def load_json(path: str) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def save_json(data: dict, path: str) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        json.dump(data, f)


def as_array(value, shape: Optional[Tuple[int, ...]] = None) -> Optional[np.ndarray]:
    try:
        arr = np.asarray(value, dtype=np.float32)
    except Exception:
        return None
    if shape is not None and arr.shape != shape:
        return None
    if not np.isfinite(arr).all():
        return None
    return arr


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


def valid_3d_ann(ann: dict) -> bool:
    if ann.get("valid3D", True) is False:
        return False
    center = as_array(ann.get("center_cam", []), (3,))
    dims = as_array(ann.get("dimensions", []), (3,))
    rot = as_array(ann.get("R_cam", []), (3, 3))
    return center is not None and dims is not None and rot is not None and center[2] > 0 and np.all(dims > 0)


def sorted_dims(ann: dict) -> Optional[np.ndarray]:
    dims = as_array(ann.get("dimensions", []), (3,))
    if dims is None or np.any(dims <= 0):
        return None
    return np.sort(dims.astype(np.float32))


def build_class_priors(reference_anns: Iterable[dict]) -> Dict[int, dict]:
    by_cat: Dict[int, List[np.ndarray]] = defaultdict(list)
    for ann in reference_anns:
        if not valid_3d_ann(ann):
            continue
        dims = sorted_dims(ann)
        if dims is not None:
            by_cat[int(ann.get("category_id", -1))].append(dims)
    priors = {}
    for cid, values in by_cat.items():
        arr = np.stack(values, axis=0)
        priors[cid] = {
            "dims_p10": np.percentile(arr, 10, axis=0).astype(np.float32),
            "dims_p50": np.percentile(arr, 50, axis=0).astype(np.float32),
            "dims_p90": np.percentile(arr, 90, axis=0).astype(np.float32),
            "n": int(arr.shape[0]),
        }
    return priors


def image_size_map(data: dict) -> Dict[int, Tuple[int, int, np.ndarray]]:
    out = {}
    for img in data.get("images", []):
        K = as_array(img.get("K", []), (3, 3))
        if K is None:
            K = np.eye(3, dtype=np.float32)
        out[int(img["id"])] = (int(img["width"]), int(img["height"]), K)
    return out


def load_depth(depth_root: str, split: str, image_id: int) -> Optional[np.ndarray]:
    path = Path(depth_root) / split / "depth" / f"{image_id}.npy"
    if not path.exists():
        return None
    try:
        arr = np.load(path)
    except Exception:
        return None
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim != 2:
        return None
    return arr


def robust_depth_anchor(
    depth: Optional[np.ndarray],
    box: Optional[Sequence[float]],
    image_w: int,
    image_h: int,
    crop_ratio: float,
    min_pixels: int,
) -> Tuple[Optional[float], int]:
    if depth is None or not valid_xyxy(box):
        return None, 0
    h, w = depth.shape[:2]
    x1, y1, x2, y2 = [float(v) for v in box[:4]]
    cx, cy = 0.5 * (x1 + x2), 0.5 * (y1 + y2)
    bw, bh = max(1.0, x2 - x1), max(1.0, y2 - y1)
    ratio = max(0.10, min(1.0, float(crop_ratio)))
    x1 = cx - 0.5 * bw * ratio
    x2 = cx + 0.5 * bw * ratio
    y1 = cy - 0.5 * bh * ratio
    y2 = cy + 0.5 * bh * ratio
    sx = float(w) / max(float(image_w), 1.0)
    sy = float(h) / max(float(image_h), 1.0)
    ix1 = int(max(0, min(w - 1, math.floor(x1 * sx))))
    ix2 = int(max(0, min(w, math.ceil(x2 * sx))))
    iy1 = int(max(0, min(h - 1, math.floor(y1 * sy))))
    iy2 = int(max(0, min(h, math.ceil(y2 * sy))))
    if ix2 <= ix1 or iy2 <= iy1:
        return None, 0
    patch = depth[iy1:iy2, ix1:ix2]
    vals = patch[np.isfinite(patch) & (patch > 0.05) & (patch < 80.0)]
    if vals.size < min_pixels:
        return None, int(vals.size)
    lo, hi = np.percentile(vals, [15, 85])
    vals = vals[(vals >= lo) & (vals <= hi)]
    if vals.size < min_pixels:
        return None, int(vals.size)
    return float(np.median(vals)), int(vals.size)


def yaw_from_R(R: np.ndarray) -> float:
    # For an upright Y-axis rotation: [[cos,0,sin],[0,1,0],[-sin,0,cos]].
    a = math.atan2(float(R[0, 2]), float(R[0, 0]))
    b = math.atan2(float(-R[2, 0]), float(R[2, 2]))
    return float(math.atan2(math.sin(a) + math.sin(b), math.cos(a) + math.cos(b)))


def upright_yaw_R(R: np.ndarray) -> np.ndarray:
    yaw = yaw_from_R(R)
    c, s = math.cos(yaw), math.sin(yaw)
    return np.asarray([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=np.float32)


def make_corners(center: Sequence[float], dims: Sequence[float], R: Sequence[Sequence[float]]) -> List[List[float]]:
    c = np.asarray(center, dtype=np.float32).reshape(1, 3)
    d = np.asarray(dims, dtype=np.float32).reshape(1, 3)
    R = np.asarray(R, dtype=np.float32).reshape(3, 3)
    signs = np.asarray(
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
    local = signs * (0.5 * d)
    corners = c + local @ R.T
    return corners.astype(float).tolist()


def project_points(corners: Sequence[Sequence[float]], K: np.ndarray, width: int, height: int) -> List[float]:
    pts = np.asarray(corners, dtype=np.float32)
    z = np.maximum(pts[:, 2], 1e-6)
    u = K[0, 0] * pts[:, 0] / z + K[0, 2]
    v = K[1, 1] * pts[:, 1] / z + K[1, 2]
    x1 = float(np.clip(np.min(u), 0, max(width - 1, 1)))
    x2 = float(np.clip(np.max(u), 0, max(width - 1, 1)))
    y1 = float(np.clip(np.min(v), 0, max(height - 1, 1)))
    y2 = float(np.clip(np.max(v), 0, max(height - 1, 1)))
    if x2 <= x1 or y2 <= y1:
        return [0.0, 0.0, 1.0, 1.0]
    return [x1, y1, x2, y2]


def choose_best_moca(ref_ann: dict, candidates: Sequence[dict], used_ids: set) -> Tuple[Optional[dict], float]:
    ref_box = ann_box_xyxy(ref_ann)
    best_ann, best_iou = None, 0.0
    for ann in candidates:
        mid = int(ann.get("id", -1))
        if mid in used_ids or not valid_3d_ann(ann):
            continue
        iou = bbox_iou(ref_box, ann_box_xyxy(ann))
        if iou > best_iou:
            best_ann, best_iou = ann, iou
    return best_ann, best_iou


def clip(value: float, low: float, high: float) -> float:
    return float(max(low, min(high, value)))


def prior_calibrate_dims(
    dims: np.ndarray,
    prior: Optional[dict],
    args: argparse.Namespace,
) -> Tuple[np.ndarray, float, float, bool]:
    if prior is None:
        return dims.copy(), 1.0, 1.0, True
    sdims = np.sort(dims.astype(np.float32))
    p50 = np.maximum(prior["dims_p50"], 1e-6)
    raw_scales = p50 / np.maximum(sdims, 1e-6)
    if args.anisotropic_prior_calibration:
        clipped = np.clip(raw_scales, float(args.prior_scale_min), float(args.prior_scale_max))
        applied_sorted_scales = (1.0 - float(args.prior_blend)) + float(args.prior_blend) * clipped
        order = np.argsort(dims.astype(np.float32))
        calibrated = dims.astype(np.float32).copy()
        calibrated[order] = sdims * applied_sorted_scales
        raw = float(np.median(raw_scales))
        applied = float(np.median(applied_sorted_scales))
        final_sorted = np.sort(calibrated)
    else:
        raw = float(np.median(raw_scales))
        applied = clip(raw, args.prior_scale_min, args.prior_scale_max)
        applied = (1.0 - float(args.prior_blend)) + float(args.prior_blend) * applied
        calibrated = dims * applied
        final_sorted = np.sort(calibrated)
    low = prior["dims_p10"] * float(args.min_dim_ratio_to_class_p10)
    high = prior["dims_p90"] * float(args.max_dim_ratio_to_class_p90)
    ok = bool(np.all(final_sorted >= low) and np.all(final_sorted <= high))
    return calibrated.astype(np.float32), applied, raw, ok


def calibrate_geometry(
    ref_ann: dict,
    moca_ann: dict,
    depth_anchor: Optional[float],
    prior: Optional[dict],
    width: int,
    height: int,
    K: np.ndarray,
    args: argparse.Namespace,
) -> Tuple[Optional[dict], dict]:
    ref_center = as_array(ref_ann.get("center_cam", []), (3,))
    moca_center = as_array(moca_ann.get("center_cam", []), (3,))
    moca_dims = as_array(moca_ann.get("dimensions", []), (3,))
    moca_R = as_array(moca_ann.get("R_cam", []), (3, 3))
    if ref_center is None or moca_center is None or moca_dims is None or moca_R is None:
        return None, {"reason": "invalid_geometry"}

    z_ref = float(ref_center[2])
    z_moca = float(moca_center[2])
    if depth_anchor is not None and math.isfinite(depth_anchor) and depth_anchor > 0:
        z_target = (1.0 - float(args.center_depth_blend)) * z_ref + float(args.center_depth_blend) * float(depth_anchor)
        depth_source = "unidepth_ref_blend"
    else:
        z_target = z_ref
        depth_source = "reference"
    depth_scale_raw = z_target / max(z_moca, 1e-6)
    depth_scale = clip(depth_scale_raw, args.depth_scale_min, args.depth_scale_max)

    dims_depth = moca_dims * depth_scale
    dims, prior_scale, prior_scale_raw, prior_ok = prior_calibrate_dims(dims_depth, prior, args)
    if np.any(~np.isfinite(dims)) or np.any(dims <= 0):
        return None, {"reason": "bad_dims"}

    scaled_moca_center = moca_center.copy()
    scaled_moca_center = scaled_moca_center * (z_target / max(z_moca, 1e-6))
    scaled_moca_center[2] = z_target
    center = ref_center.copy()
    center[2] = z_target
    alpha_xy = clip(args.center_xy_moca_blend, 0.0, 1.0)
    center[:2] = (1.0 - alpha_xy) * ref_center[:2] + alpha_xy * scaled_moca_center[:2]
    xy_shift = float(np.linalg.norm(center[:2] - ref_center[:2]) / max(z_ref, 1e-6))

    R = upright_yaw_R(moca_R) if args.upright_yaw_only else moca_R
    corners = make_corners(center, dims, R)
    proj_box = project_points(corners, K, width, height)
    input_box = ann_box_xyxy(moca_ann)
    proj_iou_after = bbox_iou(proj_box, input_box)
    area_ratio_after = bbox_area(proj_box) / max(bbox_area(input_box), 1e-6)

    ok = (
        prior_ok
        and xy_shift <= args.max_center_xy_shift_ratio
        and proj_iou_after >= args.min_after_proj_iou
        and args.min_proj_area_ratio <= area_ratio_after <= args.max_proj_area_ratio
    )
    info = {
        "depth_anchor": float(depth_anchor) if depth_anchor is not None else None,
        "depth_source": depth_source,
        "depth_scale_raw": float(depth_scale_raw),
        "depth_scale": float(depth_scale),
        "prior_scale_raw": float(prior_scale_raw),
        "prior_scale": float(prior_scale),
        "prior_ok": bool(prior_ok),
        "xy_shift": float(xy_shift),
        "proj_iou_after": float(proj_iou_after),
        "area_ratio_after": float(area_ratio_after),
        "yaw": float(yaw_from_R(moca_R)),
    }
    if not ok:
        reasons = []
        if not prior_ok:
            reasons.append("prior")
        if xy_shift > args.max_center_xy_shift_ratio:
            reasons.append("xy")
        if proj_iou_after < args.min_after_proj_iou:
            reasons.append("proj_after")
        if not (args.min_proj_area_ratio <= area_ratio_after <= args.max_proj_area_ratio):
            reasons.append("area_after")
        info["reason"] = ",".join(reasons) or "unknown"
        return None, info

    return {
        "center": center.astype(float).tolist(),
        "dims": dims.astype(float).tolist(),
        "R": R.astype(float).tolist(),
        "corners": corners,
        "bbox2D_proj": proj_box,
    }, info


def quality_weight(match_iou: float, moca_ann: dict, cal_info: dict) -> Dict[str, float]:
    proj = float(moca_ann.get("moca3d_proj_iou", 0.0) or 0.0)
    corner = float(moca_ann.get("moca3d_corner_bbox_iou", 0.0) or 0.0)
    after = float(cal_info.get("proj_iou_after", 0.0) or 0.0)
    depth_scale = float(cal_info.get("depth_scale_raw", 1.0) or 1.0)
    prior_scale = float(cal_info.get("prior_scale_raw", 1.0) or 1.0)
    depth_score = math.exp(-abs(math.log(max(depth_scale, 1e-6))))
    prior_score = math.exp(-abs(math.log(max(prior_scale, 1e-6))))
    base = (
        0.22 * clip(match_iou, 0.0, 1.0)
        + 0.22 * clip(proj, 0.0, 1.0)
        + 0.16 * clip(corner, 0.0, 1.0)
        + 0.18 * clip(after, 0.0, 1.0)
        + 0.12 * depth_score
        + 0.10 * prior_score
    )
    base = clip(base, 0.08, 1.0)
    return {
        "joint": base,
        "xy": clip(base + 0.08, 0.10, 1.0),
        "z": clip(base + 0.05, 0.10, 1.0),
        "dims": clip(base - 0.03, 0.05, 1.0),
        "pose": clip(0.45 * corner + 0.35 * after + 0.20 * base, 0.05, 1.0),
    }


def set_factorized_weights(ann: dict, weights: Dict[str, float]) -> None:
    ann["pseudo_weight"] = float(weights["joint"])
    ann["pseudo_weight_joint"] = float(weights["joint"])
    ann["pseudo_weight_xy"] = float(weights["xy"])
    ann["pseudo_weight_z"] = float(weights["z"])
    ann["pseudo_weight_dims"] = float(weights["dims"])
    ann["pseudo_weight_pose"] = float(weights["pose"])


def process_split(moca_data: dict, ref_data: dict, args: argparse.Namespace) -> Tuple[dict, dict]:
    out = copy.deepcopy(ref_data)
    images = image_size_map(ref_data)
    ref_valid = [a for a in ref_data.get("annotations", []) if valid_3d_ann(a)]
    priors = build_class_priors(ref_valid)
    moca_by_key: Dict[Tuple[int, int], List[dict]] = defaultdict(list)
    for ann in moca_data.get("annotations", []):
        if valid_3d_ann(ann):
            moca_by_key[(int(ann.get("image_id", -1)), int(ann.get("category_id", -1)))].append(ann)

    stats = defaultdict(int)
    numeric = defaultdict(list)
    used_moca_ids = set()
    depth_cache: Dict[int, Optional[np.ndarray]] = {}
    new_anns = []

    for ref_ann in ref_data.get("annotations", []):
        out_ann = copy.deepcopy(ref_ann)
        if not valid_3d_ann(ref_ann):
            out_ann["moca_calibrated_action"] = "reference_invalid"
            new_anns.append(out_ann)
            stats["reference_invalid"] += 1
            continue

        image_id = int(ref_ann.get("image_id", -1))
        cid = int(ref_ann.get("category_id", -1))
        width, height, K = images.get(image_id, (0, 0, np.eye(3, dtype=np.float32)))
        moca_ann, match_iou = choose_best_moca(ref_ann, moca_by_key.get((image_id, cid), []), used_moca_ids)
        if moca_ann is None or match_iou < args.match_iou:
            out_ann["moca_calibrated_action"] = "fallback_no_match"
            set_factorized_weights(
                out_ann,
                {
                    "joint": args.fallback_reference_weight,
                    "xy": min(1.0, args.fallback_reference_weight + 0.10),
                    "z": min(1.0, args.fallback_reference_weight + 0.08),
                    "dims": max(0.05, args.fallback_reference_weight - 0.06),
                    "pose": max(0.05, args.fallback_reference_weight - 0.12),
                },
            )
            new_anns.append(out_ann)
            stats["fallback_no_match"] += 1
            continue

        used_moca_ids.add(int(moca_ann.get("id", -1)))
        proj_iou = float(moca_ann.get("moca3d_proj_iou", 0.0) or 0.0)
        corner_iou = float(moca_ann.get("moca3d_corner_bbox_iou", 0.0) or 0.0)
        if proj_iou < args.min_moca_proj_iou or corner_iou < args.min_moca_corner_iou:
            out_ann["moca_calibrated_action"] = "fallback_low_moca_quality"
            out_ann["moca_calibrated_match_iou"] = float(match_iou)
            out_ann["moca_calibrated_proj_iou"] = float(proj_iou)
            out_ann["moca_calibrated_corner_iou"] = float(corner_iou)
            set_factorized_weights(
                out_ann,
                {
                    "joint": args.fallback_reference_weight,
                    "xy": min(1.0, args.fallback_reference_weight + 0.10),
                    "z": min(1.0, args.fallback_reference_weight + 0.08),
                    "dims": max(0.05, args.fallback_reference_weight - 0.06),
                    "pose": max(0.05, args.fallback_reference_weight - 0.12),
                },
            )
            new_anns.append(out_ann)
            stats["fallback_low_moca_quality"] += 1
            continue

        if image_id not in depth_cache:
            depth_cache[image_id] = load_depth(args.depth_root, args.split, image_id)
        depth_anchor, depth_pixels = robust_depth_anchor(
            depth_cache[image_id],
            ann_box_xyxy(moca_ann) or ann_box_xyxy(ref_ann),
            width,
            height,
            args.depth_crop_ratio,
            args.min_depth_pixels,
        )

        calibrated, cal_info = calibrate_geometry(
            ref_ann=ref_ann,
            moca_ann=moca_ann,
            depth_anchor=depth_anchor,
            prior=priors.get(cid),
            width=width,
            height=height,
            K=K,
            args=args,
        )
        for key in (
            "depth_anchor",
            "depth_scale_raw",
            "depth_scale",
            "prior_scale_raw",
            "prior_scale",
            "xy_shift",
            "proj_iou_after",
            "area_ratio_after",
        ):
            value = cal_info.get(key)
            if value is not None and math.isfinite(float(value)):
                numeric[key].append(float(value))

        out_ann["moca_calibrated_match_iou"] = float(match_iou)
        out_ann["moca_calibrated_proj_iou"] = float(proj_iou)
        out_ann["moca_calibrated_corner_iou"] = float(corner_iou)
        out_ann["moca_calibrated_depth_anchor"] = cal_info.get("depth_anchor")
        out_ann["moca_calibrated_depth_pixels"] = int(depth_pixels)
        out_ann["moca_calibrated_depth_source"] = cal_info.get("depth_source", "none")
        out_ann["moca_calibrated_depth_scale"] = float(cal_info.get("depth_scale", 1.0))
        out_ann["moca_calibrated_prior_scale"] = float(cal_info.get("prior_scale", 1.0))
        out_ann["moca_calibrated_yaw"] = float(cal_info.get("yaw", 0.0))
        out_ann["moca3d_raw_center_cam"] = copy.deepcopy(moca_ann.get("center_cam"))
        out_ann["moca3d_raw_dimensions"] = copy.deepcopy(moca_ann.get("dimensions"))
        out_ann["moca3d_projected_corners"] = copy.deepcopy(moca_ann.get("moca3d_projected_corners"))
        out_ann["moca3d_corner_depth"] = copy.deepcopy(moca_ann.get("moca3d_corner_depth"))
        out_ann["moca3d_proj_iou"] = float(proj_iou)
        out_ann["moca3d_corner_bbox_iou"] = float(corner_iou)
        out_ann["moca3d_proj_area_ratio"] = float(moca_ann.get("moca3d_proj_area_ratio", 0.0) or 0.0)

        if calibrated is None:
            out_ann["moca_calibrated_action"] = "fallback_calibration_reject"
            out_ann["moca_calibrated_reject_reason"] = str(cal_info.get("reason", "unknown"))
            set_factorized_weights(
                out_ann,
                {
                    "joint": args.fallback_reference_weight,
                    "xy": min(1.0, args.fallback_reference_weight + 0.10),
                    "z": min(1.0, args.fallback_reference_weight + 0.08),
                    "dims": max(0.05, args.fallback_reference_weight - 0.06),
                    "pose": max(0.05, args.fallback_reference_weight - 0.12),
                },
            )
            for reason in str(cal_info.get("reason", "")).split(","):
                if reason:
                    stats[f"reject_{reason}"] += 1
            stats["fallback_calibration_reject"] += 1
        else:
            out_ann["center_cam"] = calibrated["center"]
            out_ann["dimensions"] = calibrated["dims"]
            out_ann["R_cam"] = calibrated["R"]
            out_ann["bbox3D_cam"] = calibrated["corners"]
            out_ann["bbox2D_proj"] = calibrated["bbox2D_proj"]
            out_ann["moca_calibrated_action"] = "use_moca_yaw_depth_prior"
            out_ann["pseudo_label_method"] = "MoCA3D_Cube_UniDepth_Prior_Calibrated"
            set_factorized_weights(out_ann, quality_weight(match_iou, moca_ann, cal_info))
            stats["use_moca_yaw_depth_prior"] += 1
        new_anns.append(out_ann)

    if args.append_unmatched_moca:
        next_id = max([int(a.get("id", 0)) for a in new_anns] or [0]) + 1
        for ann in moca_data.get("annotations", []):
            mid = int(ann.get("id", -1))
            if mid in used_moca_ids or not valid_3d_ann(ann):
                continue
            proj_iou = float(ann.get("moca3d_proj_iou", 0.0) or 0.0)
            corner_iou = float(ann.get("moca3d_corner_bbox_iou", 0.0) or 0.0)
            if proj_iou < max(args.min_moca_proj_iou, 0.55) or corner_iou < max(args.min_moca_corner_iou, 0.45):
                continue
            out_ann = copy.deepcopy(ann)
            out_ann["id"] = next_id
            next_id += 1
            out_ann["moca_calibrated_action"] = "append_high_quality_unmatched_moca"
            out_ann["pseudo_label_method"] = "MoCA3D_Cube_Unmatched_LowWeight"
            set_factorized_weights(
                out_ann,
                {
                    "joint": args.unmatched_moca_weight,
                    "xy": args.unmatched_moca_weight,
                    "z": args.unmatched_moca_weight,
                    "dims": max(0.05, args.unmatched_moca_weight * 0.75),
                    "pose": max(0.05, args.unmatched_moca_weight * 0.75),
                },
            )
            new_anns.append(out_ann)
            stats["append_high_quality_unmatched_moca"] += 1

    out["annotations"] = new_anns
    out.setdefault("info", {})
    out["info"]["pseudo_label_method"] = "MoCA3D_Cube_UniDepth_Prior_Calibrated"
    out["info"]["moca_source_json"] = str(Path(args.moca_json).resolve())
    out["info"]["reference_source_json"] = str(Path(args.reference_json).resolve())
    out["info"]["depth_root"] = str(Path(args.depth_root).resolve())
    stats["images"] = len(out.get("images", []))
    stats["annotations"] = len(new_anns)
    stats["reference_annotations"] = len(ref_data.get("annotations", []))
    stats["moca_annotations"] = len(moca_data.get("annotations", []))
    stats["class_priors"] = len(priors)
    weights = [float(a.get("pseudo_weight", 1.0)) for a in new_anns if a.get("valid3D", True)]
    if weights:
        stats["pseudo_weight_mean_x1000"] = int(round(float(np.mean(weights)) * 1000))
        stats["pseudo_weight_p10_x1000"] = int(round(float(np.percentile(weights, 10)) * 1000))
        stats["pseudo_weight_p50_x1000"] = int(round(float(np.percentile(weights, 50)) * 1000))
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
