#!/usr/bin/env python3
"""Build pseudo labels with MoCA yaw + UniDepth point-cloud metric fitting.

Route C:

  cached GroundingSAM mask/box
  + cached UniDepth depth
  + MoCA3D-Cube projected-corner/yaw evidence
  -> fit center/dims from mask point cloud in the MoCA-yaw frame
  -> soft LLM class-prior calibration
  -> Omni3D JSON

Unlike the prior/calibrated MoCA branch, this script does not use original
OVM3D 3D boxes as center/dims anchors.  The original pipeline is only reused for
2D proposal caches, depth caches, and SUNRGBD category priors.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cubercnn.generate_label.llm_generated_prior import SUNRGBD as FALLBACK_SUNRGBD_PRIORS  # noqa: E402

try:
    from cubercnn.generate_label.priors import llm_generated_prior as ORIGINAL_LLM_PRIORS  # noqa: E402
except Exception:
    ORIGINAL_LLM_PRIORS = {}

CATEGORY_PRIORS = ORIGINAL_LLM_PRIORS.get("SUNRGBD", FALLBACK_SUNRGBD_PRIORS)

try:
    from cubercnn.generate_label.util import adaptive_erode_mask  # noqa: E402
except Exception:
    adaptive_erode_mask = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("MoCA yaw + UniDepth point cloud pseudo labels")
    parser.add_argument("--source_json", required=True)
    parser.add_argument("--moca_json", required=True)
    parser.add_argument("--pseudo_root", default="pseudo_label")
    parser.add_argument("--dataset", default="SUNRGBD")
    parser.add_argument("--split", choices=["train", "val"], default="train")
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--stats_json", default=None)
    parser.add_argument("--max_images", type=int, default=None)
    parser.add_argument("--seed", type=int, default=2026)

    parser.add_argument("--min_2d_score", type=float, default=0.0)
    parser.add_argument("--min_2d_area_ratio", type=float, default=0.00002)
    parser.add_argument("--max_2d_area_ratio", type=float, default=0.85)
    parser.add_argument("--min_mask_pixels", type=int, default=10)
    parser.add_argument("--min_points", type=int, default=20)
    parser.add_argument("--drop_invalid", action="store_true", default=True)
    parser.add_argument("--keep_invalid", dest="drop_invalid", action="store_false")

    parser.add_argument("--mask_erode_vertical", type=int, default=12)
    parser.add_argument("--mask_erode_vertical_min", type=int, default=2)
    parser.add_argument("--mask_erode_horizontal", type=int, default=6)
    parser.add_argument("--mask_erode_horizontal_min", type=int, default=2)
    parser.add_argument("--use_raw_mask_fallback", action="store_true", default=True)
    parser.add_argument("--no_raw_mask_fallback", dest="use_raw_mask_fallback", action="store_false")

    parser.add_argument("--depth_percentile_low", type=float, default=5.0)
    parser.add_argument("--depth_percentile_high", type=float, default=95.0)
    parser.add_argument("--depth_mad_scale", type=float, default=3.5)
    parser.add_argument("--min_depth_window", type=float, default=0.06)
    parser.add_argument("--extent_percentile_low", type=float, default=2.0)
    parser.add_argument("--extent_percentile_high", type=float, default=98.0)
    parser.add_argument("--thin_extent_percentile_low", type=float, default=0.5)
    parser.add_argument("--thin_extent_percentile_high", type=float, default=99.5)
    parser.add_argument("--max_fit_points", type=int, default=2500)

    parser.add_argument("--moca_match_iou", type=float, default=0.20)
    parser.add_argument("--min_moca_proj_iou", type=float, default=0.20)
    parser.add_argument("--min_moca_corner_iou", type=float, default=0.10)
    parser.add_argument("--fallback_pca_yaw", action="store_true", default=True)
    parser.add_argument("--no_fallback_pca_yaw", dest="fallback_pca_yaw", action="store_false")

    parser.add_argument("--prior_blend", type=float, default=0.45)
    parser.add_argument("--prior_scale_min", type=float, default=0.75)
    parser.add_argument("--prior_scale_max", type=float, default=1.60)
    parser.add_argument("--prior_floor_ratio", type=float, default=0.12)
    parser.add_argument("--prior_ceiling_ratio", type=float, default=8.0)
    parser.add_argument("--min_dimension", type=float, default=0.015)
    parser.add_argument("--max_dimension", type=float, default=8.0)
    parser.add_argument("--min_center_z", type=float, default=0.05)
    parser.add_argument("--max_center_z", type=float, default=20.0)

    parser.add_argument("--min_projection_iou", type=float, default=0.03)
    parser.add_argument("--max_projection_area_ratio", type=float, default=8.0)
    parser.add_argument("--min_point_support", type=float, default=0.05)
    parser.add_argument("--support_margin", type=float, default=0.10)
    parser.add_argument("--bev_nms", action="store_true", default=True)
    parser.add_argument("--no_bev_nms", dest="bev_nms", action="store_false")
    parser.add_argument("--bev_nms_iou", type=float, default=0.80)
    return parser.parse_args()


def load_json(path: str) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def save_json(data: dict, path: str) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        json.dump(data, f)


def torch_load(path: str):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def valid_xyxy(box: Optional[Sequence[float]]) -> bool:
    if box is None or len(box) < 4:
        return False
    x1, y1, x2, y2 = [float(v) for v in box[:4]]
    return math.isfinite(x1 + y1 + x2 + y2) and x2 > x1 and y2 > y1


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


def normalize_bbox(box: Sequence[float], width: int, height: int) -> Optional[List[float]]:
    if box is None or len(box) < 4:
        return None
    x1, y1, x2, y2 = [float(v) for v in box[:4]]
    x1, x2 = sorted((x1, x2))
    y1, y2 = sorted((y1, y2))
    x1 = float(np.clip(x1, 0, max(width - 1, 0)))
    y1 = float(np.clip(y1, 0, max(height - 1, 0)))
    x2 = float(np.clip(x2, 0, max(width - 1, 0)))
    y2 = float(np.clip(y2, 0, max(height - 1, 0)))
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def ann_box_xyxy(ann: dict) -> Optional[List[float]]:
    for key in ("bbox2D_tight", "bbox2D_proj", "bbox2D_trunc"):
        box = ann.get(key)
        if valid_xyxy(box):
            return [float(v) for v in box[:4]]
    return None


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


def category_maps(data: dict) -> Tuple[Dict[str, int], Dict[int, str]]:
    cats = data.get("categories", [])
    name_to_id = {str(c["name"]): int(c["id"]) for c in cats}
    id_to_name = {int(c["id"]): str(c["name"]) for c in cats}
    return name_to_id, id_to_name


def safe_adaptive_erode(mask: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    mask_arr = np.asarray(mask).astype(bool)
    if adaptive_erode_mask is None:
        return mask_arr
    try:
        return adaptive_erode_mask(
            mask_arr.astype(float),
            args.mask_erode_vertical,
            args.mask_erode_vertical_min,
            args.mask_erode_horizontal,
            args.mask_erode_horizontal_min,
        ).astype(bool)
    except Exception:
        out = np.zeros_like(mask_arr, dtype=bool)
        for idx in range(mask_arr.shape[0]):
            try:
                out[idx : idx + 1] = adaptive_erode_mask(
                    mask_arr[idx : idx + 1].astype(float),
                    args.mask_erode_vertical,
                    args.mask_erode_vertical_min,
                    args.mask_erode_horizontal,
                    args.mask_erode_horizontal_min,
                ).astype(bool)
            except Exception:
                out[idx] = mask_arr[idx]
        return out


def mask_to_points(depth: np.ndarray, mask: np.ndarray, K: np.ndarray) -> np.ndarray:
    mask_bool = np.asarray(mask).squeeze() > 0
    valid = mask_bool & np.isfinite(depth) & (depth > 0.05)
    ys, xs = np.nonzero(valid)
    if xs.size == 0:
        return np.zeros((0, 3), dtype=np.float32)
    z = depth[ys, xs].astype(np.float32)
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])
    x = (xs.astype(np.float32) - cx) * z / max(fx, 1e-6)
    y = (ys.astype(np.float32) - cy) * z / max(fy, 1e-6)
    pts = np.stack([x, y, z], axis=1)
    finite = np.all(np.isfinite(pts), axis=1) & (pts[:, 2] > 0)
    return pts[finite].astype(np.float32)


def filter_points_by_depth(points: np.ndarray, class_name: str, args: argparse.Namespace) -> Tuple[Optional[np.ndarray], dict]:
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    metrics = {"raw_points": int(pts.shape[0]), "filtered_points": 0}
    if pts.shape[0] < int(args.min_points):
        metrics["filter_reason"] = "too_few_raw"
        return None, metrics
    z = pts[:, 2].astype(np.float64)
    p_low = args.thin_extent_percentile_low if class_name in THIN_CLASSES else args.depth_percentile_low
    p_high = args.thin_extent_percentile_high if class_name in THIN_CLASSES else args.depth_percentile_high
    lo, hi = np.percentile(z, [p_low, p_high])
    med = float(np.median(z))
    mad = float(np.median(np.abs(z - med)))
    window = max(float(args.min_depth_window), float(args.depth_mad_scale) * 1.4826 * mad)
    lo = max(float(lo), med - window)
    hi = min(float(hi), med + window)
    keep = (z >= lo) & (z <= hi)
    out = pts[keep]
    if out.shape[0] < int(args.min_points):
        out = pts
        metrics["filter_reason"] = "fallback_raw"
    else:
        metrics["filter_reason"] = "depth_percentile_mad"
    metrics.update(
        {
            "filtered_points": int(out.shape[0]),
            "depth_median": med,
            "depth_window": float(window),
            "depth_keep_ratio": float(out.shape[0] / max(pts.shape[0], 1)),
        }
    )
    return out.astype(np.float32), metrics


def yaw_from_R(R: np.ndarray) -> float:
    a = math.atan2(float(R[0, 2]), float(R[0, 0]))
    b = math.atan2(float(-R[2, 0]), float(R[2, 2]))
    return float(math.atan2(math.sin(a) + math.sin(b), math.cos(a) + math.cos(b)))


def yaw_to_R(yaw: float) -> np.ndarray:
    c, s = math.cos(float(yaw)), math.sin(float(yaw))
    return np.asarray([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=np.float32)


def pca_yaw(points: np.ndarray) -> float:
    pts = np.asarray(points, dtype=np.float64)
    xz = pts[:, [0, 2]]
    xz = xz - np.median(xz, axis=0, keepdims=True)
    if xz.shape[0] < 3:
        return 0.0
    cov = np.cov(xz.T)
    vals, vecs = np.linalg.eigh(cov)
    v = vecs[:, int(np.argmax(vals))]
    return float(math.atan2(v[0], v[1]))


def make_corners(center: np.ndarray, dims: np.ndarray, R: np.ndarray) -> np.ndarray:
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
    local = signs * (0.5 * dims.reshape(1, 3))
    return (center.reshape(1, 3) + local @ R.T).astype(np.float32)


def project_corners(corners: np.ndarray, K: np.ndarray, width: int, height: int) -> List[float]:
    pts = np.asarray(corners, dtype=np.float32).reshape(-1, 3)
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


def apply_sorted_prior(dims: np.ndarray, prior: Optional[Sequence[float]], args: argparse.Namespace) -> Tuple[np.ndarray, dict]:
    metrics = {"prior_available": prior is not None, "prior_scale_median": 1.0, "prior_ok": True}
    if prior is None:
        return dims.astype(np.float32), metrics
    prior_arr = np.asarray(prior, dtype=np.float32).reshape(3)
    if np.any(prior_arr <= 0):
        return dims.astype(np.float32), metrics
    order = np.argsort(dims)
    sdims = np.sort(dims.astype(np.float32))
    sprior = np.sort(prior_arr)
    raw_scale = sprior / np.maximum(sdims, 1e-6)
    clipped = np.clip(raw_scale, float(args.prior_scale_min), float(args.prior_scale_max))
    applied = (1.0 - float(args.prior_blend)) + float(args.prior_blend) * clipped
    out = dims.astype(np.float32).copy()
    out[order] = sdims * applied
    ratio = np.sort(out) / np.maximum(sprior, 1e-6)
    ok = bool(np.all(ratio >= float(args.prior_floor_ratio)) and np.all(ratio <= float(args.prior_ceiling_ratio)))
    metrics.update(
        {
            "prior_scale_sorted": [float(v) for v in applied.tolist()],
            "prior_scale_median": float(np.median(applied)),
            "prior_ratio_sorted": [float(v) for v in ratio.tolist()],
            "prior_ok": ok,
        }
    )
    return out.astype(np.float32), metrics


def fit_yaw_frame_box(points: np.ndarray, yaw: float, class_name: str, args: argparse.Namespace) -> Tuple[Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]], dict]:
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    metrics = {"fit_points": int(pts.shape[0]), "yaw": float(yaw)}
    if pts.shape[0] < int(args.min_points):
        metrics["fit_reason"] = "too_few_points"
        return None, metrics
    if pts.shape[0] > int(args.max_fit_points):
        rng = np.random.default_rng(int(args.seed))
        pts = pts[rng.choice(pts.shape[0], size=int(args.max_fit_points), replace=False)]
        metrics["fit_points_sampled"] = int(pts.shape[0])
    R = yaw_to_R(yaw)
    seed = np.median(pts, axis=0).astype(np.float32)
    local = (pts - seed.reshape(1, 3)) @ R
    low_p = args.thin_extent_percentile_low if class_name in THIN_CLASSES else args.extent_percentile_low
    high_p = args.thin_extent_percentile_high if class_name in THIN_CLASSES else args.extent_percentile_high
    lo = np.percentile(local, low_p, axis=0).astype(np.float32)
    hi = np.percentile(local, high_p, axis=0).astype(np.float32)
    dims = np.maximum(hi - lo, float(args.min_dimension)).astype(np.float32)
    local_center = 0.5 * (lo + hi)
    center = (seed + local_center @ R.T).astype(np.float32)
    prior = CATEGORY_PRIORS.get(class_name)
    dims, prior_metrics = apply_sorted_prior(dims, prior, args)
    metrics.update(prior_metrics)
    corners = make_corners(center, dims, R)
    if center[2] < args.min_center_z or center[2] > args.max_center_z:
        metrics["fit_reason"] = "bad_center_z"
        return None, metrics
    if np.any(dims < args.min_dimension) or np.any(dims > args.max_dimension):
        metrics["fit_reason"] = "bad_dims"
        return None, metrics
    metrics.update(
        {
            "fit_reason": "valid",
            "center": [float(v) for v in center.tolist()],
            "dims": [float(v) for v in dims.tolist()],
        }
    )
    return (corners, center, dims, R), metrics


def point_support(points: np.ndarray, center: np.ndarray, dims: np.ndarray, R: np.ndarray, margin: float) -> float:
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    if pts.shape[0] == 0:
        return 0.0
    local = (pts - center.reshape(1, 3)) @ R
    half = 0.5 * dims.reshape(1, 3) + float(margin)
    inside = np.all(np.abs(local) <= half, axis=1)
    return float(np.mean(inside))


def moca_indices(moca_data: dict) -> Tuple[Dict[Tuple[int, int, int], dict], Dict[Tuple[int, int], List[dict]]]:
    by_source: Dict[Tuple[int, int, int], dict] = {}
    by_key: Dict[Tuple[int, int], List[dict]] = defaultdict(list)
    for ann in moca_data.get("annotations", []):
        if not bool(ann.get("valid3D", True)):
            continue
        image_id = int(ann.get("image_id", -1))
        cid = int(ann.get("category_id", -1))
        by_key[(image_id, cid)].append(ann)
        if "source_index" in ann:
            by_source[(image_id, cid, int(ann.get("source_index", -1)))] = ann
    return by_source, by_key


def find_moca_ann(
    image_id: int,
    category_id: int,
    source_index: int,
    bbox: Sequence[float],
    by_source: Dict[Tuple[int, int, int], dict],
    by_key: Dict[Tuple[int, int], List[dict]],
) -> Tuple[Optional[dict], float, str]:
    ann = by_source.get((int(image_id), int(category_id), int(source_index)))
    if ann is not None:
        return ann, bbox_iou(bbox, ann_box_xyxy(ann)), "source_index"
    best, best_iou = None, 0.0
    for cand in by_key.get((int(image_id), int(category_id)), []):
        iou = bbox_iou(bbox, ann_box_xyxy(cand))
        if iou > best_iou:
            best, best_iou = cand, iou
    return best, float(best_iou), "bbox_iou"


def quality_weights(proj_iou: float, moca_proj: float, moca_corner: float, support: float, prior_ok: bool, yaw_source: str) -> Dict[str, float]:
    base = 0.35 * np.clip(proj_iou, 0, 1) + 0.20 * np.clip(moca_proj, 0, 1) + 0.15 * np.clip(moca_corner, 0, 1) + 0.20 * np.clip(support, 0, 1) + 0.10 * float(prior_ok)
    if yaw_source != "moca":
        base *= 0.75
    base = float(np.clip(base, 0.08, 1.0))
    return {
        "pseudo_weight": base,
        "pseudo_weight_joint": base,
        "pseudo_weight_xy": float(np.clip(base + 0.08, 0.10, 1.0)),
        "pseudo_weight_z": float(np.clip(base + 0.06, 0.10, 1.0)),
        "pseudo_weight_dims": float(np.clip(base, 0.05, 1.0)),
        "pseudo_weight_pose": float(np.clip(base if yaw_source == "moca" else base * 0.70, 0.05, 1.0)),
    }


def invalid_vertices() -> List[List[float]]:
    return np.full((8, 3), -1.0, dtype=np.float32).tolist()


def build_invalid(ann_id: int, dataset_id: int, image_id: int, category_name: str, category_id: int, bbox: Sequence[float], score: float, extra: dict) -> dict:
    ann = {
        "id": int(dataset_id * 10000000 + ann_id),
        "image_id": int(image_id),
        "dataset_id": int(dataset_id),
        "category_name": str(category_name),
        "category_id": int(category_id),
        "valid3D": False,
        "bbox2D_tight": [float(v) for v in bbox],
        "bbox2D_trunc": [float(v) for v in bbox],
        "bbox2D_proj": [float(v) for v in bbox],
        "bbox3D_cam": invalid_vertices(),
        "center_cam": [-1.0, -1.0, -1.0],
        "dimensions": [-1.0, -1.0, -1.0],
        "R_cam": np.eye(3, dtype=np.float32).tolist(),
        "behind_camera": False,
        "visibility": 1.0,
        "truncation": 0.0,
        "segmentation_pts": -1,
        "lidar_pts": -1,
        "depth_error": -1,
        "score": float(score),
        "pseudo_weight": 0.25,
    }
    ann.update(extra)
    return ann


def build_valid(
    ann_id: int,
    dataset_id: int,
    image_id: int,
    category_name: str,
    category_id: int,
    bbox: Sequence[float],
    score: float,
    fit: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    bbox_proj: Sequence[float],
    extra: dict,
) -> dict:
    corners, center, dims, R = fit
    ann = {
        "id": int(dataset_id * 10000000 + ann_id),
        "image_id": int(image_id),
        "dataset_id": int(dataset_id),
        "category_name": str(category_name),
        "category_id": int(category_id),
        "valid3D": True,
        "bbox2D_tight": [float(v) for v in bbox],
        "bbox2D_trunc": [float(v) for v in bbox],
        "bbox2D_proj": [float(v) for v in bbox_proj],
        "bbox3D_cam": corners.astype(float).tolist(),
        "center_cam": [float(v) for v in center.tolist()],
        "dimensions": [float(v) for v in dims.tolist()],
        "R_cam": R.astype(float).tolist(),
        "behind_camera": bool(np.any(corners[:, 2] <= 0)),
        "visibility": 1.0,
        "truncation": 0.0,
        "segmentation_pts": -1,
        "lidar_pts": -1,
        "depth_error": -1,
        "score": float(score),
    }
    ann.update(extra)
    return ann


def bev_iou(a: dict, b: dict) -> float:
    ca = as_array(a.get("center_cam", []), (3,))
    cb = as_array(b.get("center_cam", []), (3,))
    da = as_array(a.get("dimensions", []), (3,))
    db = as_array(b.get("dimensions", []), (3,))
    if ca is None or cb is None or da is None or db is None:
        return 0.0
    ax1, ax2 = ca[0] - da[0] * 0.5, ca[0] + da[0] * 0.5
    az1, az2 = ca[2] - da[2] * 0.5, ca[2] + da[2] * 0.5
    bx1, bx2 = cb[0] - db[0] * 0.5, cb[0] + db[0] * 0.5
    bz1, bz2 = cb[2] - db[2] * 0.5, cb[2] + db[2] * 0.5
    ix = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    iz = max(0.0, min(az2, bz2) - max(az1, bz1))
    inter = ix * iz
    area_a = max(0.0, ax2 - ax1) * max(0.0, az2 - az1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, bz2 - bz1)
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0


def apply_bev_nms(annotations: List[dict], threshold: float, stats: dict) -> List[dict]:
    if not annotations:
        return annotations
    grouped: Dict[Tuple[int, int], List[int]] = defaultdict(list)
    for idx, ann in enumerate(annotations):
        if bool(ann.get("valid3D", True)):
            grouped[(int(ann.get("image_id", -1)), int(ann.get("category_id", -1)))].append(idx)
    keep = set(range(len(annotations)))
    suppressed = 0
    for _, idxs in grouped.items():
        idxs = sorted(idxs, key=lambda i: float(annotations[i].get("pseudo_weight", annotations[i].get("score", 0.0))), reverse=True)
        kept_local: List[int] = []
        for idx in idxs:
            if idx not in keep:
                continue
            should_suppress = False
            for kept in kept_local:
                if bev_iou(annotations[idx], annotations[kept]) >= threshold:
                    should_suppress = True
                    break
            if should_suppress:
                keep.discard(idx)
                suppressed += 1
            else:
                kept_local.append(idx)
    stats["bev_nms_suppressed"] = int(suppressed)
    return [ann for idx, ann in enumerate(annotations) if idx in keep]


THIN_CLASSES = {
    "door",
    "window",
    "blinds",
    "curtain",
    "picture",
    "painting",
    "mirror",
    "television",
    "monitor",
    "laptop",
    "floor mat",
    "board",
    "shower curtain",
}


def main() -> None:
    args = parse_args()
    source = load_json(args.source_json)
    moca_data = load_json(args.moca_json)
    name_to_id, _ = category_maps(source)
    dataset_id = int(source.get("info", {}).get("id", 12 if args.split == "train" else 13))
    input_dir = Path(args.pseudo_root) / args.dataset / args.split
    info = torch_load(str(input_dir / "info.pth"))
    moca_by_source, moca_by_key = moca_indices(moca_data)

    images = list(source.get("images", []))
    if args.max_images is not None:
        images = images[: int(args.max_images)]

    annotations: List[dict] = []
    stats = defaultdict(int)
    weights: List[float] = []
    ann_id = 1

    for img in tqdm(images, desc="MoCA yaw point-cloud fit"):
        image_id = int(img["id"])
        if image_id not in info:
            stats["images_without_info"] += 1
            continue
        depth_path = input_dir / "depth" / f"{image_id}.npy"
        mask_path = input_dir / "mask" / f"{image_id}.npy"
        if not depth_path.exists() or not mask_path.exists():
            stats["images_missing_cache"] += 1
            continue
        depth = np.load(depth_path).astype(np.float32)
        raw_mask = np.load(mask_path)
        raw_mask = np.asarray(raw_mask).squeeze(axis=1) if raw_mask.ndim == 4 and raw_mask.shape[1] == 1 else np.asarray(raw_mask)
        eroded_mask = safe_adaptive_erode(raw_mask, args)
        width, height = int(img["width"]), int(img["height"])
        K = np.asarray(img["K"], dtype=np.float32).reshape(3, 3)
        rec = info[image_id]
        phrases = list(rec.get("phrases", []))
        boxes = np.asarray(rec.get("boxes", []), dtype=np.float32)
        scores = np.asarray(rec.get("conf", np.ones(len(phrases), dtype=np.float32)), dtype=np.float32)
        n = min(len(phrases), len(boxes), raw_mask.shape[0], eroded_mask.shape[0])
        stats["objects_seen"] += int(n)
        image_area = max(float(width * height), 1.0)

        for j in range(n):
            category_name = str(phrases[j])
            category_id = int(name_to_id.get(category_name, 0))
            if category_id <= 0:
                stats["unknown_category"] += 1
                continue
            score = float(scores[j]) if j < len(scores) and math.isfinite(float(scores[j])) else 1.0
            if score < args.min_2d_score:
                stats["skipped_low_2d_score"] += 1
                continue
            bbox = normalize_bbox(boxes[j], width, height)
            if bbox is None:
                stats["skipped_bad_bbox"] += 1
                continue
            area_ratio = bbox_area(bbox) / image_area
            if area_ratio < args.min_2d_area_ratio or area_ratio > args.max_2d_area_ratio:
                stats["skipped_2d_area"] += 1
                continue

            mask = eroded_mask[j]
            mask_source = "eroded"
            if int(np.asarray(mask).sum()) < args.min_mask_pixels and args.use_raw_mask_fallback:
                mask = raw_mask[j]
                mask_source = "raw_fallback"
            if int(np.asarray(mask).sum()) < args.min_mask_pixels:
                stats["invalid_small_mask"] += 1
                if args.drop_invalid:
                    continue
                annotations.append(
                    build_invalid(
                        ann_id,
                        dataset_id,
                        image_id,
                        category_name,
                        category_id,
                        bbox,
                        score,
                        {
                            "pseudo_mask_index": int(j),
                            "moca_yaw_fit_source": "invalid_small_mask",
                            "dfu_mask_source": mask_source,
                        },
                    )
                )
                ann_id += 1
                continue

            points_raw = mask_to_points(depth, mask, K)
            points, filter_metrics = filter_points_by_depth(points_raw, category_name, args)
            if points is None:
                stats["invalid_no_points"] += 1
                if args.drop_invalid:
                    continue
                annotations.append(
                    build_invalid(
                        ann_id,
                        dataset_id,
                        image_id,
                        category_name,
                        category_id,
                        bbox,
                        score,
                        {
                            "pseudo_mask_index": int(j),
                            "moca_yaw_fit_source": "invalid_no_points",
                            "dfu_mask_source": mask_source,
                            **filter_metrics,
                        },
                    )
                )
                ann_id += 1
                continue

            moca_ann, match_iou, match_source = find_moca_ann(
                image_id,
                category_id,
                j,
                bbox,
                moca_by_source,
                moca_by_key,
            )
            moca_proj = float(moca_ann.get("moca3d_proj_iou", 0.0) or 0.0) if moca_ann else 0.0
            moca_corner = float(moca_ann.get("moca3d_corner_bbox_iou", 0.0) or 0.0) if moca_ann else 0.0
            yaw_source = "none"
            yaw = None
            if (
                moca_ann is not None
                and match_iou >= args.moca_match_iou
                and moca_proj >= args.min_moca_proj_iou
                and moca_corner >= args.min_moca_corner_iou
            ):
                R_moca = as_array(moca_ann.get("R_cam", []), (3, 3))
                if R_moca is not None:
                    yaw = yaw_from_R(R_moca)
                    yaw_source = "moca"
            if yaw is None and args.fallback_pca_yaw:
                yaw = pca_yaw(points)
                yaw_source = "fallback_pca"
            if yaw is None:
                stats["invalid_no_yaw"] += 1
                if args.drop_invalid:
                    continue
                annotations.append(
                    build_invalid(
                        ann_id,
                        dataset_id,
                        image_id,
                        category_name,
                        category_id,
                        bbox,
                        score,
                        {
                            "pseudo_mask_index": int(j),
                            "moca_yaw_fit_source": "invalid_no_yaw",
                            "moca_match_iou": float(match_iou),
                            "moca_match_source": match_source,
                        },
                    )
                )
                ann_id += 1
                continue

            fit, fit_metrics = fit_yaw_frame_box(points, yaw, category_name, args)
            if fit is None:
                stats["invalid_fit_failed"] += 1
                if args.drop_invalid:
                    continue
                annotations.append(
                    build_invalid(
                        ann_id,
                        dataset_id,
                        image_id,
                        category_name,
                        category_id,
                        bbox,
                        score,
                        {
                            "pseudo_mask_index": int(j),
                            "moca_yaw_fit_source": f"invalid_{yaw_source}",
                            "moca_match_iou": float(match_iou),
                            "moca_match_source": match_source,
                            **filter_metrics,
                            **fit_metrics,
                        },
                    )
                )
                ann_id += 1
                continue

            corners, center, dims, R = fit
            bbox_proj = project_corners(corners, K, width, height)
            proj_iou = bbox_iou(bbox_proj, bbox)
            area_ratio_after = bbox_area(bbox_proj) / max(bbox_area(bbox), 1e-6)
            support = point_support(points, center, dims, R, args.support_margin)
            if (
                proj_iou < args.min_projection_iou
                or area_ratio_after > args.max_projection_area_ratio
                or support < args.min_point_support
            ):
                stats["invalid_quality_gate"] += 1
                if args.drop_invalid:
                    continue

            stats["valid3d"] += 1
            if yaw_source == "moca":
                stats["yaw_source_moca"] += 1
            else:
                stats["yaw_source_fallback_pca"] += 1
            qweights = quality_weights(
                proj_iou,
                moca_proj,
                moca_corner,
                support,
                bool(fit_metrics.get("prior_ok", True)),
                yaw_source,
            )
            extra = {
                "pseudo_label_method": "MoCA_yaw_UniDepth_yaw_frame_extent",
                "pseudo_mask_index": int(j),
                "proposal_2d_score": float(score),
                "proposal_area_ratio": float(area_ratio),
                "dfu_mask_source": mask_source,
                "moca_yaw_fit_source": yaw_source,
                "moca_match_iou": float(match_iou),
                "moca_match_source": match_source,
                "moca3d_source_id": int(moca_ann.get("id", -1)) if moca_ann else -1,
                "moca3d_proj_iou": float(moca_proj),
                "moca3d_corner_bbox_iou": float(moca_corner),
                "moca3d_projected_corners": copy.deepcopy(moca_ann.get("moca3d_projected_corners")) if moca_ann else None,
                "moca3d_corner_depth": copy.deepcopy(moca_ann.get("moca3d_corner_depth")) if moca_ann else None,
                "yaw_frame_projection_iou": float(proj_iou),
                "yaw_frame_projection_area_ratio": float(area_ratio_after),
                "yaw_frame_point_support": float(support),
                **filter_metrics,
                **fit_metrics,
                **qweights,
            }
            ann = build_valid(
                ann_id,
                dataset_id,
                image_id,
                category_name,
                category_id,
                bbox,
                score,
                fit,
                bbox_proj,
                extra,
            )
            weights.append(float(ann.get("pseudo_weight", 1.0)))
            annotations.append(ann)
            ann_id += 1

    if args.bev_nms:
        annotations = apply_bev_nms(annotations, args.bev_nms_iou, stats)

    output = {
        "info": copy.deepcopy(source.get("info", {})),
        "images": images,
        "categories": copy.deepcopy(source.get("categories", [])),
        "annotations": annotations,
    }
    output.setdefault("info", {})
    output["info"]["pseudo_label_method"] = "MoCA_yaw_UniDepth_yaw_frame_extent"
    output["info"]["source_json"] = os.path.abspath(args.source_json)
    output["info"]["moca_json"] = os.path.abspath(args.moca_json)
    stats["images"] = len(images)
    stats["annotations"] = len(annotations)
    stats["effective3d"] = int(sum(1 for a in annotations if bool(a.get("valid3D", True))))
    weights = [float(a.get("pseudo_weight", 1.0)) for a in annotations if bool(a.get("valid3D", True))]
    if weights:
        arr = np.asarray(weights, dtype=np.float32)
        stats["pseudo_weight_mean_x1000"] = int(round(float(np.mean(arr)) * 1000))
        stats["pseudo_weight_p10_x1000"] = int(round(float(np.percentile(arr, 10)) * 1000))
        stats["pseudo_weight_p50_x1000"] = int(round(float(np.percentile(arr, 50)) * 1000))
    save_json(output, args.output_json)
    print(f"Wrote {args.output_json}")
    print(dict(stats))
    if args.stats_json:
        save_json(dict(stats), args.stats_json)
        print(f"Wrote stats {args.stats_json}")


if __name__ == "__main__":
    main()
