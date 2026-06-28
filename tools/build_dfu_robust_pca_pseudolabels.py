#!/usr/bin/env python3
import argparse
import copy
import json
import math
import os
import sys
from collections import defaultdict
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import cv2
from tqdm import tqdm

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from cubercnn.generate_label.llm_generated_prior import SUNRGBD
from cubercnn.generate_label.raytrace import calc_dis_ray_tracing, calc_inside_ratio
from cubercnn.generate_label.util import (
    adaptive_erode_mask,
    convert_box_vertices,
    erode_mask,
    extract_ground,
    generate_possible_bboxs,
    point_to_plane_distance,
    project_image_to_cam,
    rotate_y,
)

try:
    from build_ng_consistency_pseudolabels import (
        ann_box_xyxy,
        bbox_iou,
        build_index,
        consistency,
    )
except Exception:
    ann_box_xyxy = None
    bbox_iou = None
    build_index = None
    consistency = None


THIN_OR_PLANAR_CLASSES = {
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


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Build Omni3D pseudo labels from cached GroundingSAM masks and "
            "UniDepth using DFU-style filtering plus robust PCA/OBB fitting. "
            "Optional Boxer labels are used only as consistency weights."
        )
    )
    parser.add_argument("--source_json", required=True)
    parser.add_argument("--pseudo_root", default="pseudo_label")
    parser.add_argument("--dataset", default="SUNRGBD")
    parser.add_argument("--split", choices=["train", "val"], default="train")
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--reference_json", default=None)
    parser.add_argument("--max_images", type=int, default=None)
    parser.add_argument("--seed", type=int, default=2026)

    parser.add_argument("--mask_erode_vertical", type=int, default=12)
    parser.add_argument("--mask_erode_vertical_min", type=int, default=2)
    parser.add_argument("--mask_erode_horizontal", type=int, default=6)
    parser.add_argument("--mask_erode_horizontal_min", type=int, default=2)
    parser.add_argument("--ground_erode_vertical", type=int, default=4)
    parser.add_argument("--ground_erode_horizontal", type=int, default=4)

    parser.add_argument("--min_mask_pixels", type=int, default=10)
    parser.add_argument("--min_points", type=int, default=12)
    parser.add_argument("--max_pca_points", type=int, default=800)
    parser.add_argument("--min_dimension", type=float, default=0.015)

    parser.add_argument("--dfu_depth_percentile_low", type=float, default=5.0)
    parser.add_argument("--dfu_depth_percentile_high", type=float, default=95.0)
    parser.add_argument("--dfu_thin_depth_percentile_low", type=float, default=1.0)
    parser.add_argument("--dfu_thin_depth_percentile_high", type=float, default=99.0)
    parser.add_argument("--dfu_mad_scale", type=float, default=3.5)
    parser.add_argument("--dfu_thin_mad_scale", type=float, default=5.0)
    parser.add_argument("--dfu_min_depth_window", type=float, default=0.06)
    parser.add_argument("--dfu_axis_percentile_low", type=float, default=1.0)
    parser.add_argument("--dfu_axis_percentile_high", type=float, default=99.0)
    parser.add_argument("--dfu_use_axis_filter", action="store_true", default=True)
    parser.add_argument("--no_dfu_axis_filter", dest="dfu_use_axis_filter", action="store_false")
    parser.add_argument("--dfu_use_radius_filter", action="store_true")
    parser.add_argument("--dfu_radius", type=float, default=0.45)
    parser.add_argument("--dfu_radius_nb_points", type=int, default=8)
    parser.add_argument("--dfu_fallback_raw", action="store_true", default=True)
    parser.add_argument("--no_dfu_fallback_raw", dest="dfu_fallback_raw", action="store_false")

    parser.add_argument("--use_depth_edge_filter", action="store_true")
    parser.add_argument("--depth_edge_rel_threshold", type=float, default=0.025)
    parser.add_argument("--depth_edge_dilate", type=int, default=1)
    parser.add_argument("--depth_edge_min_keep_ratio", type=float, default=0.25)

    parser.add_argument("--use_frustum_dbscan", action="store_true")
    parser.add_argument("--dbscan_eps_ratio", type=float, default=0.018)
    parser.add_argument("--dbscan_eps_min", type=float, default=0.035)
    parser.add_argument("--dbscan_min_samples", type=int, default=8)
    parser.add_argument("--dbscan_max_fit_points", type=int, default=2500)
    parser.add_argument("--dbscan_min_cluster_points", type=int, default=12)
    parser.add_argument("--dbscan_min_keep_ratio", type=float, default=0.08)

    parser.add_argument("--use_normal_ground_fusion", action="store_true")
    parser.add_argument("--normal_stride", type=int, default=4)
    parser.add_argument("--normal_min_count", type=int, default=80)
    parser.add_argument("--normal_min_vertical_dot", type=float, default=0.65)
    parser.add_argument("--normal_fusion_weight", type=float, default=0.35)

    parser.add_argument("--extent_percentile_low", type=float, default=2.0)
    parser.add_argument("--extent_percentile_high", type=float, default=98.0)
    parser.add_argument("--thin_extent_percentile_low", type=float, default=0.5)
    parser.add_argument("--thin_extent_percentile_high", type=float, default=99.5)
    parser.add_argument("--height_prior_ratio", type=float, default=0.5)
    parser.add_argument("--direct_prior_ratio", type=float, default=0.5)
    parser.add_argument("--ground_snap_distance", type=float, default=0.5)
    parser.add_argument("--prior_floor_ratio", type=float, default=0.12)
    parser.add_argument("--prior_ceiling_ratio", type=float, default=8.0)

    parser.add_argument("--use_surface_box_optimization", action="store_true")
    parser.add_argument("--use_source_geometry_anchor", action="store_true")
    parser.add_argument(
        "--surface_center_mode",
        choices=["locked", "conservative", "free"],
        default="locked",
    )
    parser.add_argument("--surface_depth_percentile", type=float, default=35.0)
    parser.add_argument("--surface_max_shift_ratio", type=float, default=0.20)
    parser.add_argument("--surface_yaw_delta_deg", type=float, default=10.0)
    parser.add_argument("--surface_scale_delta", type=float, default=0.10)
    parser.add_argument("--surface_height_scale_delta", type=float, default=0.05)
    parser.add_argument("--surface_projection_weight", type=float, default=1.5)
    parser.add_argument("--surface_silhouette_weight", type=float, default=2.0)
    parser.add_argument("--surface_depth_weight", type=float, default=2.0)
    parser.add_argument("--surface_support_weight", type=float, default=2.0)
    parser.add_argument("--surface_prior_weight", type=float, default=0.15)
    parser.add_argument("--surface_min_point_support", type=float, default=0.25)
    parser.add_argument("--surface_min_support_ratio", type=float, default=0.75)
    parser.add_argument("--use_latent_box_closure", action="store_true")
    parser.add_argument("--latent_topk", type=int, default=8)
    parser.add_argument("--latent_temperature", type=float, default=0.25)
    parser.add_argument("--latent_min_attribute_weight", type=float, default=0.15)
    parser.add_argument("--latent_max_attribute_weight", type=float, default=1.0)
    parser.add_argument("--latent_store_candidates", action="store_true")

    parser.add_argument("--reference_min_iou", type=float, default=0.10)
    parser.add_argument("--min_weight", type=float, default=0.35)
    parser.add_argument("--unmatched_weight", type=float, default=0.60)
    parser.add_argument("--stats_json", default=None)
    return parser.parse_args()


def torch_load(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def create_uv_depth(depth: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
    if mask is not None:
        depth = depth * mask
    x, y = np.meshgrid(
        np.linspace(0, depth.shape[1] - 1, depth.shape[1]),
        np.linspace(0, depth.shape[0] - 1, depth.shape[0]),
    )
    uv_depth = np.stack((x, y, depth), axis=-1).reshape(-1, 3)
    return uv_depth[uv_depth[:, 2] != 0]


def normalize_vec(v: Sequence[float]) -> np.ndarray:
    arr = np.asarray(v, dtype=np.float64).reshape(3)
    n = np.linalg.norm(arr)
    if n < 1e-12:
        return arr
    return arr / n


def rotation_matrix_from_vectors_safe(vec1, vec2) -> np.ndarray:
    a = normalize_vec(vec1)
    b = normalize_vec(vec2)
    dot = float(np.clip(np.dot(a, b), -1.0, 1.0))
    if dot > 1.0 - 1e-8:
        return np.eye(3, dtype=np.float64)
    if dot < -1.0 + 1e-8:
        axis = np.cross(a, np.array([1.0, 0.0, 0.0]))
        if np.linalg.norm(axis) < 1e-8:
            axis = np.cross(a, np.array([0.0, 0.0, 1.0]))
        axis = normalize_vec(axis)
        skew = np.array(
            [[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]]
        )
        return np.eye(3) + 2.0 * (skew @ skew)
    axis = np.cross(a, b)
    skew = np.array(
        [[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]]
    )
    return np.eye(3) + skew + skew @ skew * ((1.0 - dot) / max(np.linalg.norm(axis) ** 2, 1e-12))


def clamp_percentile_pair(low: float, high: float) -> Tuple[float, float]:
    low = float(np.clip(low, 0.0, 49.0))
    high = float(np.clip(high, 51.0, 100.0))
    if low >= high:
        low, high = 5.0, 95.0
    return low, high


def safe_adaptive_erode_mask(mask: np.ndarray, args) -> np.ndarray:
    try:
        return adaptive_erode_mask(
            mask.astype(float),
            args.mask_erode_vertical,
            args.mask_erode_vertical_min,
            args.mask_erode_horizontal,
            args.mask_erode_horizontal_min,
        )
    except Exception:
        out = np.zeros_like(mask)
        for i in range(mask.shape[0]):
            if np.asarray(mask[i]).sum() == 0:
                continue
            single = mask[i : i + 1].astype(float)
            try:
                out[i : i + 1] = adaptive_erode_mask(
                    single,
                    args.mask_erode_vertical,
                    args.mask_erode_vertical_min,
                    args.mask_erode_horizontal,
                    args.mask_erode_horizontal_min,
                )
            except Exception:
                out[i] = mask[i]
        return out


def process_ground_cached(info_ground, im_id: int, depth: np.ndarray, input_folder: str, K, args):
    if im_id not in info_ground or not info_ground[im_id]:
        return False, None
    ground_mask_path = os.path.join(input_folder, "ground_mask", f"{im_id}.npy")
    if not os.path.exists(ground_mask_path):
        return False, None
    ground_mask = np.load(ground_mask_path)
    ground_mask = erode_mask(
        ground_mask.astype(float),
        args.ground_erode_vertical,
        args.ground_erode_horizontal,
    )
    conf = np.asarray(info_ground[im_id].get("conf", []))
    if conf.size == 0:
        return False, None
    idx = int(np.argmax(conf))
    if idx >= ground_mask.shape[0]:
        return False, None
    ground_depth = depth * ground_mask[idx].squeeze()
    uv_depth = create_uv_depth(ground_depth)
    if uv_depth.shape[0] <= 10:
        return False, None
    pseudo_lidar_ground = project_image_to_cam(uv_depth, np.asarray(K, dtype=np.float64))
    if pseudo_lidar_ground.shape[0] <= 10:
        return False, None
    return True, extract_ground(pseudo_lidar_ground)


def mask_to_points(depth: np.ndarray, cur_mask: np.ndarray, K) -> np.ndarray:
    mask2d = np.asarray(cur_mask).squeeze()
    cur_depth = depth * mask2d
    uv_depth = create_uv_depth(cur_depth)
    if uv_depth.shape[0] == 0:
        return np.zeros((0, 3), dtype=np.float32)
    points = project_image_to_cam(uv_depth, np.asarray(K, dtype=np.float64))
    finite = np.all(np.isfinite(points), axis=1) & (points[:, 2] > 0)
    return points[finite].astype(np.float32)


def remove_depth_edges(
    depth: np.ndarray,
    mask: np.ndarray,
    args,
) -> Tuple[np.ndarray, Dict[str, object]]:
    mask_bool = np.asarray(mask).squeeze() > 0
    metrics = {
        "depth_edge_enabled": bool(args.use_depth_edge_filter),
        "depth_edge_raw_pixels": int(mask_bool.sum()),
        "depth_edge_kept_pixels": int(mask_bool.sum()),
        "depth_edge_keep_ratio": 1.0,
        "depth_edge_fallback": False,
    }
    if not args.use_depth_edge_filter or int(mask_bool.sum()) == 0:
        return mask_bool.astype(np.float32), metrics

    depth_np = np.asarray(depth, dtype=np.float32)
    valid = np.isfinite(depth_np) & (depth_np > 0.05)
    log_depth = np.zeros_like(depth_np, dtype=np.float32)
    log_depth[valid] = np.log(np.maximum(depth_np[valid], 0.05))

    grad_x = np.zeros_like(log_depth)
    grad_y = np.zeros_like(log_depth)
    grad_x[:, 1:-1] = np.abs(log_depth[:, 2:] - log_depth[:, :-2]) * 0.5
    grad_y[1:-1, :] = np.abs(log_depth[2:, :] - log_depth[:-2, :]) * 0.5
    edge = np.maximum(grad_x, grad_y) > float(args.depth_edge_rel_threshold)
    edge |= ~valid

    dilate = max(0, int(args.depth_edge_dilate))
    if dilate > 0:
        kernel = np.ones((2 * dilate + 1, 2 * dilate + 1), dtype=np.uint8)
        edge = cv2.dilate(edge.astype(np.uint8), kernel, iterations=1).astype(bool)

    cleaned = mask_bool & ~edge
    keep_ratio = float(cleaned.sum() / max(mask_bool.sum(), 1))
    if keep_ratio < float(args.depth_edge_min_keep_ratio):
        cleaned = mask_bool
        keep_ratio = 1.0
        metrics["depth_edge_fallback"] = True

    metrics.update(
        {
            "depth_edge_kept_pixels": int(cleaned.sum()),
            "depth_edge_keep_ratio": keep_ratio,
        }
    )
    return cleaned.astype(np.float32), metrics


def estimate_depth_normal(
    depth: np.ndarray,
    K,
    args,
) -> Tuple[Optional[np.ndarray], Dict[str, object]]:
    metrics = {
        "normal_gravity_available": False,
        "normal_gravity_count": 0,
        "normal_gravity_confidence": 0.0,
    }
    if not args.use_normal_ground_fusion:
        return None, metrics

    depth_np = np.asarray(depth, dtype=np.float64)
    h, w = depth_np.shape[:2]
    if h < 5 or w < 5:
        return None, metrics

    ys, xs = np.meshgrid(
        np.arange(h, dtype=np.float64),
        np.arange(w, dtype=np.float64),
        indexing="ij",
    )
    fx, fy = float(K[0][0]), float(K[1][1])
    cx, cy = float(K[0][2]), float(K[1][2])
    z = depth_np
    points = np.stack(((xs - cx) * z / fx, (ys - cy) * z / fy, z), axis=-1)

    stride = max(1, int(args.normal_stride))
    p_left = points[1:-1:stride, :-2:stride]
    p_right = points[1:-1:stride, 2::stride]
    p_up = points[:-2:stride, 1:-1:stride]
    p_down = points[2::stride, 1:-1:stride]
    dx = p_right - p_left
    dy = p_down - p_up
    normals = np.cross(dx, dy)
    norm = np.linalg.norm(normals, axis=-1)
    finite = np.all(np.isfinite(normals), axis=-1) & (norm > 1e-8)
    normals[finite] /= norm[finite, None]
    normals = normals[finite]
    if normals.shape[0] == 0:
        return None, metrics

    normals[normals[:, 1] > 0] *= -1.0
    vertical = -normals[:, 1]
    selected = normals[vertical >= float(args.normal_min_vertical_dot)]
    metrics["normal_gravity_count"] = int(selected.shape[0])
    if selected.shape[0] < int(args.normal_min_count):
        return None, metrics

    normal = np.median(selected, axis=0)
    normal_norm = float(np.linalg.norm(normal))
    if normal_norm < 1e-8:
        return None, metrics
    normal /= normal_norm
    angular_support = float(np.mean(np.clip(selected @ normal, 0.0, 1.0)))
    confidence = angular_support * min(1.0, selected.shape[0] / 500.0)
    metrics.update(
        {
            "normal_gravity_available": True,
            "normal_gravity_confidence": confidence,
            "normal_gravity_vector": [float(x) for x in normal.tolist()],
        }
    )
    return normal.astype(np.float64), metrics


def select_frustum_cluster(
    points: np.ndarray,
    mask: np.ndarray,
    K,
    args,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, Dict[str, object]]:
    points_np = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    metrics = {
        "dbscan_enabled": bool(args.use_frustum_dbscan),
        "dbscan_input_points": int(points_np.shape[0]),
        "dbscan_output_points": int(points_np.shape[0]),
        "dbscan_clusters": 0,
        "dbscan_keep_ratio": 1.0,
        "dbscan_fallback": False,
    }
    if not args.use_frustum_dbscan or points_np.shape[0] < int(args.dbscan_min_cluster_points):
        return points_np, metrics

    try:
        from sklearn.cluster import DBSCAN
        from scipy.spatial import cKDTree
    except Exception as exc:
        metrics.update({"dbscan_fallback": True, "dbscan_reason": f"unavailable:{str(exc)[:80]}"})
        return points_np, metrics

    fit_points = points_np
    max_fit = max(int(args.dbscan_max_fit_points), int(args.dbscan_min_cluster_points))
    if fit_points.shape[0] > max_fit:
        fit_idx = rng.choice(fit_points.shape[0], size=max_fit, replace=False)
        fit_points = fit_points[fit_idx]

    median_z = float(np.median(fit_points[:, 2]))
    eps = max(float(args.dbscan_eps_min), float(args.dbscan_eps_ratio) * max(median_z, 0.25))
    labels = DBSCAN(eps=eps, min_samples=max(2, int(args.dbscan_min_samples))).fit_predict(fit_points)
    cluster_ids = [int(v) for v in np.unique(labels) if int(v) >= 0]
    metrics.update({"dbscan_clusters": len(cluster_ids), "dbscan_eps": eps})
    if not cluster_ids:
        metrics.update({"dbscan_fallback": True, "dbscan_reason": "no_cluster"})
        return points_np, metrics

    mask_yx = np.argwhere(np.asarray(mask).squeeze() > 0)
    if mask_yx.shape[0] > 0:
        mask_center_uv = np.array([mask_yx[:, 1].mean(), mask_yx[:, 0].mean()], dtype=np.float64)
        mask_diag = float(
            np.linalg.norm(
                [np.ptp(mask_yx[:, 1]) + 1.0, np.ptp(mask_yx[:, 0]) + 1.0]
            )
        )
    else:
        mask_center_uv = np.array([float(K[0][2]), float(K[1][2])], dtype=np.float64)
        mask_diag = 1.0

    best = None
    global_depth = float(np.median(fit_points[:, 2]))
    for cluster_id in cluster_ids:
        cluster = fit_points[labels == cluster_id]
        if cluster.shape[0] < int(args.dbscan_min_cluster_points):
            continue
        center = np.median(cluster, axis=0)
        if center[2] <= 1e-6:
            continue
        uv = np.array(
            [
                float(K[0][0]) * center[0] / center[2] + float(K[0][2]),
                float(K[1][1]) * center[1] / center[2] + float(K[1][2]),
            ]
        )
        size_score = cluster.shape[0] / max(fit_points.shape[0], 1)
        ray_score = math.exp(-float(np.linalg.norm(uv - mask_center_uv)) / max(mask_diag * 0.35, 1.0))
        depth_score = math.exp(-abs(float(center[2]) - global_depth) / max(0.12 * global_depth, 0.08))
        score = 0.50 * size_score + 0.30 * ray_score + 0.20 * depth_score
        if best is None or score > best["score"]:
            best = {"score": score, "cluster": cluster, "id": cluster_id}

    if best is None:
        metrics.update({"dbscan_fallback": True, "dbscan_reason": "no_valid_cluster"})
        return points_np, metrics

    tree = cKDTree(best["cluster"])
    distances, _ = tree.query(points_np, k=1)
    keep = distances <= eps * 1.5
    selected = points_np[keep]
    keep_ratio = float(selected.shape[0] / max(points_np.shape[0], 1))
    if (
        selected.shape[0] < int(args.dbscan_min_cluster_points)
        or keep_ratio < float(args.dbscan_min_keep_ratio)
    ):
        metrics.update(
            {
                "dbscan_fallback": True,
                "dbscan_reason": "selected_cluster_too_small",
            }
        )
        return points_np, metrics

    metrics.update(
        {
            "dbscan_output_points": int(selected.shape[0]),
            "dbscan_keep_ratio": keep_ratio,
            "dbscan_selected_score": float(best["score"]),
            "dbscan_selected_id": int(best["id"]),
            "dbscan_reason": "ok",
        }
    )
    return selected.astype(np.float32), metrics


def subsample(points: np.ndarray, max_points: int, rng: np.random.Generator) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    if points.shape[0] <= max_points:
        return points
    idx = rng.choice(points.shape[0], size=max_points, replace=False)
    return points[idx]


def is_thin_class(category_name: str) -> bool:
    return str(category_name).lower().strip() in THIN_OR_PLANAR_CLASSES


def radius_filter(points: np.ndarray, radius: float, nb_points: int) -> Tuple[np.ndarray, str]:
    if points.shape[0] < max(2, nb_points):
        return points, "too_few"
    try:
        from scipy.spatial import cKDTree  # type: ignore

        tree = cKDTree(points.astype(np.float32))
        try:
            counts = tree.query_ball_point(points, r=float(radius), return_length=True)
        except TypeError:
            counts = np.asarray([len(v) for v in tree.query_ball_point(points, r=float(radius))])
        keep = np.asarray(counts) >= int(nb_points)
        if int(keep.sum()) == 0:
            return points, "all_removed"
        return points[keep], "scipy_ckdtree"
    except Exception as exc:
        return points, f"unavailable:{str(exc)[:80]}"


def dfu_filter_points(
    raw_points: np.ndarray,
    category_name: str,
    args,
) -> Tuple[Optional[np.ndarray], Dict[str, object]]:
    points = np.asarray(raw_points, dtype=np.float32).reshape(-1, 3)
    finite = np.all(np.isfinite(points), axis=1) & (points[:, 2] > 0)
    points = points[finite]
    thin = is_thin_class(category_name)
    metrics = {
        "dfu_raw_points": int(points.shape[0]),
        "dfu_filtered_points": 0,
        "dfu_keep_ratio": 0.0,
        "dfu_thin_class": bool(thin),
        "dfu_radius_backend": "disabled",
        "dfu_used_raw_fallback": False,
    }
    if points.shape[0] < int(args.min_points):
        if args.dfu_fallback_raw and points.shape[0] > 0:
            metrics.update(
                {
                    "dfu_filtered_points": int(points.shape[0]),
                    "dfu_keep_ratio": 1.0,
                    "dfu_used_raw_fallback": True,
                    "dfu_reason": "too_few_raw_use_raw",
                }
            )
            return points, metrics
        metrics["dfu_reason"] = "too_few_raw"
        return None, metrics

    p_low = args.dfu_thin_depth_percentile_low if thin else args.dfu_depth_percentile_low
    p_high = args.dfu_thin_depth_percentile_high if thin else args.dfu_depth_percentile_high
    p_low, p_high = clamp_percentile_pair(p_low, p_high)
    depth = points[:, 2]
    lo, hi = np.percentile(depth, [p_low, p_high])
    keep = (depth >= lo) & (depth <= hi)
    filtered = points[keep]
    if filtered.shape[0] < int(args.min_points):
        filtered = points

    depth_f = filtered[:, 2]
    median = float(np.median(depth_f))
    mad = float(np.median(np.abs(depth_f - median)))
    sigma = 1.4826 * mad
    mad_scale = args.dfu_thin_mad_scale if thin else args.dfu_mad_scale
    window = max(float(args.dfu_min_depth_window), float(mad_scale) * sigma)
    mad_keep = np.abs(depth_f - median) <= window
    if int(mad_keep.sum()) >= int(args.min_points):
        filtered = filtered[mad_keep]

    if args.dfu_use_axis_filter and filtered.shape[0] >= int(args.min_points):
        a_low, a_high = clamp_percentile_pair(
            args.thin_extent_percentile_low if thin else args.dfu_axis_percentile_low,
            args.thin_extent_percentile_high if thin else args.dfu_axis_percentile_high,
        )
        lows, highs = np.percentile(filtered, [a_low, a_high], axis=0)
        axis_keep = np.all((filtered >= lows.reshape(1, 3)) & (filtered <= highs.reshape(1, 3)), axis=1)
        if int(axis_keep.sum()) >= int(args.min_points):
            filtered = filtered[axis_keep]

    if args.dfu_use_radius_filter and filtered.shape[0] >= int(args.min_points):
        radius_out, backend = radius_filter(
            filtered,
            args.dfu_radius,
            args.dfu_radius_nb_points,
        )
        metrics["dfu_radius_backend"] = backend
        if radius_out.shape[0] >= int(args.min_points):
            filtered = radius_out

    if filtered.shape[0] < int(args.min_points):
        if args.dfu_fallback_raw:
            filtered = points
            metrics["dfu_used_raw_fallback"] = True
            metrics["dfu_reason"] = "filter_too_few_use_raw"
        else:
            metrics["dfu_reason"] = "filter_too_few"
            return None, metrics
    else:
        metrics["dfu_reason"] = "ok"

    metrics.update(
        {
            "dfu_filtered_points": int(filtered.shape[0]),
            "dfu_keep_ratio": float(filtered.shape[0] / max(points.shape[0], 1)),
            "dfu_depth_median": float(np.median(filtered[:, 2])),
            "dfu_depth_low": float(np.min(filtered[:, 2])),
            "dfu_depth_high": float(np.max(filtered[:, 2])),
        }
    )
    return filtered.astype(np.float32), metrics


def robust_pca_yaw(points_xz: np.ndarray) -> float:
    xz = np.asarray(points_xz, dtype=np.float64).reshape(-1, 2)
    center = np.median(xz, axis=0, keepdims=True)
    centered = xz - center
    if centered.shape[0] < 3 or np.linalg.norm(centered) < 1e-8:
        return 0.0
    cov = centered.T @ centered / max(centered.shape[0] - 1, 1)
    vals, vecs = np.linalg.eigh(cov)
    vec = vecs[:, int(np.argmax(vals))]
    return float(np.arctan2(vec[1], vec[0]))


def object_prior_bounds(prior: Sequence[float], args) -> Tuple[np.ndarray, np.ndarray]:
    w, h, l = [float(v) for v in prior]
    # Object-local xyz corresponds to [dx, dy, dz], while Omni3D dimensions are [dz, dy, dx].
    prior_obj = np.asarray([l, h, w], dtype=np.float64)
    floor = np.maximum(prior_obj * float(args.prior_floor_ratio), float(args.min_dimension))
    ceil = np.maximum(prior_obj * float(args.prior_ceiling_ratio), floor * 1.25)
    return floor, ceil


def make_box(
    cx: float,
    cy: float,
    cz: float,
    dx: float,
    dy: float,
    dz: float,
    yaw: float,
    rotation_matrix: np.ndarray,
):
    vertices = convert_box_vertices(cx, cy, cz, dx, dy, dz, 0).astype(np.float64)
    vertices = (rotate_y(-yaw) @ vertices.T).T
    vertices = vertices @ rotation_matrix.T
    center_cam = vertices.mean(axis=0)
    dims_omni = np.asarray([dz, dy, dx], dtype=np.float64)
    R_cam = rotation_matrix @ rotate_y(-yaw)
    return vertices, center_cam, dims_omni, R_cam


def box_vertices_from_pose(
    center_cam: np.ndarray,
    dims_omni: np.ndarray,
    R_cam: np.ndarray,
) -> np.ndarray:
    center = np.asarray(center_cam, dtype=np.float64).reshape(3)
    dims = np.asarray(dims_omni, dtype=np.float64).reshape(3)
    # Omni3D dimensions are [width(z-local), height(y-local), length(x-local)].
    dx, dy, dz = float(dims[2]), float(dims[1]), float(dims[0])
    local = np.array(
        [
            [-dx / 2, -dy / 2, -dz / 2],
            [dx / 2, -dy / 2, -dz / 2],
            [dx / 2, dy / 2, -dz / 2],
            [-dx / 2, dy / 2, -dz / 2],
            [-dx / 2, -dy / 2, dz / 2],
            [dx / 2, -dy / 2, dz / 2],
            [dx / 2, dy / 2, dz / 2],
            [-dx / 2, dy / 2, dz / 2],
        ],
        dtype=np.float64,
    )
    return local @ np.asarray(R_cam, dtype=np.float64).reshape(3, 3).T + center


def projected_box_metrics(
    vertices: np.ndarray,
    K,
    target_bbox: Sequence[float],
    target_mask: Optional[np.ndarray],
) -> Tuple[float, float]:
    vertices_np = np.asarray(vertices, dtype=np.float64).reshape(-1, 3)
    if np.any(vertices_np[:, 2] <= 1e-4):
        return 0.0, 0.0
    K_np = np.asarray(K, dtype=np.float64).reshape(3, 3)
    projected = vertices_np @ K_np.T
    uv = projected[:, :2] / projected[:, 2:3]
    if not np.all(np.isfinite(uv)):
        return 0.0, 0.0

    proj_bbox = [
        float(np.min(uv[:, 0])),
        float(np.min(uv[:, 1])),
        float(np.max(uv[:, 0])),
        float(np.max(uv[:, 1])),
    ]
    bbox_score = bbox_iou(proj_bbox, target_bbox) if bbox_iou is not None else 0.0
    silhouette_score = bbox_score
    if target_mask is not None:
        mask_bool = np.asarray(target_mask).squeeze() > 0
        if mask_bool.ndim == 2 and int(mask_bool.sum()) > 0:
            h, w = mask_bool.shape
            hull_points = uv.copy()
            hull_points[:, 0] = np.clip(hull_points[:, 0], 0, max(w - 1, 0))
            hull_points[:, 1] = np.clip(hull_points[:, 1], 0, max(h - 1, 0))
            hull = cv2.convexHull(np.round(hull_points).astype(np.int32))
            rendered = np.zeros((h, w), dtype=np.uint8)
            if hull.shape[0] >= 3:
                cv2.fillConvexPoly(rendered, hull, 1)
                inter = int(np.logical_and(rendered > 0, mask_bool).sum())
                union = int(np.logical_or(rendered > 0, mask_bool).sum())
                silhouette_score = float(inter / union) if union > 0 else 0.0
    return float(bbox_score), float(silhouette_score)


def point_support_3d(
    points_cam: np.ndarray,
    center_cam: np.ndarray,
    dims_omni: np.ndarray,
    R_cam: np.ndarray,
    margin: float = 0.02,
) -> float:
    points = np.asarray(points_cam, dtype=np.float64).reshape(-1, 3)
    if points.shape[0] == 0:
        return 0.0
    dims = np.asarray(dims_omni, dtype=np.float64).reshape(3)
    half_obj = np.asarray([dims[2], dims[1], dims[0]], dtype=np.float64) * 0.5 + margin
    local = (points - np.asarray(center_cam).reshape(1, 3)) @ np.asarray(R_cam).reshape(3, 3)
    return float(np.mean(np.all(np.abs(local) <= half_obj.reshape(1, 3), axis=1)))


def clamp_weight(value: float, args) -> float:
    return float(
        np.clip(
            value,
            float(args.latent_min_attribute_weight),
            float(args.latent_max_attribute_weight),
        )
    )


def circular_concentration(angles: np.ndarray, weights: np.ndarray) -> float:
    if angles.size == 0:
        return 0.0
    # Upright cuboids are pi-periodic in yaw.
    vector = np.sum(weights * np.exp(2j * angles))
    return float(np.clip(np.abs(vector), 0.0, 1.0))


def latent_candidate_statistics(candidates: List[dict], args) -> dict:
    if not candidates:
        return {}

    candidates = sorted(candidates, key=lambda item: item["loss"])
    topk = candidates[: max(1, int(args.latent_topk))]
    losses = np.asarray([item["loss"] for item in topk], dtype=np.float64)
    temperature = max(float(args.latent_temperature), 1e-4)
    logits = -(losses - losses.min()) / temperature
    logits -= logits.max()
    posterior = np.exp(logits)
    posterior /= max(float(posterior.sum()), 1e-12)

    centers = np.stack([item["center"] for item in topk], axis=0)
    log_dims = np.log(
        np.maximum(np.stack([item["dims"] for item in topk], axis=0), 1e-4)
    )
    yaw_offsets = np.asarray([item["yaw_offset"] for item in topk], dtype=np.float64)
    center_mean = np.sum(centers * posterior[:, None], axis=0)
    log_dims_mean = np.sum(log_dims * posterior[:, None], axis=0)
    center_std = np.sqrt(
        np.sum(((centers - center_mean) ** 2) * posterior[:, None], axis=0)
    )
    dims_log_std = np.sqrt(
        np.sum(((log_dims - log_dims_mean) ** 2) * posterior[:, None], axis=0)
    )

    best = topk[0]
    margin = float(losses[1] - losses[0]) if losses.size > 1 else temperature
    margin_conf = float(1.0 - math.exp(-max(margin, 0.0) / temperature))
    effective_count = float(1.0 / max(np.sum(posterior**2), 1e-12))
    posterior_conf = float(
        np.clip(
            1.0 - (effective_count - 1.0) / max(len(topk) - 1, 1),
            0.0,
            1.0,
        )
    )
    mode_conf = 0.5 * margin_conf + 0.5 * posterior_conf

    bbox_conf = float(np.clip(best["bbox_iou"], 0.0, 1.0))
    silhouette_conf = float(np.clip(best["silhouette_iou"], 0.0, 1.0))
    depth_conf = float(math.exp(-3.0 * min(float(best["depth_error"]), 2.0)))
    support_conf = float(np.clip(best["support"], 0.0, 1.0))
    prior_conf = float(math.exp(-min(float(best["prior_error"]), 3.0)))
    yaw_conf = circular_concentration(yaw_offsets, posterior)

    center_xy_stability = float(math.exp(-8.0 * np.linalg.norm(center_std[:2])))
    center_z_stability = float(math.exp(-8.0 * center_std[2]))
    dims_stability = float(math.exp(-6.0 * float(np.mean(dims_log_std))))

    weight_xy = clamp_weight(
        0.35 * bbox_conf
        + 0.25 * silhouette_conf
        + 0.20 * support_conf
        + 0.10 * center_xy_stability
        + 0.10 * mode_conf,
        args,
    )
    weight_z = clamp_weight(
        0.50 * depth_conf
        + 0.25 * support_conf
        + 0.15 * center_z_stability
        + 0.10 * mode_conf,
        args,
    )
    weight_dims = clamp_weight(
        0.30 * silhouette_conf
        + 0.30 * support_conf
        + 0.15 * prior_conf
        + 0.15 * dims_stability
        + 0.10 * mode_conf,
        args,
    )
    weight_pose = clamp_weight(
        0.35 * silhouette_conf
        + 0.25 * support_conf
        + 0.25 * yaw_conf
        + 0.15 * mode_conf,
        args,
    )
    weight_joint = clamp_weight(
        float(np.exp(np.mean(np.log(np.maximum(
            [weight_xy, weight_z, weight_dims, weight_pose],
            1e-4,
        ))))),
        args,
    )

    result = {
        "latent_box_enabled": True,
        "latent_box_candidate_count": int(len(candidates)),
        "latent_box_topk": int(len(topk)),
        "latent_box_loss_margin": margin,
        "latent_box_mode_confidence": mode_conf,
        "latent_box_effective_count": effective_count,
        "latent_box_center_std": [float(v) for v in center_std],
        "latent_box_dims_log_std": [float(v) for v in dims_log_std],
        "latent_box_yaw_concentration": yaw_conf,
        "pseudo_weight_xy": weight_xy,
        "pseudo_weight_z": weight_z,
        "pseudo_weight_dims": weight_dims,
        "pseudo_weight_pose": weight_pose,
        "pseudo_weight_joint": weight_joint,
        "pseudo_weight": weight_joint,
    }
    if args.latent_store_candidates:
        result["latent_box_candidates"] = [
            {
                "loss": float(item["loss"]),
                "center_cam": [float(v) for v in item["center"]],
                "dimensions": [float(v) for v in item["dims"]],
                "yaw_offset": float(item["yaw_offset"]),
                "bbox_iou": float(item["bbox_iou"]),
                "silhouette_iou": float(item["silhouette_iou"]),
                "depth_rel_error": float(item["depth_error"]),
                "point_support": float(item["support"]),
                "posterior": float(posterior[index]),
            }
            for index, item in enumerate(topk)
        ]
    return result


def optimize_box_surface_consistency(
    fit,
    points_cam: np.ndarray,
    prior: Sequence[float],
    K,
    target_bbox: Sequence[float],
    target_mask: Optional[np.ndarray],
    ground_equ: Optional[np.ndarray],
    args,
):
    vertices, center_cam, dims_omni, R_cam = fit
    metrics = {
        "surface_opt_enabled": bool(args.use_surface_box_optimization),
        "surface_opt_applied": False,
        "surface_opt_candidates": 0,
        "latent_box_enabled": bool(args.use_latent_box_closure),
    }
    if not args.use_surface_box_optimization:
        return fit, metrics

    points = np.asarray(points_cam, dtype=np.float64).reshape(-1, 3)
    if points.shape[0] < 3:
        metrics["surface_opt_reason"] = "too_few_points"
        return fit, metrics

    center_base = np.asarray(center_cam, dtype=np.float64).reshape(3)
    dims_base = np.asarray(dims_omni, dtype=np.float64).reshape(3)
    R_base = np.asarray(R_cam, dtype=np.float64).reshape(3, 3)
    observed_surface = float(
        np.percentile(
            points[:, 2],
            np.clip(float(args.surface_depth_percentile), 1.0, 50.0),
        )
    )
    base_front = float(np.min(np.asarray(vertices)[:, 2]))
    raw_shift = observed_surface - base_front
    max_shift = float(args.surface_max_shift_ratio) * max(float(center_base[2]), 0.25)
    raw_shift = float(np.clip(raw_shift, -max_shift, max_shift))

    delta = max(0.0, float(args.surface_scale_delta))
    height_delta = max(0.0, float(args.surface_height_scale_delta))
    scale_values = sorted(set([max(0.5, 1.0 - delta), 1.0, 1.0 + delta]))
    height_scale_values = sorted(
        set([max(0.5, 1.0 - height_delta), 1.0, 1.0 + height_delta])
    )
    yaw_delta = math.radians(max(0.0, float(args.surface_yaw_delta_deg)))
    yaw_values = sorted(set([-yaw_delta, 0.0, yaw_delta]))
    if args.surface_center_mode == "locked":
        depth_blends = [0.0]
        xy_blends = [0.0]
    elif args.surface_center_mode == "conservative":
        depth_blends = [0.0, 0.25]
        xy_blends = [0.0, 0.15]
    else:
        depth_blends = [0.0, 0.5, 1.0]
        xy_blends = [0.0, 0.35]
    xy_target = np.median(points[:, :2], axis=0)
    prior_omni = np.asarray(prior, dtype=np.float64).reshape(3)
    base_bbox_iou, base_silhouette_iou = projected_box_metrics(
        vertices,
        K,
        target_bbox,
        target_mask,
    )
    base_depth_error = abs(base_front - observed_surface) / max(observed_surface, 0.10)
    base_support = point_support_3d(points, center_base, dims_base, R_base)
    min_support = max(
        float(args.surface_min_point_support),
        base_support * float(args.surface_min_support_ratio),
    )
    metrics.update(
        {
            "surface_opt_base_bbox_iou": float(base_bbox_iou),
            "surface_opt_base_silhouette_iou": float(base_silhouette_iou),
            "surface_opt_base_depth_rel_error": float(base_depth_error),
            "surface_opt_base_point_support": float(base_support),
            "surface_opt_min_point_support": float(min_support),
        }
    )

    best = None
    valid_candidates = []
    for yaw_offset in yaw_values:
        R_candidate = R_base @ rotate_y(yaw_offset)
        for scale_w in scale_values:
            for scale_h in height_scale_values:
                for scale_l in scale_values:
                    dims_candidate = dims_base * np.asarray(
                        [scale_w, scale_h, scale_l],
                        dtype=np.float64,
                    )
                    for depth_blend in depth_blends:
                        for xy_blend in xy_blends:
                            center_candidate = center_base.copy()
                            center_candidate[:2] += (xy_target - center_candidate[:2]) * xy_blend
                            center_candidate[2] += raw_shift * depth_blend
                            vertices_candidate = box_vertices_from_pose(
                                center_candidate,
                                dims_candidate,
                                R_candidate,
                            )
                            if np.any(vertices_candidate[:, 2] <= 1e-4):
                                continue

                            bbox_score, silhouette_score = projected_box_metrics(
                                vertices_candidate,
                                K,
                                target_bbox,
                                target_mask,
                            )
                            front_depth = float(np.min(vertices_candidate[:, 2]))
                            depth_error = abs(front_depth - observed_surface) / max(observed_surface, 0.10)
                            support = point_support_3d(
                                points,
                                center_candidate,
                                dims_candidate,
                                R_candidate,
                            )
                            if support < min_support:
                                continue
                            prior_error = float(
                                np.mean(
                                    np.abs(
                                        np.log(
                                            np.maximum(dims_candidate, 1e-4)
                                            / np.maximum(prior_omni, 1e-4)
                                        )
                                    )
                                )
                            )
                            ground_error = 0.0
                            if ground_equ is not None:
                                bottom_dist = min(
                                    point_to_plane_distance(ground_equ, *point)
                                    for point in vertices_candidate
                                )
                                ground_error = min(float(bottom_dist), 1.0)

                            loss = (
                                float(args.surface_projection_weight) * (1.0 - bbox_score)
                                + float(args.surface_silhouette_weight) * (1.0 - silhouette_score)
                                + float(args.surface_depth_weight) * min(depth_error, 2.0)
                                + float(args.surface_support_weight) * (1.0 - support)
                                + float(args.surface_prior_weight) * prior_error
                                + 0.10 * ground_error
                            )
                            metrics["surface_opt_candidates"] += 1
                            candidate = {
                                "loss": loss,
                                "vertices": vertices_candidate,
                                "center": center_candidate,
                                "dims": dims_candidate,
                                "R": R_candidate,
                                "bbox_iou": bbox_score,
                                "silhouette_iou": silhouette_score,
                                "depth_error": depth_error,
                                "support": support,
                                "prior_error": prior_error,
                                "front_depth": front_depth,
                                "depth_blend": depth_blend,
                                "xy_blend": xy_blend,
                                "scale_dims": [scale_w, scale_h, scale_l],
                                "yaw_offset": yaw_offset,
                            }
                            if args.use_latent_box_closure:
                                valid_candidates.append(candidate)
                            if best is None or loss < best["loss"]:
                                best = candidate

    if best is None:
        metrics["surface_opt_reason"] = "no_candidate"
        return fit, metrics

    metrics.update(
        {
            "surface_opt_applied": bool(
                abs(float(best["depth_blend"])) > 1e-6
                or abs(float(best["xy_blend"])) > 1e-6
                or any(abs(float(v) - 1.0) > 1e-6 for v in best["scale_dims"])
                or abs(float(best["yaw_offset"])) > 1e-6
            ),
            "surface_opt_reason": "valid",
            "surface_opt_loss": float(best["loss"]),
            "surface_opt_bbox_iou": float(best["bbox_iou"]),
            "surface_opt_silhouette_iou": float(best["silhouette_iou"]),
            "surface_opt_depth_rel_error": float(best["depth_error"]),
            "surface_opt_point_support": float(best["support"]),
            "surface_opt_prior_log_error": float(best["prior_error"]),
            "surface_opt_observed_depth": observed_surface,
            "surface_opt_front_depth": float(best["front_depth"]),
            "surface_opt_depth_blend": float(best["depth_blend"]),
            "surface_opt_xy_blend": float(best["xy_blend"]),
            "surface_opt_scale_dims": [float(v) for v in best["scale_dims"]],
            "surface_opt_yaw_offset": float(best["yaw_offset"]),
        }
    )
    if args.use_latent_box_closure:
        metrics.update(latent_candidate_statistics(valid_candidates, args))
    return (
        best["vertices"].astype(np.float32),
        best["center"].astype(np.float32),
        best["dims"].astype(np.float32),
        best["R"].astype(np.float32),
    ), metrics


def ray_loss(rotated_pc: np.ndarray, yaw: float, dx: float, dz: float, cx: float, cz: float) -> float:
    try:
        pc_tensor = torch.from_numpy(rotated_pc[:, [0, 2]].astype(np.float32))
        loss = calc_dis_ray_tracing(
            torch.tensor([float(dz), float(dx)], dtype=torch.float32),
            torch.tensor([float(yaw)], dtype=torch.float32),
            pc_tensor,
            torch.tensor([float(cx), float(cz)], dtype=torch.float32),
        )
        return float(loss.detach().cpu().item() if hasattr(loss, "detach") else loss)
    except Exception:
        return 0.0


def estimate_bbox_dfu_robust(
    points_cam: np.ndarray,
    prior: Sequence[float],
    category_name: str,
    ground_equ: Optional[np.ndarray],
    gravity_normal: Optional[np.ndarray],
    args,
    rng: np.random.Generator,
):
    if points_cam is None or np.asarray(points_cam).shape[0] < 3:
        return None, {"fit_reason": "too_few_points"}

    in_pc = subsample(np.asarray(points_cam, dtype=np.float32), int(args.max_pca_points), rng)
    if in_pc.shape[0] < 3:
        return None, {"fit_reason": "too_few_after_subsample"}

    w, h, l = [float(v) for v in prior]
    thin = is_thin_class(category_name)

    normal_source = "identity"
    if ground_equ is not None and np.all(np.isfinite(ground_equ[:3])):
        ground_equ = np.asarray(ground_equ, dtype=np.float64).copy()
        if np.dot(np.array([0.0, -1.0, 0.0]), ground_equ[:3]) <= 0:
            ground_equ = -ground_equ
        ground_normal = normalize_vec(ground_equ[:3])
        rotation_normal = ground_normal
        normal_source = "ground"
        if gravity_normal is not None and np.all(np.isfinite(gravity_normal)):
            depth_normal = normalize_vec(gravity_normal)
            if np.dot(depth_normal, ground_normal) < 0:
                depth_normal = -depth_normal
            blend = float(np.clip(args.normal_fusion_weight, 0.0, 1.0))
            rotation_normal = normalize_vec((1.0 - blend) * ground_normal + blend * depth_normal)
            normal_source = "ground_depth_normal_fused"
        new_ground_equ = np.array(
            [0.0, -1.0, 0.0, point_to_plane_distance(ground_equ, 0, 0, 0)],
            dtype=np.float64,
        )
        rotation_matrix = rotation_matrix_from_vectors_safe([0.0, -1.0, 0.0], rotation_normal)
        has_ground = True
    elif gravity_normal is not None and np.all(np.isfinite(gravity_normal)):
        rotation_matrix = rotation_matrix_from_vectors_safe(
            [0.0, -1.0, 0.0],
            normalize_vec(gravity_normal),
        )
        new_ground_equ = None
        has_ground = False
        normal_source = "depth_normal"
    else:
        rotation_matrix = np.eye(3, dtype=np.float64)
        new_ground_equ = None
        has_ground = False

    rotated_pc = in_pc.astype(np.float64) @ rotation_matrix
    base_yaw = robust_pca_yaw(rotated_pc[:, [0, 2]])
    yaw_candidates = [base_yaw, base_yaw + np.pi / 2.0, base_yaw - np.pi / 2.0]
    e_low = args.thin_extent_percentile_low if thin else args.extent_percentile_low
    e_high = args.thin_extent_percentile_high if thin else args.extent_percentile_high
    e_low, e_high = clamp_percentile_pair(e_low, e_high)
    prior_floor, prior_ceil = object_prior_bounds(prior, args)
    direct_ratio = float(args.direct_prior_ratio)

    best = None
    candidates_evaluated = 0
    for yaw in yaw_candidates:
        rotated_pc_2 = (rotate_y(yaw) @ rotated_pc.T).T
        lows, highs = np.percentile(rotated_pc_2, [e_low, e_high], axis=0)
        x_min, y_min, z_min = lows
        x_max, y_max, z_max = highs
        dx, dy, dz = x_max - x_min, y_max - y_min, z_max - z_min
        if not np.all(np.isfinite([dx, dy, dz])) or dx <= 0 or dy <= 0 or dz <= 0:
            continue

        dx = float(np.clip(dx, prior_floor[0], prior_ceil[0]))
        dy = float(np.clip(dy, max(float(args.min_dimension), prior_floor[1] * 0.5), prior_ceil[1]))
        dz = float(np.clip(dz, prior_floor[2], prior_ceil[2]))
        cx, cy, cz = float((x_min + x_max) / 2.0), float((y_min + y_max) / 2.0), float((z_min + z_max) / 2.0)

        if dy < h * float(args.height_prior_ratio):
            dy = float(np.clip(h, prior_floor[1], prior_ceil[1]))
            if has_ground and new_ground_equ is not None:
                cdis = point_to_plane_distance(new_ground_equ, cx, cy, cz)
                if cdis - dy / 2.0 < float(args.ground_snap_distance):
                    cy += cdis - dy / 2.0

        direct_ok = (l * direct_ratio <= dx and w * direct_ratio <= dz) or (
            l * direct_ratio <= dz and w * direct_ratio <= dx
        )

        proposal_boxes = [[cx - dx / 2.0, cx + dx / 2.0, cz - dz / 2.0, cz + dz / 2.0]]
        proposal_source = ["direct"]
        if not direct_ok:
            try:
                proposal_boxes = generate_possible_bboxs(cx, cz, dx, dz, w, l)
                proposal_source = ["prior_ray"] * len(proposal_boxes)
            except Exception:
                proposal_boxes = [[cx - dx / 2.0, cx + dx / 2.0, cz - dz / 2.0, cz + dz / 2.0]]
                proposal_source = ["direct_fallback"]

        for source, proposal in zip(proposal_source, proposal_boxes):
            px_min, px_max, pz_min, pz_max = [float(v) for v in proposal]
            pdx = max(px_max - px_min, float(args.min_dimension))
            pdz = max(pz_max - pz_min, float(args.min_dimension))
            pcx, pcz = (px_min + px_max) / 2.0, (pz_min + pz_max) / 2.0
            pdx = float(np.clip(pdx, prior_floor[0], prior_ceil[0]))
            pdz = float(np.clip(pdz, prior_floor[2], prior_ceil[2]))

            inside = float(
                calc_inside_ratio(
                    rotated_pc_2.T,
                    pcx - pdx / 2.0,
                    pcx + pdx / 2.0,
                    pcz - pdz / 2.0,
                    pcz + pdz / 2.0,
                )
            )
            rloss = ray_loss(rotated_pc, yaw, pdx, pdz, pcx, pcz)
            prior_obj = np.asarray([l, h, w], dtype=np.float64)
            cand_obj = np.asarray([pdx, dy, pdz], dtype=np.float64)
            prior_error = float(np.mean(np.abs(np.log(np.maximum(cand_obj, 1e-4) / np.maximum(prior_obj, 1e-4)))))
            loss = float(rloss + 5.0 * (1.0 - inside) + 0.15 * prior_error)
            candidates_evaluated += 1
            if best is None or loss < best["loss"]:
                vertices, center_cam, dims_omni, R_cam = make_box(
                    pcx, cy, pcz, pdx, dy, pdz, yaw, rotation_matrix
                )
                best = {
                    "loss": loss,
                    "vertices": vertices,
                    "center_cam": center_cam,
                    "dimensions": dims_omni,
                    "R_cam": R_cam,
                    "yaw": float(yaw),
                    "inside_ratio": inside,
                    "ray_loss": rloss,
                    "prior_log_error": prior_error,
                    "fit_source": source,
                    "local_extent_xyz": [float(pdx), float(dy), float(pdz)],
                    "direct_ok": bool(direct_ok),
                }

    if best is None:
        return None, {"fit_reason": "no_candidate"}

    metrics = {
        "fit_reason": "valid",
        "fit_source": best["fit_source"],
        "fit_candidates": int(candidates_evaluated),
        "fit_loss": float(best["loss"]),
        "fit_inside_ratio": float(best["inside_ratio"]),
        "fit_ray_loss": float(best["ray_loss"]),
        "fit_prior_log_error": float(best["prior_log_error"]),
        "fit_yaw": float(best["yaw"]),
        "fit_has_ground": bool(has_ground),
        "fit_rotation_source": normal_source,
        "fit_direct_ok": bool(best["direct_ok"]),
        "fit_local_extent_xyz": best["local_extent_xyz"],
    }
    return (
        best["vertices"].astype(np.float32),
        best["center_cam"].astype(np.float32),
        best["dimensions"].astype(np.float32),
        best["R_cam"].astype(np.float32),
    ), metrics


def to_list(x):
    if hasattr(x, "tolist"):
        return x.tolist()
    return x


def normalize_bbox_to_pixels(bbox, width: int, height: int) -> List[float]:
    box = np.asarray(to_list(bbox), dtype=np.float64).reshape(-1)
    if box.size != 4 or not np.all(np.isfinite(box)):
        return [-1.0, -1.0, -1.0, -1.0]
    if float(np.max(np.abs(box))) <= 2.0:
        box = np.asarray([box[0] * width, box[1] * height, box[2] * width, box[3] * height])
    x1, y1, x2, y2 = box.tolist()
    return [float(x1), float(y1), float(x2), float(y2)]


def invalid_box_vertices():
    return np.full((8, 3), -1.0, dtype=np.float32)


def build_invalid_annotation(ann_id, dataset_id, im_id, category_name, category_id, bbox, score, extra):
    obj = {
        "id": int(dataset_id * 10000000 + ann_id),
        "image_id": int(im_id),
        "dataset_id": int(dataset_id),
        "category_name": str(category_name),
        "category_id": int(category_id),
        "valid3D": False,
        "bbox2D_tight": [float(x) for x in bbox],
        "bbox2D_trunc": [float(x) for x in bbox],
        "bbox2D_proj": [float(x) for x in bbox],
        "bbox3D_cam": invalid_box_vertices().tolist(),
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
        "pseudo_weight": 0.35,
    }
    obj.update(extra)
    return obj


def build_source_anchor_index(source: dict):
    index = defaultdict(list)
    for ann in source.get("annotations", []):
        if not bool(ann.get("valid3D", True)):
            continue
        if ann_box_xyxy is None:
            continue
        box = ann_box_xyxy(ann)
        if box is None:
            continue
        key = (int(ann["image_id"]), int(ann["category_id"]))
        index[key].append((ann, box))
    return index


def match_source_anchor(
    anchor_index,
    used_anchor_ids: set,
    image_id: int,
    category_id: int,
    target_bbox: Sequence[float],
):
    if anchor_index is None or bbox_iou is None:
        return None, 0.0
    best = None
    for ann, source_box in anchor_index.get((int(image_id), int(category_id)), []):
        ann_id = int(ann.get("id", -1))
        if ann_id in used_anchor_ids:
            continue
        overlap = bbox_iou(target_bbox, source_box)
        if best is None or overlap > best[0]:
            best = (overlap, ann)
    if best is None:
        return None, 0.0
    used_anchor_ids.add(int(best[1].get("id", -1)))
    return best[1], float(best[0])


def annotation_to_fit(ann: dict):
    try:
        vertices = np.asarray(ann["bbox3D_cam"], dtype=np.float32).reshape(8, 3)
        center = np.asarray(ann["center_cam"], dtype=np.float32).reshape(3)
        dims = np.asarray(ann["dimensions"], dtype=np.float32).reshape(3)
        rotation = np.asarray(ann["R_cam"], dtype=np.float32).reshape(3, 3)
    except Exception:
        return None
    if (
        not np.all(np.isfinite(vertices))
        or not np.all(np.isfinite(center))
        or not np.all(np.isfinite(dims))
        or not np.all(np.isfinite(rotation))
        or np.any(dims <= 0)
        or center[2] <= 0
    ):
        return None
    return vertices, center, dims, rotation


def build_valid_annotation(
    ann_id,
    dataset_id,
    im_id,
    category_name,
    category_id,
    bbox,
    score,
    fit,
    extra,
):
    vertices, center_cam, dimensions, R_cam = fit
    obj = {
        "id": int(dataset_id * 10000000 + ann_id),
        "image_id": int(im_id),
        "dataset_id": int(dataset_id),
        "category_name": str(category_name),
        "category_id": int(category_id),
        "valid3D": True,
        "bbox2D_tight": [float(x) for x in bbox],
        "bbox2D_trunc": [float(x) for x in bbox],
        "bbox2D_proj": [float(x) for x in bbox],
        "bbox3D_cam": [[float(x) for x in row] for row in vertices.tolist()],
        "center_cam": [float(x) for x in center_cam.tolist()],
        "dimensions": [float(x) for x in dimensions.tolist()],
        "R_cam": [[float(x) for x in row] for row in R_cam.tolist()],
        "behind_camera": False,
        "visibility": 1.0,
        "truncation": 0.0,
        "segmentation_pts": -1,
        "lidar_pts": -1,
        "depth_error": -1,
        "score": float(score),
    }
    obj.update(extra)
    return obj


def reference_match_update(ann: dict, ref_index, args) -> dict:
    if ref_index is None or ann_box_xyxy is None or bbox_iou is None or consistency is None:
        ann["pseudo_weight"] = float(ann.get("pseudo_weight", 1.0))
        ann["ng_match_found"] = False
        return ann

    if not bool(ann.get("valid3D", True)):
        ann["pseudo_weight"] = float(args.min_weight)
        ann["ng_match_found"] = False
        return ann

    box = ann_box_xyxy(ann)
    key = (int(ann["image_id"]), int(ann["category_id"]))
    best = None
    for ref_ann, ref_box in ref_index.get(key, []):
        iou = bbox_iou(box, ref_box)
        if best is None or iou > best[0]:
            best = (iou, ref_ann)

    if best is None or best[0] < float(args.reference_min_iou):
        ann["pseudo_weight"] = float(args.unmatched_weight)
        ann["ng_consistency_score"] = 0.0
        ann["ng_match_iou_2d"] = float(best[0]) if best is not None else 0.0
        ann["ng_match_found"] = False
        return ann
    ann.update(consistency(ann, best[1], best[0], float(args.min_weight)))
    return ann


def main():
    args = parse_args()
    rng = np.random.default_rng(int(args.seed))

    with open(args.source_json, "r") as f:
        source = json.load(f)

    input_folder = os.path.join(args.pseudo_root, args.dataset, args.split)
    info_path = os.path.join(input_folder, "info.pth")
    ground_path = os.path.join(input_folder, "info_ground.pth")
    info = torch_load(info_path)
    info_ground = torch_load(ground_path) if os.path.exists(ground_path) else {}

    ref_index = None
    if args.reference_json:
        if build_index is None:
            raise RuntimeError("Could not import build_ng_consistency_pseudolabels helpers")
        with open(args.reference_json, "r") as f:
            ref_data = json.load(f)
        ref_index = build_index(ref_data)

    categories = source.get("categories", [])
    category_name_to_id = {str(cat["name"]): int(cat["id"]) for cat in categories}
    dataset_info = source.get("info", {})
    dataset_id = int(dataset_info.get("id", 12 if args.split == "train" else 13))
    images = list(source.get("images", []))
    if args.max_images is not None:
        images = images[: int(args.max_images)]

    annotations = []
    stats = defaultdict(int)
    weight_values = []
    ann_id = 1
    source_anchor_index = build_source_anchor_index(source) if args.use_source_geometry_anchor else None

    for img_info in tqdm(images, desc="DFU robust PCA"):
        im_id = int(img_info["id"])
        if im_id not in info or not info[im_id]:
            stats["images_without_info"] += 1
            continue

        depth_path = os.path.join(input_folder, "depth", f"{im_id}.npy")
        mask_path = os.path.join(input_folder, "mask", f"{im_id}.npy")
        if not os.path.exists(depth_path) or not os.path.exists(mask_path):
            stats["images_missing_cache"] += 1
            continue

        depth = np.load(depth_path).astype(np.float32)
        raw_mask = np.load(mask_path)
        mask = safe_adaptive_erode_mask(raw_mask, args)
        K = img_info["K"]
        width, height = int(img_info["width"]), int(img_info["height"])
        has_ground, ground_equ = process_ground_cached(info_ground, im_id, depth, input_folder, K, args)
        gravity_normal, gravity_metrics = estimate_depth_normal(depth, K, args)

        img_data = info[im_id]
        phrases = list(img_data.get("phrases", []))
        boxes = img_data.get("boxes", [])
        scores = img_data.get("conf", np.ones(len(phrases), dtype=np.float32))
        n = min(len(phrases), len(boxes), mask.shape[0], raw_mask.shape[0])
        stats["objects_seen"] += int(n)
        used_anchor_ids = set()

        for j in range(n):
            category_name = str(phrases[j])
            category_id = category_name_to_id.get(category_name, 0)
            if category_id == 0:
                stats["unknown_category"] += 1
            score = float(np.asarray(scores[j]).reshape(-1)[0])
            bbox = normalize_bbox_to_pixels(boxes[j], width, height)
            extra = {"pseudo_mask_index": int(j)}
            source_anchor, source_anchor_iou = match_source_anchor(
                source_anchor_index,
                used_anchor_ids,
                im_id,
                category_id,
                bbox,
            )
            source_anchor_fit = annotation_to_fit(source_anchor) if source_anchor is not None else None
            extra["source_anchor_found"] = bool(source_anchor_fit is not None)
            extra["source_anchor_match_iou"] = float(source_anchor_iou)
            extra["source_anchor_id"] = int(source_anchor.get("id", -1)) if source_anchor is not None else -1

            cur_mask = mask[j]
            mask_source = "eroded"
            if (
                np.asarray(cur_mask).sum() < int(args.min_mask_pixels)
                and np.asarray(raw_mask[j]).sum() >= int(args.min_mask_pixels)
            ):
                cur_mask = raw_mask[j]
                mask_source = "raw_fallback"
            extra["dfu_mask_source"] = mask_source
            if np.asarray(cur_mask).sum() < int(args.min_mask_pixels):
                stats["invalid_small_mask"] += 1
                ann = build_invalid_annotation(
                    ann_id,
                    dataset_id,
                    im_id,
                    category_name,
                    category_id,
                    bbox,
                    score,
                    {"dfu_reason": "small_mask", "dfu_mask_source": mask_source},
                )
                ann = reference_match_update(ann, ref_index, args)
                annotations.append(ann)
                ann_id += 1
                continue

            clean_mask, edge_metrics = remove_depth_edges(depth, cur_mask, args)
            extra.update(edge_metrics)
            extra.update(gravity_metrics)
            raw_points = mask_to_points(depth, clean_mask, K)
            clustered_points, cluster_metrics = select_frustum_cluster(
                raw_points,
                clean_mask,
                K,
                args,
                rng,
            )
            extra.update(cluster_metrics)
            filtered_points, dfu_metrics = dfu_filter_points(clustered_points, category_name, args)
            extra.update(dfu_metrics)
            if filtered_points is None:
                stats["invalid_no_points"] += 1
                ann = build_invalid_annotation(
                    ann_id,
                    dataset_id,
                    im_id,
                    category_name,
                    category_id,
                    bbox,
                    score,
                    extra,
                )
                ann = reference_match_update(ann, ref_index, args)
                annotations.append(ann)
                ann_id += 1
                continue

            prior = SUNRGBD.get(category_name)
            if prior is None:
                stats["unknown_prior"] += 1
                prior = [0.5, 0.5, 0.5]

            fit, fit_metrics = estimate_bbox_dfu_robust(
                filtered_points,
                prior,
                category_name,
                ground_equ if has_ground else None,
                gravity_normal,
                args,
                rng,
            )
            extra.update(fit_metrics)
            extra["dfu_has_ground"] = bool(has_ground)
            extra["dfu_prior_whl"] = [float(x) for x in prior]
            if source_anchor_fit is not None:
                fit = source_anchor_fit
                extra["fit_source_before_surface"] = "source_geometry_anchor"
                stats["source_anchor_used"] += 1
            else:
                extra["fit_source_before_surface"] = "dfu_robust_pca"
                stats["source_anchor_missing"] += 1

            if fit is None:
                stats["invalid_fit_failed"] += 1
                ann = build_invalid_annotation(
                    ann_id,
                    dataset_id,
                    im_id,
                    category_name,
                    category_id,
                    bbox,
                    score,
                    extra,
                )
            else:
                fit, surface_metrics = optimize_box_surface_consistency(
                    fit,
                    filtered_points,
                    prior,
                    K,
                    bbox,
                    clean_mask,
                    ground_equ if has_ground else None,
                    args,
                )
                extra.update(surface_metrics)
                if (
                    args.use_latent_box_closure
                    and "pseudo_weight_joint" not in extra
                ):
                    fallback_weight = clamp_weight(0.35, args)
                    extra.update(
                        {
                            "latent_box_fallback": True,
                            "pseudo_weight_xy": fallback_weight,
                            "pseudo_weight_z": fallback_weight,
                            "pseudo_weight_dims": fallback_weight,
                            "pseudo_weight_pose": fallback_weight,
                            "pseudo_weight_joint": fallback_weight,
                            "pseudo_weight": fallback_weight,
                        }
                    )
                stats["valid3d"] += 1
                ann = build_valid_annotation(
                    ann_id,
                    dataset_id,
                    im_id,
                    category_name,
                    category_id,
                    bbox,
                    score,
                    fit,
                    extra,
                )
                if source_anchor is not None:
                    for box_key in ("bbox2D_tight", "bbox2D_trunc", "bbox2D_proj"):
                        source_box = source_anchor.get(box_key)
                        if source_box is not None and len(source_box) == 4:
                            ann[box_key] = [float(x) for x in source_box]

            ann = reference_match_update(ann, ref_index, args)
            if bool(ann.get("valid3D", True)):
                weight_values.append(float(ann.get("pseudo_weight", 1.0)))
            annotations.append(ann)
            ann_id += 1

    output = {
        "info": copy.deepcopy(source.get("info", {})),
        "images": images,
        "categories": categories,
        "annotations": annotations,
    }
    output.setdefault("info", {})
    output["info"]["pseudo_label_method"] = (
        "imov3d_depth_edge_dbscan_normal_surface_optimization"
        if (
            args.use_depth_edge_filter
            or args.use_frustum_dbscan
            or args.use_normal_ground_fusion
            or args.use_surface_box_optimization
        )
        else "dfu_robust_pca_with_boxer_consistency_weight"
    )
    output["info"]["pseudo_label_source_json"] = os.path.abspath(args.source_json)
    output["info"]["pseudo_label_cache_root"] = os.path.abspath(input_folder)
    output["info"]["imov3d_depth_edge_filter"] = bool(args.use_depth_edge_filter)
    output["info"]["imov3d_frustum_dbscan"] = bool(args.use_frustum_dbscan)
    output["info"]["imov3d_normal_ground_fusion"] = bool(args.use_normal_ground_fusion)
    output["info"]["surface_box_optimization"] = bool(args.use_surface_box_optimization)
    output["info"]["source_geometry_anchor"] = bool(args.use_source_geometry_anchor)
    output["info"]["surface_center_mode"] = str(args.surface_center_mode)
    if args.reference_json:
        output["info"]["pseudo_label_reference_json"] = os.path.abspath(args.reference_json)
    if weight_values:
        output["info"]["ng_mean_pseudo_weight"] = float(np.mean(weight_values))

    os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump(output, f)

    stats["images"] = len(images)
    stats["annotations"] = len(annotations)
    stats["valid3d"] = int(sum(1 for a in annotations if bool(a.get("valid3D", True))))
    stats["invalid3d"] = int(len(annotations) - stats["valid3d"])
    if weight_values:
        stats["pseudo_weight_mean_x1000"] = int(round(float(np.mean(weight_values)) * 1000))
        stats["pseudo_weight_p10_x1000"] = int(round(float(np.percentile(weight_values, 10)) * 1000))
        stats["pseudo_weight_p50_x1000"] = int(round(float(np.percentile(weight_values, 50)) * 1000))

    print(f"Wrote {args.output_json}")
    print(dict(stats))

    if args.stats_json:
        os.makedirs(os.path.dirname(args.stats_json) or ".", exist_ok=True)
        with open(args.stats_json, "w") as f:
            json.dump(dict(stats), f, indent=2)


if __name__ == "__main__":
    main()
