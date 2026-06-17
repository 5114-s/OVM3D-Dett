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

    parser.add_argument("--extent_percentile_low", type=float, default=2.0)
    parser.add_argument("--extent_percentile_high", type=float, default=98.0)
    parser.add_argument("--thin_extent_percentile_low", type=float, default=0.5)
    parser.add_argument("--thin_extent_percentile_high", type=float, default=99.5)
    parser.add_argument("--height_prior_ratio", type=float, default=0.5)
    parser.add_argument("--direct_prior_ratio", type=float, default=0.5)
    parser.add_argument("--ground_snap_distance", type=float, default=0.5)
    parser.add_argument("--prior_floor_ratio", type=float, default=0.12)
    parser.add_argument("--prior_ceiling_ratio", type=float, default=8.0)

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

    if ground_equ is not None and np.all(np.isfinite(ground_equ[:3])):
        ground_equ = np.asarray(ground_equ, dtype=np.float64).copy()
        if np.dot(np.array([0.0, -1.0, 0.0]), ground_equ[:3]) <= 0:
            ground_equ = -ground_equ
        new_ground_equ = np.array(
            [0.0, -1.0, 0.0, point_to_plane_distance(ground_equ, 0, 0, 0)],
            dtype=np.float64,
        )
        rotation_matrix = rotation_matrix_from_vectors_safe([0.0, -1.0, 0.0], ground_equ[:3])
        has_ground = True
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

        img_data = info[im_id]
        phrases = list(img_data.get("phrases", []))
        boxes = img_data.get("boxes", [])
        scores = img_data.get("conf", np.ones(len(phrases), dtype=np.float32))
        n = min(len(phrases), len(boxes), mask.shape[0], raw_mask.shape[0])
        stats["objects_seen"] += int(n)

        for j in range(n):
            category_name = str(phrases[j])
            category_id = category_name_to_id.get(category_name, 0)
            if category_id == 0:
                stats["unknown_category"] += 1
            score = float(np.asarray(scores[j]).reshape(-1)[0])
            bbox = normalize_bbox_to_pixels(boxes[j], width, height)
            extra = {}

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

            raw_points = mask_to_points(depth, cur_mask, K)
            filtered_points, dfu_metrics = dfu_filter_points(raw_points, category_name, args)
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
                args,
                rng,
            )
            extra.update(fit_metrics)
            extra["dfu_has_ground"] = bool(has_ground)
            extra["dfu_prior_whl"] = [float(x) for x in prior]

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
    output["info"]["pseudo_label_method"] = "dfu_robust_pca_with_boxer_consistency_weight"
    output["info"]["pseudo_label_source_json"] = os.path.abspath(args.source_json)
    output["info"]["pseudo_label_cache_root"] = os.path.abspath(input_folder)
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
