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

    parser.add_argument("--use_depth_aware_mask_selector", action="store_true")
    parser.add_argument("--mask_selector_depth_percentile_low", type=float, default=5.0)
    parser.add_argument("--mask_selector_depth_percentile_high", type=float, default=95.0)
    parser.add_argument("--mask_selector_mad_scale", type=float, default=2.5)
    parser.add_argument("--mask_selector_min_depth_window", type=float, default=0.05)
    parser.add_argument("--mask_selector_min_keep_ratio", type=float, default=0.18)
    parser.add_argument("--mask_selector_min_score_gain", type=float, default=0.03)
    parser.add_argument("--mask_selector_ring_dilate", type=int, default=5)
    parser.add_argument("--mask_selector_store_candidates", action="store_true")

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
    parser.add_argument("--surface_require_improvement", action="store_true", default=True)
    parser.add_argument("--no_surface_require_improvement", dest="surface_require_improvement", action="store_false")
    parser.add_argument("--surface_min_loss_gain", type=float, default=0.03)
    parser.add_argument("--surface_max_bbox_iou_drop", type=float, default=0.03)
    parser.add_argument("--surface_max_depth_worsen_ratio", type=float, default=1.10)
    parser.add_argument("--surface_max_support_drop", type=float, default=0.05)
    parser.add_argument("--surface_include_right_angle_yaws", action="store_true")
    parser.add_argument("--surface_enable_dims_swap", action="store_true")
    parser.add_argument("--use_latent_box_closure", action="store_true")
    parser.add_argument("--latent_topk", type=int, default=8)
    parser.add_argument("--latent_temperature", type=float, default=0.25)
    parser.add_argument("--latent_min_attribute_weight", type=float, default=0.15)
    parser.add_argument("--latent_max_attribute_weight", type=float, default=1.0)
    parser.add_argument("--latent_store_candidates", action="store_true")

    parser.add_argument("--reference_min_iou", type=float, default=0.10)
    parser.add_argument("--min_weight", type=float, default=0.35)
    parser.add_argument("--unmatched_weight", type=float, default=0.60)
    parser.add_argument("--use_imov3d_quality_weight", action="store_true")
    parser.add_argument("--imov3d_quality_min", type=float, default=0.65)
    parser.add_argument("--imov3d_prior_quality_strength", type=float, default=0.45)
    parser.add_argument("--imov3d_prior_error_cap", type=float, default=2.5)
    parser.add_argument("--imov3d_cluster_fallback_weight", type=float, default=0.88)
    parser.add_argument("--imov3d_normal_missing_weight", type=float, default=0.95)
    parser.add_argument("--use_projected_corner_depth_score", action="store_true")
    parser.add_argument("--pag_front_depth_percentile", type=float, default=35.0)
    parser.add_argument("--pag_corner_depth_radius", type=int, default=2)
    parser.add_argument("--pag_min_corner_depth_samples", type=int, default=2)
    parser.add_argument("--pag_min_score", type=float, default=0.60)
    parser.add_argument("--pag_apply_to_weight", action="store_true")
    parser.add_argument("--pag_weight_strength", type=float, default=0.35)
    parser.add_argument("--pag_store_projection", action="store_true")
    parser.add_argument("--use_locate3d_factorized_curriculum", action="store_true")
    parser.add_argument("--curriculum_xy_floor", type=float, default=0.75)
    parser.add_argument("--curriculum_z_floor", type=float, default=0.75)
    parser.add_argument("--curriculum_dims_floor", type=float, default=0.55)
    parser.add_argument("--curriculum_pose_floor", type=float, default=0.35)
    parser.add_argument("--curriculum_joint_floor", type=float, default=0.45)
    parser.add_argument("--use_bev_nms", action="store_true")
    parser.add_argument("--bev_nms_iou_threshold", type=float, default=0.45)
    parser.add_argument("--bev_nms_score_weight", type=float, default=0.50)
    parser.add_argument(
        "--use_external_strict_3d",
        action="store_true",
        help=(
            "Treat external 2D proposals, such as Detic+SAM2, as recall "
            "candidates: keep their 2D boxes but cap/down-weight noisy 3D "
            "supervision unless geometry evidence is strong."
        ),
    )
    parser.add_argument(
        "--external_strict_sources",
        default="detic,external_2d,detany3d",
        help="Comma-separated source names considered external for strict 3D gating.",
    )
    parser.add_argument("--external_strict_min_score", type=float, default=0.35)
    parser.add_argument("--external_strict_accept_quality", type=float, default=0.62)
    parser.add_argument("--external_strict_low_quality", type=float, default=0.35)
    parser.add_argument("--external_strict_high_cap", type=float, default=0.72)
    parser.add_argument("--external_strict_mid_cap", type=float, default=0.45)
    parser.add_argument("--external_strict_low_cap", type=float, default=0.18)
    parser.add_argument("--external_strict_xy_floor", type=float, default=0.35)
    parser.add_argument("--external_strict_z_floor", type=float, default=0.35)
    parser.add_argument("--external_strict_dims_floor", type=float, default=0.12)
    parser.add_argument("--external_strict_pose_floor", type=float, default=0.08)
    parser.add_argument(
        "--external_strict_mark_low_valid3d_false",
        action="store_true",
        help="For very weak external boxes, mark valid3D false instead of only down-weighting.",
    )
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


def largest_connected_component(mask_bool: np.ndarray) -> np.ndarray:
    mask_u8 = np.asarray(mask_bool, dtype=np.uint8)
    if int(mask_u8.sum()) == 0:
        return mask_u8.astype(bool)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, 8)
    if num_labels <= 1:
        return mask_u8.astype(bool)
    areas = stats[1:, cv2.CC_STAT_AREA]
    largest = int(np.argmax(areas)) + 1
    return labels == largest


def bbox_area_pixels(bbox: Sequence[float], h: int, w: int) -> float:
    x1, y1, x2, y2 = [float(v) for v in bbox]
    x1 = float(np.clip(x1, 0, max(w - 1, 0)))
    x2 = float(np.clip(x2, 0, max(w - 1, 0)))
    y1 = float(np.clip(y1, 0, max(h - 1, 0)))
    y2 = float(np.clip(y2, 0, max(h - 1, 0)))
    return max(1.0, (x2 - x1 + 1.0) * (y2 - y1 + 1.0))


def mask_depth_selector_metrics(
    depth: np.ndarray,
    mask_bool: np.ndarray,
    raw_mask_bool: np.ndarray,
    bbox: Sequence[float],
    args,
) -> Dict[str, object]:
    mask_bool = np.asarray(mask_bool).squeeze() > 0
    raw_mask_bool = np.asarray(raw_mask_bool).squeeze() > 0
    depth_np = np.asarray(depth, dtype=np.float32)
    valid = np.isfinite(depth_np) & (depth_np > 0.05)
    selected = mask_bool & valid
    pixel_count = int(mask_bool.sum())
    valid_count = int(selected.sum())
    raw_pixels = max(int(raw_mask_bool.sum()), 1)
    metrics: Dict[str, object] = {
        "pixels": pixel_count,
        "valid_pixels": valid_count,
        "keep_ratio": float(pixel_count / raw_pixels),
        "score": -1.0,
        "valid_ratio": 0.0,
        "compactness": 0.0,
        "inlier_ratio": 0.0,
        "area_score": 0.0,
        "separation_score": 0.5,
    }
    if pixel_count <= 0 or valid_count <= 0:
        return metrics

    values = depth_np[selected].astype(np.float64)
    median = float(np.median(values))
    q25, q75 = np.percentile(values, [25.0, 75.0])
    iqr = float(max(q75 - q25, 0.0))
    mad = float(np.median(np.abs(values - median)))
    sigma = 1.4826 * mad
    window = max(float(args.mask_selector_min_depth_window), float(args.mask_selector_mad_scale) * sigma)
    inlier_ratio = float(np.mean(np.abs(values - median) <= window))
    valid_ratio = float(valid_count / max(pixel_count, 1))
    compactness = float(math.exp(-iqr / max(0.12 * median, 0.05)))
    keep_ratio = float(pixel_count / raw_pixels)
    h, w = depth_np.shape[:2]
    bbox_area = bbox_area_pixels(bbox, h, w)
    fill_ratio = float(pixel_count / max(bbox_area, 1.0))
    # Prefer masks that are neither tiny fragments nor entire loose 2D boxes.
    area_score = float(
        np.clip(keep_ratio / max(float(args.mask_selector_min_keep_ratio), 1e-4), 0.0, 1.0)
        * np.clip(fill_ratio / 0.65, 0.0, 1.0)
    )

    separation_score = 0.5
    dilate = max(1, int(args.mask_selector_ring_dilate))
    kernel = np.ones((2 * dilate + 1, 2 * dilate + 1), dtype=np.uint8)
    ring = cv2.dilate(mask_bool.astype(np.uint8), kernel, iterations=1).astype(bool) & ~mask_bool
    ring_values = depth_np[ring & valid]
    if ring_values.size >= 8:
        bg_median = float(np.median(ring_values))
        # Foreground objects should usually be no farther than the local ring.
        separation_score = float(
            np.clip(0.5 + (bg_median - median) / max(0.35 * median, 0.10), 0.0, 1.0)
        )

    score = (
        0.28 * valid_ratio
        + 0.27 * compactness
        + 0.22 * inlier_ratio
        + 0.13 * area_score
        + 0.10 * separation_score
    )
    metrics.update(
        {
            "score": float(score),
            "valid_ratio": valid_ratio,
            "compactness": compactness,
            "inlier_ratio": inlier_ratio,
            "area_score": area_score,
            "separation_score": separation_score,
            "depth_median": median,
            "depth_iqr": iqr,
            "depth_mad": mad,
            "depth_window": float(window),
            "fill_ratio": fill_ratio,
        }
    )
    return metrics


def build_depth_core_mask(
    depth: np.ndarray,
    mask_bool: np.ndarray,
    args,
) -> np.ndarray:
    mask_bool = np.asarray(mask_bool).squeeze() > 0
    depth_np = np.asarray(depth, dtype=np.float32)
    valid = mask_bool & np.isfinite(depth_np) & (depth_np > 0.05)
    if int(valid.sum()) == 0:
        return np.zeros_like(mask_bool, dtype=bool)
    values = depth_np[valid].astype(np.float64)
    p_low, p_high = clamp_percentile_pair(
        args.mask_selector_depth_percentile_low,
        args.mask_selector_depth_percentile_high,
    )
    lo, hi = np.percentile(values, [p_low, p_high])
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    sigma = 1.4826 * mad
    window = max(float(args.mask_selector_min_depth_window), float(args.mask_selector_mad_scale) * sigma)
    lo = max(float(lo), median - window)
    hi = min(float(hi), median + window)
    core = valid & (depth_np >= lo) & (depth_np <= hi)
    if int(core.sum()) == 0:
        return core
    kernel = np.ones((3, 3), dtype=np.uint8)
    core = cv2.morphologyEx(core.astype(np.uint8), cv2.MORPH_OPEN, kernel, iterations=1).astype(bool)
    return core


def select_depth_aware_mask(
    depth: np.ndarray,
    eroded_mask: np.ndarray,
    raw_mask: np.ndarray,
    bbox: Sequence[float],
    args,
) -> Tuple[np.ndarray, Dict[str, object]]:
    raw_bool = np.asarray(raw_mask).squeeze() > 0
    eroded_bool = np.asarray(eroded_mask).squeeze() > 0
    metrics: Dict[str, object] = {
        "mask_selector_enabled": bool(args.use_depth_aware_mask_selector),
        "mask_selector_selected": "eroded",
        "mask_selector_replaced": False,
    }
    if not args.use_depth_aware_mask_selector:
        return eroded_bool.astype(np.float32), metrics

    candidates: List[Tuple[str, np.ndarray]] = []
    if int(eroded_bool.sum()) > 0:
        candidates.append(("eroded", eroded_bool))
        candidates.append(("eroded_lcc", largest_connected_component(eroded_bool)))
    if int(raw_bool.sum()) > 0:
        candidates.append(("raw", raw_bool))
        core = build_depth_core_mask(depth, raw_bool, args)
        if int(core.sum()) > 0:
            candidates.append(("depth_core", core))
            candidates.append(("depth_core_lcc", largest_connected_component(core)))

    if not candidates:
        metrics["mask_selector_reason"] = "no_candidates"
        return eroded_bool.astype(np.float32), metrics

    evaluated = []
    raw_pixels = max(int(raw_bool.sum()), 1)
    min_pixels = max(int(args.min_mask_pixels), int(round(raw_pixels * float(args.mask_selector_min_keep_ratio))))
    for name, candidate in candidates:
        candidate_bool = np.asarray(candidate).squeeze() > 0
        item = mask_depth_selector_metrics(depth, candidate_bool, raw_bool, bbox, args)
        item["name"] = name
        if int(item["pixels"]) < min_pixels:
            item["eligible"] = False
            item["reject_reason"] = "too_small"
        elif int(item["valid_pixels"]) < int(args.min_points):
            item["eligible"] = False
            item["reject_reason"] = "too_few_depth_pixels"
        else:
            item["eligible"] = True
            item["reject_reason"] = "ok"
        evaluated.append((name, candidate_bool, item))

    default_item = next((item for name, _, item in evaluated if name == "eroded"), evaluated[0][2])
    eligible = [entry for entry in evaluated if bool(entry[2].get("eligible", False))]
    if not eligible:
        metrics.update(
            {
                "mask_selector_reason": "no_eligible_candidate",
                "mask_selector_default_score": float(default_item.get("score", -1.0)),
                "mask_selector_candidates": len(evaluated),
            }
        )
        return eroded_bool.astype(np.float32), metrics

    best_name, best_mask, best_item = max(eligible, key=lambda entry: float(entry[2].get("score", -1.0)))
    default_score = float(default_item.get("score", -1.0))
    best_score = float(best_item.get("score", -1.0))
    replacement_ok = (
        best_name == "eroded"
        or best_score >= default_score + float(args.mask_selector_min_score_gain)
        or int(eroded_bool.sum()) < int(args.min_mask_pixels)
    )
    selected_name = best_name if replacement_ok else "eroded"
    selected_mask = best_mask if replacement_ok else eroded_bool
    selected_item = best_item if replacement_ok else default_item
    metrics.update(
        {
            "mask_selector_selected": selected_name,
            "mask_selector_replaced": bool(selected_name != "eroded"),
            "mask_selector_reason": "selected" if replacement_ok else "kept_eroded_by_gain_gate",
            "mask_selector_candidates": len(evaluated),
            "mask_selector_default_score": default_score,
            "mask_selector_best_score": best_score,
            "mask_selector_score": float(selected_item.get("score", -1.0)),
            "mask_selector_pixels": int(selected_item.get("pixels", 0)),
            "mask_selector_valid_pixels": int(selected_item.get("valid_pixels", 0)),
            "mask_selector_keep_ratio": float(selected_item.get("keep_ratio", 0.0)),
            "mask_selector_depth_iqr": float(selected_item.get("depth_iqr", 0.0)),
            "mask_selector_inlier_ratio": float(selected_item.get("inlier_ratio", 0.0)),
        }
    )
    if args.mask_selector_store_candidates:
        metrics["mask_selector_candidate_scores"] = [
            {
                "name": str(item["name"]),
                "score": float(item.get("score", -1.0)),
                "pixels": int(item.get("pixels", 0)),
                "valid_pixels": int(item.get("valid_pixels", 0)),
                "eligible": bool(item.get("eligible", False)),
                "reject_reason": str(item.get("reject_reason", "")),
            }
            for _, _, item in evaluated
        ]
    return selected_mask.astype(np.float32), metrics


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
    local_yaw_offsets = [-yaw_delta, 0.0, yaw_delta]
    yaw_anchors = [0.0]
    if bool(args.surface_include_right_angle_yaws):
        yaw_anchors.extend([-0.5 * math.pi, 0.5 * math.pi])
    yaw_values = sorted(
        set(
            float(anchor + local)
            for anchor in yaw_anchors
            for local in local_yaw_offsets
        )
    )
    dims_variants = [("base", dims_base.copy())]
    if bool(args.surface_enable_dims_swap):
        # Omni3D dimensions are [local-z width, local-y height, local-x length].
        # Swapping local x/z covers the common 90-degree yaw / length-width ambiguity.
        swapped = dims_base[[2, 1, 0]].copy()
        if not np.allclose(swapped, dims_base, rtol=1e-4, atol=1e-4):
            dims_variants.append(("width_length_swap", swapped))
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
    base_prior_error = float(
        np.mean(
            np.abs(
                np.log(
                    np.maximum(dims_base, 1e-4)
                    / np.maximum(prior_omni, 1e-4)
                )
            )
        )
    )
    base_ground_error = 0.0
    if ground_equ is not None:
        base_ground_error = min(
            float(
                min(
                    point_to_plane_distance(ground_equ, *point)
                    for point in np.asarray(vertices, dtype=np.float64)
                )
            ),
            1.0,
        )
    base_loss = (
        float(args.surface_projection_weight) * (1.0 - base_bbox_iou)
        + float(args.surface_silhouette_weight) * (1.0 - base_silhouette_iou)
        + float(args.surface_depth_weight) * min(base_depth_error, 2.0)
        + float(args.surface_support_weight) * (1.0 - base_support)
        + float(args.surface_prior_weight) * base_prior_error
        + 0.10 * base_ground_error
    )
    base_candidate = {
        "loss": float(base_loss),
        "vertices": np.asarray(vertices, dtype=np.float64),
        "center": center_base,
        "dims": dims_base,
        "R": R_base,
        "bbox_iou": float(base_bbox_iou),
        "silhouette_iou": float(base_silhouette_iou),
        "depth_error": float(base_depth_error),
        "support": float(base_support),
        "prior_error": float(base_prior_error),
        "front_depth": float(base_front),
        "depth_blend": 0.0,
        "xy_blend": 0.0,
        "scale_dims": [1.0, 1.0, 1.0],
        "dims_variant": "base",
        "yaw_offset": 0.0,
        "is_base": True,
    }
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
            "surface_opt_base_prior_log_error": float(base_prior_error),
            "surface_opt_base_loss": float(base_loss),
            "surface_opt_min_point_support": float(min_support),
        }
    )

    best = base_candidate
    valid_candidates = [base_candidate] if args.use_latent_box_closure else []
    metrics["surface_opt_candidates"] = 1
    metrics["surface_opt_yaw_family_count"] = int(len(yaw_values))
    metrics["surface_opt_dims_variant_count"] = int(len(dims_variants))
    for dims_variant_name, dims_seed in dims_variants:
        for yaw_offset in yaw_values:
            R_candidate = R_base @ rotate_y(yaw_offset)
            for scale_w in scale_values:
                for scale_h in height_scale_values:
                    for scale_l in scale_values:
                        dims_candidate = dims_seed * np.asarray(
                            [scale_w, scale_h, scale_l],
                            dtype=np.float64,
                        )
                        for depth_blend in depth_blends:
                            for xy_blend in xy_blends:
                                center_candidate = center_base.copy()
                                center_candidate[:2] += (xy_target - center_candidate[:2]) * xy_blend
                                center_candidate[2] += raw_shift * depth_blend
                                is_base_candidate = (
                                    dims_variant_name == "base"
                                    and abs(float(yaw_offset)) <= 1e-9
                                    and abs(float(scale_w) - 1.0) <= 1e-9
                                    and abs(float(scale_h) - 1.0) <= 1e-9
                                    and abs(float(scale_l) - 1.0) <= 1e-9
                                    and abs(float(depth_blend)) <= 1e-9
                                    and abs(float(xy_blend)) <= 1e-9
                                )
                                if is_base_candidate:
                                    continue
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
                                    "dims_variant": dims_variant_name,
                                    "yaw_offset": yaw_offset,
                                    "is_base": False,
                                }
                                if args.use_latent_box_closure:
                                    valid_candidates.append(candidate)
                                if best is None or loss < best["loss"]:
                                    best = candidate

    chosen = best if best is not None else base_candidate
    gate_reason = "valid"
    if (
        bool(args.surface_require_improvement)
        and not bool(chosen.get("is_base", False))
    ):
        loss_gain = float(base_candidate["loss"] - chosen["loss"])
        bbox_ok = (
            float(chosen["bbox_iou"])
            >= float(base_candidate["bbox_iou"]) - float(args.surface_max_bbox_iou_drop)
        )
        depth_ok = (
            float(chosen["depth_error"])
            <= float(base_candidate["depth_error"]) * float(args.surface_max_depth_worsen_ratio)
            + 1e-6
        )
        support_ok = (
            float(chosen["support"])
            >= float(base_candidate["support"]) - float(args.surface_max_support_drop)
        )
        gain_ok = loss_gain >= float(args.surface_min_loss_gain)
        if not (gain_ok and bbox_ok and depth_ok and support_ok):
            chosen = base_candidate
            gate_reason = "kept_base_by_improvement_gate"

    metrics.update(
        {
            "surface_opt_applied": bool(
                abs(float(chosen["depth_blend"])) > 1e-6
                or abs(float(chosen["xy_blend"])) > 1e-6
                or any(abs(float(v) - 1.0) > 1e-6 for v in chosen["scale_dims"])
                or str(chosen.get("dims_variant", "base")) != "base"
                or abs(float(chosen["yaw_offset"])) > 1e-6
            ),
            "surface_opt_reason": gate_reason,
            "surface_opt_best_loss": float(best["loss"]) if best is not None else float(base_loss),
            "surface_opt_loss": float(chosen["loss"]),
            "surface_opt_loss_gain": float(base_candidate["loss"] - chosen["loss"]),
            "surface_opt_bbox_iou": float(chosen["bbox_iou"]),
            "surface_opt_silhouette_iou": float(chosen["silhouette_iou"]),
            "surface_opt_depth_rel_error": float(chosen["depth_error"]),
            "surface_opt_point_support": float(chosen["support"]),
            "surface_opt_prior_log_error": float(chosen["prior_error"]),
            "surface_opt_observed_depth": observed_surface,
            "surface_opt_front_depth": float(chosen["front_depth"]),
            "surface_opt_depth_blend": float(chosen["depth_blend"]),
            "surface_opt_xy_blend": float(chosen["xy_blend"]),
            "surface_opt_scale_dims": [float(v) for v in chosen["scale_dims"]],
            "surface_opt_dims_variant": str(chosen.get("dims_variant", "base")),
            "surface_opt_yaw_offset": float(chosen["yaw_offset"]),
        }
    )
    if args.use_latent_box_closure:
        metrics.update(latent_candidate_statistics(valid_candidates, args))
    return (
        chosen["vertices"].astype(np.float32),
        chosen["center"].astype(np.float32),
        chosen["dims"].astype(np.float32),
        chosen["R"].astype(np.float32),
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


def finite_float(value, default: float = 0.0) -> float:
    try:
        value_f = float(value)
    except Exception:
        return float(default)
    return value_f if math.isfinite(value_f) else float(default)


def sanitize_detection_score(score, default: float = 0.05) -> float:
    return float(np.clip(finite_float(score, default), 0.0, 1.0))


def split_external_sources(value: str) -> set:
    return {item.strip().lower() for item in str(value).split(",") if item.strip()}


def is_external_proposal(ann: dict, args) -> bool:
    if bool(ann.get("proposal_external", False)):
        return True
    source = str(ann.get("proposal_source", "")).lower().strip()
    return source in split_external_sources(args.external_strict_sources)


def external_strict_quality(ann: dict, args) -> Tuple[float, Dict[str, float]]:
    score = sanitize_detection_score(ann.get("score", ann.get("proposal_2d_score", 0.0)))
    min_score = float(np.clip(args.external_strict_min_score, 0.0, 0.99))
    score_conf = float(np.clip((score - min_score) / max(1.0 - min_score, 1e-6), 0.0, 1.0))

    support = max(
        finite_float(ann.get("surface_opt_point_support", 0.0)),
        finite_float(ann.get("fit_inside_ratio", 0.0)),
        finite_float(ann.get("ng_dfu_box_support", 0.0)),
    )
    support_conf = float(np.clip(support, 0.0, 1.0))

    depth_error = min(
        finite_float(ann.get("surface_opt_depth_rel_error", 1.0), 1.0),
        finite_float(ann.get("ng_depth_rel_error", 1.0), 1.0),
    )
    depth_conf = float(np.exp(-min(max(depth_error, 0.0), 2.0) / 0.45))

    projection = max(
        finite_float(ann.get("surface_opt_bbox_iou", 0.0)),
        finite_float(ann.get("surface_opt_silhouette_iou", 0.0)),
        finite_float(ann.get("ng_match_iou_2d", 0.0)),
    )
    projection_conf = float(np.clip(projection, 0.0, 1.0))

    prior_error = min(
        finite_float(ann.get("surface_opt_prior_log_error", 2.0), 2.0),
        finite_float(ann.get("fit_prior_log_error", 2.0), 2.0),
        finite_float(ann.get("ng_dim_log_error", 2.0), 2.0),
    )
    prior_conf = float(np.exp(-min(max(prior_error, 0.0), 2.5) / 1.1))

    source_anchor_conf = 1.0 if bool(ann.get("source_anchor_found", False)) else 0.65
    if bool(ann.get("ng_match_found", False)):
        source_anchor_conf = max(source_anchor_conf, 0.85)

    quality = float(
        np.clip(
            0.23 * score_conf
            + 0.25 * support_conf
            + 0.22 * depth_conf
            + 0.15 * projection_conf
            + 0.10 * prior_conf
            + 0.05 * source_anchor_conf,
            0.0,
            1.0,
        )
    )
    details = {
        "external_strict_score_conf": score_conf,
        "external_strict_support_conf": support_conf,
        "external_strict_depth_conf": depth_conf,
        "external_strict_projection_conf": projection_conf,
        "external_strict_prior_conf": prior_conf,
        "external_strict_source_anchor_conf": source_anchor_conf,
    }
    return quality, details


def cap_factorized_weights(
    ann: dict,
    *,
    joint_cap: float,
    xy_cap: float,
    z_cap: float,
    dims_cap: float,
    pose_cap: float,
    args,
    min_floor: float = 0.05,
) -> dict:
    caps = {
        "pseudo_weight_xy": xy_cap,
        "pseudo_weight_z": z_cap,
        "pseudo_weight_dims": dims_cap,
        "pseudo_weight_pose": pose_cap,
        "pseudo_weight_joint": joint_cap,
        "pseudo_weight": joint_cap,
    }
    for key, cap in caps.items():
        current = finite_float(ann.get(key, ann.get("pseudo_weight", 1.0)), 1.0)
        ann[key] = float(np.clip(min(current, float(cap)), float(min_floor), 1.0))
    return ann


def external_strict_3d_update(ann: dict, args, stats=None) -> dict:
    if not bool(args.use_external_strict_3d):
        return ann
    if not is_external_proposal(ann, args):
        return ann

    ann["external_strict_3d_enabled"] = True
    if stats is not None:
        stats["external_strict_seen"] += 1

    if not bool(ann.get("valid3D", True)):
        ann["pseudo_weight"] = float(args.min_weight)
        ann["pseudo_weight_joint"] = float(args.min_weight)
        if stats is not None:
            stats["external_strict_invalid"] += 1
        return ann

    quality, details = external_strict_quality(ann, args)
    ann.update(details)
    ann["external_strict_quality"] = quality

    accept_quality = float(args.external_strict_accept_quality)
    low_quality = float(args.external_strict_low_quality)
    if quality >= accept_quality:
        tier = "accepted_3d"
        joint_cap = float(args.external_strict_high_cap)
        xy_cap = max(joint_cap, float(args.external_strict_xy_floor))
        z_cap = max(joint_cap, float(args.external_strict_z_floor))
        dims_cap = joint_cap
        pose_cap = min(joint_cap, 0.60)
        stat_key = "external_strict_accepted"
    elif quality >= low_quality:
        tier = "weak_3d"
        joint_cap = float(args.external_strict_mid_cap)
        xy_cap = max(joint_cap, float(args.external_strict_xy_floor))
        z_cap = max(joint_cap, float(args.external_strict_z_floor))
        dims_cap = min(joint_cap, 0.30)
        pose_cap = min(joint_cap, 0.20)
        stat_key = "external_strict_weak"
    else:
        tier = "very_weak_3d"
        if bool(args.external_strict_mark_low_valid3d_false):
            ann["valid3D"] = False
        joint_cap = float(args.external_strict_low_cap)
        xy_cap = max(joint_cap, float(args.external_strict_xy_floor))
        z_cap = max(joint_cap, float(args.external_strict_z_floor))
        dims_cap = max(float(args.external_strict_dims_floor), joint_cap)
        pose_cap = max(float(args.external_strict_pose_floor), min(joint_cap, 0.15))
        stat_key = "external_strict_very_weak"

    ann["external_strict_tier"] = tier
    cap_factorized_weights(
        ann,
        joint_cap=joint_cap,
        xy_cap=xy_cap,
        z_cap=z_cap,
        dims_cap=dims_cap,
        pose_cap=pose_cap,
        args=args,
    )
    if stats is not None:
        stats[stat_key] += 1
    return ann


def imov3d_quality_weight_update(ann: dict, args) -> dict:
    """Conservatively fold ImOV3D-style revision quality into pseudo weights.

    The reference/ng weight still carries the main supervision confidence. This
    extra factor only down-weights boxes whose revised geometry is suspicious:
    size far from class prior, failed clustering, missing gravity evidence, or
    weak projection/depth/surface support.
    """
    if not bool(args.use_imov3d_quality_weight):
        return ann

    if not bool(ann.get("valid3D", True)):
        ann["imov3d_quality_weight"] = float(args.min_weight)
        return ann

    prior_error = float(
        ann.get(
            "surface_opt_prior_log_error",
            ann.get("fit_prior_log_error", 0.0),
        )
    )
    prior_conf = float(
        math.exp(
            -float(args.imov3d_prior_quality_strength)
            * min(max(prior_error, 0.0), float(args.imov3d_prior_error_cap))
        )
    )

    cluster_conf = 1.0
    if bool(ann.get("dbscan_enabled", False)):
        if bool(ann.get("dbscan_fallback", False)):
            cluster_conf = float(args.imov3d_cluster_fallback_weight)
        else:
            selected_score = float(ann.get("dbscan_selected_score", 0.5))
            keep_ratio = float(ann.get("dbscan_keep_ratio", 1.0))
            cluster_conf = float(np.clip(0.72 + 0.20 * selected_score + 0.08 * keep_ratio, 0.55, 1.0))

    normal_conf = 1.0
    if bool(args.use_normal_ground_fusion):
        if bool(ann.get("normal_gravity_available", False)):
            normal_conf = float(
                np.clip(
                    0.88 + 0.12 * float(ann.get("normal_gravity_confidence", 0.0)),
                    0.82,
                    1.0,
                )
            )
        else:
            normal_conf = float(args.imov3d_normal_missing_weight)

    mask_conf = 1.0
    if bool(ann.get("mask_selector_enabled", False)):
        mask_conf = float(
            np.clip(0.70 + 0.30 * float(ann.get("mask_selector_score", 0.5)), 0.60, 1.0)
        )

    surface_conf = 1.0
    if "surface_opt_depth_rel_error" in ann:
        depth_conf = math.exp(-min(float(ann.get("surface_opt_depth_rel_error", 0.0)), 2.0))
        support_conf = float(np.clip(float(ann.get("surface_opt_point_support", 0.5)), 0.0, 1.0))
        silhouette_conf = float(np.clip(float(ann.get("surface_opt_silhouette_iou", 0.0)), 0.0, 1.0))
        surface_conf = float(
            np.clip(0.35 * depth_conf + 0.45 * support_conf + 0.20 * silhouette_conf, 0.50, 1.0)
        )

    quality = float(
        np.clip(
            prior_conf * cluster_conf * normal_conf * mask_conf * surface_conf,
            float(args.imov3d_quality_min),
            1.0,
        )
    )
    ann["imov3d_prior_confidence"] = prior_conf
    ann["imov3d_cluster_confidence"] = cluster_conf
    ann["imov3d_normal_confidence"] = normal_conf
    ann["imov3d_mask_confidence"] = mask_conf
    ann["imov3d_surface_confidence"] = surface_conf
    ann["imov3d_quality_weight"] = quality

    base_weight = float(ann.get("pseudo_weight", 1.0))
    ann["pseudo_weight_before_imov3d_quality"] = base_weight
    ann["pseudo_weight"] = float(np.clip(base_weight * quality, float(args.min_weight), 1.0))

    # Make the prior mainly affect dimensions/yaw while keeping center/depth
    # supervision relatively stable.
    for key in ("pseudo_weight_dims", "pseudo_weight_pose", "pseudo_weight_joint"):
        if key in ann:
            ann[key] = float(np.clip(float(ann[key]) * quality, float(args.min_weight), 1.0))
    for key in ("pseudo_weight_xy", "pseudo_weight_z"):
        if key in ann:
            relaxed = 0.5 + 0.5 * quality
            ann[key] = float(np.clip(float(ann[key]) * relaxed, float(args.min_weight), 1.0))
    return ann


def safe_bbox_iou_np(box_a: Sequence[float], box_b: Sequence[float]) -> float:
    if bbox_iou is not None:
        return float(bbox_iou(box_a, box_b))
    ax1, ay1, ax2, ay2 = [float(v) for v in box_a]
    bx1, by1, bx2, by2 = [float(v) for v in box_b]
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(ix2 - ix1, 0.0), max(iy2 - iy1, 0.0)
    inter = iw * ih
    area_a = max(ax2 - ax1, 0.0) * max(ay2 - ay1, 0.0)
    area_b = max(bx2 - bx1, 0.0) * max(by2 - by1, 0.0)
    denom = area_a + area_b - inter
    return float(inter / denom) if denom > 1e-12 else 0.0


def sample_depth_patch(
    depth: np.ndarray,
    mask: Optional[np.ndarray],
    u: float,
    v: float,
    radius: int,
) -> Optional[float]:
    depth_np = np.asarray(depth, dtype=np.float32)
    h, w = depth_np.shape[:2]
    ui = int(round(float(u)))
    vi = int(round(float(v)))
    if ui < 0 or ui >= w or vi < 0 or vi >= h:
        return None
    radius = max(0, int(radius))
    x1, x2 = max(ui - radius, 0), min(ui + radius + 1, w)
    y1, y2 = max(vi - radius, 0), min(vi + radius + 1, h)
    patch = depth_np[y1:y2, x1:x2]
    valid = np.isfinite(patch) & (patch > 0.05)
    if mask is not None:
        mask_np = np.asarray(mask).squeeze() > 0
        if mask_np.shape[:2] == depth_np.shape[:2]:
            valid &= mask_np[y1:y2, x1:x2]
    values = patch[valid]
    if values.size == 0:
        return None
    return float(np.percentile(values.astype(np.float64), 35.0))


def projected_corner_depth_score_update(
    ann: dict,
    depth: np.ndarray,
    target_mask: Optional[np.ndarray],
    target_bbox: Sequence[float],
    K,
    points_cam: Optional[np.ndarray],
    prior: Sequence[float],
    args,
) -> dict:
    """Add MoCA3D/PAG-style image-plane geometry evidence to one annotation.

    This does not remove labels. It stores projected cuboid corners, front-depth
    agreement, and a pixel-aligned geometry score that later becomes a soft
    pseudo-label weight.
    """
    if not bool(args.use_projected_corner_depth_score):
        return ann
    if not bool(ann.get("valid3D", True)):
        ann["pag_score"] = float(args.min_weight)
        return ann

    try:
        vertices = np.asarray(ann["bbox3D_cam"], dtype=np.float64).reshape(8, 3)
        center = np.asarray(ann["center_cam"], dtype=np.float64).reshape(3)
        dims = np.asarray(ann["dimensions"], dtype=np.float64).reshape(3)
        rotation = np.asarray(ann["R_cam"], dtype=np.float64).reshape(3, 3)
        K_np = np.asarray(K, dtype=np.float64).reshape(3, 3)
    except Exception:
        return ann
    if (
        not np.all(np.isfinite(vertices))
        or np.any(vertices[:, 2] <= 1e-4)
        or not np.all(np.isfinite(K_np))
    ):
        ann["pag_score"] = float(args.min_weight)
        return ann

    projected = vertices @ K_np.T
    uv = projected[:, :2] / projected[:, 2:3]
    if not np.all(np.isfinite(uv)):
        ann["pag_score"] = float(args.min_weight)
        return ann

    depth_np = np.asarray(depth, dtype=np.float32)
    h, w = depth_np.shape[:2]
    in_image = (
        (uv[:, 0] >= 0.0)
        & (uv[:, 0] < float(w))
        & (uv[:, 1] >= 0.0)
        & (uv[:, 1] < float(h))
    )
    in_image_ratio = float(np.mean(in_image))
    projected_bbox = [
        float(np.min(uv[:, 0])),
        float(np.min(uv[:, 1])),
        float(np.max(uv[:, 0])),
        float(np.max(uv[:, 1])),
    ]
    projection_iou = safe_bbox_iou_np(projected_bbox, target_bbox)
    _bbox_score, silhouette_iou = projected_box_metrics(
        vertices,
        K_np,
        target_bbox,
        target_mask,
    )

    mask_bool = (
        np.asarray(target_mask).squeeze() > 0
        if target_mask is not None
        else np.zeros(depth_np.shape[:2], dtype=bool)
    )
    valid_depth = np.isfinite(depth_np) & (depth_np > 0.05)
    masked_values = depth_np[mask_bool & valid_depth]
    if masked_values.size >= 4:
        observed_front_depth = float(
            np.percentile(
                masked_values.astype(np.float64),
                np.clip(float(args.pag_front_depth_percentile), 1.0, 50.0),
            )
        )
    else:
        observed_front_depth = float(np.percentile(vertices[:, 2], 10.0))
    box_front_depth = float(np.min(vertices[:, 2]))
    front_depth_rel_error = abs(box_front_depth - observed_front_depth) / max(
        observed_front_depth,
        0.10,
    )
    front_depth_conf = float(math.exp(-2.5 * min(front_depth_rel_error, 2.0)))

    front_threshold = box_front_depth + max(0.08, 0.04 * box_front_depth)
    front_corner_indices = np.where(vertices[:, 2] <= front_threshold)[0]
    corner_errors = []
    corner_depth_samples = []
    radius = int(args.pag_corner_depth_radius)
    for corner_idx in front_corner_indices:
        if not in_image[corner_idx]:
            continue
        observed = sample_depth_patch(
            depth_np,
            mask_bool,
            uv[corner_idx, 0],
            uv[corner_idx, 1],
            radius,
        )
        if observed is None:
            continue
        predicted = float(vertices[corner_idx, 2])
        rel_error = abs(predicted - observed) / max(observed, 0.10)
        corner_errors.append(float(rel_error))
        corner_depth_samples.append(
            {
                "corner": int(corner_idx),
                "u": float(uv[corner_idx, 0]),
                "v": float(uv[corner_idx, 1]),
                "pred_depth": predicted,
                "obs_depth": observed,
                "rel_error": float(rel_error),
            }
        )

    if len(corner_errors) >= int(args.pag_min_corner_depth_samples):
        corner_depth_rel_error = float(np.median(corner_errors))
        corner_depth_conf = float(math.exp(-3.0 * min(corner_depth_rel_error, 2.0)))
    else:
        corner_depth_rel_error = front_depth_rel_error
        # Missing projected corner samples should not kill the label; fall back
        # to front-surface evidence but mark the evidence as weaker.
        corner_depth_conf = float(0.75 * front_depth_conf)

    support_conf = float(
        np.clip(
            ann.get(
                "surface_opt_point_support",
                point_support_3d(points_cam, center, dims, rotation) if points_cam is not None else 0.5,
            ),
            0.0,
            1.0,
        )
    )
    prior_error = float(
        ann.get("surface_opt_prior_log_error", ann.get("fit_prior_log_error", 0.0))
    )
    prior_conf = float(math.exp(-min(max(prior_error, 0.0), 3.0)))
    if prior is not None and len(prior) == 3:
        prior_arr = np.asarray([prior[0], prior[1], prior[2]], dtype=np.float64)
        dims_error = float(
            np.mean(
                np.abs(
                    np.log(
                        np.maximum(dims, 1e-4)
                        / np.maximum(prior_arr, 1e-4)
                    )
                )
            )
        )
        prior_conf = max(prior_conf, float(math.exp(-min(dims_error, 3.0))))

    pag_score_raw = float(
        np.clip(
            0.22 * projection_iou
            + 0.18 * silhouette_iou
            + 0.24 * front_depth_conf
            + 0.16 * corner_depth_conf
            + 0.12 * support_conf
            + 0.05 * in_image_ratio
            + 0.03 * prior_conf,
            0.0,
            1.0,
        )
    )
    pag_score = float(
        np.clip(pag_score_raw, float(args.pag_min_score), 1.0)
    )

    ann.update(
        {
            "moca3d_projected_corner_depth_enabled": True,
            "moca3d_projected_corner_depth_score": float(pag_score),
            "moca3d_projected_bbox": [float(v) for v in projected_bbox],
            "moca3d_projection_iou": float(projection_iou),
            "moca3d_silhouette_iou": float(silhouette_iou),
            "moca3d_corner_in_image_ratio": float(in_image_ratio),
            "moca3d_box_front_depth": float(box_front_depth),
            "moca3d_observed_front_depth": float(observed_front_depth),
            "moca3d_front_depth_rel_error": float(front_depth_rel_error),
            "moca3d_front_depth_confidence": float(front_depth_conf),
            "moca3d_corner_depth_rel_error": float(corner_depth_rel_error),
            "moca3d_corner_depth_confidence": float(corner_depth_conf),
            "moca3d_corner_depth_samples": int(len(corner_errors)),
            "moca3d_support_confidence": float(support_conf),
            "moca3d_prior_confidence": float(prior_conf),
            "pag_score_raw": float(pag_score_raw),
            "pag_score": float(pag_score),
        }
    )
    if bool(args.pag_store_projection):
        ann["moca3d_projected_corners_2d"] = [
            [float(v) for v in row] for row in uv.tolist()
        ]
        ann["moca3d_corner_depths"] = [float(v) for v in vertices[:, 2].tolist()]
        ann["moca3d_front_corner_depth_samples"] = corner_depth_samples

    if bool(args.pag_apply_to_weight):
        strength = float(np.clip(args.pag_weight_strength, 0.0, 1.0))
        factor = float((1.0 - strength) + strength * pag_score)
        base_weight = float(ann.get("pseudo_weight", 1.0))
        ann["pseudo_weight_before_pag"] = base_weight
        ann["pseudo_weight"] = float(
            np.clip(base_weight * factor, float(args.min_weight), 1.0)
        )
        relaxed = 0.5 + 0.5 * factor
        for key in ("pseudo_weight_xy", "pseudo_weight_z"):
            if key in ann:
                ann[key] = float(
                    np.clip(float(ann[key]) * relaxed, float(args.min_weight), 1.0)
                )
        for key in ("pseudo_weight_dims", "pseudo_weight_pose", "pseudo_weight_joint"):
            if key in ann:
                ann[key] = float(
                    np.clip(float(ann[key]) * factor, float(args.min_weight), 1.0)
                )
    return ann


def locate3d_factorized_curriculum_update(ann: dict, args) -> dict:
    """LocateAnything3D-style chain weighting for pseudo supervision.

    Center/depth are allowed to remain strong when 2D/depth evidence is stable;
    dimensions are moderated by silhouette/prior/support; yaw is the most
    conservative factor and is lowered when projected geometry is weak.
    """
    if not bool(args.use_locate3d_factorized_curriculum):
        return ann
    if not bool(ann.get("valid3D", True)):
        return ann

    base = float(np.clip(float(ann.get("pseudo_weight", 1.0)), float(args.min_weight), 1.0))
    pag = float(np.clip(float(ann.get("pag_score", ann.get("pag_score_raw", 1.0))), 0.0, 1.0))
    projection = float(
        np.clip(
            ann.get("moca3d_projection_iou", ann.get("surface_opt_bbox_iou", pag)),
            0.0,
            1.0,
        )
    )
    silhouette = float(
        np.clip(
            ann.get("moca3d_silhouette_iou", ann.get("surface_opt_silhouette_iou", pag)),
            0.0,
            1.0,
        )
    )
    depth_conf = float(
        np.clip(
            ann.get("moca3d_front_depth_confidence", math.exp(-min(float(ann.get("surface_opt_depth_rel_error", 0.0)), 2.0))),
            0.0,
            1.0,
        )
    )
    corner_depth_conf = float(
        np.clip(ann.get("moca3d_corner_depth_confidence", depth_conf), 0.0, 1.0)
    )
    support = float(
        np.clip(
            ann.get("surface_opt_point_support", ann.get("fit_inside_ratio", 0.5)),
            0.0,
            1.0,
        )
    )
    prior_conf = float(
        np.clip(
            ann.get("moca3d_prior_confidence", ann.get("imov3d_prior_confidence", 1.0)),
            0.0,
            1.0,
        )
    )

    xy_conf = float(
        np.clip(
            0.35 * projection + 0.25 * silhouette + 0.25 * pag + 0.15 * support,
            float(args.curriculum_xy_floor),
            1.0,
        )
    )
    z_conf = float(
        np.clip(
            0.45 * depth_conf + 0.25 * corner_depth_conf + 0.20 * pag + 0.10 * support,
            float(args.curriculum_z_floor),
            1.0,
        )
    )
    dims_conf = float(
        np.clip(
            0.25 * silhouette + 0.25 * support + 0.25 * prior_conf + 0.25 * pag,
            float(args.curriculum_dims_floor),
            1.0,
        )
    )
    pose_conf = float(
        np.clip(
            0.35 * silhouette + 0.20 * support + 0.20 * prior_conf + 0.25 * pag,
            float(args.curriculum_pose_floor),
            1.0,
        )
    )

    base_boost = 0.5 + 0.5 * base
    prev = {
        "xy": float(ann.get("pseudo_weight_xy", base)),
        "z": float(ann.get("pseudo_weight_z", base)),
        "dims": float(ann.get("pseudo_weight_dims", base)),
        "pose": float(ann.get("pseudo_weight_pose", base)),
        "joint": float(ann.get("pseudo_weight_joint", base)),
    }
    xy_weight = max(prev["xy"], xy_conf * base_boost)
    z_weight = max(prev["z"], z_conf * base_boost)
    dims_weight = min(prev["dims"], max(float(args.curriculum_dims_floor), dims_conf * base_boost))
    pose_weight = min(prev["pose"], max(float(args.curriculum_pose_floor), pose_conf * base_boost))
    joint_weight = float(
        np.exp(
            np.mean(
                np.log(
                    np.maximum(
                        [xy_weight, z_weight, dims_weight, pose_weight],
                        1e-4,
                    )
                )
            )
        )
    )
    joint_weight = float(
        np.clip(joint_weight, float(args.curriculum_joint_floor), 1.0)
    )

    ann.update(
        {
            "locate3d_factorized_curriculum": True,
            "locate3d_curriculum_xy_conf": xy_conf,
            "locate3d_curriculum_z_conf": z_conf,
            "locate3d_curriculum_dims_conf": dims_conf,
            "locate3d_curriculum_pose_conf": pose_conf,
            "pseudo_weight_before_locate3d_curriculum": base,
            "pseudo_weight_xy": float(np.clip(xy_weight, float(args.min_weight), 1.0)),
            "pseudo_weight_z": float(np.clip(z_weight, float(args.min_weight), 1.0)),
            "pseudo_weight_dims": float(np.clip(dims_weight, float(args.min_weight), 1.0)),
            "pseudo_weight_pose": float(np.clip(pose_weight, float(args.min_weight), 1.0)),
            "pseudo_weight_joint": float(np.clip(joint_weight, float(args.min_weight), 1.0)),
            "pseudo_weight": float(np.clip(joint_weight, float(args.min_weight), 1.0)),
        }
    )
    return ann


def ann_bev_box(ann: dict) -> Optional[np.ndarray]:
    if not bool(ann.get("valid3D", True)):
        return None
    try:
        vertices = np.asarray(ann["bbox3D_cam"], dtype=np.float64).reshape(8, 3)
    except Exception:
        return None
    if not np.all(np.isfinite(vertices)):
        return None
    x_min, z_min = vertices[:, [0, 2]].min(axis=0)
    x_max, z_max = vertices[:, [0, 2]].max(axis=0)
    if x_max <= x_min or z_max <= z_min:
        return None
    return np.asarray([x_min, z_min, x_max, z_max], dtype=np.float64)


def bev_iou_np(box_a: np.ndarray, box_b: np.ndarray) -> float:
    inter_min = np.maximum(box_a[:2], box_b[:2])
    inter_max = np.minimum(box_a[2:], box_b[2:])
    inter_size = np.maximum(inter_max - inter_min, 0.0)
    inter = float(inter_size[0] * inter_size[1])
    area_a = float(max(box_a[2] - box_a[0], 0.0) * max(box_a[3] - box_a[1], 0.0))
    area_b = float(max(box_b[2] - box_b[0], 0.0) * max(box_b[3] - box_b[1], 0.0))
    denom = area_a + area_b - inter
    return float(inter / denom) if denom > 1e-12 else 0.0


def annotation_nms_quality(ann: dict, args) -> float:
    score = sanitize_detection_score(ann.get("score", 1.0), default=0.05)
    weight = float(
        np.clip(
            float(ann.get("pseudo_weight_joint", ann.get("pseudo_weight", 1.0))),
            0.05,
            1.0,
        )
    )
    alpha = float(np.clip(args.bev_nms_score_weight, 0.0, 1.0))
    quality = score * ((1.0 - alpha) + alpha * weight)
    if "surface_opt_loss" in ann:
        quality *= float(1.0 / (1.0 + max(float(ann["surface_opt_loss"]), 0.0)))
    elif "ng_consistency_score" in ann:
        quality *= float(0.5 + 0.5 * np.clip(float(ann["ng_consistency_score"]), 0.0, 1.0))
    return float(quality)


def apply_bev_nms(annotations: List[dict], args, stats) -> List[dict]:
    if not bool(args.use_bev_nms):
        return annotations

    threshold = float(np.clip(args.bev_nms_iou_threshold, 0.0, 1.0))
    grouped = defaultdict(list)
    passthrough = []
    for index, ann in enumerate(annotations):
        bev_box = ann_bev_box(ann)
        if bev_box is None:
            passthrough.append((index, ann))
            continue
        key = (int(ann.get("image_id", -1)), int(ann.get("category_id", -1)))
        grouped[key].append((index, ann, bev_box, annotation_nms_quality(ann, args)))

    keep_indices = {index for index, _ in passthrough}
    suppressed = 0
    for group_items in grouped.values():
        group_items = sorted(group_items, key=lambda item: item[3], reverse=True)
        kept_boxes = []
        for index, ann, bev_box, _quality in group_items:
            if any(bev_iou_np(bev_box, kept) > threshold for kept in kept_boxes):
                suppressed += 1
                continue
            keep_indices.add(index)
            kept_boxes.append(bev_box)

    stats["bev_nms_enabled"] = 1
    stats["bev_nms_suppressed"] += int(suppressed)
    stats["bev_nms_threshold_x1000"] = int(round(threshold * 1000))
    return [ann for index, ann in enumerate(annotations) if index in keep_indices]


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
    pag_values = []
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
        proposal_sources = list(img_data.get("proposal_sources", []))
        proposal_external_flags = np.asarray(
            img_data.get("proposal_external_flags", np.zeros(len(phrases), dtype=bool))
        ).reshape(-1)
        proposal_source_indices = np.asarray(
            img_data.get("proposal_source_indices", np.arange(len(phrases), dtype=np.int64))
        ).reshape(-1)
        n = min(len(phrases), len(boxes), mask.shape[0], raw_mask.shape[0])
        stats["objects_seen"] += int(n)
        used_anchor_ids = set()

        for j in range(n):
            category_name = str(phrases[j])
            category_id = category_name_to_id.get(category_name, 0)
            if category_id == 0:
                stats["unknown_category"] += 1
            score = sanitize_detection_score(np.asarray(scores[j]).reshape(-1)[0])
            bbox = normalize_bbox_to_pixels(boxes[j], width, height)
            proposal_source = (
                str(proposal_sources[j])
                if j < len(proposal_sources)
                else "groundingsam"
            )
            proposal_external = (
                bool(proposal_external_flags[j])
                if j < len(proposal_external_flags)
                else False
            )
            proposal_source_index = (
                int(proposal_source_indices[j])
                if j < len(proposal_source_indices)
                else int(j)
            )
            extra = {
                "pseudo_mask_index": int(j),
                "proposal_source": proposal_source,
                "proposal_external": bool(proposal_external),
                "proposal_source_index": int(proposal_source_index),
                "proposal_2d_score": float(score),
            }
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

            if bool(args.use_depth_aware_mask_selector):
                cur_mask, selector_metrics = select_depth_aware_mask(
                    depth,
                    mask[j],
                    raw_mask[j],
                    bbox,
                    args,
                )
                extra.update(selector_metrics)
                mask_source = "selector_" + str(selector_metrics.get("mask_selector_selected", "unknown"))
            else:
                cur_mask = mask[j]
                mask_source = "eroded"
                if (
                    np.asarray(cur_mask).sum() < int(args.min_mask_pixels)
                    and np.asarray(raw_mask[j]).sum() >= int(args.min_mask_pixels)
                ):
                    cur_mask = raw_mask[j]
                    mask_source = "raw_fallback"
            if (
                np.asarray(cur_mask).sum() < int(args.min_mask_pixels)
                and np.asarray(raw_mask[j]).sum() >= int(args.min_mask_pixels)
            ):
                cur_mask = raw_mask[j]
                mask_source = "raw_fallback_after_selector"
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
                    {
                        **extra,
                        "dfu_reason": "small_mask",
                        "dfu_mask_source": mask_source,
                    },
                )
                ann = reference_match_update(ann, ref_index, args)
                ann = imov3d_quality_weight_update(ann, args)
                ann = external_strict_3d_update(ann, args, stats)
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
                ann = imov3d_quality_weight_update(ann, args)
                ann = external_strict_3d_update(ann, args, stats)
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
            ann = imov3d_quality_weight_update(ann, args)
            if bool(ann.get("valid3D", True)):
                ann = projected_corner_depth_score_update(
                    ann,
                    depth,
                    clean_mask,
                    bbox,
                    K,
                    filtered_points,
                    prior,
                    args,
                )
                ann = locate3d_factorized_curriculum_update(ann, args)
            ann = external_strict_3d_update(ann, args, stats)
            if bool(ann.get("valid3D", True)):
                if bool(ann.get("dbscan_enabled", False)):
                    stats["dfu3d_cluster_cleaning_seen"] += 1
                    if not bool(ann.get("dbscan_fallback", False)):
                        stats["dfu3d_cluster_cleaning_applied"] += 1
                if bool(ann.get("normal_gravity_available", False)):
                    stats["imov3d_normal_gravity_available"] += 1
                if "imov3d_quality_weight" in ann:
                    stats["imov3d_quality_weighted"] += 1
                if bool(ann.get("moca3d_projected_corner_depth_enabled", False)):
                    stats["moca3d_corner_depth_scored"] += 1
            if bool(ann.get("valid3D", True)):
                weight_values.append(float(ann.get("pseudo_weight", 1.0)))
                if "pag_score" in ann:
                    pag_values.append(float(ann.get("pag_score", 0.0)))
            annotations.append(ann)
            ann_id += 1

    annotations = apply_bev_nms(annotations, args, stats)
    weight_values = [
        float(ann.get("pseudo_weight", 1.0))
        for ann in annotations
        if bool(ann.get("valid3D", True))
    ]
    pag_values = [
        float(ann.get("pag_score", 0.0))
        for ann in annotations
        if bool(ann.get("valid3D", True)) and "pag_score" in ann
    ]

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
            or args.use_depth_aware_mask_selector
            or args.use_frustum_dbscan
            or args.use_normal_ground_fusion
            or args.use_surface_box_optimization
        )
        else "dfu_robust_pca_with_boxer_consistency_weight"
    )
    output["info"]["pseudo_label_source_json"] = os.path.abspath(args.source_json)
    output["info"]["pseudo_label_cache_root"] = os.path.abspath(input_folder)
    output["info"]["depth_aware_mask_selector"] = bool(args.use_depth_aware_mask_selector)
    output["info"]["imov3d_depth_edge_filter"] = bool(args.use_depth_edge_filter)
    output["info"]["imov3d_frustum_dbscan"] = bool(args.use_frustum_dbscan)
    output["info"]["imov3d_normal_ground_fusion"] = bool(args.use_normal_ground_fusion)
    output["info"]["imov3d_quality_weight"] = bool(args.use_imov3d_quality_weight)
    output["info"]["imov3d_quality_min"] = float(args.imov3d_quality_min)
    output["info"]["moca3d_projected_corner_depth_score"] = bool(args.use_projected_corner_depth_score)
    output["info"]["pag_apply_to_weight"] = bool(args.pag_apply_to_weight)
    output["info"]["locate3d_factorized_curriculum"] = bool(args.use_locate3d_factorized_curriculum)
    output["info"]["surface_box_optimization"] = bool(args.use_surface_box_optimization)
    output["info"]["surface_include_right_angle_yaws"] = bool(args.surface_include_right_angle_yaws)
    output["info"]["surface_enable_dims_swap"] = bool(args.surface_enable_dims_swap)
    output["info"]["source_geometry_anchor"] = bool(args.use_source_geometry_anchor)
    output["info"]["surface_center_mode"] = str(args.surface_center_mode)
    output["info"]["surface_require_improvement"] = bool(args.surface_require_improvement)
    output["info"]["bev_nms"] = bool(args.use_bev_nms)
    output["info"]["bev_nms_iou_threshold"] = float(args.bev_nms_iou_threshold)
    output["info"]["external_strict_3d"] = bool(args.use_external_strict_3d)
    output["info"]["external_strict_sources"] = str(args.external_strict_sources)
    output["info"]["external_strict_accept_quality"] = float(args.external_strict_accept_quality)
    if args.reference_json:
        output["info"]["pseudo_label_reference_json"] = os.path.abspath(args.reference_json)
    if weight_values:
        output["info"]["ng_mean_pseudo_weight"] = float(np.mean(weight_values))
    if pag_values:
        output["info"]["pag_mean_score"] = float(np.mean(pag_values))

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
    if pag_values:
        stats["pag_score_mean_x1000"] = int(round(float(np.mean(pag_values)) * 1000))
        stats["pag_score_p10_x1000"] = int(round(float(np.percentile(pag_values, 10)) * 1000))
        stats["pag_score_p50_x1000"] = int(round(float(np.percentile(pag_values, 50)) * 1000))

    print(f"Wrote {args.output_json}")
    print(dict(stats))

    if args.stats_json:
        os.makedirs(os.path.dirname(args.stats_json) or ".", exist_ok=True)
        with open(args.stats_json, "w") as f:
            json.dump(dict(stats), f, indent=2)


if __name__ == "__main__":
    main()
