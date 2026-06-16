#!/usr/bin/env python3
"""Shared utilities for Boxer-Residual-LIFT.

The base feature extractor uses box geometry, camera rays, Boxer outputs, and
Boxer/depth quality metrics.  An optional ROI branch adds lightweight image-crop
features so the residual head can condition on object appearance in the same
spirit as OVMono3D-LIFT, while keeping the pseudo-label pipeline independent
from the main OVM3D detector.
"""

from __future__ import annotations

import math
import os
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import torch
from torch import nn
from PIL import Image


EPS = 1e-6


BASE_FEATURE_NAMES = [
    "bbox_x1_norm",
    "bbox_y1_norm",
    "bbox_x2_norm",
    "bbox_y2_norm",
    "bbox_cx_norm",
    "bbox_cy_norm",
    "bbox_w_norm",
    "bbox_h_norm",
    "bbox_area_norm",
    "bbox_aspect_log",
    "fx_norm",
    "fy_norm",
    "cx_norm",
    "cy_norm",
    "ray_x",
    "ray_y",
    "src_x_over_z",
    "src_y_over_z",
    "src_log_z",
    "src_log_w",
    "src_log_h",
    "src_log_l",
    "src_yaw_sin",
    "src_yaw_cos",
    "score",
    "boxer_projection_iou",
    "boxer_depth_support",
    "boxer_depth_rel_error",
    "boxer_depth_median_log",
    "boxer_quality",
    "prior_ratio_w_log",
    "prior_ratio_h_log",
    "prior_ratio_l_log",
    "ground_snap_distance",
    "ground_min_corner_distance",
    "depth_refine_shift_over_z",
]
FEATURE_NAMES = BASE_FEATURE_NAMES


def roi_feature_names(mode: str = "none", grid_size: int = 4) -> List[str]:
    if mode == "none":
        return []
    if mode != "color_grid":
        raise ValueError(f"Unsupported roi feature mode: {mode}")
    grid_size = max(int(grid_size), 1)
    names = ["roi_valid"]
    for gy in range(grid_size):
        for gx in range(grid_size):
            for channel in ("r", "g", "b"):
                names.append(f"roi_grid_{gy}_{gx}_{channel}")
    for stat in ("mean", "std", "min", "max"):
        for channel in ("r", "g", "b"):
            names.append(f"roi_rgb_{stat}_{channel}")
    names.extend(["roi_gray_mean", "roi_gray_std", "roi_edge_mean", "roi_edge_std"])
    return names


def feature_names_with_roi(mode: str = "none", grid_size: int = 4) -> List[str]:
    return BASE_FEATURE_NAMES + roi_feature_names(mode, grid_size)


def load_roi_feature_cache(path: str | None) -> Tuple[Dict[int, np.ndarray], List[str], Dict]:
    if not path:
        return {}, [], {}
    data = torch.load(path, map_location="cpu")
    ann_ids = data.get("ann_ids")
    features = data.get("features")
    if ann_ids is None or features is None:
        raise ValueError(f"ROI feature cache missing ann_ids/features: {path}")
    ann_ids = ann_ids.long().cpu().numpy().tolist()
    features_np = features.float().cpu().numpy()
    if len(ann_ids) != int(features_np.shape[0]):
        raise ValueError(f"ROI feature cache length mismatch: {path}")
    names = data.get("feature_names")
    if names is None:
        names = [f"cached_roi_{idx}" for idx in range(int(features_np.shape[1]))]
    cache = {int(ann_id): features_np[idx].astype(np.float32) for idx, ann_id in enumerate(ann_ids)}
    return cache, list(names), dict(data.get("config") or {})


def cached_roi_feature(
    ann: Mapping,
    cache: Mapping[int, np.ndarray],
    dim: int,
) -> np.ndarray:
    if dim <= 0:
        return np.zeros(0, dtype=np.float32)
    ann_id = int(ann.get("id", -1))
    feat = cache.get(ann_id)
    if feat is None:
        return np.zeros(dim, dtype=np.float32)
    feat = np.asarray(feat, dtype=np.float32).reshape(-1)
    if feat.shape[0] != dim:
        out = np.zeros(dim, dtype=np.float32)
        n = min(dim, feat.shape[0])
        out[:n] = feat[:n]
        return out
    return feat


def as_float_array(values: Sequence[float], shape: Tuple[int, ...] | None = None) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    if shape is not None:
        arr = arr.reshape(shape)
    return arr


def finite_float(value, default: float = 0.0) -> float:
    try:
        value = float(value)
    except Exception:
        return float(default)
    if not math.isfinite(value):
        return float(default)
    return float(value)


def resolve_image_path(image_root: str, file_path: str) -> str:
    if os.path.isabs(file_path):
        return file_path
    return os.path.join(image_root, file_path)


def load_image_rgb(image_root: str, image: Mapping) -> np.ndarray | None:
    file_path = image.get("file_path")
    if not file_path:
        return None
    path = resolve_image_path(image_root, str(file_path))
    try:
        with Image.open(path) as im:
            return np.asarray(im.convert("RGB"), dtype=np.uint8)
    except Exception:
        return None


def normalize_angle(angle: float) -> float:
    return float((angle + math.pi) % (2.0 * math.pi) - math.pi)


def yaw_from_R(R_cam: Sequence[Sequence[float]]) -> float:
    R = as_float_array(R_cam, (3, 3))
    if not np.all(np.isfinite(R)):
        return 0.0
    # Same convention used by the Fast-SAM3D conversion utilities in this repo.
    return normalize_angle(math.atan2(float(R[0, 2] - R[2, 0]), float(R[0, 0] + R[2, 2]) + EPS))


def rotation_y(yaw: float) -> np.ndarray:
    c = math.cos(yaw)
    s = math.sin(yaw)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=np.float32)


def xyxy_from_ann(ann: Mapping) -> np.ndarray:
    for key in ("bbox2D_tight", "bbox2D_proj", "bbox2D_trunc", "bbox"):
        if key not in ann:
            continue
        box = np.asarray(ann[key], dtype=np.float32).reshape(-1)
        if box.shape[0] < 4:
            continue
        box = box[:4].copy()
        if np.any(~np.isfinite(box)) or np.all(box == -1):
            continue
        if key == "bbox":
            box[2] += box[0]
            box[3] += box[1]
        if box[2] <= box[0] or box[3] <= box[1]:
            continue
        return box
    return np.array([-1.0, -1.0, -1.0, -1.0], dtype=np.float32)


def _roi_zeros(mode: str, grid_size: int) -> np.ndarray:
    return np.zeros(len(roi_feature_names(mode, grid_size)), dtype=np.float32)


def extract_roi_feature(
    image_rgb: np.ndarray | None,
    ann: Mapping,
    image: Mapping,
    *,
    mode: str = "none",
    grid_size: int = 4,
    context_scale: float = 1.15,
) -> np.ndarray:
    if mode == "none":
        return np.zeros(0, dtype=np.float32)
    if mode != "color_grid":
        raise ValueError(f"Unsupported roi feature mode: {mode}")
    if image_rgb is None or image_rgb.ndim != 3 or image_rgb.shape[2] < 3:
        return _roi_zeros(mode, grid_size)

    width = max(float(image.get("width", image_rgb.shape[1])), 1.0)
    height = max(float(image.get("height", image_rgb.shape[0])), 1.0)
    box = xyxy_from_ann(ann)
    if np.any(box < 0):
        box = np.array([0.0, 0.0, width, height], dtype=np.float32)
    x1, y1, x2, y2 = [float(x) for x in box]
    bw = max(x2 - x1, 1.0)
    bh = max(y2 - y1, 1.0)
    cx = x1 + 0.5 * bw
    cy = y1 + 0.5 * bh
    scale = max(float(context_scale), 1.0)
    x1 = cx - 0.5 * bw * scale
    x2 = cx + 0.5 * bw * scale
    y1 = cy - 0.5 * bh * scale
    y2 = cy + 0.5 * bh * scale
    h_img, w_img = image_rgb.shape[:2]
    ix1 = int(np.floor(np.clip(x1, 0, w_img - 1)))
    iy1 = int(np.floor(np.clip(y1, 0, h_img - 1)))
    ix2 = int(np.ceil(np.clip(x2, ix1 + 1, w_img)))
    iy2 = int(np.ceil(np.clip(y2, iy1 + 1, h_img)))
    if ix2 <= ix1 or iy2 <= iy1:
        return _roi_zeros(mode, grid_size)

    crop = image_rgb[iy1:iy2, ix1:ix2, :3].astype(np.float32) / 255.0
    if crop.size == 0:
        return _roi_zeros(mode, grid_size)

    grid_size = max(int(grid_size), 1)
    crop_pil = Image.fromarray(np.clip(crop * 255.0, 0, 255).astype(np.uint8))
    grid = np.asarray(
        crop_pil.resize((grid_size, grid_size), Image.BILINEAR),
        dtype=np.float32,
    )[:, :, :3] / 255.0
    rgb_mean = crop.reshape(-1, 3).mean(axis=0)
    rgb_std = crop.reshape(-1, 3).std(axis=0)
    rgb_min = crop.reshape(-1, 3).min(axis=0)
    rgb_max = crop.reshape(-1, 3).max(axis=0)
    gray = 0.299 * crop[:, :, 0] + 0.587 * crop[:, :, 1] + 0.114 * crop[:, :, 2]
    dx = np.diff(gray, axis=1)
    dy = np.diff(gray, axis=0)
    edge = np.concatenate([np.abs(dx).reshape(-1), np.abs(dy).reshape(-1)], axis=0)
    if edge.size == 0:
        edge = np.zeros(1, dtype=np.float32)
    values = np.concatenate(
        [
            np.array([1.0], dtype=np.float32),
            grid.reshape(-1).astype(np.float32),
            rgb_mean.astype(np.float32),
            rgb_std.astype(np.float32),
            rgb_min.astype(np.float32),
            rgb_max.astype(np.float32),
            np.array(
                [
                    float(gray.mean()),
                    float(gray.std()),
                    float(edge.mean()),
                    float(edge.std()),
                ],
                dtype=np.float32,
            ),
        ],
        axis=0,
    )
    return values.astype(np.float32)


def box_iou_xyxy(a: Sequence[float], b: Sequence[float]) -> float:
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    ix1 = max(float(a[0]), float(b[0]))
    iy1 = max(float(a[1]), float(b[1]))
    ix2 = min(float(a[2]), float(b[2]))
    iy2 = min(float(a[3]), float(b[3]))
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, float(a[2] - a[0])) * max(0.0, float(a[3] - a[1]))
    area_b = max(0.0, float(b[2] - b[0])) * max(0.0, float(b[3] - b[1]))
    return float(inter / max(area_a + area_b - inter, EPS))


def cuboid_corners(center: Sequence[float], dims_whl: Sequence[float], R_cam: Sequence[Sequence[float]]) -> np.ndarray:
    center = as_float_array(center, (3,))
    w, h, l = [float(x) for x in as_float_array(dims_whl, (3,))]
    R = as_float_array(R_cam, (3, 3))
    local = np.array(
        [
            [-l / 2, -h / 2, -w / 2],
            [l / 2, -h / 2, -w / 2],
            [l / 2, h / 2, -w / 2],
            [-l / 2, h / 2, -w / 2],
            [-l / 2, -h / 2, w / 2],
            [l / 2, -h / 2, w / 2],
            [l / 2, h / 2, w / 2],
            [-l / 2, h / 2, w / 2],
        ],
        dtype=np.float32,
    )
    return local @ R.T + center.reshape(1, 3)


def project_points(points_cam: np.ndarray, K: Sequence[Sequence[float]]) -> np.ndarray:
    K = as_float_array(K, (3, 3))
    pts = np.asarray(points_cam, dtype=np.float32).reshape(-1, 3)
    z = np.maximum(pts[:, 2], EPS)
    uvw = pts @ K.T
    uv = uvw[:, :2] / z[:, None]
    return uv.astype(np.float32)


def projected_box_from_corners(
    corners_cam: np.ndarray,
    K: Sequence[Sequence[float]],
    width: float,
    height: float,
) -> List[float]:
    corners = np.asarray(corners_cam, dtype=np.float32).reshape(-1, 3)
    if np.any(corners[:, 2] <= EPS):
        return [-1.0, -1.0, -1.0, -1.0]
    uv = project_points(corners, K)
    x1, y1 = uv.min(axis=0)
    x2, y2 = uv.max(axis=0)
    x1 = float(np.clip(x1, 0, width))
    y1 = float(np.clip(y1, 0, height))
    x2 = float(np.clip(x2, 0, width))
    y2 = float(np.clip(y2, 0, height))
    if x2 <= x1 or y2 <= y1:
        return [-1.0, -1.0, -1.0, -1.0]
    return [x1, y1, x2, y2]


def build_category_index(categories: Iterable[Mapping]) -> Dict[int, int]:
    cat_ids = sorted({int(cat["id"]) for cat in categories})
    return {cat_id: idx for idx, cat_id in enumerate(cat_ids)}


def build_feature_vector(ann: Mapping, image: Mapping) -> np.ndarray:
    width = max(float(image.get("width", 1.0)), 1.0)
    height = max(float(image.get("height", 1.0)), 1.0)
    K = as_float_array(image.get("K", np.eye(3)), (3, 3))
    box = xyxy_from_ann(ann)
    if np.any(box < 0):
        box = np.array([0.0, 0.0, width, height], dtype=np.float32)
    x1, y1, x2, y2 = [float(v) for v in box]
    bw = max(x2 - x1, 1.0)
    bh = max(y2 - y1, 1.0)
    bx = x1 + 0.5 * bw
    by = y1 + 0.5 * bh

    center = as_float_array(ann.get("center_cam", [0.0, 0.0, 1.0]), (3,))
    dims = np.maximum(as_float_array(ann.get("dimensions", [1.0, 1.0, 1.0]), (3,)), EPS)
    z = max(float(center[2]), EPS)
    yaw = yaw_from_R(ann.get("R_cam", np.eye(3)))

    prior_ratio = ann.get("boxer_prior_ratio", [1.0, 1.0, 1.0])
    if not isinstance(prior_ratio, (list, tuple)) or len(prior_ratio) != 3:
        prior_ratio = [1.0, 1.0, 1.0]

    fx = max(float(K[0, 0]), EPS)
    fy = max(float(K[1, 1]), EPS)
    cx = float(K[0, 2])
    cy = float(K[1, 2])
    ray_x = (bx - cx) / fx
    ray_y = (by - cy) / fy

    values = [
        x1 / width,
        y1 / height,
        x2 / width,
        y2 / height,
        bx / width,
        by / height,
        bw / width,
        bh / height,
        (bw * bh) / max(width * height, EPS),
        math.log(max(bw / bh, EPS)),
        fx / width,
        fy / height,
        cx / width,
        cy / height,
        ray_x,
        ray_y,
        float(center[0]) / z,
        float(center[1]) / z,
        math.log(z),
        math.log(float(dims[0])),
        math.log(float(dims[1])),
        math.log(float(dims[2])),
        math.sin(yaw),
        math.cos(yaw),
        finite_float(ann.get("score"), 0.0),
        finite_float(ann.get("boxer_projection_iou"), 0.0),
        finite_float(ann.get("boxer_depth_support"), 0.0),
        finite_float(ann.get("boxer_depth_rel_error"), 1.0),
        math.log(max(finite_float(ann.get("boxer_depth_median"), z), EPS)),
        finite_float(ann.get("boxer_quality"), 0.0),
        math.log(max(finite_float(prior_ratio[0], 1.0), EPS)),
        math.log(max(finite_float(prior_ratio[1], 1.0), EPS)),
        math.log(max(finite_float(prior_ratio[2], 1.0), EPS)),
        finite_float(ann.get("boxer_ground_snap_distance"), 0.0),
        finite_float(ann.get("boxer_ground_min_corner_distance"), 0.0),
        finite_float(ann.get("boxer_depth_refine_shift"), 0.0) / z,
    ]
    return np.asarray(values, dtype=np.float32)


def build_feature_vector_with_roi(
    ann: Mapping,
    image: Mapping,
    image_rgb: np.ndarray | None = None,
    *,
    roi_feature_mode: str = "none",
    roi_grid_size: int = 4,
    roi_context_scale: float = 1.15,
) -> np.ndarray:
    base = build_feature_vector(ann, image)
    roi = extract_roi_feature(
        image_rgb,
        ann,
        image,
        mode=roi_feature_mode,
        grid_size=roi_grid_size,
        context_scale=roi_context_scale,
    )
    if roi.size == 0:
        return base
    return np.concatenate([base, roi], axis=0).astype(np.float32)


def target_from_pair(source_ann: Mapping, target_ann: Mapping) -> np.ndarray:
    src_center = as_float_array(source_ann["center_cam"], (3,))
    tgt_center = as_float_array(target_ann["center_cam"], (3,))
    src_dims = np.maximum(as_float_array(source_ann["dimensions"], (3,)), EPS)
    tgt_dims = np.maximum(as_float_array(target_ann["dimensions"], (3,)), EPS)
    src_z = max(float(src_center[2]), EPS)
    src_yaw = yaw_from_R(source_ann.get("R_cam", np.eye(3)))
    tgt_yaw = yaw_from_R(target_ann.get("R_cam", np.eye(3)))
    delta_yaw = normalize_angle(tgt_yaw - src_yaw)
    target = np.concatenate(
        [
            (tgt_center - src_center) / src_z,
            np.log(tgt_dims / src_dims),
            np.array([math.sin(delta_yaw), math.cos(delta_yaw)], dtype=np.float32),
        ]
    )
    return target.astype(np.float32)


def apply_residual(
    ann: Mapping,
    residual: Sequence[float],
    *,
    max_center_shift_ratio: float = 0.35,
    max_log_dim_delta: float = 0.7,
    max_yaw_delta: float = math.pi / 3,
    blend: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray, float, Dict[str, float]]:
    center = as_float_array(ann["center_cam"], (3,)).astype(np.float32)
    dims = np.maximum(as_float_array(ann["dimensions"], (3,)).astype(np.float32), EPS)
    yaw = yaw_from_R(ann.get("R_cam", np.eye(3)))
    z = max(float(center[2]), EPS)
    res = np.asarray(residual, dtype=np.float32).reshape(-1)
    center_delta = np.clip(res[:3], -max_center_shift_ratio, max_center_shift_ratio) * z
    log_dim_delta = np.clip(res[3:6], -max_log_dim_delta, max_log_dim_delta)
    raw_yaw_delta = normalize_angle(math.atan2(float(res[6]), float(res[7]) + EPS))
    yaw_delta = float(np.clip(raw_yaw_delta, -max_yaw_delta, max_yaw_delta))
    blend = float(np.clip(blend, 0.0, 1.0))
    new_center = center + blend * center_delta
    new_dims = dims * np.exp(blend * log_dim_delta)
    new_yaw = normalize_angle(yaw + blend * yaw_delta)
    metrics = {
        "lift_delta_center_norm": float(np.linalg.norm(center_delta)),
        "lift_delta_log_dim_abs_mean": float(np.mean(np.abs(log_dim_delta))),
        "lift_delta_yaw": float(yaw_delta),
    }
    return new_center.astype(np.float32), new_dims.astype(np.float32), new_yaw, metrics


class ResidualLiftHead(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        num_categories: int,
        category_embed_dim: int = 32,
        hidden_dim: int = 256,
        num_layers: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.category_embed = nn.Embedding(max(num_categories, 1), category_embed_dim)
        layers: List[nn.Module] = []
        in_dim = feature_dim + category_embed_dim
        for _ in range(max(num_layers, 1)):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, 8))
        self.net = nn.Sequential(*layers)

    def forward(self, features: torch.Tensor, category_indices: torch.Tensor) -> torch.Tensor:
        cat = self.category_embed(category_indices.clamp(min=0, max=self.category_embed.num_embeddings - 1))
        return self.net(torch.cat([features, cat], dim=-1))
