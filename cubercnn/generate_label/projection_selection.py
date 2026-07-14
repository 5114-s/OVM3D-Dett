"""Depth-aware projective selection for released OVM3D box candidates.

This module never creates, moves, rotates, or resizes a candidate.  It only
re-ranks the candidates already emitted by ``generate_possible_bboxs`` using
the raw mask, the adaptively eroded core mask, and the cached monocular depth.
"""

import math
import os

import cv2
import numpy as np


def _env_bool(name, default=False):
    value = os.environ.get(name)
    if value is None:
        return bool(default)
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _env_float(name, default):
    value = os.environ.get(name)
    if value is None or str(value).strip() == "":
        return float(default)
    try:
        return float(value)
    except ValueError:
        return float(default)


def _env_int(name, default):
    value = os.environ.get(name)
    if value is None or str(value).strip() == "":
        return int(default)
    try:
        return int(value)
    except ValueError:
        return int(default)


def get_projection_selection_cfg():
    """Read the opt-in module configuration from environment variables."""
    return {
        "enabled": _env_bool("OVM3D_USE_PROJECTION_SELECTION", False),
        "raw_weight": _env_float("OVM3D_PROJ_RAW_WEIGHT", 0.20),
        "core_weight": _env_float("OVM3D_PROJ_CORE_WEIGHT", 0.20),
        "depth_weight": _env_float("OVM3D_PROJ_DEPTH_WEIGHT", 0.40),
        "inside_weight": _env_float("OVM3D_PROJ_INSIDE_WEIGHT", 0.20),
        "switch_margin": _env_float("OVM3D_PROJ_SWITCH_MARGIN", 0.03),
        "min_score": _env_float("OVM3D_PROJ_MIN_SCORE", 0.20),
        "max_inside_drop": _env_float("OVM3D_PROJ_MAX_INSIDE_DROP", 0.05),
        "depth_scale": _env_float("OVM3D_PROJ_DEPTH_SCALE", 0.15),
        "max_depth_samples": _env_int("OVM3D_PROJ_MAX_DEPTH_SAMPLES", 256),
        "min_depth_samples": _env_int("OVM3D_PROJ_MIN_DEPTH_SAMPLES", 8),
    }


def disabled_selection_metric(reason="disabled"):
    return {
        "enabled": False,
        "eligible": False,
        "switched": False,
        "reason": str(reason),
    }


def _as_binary_mask(mask, shape_hw):
    if mask is None:
        return None
    arr = np.asarray(mask).squeeze()
    if arr.shape != tuple(shape_hw):
        return None
    return np.isfinite(arr) & (arr > 0.5)


def _project_cuboid_hull(vertices_cam, K, shape_hw):
    """Return a clipped crop containing the projected convex silhouette."""
    vertices = np.asarray(vertices_cam, dtype=np.float64)
    intrinsic = np.asarray(K, dtype=np.float64)
    if vertices.shape != (8, 3) or intrinsic.shape != (3, 3):
        return None
    if np.any(~np.isfinite(vertices)) or np.any(vertices[:, 2] <= 1e-4):
        return None

    projected = (intrinsic @ vertices.T).T
    uv = projected[:, :2] / projected[:, 2:3]
    if np.any(~np.isfinite(uv)):
        return None
    hull = cv2.convexHull(uv.astype(np.float32)).reshape(-1, 2)
    if hull.shape[0] < 3 or abs(float(cv2.contourArea(hull))) < 1.0:
        return None

    height, width = int(shape_hw[0]), int(shape_hw[1])
    x1 = max(0, int(math.floor(float(hull[:, 0].min()))))
    y1 = max(0, int(math.floor(float(hull[:, 1].min()))))
    x2 = min(width, int(math.ceil(float(hull[:, 0].max()))) + 1)
    y2 = min(height, int(math.ceil(float(hull[:, 1].max()))) + 1)
    if x2 <= x1 or y2 <= y1:
        return None

    local_hull = np.rint(
        hull - np.array([x1, y1], dtype=np.float32)
    ).astype(np.int32)
    crop = np.zeros((y2 - y1, x2 - x1), dtype=np.uint8)
    cv2.fillConvexPoly(crop, local_hull, 1)
    if int(crop.sum()) == 0:
        return None
    return x1, y1, x2, y2, crop.astype(bool)


def _deterministic_depth_samples(core_mask, depth, max_samples):
    depth_arr = np.asarray(depth, dtype=np.float64)
    valid = core_mask & np.isfinite(depth_arr) & (depth_arr > 1e-4)
    ys, xs = np.nonzero(valid)
    if ys.size == 0:
        return None
    if ys.size > int(max_samples):
        indices = np.linspace(0, ys.size - 1, int(max_samples), dtype=np.int64)
        ys, xs = ys[indices], xs[indices]
    return (
        xs.astype(np.float64),
        ys.astype(np.float64),
        depth_arr[ys, xs],
    )


def _ray_box_front_depths(sample, center_cam, R_cam, dimensions_xyz, K):
    """Intersect z-parameterized camera rays with an oriented cuboid."""
    if sample is None:
        return None, None
    xs, ys, _ = sample
    intrinsic = np.asarray(K, dtype=np.float64)
    rotation = np.asarray(R_cam, dtype=np.float64)
    center = np.asarray(center_cam, dtype=np.float64).reshape(3)
    dimensions = np.asarray(dimensions_xyz, dtype=np.float64).reshape(3)
    if (
        intrinsic.shape != (3, 3)
        or rotation.shape != (3, 3)
        or np.any(~np.isfinite(rotation))
        or np.any(~np.isfinite(center))
        or np.any(~np.isfinite(dimensions))
        or np.any(dimensions <= 1e-5)
        or abs(float(intrinsic[0, 0])) < 1e-9
        or abs(float(intrinsic[1, 1])) < 1e-9
    ):
        return None, None

    rays_cam = np.stack(
        [
            (xs - intrinsic[0, 2]) / intrinsic[0, 0],
            (ys - intrinsic[1, 2]) / intrinsic[1, 1],
            np.ones_like(xs),
        ],
        axis=1,
    )
    # Column convention: p_cam = R_cam @ p_local + center_cam.
    # Row-vector multiplication by R_cam performs the inverse rotation.
    origin_local = (-center) @ rotation
    direction_local = rays_cam @ rotation
    half = dimensions / 2.0
    safe_direction = np.where(
        np.abs(direction_local) < 1e-9,
        np.where(direction_local < 0, -1e-9, 1e-9),
        direction_local,
    )
    t1 = (-half - origin_local[None, :]) / safe_direction
    t2 = (half - origin_local[None, :]) / safe_direction
    t_near = np.max(np.minimum(t1, t2), axis=1)
    t_far = np.min(np.maximum(t1, t2), axis=1)
    valid = (
        np.isfinite(t_near)
        & np.isfinite(t_far)
        & (t_far >= t_near)
        & (t_far > 0)
    )
    front_depth = np.where(valid, np.maximum(t_near, 0.0), np.nan)
    return front_depth, valid


def score_projected_candidate(
    candidate,
    raw_mask,
    core_mask,
    depth,
    K,
    cfg,
    depth_sample=None,
):
    """Compute mask, depth, and point-support scores for one candidate."""
    shape_hw = np.asarray(depth).shape[:2]
    raw_binary = _as_binary_mask(raw_mask, shape_hw)
    core_binary = _as_binary_mask(core_mask, shape_hw)
    inside_ratio = float(candidate.get("inside_ratio", 0.0))
    metric = {
        "valid": False,
        "score": float("-inf"),
        "raw_iou": 0.0,
        "core_coverage": 0.0,
        "depth_score": 0.0,
        "depth_error": float("inf"),
        "depth_hit_ratio": 0.0,
        "inside_ratio": inside_ratio if np.isfinite(inside_ratio) else 0.0,
    }
    if raw_binary is None or core_binary is None:
        return metric

    projected = _project_cuboid_hull(candidate["vertices_cam"], K, shape_hw)
    if projected is None:
        return metric
    x1, y1, x2, y2, projected_crop = projected
    raw_crop = raw_binary[y1:y2, x1:x2]
    core_crop = core_binary[y1:y2, x1:x2]
    pred_area = int(projected_crop.sum())
    raw_intersection = int(np.logical_and(projected_crop, raw_crop).sum())
    core_intersection = int(np.logical_and(projected_crop, core_crop).sum())
    raw_union = pred_area + int(raw_binary.sum()) - raw_intersection
    raw_iou = float(raw_intersection / max(raw_union, 1))
    core_coverage = float(core_intersection / max(int(core_binary.sum()), 1))

    if depth_sample is None:
        depth_sample = _deterministic_depth_samples(
            core_binary, depth, int(cfg["max_depth_samples"])
        )
    front_depth, depth_valid = _ray_box_front_depths(
        depth_sample,
        candidate["center_cam"],
        candidate["R_cam"],
        candidate["dimensions_xyz"],
        K,
    )
    depth_score = 0.0
    depth_error = float("inf")
    depth_hit_ratio = 0.0
    if front_depth is not None and depth_valid is not None and depth_sample is not None:
        observed = depth_sample[2]
        valid_count = int(depth_valid.sum())
        depth_hit_ratio = float(valid_count / max(observed.size, 1))
        if valid_count >= int(cfg["min_depth_samples"]):
            relative_error = np.abs(
                front_depth[depth_valid] - observed[depth_valid]
            ) / np.maximum(observed[depth_valid], 1e-3)
            depth_error = float(np.median(np.clip(relative_error, 0.0, 1.0)))
            depth_score = float(
                depth_hit_ratio
                * math.exp(
                    -depth_error / max(float(cfg["depth_scale"]), 1e-6)
                )
            )

    inside_score = float(np.clip(inside_ratio, 0.0, 1.0))
    weights = np.asarray(
        [
            cfg["raw_weight"],
            cfg["core_weight"],
            cfg["depth_weight"],
            cfg["inside_weight"],
        ],
        dtype=np.float64,
    )
    weight_sum = float(weights.sum())
    if weight_sum <= 0:
        return metric
    score = float(
        np.dot(weights, [raw_iou, core_coverage, depth_score, inside_score])
        / weight_sum
    )
    metric.update(
        {
            "valid": True,
            "score": score,
            "raw_iou": raw_iou,
            "core_coverage": core_coverage,
            "depth_score": depth_score,
            "depth_error": depth_error,
            "depth_hit_ratio": depth_hit_ratio,
        }
    )
    return metric


def select_projected_candidate(
    candidates,
    original_index,
    raw_mask,
    core_mask,
    depth,
    K,
    cfg,
):
    """Conservatively replace the released objective's winning candidate."""
    metric = {
        "enabled": bool(cfg.get("enabled", False)),
        "eligible": False,
        "switched": False,
        "reason": "disabled",
        "candidate_count": int(len(candidates)),
        "original_index": int(original_index) if original_index is not None else -1,
        "selected_index": int(original_index) if original_index is not None else -1,
    }
    if not cfg.get("enabled", False):
        return original_index, metric
    if original_index is None or not (0 <= int(original_index) < len(candidates)):
        metric["reason"] = "no_original_candidate"
        return original_index, metric
    if raw_mask is None or core_mask is None or depth is None or K is None:
        metric["reason"] = "missing_projection_inputs"
        return original_index, metric

    core_binary = _as_binary_mask(core_mask, np.asarray(depth).shape[:2])
    if core_binary is None:
        metric["reason"] = "invalid_core_mask"
        return original_index, metric
    depth_sample = _deterministic_depth_samples(
        core_binary, depth, int(cfg["max_depth_samples"])
    )
    scores = [
        score_projected_candidate(
            candidate,
            raw_mask,
            core_mask,
            depth,
            K,
            cfg,
            depth_sample=depth_sample,
        )
        for candidate in candidates
    ]
    valid_indices = [i for i, score in enumerate(scores) if score["valid"]]
    if not valid_indices or not scores[int(original_index)]["valid"]:
        metric["reason"] = "invalid_projection_score"
        return original_index, metric

    best_index = max(valid_indices, key=lambda i: (scores[i]["score"], -i))
    original_score = scores[int(original_index)]["score"]
    best_score = scores[best_index]["score"]
    improvement = float(best_score - original_score)
    original_inside = float(candidates[int(original_index)]["inside_ratio"])
    best_inside = float(candidates[best_index]["inside_ratio"])
    metric.update(
        {
            "eligible": True,
            "reason": "kept_original",
            "best_index": int(best_index),
            "original_score": float(original_score),
            "best_score": float(best_score),
            "improvement": improvement,
            "original_components": scores[int(original_index)],
            "best_components": scores[best_index],
        }
    )

    should_switch = (
        best_index != int(original_index)
        and best_score >= float(cfg["min_score"])
        and improvement >= float(cfg["switch_margin"])
        and best_inside
        >= original_inside - float(cfg["max_inside_drop"])
    )
    if should_switch:
        metric.update(
            {
                "switched": True,
                "reason": "projection_improved",
                "selected_index": int(best_index),
            }
        )
        return best_index, metric
    return original_index, metric

