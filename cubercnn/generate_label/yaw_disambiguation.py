"""Prior-conditioned 90-degree yaw disambiguation for indoor pseudo labels.

The released OVM3D-Det path uses the first PCA component as yaw.  A visible
front/rear surface can make that component follow object width instead of
length.  This opt-in module compares the released yaw with ``yaw + pi/2`` and
switches only when robust point support agrees with the category dimensions.

The robust boundary statistic follows MonoSOWA's official stationary-car
closeness criterion (10/90 percentiles and a saturated sigmoid):
https://github.com/jskvrna/MonoSOWA/blob/main/pseudo_label_generator/3d/scripts/dimension_estimator.py

MonoSOWA's criterion is intentionally symmetric under a 90-degree axis swap,
so it is used here as a quality gate.  OVM3D's existing physical ``(w,h,l)``
prior supplies the missing length/width assignment.  The module is disabled by
default and does not affect released behavior unless ``OVM3D_USE_PRIOR_YAW=1``.
"""

import math
import os

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


def get_prior_yaw_cfg():
    """Read conservative, opt-in settings from environment variables."""
    return {
        "enabled": _env_bool("OVM3D_USE_PRIOR_YAW", False),
        "min_points": _env_int("OVM3D_YAW_MIN_POINTS", 20),
        "min_prior_aspect": _env_float("OVM3D_YAW_MIN_PRIOR_ASPECT", 1.50),
        "quantile_low": _env_float("OVM3D_YAW_QUANTILE_LOW", 0.10),
        "quantile_high": _env_float("OVM3D_YAW_QUANTILE_HIGH", 0.90),
        # MonoSOWA official default.
        "steepness": _env_float("OVM3D_YAW_STEEPNESS", 10.0),
        "min_scc_quality": _env_float("OVM3D_YAW_MIN_SCC_QUALITY", 0.15),
        "switch_margin": _env_float("OVM3D_YAW_SWITCH_MARGIN", 0.15),
        "min_selected_score": _env_float("OVM3D_YAW_MIN_SCORE", 0.60),
        "overflow_tolerance": _env_float("OVM3D_YAW_OVERFLOW_TOL", 0.10),
        "overflow_scale": _env_float("OVM3D_YAW_OVERFLOW_SCALE", 0.25),
        "max_selected_overflow": _env_float(
            "OVM3D_YAW_MAX_SELECTED_OVERFLOW", 0.35
        ),
        "secondary_support_weight": _env_float(
            "OVM3D_YAW_SECONDARY_WEIGHT", 0.15
        ),
    }


def disabled_yaw_metric(reason="disabled"):
    return {
        "enabled": False,
        "eligible": False,
        "switched": False,
        "reason": str(reason),
    }


def _project_ground(points, yaw):
    points = np.asarray(points, dtype=np.float64)
    cosine, sine = math.cos(float(yaw)), math.sin(float(yaw))
    x = points[:, 0]
    z = points[:, 2]
    return cosine * x + sine * z, -sine * x + cosine * z


def _robust_axis_statistics(points, yaw, cfg):
    """Return robust spans and normalized MonoSOWA SCC quality."""
    c1, c2 = _project_ground(points, yaw)
    q_low = float(cfg["quantile_low"])
    q_high = float(cfg["quantile_high"])
    q1_low, q1_high = np.quantile(c1, [q_low, q_high])
    q2_low, q2_high = np.quantile(c2, [q_low, q_high])

    d1 = np.minimum(np.abs(c1 - q1_low), np.abs(q1_high - c1))
    d2 = np.minimum(np.abs(c2 - q2_low), np.abs(q2_high - c2))
    steepness = float(cfg["steepness"])
    # Clip only the exponential argument; this is numerically identical to
    # MonoSOWA's sigmoid over its normal operating range.
    s1 = 1.0 / (1.0 + np.exp(-np.clip(d1 * steepness, -60.0, 60.0)))
    s2 = 1.0 / (1.0 + np.exp(-np.clip(d2 * steepness, -60.0, 60.0)))
    closeness = -float(np.sum(np.minimum(s1, s2)))
    # Each saturated term lies in [0.5, 1]. Map the official maximization
    # criterion to [1, 0] for convenient gating and reporting.
    scc_quality = 2.0 * (1.0 + closeness / max(c1.size, 1))
    return {
        "span_length_axis": float(max(q1_high - q1_low, 0.0)),
        "span_width_axis": float(max(q2_high - q2_low, 0.0)),
        "scc_quality": float(np.clip(scc_quality, 0.0, 1.0)),
    }


def _prior_axis_score(stats, length, width, cfg):
    ratios = np.array(
        [
            stats["span_length_axis"] / float(length),
            stats["span_width_axis"] / float(width),
        ],
        dtype=np.float64,
    )
    if np.any(~np.isfinite(ratios)):
        return float("-inf"), float("inf"), [0.0, 0.0]

    coverage = np.clip(ratios, 0.0, 1.0)
    secondary_weight = float(
        np.clip(cfg["secondary_support_weight"], 0.0, 0.5)
    )
    support = (
        (1.0 - secondary_weight) * float(np.max(coverage))
        + secondary_weight * float(np.min(coverage))
    )
    overflow = float(
        np.mean(
            np.maximum(
                ratios - 1.0 - float(cfg["overflow_tolerance"]), 0.0
            )
        )
    )
    overflow_scale = max(float(cfg["overflow_scale"]), 1e-6)
    score = support * math.exp(-overflow / overflow_scale)
    return float(score), overflow, coverage.tolist()


def select_prior_conditioned_yaw(points, prior, original_yaw, cfg=None):
    """Choose between PCA yaw and its orthogonal axis with safe fallback.

    Args:
        points: Ground-aligned Nx3 pseudo-LiDAR points.
        prior: OVM3D category dimensions in ``(width, height, length)`` order.
        original_yaw: Released PCA yaw in radians.
        cfg: Optional dictionary returned by :func:`get_prior_yaw_cfg`.

    Returns:
        ``(selected_yaw, metric)``.  Angles are not wrapped because all later
        OVM3D operations are periodic and retaining the original value makes
        the no-switch path bit-for-bit unchanged.
    """
    if cfg is None:
        cfg = get_prior_yaw_cfg()
    if not cfg.get("enabled", False):
        return float(original_yaw), disabled_yaw_metric()

    metric = {
        "enabled": True,
        "eligible": False,
        "switched": False,
        "reason": "invalid_input",
    }
    points = np.asarray(points, dtype=np.float64)
    prior = np.asarray(prior, dtype=np.float64).reshape(-1)
    if (
        points.ndim != 2
        or points.shape[1] < 3
        or points.shape[0] < int(cfg["min_points"])
        or np.any(~np.isfinite(points[:, :3]))
    ):
        metric["reason"] = "too_few_or_invalid_points"
        return float(original_yaw), metric
    if (
        prior.size < 3
        or np.any(~np.isfinite(prior[:3]))
        or float(prior[0]) <= 0.0
        or float(prior[2]) <= 0.0
    ):
        metric["reason"] = "invalid_prior"
        return float(original_yaw), metric

    width, length = float(prior[0]), float(prior[2])
    prior_aspect = max(length, width) / min(length, width)
    metric["prior_aspect"] = float(prior_aspect)
    if prior_aspect < float(cfg["min_prior_aspect"]):
        metric["reason"] = "near_square_prior"
        return float(original_yaw), metric

    alternate_yaw = float(original_yaw) + math.pi / 2.0
    original_stats = _robust_axis_statistics(points, original_yaw, cfg)
    alternate_stats = _robust_axis_statistics(points, alternate_yaw, cfg)
    original_score, original_overflow, original_coverage = _prior_axis_score(
        original_stats, length, width, cfg
    )
    alternate_score, alternate_overflow, alternate_coverage = _prior_axis_score(
        alternate_stats, length, width, cfg
    )
    scc_quality = min(
        original_stats["scc_quality"], alternate_stats["scc_quality"]
    )
    improvement = alternate_score - original_score
    metric.update(
        {
            "eligible": True,
            "original_yaw": float(original_yaw),
            "alternate_yaw": float(alternate_yaw),
            "original_score": float(original_score),
            "alternate_score": float(alternate_score),
            "score_improvement": float(improvement),
            "original_overflow": float(original_overflow),
            "alternate_overflow": float(alternate_overflow),
            "original_coverage": [float(v) for v in original_coverage],
            "alternate_coverage": [float(v) for v in alternate_coverage],
            "original_spans": [
                original_stats["span_length_axis"],
                original_stats["span_width_axis"],
            ],
            "alternate_spans": [
                alternate_stats["span_length_axis"],
                alternate_stats["span_width_axis"],
            ],
            "original_scc_quality": float(original_stats["scc_quality"]),
            "alternate_scc_quality": float(alternate_stats["scc_quality"]),
            "scc_quality": float(scc_quality),
        }
    )

    if not np.isfinite(original_score) or not np.isfinite(alternate_score):
        metric["reason"] = "invalid_score"
        return float(original_yaw), metric
    if scc_quality < float(cfg["min_scc_quality"]):
        metric["reason"] = "low_scc_quality"
        return float(original_yaw), metric
    if alternate_overflow > float(cfg["max_selected_overflow"]):
        metric["reason"] = "alternate_overflow"
        return float(original_yaw), metric
    if alternate_score < float(cfg["min_selected_score"]):
        metric["reason"] = "alternate_weak_support"
        return float(original_yaw), metric
    if improvement < float(cfg["switch_margin"]):
        metric["reason"] = "kept_original"
        return float(original_yaw), metric

    metric["switched"] = True
    metric["reason"] = "prior_axis_improved"
    return alternate_yaw, metric
