#!/usr/bin/env python3
"""LocateAnything3D-style pseudo-label reweighting.

This is a lightweight pseudo-label post-processor. It does not rerun
GroundingSAM, UniDepth, PCA, Boxer, SOR, Detic, or any generator. Instead it
uses the existing Omni3D pseudo labels and adds factorized confidence fields
inspired by LocateAnything3D:

  1. 2D evidence first: projected 3D box must agree with the visible 2D box.
  2. Near-to-far curriculum: nearer objects receive stronger supervision.
  3. Center -> depth -> size -> yaw factorization: unreliable attributes are
     down-weighted more aggressively, but labels are kept for recall.

The output JSON is a separate route and is safe to compare against existing
MoCA3D-style, Detic-fusion, and ZEM routes.
"""

import argparse
import copy
import json
import math
import os
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(
        description="Add LocateAnything3D-style factorized pseudo weights."
    )
    parser.add_argument("--source_json", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--stats_json", default=None)
    parser.add_argument("--max_images", type=int, default=None)

    parser.add_argument("--near_z", type=float, default=1.0)
    parser.add_argument("--far_z", type=float, default=6.0)
    parser.add_argument("--min_near_weight", type=float, default=0.55)
    parser.add_argument("--xy_floor", type=float, default=0.80)
    parser.add_argument("--z_floor", type=float, default=0.75)
    parser.add_argument("--dims_floor", type=float, default=0.45)
    parser.add_argument("--pose_floor", type=float, default=0.25)
    parser.add_argument("--joint_floor", type=float, default=0.40)

    parser.add_argument("--projection_weight", type=float, default=0.45)
    parser.add_argument("--quality_weight", type=float, default=0.25)
    parser.add_argument("--area_weight", type=float, default=0.15)
    parser.add_argument("--visibility_weight", type=float, default=0.15)
    parser.add_argument("--good_box_sqrt_area", type=float, default=48.0)
    parser.add_argument("--min_box_sqrt_area", type=float, default=8.0)
    parser.add_argument("--border_margin", type=float, default=2.0)
    parser.add_argument(
        "--combine_mode",
        choices=["multiply", "replace", "max"],
        default="multiply",
        help=(
            "How new LocateAnything3D weights combine with existing "
            "pseudo_weight_* fields."
        ),
    )
    parser.add_argument(
        "--base_weight_default",
        type=float,
        default=1.0,
        help="Base pseudo weight when the source JSON has no pseudo_weight.",
    )
    parser.add_argument(
        "--write_projection_box",
        action="store_true",
        help="Store the 3D-projected box as locate3d_projected_box.",
    )
    return parser.parse_args()


def valid_box_xyxy(box: Optional[Sequence[float]]) -> bool:
    if box is None or len(box) != 4:
        return False
    arr = np.asarray(box, dtype=np.float64)
    if not np.isfinite(arr).all():
        return False
    if np.all(arr == -1):
        return False
    return bool(arr[2] > arr[0] and arr[3] > arr[1])


def ann_target_box_xyxy(ann: Dict) -> Optional[np.ndarray]:
    for key in ("bbox2D_tight", "bbox2D_trunc", "bbox2D_proj"):
        box = ann.get(key)
        if valid_box_xyxy(box):
            return np.asarray(box, dtype=np.float64)
    bbox = ann.get("bbox")
    if bbox is not None and len(bbox) == 4:
        x, y, w, h = [float(v) for v in bbox]
        if w > 0 and h > 0:
            return np.asarray([x, y, x + w, y + h], dtype=np.float64)
    return None


def bbox_iou_xyxy(box_a: Sequence[float], box_b: Sequence[float]) -> float:
    a = np.asarray(box_a, dtype=np.float64)
    b = np.asarray(box_b, dtype=np.float64)
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    denom = area_a + area_b - inter
    if denom <= 0:
        return 0.0
    return float(inter / denom)


def clip_box_xyxy(box: Sequence[float], width: float, height: float) -> np.ndarray:
    arr = np.asarray(box, dtype=np.float64).copy()
    arr[0::2] = np.clip(arr[0::2], 0.0, max(width - 1.0, 0.0))
    arr[1::2] = np.clip(arr[1::2], 0.0, max(height - 1.0, 0.0))
    return arr


def project_corners_to_box(
    corners: Sequence[Sequence[float]],
    K: Sequence[Sequence[float]],
    width: int,
    height: int,
) -> Tuple[Optional[np.ndarray], float, float]:
    corners_np = np.asarray(corners, dtype=np.float64)
    K_np = np.asarray(K, dtype=np.float64)
    if corners_np.shape != (8, 3) or K_np.shape != (3, 3):
        return None, 0.0, 0.0
    finite = np.isfinite(corners_np).all(axis=1)
    front = finite & (corners_np[:, 2] > 1e-4)
    front_ratio = float(front.mean()) if corners_np.shape[0] else 0.0
    if front.sum() < 2:
        return None, front_ratio, 0.0
    projected = (K_np @ corners_np[front].T).T
    uv = projected[:, :2] / np.clip(projected[:, 2:3], 1e-4, None)
    raw_box = np.asarray(
        [uv[:, 0].min(), uv[:, 1].min(), uv[:, 0].max(), uv[:, 1].max()],
        dtype=np.float64,
    )
    clipped = clip_box_xyxy(raw_box, width, height)
    if not valid_box_xyxy(clipped):
        return raw_box, front_ratio, 0.0
    image_box = np.asarray([0.0, 0.0, width - 1.0, height - 1.0], dtype=np.float64)
    frustum_iou = bbox_iou_xyxy(raw_box, image_box)
    return raw_box, front_ratio, float(np.clip(frustum_iou, 0.0, 1.0))


def box_area_score(
    target_box: np.ndarray,
    min_box_sqrt_area: float,
    good_box_sqrt_area: float,
) -> float:
    area = max(0.0, float(target_box[2] - target_box[0])) * max(
        0.0,
        float(target_box[3] - target_box[1]),
    )
    sqrt_area = math.sqrt(area)
    denom = max(good_box_sqrt_area - min_box_sqrt_area, 1e-6)
    return float(np.clip((sqrt_area - min_box_sqrt_area) / denom, 0.0, 1.0))


def border_visibility_score(
    target_box: np.ndarray,
    width: int,
    height: int,
    margin: float,
) -> float:
    if width <= 1 or height <= 1:
        return 1.0
    touches = (
        target_box[0] <= margin
        or target_box[1] <= margin
        or target_box[2] >= width - 1 - margin
        or target_box[3] >= height - 1 - margin
    )
    return 0.70 if touches else 1.0


def safe_float(value, default: float = 1.0) -> float:
    try:
        out = float(value)
    except Exception:
        return default
    if not math.isfinite(out):
        return default
    return out


def combine_weight(base: float, locate: float, mode: str) -> float:
    base = float(np.clip(base, 0.05, 1.0))
    locate = float(np.clip(locate, 0.05, 1.0))
    if mode == "replace":
        return locate
    if mode == "max":
        return max(base, locate)
    return base * locate


def process_annotation(
    ann: Dict,
    image: Dict,
    args,
) -> Tuple[Dict, Dict[str, float]]:
    out = ann
    width = int(image.get("width", 0))
    height = int(image.get("height", 0))
    target_box = ann_target_box_xyxy(ann)
    projected_box, front_ratio, frustum_score = project_corners_to_box(
        ann.get("bbox3D_cam", []),
        image.get("K", []),
        width,
        height,
    )

    if target_box is not None and projected_box is not None:
        projection_iou = bbox_iou_xyxy(projected_box, target_box)
        clipped_projection_iou = bbox_iou_xyxy(
            clip_box_xyxy(projected_box, width, height),
            clip_box_xyxy(target_box, width, height),
        )
        area_score = box_area_score(
            target_box,
            args.min_box_sqrt_area,
            args.good_box_sqrt_area,
        )
        visibility_score = (
            border_visibility_score(target_box, width, height, args.border_margin)
            * front_ratio
            * max(frustum_score, 0.25)
        )
    else:
        projection_iou = 0.0
        clipped_projection_iou = 0.0
        area_score = 0.0
        visibility_score = 0.0

    existing_quality = safe_float(
        ann.get(
            "pag_score",
            ann.get(
                "ng_consistency_score",
                ann.get("pseudo_weight", args.base_weight_default),
            ),
        ),
        args.base_weight_default,
    )
    if "moca3d_projected_corner_depth_score" in ann:
        corner_score = ann.get("moca3d_projected_corner_depth_score")
        if isinstance(corner_score, bool):
            corner_score = existing_quality
        existing_quality = min(
            existing_quality,
            safe_float(corner_score, existing_quality),
        )
    existing_quality = float(np.clip(existing_quality, 0.05, 1.0))

    evidence_den = max(
        args.projection_weight
        + args.quality_weight
        + args.area_weight
        + args.visibility_weight,
        1e-6,
    )
    evidence = (
        args.projection_weight * float(np.clip(clipped_projection_iou, 0.0, 1.0))
        + args.quality_weight * existing_quality
        + args.area_weight * area_score
        + args.visibility_weight * visibility_score
    ) / evidence_den
    evidence = float(np.clip(evidence, 0.0, 1.0))

    z = safe_float((ann.get("center_cam") or [0, 0, args.far_z])[2], args.far_z)
    z_span = max(args.far_z - args.near_z, 1e-6)
    near_progress = 1.0 - np.clip((z - args.near_z) / z_span, 0.0, 1.0)
    near_weight = float(
        np.clip(args.min_near_weight + (1.0 - args.min_near_weight) * near_progress, 0.0, 1.0)
    )
    easy_score = float(np.clip(evidence * near_weight, 0.0, 1.0))

    locate_xy = args.xy_floor + (1.0 - args.xy_floor) * evidence
    locate_z = args.z_floor + (1.0 - args.z_floor) * easy_score
    locate_dims = args.dims_floor + (1.0 - args.dims_floor) * (easy_score ** 1.25)
    locate_pose = args.pose_floor + (1.0 - args.pose_floor) * (easy_score ** 1.75)
    locate_joint = args.joint_floor + (1.0 - args.joint_floor) * (easy_score ** 1.50)

    base_joint = safe_float(ann.get("pseudo_weight", args.base_weight_default), args.base_weight_default)
    base_weights = {
        "xy": safe_float(ann.get("pseudo_weight_xy", base_joint), base_joint),
        "z": safe_float(ann.get("pseudo_weight_z", base_joint), base_joint),
        "dims": safe_float(ann.get("pseudo_weight_dims", base_joint), base_joint),
        "pose": safe_float(ann.get("pseudo_weight_pose", base_joint), base_joint),
        "joint": safe_float(ann.get("pseudo_weight_joint", base_joint), base_joint),
    }
    locate_weights = {
        "xy": locate_xy,
        "z": locate_z,
        "dims": locate_dims,
        "pose": locate_pose,
        "joint": locate_joint,
    }
    final_weights = {
        name: combine_weight(base_weights[name], locate_weights[name], args.combine_mode)
        for name in locate_weights
    }

    out["pseudo_weight_xy"] = float(final_weights["xy"])
    out["pseudo_weight_z"] = float(final_weights["z"])
    out["pseudo_weight_dims"] = float(final_weights["dims"])
    out["pseudo_weight_pose"] = float(final_weights["pose"])
    out["pseudo_weight_joint"] = float(final_weights["joint"])
    out["pseudo_weight"] = float(final_weights["joint"])
    out["locate3d_cos_enabled"] = True
    out["locate3d_2d_evidence"] = float(evidence)
    out["locate3d_projection_iou"] = float(projection_iou)
    out["locate3d_projection_iou_clipped"] = float(clipped_projection_iou)
    out["locate3d_area_score"] = float(area_score)
    out["locate3d_visibility_score"] = float(visibility_score)
    out["locate3d_near_weight"] = float(near_weight)
    out["locate3d_easy_score"] = float(easy_score)
    out["locate3d_xy_conf"] = float(locate_xy)
    out["locate3d_z_conf"] = float(locate_z)
    out["locate3d_dims_conf"] = float(locate_dims)
    out["locate3d_pose_conf"] = float(locate_pose)
    out["locate3d_joint_conf"] = float(locate_joint)
    if args.write_projection_box and projected_box is not None:
        out["locate3d_projected_box"] = [float(x) for x in projected_box.tolist()]

    stats = {
        "evidence": evidence,
        "projection_iou": projection_iou,
        "projection_iou_clipped": clipped_projection_iou,
        "area_score": area_score,
        "visibility_score": visibility_score,
        "near_weight": near_weight,
        "easy_score": easy_score,
        "xy": final_weights["xy"],
        "z": final_weights["z"],
        "dims": final_weights["dims"],
        "pose": final_weights["pose"],
        "joint": final_weights["joint"],
    }
    return out, stats


def summarize(values: List[float]) -> Dict[str, float]:
    if not values:
        return {"mean": 0.0, "p10": 0.0, "p50": 0.0, "p90": 0.0}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(arr.mean()),
        "p10": float(np.percentile(arr, 10)),
        "p50": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
    }


def main():
    args = parse_args()
    with open(args.source_json, "r") as f:
        data = json.load(f)
    output = copy.deepcopy(data)
    images = output.get("images", [])
    annotations = output.get("annotations", [])
    image_by_id = {img["id"]: img for img in images}

    if args.max_images is not None:
        keep_image_ids = {img["id"] for img in images[: args.max_images]}
        output["images"] = [img for img in images if img["id"] in keep_image_ids]
        annotations = [ann for ann in annotations if ann.get("image_id") in keep_image_ids]
        output["annotations"] = annotations
    else:
        keep_image_ids = {img["id"] for img in images}

    stats_values: Dict[str, List[float]] = {
        name: []
        for name in (
            "evidence",
            "projection_iou",
            "projection_iou_clipped",
            "area_score",
            "visibility_score",
            "near_weight",
            "easy_score",
            "xy",
            "z",
            "dims",
            "pose",
            "joint",
        )
    }
    valid3d = 0
    missing_image = 0
    for ann in tqdm(output["annotations"], desc="Locate3D-CoS pseudo weights"):
        image = image_by_id.get(ann.get("image_id"))
        if image is None:
            missing_image += 1
            continue
        if not ann.get("valid3D", True):
            continue
        valid3d += 1
        _out, ann_stats = process_annotation(ann, image, args)
        for key, value in ann_stats.items():
            stats_values[key].append(value)

    info = output.setdefault("info", {})
    info["pseudo_label_method"] = (
        str(info.get("pseudo_label_method", "source"))
        + "+locate3d_cos_factorized_reweight"
    )
    info["locate3d_cos_factorized_reweight"] = True
    info["locate3d_cos_source_json"] = args.source_json
    info["locate3d_cos_combine_mode"] = args.combine_mode
    info["locate3d_cos_near_z"] = float(args.near_z)
    info["locate3d_cos_far_z"] = float(args.far_z)

    os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump(output, f)

    summary = {
        "source_json": args.source_json,
        "output_json": args.output_json,
        "images": len(output.get("images", [])),
        "annotations": len(output.get("annotations", [])),
        "valid3d": valid3d,
        "missing_image": missing_image,
        "combine_mode": args.combine_mode,
        "metrics": {name: summarize(values) for name, values in stats_values.items()},
    }
    if args.stats_json:
        os.makedirs(os.path.dirname(args.stats_json) or ".", exist_ok=True)
        with open(args.stats_json, "w") as f:
            json.dump(summary, f, indent=2)

    print(f"Wrote {args.output_json}")
    print(
        {
            "images": summary["images"],
            "annotations": summary["annotations"],
            "valid3d": summary["valid3d"],
            "evidence_mean": round(summary["metrics"]["evidence"]["mean"], 4),
            "joint_mean": round(summary["metrics"]["joint"]["mean"], 4),
            "pose_mean": round(summary["metrics"]["pose"]["mean"], 4),
        }
    )


if __name__ == "__main__":
    main()
