#!/usr/bin/env python3
"""Apply a trained Boxer-Residual-LIFT head to an Omni3D JSON."""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import sys
from typing import Dict, List, Mapping, Set

import numpy as np
import torch
from tqdm import tqdm

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from tools.lifthead_common import (  # noqa: E402
    ResidualLiftHead,
    apply_residual,
    box_iou_xyxy,
    build_feature_vector_with_roi,
    cached_roi_feature,
    cuboid_corners,
    finite_float,
    load_depth_map,
    load_image_rgb,
    load_roi_feature_cache,
    projected_box_from_corners,
    rotation_y,
    xyxy_from_ann,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Correct Boxer Omni3D labels with Residual-LIFT.")
    parser.add_argument("--source_json", required=True, help="Input Boxer Omni3D JSON.")
    parser.add_argument("--image_root", default="datasets", help="Dataset root used for ROI image crops.")
    parser.add_argument("--checkpoint", required=True, help="LiftHead checkpoint, e.g. best.pth.")
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--force_cpu", action="store_true")
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--blend", type=float, default=0.7)
    parser.add_argument("--max_center_shift_ratio", type=float, default=0.35)
    parser.add_argument("--max_log_dim_delta", type=float, default=0.7)
    parser.add_argument("--max_yaw_delta", type=float, default=math.pi / 3)
    parser.add_argument("--update_yaw", action="store_true", help="Replace R_cam by a yaw-only rotation.")
    parser.add_argument(
        "--min_after_proj_iou",
        type=float,
        default=0.0,
        help="If >0 and corrected box projection IoU is lower, keep the original annotation.",
    )
    parser.add_argument("--max_images", type=int, default=None)
    parser.add_argument("--no_repair_score", action="store_true")
    parser.add_argument(
        "--roi_feature_mode",
        choices=["auto", "none", "color_grid"],
        default="auto",
        help="Use checkpoint ROI config by default; override only for debugging.",
    )
    parser.add_argument(
        "--roi_feature_cache",
        default=None,
        help="Optional cached ROI features matching source_json annotation ids.",
    )
    parser.add_argument("--depth_root", default=None, help="Optional cached metric depth root.")
    parser.add_argument("--depth_split", default=None, help="Depth split under --depth_root.")
    parser.add_argument(
        "--depth_feature_mode",
        choices=["auto", "none", "box_stats"],
        default="auto",
        help="Use checkpoint depth config by default; override only for debugging.",
    )
    parser.add_argument(
        "--only_categories",
        default="",
        help="Optional comma-separated category names/ids to correct. Others are kept unchanged.",
    )
    parser.add_argument(
        "--skip_categories",
        default="",
        help="Optional comma-separated category names/ids to keep unchanged.",
    )
    return parser.parse_args()


def load_json(path: str) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def is_valid_ann(ann: dict) -> bool:
    if not bool(ann.get("valid3D", False)):
        return False
    center = np.asarray(ann.get("center_cam", []), dtype=np.float32).reshape(-1)
    dims = np.asarray(ann.get("dimensions", []), dtype=np.float32).reshape(-1)
    return center.shape[0] == 3 and dims.shape[0] == 3 and np.all(np.isfinite(center)) and np.all(np.isfinite(dims)) and center[2] > 0.05 and np.all(dims > 0)


def _normalize_category_token(value: object) -> str:
    return str(value).strip().lower().replace("_", " ")


def parse_category_filter(raw: str) -> Set[str]:
    if not raw:
        return set()
    return {_normalize_category_token(tok) for tok in raw.split(",") if tok.strip()}


def category_names_by_id(categories: List[Mapping]) -> Dict[int, str]:
    names: Dict[int, str] = {}
    for cat in categories or []:
        if "id" not in cat:
            continue
        names[int(cat["id"])] = str(cat.get("name", cat.get("name_readable", cat["id"])))
    return names


def category_allowed(
    ann: Mapping,
    names_by_id: Mapping[int, str],
    only: Set[str],
    skip: Set[str],
) -> bool:
    cat_id = int(ann.get("category_id", -1))
    tokens = {
        _normalize_category_token(cat_id),
        _normalize_category_token(names_by_id.get(cat_id, cat_id)),
    }
    if only and tokens.isdisjoint(only):
        return False
    if skip and not tokens.isdisjoint(skip):
        return False
    return True


def main() -> None:
    args = parse_args()
    if torch.cuda.is_available() and not args.force_cpu:
        torch.cuda.set_device(args.gpu)
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    source = load_json(args.source_json)
    only_categories = parse_category_filter(args.only_categories)
    skip_categories = parse_category_filter(args.skip_categories)
    names_by_id = category_names_by_id(source.get("categories", []))
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    model_args = ckpt["model_args"]
    model = ResidualLiftHead(**model_args)
    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()
    feature_mean = ckpt["feature_mean"].float().to(device)
    feature_std = ckpt["feature_std"].float().clamp(min=1e-4).to(device)
    cat_to_idx = {int(k): int(v) for k, v in ckpt.get("category_id_to_index", {}).items()}
    roi_cfg = dict(ckpt.get("roi_feature_config") or {"mode": "none"})
    if args.roi_feature_mode != "auto":
        roi_cfg["mode"] = args.roi_feature_mode
    roi_cfg.setdefault("mode", "none")
    roi_cfg.setdefault("grid_size", 4)
    roi_cfg.setdefault("context_scale", 1.15)
    depth_cfg = dict(ckpt.get("depth_feature_config") or {"mode": "none"})
    if args.depth_feature_mode != "auto":
        depth_cfg["mode"] = args.depth_feature_mode
    depth_cfg.setdefault("mode", "none")
    depth_cfg.setdefault("context_scale", 1.05)
    if args.depth_root:
        depth_cfg["root"] = args.depth_root
    if args.depth_split:
        depth_cfg["split"] = args.depth_split
    roi_cache, roi_cache_names, _roi_cache_config = load_roi_feature_cache(args.roi_feature_cache)
    expected_cache_names = list(ckpt.get("roi_feature_cache_names") or [])
    roi_cache_dim = len(expected_cache_names)
    if roi_cache_dim > 0 and not args.roi_feature_cache:
        raise RuntimeError(
            "This checkpoint was trained with cached ROI features. "
            "Please pass --roi_feature_cache for the source_json being corrected."
        )
    if roi_cache_dim > 0 and len(roi_cache_names) != roi_cache_dim:
        raise RuntimeError(
            f"ROI cache dim mismatch: cache has {len(roi_cache_names)}, checkpoint expects {roi_cache_dim}."
        )

    output = copy.deepcopy(source)
    image_by_id = {int(im["id"]): im for im in output.get("images", [])}
    if args.max_images is not None:
        selected = {int(im["id"]) for im in output.get("images", [])[: args.max_images]}
    else:
        selected = None

    ann_indices: List[int] = []
    features = []
    cats = []
    image_cache: Dict[int, np.ndarray | None] = {}
    depth_cache: Dict[int, np.ndarray | None] = {}
    use_roi = roi_cfg.get("mode", "none") != "none"
    use_depth = depth_cfg.get("mode", "none") != "none"

    def get_image_rgb(image: dict) -> np.ndarray | None:
        if not use_roi:
            return None
        image_id = int(image["id"])
        if image_id not in image_cache:
            image_cache[image_id] = load_image_rgb(args.image_root, image)
        return image_cache[image_id]

    def get_depth_map(image: dict) -> np.ndarray | None:
        if not use_depth:
            return None
        image_id = int(image["id"])
        if image_id not in depth_cache:
            depth_cache[image_id] = load_depth_map(
                depth_cfg.get("root"),
                image_id,
                depth_cfg.get("split"),
            )
        return depth_cache[image_id]

    for idx, ann in enumerate(output.get("annotations", [])):
        if selected is not None and int(ann.get("image_id", -1)) not in selected:
            continue
        if not is_valid_ann(ann):
            continue
        if not category_allowed(ann, names_by_id, only_categories, skip_categories):
            continue
        image = image_by_id.get(int(ann["image_id"]))
        if image is None:
            continue
        ann_indices.append(idx)
        feature = build_feature_vector_with_roi(
            ann,
            image,
            get_image_rgb(image),
            roi_feature_mode=str(roi_cfg.get("mode", "none")),
            roi_grid_size=int(roi_cfg.get("grid_size", 4)),
            roi_context_scale=float(roi_cfg.get("context_scale", 1.15)),
            depth_map=get_depth_map(image),
            depth_feature_mode=str(depth_cfg.get("mode", "none")),
            depth_context_scale=float(depth_cfg.get("context_scale", 1.05)),
        )
        if roi_cache_dim > 0:
            feature = np.concatenate(
                [feature, cached_roi_feature(ann, roi_cache, roi_cache_dim)],
                axis=0,
            ).astype(np.float32)
        features.append(feature)
        cats.append(cat_to_idx.get(int(ann.get("category_id", -1)), 0))

    if features and len(features[0]) != int(model_args["feature_dim"]):
        raise RuntimeError(
            f"Feature dimension mismatch: built {len(features[0])}, checkpoint expects "
            f"{model_args['feature_dim']}. Check --image_root/--roi_feature_mode and checkpoint ROI config."
        )

    stats = {
        "source_annotations": len(output.get("annotations", [])),
        "candidate_annotations": len(ann_indices),
        "only_categories": sorted(only_categories),
        "skip_categories": sorted(skip_categories),
        "skipped_by_category": len(
            [
                ann
                for ann in output.get("annotations", [])
                if is_valid_ann(ann)
                and not category_allowed(ann, names_by_id, only_categories, skip_categories)
            ]
        ),
        "corrected": 0,
        "kept_original_projection_gate": 0,
        "score_repaired": 0,
    }

    for start in tqdm(range(0, len(ann_indices), args.batch_size), desc="Residual-LIFT inference"):
        end = min(start + args.batch_size, len(ann_indices))
        batch_features = torch.tensor(np.stack(features[start:end]), dtype=torch.float32, device=device)
        batch_features = (batch_features - feature_mean) / feature_std
        batch_cats = torch.tensor(cats[start:end], dtype=torch.long, device=device)
        with torch.no_grad():
            pred = model(batch_features, batch_cats).detach().cpu().numpy()

        for local_idx, residual in enumerate(pred):
            ann_idx = ann_indices[start + local_idx]
            ann = output["annotations"][ann_idx]
            original = copy.deepcopy(ann)
            image = image_by_id[int(ann["image_id"])]
            R_old = np.asarray(ann.get("R_cam", np.eye(3)), dtype=np.float32).reshape(3, 3)
            new_center, new_dims, new_yaw, metrics = apply_residual(
                ann,
                residual,
                max_center_shift_ratio=args.max_center_shift_ratio,
                max_log_dim_delta=args.max_log_dim_delta,
                max_yaw_delta=args.max_yaw_delta,
                blend=args.blend,
            )
            R_new = rotation_y(new_yaw) if args.update_yaw else R_old
            corners = cuboid_corners(new_center, new_dims, R_new)
            proj_box = projected_box_from_corners(
                corners,
                image["K"],
                float(image.get("width", 1.0)),
                float(image.get("height", 1.0)),
            )
            proj_iou = box_iou_xyxy(proj_box, xyxy_from_ann(original)) if proj_box[0] >= 0 else 0.0
            if args.min_after_proj_iou > 0 and proj_iou < args.min_after_proj_iou:
                if not args.no_repair_score:
                    score = finite_float(original.get("score"), float("nan"))
                    if not math.isfinite(score):
                        original["score"] = finite_float(original.get("boxer_quality"), 0.05)
                        stats["score_repaired"] += 1
                output["annotations"][ann_idx] = original
                output["annotations"][ann_idx]["lifthead_kept_original_reason"] = "projection_gate"
                output["annotations"][ann_idx]["lifthead_after_proj_iou"] = float(proj_iou)
                stats["kept_original_projection_gate"] += 1
                continue

            ann["lifthead_corrected"] = True
            ann["lifthead_checkpoint"] = os.path.abspath(args.checkpoint)
            ann["lifthead_raw_center_cam"] = original.get("center_cam")
            ann["lifthead_raw_dimensions"] = original.get("dimensions")
            ann["lifthead_after_proj_iou"] = float(proj_iou)
            ann.update(metrics)
            ann["center_cam"] = [float(x) for x in new_center.tolist()]
            ann["dimensions"] = [float(x) for x in new_dims.tolist()]
            ann["R_cam"] = [[float(x) for x in row] for row in R_new.tolist()]
            ann["bbox3D_cam"] = [[float(x) for x in row] for row in corners.tolist()]
            ann["bbox2D_proj"] = [float(x) for x in proj_box]
            ann["depth_error"] = finite_float(ann.get("boxer_depth_rel_error"), finite_float(ann.get("depth_error"), -1.0))
            if not args.no_repair_score:
                score = finite_float(ann.get("score"), float("nan"))
                if not math.isfinite(score):
                    ann["score"] = finite_float(ann.get("boxer_quality"), 0.05)
                    stats["score_repaired"] += 1
            stats["corrected"] += 1

    info = output.setdefault("info", {})
    info["pseudo_label_method"] = "BoxerNet+ResidualLiftHead"
    info["lifthead_source_json"] = os.path.abspath(args.source_json)
    info["lifthead_checkpoint"] = os.path.abspath(args.checkpoint)
    info["lifthead_blend"] = args.blend
    info["lifthead_update_yaw"] = bool(args.update_yaw)
    info["lifthead_roi_feature_config"] = roi_cfg
    info["lifthead_depth_feature_config"] = depth_cfg
    info["lifthead_roi_feature_cache"] = os.path.abspath(args.roi_feature_cache) if args.roi_feature_cache else None
    info["lifthead_stats"] = stats

    os.makedirs(os.path.dirname(os.path.abspath(args.output_json)), exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump(output, f)
    print(f"Wrote Residual-LIFT Omni3D JSON: {args.output_json}")
    print(stats)


if __name__ == "__main__":
    main()
