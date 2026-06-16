#!/usr/bin/env python3
"""Prepare training pairs for Boxer-Residual-LIFT.

The script matches Boxer pseudo annotations to a target 3D annotation source
by image/category/2D IoU and saves compact tensors for train_lifthead.py.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from typing import Dict, List, Mapping, Tuple

import numpy as np
import torch
from tqdm import tqdm

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from tools.lifthead_common import (  # noqa: E402
    box_iou_xyxy,
    build_category_index,
    build_feature_vector_with_roi,
    cached_roi_feature,
    feature_names_with_roi,
    finite_float,
    load_image_rgb,
    load_roi_feature_cache,
    target_from_pair,
    xyxy_from_ann,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build residual LiftHead training tensors.")
    parser.add_argument("--source_json", required=True, help="Boxer pseudo-label Omni3D JSON.")
    parser.add_argument("--image_root", default="datasets", help="Dataset root used for ROI image crops.")
    parser.add_argument(
        "--target_json",
        required=True,
        help="Target 3D labels, e.g. Omni3D GT or high-quality original PCA pseudo labels.",
    )
    parser.add_argument("--output", required=True, help="Output .pth file.")
    parser.add_argument("--min_match_iou", type=float, default=0.30)
    parser.add_argument("--min_source_quality", type=float, default=0.0)
    parser.add_argument("--min_source_proj_iou", type=float, default=0.0)
    parser.add_argument("--max_source_depth_error", type=float, default=10.0)
    parser.add_argument("--max_pairs_per_image", type=int, default=0, help="0 means no limit.")
    parser.add_argument(
        "--weight_quality",
        action="store_true",
        help="Use Boxer quality/projection/depth metrics as sample weights.",
    )
    parser.add_argument(
        "--roi_feature_mode",
        choices=["none", "color_grid"],
        default="none",
        help="Append lightweight ROI image-crop features to the LiftHead input.",
    )
    parser.add_argument("--roi_grid_size", type=int, default=4)
    parser.add_argument("--roi_context_scale", type=float, default=1.15)
    parser.add_argument(
        "--roi_feature_cache",
        default=None,
        help="Optional cached ROI features from extract_lifthead_roi_features.py.",
    )
    return parser.parse_args()


def load_json(path: str) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def is_valid_3d(ann: Mapping) -> bool:
    if not bool(ann.get("valid3D", False)):
        return False
    center = np.asarray(ann.get("center_cam", []), dtype=np.float32).reshape(-1)
    dims = np.asarray(ann.get("dimensions", []), dtype=np.float32).reshape(-1)
    if center.shape[0] != 3 or dims.shape[0] != 3:
        return False
    if not (np.all(np.isfinite(center)) and np.all(np.isfinite(dims))):
        return False
    if center[2] <= 0.05 or np.any(dims <= 0):
        return False
    return True


def source_ok(ann: Mapping, args: argparse.Namespace) -> bool:
    if not is_valid_3d(ann):
        return False
    if finite_float(ann.get("boxer_quality"), 1.0) < args.min_source_quality:
        return False
    if finite_float(ann.get("boxer_projection_iou"), 1.0) < args.min_source_proj_iou:
        return False
    if finite_float(ann.get("boxer_depth_rel_error"), 0.0) > args.max_source_depth_error:
        return False
    box = xyxy_from_ann(ann)
    if np.any(box < 0):
        return False
    return True


def group_annotations(anns: List[Mapping], keep_fn) -> Dict[Tuple[int, int], List[Mapping]]:
    grouped: Dict[Tuple[int, int], List[Mapping]] = defaultdict(list)
    for ann in anns:
        if not keep_fn(ann):
            continue
        grouped[(int(ann["image_id"]), int(ann["category_id"]))].append(ann)
    return grouped


def greedy_match(
    sources: List[Mapping],
    targets: List[Mapping],
    min_iou: float,
) -> List[Tuple[Mapping, Mapping, float]]:
    pairs = []
    for si, src in enumerate(sources):
        src_box = xyxy_from_ann(src)
        for ti, tgt in enumerate(targets):
            iou = box_iou_xyxy(src_box, xyxy_from_ann(tgt))
            if iou >= min_iou:
                pairs.append((iou, si, ti))
    pairs.sort(reverse=True)
    used_s = set()
    used_t = set()
    matched = []
    for iou, si, ti in pairs:
        if si in used_s or ti in used_t:
            continue
        used_s.add(si)
        used_t.add(ti)
        matched.append((sources[si], targets[ti], float(iou)))
    return matched


def sample_weight(source_ann: Mapping, match_iou: float, use_quality: bool) -> float:
    if not use_quality:
        return 1.0
    quality = finite_float(source_ann.get("boxer_quality"), 0.5)
    proj_iou = finite_float(source_ann.get("boxer_projection_iou"), 0.5)
    depth_err = finite_float(source_ann.get("boxer_depth_rel_error"), 0.2)
    depth_term = float(np.clip(1.0 - depth_err, 0.1, 1.0))
    return float(np.clip(0.25 + 0.75 * quality * proj_iou * depth_term * match_iou, 0.1, 1.0))


def main() -> None:
    args = parse_args()
    source = load_json(args.source_json)
    target = load_json(args.target_json)
    image_by_id = {int(im["id"]): im for im in source["images"]}
    if not image_by_id:
        raise ValueError("source_json has no images")

    cat_to_idx = build_category_index(source.get("categories", target.get("categories", [])))
    target_grouped = group_annotations(target.get("annotations", []), is_valid_3d)
    source_grouped = group_annotations(source.get("annotations", []), lambda ann: source_ok(ann, args))

    features = []
    category_indices = []
    targets = []
    weights = []
    match_ious = []
    source_ann_ids = []
    target_ann_ids = []
    image_ids = []
    image_cache: Dict[int, np.ndarray | None] = {}
    use_roi = args.roi_feature_mode != "none"
    roi_cache, roi_cache_names, roi_cache_config = load_roi_feature_cache(args.roi_feature_cache)
    roi_cache_dim = len(roi_cache_names)

    def get_image_rgb(image: Mapping) -> np.ndarray | None:
        if not use_roi:
            return None
        image_id = int(image["id"])
        if image_id not in image_cache:
            image_cache[image_id] = load_image_rgb(args.image_root, image)
        return image_cache[image_id]

    total_groups = 0
    total_matches = 0
    for key, src_anns in tqdm(source_grouped.items(), desc="Matching source to target"):
        tgt_anns = target_grouped.get(key)
        if not tgt_anns:
            continue
        total_groups += 1
        matches = greedy_match(src_anns, tgt_anns, args.min_match_iou)
        if args.max_pairs_per_image > 0:
            matches = matches[: args.max_pairs_per_image]
        for src_ann, tgt_ann, iou in matches:
            image = image_by_id[int(src_ann["image_id"])]
            feature = build_feature_vector_with_roi(
                src_ann,
                image,
                get_image_rgb(image),
                roi_feature_mode=args.roi_feature_mode,
                roi_grid_size=args.roi_grid_size,
                roi_context_scale=args.roi_context_scale,
            )
            if roi_cache_dim > 0:
                feature = np.concatenate(
                    [feature, cached_roi_feature(src_ann, roi_cache, roi_cache_dim)],
                    axis=0,
                ).astype(np.float32)
            features.append(feature)
            category_indices.append(cat_to_idx.get(int(src_ann["category_id"]), 0))
            targets.append(target_from_pair(src_ann, tgt_ann))
            weights.append(sample_weight(src_ann, iou, args.weight_quality))
            match_ious.append(iou)
            source_ann_ids.append(int(src_ann.get("id", -1)))
            target_ann_ids.append(int(tgt_ann.get("id", -1)))
            image_ids.append(int(src_ann["image_id"]))
        total_matches += len(matches)

    if not features:
        raise RuntimeError(
            "No matched pairs were produced. Lower --min_match_iou or verify source/target JSON alignment."
        )

    out = {
        "features": torch.tensor(np.stack(features), dtype=torch.float32),
        "category_indices": torch.tensor(category_indices, dtype=torch.long),
        "targets": torch.tensor(np.stack(targets), dtype=torch.float32),
        "weights": torch.tensor(weights, dtype=torch.float32),
        "match_ious": torch.tensor(match_ious, dtype=torch.float32),
        "image_ids": torch.tensor(image_ids, dtype=torch.long),
        "source_ann_ids": torch.tensor(source_ann_ids, dtype=torch.long),
        "target_ann_ids": torch.tensor(target_ann_ids, dtype=torch.long),
        "feature_names": feature_names_with_roi(args.roi_feature_mode, args.roi_grid_size)
        + roi_cache_names,
        "roi_feature_config": {
            "mode": args.roi_feature_mode,
            "grid_size": int(args.roi_grid_size),
            "context_scale": float(args.roi_context_scale),
        },
        "roi_feature_cache_config": roi_cache_config,
        "roi_feature_cache_names": roi_cache_names,
        "category_id_to_index": cat_to_idx,
        "source_json": os.path.abspath(args.source_json),
        "target_json": os.path.abspath(args.target_json),
        "stats": {
            "source_annotations": len(source.get("annotations", [])),
            "target_annotations": len(target.get("annotations", [])),
            "source_groups": len(source_grouped),
            "target_groups": len(target_grouped),
            "matched_groups": total_groups,
            "matched_pairs": total_matches,
            "min_match_iou": args.min_match_iou,
        },
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    torch.save(out, args.output)
    print(f"Wrote LiftHead data: {args.output}")
    print(out["stats"])


if __name__ == "__main__":
    main()
