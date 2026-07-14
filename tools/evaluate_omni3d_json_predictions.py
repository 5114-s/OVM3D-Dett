#!/usr/bin/env python3
"""Evaluate an Omni3D-style prediction JSON with the Omni3D AP evaluator.

This is useful for OVMono3D-like direct-lift experiments where a pseudo-label
JSON is treated as model predictions rather than as training labels.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List, Mapping, Sequence, Set

import numpy as np
from detectron2.structures import BoxMode

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from cubercnn.data import Omni3D, get_filter_settings_from_cfg  # noqa: E402
from cubercnn.evaluation.omni3d_evaluation import _evaluate_predictions_on_omni  # noqa: E402


OVMONO3D_BASE_CATEGORIES = (
    "chair,table,cabinet,car,lamp,books,sofa,pedestrian,picture,window,pillow,"
    "truck,door,blinds,sink,shelves,television,shoes,cup,bottle,bookcase,laptop,"
    "desk,cereal box,floor mat,traffic cone,mirror,barrier,counter,camera,bicycle,"
    "toilet,bus,bed,refrigerator,trailer,box,oven,clothes,van,towel,motorcycle,"
    "night stand,stove,machine,stationery,bathtub,cyclist,curtain,bin"
)
OVMONO3D_NOVEL_CATEGORIES = (
    "monitor,bag,dresser,board,printer,keyboard,painting,drawers,microwave,"
    "computer,kitchen pan,potted plant,tissues,rack,tray,toys,phone,podium,"
    "cart,soundsystem,fire place,tram"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Omni3D-style prediction JSON.")
    parser.add_argument("--pred_json", required=True)
    parser.add_argument("--gt_json", required=True)
    parser.add_argument("--output_dir", default="outputs/eval_json_predictions")
    parser.add_argument(
        "--cat_mode",
        choices=["custom", "base", "novel", "all"],
        default="custom",
        help="Category subset to evaluate. custom uses --categories.",
    )
    parser.add_argument("--categories", default="", help="Comma-separated category names for custom mode.")
    parser.add_argument(
        "--bbox_field",
        default="bbox2D_tight,bbox2D_proj,bbox2D_trunc,bbox",
        help="Comma-separated prediction 2D bbox fields, first valid one is used.",
    )
    parser.add_argument("--score_field", default="score")
    parser.add_argument("--min_score", type=float, default=0.0)
    parser.add_argument("--max_dets_per_image", type=int, default=300)
    parser.add_argument("--only_2d", action="store_true")
    parser.add_argument(
        "--eval_prox",
        action="store_true",
        help="Enable SUNRGBD/Objectron near/medium/far AP analysis.",
    )
    return parser.parse_args()


def load_json(path: str) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def parse_categories(raw: str) -> List[str]:
    return [x.strip() for x in raw.split(",") if x.strip()]


def category_names_for_mode(mode: str, custom: str, gt: Mapping) -> List[str]:
    if mode == "base":
        return parse_categories(OVMONO3D_BASE_CATEGORIES)
    if mode == "novel":
        return parse_categories(OVMONO3D_NOVEL_CATEGORIES)
    if mode == "custom":
        cats = parse_categories(custom)
        if cats:
            return cats
    return [str(cat["name"]) for cat in gt.get("categories", [])]


def cat_maps(data: Mapping) -> tuple[Dict[int, str], Dict[str, int]]:
    id_to_name: Dict[int, str] = {}
    name_to_id: Dict[str, int] = {}
    for cat in data.get("categories", []):
        cat_id = int(cat["id"])
        name = str(cat.get("name", cat.get("name_readable", cat_id)))
        id_to_name[cat_id] = name
        name_to_id[name] = cat_id
    return id_to_name, name_to_id


def valid_xyxy(box: Sequence[float]) -> bool:
    arr = np.asarray(box, dtype=np.float32).reshape(-1)
    return arr.shape[0] >= 4 and np.all(np.isfinite(arr[:4])) and arr[2] > arr[0] and arr[3] > arr[1]


def choose_bbox(ann: Mapping, fields: Sequence[str]) -> List[float] | None:
    for field in fields:
        if field not in ann:
            continue
        box = np.asarray(ann[field], dtype=np.float32).reshape(-1)
        if box.shape[0] < 4 or not np.all(np.isfinite(box[:4])):
            continue
        box = box[:4].copy()
        if np.all(box == -1):
            continue
        if field == "bbox":
            if box[2] <= 0 or box[3] <= 0:
                continue
            return [float(x) for x in box.tolist()]
        if valid_xyxy(box):
            xywh = BoxMode.convert(box.tolist(), BoxMode.XYXY_ABS, BoxMode.XYWH_ABS)
            return [float(x) for x in xywh]
    return None


def valid_3d(ann: Mapping) -> bool:
    if not bool(ann.get("valid3D", True)):
        return False
    corners = np.asarray(ann.get("bbox3D_cam", ann.get("bbox3D", [])), dtype=np.float32)
    if corners.shape != (8, 3) or not np.all(np.isfinite(corners)):
        return False
    dims = np.asarray(ann.get("dimensions", [1.0, 1.0, 1.0]), dtype=np.float32).reshape(-1)
    center = np.asarray(ann.get("center_cam", [0.0, 0.0, float(np.mean(corners[:, 2]))]), dtype=np.float32).reshape(-1)
    return dims.shape[0] >= 3 and center.shape[0] >= 3 and np.all(dims[:3] > 0) and float(center[2]) > 0.05


def build_results(
    pred: Mapping,
    category_names: Sequence[str],
    bbox_fields: Sequence[str],
    score_field: str,
    min_score: float,
    max_dets_per_image: int,
) -> List[dict]:
    id_to_name, name_to_id = cat_maps(pred)
    keep_names: Set[str] = set(category_names)
    keep_ids: Set[int] = {name_to_id[name] for name in keep_names if name in name_to_id}
    by_image: Dict[int, List[dict]] = {}

    for ann in pred.get("annotations", []):
        cat_id = int(ann.get("category_id", -1))
        cat_name = str(ann.get("category_name", id_to_name.get(cat_id, cat_id)))
        if keep_names and cat_name not in keep_names and cat_id not in keep_ids:
            continue
        if not valid_3d(ann):
            continue
        bbox = choose_bbox(ann, bbox_fields)
        if bbox is None:
            continue
        score = float(ann.get(score_field, ann.get("boxer_quality", 1.0)))
        if not np.isfinite(score) or score < min_score:
            continue
        corners = np.asarray(ann.get("bbox3D_cam", ann.get("bbox3D")), dtype=np.float32).reshape(8, 3)
        result = {
            "image_id": int(ann["image_id"]),
            "category_id": cat_id,
            "bbox": bbox,
            "score": float(score),
            "depth": float(np.mean(corners[:, 2])),
            "bbox3D": [[float(x) for x in row] for row in corners.tolist()],
            "center_cam": [float(x) for x in ann.get("center_cam", [0.0, 0.0, float(np.mean(corners[:, 2]))])],
            "dimensions": [float(x) for x in ann.get("dimensions", [1.0, 1.0, 1.0])],
            "pose": ann.get("R_cam", ann.get("pose", np.eye(3).tolist())),
        }
        by_image.setdefault(int(ann["image_id"]), []).append(result)

    results = []
    for image_id, anns in by_image.items():
        anns.sort(key=lambda x: float(x["score"]), reverse=True)
        results.extend(anns[:max_dets_per_image])
    return results


def main() -> None:
    args = parse_args()
    pred = load_json(args.pred_json)
    gt = load_json(args.gt_json)
    category_names = category_names_for_mode(args.cat_mode, args.categories, gt)
    bbox_fields = [x.strip() for x in args.bbox_field.split(",") if x.strip()]

    filter_settings = get_filter_settings_from_cfg(None)
    filter_settings["category_names"] = list(category_names)
    filter_settings["trunc_2D_boxes"] = False
    filter_settings["modal_2D_boxes"] = False

    omni_gt = Omni3D([args.gt_json], filter_settings)
    results = build_results(
        pred,
        category_names,
        bbox_fields,
        args.score_field,
        args.min_score,
        args.max_dets_per_image,
    )
    if not results:
        raise RuntimeError("No valid predictions after filtering.")

    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "omni_instances_results.json"), "w") as f:
        json.dump(results, f)
    evals, log_strs = _evaluate_predictions_on_omni(
        omni_gt,
        results,
        "bbox",
        only_2d=args.only_2d,
        eval_prox=args.eval_prox,
    )
    summary = {
        "pred_json": os.path.abspath(args.pred_json),
        "gt_json": os.path.abspath(args.gt_json),
        "cat_mode": args.cat_mode,
        "category_names": list(category_names),
        "num_predictions": len(results),
        "log_str_2D": log_strs.get("2D"),
        "log_str_3D": log_strs.get("3D"),
    }
    with open(os.path.join(args.output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
