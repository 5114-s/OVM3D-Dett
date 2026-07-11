#!/usr/bin/env python3
"""A/B test MoCA3D-Cube with GT tight boxes vs GroundedSAM boxes.

For the same SUNRGBD GT objects, this script runs MoCA3D-Cube twice:

  1. image + GT bbox2D_tight
  2. image + matched GroundedSAM box of the same class

It then compares the predicted 3D boxes against the GT 3D boxes. The goal is to
measure whether MoCA's poor pseudo-label quality is mainly caused by the 2D box
distribution mismatch.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from run_moca3d_omni3d import (
    DEFAULT_MOCA_ROOT,
    bbox_iou_xyxy,
    box_to_center_dims_R,
    build_models,
    category_maps,
    image_to_tensor,
    letterbox_image,
    load_gsam_info,
    load_json,
    project_points,
    run_moca_on_image,
    valid_padding_mask,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Compare MoCA with GT 2D boxes vs GroundedSAM boxes")
    parser.add_argument("--json_file", default="datasets/Omni3D/SUNRGBD_train.json")
    parser.add_argument("--image_root", default="datasets")
    parser.add_argument("--dataset", default="SUNRGBD")
    parser.add_argument("--split", default="train")
    parser.add_argument("--original_pseudo_root", default="pseudo_label")
    parser.add_argument("--output_json", default="outputs/moca3d_gt_vs_gsam_50/compare.json")
    parser.add_argument("--num_samples", type=int, default=50)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--min_match_iou", type=float, default=0.05)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--force_cpu", action="store_true")
    parser.add_argument("--moca_root", default=str(DEFAULT_MOCA_ROOT))
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--precision", choices=["float32", "float16", "bfloat16"], default="float32")
    parser.add_argument("--prefer_ema", action="store_true")
    parser.add_argument("--moca_checkpoint", default=None)
    parser.add_argument("--cube_checkpoint", default=None)
    parser.add_argument("--joint_checkpoint", default=None)
    parser.add_argument("--moca_config", default=None)
    parser.add_argument("--cube_config", default=None)
    return parser.parse_args()


def valid_xyxy(box: Optional[Sequence[float]]) -> bool:
    if box is None or len(box) < 4:
        return False
    x1, y1, x2, y2 = [float(v) for v in box[:4]]
    return math.isfinite(x1 + y1 + x2 + y2) and x2 > x1 and y2 > y1 and x1 >= 0 and y1 >= 0


def valid_gt_ann(ann: dict) -> bool:
    if not ann.get("valid3D", False):
        return False
    if not valid_xyxy(ann.get("bbox2D_tight")):
        return False
    center = np.asarray(ann.get("center_cam", []), dtype=np.float32)
    dims = np.asarray(ann.get("dimensions", []), dtype=np.float32)
    box3d = np.asarray(ann.get("bbox3D_cam", []), dtype=np.float32)
    return center.shape == (3,) and dims.shape == (3,) and box3d.shape == (8, 3) and np.isfinite(center).all() and np.isfinite(dims).all() and np.isfinite(box3d).all() and center[2] > 0 and np.all(dims > 0)


def normalize_box(box: Sequence[float], width: int, height: int) -> Optional[List[float]]:
    if box is None or len(box) < 4:
        return None
    x1, y1, x2, y2 = [float(v) for v in box[:4]]
    x1, x2 = sorted((x1, x2))
    y1, y2 = sorted((y1, y2))
    x1 = max(0.0, min(float(width - 1), x1))
    y1 = max(0.0, min(float(height - 1), y1))
    x2 = max(0.0, min(float(width), x2))
    y2 = max(0.0, min(float(height), y2))
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def best_gsam_box(record: dict, category_name: str, gt_box: Sequence[float], width: int, height: int) -> Tuple[Optional[List[float]], float, float]:
    boxes = np.asarray(record.get("boxes", []), dtype=np.float32)
    phrases = list(record.get("phrases", []))
    conf = np.asarray(record.get("conf", []), dtype=np.float32)
    if boxes.ndim != 2 or boxes.shape[1] < 4:
        return None, 0.0, 0.0
    best_box, best_iou, best_score = None, 0.0, 0.0
    for idx, raw in enumerate(boxes):
        if idx >= len(phrases) or str(phrases[idx]) != str(category_name):
            continue
        box = normalize_box(raw, width, height)
        if box is None:
            continue
        iou = bbox_iou_xyxy(box, gt_box)
        if iou > best_iou:
            best_box = box
            best_iou = float(iou)
            best_score = float(conf[idx]) if idx < len(conf) else 1.0
    return best_box, best_iou, best_score


def summarize(vals: Sequence[float]) -> dict:
    arr = np.asarray([float(v) for v in vals if math.isfinite(float(v))], dtype=np.float32)
    if arr.size == 0:
        return {"n": 0}
    return {
        "n": int(arr.size),
        "mean": round(float(arr.mean()), 6),
        "p10": round(float(np.percentile(arr, 10)), 6),
        "p50": round(float(np.percentile(arr, 50)), 6),
        "p90": round(float(np.percentile(arr, 90)), 6),
        "max": round(float(arr.max()), 6),
    }


def sorted_dim_rel_error(pred_dims: Sequence[float], gt_dims: Sequence[float]) -> float:
    pred = np.sort(np.asarray(pred_dims, dtype=np.float32))
    gt = np.sort(np.asarray(gt_dims, dtype=np.float32))
    return float(np.mean(np.abs(pred - gt) / np.maximum(gt, 1e-6)))


def center_rel_error(pred_center: Sequence[float], gt_center: Sequence[float]) -> float:
    pred = np.asarray(pred_center, dtype=np.float32)
    gt = np.asarray(gt_center, dtype=np.float32)
    return float(np.linalg.norm(pred - gt) / max(float(gt[2]), 1e-6))


def z_rel_error(pred_center: Sequence[float], gt_center: Sequence[float]) -> float:
    return float(abs(float(pred_center[2]) - float(gt_center[2])) / max(float(gt_center[2]), 1e-6))


def make_model_args(args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        moca_root=args.moca_root,
        moca_config=args.moca_config,
        moca_checkpoint=args.moca_checkpoint,
        cube_config=args.cube_config,
        cube_checkpoint=args.cube_checkpoint,
        joint_checkpoint=args.joint_checkpoint,
        prefer_ema=args.prefer_ema,
        strict_moca=False,
        strict_cube=False,
        precision=args.precision,
        batch_size=args.batch_size,
        min_depth=0.05,
        max_depth=80.0,
        min_dimension=0.01,
        max_dimension=20.0,
    )


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    data = load_json(args.json_file)
    name_to_id, id_to_name = category_maps(data)
    images = {int(img["id"]): img for img in data.get("images", [])}
    gsam_info = load_gsam_info(args.original_pseudo_root, args.dataset, args.split)

    candidates = []
    for ann in data.get("annotations", []):
        if not valid_gt_ann(ann):
            continue
        img = images.get(int(ann["image_id"]))
        if img is None:
            continue
        gt_box = normalize_box(ann["bbox2D_tight"], int(img["width"]), int(img["height"]))
        if gt_box is None:
            continue
        record = gsam_info.get(int(ann["image_id"]), {})
        gsam_box, match_iou, gsam_score = best_gsam_box(record, ann.get("category_name", id_to_name.get(int(ann["category_id"]), "")), gt_box, int(img["width"]), int(img["height"]))
        if gsam_box is None or match_iou < args.min_match_iou:
            continue
        candidates.append(
            {
                "ann": ann,
                "image": img,
                "gt_box": gt_box,
                "gsam_box": gsam_box,
                "gsam_match_iou": match_iou,
                "gsam_score": gsam_score,
            }
        )

    if not candidates:
        raise RuntimeError("No valid GT annotations matched to GroundedSAM boxes.")
    random.shuffle(candidates)
    samples = candidates[: int(args.num_samples)]
    print(f"Candidates={len(candidates)} sampled={len(samples)}")

    device = torch.device("cpu" if args.force_cpu or not torch.cuda.is_available() else f"cuda:{args.gpu}")
    print(f"device={device}")
    model_args = make_model_args(args)
    moca_model, cube_model = build_models(model_args, device)
    from losses.bbx3d_loss import BBox3DLoss

    criterion = BBox3DLoss().to(device).eval()

    by_image: Dict[int, List[dict]] = defaultdict(list)
    for sample in samples:
        by_image[int(sample["image"]["id"])].append(sample)

    results = []
    for image_id, image_samples in tqdm(by_image.items(), desc="MoCA GT-vs-GSAM"):
        img = image_samples[0]["image"]
        width, height = int(img["width"]), int(img["height"])
        image_path = Path(args.image_root) / img["file_path"]
        image = Image.open(image_path).convert("RGB")
        letterboxed, scale, pad_left, pad_top, new_w, new_h = letterbox_image(image, 512)
        image_tensor = image_to_tensor(letterboxed)
        padding_mask = valid_padding_mask(512, pad_left, pad_top, new_w, new_h)
        K = torch.as_tensor(np.asarray(img["K"], dtype=np.float32).reshape(3, 3), dtype=torch.float32)
        K_np = K.numpy()

        entries = []
        boxes = []
        for sample in image_samples:
            entries.append((sample, "gt_box"))
            boxes.append(sample["gt_box"])
            entries.append((sample, "gsam_box"))
            boxes.append(sample["gsam_box"])

        pred_boxes, _, _ = run_moca_on_image(
            moca_model=moca_model,
            cube_model=cube_model,
            criterion=criterion,
            image_tensor=image_tensor,
            padding_mask=padding_mask,
            K=K,
            boxes_xyxy=boxes,
            scale=scale,
            pad_left=pad_left,
            pad_top=pad_top,
            real_h=height,
            device=device,
            args=model_args,
        )

        for (sample, source), corners, in_box in zip(entries, pred_boxes, boxes):
            center, dims, _ = box_to_center_dims_R(corners)
            gt_ann = sample["ann"]
            gt_center = gt_ann["center_cam"]
            gt_dims = gt_ann["dimensions"]
            gt_proj = gt_ann.get("bbox2D_proj")
            gt_tight = sample["gt_box"]
            pred_proj = project_points(corners, K_np, width, height)
            results.append(
                {
                    "ann_id": int(gt_ann["id"]),
                    "image_id": int(gt_ann["image_id"]),
                    "category_name": gt_ann.get("category_name", id_to_name.get(int(gt_ann["category_id"]), "")),
                    "source": source,
                    "input_box": [float(x) for x in in_box],
                    "gsam_match_iou": float(sample["gsam_match_iou"]),
                    "pred_center": [float(x) for x in center],
                    "pred_dims": [float(x) for x in dims],
                    "gt_center": [float(x) for x in gt_center],
                    "gt_dims": [float(x) for x in gt_dims],
                    "dim_rel_error": sorted_dim_rel_error(dims, gt_dims),
                    "z_rel_error": z_rel_error(center, gt_center),
                    "center_rel_error": center_rel_error(center, gt_center),
                    "pred_proj_iou_to_input": float(bbox_iou_xyxy(pred_proj, in_box)),
                    "pred_proj_iou_to_gt_tight": float(bbox_iou_xyxy(pred_proj, gt_tight)),
                    "pred_proj_iou_to_gt_proj": float(bbox_iou_xyxy(pred_proj, gt_proj)) if valid_xyxy(gt_proj) else None,
                }
            )

    summary = {"num_samples": len(samples), "num_predictions": len(results)}
    for source in ("gt_box", "gsam_box"):
        part = [r for r in results if r["source"] == source]
        summary[source] = {
            "dim_rel_error": summarize([r["dim_rel_error"] for r in part]),
            "z_rel_error": summarize([r["z_rel_error"] for r in part]),
            "center_rel_error": summarize([r["center_rel_error"] for r in part]),
            "pred_proj_iou_to_input": summarize([r["pred_proj_iou_to_input"] for r in part]),
            "pred_proj_iou_to_gt_tight": summarize([r["pred_proj_iou_to_gt_tight"] for r in part]),
            "pred_proj_iou_to_gt_proj": summarize([r["pred_proj_iou_to_gt_proj"] for r in part if r["pred_proj_iou_to_gt_proj"] is not None]),
        }

    gt_dim = summary["gt_box"]["dim_rel_error"]["p50"]
    gs_dim = summary["gsam_box"]["dim_rel_error"]["p50"]
    gt_z = summary["gt_box"]["z_rel_error"]["p50"]
    gs_z = summary["gsam_box"]["z_rel_error"]["p50"]
    summary["interpretation"] = {
        "dim_error_delta_gsam_minus_gt": round(float(gs_dim - gt_dim), 6),
        "z_error_delta_gsam_minus_gt": round(float(gs_z - gt_z), 6),
        "box_distribution_likely_problem": bool(gs_dim > gt_dim * 1.10 or gs_z > gt_z * 1.10),
    }

    out = {"summary": summary, "samples": results}
    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(summary, indent=2))
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
