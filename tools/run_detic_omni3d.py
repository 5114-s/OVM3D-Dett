#!/usr/bin/env python3
"""Run Detic on Omni3D images and export common external_2d proposals.

This is intentionally a thin wrapper around the Detic demo API used by
ImOV3D.  The output schema is the same as the external_2d JSON consumed by
other tools in this repository:

    {"annotations": [{"image_id", "category_name", "category_id",
                      "bbox" [x1,y1,x2,y2], "score", "source"}]}

Detic is only used as an extra 2D proposal source.  The 3D pseudo-label
generation remains in the OVM3D-Det/SOR pipeline.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DETIC_ROOT = (
    REPO_ROOT
    / "third_party"
    / "Detic"
)
DEFAULT_CONFIG = "configs/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.yaml"
DEFAULT_WEIGHTS = "models/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate Detic 2D proposals for Omni3D/SUNRGBD images."
    )
    parser.add_argument("--json_file", required=True, help="Input Omni3D JSON.")
    parser.add_argument("--image_root", default="datasets")
    parser.add_argument("--output_json", required=True)
    parser.add_argument(
        "--detic_root",
        default=str(DEFAULT_DETIC_ROOT),
        help=(
            "Detic project root. If using ImOV3D's layout, clone/download the "
            "official Detic project under third_party/ImOV3D/Data_Maker/"
            "2DBranch_BBOX_GEN/Detic."
        ),
    )
    parser.add_argument("--config_file", default=DEFAULT_CONFIG)
    parser.add_argument("--weights", default=DEFAULT_WEIGHTS)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--score_threshold", type=float, default=0.35)
    parser.add_argument("--max_images", type=int, default=None)
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--skip_images", type=int, default=1)
    parser.add_argument("--max_detections_per_image", type=int, default=100)
    parser.add_argument("--min_2d_area_ratio", type=float, default=0.0002)
    parser.add_argument("--max_2d_area_ratio", type=float, default=0.85)
    parser.add_argument(
        "--custom_vocabulary",
        default=None,
        help=(
            "Comma-separated Detic custom vocabulary. Defaults to categories "
            "from --json_file, which keeps the output aligned to SUNRGBD 38 classes."
        ),
    )
    parser.add_argument(
        "--pred_all_class",
        action="store_true",
        help="Forwarded to Detic; keep all class scores per proposal.",
    )
    parser.add_argument(
        "--opts",
        nargs=argparse.REMAINDER,
        default=[],
        help="Extra Detic cfg opts after MODEL.WEIGHTS.",
    )
    return parser.parse_args()


def resolve_path(root: str, path: str) -> str:
    if os.path.isabs(path):
        return path
    return os.path.join(root, path)


def valid_bbox_xyxy(
    bbox: Sequence[float], width: int, height: int
) -> Optional[List[float]]:
    if bbox is None or len(bbox) != 4:
        return None
    vals = [float(v) for v in bbox]
    if any(not math.isfinite(v) for v in vals):
        return None
    x1, y1, x2, y2 = vals
    if x2 <= x1 or y2 <= y1:
        return None
    x1 = max(0.0, min(float(width - 1), x1))
    y1 = max(0.0, min(float(height - 1), y1))
    x2 = max(0.0, min(float(width - 1), x2))
    y2 = max(0.0, min(float(height - 1), y2))
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def bbox_area_ratio(bbox: Sequence[float], width: int, height: int) -> float:
    x1, y1, x2, y2 = [float(x) for x in bbox]
    return float(max(x2 - x1, 0.0) * max(y2 - y1, 0.0) / max(width * height, 1))


def load_omni3d(path: str):
    with open(path, "r") as f:
        data = json.load(f)
    id_to_name: Dict[int, str] = {}
    name_to_id: Dict[str, int] = {}
    categories_sorted = sorted(data.get("categories", []), key=lambda c: int(c["id"]))
    for cat in categories_sorted:
        cid = int(cat["id"])
        name = str(cat["name"]).lower().strip()
        id_to_name[cid] = str(cat["name"])
        name_to_id[name] = cid
    return data, categories_sorted, id_to_name, name_to_id


def add_detic_paths(detic_root: str) -> Path:
    root = Path(detic_root).resolve()
    candidates = [
        root,
        root / "third_party" / "CenterNet2",
        root.parent / "CenterNet2",
        REPO_ROOT / "third_party" / "detectron2",
    ]
    for candidate in candidates:
        if candidate.exists() and str(candidate) not in sys.path:
            sys.path.insert(0, str(candidate))
    return root


def load_detic_demo(args: argparse.Namespace, vocabulary: str):
    detic_root = add_detic_paths(args.detic_root)
    config_file = args.config_file
    if not os.path.isabs(config_file):
        config_file = str(detic_root / config_file)
    weights = args.weights
    if not os.path.isabs(weights):
        weights = str(detic_root / weights)

    try:
        from detectron2.config import get_cfg
        from detectron2.data.detection_utils import read_image
        from detectron2.utils.logger import setup_logger
        from centernet.config import add_centernet_config
        from detic.config import add_detic_config
        from detic.predictor import VisualizationDemo
    except Exception as exc:
        raise RuntimeError(
            "Could not import Detic. Install/clone the official Detic project "
            f"under {detic_root} and make sure CenterNet2/detectron2 are importable. "
            f"Original error: {exc}"
        ) from exc

    cfg = get_cfg()
    if args.cpu:
        cfg.MODEL.DEVICE = "cpu"
    else:
        torch.cuda.set_device(args.gpu)
        cfg.MODEL.DEVICE = "cuda"
    add_centernet_config(cfg)
    add_detic_config(cfg)
    cfg.merge_from_file(config_file)
    cfg.merge_from_list(["MODEL.WEIGHTS", weights] + list(args.opts))
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = float(args.score_threshold)
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = float(args.score_threshold)
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = float(
        args.score_threshold
    )
    cfg.MODEL.ROI_BOX_HEAD.ZEROSHOT_WEIGHT_PATH = "rand"
    if not args.pred_all_class:
        cfg.MODEL.ROI_HEADS.ONE_CLASS_PER_PROPOSAL = True
    cfg.freeze()

    setup_logger(name="fvcore")
    detic_args = SimpleNamespace(
        vocabulary="custom",
        custom_vocabulary=vocabulary,
        pred_all_class=bool(args.pred_all_class),
        confidence_threshold=float(args.score_threshold),
    )
    return VisualizationDemo(cfg, detic_args), read_image, config_file, weights


def main() -> None:
    args = parse_args()
    data, categories_sorted, _id_to_name, name_to_id = load_omni3d(args.json_file)
    if args.custom_vocabulary:
        vocabulary = args.custom_vocabulary
    else:
        vocabulary = ",".join(str(cat["name"]) for cat in categories_sorted)

    demo, read_image, config_file, weights = load_detic_demo(args, vocabulary)

    images = list(data.get("images", []))
    selected = images[args.start_index :: max(int(args.skip_images), 1)]
    if args.max_images is not None:
        selected = selected[: int(args.max_images)]

    annotations: List[dict] = []
    stats = {
        "images_total": len(images),
        "images_selected": len(selected),
        "raw_detections": 0,
        "kept_detections": 0,
        "filtered_category": 0,
        "filtered_bbox": 0,
        "filtered_area": 0,
        "score_threshold": float(args.score_threshold),
        "custom_vocabulary": vocabulary,
        "detic_root": os.path.abspath(args.detic_root),
        "config_file": os.path.abspath(config_file),
        "weights": os.path.abspath(weights),
    }

    for img_info in tqdm(selected, desc="Detic Omni3D"):
        image_id = int(img_info["id"])
        width, height = int(img_info["width"]), int(img_info["height"])
        image_path = resolve_path(args.image_root, img_info["file_path"])
        img = read_image(image_path, format="BGR")
        with torch.no_grad():
            predictions, _visualized = demo.run_on_image(img)
        instances = predictions.get("instances")
        if instances is None or len(instances) == 0:
            continue
        instances = instances.to("cpu")
        boxes = instances.pred_boxes.tensor.numpy()
        scores = instances.scores.numpy()
        classes = instances.pred_classes.numpy()
        class_names = list(getattr(demo.metadata, "thing_classes", []))
        order = np.argsort(-scores)
        if args.max_detections_per_image is not None:
            order = order[: int(args.max_detections_per_image)]

        for local_rank, det_idx in enumerate(order.tolist()):
            stats["raw_detections"] += 1
            class_index = int(classes[det_idx])
            if class_index < 0 or class_index >= len(class_names):
                stats["filtered_category"] += 1
                continue
            label = str(class_names[class_index]).lower().strip()
            category_id = name_to_id.get(label)
            if category_id is None:
                stats["filtered_category"] += 1
                continue
            bbox = valid_bbox_xyxy(boxes[det_idx].tolist(), width, height)
            if bbox is None:
                stats["filtered_bbox"] += 1
                continue
            area_ratio = bbox_area_ratio(bbox, width, height)
            if (
                area_ratio < float(args.min_2d_area_ratio)
                or area_ratio > float(args.max_2d_area_ratio)
            ):
                stats["filtered_area"] += 1
                continue
            annotations.append(
                {
                    "image_id": image_id,
                    "category_name": label,
                    "category_id": int(category_id),
                    "bbox": [float(x) for x in bbox],
                    "score": float(scores[det_idx]),
                    "source": "detic",
                    "source_index": int(local_rank),
                }
            )
            stats["kept_detections"] += 1

    output = {
        "info": {
            "source": "detic",
            "omni3d_json": os.path.abspath(args.json_file),
            "bbox_format": "xyxy",
            "stats": stats,
        },
        "annotations": annotations,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.output_json)), exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Wrote Detic external_2d JSON: {args.output_json}")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
