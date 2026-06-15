#!/usr/bin/env python3
"""Add SAM2 masks to DetAny3D 2D detections using box prompts.

This prepares the strongest Boxer++ input branch:
    DetAny3D 2D boxes -> SAM2 masks -> Boxer++ mask-aware depth gate.

The output keeps DetAny3D-style records and adds mask_path for each detection.
Use it with:
    tools/run_boxer_omni3d.py --box_source detany3d --use_mask_depth_gate
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parent.parent
GROUNDED_SAM2_ROOT = REPO_ROOT / "third_party" / "Grounded-SAM-2"
if str(GROUNDED_SAM2_ROOT) not in sys.path:
    sys.path.insert(0, str(GROUNDED_SAM2_ROOT))
if str(GROUNDED_SAM2_ROOT / "sam2") not in sys.path:
    sys.path.insert(0, str(GROUNDED_SAM2_ROOT / "sam2"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Use SAM2 box prompts to add mask_path to DetAny3D detections."
    )
    parser.add_argument("--detany3d_json", required=True, help="DetAny3D output JSON.")
    parser.add_argument(
        "--omni3d_json",
        required=True,
        help="Reference Omni3D JSON with images/categories.",
    )
    parser.add_argument("--image_root", default="datasets")
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--mask_dir", required=True)
    parser.add_argument(
        "--sam_checkpoint",
        default=None,
        help="Path to sam2.1_hiera_large.pt. If omitted, common local paths are tried.",
    )
    parser.add_argument(
        "--sam2_config",
        default="configs/sam2.1/sam2.1_hiera_l.yaml",
        help="SAM2 config path relative to third_party/Grounded-SAM-2/sam2.",
    )
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--force_cpu", action="store_true")
    parser.add_argument(
        "--input_bbox_format",
        choices=["xywh", "xyxy"],
        default="xywh",
        help="DetAny3D official output uses xywh.",
    )
    parser.add_argument(
        "--output_bbox_format",
        choices=["same", "xywh", "xyxy"],
        default="same",
        help="BBox format written to output JSON.",
    )
    parser.add_argument("--score_threshold", type=float, default=0.0)
    parser.add_argument("--max_images", type=int, default=None)
    parser.add_argument("--max_boxes_per_image", type=int, default=None)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse existing mask .npy files and skip their SAM2 prediction.",
    )
    parser.add_argument(
        "--mask_path_relative_to",
        choices=["output_json", "cwd", "absolute"],
        default="output_json",
        help="How mask_path should be stored in the output JSON.",
    )
    parser.add_argument(
        "--multimask_output",
        action="store_true",
        help="Ask SAM2 for multiple masks and keep the highest predicted IoU mask.",
    )
    return parser.parse_args()


def resolve_path(root: str, path: str) -> str:
    if os.path.isabs(path):
        return path
    return os.path.join(root, path)


def load_reference_maps(omni3d_json: str):
    with open(omni3d_json, "r") as f:
        data = json.load(f)
    images_by_id = {int(img["id"]): img for img in data.get("images", [])}
    id_to_name: Dict[int, str] = {}
    name_to_id: Dict[str, int] = {}
    for cat in data.get("categories", []):
        cid = int(cat["id"])
        name = str(cat["name"])
        id_to_name[cid] = name
        name_to_id[name.lower()] = cid
    return images_by_id, id_to_name, name_to_id


def normalize_records(data) -> List[dict]:
    if isinstance(data, list):
        return [item for item in data if isinstance(item, dict)]
    if not isinstance(data, dict):
        return []
    for key in ("annotations", "detections", "results", "proposals", "objects"):
        value = data.get(key)
        if isinstance(value, list):
            return [item for item in value if isinstance(item, dict)]
    records: List[dict] = []
    for image_id, value in data.items():
        if not isinstance(value, list):
            continue
        for idx, item in enumerate(value):
            if not isinstance(item, dict):
                continue
            record = dict(item)
            record.setdefault("image_id", image_id)
            record.setdefault("source_index", idx)
            records.append(record)
    return records


def valid_bbox_xyxy(bbox: Sequence[float], width: int, height: int) -> Optional[List[float]]:
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


def bbox_to_xyxy(
    bbox: Sequence[float], width: int, height: int, bbox_format: str
) -> Optional[List[float]]:
    if bbox is None or len(bbox) != 4:
        return None
    vals = [float(v) for v in bbox]
    if bbox_format == "xywh":
        x, y, w_box, h_box = vals
        vals = [x, y, x + w_box, y + h_box]
    return valid_bbox_xyxy(vals, width, height)


def bbox_from_xyxy(xyxy: Sequence[float], bbox_format: str) -> List[float]:
    x1, y1, x2, y2 = [float(x) for x in xyxy]
    if bbox_format == "xywh":
        return [x1, y1, x2 - x1, y2 - y1]
    return [x1, y1, x2, y2]


def get_record_bbox(record: dict):
    return (
        record.get("bbox")
        or record.get("bbox_xyxy")
        or record.get("bbox2D_tight")
        or record.get("bbox2D_proj")
        or record.get("bbox2d")
        or record.get("box")
    )


def get_label_and_category(record: dict, id_to_name: Dict[int, str], name_to_id: Dict[str, int]):
    category_id = record.get("category_id", record.get("class_id", None))
    label = (
        record.get("category_name")
        or record.get("label")
        or record.get("name")
        or record.get("phrase")
        or record.get("class_name")
        or record.get("text")
    )
    if label is None and category_id is not None:
        try:
            label = id_to_name.get(int(category_id))
        except Exception:
            label = None
    if label is None:
        return None, -1
    label = str(label).lower().strip()
    mapped_id = name_to_id.get(label)
    if mapped_id is None:
        try:
            mapped_id = int(category_id)
        except Exception:
            mapped_id = -1
    return label, int(mapped_id)


def resolve_sam_checkpoint(user_path: Optional[str]) -> str:
    candidates = []
    if user_path:
        candidates.append(user_path)
    candidates.extend(
        [
            str(GROUNDED_SAM2_ROOT / "checkpoints" / "sam2.1_hiera_large.pt"),
            str(REPO_ROOT / "third_party" / "Grounded-SAM-2" / "checkpoints" / "sam2.1_hiera_large.pt"),
            "/data/ZhaoX/ovmono3d/checkpoints/sam2.1_hiera_large.pt",
            "/extra/OVM3D-Det-1/OVM3D-Det-1/Grounded-SAM-2/checkpoints/sam2.1_hiera_large.pt",
        ]
    )
    for path in candidates:
        if path and os.path.exists(path):
            return path
    raise FileNotFoundError(
        "SAM2 checkpoint not found. Pass --sam_checkpoint /path/to/sam2.1_hiera_large.pt"
    )


def load_sam2_predictor(args: argparse.Namespace, device: str):
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    checkpoint = resolve_sam_checkpoint(args.sam_checkpoint)
    sam2_dir = str(GROUNDED_SAM2_ROOT / "sam2")
    original_cwd = os.getcwd()
    try:
        os.chdir(sam2_dir)
        model = build_sam2(
            config_file=args.sam2_config,
            ckpt_path=checkpoint,
            device=device,
        )
    finally:
        os.chdir(original_cwd)
    return SAM2ImagePredictor(model), checkpoint


def mask_path_for_record(mask_dir: str, image_id: int, index: int, label: str) -> str:
    label_safe = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in label)
    return os.path.join(mask_dir, f"{int(image_id)}_{int(index):05d}_{label_safe}.npy")


def stored_mask_path(mask_path: str, args: argparse.Namespace) -> str:
    if args.mask_path_relative_to == "absolute":
        return os.path.abspath(mask_path)
    if args.mask_path_relative_to == "cwd":
        return os.path.relpath(mask_path, os.getcwd())
    return os.path.relpath(mask_path, os.path.dirname(os.path.abspath(args.output_json)))


def select_sam_mask(masks: np.ndarray, scores: np.ndarray) -> Tuple[np.ndarray, float]:
    masks_np = np.asarray(masks)
    scores_np = np.asarray(scores).reshape(-1)
    if masks_np.ndim == 2:
        return masks_np.astype(bool), float(scores_np[0]) if scores_np.size else 0.0
    if masks_np.ndim == 3:
        best = int(np.argmax(scores_np)) if scores_np.size == masks_np.shape[0] else 0
        return masks_np[best].astype(bool), float(scores_np[best]) if scores_np.size else 0.0
    if masks_np.ndim == 4:
        masks_np = masks_np.squeeze(1)
        best = int(np.argmax(scores_np)) if scores_np.size == masks_np.shape[0] else 0
        return masks_np[best].astype(bool), float(scores_np[best]) if scores_np.size else 0.0
    raise ValueError(f"Unsupported SAM2 mask shape: {masks_np.shape}")


def main() -> None:
    args = parse_args()
    os.makedirs(args.mask_dir, exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.output_json)), exist_ok=True)

    images_by_id, id_to_name, name_to_id = load_reference_maps(args.omni3d_json)
    with open(args.detany3d_json, "r") as f:
        detany3d_data = json.load(f)
    records = normalize_records(detany3d_data)

    grouped = defaultdict(list)
    output_records: List[dict] = []
    stats = {
        "raw_count": len(records),
        "kept_count": 0,
        "filtered_image": 0,
        "filtered_category": 0,
        "filtered_score": 0,
        "filtered_bbox": 0,
        "masks_written": 0,
        "masks_reused": 0,
        "images_processed": 0,
        "input_bbox_format": args.input_bbox_format,
        "output_bbox_format": args.output_bbox_format,
    }

    output_bbox_format = args.input_bbox_format if args.output_bbox_format == "same" else args.output_bbox_format

    for idx, record in enumerate(records):
        try:
            image_id = int(record.get("image_id", record.get("id")))
        except Exception:
            stats["filtered_image"] += 1
            continue
        img = images_by_id.get(image_id)
        if img is None:
            stats["filtered_image"] += 1
            continue

        label, category_id = get_label_and_category(record, id_to_name, name_to_id)
        if label is None or category_id < 0:
            stats["filtered_category"] += 1
            continue

        score = float(record.get("score", record.get("confidence", record.get("conf", 1.0))))
        if score < float(args.score_threshold):
            stats["filtered_score"] += 1
            continue

        bbox_xyxy = bbox_to_xyxy(
            get_record_bbox(record),
            int(img["width"]),
            int(img["height"]),
            args.input_bbox_format,
        )
        if bbox_xyxy is None:
            stats["filtered_bbox"] += 1
            continue

        out_record = dict(record)
        out_record["image_id"] = image_id
        out_record["category_name"] = label
        out_record["category_id"] = category_id
        out_record["bbox"] = bbox_from_xyxy(bbox_xyxy, output_bbox_format)
        out_record["score"] = score
        out_record["source"] = out_record.get("source", "detany3d_sam2")
        out_record["source_index"] = int(out_record.get("source_index", idx))
        group_item = {
            "record_index": len(output_records),
            "source_index": idx,
            "bbox_xyxy": bbox_xyxy,
            "label": label,
        }
        grouped[image_id].append(group_item)
        output_records.append(out_record)
        stats["kept_count"] += 1

    selected_image_ids = list(grouped.keys())
    if args.max_images is not None:
        selected_image_ids = selected_image_ids[: args.max_images]
    selected_image_id_set = set(selected_image_ids)

    if args.force_cpu:
        device = "cpu"
    elif torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
        device = "cuda"
    else:
        device = "cpu"
    predictor, checkpoint = load_sam2_predictor(args, device)

    for image_id in tqdm(selected_image_ids, desc="SAM2 masks from DetAny3D boxes"):
        items = grouped[image_id]
        if args.max_boxes_per_image is not None:
            items = items[: args.max_boxes_per_image]
        if not items:
            continue
        img_info = images_by_id[image_id]
        image_path = resolve_path(args.image_root, img_info["file_path"])
        image = np.asarray(Image.open(image_path).convert("RGB"))
        predictor.set_image(image)
        stats["images_processed"] += 1

        for item in items:
            record = output_records[item["record_index"]]
            mask_path = mask_path_for_record(
                args.mask_dir,
                image_id,
                item["source_index"],
                item["label"],
            )
            if args.resume and os.path.exists(mask_path):
                record["mask_path"] = stored_mask_path(mask_path, args)
                stats["masks_reused"] += 1
                continue

            box = np.asarray(item["bbox_xyxy"], dtype=np.float32)
            with torch.no_grad():
                if device == "cuda":
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        masks, scores, _ = predictor.predict(
                            point_coords=None,
                            point_labels=None,
                            box=box,
                            multimask_output=bool(args.multimask_output),
                            normalize_coords=False,
                        )
                else:
                    masks, scores, _ = predictor.predict(
                        point_coords=None,
                        point_labels=None,
                        box=box,
                        multimask_output=bool(args.multimask_output),
                        normalize_coords=False,
                    )
            mask, sam_score = select_sam_mask(masks, scores)
            np.save(mask_path, mask.astype(bool))
            record["mask_path"] = stored_mask_path(mask_path, args)
            record["sam2_mask_score"] = float(sam_score)
            stats["masks_written"] += 1

    final_records = output_records
    if args.max_images is not None:
        final_records = [
            record for record in output_records if int(record.get("image_id", -1)) in selected_image_id_set
        ]
    stats["output_count"] = len(final_records)

    output = {
        "info": {
            "source": "detany3d_sam2",
            "detany3d_json": os.path.abspath(args.detany3d_json),
            "omni3d_json": os.path.abspath(args.omni3d_json),
            "mask_dir": os.path.abspath(args.mask_dir),
            "sam_checkpoint": os.path.abspath(checkpoint),
            "bbox_format": output_bbox_format,
            "stats": stats,
        },
        "annotations": final_records,
    }
    with open(args.output_json, "w") as f:
        json.dump(output, f, indent=2)

    print(f"Wrote DetAny3D+SAM2 JSON: {args.output_json}")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
