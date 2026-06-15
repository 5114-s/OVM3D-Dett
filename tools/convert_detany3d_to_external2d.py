#!/usr/bin/env python3
"""Convert DetAny3D inference output to Boxer++ external_2d proposals.

DetAny3D's official inference output is a list of records with fields such as:
    image_id, bbox [x, y, w, h], category_id, score

This script converts those records into the common external_2d schema consumed
by tools/run_boxer_omni3d.py:
    annotations: [{image_id, category_name, category_id, bbox [x1,y1,x2,y2],
                   score, optional mask_path}]
"""

from __future__ import annotations

import argparse
import json
import math
import os
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert DetAny3D output JSON to Boxer++ external_2d JSON."
    )
    parser.add_argument("--detany3d_json", required=True, help="DetAny3D output JSON.")
    parser.add_argument(
        "--omni3d_json",
        required=True,
        help="Reference Omni3D JSON used for image/category mapping.",
    )
    parser.add_argument("--output_json", required=True, help="Converted output JSON.")
    parser.add_argument(
        "--input_bbox_format",
        choices=["xywh", "xyxy"],
        default="xywh",
        help="DetAny3D official output uses xywh.",
    )
    parser.add_argument(
        "--score_threshold",
        type=float,
        default=0.0,
        help="Drop detections below this score.",
    )
    parser.add_argument(
        "--mask_root",
        default="",
        help="Optional root used to resolve/canonicalize relative mask_path values.",
    )
    parser.add_argument(
        "--keep_3d_fields",
        action="store_true",
        help="Keep DetAny3D 3D fields under a detany3d_extra object for debugging.",
    )
    return parser.parse_args()


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
    bbox: Sequence[float],
    width: int,
    height: int,
    bbox_format: str,
) -> Optional[List[float]]:
    if bbox is None or len(bbox) != 4:
        return None
    vals = [float(v) for v in bbox]
    if bbox_format == "xywh":
        x, y, w_box, h_box = vals
        vals = [x, y, x + w_box, y + h_box]
    return valid_bbox_xyxy(vals, width, height)


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


def load_reference_maps(omni3d_json: str) -> Tuple[Dict[int, dict], Dict[int, str], Dict[str, int]]:
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


def resolve_mask_path(mask_path: Optional[str], mask_root: str) -> Optional[str]:
    if not isinstance(mask_path, str) or not mask_path:
        return None
    if os.path.isabs(mask_path):
        return mask_path
    if mask_root:
        candidate = os.path.join(mask_root, mask_path)
        if os.path.exists(candidate):
            return candidate
    return mask_path


def main() -> None:
    args = parse_args()
    images_by_id, id_to_name, name_to_id = load_reference_maps(args.omni3d_json)
    with open(args.detany3d_json, "r") as f:
        detany3d_data = json.load(f)

    records = normalize_records(detany3d_data)
    annotations = []
    stats = {
        "raw_count": len(records),
        "kept_count": 0,
        "filtered_image": 0,
        "filtered_category": 0,
        "filtered_score": 0,
        "filtered_bbox": 0,
        "with_mask": 0,
        "input_bbox_format": args.input_bbox_format,
    }

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

        bbox = (
            record.get("bbox")
            or record.get("bbox_xyxy")
            or record.get("bbox2D_tight")
            or record.get("bbox2D_proj")
            or record.get("bbox2d")
            or record.get("box")
        )
        bbox_xyxy = bbox_to_xyxy(
            bbox,
            int(img["width"]),
            int(img["height"]),
            args.input_bbox_format,
        )
        if bbox_xyxy is None:
            stats["filtered_bbox"] += 1
            continue

        mask_path = resolve_mask_path(
            record.get("mask_path")
            or record.get("mask_file")
            or record.get("segmentation_path"),
            args.mask_root,
        )
        if mask_path is not None:
            stats["with_mask"] += 1

        output = {
            "image_id": image_id,
            "category_name": label,
            "category_id": category_id,
            "bbox": [float(x) for x in bbox_xyxy],
            "score": score,
            "source": "detany3d",
            "source_index": int(record.get("source_index", idx)),
        }
        if mask_path is not None:
            output["mask_path"] = mask_path
        if args.keep_3d_fields:
            output["detany3d_extra"] = {
                key: record[key]
                for key in (
                    "depth",
                    "bbox3D",
                    "center_cam",
                    "center_2D",
                    "pose",
                    "dimensions",
                    "area",
                    "yaw",
                )
                if key in record
            }
        annotations.append(output)
        stats["kept_count"] += 1

    os.makedirs(os.path.dirname(os.path.abspath(args.output_json)), exist_ok=True)
    converted = {
        "info": {
            "source": "detany3d",
            "detany3d_json": os.path.abspath(args.detany3d_json),
            "omni3d_json": os.path.abspath(args.omni3d_json),
            "bbox_format": "xyxy",
            "stats": stats,
        },
        "annotations": annotations,
    }
    with open(args.output_json, "w") as f:
        json.dump(converted, f, indent=2)

    print(f"Wrote external_2d JSON: {args.output_json}")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
