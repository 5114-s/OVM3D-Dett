#!/usr/bin/env python3
"""Fuse external 2D proposals into the original GroundingSAM pseudo cache.

The original OVM3D pseudo-label builders consume:

    pseudo_label/<dataset>/<split>/info.pth
    pseudo_label/<dataset>/<split>/mask/<image_id>.npy
    pseudo_label/<dataset>/<split>/depth/<image_id>.npy
    pseudo_label/<dataset>/<split>/info_ground.pth

This script appends external proposals, such as Detic+SAM2 boxes/masks, to that
cache format.  It does not generate 3D boxes; downstream SOR/PCA scripts keep
doing the geometry step.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Append external 2D proposals into pseudo_label cache."
    )
    parser.add_argument("--omni3d_json", required=True)
    parser.add_argument("--external_2d_json", required=True)
    parser.add_argument("--external_2d_root", default="")
    parser.add_argument("--base_pseudo_root", default="pseudo_label")
    parser.add_argument("--output_pseudo_root", required=True)
    parser.add_argument("--dataset", default="SUNRGBD")
    parser.add_argument("--split", choices=["train", "val"], default="train")
    parser.add_argument(
        "--external_bbox_format",
        choices=["xyxy", "xywh"],
        default="xyxy",
    )
    parser.add_argument("--score_threshold", type=float, default=0.25)
    parser.add_argument("--fusion_iou_threshold", type=float, default=0.65)
    parser.add_argument("--class_agnostic_iou_threshold", type=float, default=0.90)
    parser.add_argument("--min_2d_area_ratio", type=float, default=0.0002)
    parser.add_argument("--max_2d_area_ratio", type=float, default=0.85)
    parser.add_argument("--max_external_per_image", type=int, default=80)
    parser.add_argument(
        "--require_mask",
        action="store_true",
        help="Drop external proposals without mask_path instead of using bbox masks.",
    )
    parser.add_argument(
        "--no_aux_symlinks",
        action="store_true",
        help="Do not symlink/copy depth and info_ground from the base pseudo cache.",
    )
    parser.add_argument(
        "--allow_inplace",
        action="store_true",
        help="Allow writing into the same split directory as the base pseudo cache.",
    )
    return parser.parse_args()


def torch_load(path: str):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


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


def bbox_to_xyxy(
    bbox: Sequence[float], width: int, height: int, bbox_format: str
) -> Optional[List[float]]:
    if bbox is None or len(bbox) != 4:
        return None
    vals = [float(v) for v in bbox]
    if bbox_format == "xywh":
        x, y, w, h = vals
        vals = [x, y, x + w, y + h]
    return valid_bbox_xyxy(vals, width, height)


def bbox_iou(a: Sequence[float], b: Sequence[float]) -> float:
    ax1, ay1, ax2, ay2 = [float(x) for x in a]
    bx1, by1, bx2, by2 = [float(x) for x in b]
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(ix2 - ix1, 0.0), max(iy2 - iy1, 0.0)
    inter = iw * ih
    area_a = max(ax2 - ax1, 0.0) * max(ay2 - ay1, 0.0)
    area_b = max(bx2 - bx1, 0.0) * max(by2 - by1, 0.0)
    denom = area_a + area_b - inter
    return float(inter / denom) if denom > 1e-12 else 0.0


def bbox_area_ratio(bbox: Sequence[float], width: int, height: int) -> float:
    x1, y1, x2, y2 = [float(x) for x in bbox]
    return float(max(x2 - x1, 0.0) * max(y2 - y1, 0.0) / max(width * height, 1))


def normalize_records(data) -> List[dict]:
    if isinstance(data, list):
        return [x for x in data if isinstance(x, dict)]
    if not isinstance(data, dict):
        return []
    for key in ("annotations", "detections", "results", "proposals", "objects"):
        value = data.get(key)
        if isinstance(value, list):
            return [x for x in value if isinstance(x, dict)]
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


def get_record_bbox(record: dict):
    return (
        record.get("bbox")
        or record.get("bbox_xyxy")
        or record.get("bbox2D_tight")
        or record.get("bbox2D_proj")
        or record.get("bbox2d")
        or record.get("box")
    )


def load_reference(omni3d_json: str):
    with open(omni3d_json, "r") as f:
        data = json.load(f)
    images_by_id = {int(img["id"]): img for img in data.get("images", [])}
    id_to_name: Dict[int, str] = {}
    name_to_id: Dict[str, int] = {}
    for cat in data.get("categories", []):
        cid = int(cat["id"])
        name = str(cat["name"])
        id_to_name[cid] = name
        name_to_id[name.lower().strip()] = cid
    return data, images_by_id, id_to_name, name_to_id


def get_label_and_category(
    record: dict, id_to_name: Dict[int, str], name_to_id: Dict[str, int]
) -> Tuple[Optional[str], int]:
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
            label = str(id_to_name.get(mapped_id, label)).lower().strip()
        except Exception:
            mapped_id = -1
    return label, int(mapped_id)


def resolve_mask_path(mask_value, external_root: str, json_dir: str) -> Optional[str]:
    if not isinstance(mask_value, str) or not mask_value:
        return None
    candidates = [mask_value]
    if not os.path.isabs(mask_value):
        if external_root:
            candidates.append(os.path.join(external_root, mask_value))
        candidates.append(os.path.join(json_dir, mask_value))
        candidates.append(os.path.join(os.getcwd(), mask_value))
    for path in candidates:
        if os.path.exists(path):
            return path
    return None


def load_mask(mask_path: Optional[str], width: int, height: int, bbox: Sequence[float]):
    if mask_path is None:
        return None
    try:
        if mask_path.lower().endswith(".npy"):
            mask = np.load(mask_path, allow_pickle=False)
        else:
            mask = np.asarray(Image.open(mask_path))
    except Exception:
        return None
    mask = np.asarray(mask)
    if mask.ndim > 2:
        mask = np.squeeze(mask)
    if mask.ndim != 2:
        return None
    if mask.shape != (height, width):
        return None
    return mask.astype(bool)


def bbox_mask(width: int, height: int, bbox: Sequence[float]) -> np.ndarray:
    x1, y1, x2, y2 = [int(round(float(x))) for x in bbox]
    x1, y1 = max(x1, 0), max(y1, 0)
    x2, y2 = min(x2, width - 1), min(y2, height - 1)
    mask = np.zeros((height, width), dtype=bool)
    if x2 > x1 and y2 > y1:
        mask[y1 : y2 + 1, x1 : x2 + 1] = True
    return mask


def squeeze_base_mask(mask: np.ndarray, height: int, width: int) -> np.ndarray:
    mask = np.asarray(mask).astype(bool)
    if mask.ndim == 4:
        return mask
    if mask.ndim == 3:
        if mask.shape[-2:] == (height, width):
            return mask[:, None, :, :]
        if mask.shape[1:] == (height, width):
            return mask[:, None, :, :]
    if mask.ndim == 2:
        return mask[None, None, :, :]
    return np.zeros((0, 1, height, width), dtype=bool)


def ensure_aux_links(base_split_dir: Path, out_split_dir: Path):
    for name in ("depth", "info_ground.pth"):
        src = base_split_dir / name
        dst = out_split_dir / name
        if dst.exists() or dst.is_symlink() or not src.exists():
            continue
        try:
            os.symlink(src.resolve(), dst)
        except OSError:
            if src.is_dir():
                shutil.copytree(src, dst)
            else:
                shutil.copy2(src, dst)


def build_external_groups(args, images_by_id, id_to_name, name_to_id):
    with open(args.external_2d_json, "r") as f:
        data = json.load(f)
    records = normalize_records(data)
    json_dir = os.path.dirname(os.path.abspath(args.external_2d_json))
    grouped = defaultdict(list)
    stats = defaultdict(int)
    stats["raw_count"] = len(records)

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
        bbox = bbox_to_xyxy(
            get_record_bbox(record),
            int(img["width"]),
            int(img["height"]),
            args.external_bbox_format,
        )
        if bbox is None:
            stats["filtered_bbox"] += 1
            continue
        area_ratio = bbox_area_ratio(bbox, int(img["width"]), int(img["height"]))
        if area_ratio < args.min_2d_area_ratio or area_ratio > args.max_2d_area_ratio:
            stats["filtered_area"] += 1
            continue
        mask_value = (
            record.get("mask_path")
            or record.get("mask")
            or record.get("segmentation_path")
            or record.get("mask_file")
        )
        mask_path = resolve_mask_path(mask_value, args.external_2d_root, json_dir)
        if mask_path is None:
            stats["missing_mask"] += 1
            if args.require_mask:
                continue
        grouped[image_id].append(
            {
                "label": label,
                "category_id": category_id,
                "score": score,
                "bbox": [float(x) for x in bbox],
                "mask_path": mask_path,
                "source_index": int(record.get("source_index", idx)),
                "source": str(record.get("source", "external_2d")),
            }
        )
        stats["kept_count"] += 1
    for image_id in grouped:
        grouped[image_id].sort(key=lambda x: float(x["score"]), reverse=True)
        if args.max_external_per_image is not None:
            grouped[image_id] = grouped[image_id][: int(args.max_external_per_image)]
    return grouped, stats


def main() -> None:
    args = parse_args()
    _source, images_by_id, id_to_name, name_to_id = load_reference(args.omni3d_json)

    base_split_dir = Path(args.base_pseudo_root) / args.dataset / args.split
    out_split_dir = Path(args.output_pseudo_root) / args.dataset / args.split
    if (
        base_split_dir.resolve() == out_split_dir.resolve()
        and not bool(args.allow_inplace)
    ):
        raise ValueError(
            "Refusing to overwrite the base pseudo cache in-place. "
            "Use a different --output_pseudo_root or pass --allow_inplace."
        )
    out_mask_dir = out_split_dir / "mask"
    out_split_dir.mkdir(parents=True, exist_ok=True)
    out_mask_dir.mkdir(parents=True, exist_ok=True)

    base_info_path = base_split_dir / "info.pth"
    if not base_info_path.exists():
        raise FileNotFoundError(f"Missing base info cache: {base_info_path}")
    base_info = torch_load(str(base_info_path))
    output_info = copy.deepcopy(base_info)
    external_grouped, external_stats = build_external_groups(
        args, images_by_id, id_to_name, name_to_id
    )

    stats = defaultdict(int)
    stats.update({f"external_{k}": int(v) for k, v in external_stats.items()})

    for image_id, img in tqdm(images_by_id.items(), desc="Fuse external 2D into cache"):
        width, height = int(img["width"]), int(img["height"])
        base_entry = output_info.get(image_id, {})
        base_boxes = np.asarray(base_entry.get("boxes", []), dtype=np.float32).reshape(-1, 4)
        base_scores = np.asarray(base_entry.get("conf", []), dtype=np.float32).reshape(-1)
        base_phrases = list(base_entry.get("phrases", []))
        if base_scores.shape[0] != base_boxes.shape[0]:
            base_scores = np.ones((base_boxes.shape[0],), dtype=np.float32)

        base_mask_path = base_split_dir / "mask" / f"{image_id}.npy"
        if base_mask_path.exists():
            base_masks = squeeze_base_mask(np.load(base_mask_path), height, width)
        else:
            base_masks = np.zeros((base_boxes.shape[0], 1, height, width), dtype=bool)

        boxes = [box.astype(np.float32) for box in base_boxes]
        scores = [float(x) for x in base_scores.tolist()]
        phrases = [str(x).lower().strip() for x in base_phrases]
        base_sources = list(base_entry.get("proposal_sources", []))
        base_external_flags = list(base_entry.get("proposal_external_flags", []))
        base_source_indices = list(base_entry.get("proposal_source_indices", []))
        sources = []
        external_flags = []
        source_indices = []
        for i in range(len(boxes)):
            sources.append(str(base_sources[i]) if i < len(base_sources) else "groundingsam")
            external_flags.append(
                bool(base_external_flags[i]) if i < len(base_external_flags) else False
            )
            source_indices.append(
                int(base_source_indices[i]) if i < len(base_source_indices) else int(i)
            )
        masks = [base_masks[i].astype(bool) for i in range(min(len(boxes), base_masks.shape[0]))]
        while len(masks) < len(boxes):
            masks.append(bbox_mask(width, height, boxes[len(masks)])[None, :, :])

        stats["base_objects"] += len(boxes)
        for proposal in external_grouped.get(image_id, []):
            prop_box = proposal["bbox"]
            prop_label = proposal["label"]
            duplicate = False
            for old_box, old_label in zip(boxes, phrases):
                overlap = bbox_iou(prop_box, old_box)
                if old_label == prop_label and overlap >= float(args.fusion_iou_threshold):
                    duplicate = True
                    stats["skipped_same_class_duplicate"] += 1
                    break
                if overlap >= float(args.class_agnostic_iou_threshold):
                    duplicate = True
                    stats["skipped_class_agnostic_duplicate"] += 1
                    break
            if duplicate:
                continue

            mask = load_mask(proposal.get("mask_path"), width, height, prop_box)
            if mask is None:
                if args.require_mask:
                    stats["skipped_missing_mask"] += 1
                    continue
                mask = bbox_mask(width, height, prop_box)
                stats["used_bbox_mask"] += 1
            boxes.append(np.asarray(prop_box, dtype=np.float32))
            scores.append(float(proposal["score"]))
            phrases.append(prop_label)
            sources.append(str(proposal.get("source", "external_2d")))
            external_flags.append(True)
            source_indices.append(int(proposal.get("source_index", -1)))
            masks.append(mask[None, :, :].astype(bool))
            stats["external_added"] += 1

        if boxes:
            output_info[image_id] = {
                "boxes": np.stack(boxes, axis=0).astype(np.float32),
                "conf": np.asarray(scores, dtype=np.float32),
                "phrases": phrases,
                "proposal_sources": sources,
                "proposal_external_flags": np.asarray(external_flags, dtype=np.bool_),
                "proposal_source_indices": np.asarray(source_indices, dtype=np.int64),
            }
            out_masks = np.stack(masks, axis=0).astype(bool)
        else:
            output_info[image_id] = {
                "boxes": np.zeros((0, 4), dtype=np.float32),
                "conf": np.zeros((0,), dtype=np.float32),
                "phrases": [],
                "proposal_sources": [],
                "proposal_external_flags": np.zeros((0,), dtype=np.bool_),
                "proposal_source_indices": np.zeros((0,), dtype=np.int64),
            }
            out_masks = np.zeros((0, 1, height, width), dtype=bool)
        np.save(out_mask_dir / f"{image_id}.npy", out_masks)
        stats["output_objects"] += len(output_info[image_id]["phrases"])

    torch.save(output_info, out_split_dir / "info.pth")
    if not args.no_aux_symlinks:
        ensure_aux_links(base_split_dir, out_split_dir)

    stats["images"] = len(images_by_id)
    stats["base_info"] = str(base_info_path)
    stats["output_info"] = str(out_split_dir / "info.pth")
    stats["output_mask_dir"] = str(out_mask_dir)
    stats_path = out_split_dir / "fusion_stats.json"
    with open(stats_path, "w") as f:
        json.dump(dict(stats), f, indent=2)
    print(f"Wrote fused pseudo cache: {out_split_dir}")
    print(json.dumps(dict(stats), indent=2))


if __name__ == "__main__":
    main()
