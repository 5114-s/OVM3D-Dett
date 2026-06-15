#!/usr/bin/env python3
"""Run BoxerNet on Omni3D-style JSON and write trainable Omni3D pseudo labels.

Pipeline:
    RGB + 2D boxes + K + metric depth/gravity -> BoxerNet 3D OBB
    -> camera-frame Omni3D JSON.

This script keeps Boxer as an independent pseudo-label branch. It writes both
the official Boxer CSV and an Omni3D JSON that matches the fields consumed by
OVM3D training.
"""

from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import math
import os
import sys
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BOXER_ROOT = os.path.join(REPO_ROOT, "third_party", "boxer")
if BOXER_ROOT not in sys.path:
    sys.path.insert(0, BOXER_ROOT)
if REPO_ROOT not in sys.path:
    sys.path.append(REPO_ROOT)

from boxernet.boxernet import BoxerNet  # noqa: E402
from loaders.base_loader import BaseLoader  # noqa: E402
from loaders.omni_loader import load_sunrgbd_extrinsics  # noqa: E402
from utils.file_io import ObbCsvWriter2  # noqa: E402
from utils.tw.obb import ObbTW  # noqa: E402
from utils.tw.pose import PoseTW  # noqa: E402
from utils.tw.tensor_utils import pad_string, string2tensor  # noqa: E402

from cubercnn.generate_label import llm_generated_prior  # noqa: E402
from cubercnn.generate_label.util import (  # noqa: E402
    erode_mask,
    extract_ground,
    point_to_plane_distance,
    project_image_to_cam,
)


DEFAULT_CKPT = os.path.join(
    BOXER_ROOT, "ckpts", "boxernet_hw960in4x6d768-3e37cfc4.ckpt"
)
_GSAM_MODULE = None


@dataclass
class Box2DEntry:
    ann: dict
    bbox_xyxy: List[float]
    label: str
    category_id: int
    score: float
    source_index: int = -1
    mask: Optional[np.ndarray] = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate Omni3D pseudo labels with BoxerNet."
    )
    parser.add_argument("--json_file", required=True, help="Input Omni3D JSON.")
    parser.add_argument(
        "--image_root",
        default="datasets",
        help="Dataset root joined with image file_path and depth file_path.",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Directory for boxer_3dbbs.csv, JSON, and stats.",
    )
    parser.add_argument(
        "--output_json",
        default=None,
        help="Output Omni3D JSON. Defaults to output_dir/<input>_boxer.json.",
    )
    parser.add_argument("--ckpt", default=DEFAULT_CKPT, help="BoxerNet checkpoint.")
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--max_images", type=int, default=None)
    parser.add_argument("--skip_images", type=int, default=1)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--force_cpu", action="store_true")
    parser.add_argument(
        "--force_precision",
        choices=["float32", "bfloat16"],
        default=None,
        help="Override autocast precision.",
    )
    parser.add_argument(
        "--bbox_fields",
        default="bbox2D_tight,bbox2D_proj,bbox2D_trunc",
        help="Comma-separated 2D bbox priority list.",
    )
    parser.add_argument("--thresh3d", type=float, default=0.15)
    parser.add_argument(
        "--box_source",
        choices=["json", "gsam", "original_gsam", "external_2d", "detany3d"],
        default="json",
        help=(
            "Use 2D boxes from input JSON, generate them with Grounding-SAM2, "
            "read original OVM3D Step-2 pseudo_label/<dataset>/<split>/info.pth, "
            "read an external 2D aggregator JSON, or read DetAny3D inference output."
        ),
    )
    parser.add_argument(
        "--external_2d_json",
        default=None,
        help=(
            "External 2D proposal JSON for --box_source external_2d. Supports "
            "a list, {'annotations': [...]}, {'detections': [...]}, or "
            "{image_id: [...]} records."
        ),
    )
    parser.add_argument(
        "--external_2d_root",
        default="",
        help="Root path used to resolve relative external mask_path values.",
    )
    parser.add_argument(
        "--external_2d_bbox_format",
        choices=["xyxy", "xywh"],
        default="xyxy",
        help="Format of external bbox values.",
    )
    parser.add_argument(
        "--external_2d_score_threshold",
        type=float,
        default=0.0,
        help="Drop external 2D proposals below this score before Boxer.",
    )
    parser.add_argument(
        "--detany3d_json",
        default=None,
        help=(
            "DetAny3D output JSON for --box_source detany3d. If omitted, "
            "--external_2d_json is used as a fallback."
        ),
    )
    parser.add_argument(
        "--detany3d_root",
        default="",
        help="Root path used to resolve relative DetAny3D mask_path values.",
    )
    parser.add_argument(
        "--detany3d_bbox_format",
        choices=["xywh", "xyxy"],
        default="xywh",
        help="DetAny3D official output uses xywh bbox; use xyxy for custom dumps.",
    )
    parser.add_argument(
        "--detany3d_score_threshold",
        type=float,
        default=0.0,
        help="Drop DetAny3D proposals below this score before Boxer.",
    )
    parser.add_argument(
        "--depth_source",
        choices=["sunrgbd", "unidepth", "original_unidepth"],
        default="sunrgbd",
        help=(
            "Use SUNRGBD metric depth files, independent UniDepth, or original "
            "OVM3D Step-1 pseudo_label/<dataset>/<split>/depth/*.npy."
        ),
    )
    parser.add_argument(
        "--dataset",
        default=None,
        help="Dataset name for original_gsam/original_unidepth paths. Auto-inferred by default.",
    )
    parser.add_argument(
        "--split",
        choices=["train", "val"],
        default=None,
        help="Dataset split for original_gsam/original_unidepth paths. Auto-inferred by default.",
    )
    parser.add_argument(
        "--original_pseudo_root",
        default="pseudo_label",
        help="Root folder containing original OVM3D Step-1/2 outputs.",
    )
    parser.add_argument(
        "--text_prompt",
        default=None,
        help="Grounding-SAM2 text prompt. Defaults to SUNRGBD 38 classes.",
    )
    parser.add_argument("--grounding_dino_checkpoint", default=None)
    parser.add_argument("--sam_checkpoint", default=None)
    parser.add_argument(
        "--use_large_gdino",
        action="store_true",
        help="Use Grounding DINO Swin-B, matching the original stronger setup.",
    )
    parser.add_argument("--box_threshold", type=float, default=0.2)
    parser.add_argument("--text_threshold", type=float, default=0.2)
    parser.add_argument("--gsam_iou_threshold", type=float, default=0.9)
    parser.add_argument(
        "--gsam_box_from",
        choices=["mask", "box"],
        default="mask",
        help="Use SAM2 mask bounding boxes or raw GroundingDINO boxes.",
    )
    parser.add_argument("--min_2d_area_ratio", type=float, default=0.0002)
    parser.add_argument("--max_2d_area_ratio", type=float, default=0.85)
    parser.add_argument(
        "--save_unidepth",
        action="store_true",
        help="Save per-image UniDepth maps as .npy under output_dir/unidepth.",
    )
    parser.add_argument("--min_proj_iou", type=float, default=0.10)
    parser.add_argument("--min_depth_pixels", type=int, default=16)
    parser.add_argument("--min_depth_support", type=float, default=0.05)
    parser.add_argument("--max_rel_depth_error", type=float, default=0.70)
    parser.add_argument("--depth_margin", type=float, default=0.25)
    parser.add_argument("--min_dimension", type=float, default=0.03)
    parser.add_argument("--max_dimension", type=float, default=6.0)
    parser.add_argument("--prior_min_ratio", type=float, default=0.35)
    parser.add_argument("--prior_max_ratio", type=float, default=3.0)
    parser.add_argument(
        "--prior_adjust_min_ratio",
        type=float,
        default=0.50,
        help=(
            "If Boxer predicts a dimension below this fraction of the original "
            "OVM3D category prior, expand that dimension to the prior before "
            "writing Omni3D JSON."
        ),
    )
    parser.add_argument(
        "--no_prior_adjust",
        action="store_true",
        help="Disable original-prior dimension expansion before JSON export.",
    )
    parser.add_argument(
        "--no_prior_gate",
        action="store_true",
        help="Disable original OVM3D category-prior dimension gate.",
    )
    parser.add_argument("--ground_max_distance", type=float, default=0.55)
    parser.add_argument(
        "--no_ground_snap",
        action="store_true",
        help="Disable snapping ground-supported classes to the original ground plane.",
    )
    parser.add_argument(
        "--no_ground_gate",
        action="store_true",
        help="Disable original OVM3D floor/ground consistency gate.",
    )
    parser.add_argument(
        "--require_ground",
        action="store_true",
        help="Reject ground-supported classes when original floor/ground is missing.",
    )
    parser.add_argument(
        "--require_depth",
        action="store_true",
        help="Drop boxes from images whose metric depth is missing.",
    )
    parser.add_argument(
        "--no_depth_gate",
        action="store_true",
        help="Use depth as Boxer input but do not filter by depth support.",
    )
    parser.add_argument(
        "--no_projection_gate",
        action="store_true",
        help="Do not filter by projection IoU.",
    )
    parser.add_argument(
        "--use_mask_depth_gate",
        action="store_true",
        help=(
            "Use instance-mask pixels, when available, for depth support/error "
            "instead of the whole 2D box patch."
        ),
    )
    parser.add_argument(
        "--require_mask_depth",
        action="store_true",
        help="Reject depth gate when --use_mask_depth_gate is enabled but no usable mask depth exists.",
    )
    parser.add_argument(
        "--boxer_refine_with_depth",
        action="store_true",
        help=(
            "Conservatively shift Boxer center_z so the box front face agrees "
            "with the mask/bbox median metric depth."
        ),
    )
    parser.add_argument("--depth_refine_max_shift_ratio", type=float, default=0.25)
    parser.add_argument(
        "--classwise_quality_gate",
        action="store_true",
        help="Apply a Boxer++ quality threshold after projection/depth/prior/ground gates.",
    )
    parser.add_argument("--quality_threshold", type=float, default=0.40)
    parser.add_argument("--thin_quality_threshold", type=float, default=0.55)
    parser.add_argument("--large_quality_threshold", type=float, default=0.42)
    parser.add_argument(
        "--boxer_nms",
        action="store_true",
        help="Apply same-class 2D/3D duplicate suppression after Boxer++ quality scoring.",
    )
    parser.add_argument("--boxer_nms_iou", type=float, default=0.75)
    parser.add_argument("--boxer_nms_center_ratio", type=float, default=0.35)
    parser.add_argument(
        "--save_invalid",
        action="store_true",
        help="Write rejected predictions as valid3D=false annotations.",
    )
    parser.add_argument("--stats_json", default=None)
    return parser.parse_args()


def resolve_path(root: str, rel_path: str) -> str:
    if os.path.isabs(rel_path):
        return rel_path
    return os.path.join(root, rel_path)


def valid_bbox_xyxy(bbox: Sequence[float], width: int, height: int) -> Optional[List[float]]:
    if bbox is None or len(bbox) != 4:
        return None
    vals = [float(v) for v in bbox]
    if any(not math.isfinite(v) for v in vals):
        return None
    if vals == [-1.0, -1.0, -1.0, -1.0]:
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
    x1, y1, x2, y2 = bbox
    area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    image_area = max(1.0, float(width * height))
    return float(area / image_area)


def choose_bbox(
    ann: dict, fields: Sequence[str], width: int, height: int
) -> Optional[List[float]]:
    for field in fields:
        bbox = valid_bbox_xyxy(ann.get(field), width, height)
        if bbox is not None:
            return bbox
    return None


def group_annotations_by_image(
    annotations: Iterable[dict],
    images_by_id: Dict[int, dict],
    bbox_fields: Sequence[str],
    min_area_ratio: float = 0.0,
    max_area_ratio: float = 1.0,
) -> Dict[int, List[Box2DEntry]]:
    grouped: Dict[int, List[Box2DEntry]] = {}
    for ann in annotations:
        img = images_by_id.get(int(ann["image_id"]))
        if img is None:
            continue
        bbox = choose_bbox(ann, bbox_fields, int(img["width"]), int(img["height"]))
        if bbox is None:
            continue
        area_ratio = bbox_area_ratio(bbox, int(img["width"]), int(img["height"]))
        if area_ratio < min_area_ratio or area_ratio > max_area_ratio:
            continue
        label = ann.get("category_name") or str(ann.get("category_id", "object"))
        category_id = int(ann.get("category_id", -1))
        score = float(ann.get("score", ann.get("confidence", 1.0)))
        grouped.setdefault(int(ann["image_id"]), []).append(
            Box2DEntry(
                ann=ann,
                bbox_xyxy=bbox,
                label=str(label),
                category_id=category_id,
                score=score,
            )
        )
    return grouped


def load_external_mask(mask_value, external_root: str) -> Optional[np.ndarray]:
    if mask_value is None:
        return None
    if isinstance(mask_value, np.ndarray):
        return squeeze_instance_mask(mask_value)
    if isinstance(mask_value, (list, tuple)):
        try:
            return squeeze_instance_mask(np.asarray(mask_value))
        except Exception:
            return None
    if not isinstance(mask_value, str) or not mask_value:
        return None

    mask_path = mask_value
    if not os.path.isabs(mask_path):
        roots = [external_root, REPO_ROOT]
        for root in roots:
            if not root:
                continue
            candidate = os.path.join(root, mask_path)
            if os.path.exists(candidate):
                mask_path = candidate
                break
    if not os.path.exists(mask_path):
        return None
    try:
        if mask_path.lower().endswith(".npy"):
            return squeeze_instance_mask(np.load(mask_path, allow_pickle=False))
        return squeeze_instance_mask(np.asarray(Image.open(mask_path)))
    except Exception:
        return None


def external_bbox_to_xyxy(
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


def normalize_external_2d_records(external_data) -> List[dict]:
    if isinstance(external_data, list):
        return [r for r in external_data if isinstance(r, dict)]

    if not isinstance(external_data, dict):
        return []

    for key in ("annotations", "detections", "results", "proposals", "objects"):
        value = external_data.get(key)
        if isinstance(value, list):
            return [r for r in value if isinstance(r, dict)]

    records: List[dict] = []
    for image_id, value in external_data.items():
        if not isinstance(value, list):
            continue
        for idx, record in enumerate(value):
            if not isinstance(record, dict):
                continue
            item = dict(record)
            item.setdefault("image_id", image_id)
            item.setdefault("source_index", idx)
            records.append(item)
    return records


def get_external_record_label(record: dict, sem_id_to_name: Dict[int, str]) -> Tuple[Optional[str], int]:
    label = (
        record.get("category_name")
        or record.get("label")
        or record.get("name")
        or record.get("phrase")
        or record.get("class_name")
        or record.get("text")
    )
    category_id = record.get("category_id", record.get("class_id", None))
    if label is None and category_id is not None:
        try:
            label = sem_id_to_name.get(int(category_id))
        except Exception:
            label = None
    if label is None:
        return None, -1
    return str(label).lower().strip(), int(category_id) if category_id is not None else -1


def build_external_2d_grouped(
    external_json: str,
    images_by_id: Dict[int, dict],
    cat_name_to_id: Dict[str, int],
    sem_id_to_name: Dict[int, str],
    args: argparse.Namespace,
) -> Tuple[Dict[int, List[Box2DEntry]], dict]:
    if not external_json:
        raise ValueError("--external_2d_json is required when --box_source external_2d")
    with open(external_json, "r") as f:
        external_data = json.load(f)

    records = normalize_external_2d_records(external_data)
    grouped: Dict[int, List[Box2DEntry]] = {}
    stats = {
        "path": os.path.abspath(external_json),
        "raw_count": len(records),
        "kept_count": 0,
        "filtered_score": 0,
        "filtered_image": 0,
        "filtered_category": 0,
        "filtered_bbox": 0,
        "filtered_area": 0,
        "with_mask": 0,
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

        label, category_id = get_external_record_label(record, sem_id_to_name)
        if label is None:
            stats["filtered_category"] += 1
            continue
        mapped_category_id = cat_name_to_id.get(label, category_id)
        if mapped_category_id is None or int(mapped_category_id) < 0:
            stats["filtered_category"] += 1
            continue

        score = float(record.get("score", record.get("confidence", record.get("conf", 1.0))))
        if score < float(args.external_2d_score_threshold):
            stats["filtered_score"] += 1
            continue

        bbox = (
            record.get("bbox")
            or record.get("bbox_xyxy")
            or record.get("box")
            or record.get("bbox2D_tight")
            or record.get("bbox2D_proj")
            or record.get("bbox2d")
        )
        bbox_xyxy = external_bbox_to_xyxy(
            bbox,
            int(img["width"]),
            int(img["height"]),
            args.external_2d_bbox_format,
        )
        if bbox_xyxy is None:
            stats["filtered_bbox"] += 1
            continue
        area_ratio = bbox_area_ratio(bbox_xyxy, int(img["width"]), int(img["height"]))
        if area_ratio < args.min_2d_area_ratio or area_ratio > args.max_2d_area_ratio:
            stats["filtered_area"] += 1
            continue

        mask_value = (
            record.get("mask_path")
            or record.get("mask")
            or record.get("segmentation_path")
            or record.get("mask_file")
        )
        mask = load_external_mask(mask_value, args.external_2d_root)
        if mask is not None:
            stats["with_mask"] += 1

        ann_stub = {
            "image_id": image_id,
            "category_name": label,
            "category_id": int(mapped_category_id),
            "bbox2D_tight": [float(x) for x in bbox_xyxy],
            "bbox2D_proj": [float(x) for x in bbox_xyxy],
            "score": float(score),
            "visibility": float(record.get("visibility", 1.0)),
            "truncation": float(record.get("truncation", 0.0)),
        }
        grouped.setdefault(image_id, []).append(
            Box2DEntry(
                ann=ann_stub,
                bbox_xyxy=[float(x) for x in bbox_xyxy],
                label=label,
                category_id=int(mapped_category_id),
                score=float(score),
                source_index=int(record.get("source_index", idx)),
                mask=mask,
            )
        )
        stats["kept_count"] += 1

    return grouped, stats


def infer_split_from_json(json_file: str, source: dict) -> str:
    name = os.path.basename(json_file).lower()
    if "_val" in name or name.endswith("val.json"):
        return "val"
    if "_train" in name or name.endswith("train.json"):
        return "train"
    split = str(source.get("info", {}).get("split", "")).lower()
    if "val" in split:
        return "val"
    return "train"


def category_maps(source: dict) -> Tuple[Dict[str, int], Dict[int, str]]:
    name_to_id = {}
    id_to_name = {}
    for cat in source.get("categories", []):
        name = str(cat["name"])
        cid = int(cat["id"])
        name_to_id[name.lower()] = cid
        id_to_name[cid] = name
    return name_to_id, id_to_name


def get_category_prior(dataset_name: str) -> Dict[str, Sequence[float]]:
    prior = getattr(llm_generated_prior, dataset_name, None)
    if prior is None:
        prior = getattr(llm_generated_prior, dataset_name.upper(), None)
    if prior is None and dataset_name.lower() == "sunrgbd":
        prior = llm_generated_prior.SUNRGBD
    return {str(k).lower(): v for k, v in (prior or {}).items()}


GROUND_SUPPORTED_CLASSES = {
    "bathtub",
    "bed",
    "bin",
    "bookcase",
    "box",
    "cabinet",
    "chair",
    "counter",
    "desk",
    "floor mat",
    "machine",
    "night stand",
    "oven",
    "refrigerator",
    "shelves",
    "sofa",
    "stove",
    "table",
    "toilet",
}


THIN_OR_WALL_CLASSES = {
    "blinds",
    "curtain",
    "door",
    "floor mat",
    "mirror",
    "picture",
    "television",
    "towel",
    "window",
}


LARGE_FURNITURE_CLASSES = {
    "bathtub",
    "bed",
    "bookcase",
    "cabinet",
    "chair",
    "counter",
    "desk",
    "refrigerator",
    "shelves",
    "sofa",
    "stove",
    "table",
    "toilet",
}


def create_uv_depth(depth: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
    if mask is not None:
        depth = depth * mask
    x, y = np.meshgrid(
        np.linspace(0, depth.shape[1] - 1, depth.shape[1]),
        np.linspace(0, depth.shape[0] - 1, depth.shape[0]),
    )
    uv_depth = np.stack((x, y, depth), axis=-1).reshape(-1, 3)
    return uv_depth[uv_depth[:, 2] != 0]


def load_original_ground_info(
    pseudo_root: str,
    dataset_name: str,
    split: str,
) -> Tuple[Optional[dict], Optional[str]]:
    info_path = os.path.join(pseudo_root, dataset_name, split, "info_ground.pth")
    if not os.path.exists(info_path):
        return None, None
    return torch.load(info_path, map_location="cpu"), info_path


def get_original_record(info: Optional[dict], image_id: int):
    if info is None:
        return None
    record = info.get(int(image_id))
    if record is None:
        record = info.get(str(int(image_id)))
    if not record:
        return None
    return record


def orient_ground_equation_like_original(ground_equ: np.ndarray) -> Optional[np.ndarray]:
    ground_equ = np.asarray(ground_equ, dtype=np.float64).reshape(-1)
    if ground_equ.size != 4 or not np.all(np.isfinite(ground_equ[:4])):
        return None
    if np.dot([0.0, -1.0, 0.0], ground_equ[:3]) <= 0:
        ground_equ = -ground_equ
    return ground_equ


def estimate_original_ground_equation(
    info_ground: Optional[dict],
    pseudo_root: str,
    dataset_name: str,
    split: str,
    image_id: int,
    depth_np: Optional[np.ndarray],
    K: Sequence[Sequence[float]],
) -> Optional[np.ndarray]:
    if depth_np is None:
        return None
    record = get_original_record(info_ground, image_id)
    if record is None:
        return None

    conf = np.asarray(record.get("conf", []), dtype=np.float64).reshape(-1)
    if conf.size == 0:
        return None
    mask_path = os.path.join(
        pseudo_root, dataset_name, split, "ground_mask", f"{int(image_id)}.npy"
    )
    if not os.path.exists(mask_path):
        return None

    ground_mask = np.load(mask_path)
    ground_mask = erode_mask(ground_mask.astype(float), 4, 4)
    best_idx = int(np.argmax(conf))
    if best_idx >= len(ground_mask):
        return None
    mask = np.asarray(ground_mask[best_idx]).squeeze()
    if mask.shape != depth_np.shape[:2]:
        return None

    uv_depth = create_uv_depth(np.asarray(depth_np, dtype=np.float32), mask)
    if uv_depth.shape[0] <= 10:
        return None
    pseudo_lidar_ground = project_image_to_cam(uv_depth, np.asarray(K, dtype=np.float64))
    if pseudo_lidar_ground.shape[0] <= 10:
        return None
    return orient_ground_equation_like_original(extract_ground(pseudo_lidar_ground))


def category_prior_gate(
    label: str,
    dims: np.ndarray,
    category_prior: Dict[str, Sequence[float]],
    min_ratio: float,
    max_ratio: float,
) -> Tuple[bool, dict]:
    prior_xyz = get_prior_array(label, category_prior)
    if prior_xyz is None:
        return True, {"prior_available": False}
    dims = np.asarray(dims, dtype=np.float64).reshape(3)
    if np.any(prior_xyz <= 0) or np.any(dims <= 0):
        return False, {"prior_available": True, "prior_ok": False}

    candidates = [
        prior_xyz,
        np.array([prior_xyz[2], prior_xyz[1], prior_xyz[0]], dtype=np.float64),
    ]
    best = None
    for candidate in candidates:
        ratio = dims / np.maximum(candidate, 1e-6)
        violation = np.maximum(min_ratio / np.maximum(ratio, 1e-6), ratio / max_ratio)
        score = float(np.max(violation))
        if best is None or score < best[0]:
            best = (score, ratio, candidate)

    score, ratio, matched_prior = best
    ok = bool(np.all(ratio >= min_ratio) and np.all(ratio <= max_ratio))
    return ok, {
        "prior_available": True,
        "prior_ok": ok,
        "prior_dims": [float(x) for x in prior_xyz.tolist()],
        "matched_prior_dims": [float(x) for x in matched_prior.tolist()],
        "prior_ratio": [float(x) for x in ratio.tolist()],
        "prior_violation": float(score),
    }


def get_prior_array(
    label: str, category_prior: Dict[str, Sequence[float]]
) -> Optional[np.ndarray]:
    prior = category_prior.get(str(label).lower())
    if prior is None:
        return None
    prior_arr = np.asarray(prior, dtype=np.float64).reshape(-1)
    if prior_arr.size != 3 or not np.all(np.isfinite(prior_arr)):
        return None
    return prior_arr


def ground_consistency_gate(
    label: str,
    corners_cam: np.ndarray,
    ground_equ: Optional[np.ndarray],
    max_distance: float,
    require_ground: bool,
) -> Tuple[bool, dict]:
    label_norm = str(label).lower()
    if label_norm not in GROUND_SUPPORTED_CLASSES:
        return True, {"ground_required": False}
    if ground_equ is None:
        return (not require_ground), {
            "ground_required": True,
            "ground_available": False,
            "ground_ok": not require_ground,
        }

    distances = [
        point_to_plane_distance(ground_equ, float(x), float(y), float(z))
        for x, y, z in np.asarray(corners_cam, dtype=np.float64)
    ]
    min_distance = float(np.min(distances)) if distances else float("inf")
    ok = bool(np.isfinite(min_distance) and min_distance <= max_distance)
    return ok, {
        "ground_required": True,
        "ground_available": True,
        "ground_ok": ok,
        "ground_min_corner_distance": min_distance,
    }


def boxer_dims_to_omni(dims_object_xyz: np.ndarray) -> np.ndarray:
    """Convert Boxer object-axis [x, y, z] to CubeRCNN/Omni3D [w, h, l]."""
    dims_object_xyz = np.asarray(dims_object_xyz, dtype=np.float64).reshape(3)
    return np.array(
        [dims_object_xyz[2], dims_object_xyz[1], dims_object_xyz[0]],
        dtype=np.float64,
    )


def omni_dims_to_object_xyz(dims_omni_whl: np.ndarray) -> np.ndarray:
    """Convert CubeRCNN/Omni3D [w, h, l] to object-axis [x, y, z]."""
    dims_omni_whl = np.asarray(dims_omni_whl, dtype=np.float64).reshape(3)
    return np.array(
        [dims_omni_whl[2], dims_omni_whl[1], dims_omni_whl[0]],
        dtype=np.float64,
    )


def cubercnn_box_corners(
    center_cam: np.ndarray,
    dims_omni_whl: np.ndarray,
    r_cam_obj: np.ndarray,
) -> np.ndarray:
    """Build 8 camera-frame cuboid corners using CubeRCNN's [w, h, l] convention."""
    center_cam = np.asarray(center_cam, dtype=np.float64).reshape(3)
    dims_xyz = omni_dims_to_object_xyz(dims_omni_whl)
    r_cam_obj = np.asarray(r_cam_obj, dtype=np.float64).reshape(3, 3)
    dx, dy, dz = [float(x) for x in dims_xyz]
    local_corners = np.array(
        [
            [-dx / 2, -dy / 2, -dz / 2],
            [dx / 2, -dy / 2, -dz / 2],
            [dx / 2, dy / 2, -dz / 2],
            [-dx / 2, dy / 2, -dz / 2],
            [-dx / 2, -dy / 2, dz / 2],
            [dx / 2, -dy / 2, dz / 2],
            [dx / 2, dy / 2, dz / 2],
            [-dx / 2, dy / 2, dz / 2],
        ],
        dtype=np.float64,
    )
    return local_corners @ r_cam_obj.T + center_cam.reshape(1, 3)


def signed_plane_distances(ground_equ: np.ndarray, points: np.ndarray) -> np.ndarray:
    ground_equ = np.asarray(ground_equ, dtype=np.float64).reshape(4)
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    normal = ground_equ[:3]
    norm = float(np.linalg.norm(normal))
    if norm <= 1e-8:
        return np.full(points.shape[0], np.inf, dtype=np.float64)
    return (points @ normal + ground_equ[3]) / norm


def apply_original_prior_ground_adjustments(
    label: str,
    center_cam: np.ndarray,
    dims_omni: np.ndarray,
    r_cam_obj: np.ndarray,
    category_prior: Dict[str, Sequence[float]],
    ground_equ: Optional[np.ndarray],
    args: argparse.Namespace,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    center = np.asarray(center_cam, dtype=np.float64).reshape(3).copy()
    dims = np.asarray(dims_omni, dtype=np.float64).reshape(3).copy()
    r_cam_obj = np.asarray(r_cam_obj, dtype=np.float64).reshape(3, 3)
    raw_center = center.copy()
    raw_dims = dims.copy()

    prior_adjusted_axes: List[str] = []
    prior = get_prior_array(label, category_prior)
    if prior is not None and not args.no_prior_adjust:
        axis_names = ["w", "h", "l"]
        for axis, axis_name in enumerate(axis_names):
            threshold = float(prior[axis]) * float(args.prior_adjust_min_ratio)
            if (
                np.isfinite(dims[axis])
                and dims[axis] > 0
                and threshold > 0
                and dims[axis] < threshold
            ):
                dims[axis] = float(prior[axis])
                prior_adjusted_axes.append(axis_name)

    corners = cubercnn_box_corners(center, dims, r_cam_obj)
    ground_snapped = False
    ground_snap_distance = 0.0
    label_norm = str(label).lower()
    if (
        not args.no_ground_snap
        and ground_equ is not None
        and label_norm in GROUND_SUPPORTED_CLASSES
    ):
        ground_equ_np = np.asarray(ground_equ, dtype=np.float64).reshape(4)
        normal = ground_equ_np[:3]
        normal_norm = float(np.linalg.norm(normal))
        if normal_norm > 1e-8:
            normal_unit = normal / normal_norm
            signed = signed_plane_distances(ground_equ_np, corners)
            bottom_signed = float(np.min(signed)) if signed.size else float("inf")
            if np.isfinite(bottom_signed) and abs(bottom_signed) <= args.ground_max_distance:
                shift = -bottom_signed * normal_unit
                center = center + shift
                corners = corners + shift.reshape(1, 3)
                ground_snapped = True
                ground_snap_distance = bottom_signed

    metrics = {
        "prior_adjusted": bool(prior_adjusted_axes),
        "prior_adjusted_axes": prior_adjusted_axes,
        "ground_snapped": bool(ground_snapped),
        "ground_snap_distance": float(ground_snap_distance),
        "raw_center_cam": [float(x) for x in raw_center.tolist()],
        "raw_dimensions": [float(x) for x in raw_dims.tolist()],
    }
    return (
        center.astype(np.float32),
        dims.astype(np.float32),
        corners.astype(np.float32),
        metrics,
    )


def load_original_gsam_info(
    pseudo_root: str,
    dataset_name: str,
    split: str,
) -> Tuple[dict, str]:
    info_path = os.path.join(pseudo_root, dataset_name, split, "info.pth")
    if not os.path.exists(info_path):
        raise FileNotFoundError(
            f"Original Grounded-SAM info not found: {info_path}. "
            "Run original Step 2 first or use --box_source json with bbox2D_proj."
        )
    return torch.load(info_path, map_location="cpu"), info_path


def squeeze_instance_mask(mask: Optional[np.ndarray]) -> Optional[np.ndarray]:
    if mask is None:
        return None
    mask_np = np.asarray(mask)
    if mask_np.size == 0:
        return None
    mask_np = np.squeeze(mask_np)
    if mask_np.ndim != 2:
        return None
    return mask_np.astype(bool)


def resize_bool_mask(mask: np.ndarray, height: int, width: int) -> np.ndarray:
    mask_np = squeeze_instance_mask(mask)
    if mask_np is None:
        return np.zeros((height, width), dtype=bool)
    if mask_np.shape == (height, width):
        return mask_np.astype(bool)
    pil = Image.fromarray(mask_np.astype(np.uint8) * 255)
    try:
        resample = Image.Resampling.NEAREST
    except AttributeError:
        resample = Image.NEAREST
    return np.asarray(pil.resize((width, height), resample), dtype=np.uint8) > 0


def load_original_instance_masks(
    pseudo_root: str,
    dataset_name: str,
    split: str,
    image_id: int,
) -> Optional[np.ndarray]:
    mask_path = os.path.join(
        pseudo_root, dataset_name, split, "mask", f"{int(image_id)}.npy"
    )
    if not os.path.exists(mask_path):
        return None
    try:
        return np.load(mask_path, allow_pickle=False)
    except Exception:
        return None


def build_original_gsam_entries(
    img_info: dict,
    original_info: dict,
    cat_name_to_id: Dict[str, int],
    dataset_name: str,
    split: str,
    args: argparse.Namespace,
) -> Tuple[List[Box2DEntry], dict]:
    img_id = int(img_info["id"])
    record = original_info.get(img_id)
    if record is None:
        record = original_info.get(str(img_id))
    if not record:
        return [], {"raw_count": 0}

    boxes = record.get("boxes", [])
    phrases = record.get("phrases", [])
    confs = record.get("conf", [])
    raw_count = min(len(boxes), len(phrases))
    entries: List[Box2DEntry] = []
    width = int(img_info["width"])
    height = int(img_info["height"])
    masks = None
    if args.use_mask_depth_gate or args.boxer_refine_with_depth:
        masks = load_original_instance_masks(
            args.original_pseudo_root, dataset_name, split, img_id
        )

    for idx in range(raw_count):
        label = str(phrases[idx]).lower().strip()
        category_id = cat_name_to_id.get(label, -1)
        if category_id < 0:
            continue

        bbox = valid_bbox_xyxy(np.asarray(boxes[idx]).reshape(-1).tolist(), width, height)
        if bbox is None:
            continue
        area_ratio = bbox_area_ratio(bbox, width, height)
        if area_ratio < args.min_2d_area_ratio or area_ratio > args.max_2d_area_ratio:
            continue

        score = 1.0
        if idx < len(confs):
            try:
                score = float(np.asarray(confs[idx]).reshape(-1)[0])
            except Exception:
                score = 1.0

        ann_stub = {
            "image_id": img_id,
            "category_name": label,
            "category_id": int(category_id),
            "bbox2D_tight": [float(x) for x in bbox],
            "bbox2D_proj": [float(x) for x in bbox],
            "score": float(score),
            "visibility": 1.0,
            "truncation": 0.0,
        }
        entry_mask = None
        if masks is not None and idx < len(masks):
            entry_mask = squeeze_instance_mask(masks[idx])
        entries.append(
            Box2DEntry(
                ann=ann_stub,
                bbox_xyxy=[float(x) for x in bbox],
                label=label,
                category_id=int(category_id),
                score=float(score),
                source_index=int(idx),
                mask=entry_mask,
            )
        )
    return entries, {"raw_count": raw_count, "has_masks": masks is not None}


def load_gsam_helpers():
    global _GSAM_MODULE
    if _GSAM_MODULE is None:
        module_path = os.path.join(REPO_ROOT, "tools", "generate_pseudo_label_gsam.py")
        spec = importlib.util.spec_from_file_location("generate_pseudo_label_gsam", module_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load {module_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _GSAM_MODULE = module

    return {
        "SUNRGBD_38_CLASSES": _GSAM_MODULE.SUNRGBD_38_CLASSES,
        "convert_detection_box_to_xyxy_pixels": _GSAM_MODULE.convert_detection_box_to_xyxy_pixels,
        "load_grounding_sam2_models": _GSAM_MODULE.load_grounding_sam2_models,
        "mask_to_xyxy_pixels": _GSAM_MODULE.mask_to_xyxy_pixels,
        "segment_with_grounding_sam2": _GSAM_MODULE.segment_with_grounding_sam2,
        "run_independent_unidepth": _GSAM_MODULE.run_independent_unidepth,
    }


def build_gsam_entries(
    image_path: str,
    image_width: int,
    image_height: int,
    image_id: int,
    text_prompt: str,
    grounding_dino_model,
    sam_predictor,
    gsam_helpers: dict,
    cat_name_to_id: Dict[str, int],
    args: argparse.Namespace,
) -> Tuple[List[Box2DEntry], dict]:
    from grounding_dino.groundingdino.util.inference import load_image

    image_source, image_transformed = load_image(image_path)
    valid_categories = [cat.strip() for cat in text_prompt.split(".") if cat.strip()]
    seg = gsam_helpers["segment_with_grounding_sam2"](
        image_source=image_source,
        image_transformed=image_transformed,
        text_prompt=text_prompt,
        grounding_dino_model=grounding_dino_model,
        sam2_predictor=sam_predictor,
        box_threshold=args.box_threshold,
        text_threshold=args.text_threshold,
        iou_threshold=args.gsam_iou_threshold,
        valid_categories=valid_categories,
    )

    entries: List[Box2DEntry] = []
    det_records = []
    for idx, (box, label, score) in enumerate(
        zip(seg["boxes"], seg["labels"], seg["scores"])
    ):
        bbox = None
        if args.gsam_box_from == "mask" and idx < len(seg.get("masks", [])):
            bbox = gsam_helpers["mask_to_xyxy_pixels"](seg["masks"][idx])
        if bbox is None:
            bbox = gsam_helpers["convert_detection_box_to_xyxy_pixels"](
                box, image_width, image_height
            )
        bbox = valid_bbox_xyxy(bbox, image_width, image_height)
        if bbox is None:
            continue
        area_ratio = bbox_area_ratio(bbox, image_width, image_height)
        if area_ratio < args.min_2d_area_ratio or area_ratio > args.max_2d_area_ratio:
            continue

        label_str = str(label).lower().strip()
        category_id = cat_name_to_id.get(label_str, -1)
        if category_id < 0:
            continue
        ann_stub = {
            "image_id": int(image_id),
            "category_name": label_str,
            "category_id": int(category_id),
            "bbox2D_tight": [float(x) for x in bbox],
            "score": float(score),
            "visibility": 1.0,
            "truncation": 0.0,
        }
        entry_mask = None
        if idx < len(seg.get("masks", [])):
            entry_mask = squeeze_instance_mask(seg["masks"][idx])
        entries.append(
            Box2DEntry(
                ann=ann_stub,
                bbox_xyxy=[float(x) for x in bbox],
                label=label_str,
                category_id=int(category_id),
                score=float(score),
                source_index=int(idx),
                mask=entry_mask,
            )
        )
        det_records.append(
            {
                "image_id": int(image_id),
                "category_name": label_str,
                "category_id": int(category_id),
                "bbox": [float(x) for x in bbox],
                "score": float(score),
                "source": f"grounding_sam2_{args.gsam_box_from}",
            }
        )
    return entries, {"detections": det_records, "raw_count": len(seg["labels"])}


def load_rgb(path: str) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB")).copy()


def resize_rgb(img_np: np.ndarray, size: int) -> np.ndarray:
    pil = Image.fromarray(img_np)
    try:
        resample = Image.Resampling.BILINEAR
    except AttributeError:
        resample = Image.BILINEAR
    return np.asarray(pil.resize((size, size), resample)).copy()


def load_depth_for_sunrgbd(image_root: str, file_path: str) -> Optional[np.ndarray]:
    candidates = []
    depth_path = file_path.replace("/image/", "/depth/").replace(".jpg", ".png")
    candidates.append(resolve_path(image_root, depth_path))
    depth_bfx_path = file_path.replace("/image/", "/depth_bfx/").replace(".jpg", ".png")
    candidates.append(resolve_path(image_root, depth_bfx_path))

    for path in candidates:
        if os.path.exists(path):
            return np.asarray(Image.open(path), dtype=np.float32) / 8000.0

    depth_dir = os.path.dirname(candidates[0])
    if os.path.isdir(depth_dir):
        pngs = [f for f in os.listdir(depth_dir) if f.lower().endswith(".png")]
        if len(pngs) == 1:
            path = os.path.join(depth_dir, pngs[0])
            return np.asarray(Image.open(path), dtype=np.float32) / 8000.0
    return None


def load_original_unidepth(
    pseudo_root: str,
    dataset_name: str,
    split: str,
    image_id: int,
) -> Optional[np.ndarray]:
    path = os.path.join(pseudo_root, dataset_name, split, "depth", f"{int(image_id)}.npy")
    if not os.path.exists(path):
        return None
    return np.load(path).astype(np.float32)


def run_unidepth_for_image(image_rgb: np.ndarray, K: Sequence[Sequence[float]]) -> np.ndarray:
    depth = load_gsam_helpers()["run_independent_unidepth"](
        image_rgb, np.asarray(K, dtype=np.float32)
    )
    return np.asarray(depth, dtype=np.float32)


def resize_depth(depth_np: np.ndarray, size: int) -> np.ndarray:
    pil = Image.fromarray(depth_np)
    try:
        resample = Image.Resampling.NEAREST
    except AttributeError:
        resample = Image.NEAREST
    return np.asarray(pil.resize((size, size), resample), dtype=np.float32)


def world_from_camera_pose(image_root: str, file_path: str, dataset_name: str) -> np.ndarray:
    if dataset_name.upper() == "SUNRGBD":
        r_ext = load_sunrgbd_extrinsics(image_root, file_path)
        r_yz_swap = np.array(
            [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]],
            dtype=np.float32,
        )
        if r_ext is not None:
            r_wc = r_yz_swap @ r_ext.astype(np.float32)
        else:
            r_wc = r_yz_swap
    else:
        r_wc = np.array(
            [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]],
            dtype=np.float32,
        )
    t_wc = np.zeros(3, dtype=np.float32)
    return np.concatenate([r_wc.reshape(-1), t_wc], axis=0).astype(np.float32)


def scale_intrinsics(K: Sequence[Sequence[float]], width: int, height: int, size: int):
    K_np = np.asarray(K, dtype=np.float32)
    sx = float(size) / float(width)
    sy = float(size) / float(height)
    fx = float(K_np[0, 0] * sx)
    fy = float(K_np[1, 1] * sy)
    cx = float(K_np[0, 2] * sx)
    cy = float(K_np[1, 2] * sy)
    return fx, fy, cx, cy, sx, sy


def make_boxer_datum(
    img_info: dict,
    entries: List[Box2DEntry],
    image_root: str,
    dataset_name: str,
    boxer_hw: int,
    depth_source: str,
    pseudo_root: str = "pseudo_label",
    split: str = "train",
    save_unidepth_dir: Optional[str] = None,
) -> Tuple[dict, dict]:
    file_path = img_info["file_path"]
    image_path = resolve_path(image_root, file_path)
    img_np = load_rgb(image_path)
    h, w = img_np.shape[:2]
    if int(img_info["width"]) != w or int(img_info["height"]) != h:
        w = int(img_info["width"])
        h = int(img_info["height"])

    fx, fy, cx, cy, sx, sy = scale_intrinsics(img_info["K"], w, h, boxer_hw)
    img_resized = resize_rgb(img_np, boxer_hw)
    img_torch = BaseLoader.img_to_tensor(img_resized).float()
    cam = BaseLoader.pinhole_from_K(boxer_hw, boxer_hw, fx, fy, cx, cy).float()

    pose_data_np = world_from_camera_pose(image_root, file_path, dataset_name)
    pose_data = torch.from_numpy(pose_data_np).float()
    r_wc = pose_data_np[:9].reshape(3, 3).astype(np.float32)
    t_wc = pose_data_np[9:].astype(np.float32)

    if depth_source == "unidepth":
        depth_np = run_unidepth_for_image(img_np, img_info["K"])
        if save_unidepth_dir is not None:
            os.makedirs(save_unidepth_dir, exist_ok=True)
            np.save(
                os.path.join(save_unidepth_dir, f"{int(img_info['id'])}.npy"),
                depth_np.astype(np.float32),
            )
    elif depth_source == "original_unidepth":
        depth_np = load_original_unidepth(
            pseudo_root=pseudo_root,
            dataset_name=dataset_name,
            split=split,
            image_id=int(img_info["id"]),
        )
    else:
        depth_np = load_depth_for_sunrgbd(image_root, file_path)
    if depth_np is not None:
        depth_resized = resize_depth(depth_np, boxer_hw)
        sdp_w = BaseLoader.sdp_from_depth(depth_resized, fx, fy, cx, cy, r_wc, t_wc)
    else:
        depth_resized = None
        sdp_w = torch.zeros(0, 3, dtype=torch.float32)

    bb2d = []
    labels = []
    category_ids = []
    scores2d = []
    for entry in entries:
        x1, y1, x2, y2 = entry.bbox_xyxy
        bb2d.append([x1 * sx, x2 * sx, y1 * sy, y2 * sy])
        labels.append(entry.label)
        category_ids.append(entry.category_id)
        scores2d.append(entry.score)

    datum = {
        "img0": img_torch,
        "cam0": cam,
        "T_world_rig0": PoseTW(pose_data),
        "sdp_w": sdp_w.float(),
        "time_ns0": int(img_info["id"]),
        "rotated0": torch.tensor(False).reshape(1),
        "bb2d": torch.tensor(bb2d, dtype=torch.float32),
    }
    meta = {
        "depth": depth_np,
        "depth_resized": depth_resized,
        "r_wc": r_wc,
        "t_wc": t_wc,
        "scale_x": sx,
        "scale_y": sy,
        "labels": labels,
        "category_ids": category_ids,
        "scores2d": torch.tensor(scores2d, dtype=torch.float32),
        "image_width": w,
        "image_height": h,
    }
    return datum, meta


def select_precision(device: str, force_precision: Optional[str]):
    if force_precision == "bfloat16":
        return torch.bfloat16
    if force_precision == "float32":
        return torch.float32
    if device == "cuda" and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float32


def run_boxernet(
    boxernet: BoxerNet,
    datum: dict,
    labels: List[str],
    category_ids: List[int],
    scores2d: torch.Tensor,
    thresh3d: float,
    device: str,
    precision_dtype: torch.dtype,
) -> Tuple[ObbTW, List[str], List[int], torch.Tensor, List[int]]:
    if device == "mps":
        outputs = boxernet.forward(datum)
    else:
        with torch.autocast(device_type=device, dtype=precision_dtype):
            outputs = boxernet.forward(datum)

    obb_pr_w = outputs["obbs_pr_w"].cpu()[0]
    if len(obb_pr_w) == 0:
        return obb_pr_w, [], [], torch.zeros(0), []

    sem_ids = torch.tensor(category_ids, dtype=torch.int32)
    obb_pr_w.set_sem_id(sem_ids)
    scores3d = obb_pr_w.prob.squeeze(-1).clone()
    keep = scores3d >= thresh3d
    keep_indices = [i for i in range(len(labels)) if bool(keep[i])]
    obb_pr_w = obb_pr_w[keep].clone()
    labels = [labels[i] for i in keep_indices]
    category_ids = [category_ids[i] for i in keep_indices]
    scores2d = scores2d[keep]
    scores3d = scores3d[keep]
    if len(obb_pr_w) == 0:
        return obb_pr_w, labels, category_ids, torch.zeros(0), keep_indices

    mean_scores = (scores2d + scores3d) / 2.0
    obb_pr_w.set_prob(mean_scores)
    text = torch.stack([string2tensor(pad_string(label, max_len=128)) for label in labels])
    obb_pr_w.set_text(text)
    return obb_pr_w, labels, category_ids, mean_scores, keep_indices


def transform_points_world_to_cam(
    points_world: np.ndarray, r_wc: np.ndarray, t_wc: np.ndarray
) -> np.ndarray:
    return (points_world - t_wc.reshape(1, 3)) @ r_wc


def transform_rotation_world_to_cam(r_world_obj: np.ndarray, r_wc: np.ndarray) -> np.ndarray:
    return r_wc.T @ r_world_obj


def project_points(K: Sequence[Sequence[float]], points_cam: np.ndarray) -> np.ndarray:
    K_np = np.asarray(K, dtype=np.float32)
    z = np.clip(points_cam[:, 2], 1e-6, None)
    u = K_np[0, 0] * points_cam[:, 0] / z + K_np[0, 2]
    v = K_np[1, 1] * points_cam[:, 1] / z + K_np[1, 2]
    return np.stack([u, v], axis=1)


def bbox_from_projected(pts2d: np.ndarray, width: int, height: int) -> List[float]:
    x1 = float(np.min(pts2d[:, 0]))
    y1 = float(np.min(pts2d[:, 1]))
    x2 = float(np.max(pts2d[:, 0]))
    y2 = float(np.max(pts2d[:, 1]))
    x1 = max(0.0, min(float(width - 1), x1))
    y1 = max(0.0, min(float(height - 1), y1))
    x2 = max(0.0, min(float(width - 1), x2))
    y2 = max(0.0, min(float(height - 1), y2))
    return [x1, y1, x2, y2]


def bbox_iou(a: Sequence[float], b: Sequence[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    if union <= 0:
        return 0.0
    return float(inter / union)


def depth_values_for_entry(
    depth_np: Optional[np.ndarray],
    bbox_xyxy: Sequence[float],
    min_pixels: int,
    mask: Optional[np.ndarray] = None,
    use_mask: bool = False,
    require_mask: bool = False,
) -> Tuple[Optional[np.ndarray], str, int]:
    if depth_np is None:
        return None, "missing", 0
    h, w = depth_np.shape[:2]
    x1, y1, x2, y2 = bbox_xyxy
    ix1 = max(0, min(w - 1, int(math.floor(x1))))
    iy1 = max(0, min(h - 1, int(math.floor(y1))))
    ix2 = max(0, min(w, int(math.ceil(x2))))
    iy2 = max(0, min(h, int(math.ceil(y2))))
    if ix2 <= ix1 or iy2 <= iy1:
        return None, "empty_box", 0

    depth = np.asarray(depth_np, dtype=np.float32)
    valid_depth = np.isfinite(depth) & (depth > 0)
    roi = np.zeros((h, w), dtype=bool)
    roi[iy1:iy2, ix1:ix2] = True

    if use_mask:
        mask_np = resize_bool_mask(mask, h, w) if mask is not None else None
        if mask_np is not None and int(mask_np.sum()) > 0:
            region = roi & mask_np
            vals = depth[region & valid_depth]
            if vals.size >= min_pixels:
                return vals.astype(np.float32), "mask", int(vals.size)
            if require_mask:
                return vals.astype(np.float32), "mask_too_few", int(vals.size)
        elif require_mask:
            return None, "missing_mask", 0

    patch = depth[iy1:iy2, ix1:ix2]
    vals = patch[np.isfinite(patch) & (patch > 0)]
    return vals.astype(np.float32), "bbox", int(vals.size)


def depth_gate(
    depth_np: Optional[np.ndarray],
    bbox_xyxy: Sequence[float],
    z_min: float,
    z_max: float,
    center_z: float,
    min_pixels: int,
    min_support: float,
    max_rel_error: float,
    margin: float,
    mask: Optional[np.ndarray] = None,
    use_mask: bool = False,
    require_mask: bool = False,
) -> Tuple[bool, float, float, int, float, str]:
    if depth_np is None:
        return True, -1.0, -1.0, 0, float("nan"), "missing"
    vals, value_source, value_count = depth_values_for_entry(
        depth_np,
        bbox_xyxy,
        min_pixels,
        mask=mask,
        use_mask=use_mask,
        require_mask=require_mask,
    )
    if vals is None:
        return False, 0.0, float("inf"), int(value_count), float("nan"), value_source
    if vals.size < min_pixels:
        return False, 0.0, float("inf"), int(vals.size), float("nan"), value_source
    lo = z_min - margin
    hi = z_max + margin
    support = float(np.mean((vals >= lo) & (vals <= hi)))
    median_depth = float(np.median(vals))
    rel_err = abs(median_depth - center_z) / max(abs(center_z), 1e-3)
    ok = support >= min_support and rel_err <= max_rel_error
    return ok, support, rel_err, int(vals.size), median_depth, value_source


def refine_center_z_with_depth(
    center_cam: np.ndarray,
    corners_cam: np.ndarray,
    depth_np: Optional[np.ndarray],
    entry: Box2DEntry,
    args: argparse.Namespace,
) -> Tuple[np.ndarray, np.ndarray, dict]:
    center = np.asarray(center_cam, dtype=np.float64).reshape(3).copy()
    corners = np.asarray(corners_cam, dtype=np.float64).reshape(-1, 3).copy()
    metrics = {
        "depth_refined": False,
        "depth_refine_source": "disabled",
        "depth_refine_shift": 0.0,
        "depth_refine_target_center_z": float(center[2]),
    }
    if depth_np is None or not np.isfinite(center[2]) or center[2] <= 1e-4:
        metrics["depth_refine_source"] = "missing_depth"
        return center.astype(np.float32), corners.astype(np.float32), metrics

    vals, value_source, value_count = depth_values_for_entry(
        depth_np,
        entry.bbox_xyxy,
        args.min_depth_pixels,
        mask=entry.mask,
        use_mask=args.use_mask_depth_gate,
        require_mask=args.require_mask_depth,
    )
    metrics["depth_refine_source"] = value_source
    metrics["depth_refine_pixels"] = int(value_count)
    if vals is None or vals.size < args.min_depth_pixels:
        return center.astype(np.float32), corners.astype(np.float32), metrics

    median_depth = float(np.median(vals))
    z_min = float(np.min(corners[:, 2]))
    front_offset = max(0.0, float(center[2] - z_min))
    target_center_z = median_depth + front_offset
    raw_shift = target_center_z - float(center[2])
    max_shift = max(0.0, float(args.depth_refine_max_shift_ratio)) * max(float(center[2]), 1e-3)
    shift = float(np.clip(raw_shift, -max_shift, max_shift))
    if abs(shift) < 1e-3:
        metrics.update(
            {
                "depth_refine_median_depth": median_depth,
                "depth_refine_front_offset": front_offset,
                "depth_refine_target_center_z": target_center_z,
            }
        )
        return center.astype(np.float32), corners.astype(np.float32), metrics

    center[2] += shift
    corners[:, 2] += shift
    metrics.update(
        {
            "depth_refined": True,
            "depth_refine_shift": shift,
            "depth_refine_raw_shift": float(raw_shift),
            "depth_refine_median_depth": median_depth,
            "depth_refine_front_offset": front_offset,
            "depth_refine_target_center_z": target_center_z,
        }
    )
    return center.astype(np.float32), corners.astype(np.float32), metrics


def classwise_quality_threshold(label: str, args: argparse.Namespace) -> float:
    label_norm = str(label).lower()
    if label_norm in THIN_OR_WALL_CLASSES:
        return float(args.thin_quality_threshold)
    if label_norm in LARGE_FURNITURE_CLASSES:
        return float(args.large_quality_threshold)
    return float(args.quality_threshold)


def boxer_quality_score(
    boxer_score: float,
    proj_iou: float,
    depth_support: float,
    depth_rel_error: float,
    prior_metrics: dict,
    ground_metrics: dict,
    args: argparse.Namespace,
) -> float:
    score_term = float(np.clip(boxer_score, 0.0, 1.0))
    proj_term = float(np.clip(proj_iou / max(float(args.min_proj_iou) * 4.0, 0.20), 0.0, 1.0))
    support_term = 0.5 if depth_support < 0 else float(np.clip(depth_support, 0.0, 1.0))
    if math.isfinite(depth_rel_error) and depth_rel_error >= 0:
        depth_term = 1.0 - float(np.clip(depth_rel_error / max(float(args.max_rel_depth_error), 1e-3), 0.0, 1.0))
    else:
        depth_term = 0.5

    prior_penalty = 0.0
    prior_violation = prior_metrics.get("prior_violation")
    if prior_violation is not None and math.isfinite(float(prior_violation)):
        prior_penalty = max(0.0, float(prior_violation) - 1.0)

    ground_penalty = 0.0
    if ground_metrics.get("ground_required") and not ground_metrics.get("ground_ok", True):
        ground_penalty = 0.5

    quality = (
        0.35 * score_term
        + 0.25 * proj_term
        + 0.25 * support_term
        + 0.15 * depth_term
        - 0.10 * min(prior_penalty, 2.0)
        - 0.15 * ground_penalty
    )
    return float(np.clip(quality, 0.0, 1.0))


def obb_to_omni3d_fields(
    obb: ObbTW,
    img_info: dict,
    entry: Box2DEntry,
    r_wc: np.ndarray,
    t_wc: np.ndarray,
    depth_np: Optional[np.ndarray],
    category_prior: Dict[str, Sequence[float]],
    ground_equ: Optional[np.ndarray],
    args: argparse.Namespace,
) -> Tuple[dict, dict]:
    corners_world = obb.bb3corners_world.detach().cpu().numpy().astype(np.float32)
    corners_cam = transform_points_world_to_cam(corners_world, r_wc, t_wc)
    center_world = obb.bb3_center_world.detach().cpu().numpy().astype(np.float32)
    center_cam = transform_points_world_to_cam(center_world.reshape(1, 3), r_wc, t_wc)[0]
    dims_object_xyz = obb.bb3_diagonal.detach().cpu().numpy().astype(np.float32)
    dims = boxer_dims_to_omni(dims_object_xyz).astype(np.float32)
    r_world_obj = obb.T_world_object.R.detach().cpu().numpy().astype(np.float32)
    r_cam_obj = transform_rotation_world_to_cam(r_world_obj, r_wc)
    score = float(obb.prob.squeeze(-1).item())

    center_cam, dims, corners_cam, adjust_metrics = apply_original_prior_ground_adjustments(
        entry.label,
        center_cam,
        dims,
        r_cam_obj,
        category_prior,
        ground_equ,
        args,
    )
    depth_refine_metrics = {
        "depth_refined": False,
        "depth_refine_source": "disabled",
        "depth_refine_shift": 0.0,
    }
    if args.boxer_refine_with_depth:
        center_cam, corners_cam, depth_refine_metrics = refine_center_z_with_depth(
            center_cam,
            corners_cam,
            depth_np,
            entry,
            args,
        )

    width = int(img_info["width"])
    height = int(img_info["height"])
    behind_camera = bool(np.any(corners_cam[:, 2] <= 1e-4) or center_cam[2] <= 1e-4)
    pts2d = project_points(img_info["K"], corners_cam)
    bbox_proj = bbox_from_projected(pts2d, width, height)
    proj_iou = bbox_iou(entry.bbox_xyxy, bbox_proj)

    z_min = float(np.min(corners_cam[:, 2]))
    z_max = float(np.max(corners_cam[:, 2]))
    depth_ok, depth_support, depth_rel_err, depth_pixels, depth_median, depth_value_source = depth_gate(
        depth_np,
        entry.bbox_xyxy,
        z_min,
        z_max,
        float(center_cam[2]),
        args.min_depth_pixels,
        args.min_depth_support,
        args.max_rel_depth_error,
        args.depth_margin,
        mask=entry.mask,
        use_mask=args.use_mask_depth_gate,
        require_mask=args.require_mask_depth,
    )

    dim_ok = bool(
        np.all(np.isfinite(dims))
        and np.all(dims >= args.min_dimension)
        and np.all(dims <= args.max_dimension)
    )
    prior_ok, prior_metrics = category_prior_gate(
        entry.label,
        dims,
        category_prior,
        args.prior_min_ratio,
        args.prior_max_ratio,
    )
    if args.no_prior_gate:
        prior_ok = True

    ground_ok, ground_metrics = ground_consistency_gate(
        entry.label,
        corners_cam,
        ground_equ,
        args.ground_max_distance,
        args.require_ground,
    )
    if args.no_ground_gate:
        ground_ok = True

    proj_ok = args.no_projection_gate or proj_iou >= args.min_proj_iou
    depth_gate_ok = args.no_depth_gate or depth_ok
    quality = boxer_quality_score(
        score,
        proj_iou,
        depth_support,
        depth_rel_err,
        prior_metrics,
        ground_metrics,
        args,
    )
    quality_threshold = classwise_quality_threshold(entry.label, args)
    quality_ok = (not args.classwise_quality_gate) or quality >= quality_threshold
    valid = bool(
        (not behind_camera)
        and dim_ok
        and prior_ok
        and ground_ok
        and proj_ok
        and depth_gate_ok
        and quality_ok
    )

    metrics = {
        "valid": valid,
        "behind_camera": behind_camera,
        "dim_ok": dim_ok,
        "prior_ok": bool(prior_ok),
        "ground_ok": bool(ground_ok),
        "projection_iou": float(proj_iou),
        "depth_ok": bool(depth_ok),
        "depth_support": float(depth_support),
        "depth_rel_error": float(depth_rel_err),
        "depth_pixels": int(depth_pixels),
        "depth_median": float(depth_median) if math.isfinite(depth_median) else -1.0,
        "depth_value_source": depth_value_source,
        "score": score,
        "quality_ok": bool(quality_ok),
        "boxer_quality": float(quality),
        "boxer_quality_threshold": float(quality_threshold),
        "nms_suppressed": False,
        **adjust_metrics,
        **depth_refine_metrics,
        **prior_metrics,
        **ground_metrics,
    }
    fields = {
        "category_name": entry.label,
        "category_id": entry.category_id,
        "valid3D": valid,
        "bbox2D_tight": [float(x) for x in entry.bbox_xyxy],
        "bbox2D_trunc": [float(x) for x in entry.bbox_xyxy],
        "bbox2D_proj": [float(x) for x in bbox_proj],
        "bbox3D_cam": [[float(v) for v in row] for row in corners_cam.tolist()],
        "center_cam": [float(v) for v in center_cam.tolist()],
        "dimensions": [float(v) for v in dims.tolist()],
        "R_cam": [[float(v) for v in row] for row in r_cam_obj.tolist()],
        "behind_camera": behind_camera,
        "visibility": float(entry.ann.get("visibility", 1.0)),
        "truncation": float(entry.ann.get("truncation", 0.0)),
        "segmentation_pts": int(entry.ann.get("segmentation_pts", -1)),
        "lidar_pts": int(entry.ann.get("lidar_pts", -1)),
        "depth_error": float(depth_rel_err) if math.isfinite(depth_rel_err) else -1.0,
        "score": score,
        "boxer_projection_iou": float(proj_iou),
        "boxer_depth_support": float(depth_support),
        "boxer_depth_rel_error": float(depth_rel_err)
        if math.isfinite(depth_rel_err)
        else -1.0,
        "boxer_depth_median": float(depth_median) if math.isfinite(depth_median) else -1.0,
        "boxer_depth_value_source": depth_value_source,
        "boxer_quality": float(quality),
        "boxer_quality_threshold": float(quality_threshold),
        "boxer_quality_ok": bool(quality_ok),
        "boxer_prior_ok": bool(prior_ok),
        "boxer_ground_ok": bool(ground_ok),
        "boxer_dimensions_raw": adjust_metrics["raw_dimensions"],
        "boxer_center_cam_raw": adjust_metrics["raw_center_cam"],
        "boxer_depth_refined": bool(depth_refine_metrics.get("depth_refined", False)),
        "boxer_depth_refine_shift": float(depth_refine_metrics.get("depth_refine_shift", 0.0)),
        "boxer_prior_adjusted": bool(adjust_metrics["prior_adjusted"]),
        "boxer_prior_adjusted_axes": adjust_metrics["prior_adjusted_axes"],
        "boxer_ground_snapped": bool(adjust_metrics["ground_snapped"]),
        "boxer_ground_snap_distance": float(adjust_metrics["ground_snap_distance"]),
    }
    if prior_metrics.get("prior_available"):
        fields["boxer_prior_ratio"] = prior_metrics.get("prior_ratio", [])
    if ground_metrics.get("ground_available"):
        fields["boxer_ground_min_corner_distance"] = float(
            ground_metrics.get("ground_min_corner_distance", -1.0)
        )
    return fields, metrics


def to_device_datum(datum: dict, device: str) -> dict:
    moved = {}
    for key, value in datum.items():
        if isinstance(value, torch.Tensor):
            if "rotated" in key:
                moved[key] = value.to(device).bool()
            else:
                moved[key] = value.to(device).float()
        else:
            moved[key] = value
    return moved


def update_stats(stats: dict, metrics: dict, reason_prefix: str = ""):
    if metrics.get("prior_adjusted", False):
        stats["prior_adjusted"] += 1
    if metrics.get("ground_snapped", False):
        stats["ground_snapped"] += 1
    if metrics.get("depth_refined", False):
        stats["depth_refined"] += 1
    depth_source = metrics.get("depth_value_source")
    if depth_source == "mask":
        stats["depth_gate_mask"] += 1
    elif depth_source == "bbox":
        stats["depth_gate_bbox"] += 1
    if metrics["valid"]:
        stats["valid3d"] += 1
        return
    stats["rejected"] += 1
    if metrics["behind_camera"]:
        stats["reject_behind_camera"] += 1
    if not metrics["dim_ok"]:
        stats["reject_dimension"] += 1
    if not metrics.get("prior_ok", True):
        stats["reject_prior"] += 1
    if not metrics.get("ground_ok", True):
        stats["reject_ground"] += 1
    if metrics["projection_iou"] < stats["min_proj_iou_threshold"]:
        stats["reject_projection"] += 1
    if not metrics["depth_ok"]:
        stats["reject_depth"] += 1
    if not metrics.get("quality_ok", True):
        stats["reject_quality"] += 1
    if metrics.get("nms_suppressed", False):
        stats["reject_nms"] += 1


def candidate_nms_key(candidate: dict) -> float:
    fields = candidate["fields"]
    metrics = candidate["metrics"]
    return float(fields.get("boxer_quality", metrics.get("boxer_quality", fields.get("score", 0.0))))


def suppress_candidate_by_nms(candidate: dict, kept: dict, args: argparse.Namespace) -> bool:
    fields = candidate["fields"]
    kept_fields = kept["fields"]
    if int(fields.get("category_id", -1)) != int(kept_fields.get("category_id", -2)):
        return False

    iou2d = bbox_iou(fields["bbox2D_tight"], kept_fields["bbox2D_tight"])
    if iou2d >= float(args.boxer_nms_iou):
        return True

    center = np.asarray(fields.get("center_cam", [0.0, 0.0, 0.0]), dtype=np.float64)
    kept_center = np.asarray(kept_fields.get("center_cam", [0.0, 0.0, 0.0]), dtype=np.float64)
    dims = np.asarray(fields.get("dimensions", [1.0, 1.0, 1.0]), dtype=np.float64)
    kept_dims = np.asarray(kept_fields.get("dimensions", [1.0, 1.0, 1.0]), dtype=np.float64)
    if center.size != 3 or kept_center.size != 3 or dims.size != 3 or kept_dims.size != 3:
        return False
    if not (
        np.all(np.isfinite(center))
        and np.all(np.isfinite(kept_center))
        and np.all(np.isfinite(dims))
        and np.all(np.isfinite(kept_dims))
    ):
        return False
    avg_diag = 0.5 * (float(np.linalg.norm(dims)) + float(np.linalg.norm(kept_dims)))
    if avg_diag <= 1e-6:
        return False
    center_ratio = float(np.linalg.norm(center - kept_center) / avg_diag)
    return bool(iou2d >= 0.30 and center_ratio <= float(args.boxer_nms_center_ratio))


def apply_boxer_candidate_nms(candidates: List[dict], args: argparse.Namespace) -> List[dict]:
    if not args.boxer_nms:
        return candidates

    valid_candidates = [c for c in candidates if c["metrics"].get("valid", False)]
    sorted_valid = sorted(valid_candidates, key=candidate_nms_key, reverse=True)
    kept: List[dict] = []
    suppressed = set()

    for candidate in sorted_valid:
        idx = candidate["index"]
        if any(suppress_candidate_by_nms(candidate, kept_candidate, args) for kept_candidate in kept):
            suppressed.add(idx)
            continue
        kept.append(candidate)

    for candidate in candidates:
        if candidate["index"] not in suppressed:
            continue
        candidate["metrics"]["valid"] = False
        candidate["metrics"]["nms_suppressed"] = True
        candidate["fields"]["valid3D"] = False
        candidate["fields"]["boxer_nms_suppressed"] = True
    return candidates


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    bbox_fields = [x.strip() for x in args.bbox_fields.split(",") if x.strip()]
    dataset_name = "SUNRGBD"

    with open(args.json_file, "r") as f:
        source = json.load(f)

    if source.get("info", {}).get("source"):
        dataset_name = str(source["info"]["source"])
    if args.dataset is not None:
        dataset_name = str(args.dataset)
    split = args.split or infer_split_from_json(args.json_file, source)

    images = source.get("images", [])
    images_by_id = {int(img["id"]): img for img in images}
    cat_name_to_id, sem_id_to_name = category_maps(source)
    category_prior = get_category_prior(dataset_name)

    grouped = {}
    external_2d_stats = None
    if args.box_source == "json":
        grouped = group_annotations_by_image(
            source.get("annotations", []),
            images_by_id,
            bbox_fields,
            min_area_ratio=args.min_2d_area_ratio,
            max_area_ratio=args.max_2d_area_ratio,
        )
    elif args.box_source == "external_2d":
        grouped, external_2d_stats = build_external_2d_grouped(
            args.external_2d_json,
            images_by_id,
            cat_name_to_id,
            sem_id_to_name,
            args,
        )
    elif args.box_source == "detany3d":
        detany3d_json = args.detany3d_json or args.external_2d_json
        if not detany3d_json:
            raise ValueError("--detany3d_json is required when --box_source detany3d")
        detany3d_args = copy.copy(args)
        detany3d_args.external_2d_root = args.detany3d_root or args.external_2d_root
        detany3d_args.external_2d_bbox_format = args.detany3d_bbox_format
        detany3d_args.external_2d_score_threshold = args.detany3d_score_threshold
        grouped, external_2d_stats = build_external_2d_grouped(
            detany3d_json,
            images_by_id,
            cat_name_to_id,
            sem_id_to_name,
            detany3d_args,
        )

    original_gsam_info = None
    original_gsam_info_path = None
    if args.box_source == "original_gsam":
        original_gsam_info, original_gsam_info_path = load_original_gsam_info(
            args.original_pseudo_root,
            dataset_name,
            split,
        )

    original_ground_info, original_ground_info_path = load_original_ground_info(
        args.original_pseudo_root,
        dataset_name,
        split,
    )

    selected = images[args.start_index :: max(args.skip_images, 1)]
    if args.max_images is not None:
        selected = selected[: args.max_images]

    if args.output_json is None:
        stem = os.path.splitext(os.path.basename(args.json_file))[0]
        args.output_json = os.path.join(args.output_dir, f"{stem}_boxer.json")
    csv_path = os.path.join(args.output_dir, "boxer_3dbbs.csv")
    stats_path = args.stats_json or os.path.join(args.output_dir, "boxer_stats.json")
    gsam_jsonl_path = os.path.join(args.output_dir, "gsam_2dbbs.jsonl")
    unidepth_dir = os.path.join(args.output_dir, "unidepth") if args.save_unidepth else None

    if args.force_cpu:
        device = "cpu"
    elif torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
        device = "cuda"
    else:
        device = "cpu"

    gsam_helpers = None
    grounding_dino_model = None
    sam_predictor = None
    text_prompt = args.text_prompt
    gsam_jsonl = None
    if args.box_source == "gsam":
        if device != "cuda":
            raise RuntimeError("Grounding-SAM2 mode requires CUDA in this project setup.")
        gsam_helpers = load_gsam_helpers()
        if text_prompt is None:
            text_prompt = gsam_helpers["SUNRGBD_38_CLASSES"]
        print("Loading Grounding-SAM2 models for 2D boxes...")
        grounding_dino_model, sam_predictor = gsam_helpers["load_grounding_sam2_models"](
            grounding_dino_checkpoint=args.grounding_dino_checkpoint,
            sam_checkpoint=args.sam_checkpoint,
            use_large_gdino=args.use_large_gdino,
        )
        gsam_jsonl = open(gsam_jsonl_path, "w")

    print(f"Loading BoxerNet checkpoint: {args.ckpt}")
    boxernet = BoxerNet.load_from_checkpoint(args.ckpt, device=device)
    boxernet.eval()
    boxer_hw = int(boxernet.hw)
    precision_dtype = select_precision(device, args.force_precision)
    print(f"Using device={device}, boxer_hw={boxer_hw}, precision={precision_dtype}")
    print(f"Input images={len(images)}, selected={len(selected)}")
    print(f"Loaded original category priors: {len(category_prior)} for {dataset_name}")
    if original_ground_info_path is not None:
        print(f"Loaded original ground info: {original_ground_info_path}")
    elif not args.no_ground_gate:
        print("Original ground info not found; ground gate will pass unless --require_ground is set.")

    writer = ObbCsvWriter2(csv_path)
    annotations_out = []
    dataset_id = int(source.get("info", {}).get("id", source.get("dataset_id", 12)))
    ann_serial = 1

    stats = {
        "box_source": args.box_source,
        "depth_source": args.depth_source,
        "dataset": dataset_name,
        "split": split,
        "original_pseudo_root": os.path.abspath(args.original_pseudo_root),
        "original_gsam_info": os.path.abspath(original_gsam_info_path)
        if original_gsam_info_path is not None
        else None,
        "original_ground_info": os.path.abspath(original_ground_info_path)
        if original_ground_info_path is not None
        else None,
        "external_2d_json": os.path.abspath(args.external_2d_json)
        if args.external_2d_json is not None
        else None,
        "detany3d_json": os.path.abspath(args.detany3d_json)
        if args.detany3d_json is not None
        else None,
        "external_2d_stats": external_2d_stats,
        "category_priors": len(category_prior),
        "images_total": len(images),
        "images_selected": len(selected),
        "images_seen": 0,
        "images_with_2d": 0,
        "images_missing_depth": 0,
        "gsam_raw_2d": 0,
        "input_2d": 0,
        "filtered_2d": 0,
        "boxer_pred": 0,
        "valid3d": 0,
        "rejected": 0,
        "reject_behind_camera": 0,
        "reject_dimension": 0,
        "reject_prior": 0,
        "reject_ground": 0,
        "reject_projection": 0,
        "reject_depth": 0,
        "reject_quality": 0,
        "reject_nms": 0,
        "images_with_ground": 0,
        "images_missing_ground": 0,
        "images_with_masks": 0,
        "prior_adjusted": 0,
        "ground_snapped": 0,
        "depth_refined": 0,
        "depth_gate_mask": 0,
        "depth_gate_bbox": 0,
        "min_proj_iou_threshold": args.min_proj_iou,
        "thresh3d": args.thresh3d,
        "prior_min_ratio": args.prior_min_ratio,
        "prior_max_ratio": args.prior_max_ratio,
        "prior_adjust_min_ratio": args.prior_adjust_min_ratio,
        "prior_adjust_enabled": not args.no_prior_adjust,
        "ground_max_distance": args.ground_max_distance,
        "ground_snap_enabled": not args.no_ground_snap,
        "mask_depth_gate_enabled": bool(args.use_mask_depth_gate),
        "depth_refine_enabled": bool(args.boxer_refine_with_depth),
        "classwise_quality_gate_enabled": bool(args.classwise_quality_gate),
        "quality_threshold": args.quality_threshold,
        "thin_quality_threshold": args.thin_quality_threshold,
        "large_quality_threshold": args.large_quality_threshold,
        "boxer_nms_enabled": bool(args.boxer_nms),
        "boxer_nms_iou": args.boxer_nms_iou,
        "boxer_nms_center_ratio": args.boxer_nms_center_ratio,
    }

    pbar = tqdm(selected, desc="Boxer Omni3D")
    for img_info in pbar:
        stats["images_seen"] += 1
        img_id = int(img_info["id"])
        if args.box_source == "gsam":
            image_path = resolve_path(args.image_root, img_info["file_path"])
            entries, gsam_meta = build_gsam_entries(
                image_path=image_path,
                image_width=int(img_info["width"]),
                image_height=int(img_info["height"]),
                image_id=img_id,
                text_prompt=text_prompt,
                grounding_dino_model=grounding_dino_model,
                sam_predictor=sam_predictor,
                gsam_helpers=gsam_helpers,
                cat_name_to_id=cat_name_to_id,
                args=args,
            )
            stats["gsam_raw_2d"] += int(gsam_meta["raw_count"])
            stats["filtered_2d"] += max(0, int(gsam_meta["raw_count"]) - len(entries))
            if any(entry.mask is not None for entry in entries):
                stats["images_with_masks"] += 1
            if gsam_jsonl is not None:
                for record in gsam_meta["detections"]:
                    gsam_jsonl.write(json.dumps(record) + "\n")
                gsam_jsonl.flush()
        elif args.box_source == "original_gsam":
            entries, gsam_meta = build_original_gsam_entries(
                img_info=img_info,
                original_info=original_gsam_info,
                cat_name_to_id=cat_name_to_id,
                dataset_name=dataset_name,
                split=split,
                args=args,
            )
            stats["gsam_raw_2d"] += int(gsam_meta["raw_count"])
            stats["filtered_2d"] += max(0, int(gsam_meta["raw_count"]) - len(entries))
            if gsam_meta.get("has_masks", False):
                stats["images_with_masks"] += 1
        else:
            entries = grouped.get(img_id, [])
            if args.box_source in ("external_2d", "detany3d"):
                stats["gsam_raw_2d"] += len(entries)
                if any(entry.mask is not None for entry in entries):
                    stats["images_with_masks"] += 1
        if len(entries) == 0:
            continue
        stats["images_with_2d"] += 1
        stats["input_2d"] += len(entries)

        datum, meta = make_boxer_datum(
            img_info,
            entries,
            args.image_root,
            dataset_name,
            boxer_hw,
            depth_source=args.depth_source,
            pseudo_root=args.original_pseudo_root,
            split=split,
            save_unidepth_dir=unidepth_dir,
        )
        if meta["depth"] is None:
            stats["images_missing_depth"] += 1
            if args.require_depth:
                continue

        ground_equ = estimate_original_ground_equation(
            info_ground=original_ground_info,
            pseudo_root=args.original_pseudo_root,
            dataset_name=dataset_name,
            split=split,
            image_id=img_id,
            depth_np=meta["depth"],
            K=img_info["K"],
        )
        if ground_equ is None:
            stats["images_missing_ground"] += 1
        else:
            stats["images_with_ground"] += 1

        datum = to_device_datum(datum, device)
        with torch.no_grad():
            obbs, labels, category_ids, _scores, keep_indices = run_boxernet(
                boxernet,
                datum,
                meta["labels"],
                meta["category_ids"],
                meta["scores2d"],
                args.thresh3d,
                device,
                precision_dtype,
            )

        stats["boxer_pred"] += len(obbs)
        if len(obbs) == 0:
            pbar.set_postfix_str(
                f"valid={stats['valid3d']} input2d={stats['input_2d']}"
            )
            continue

        candidates = []
        for local_idx, obb in enumerate(obbs):
            entry = entries[keep_indices[local_idx]]
            fields, metrics = obb_to_omni3d_fields(
                obb,
                img_info,
                entry,
                meta["r_wc"],
                meta["t_wc"],
                meta["depth"],
                category_prior,
                ground_equ,
                args,
            )
            candidates.append(
                {
                    "index": int(local_idx),
                    "obb": obb,
                    "fields": fields,
                    "metrics": metrics,
                }
            )

        candidates = apply_boxer_candidate_nms(candidates, args)

        kept_obbs = []
        for candidate in candidates:
            fields = candidate["fields"]
            metrics = candidate["metrics"]
            obb = candidate["obb"]
            update_stats(stats, metrics)
            if metrics["valid"] or args.save_invalid:
                out_ann = {
                    "id": int(dataset_id * 10000000 + ann_serial),
                    "image_id": img_id,
                    "dataset_id": dataset_id,
                    **fields,
                }
                annotations_out.append(out_ann)
                ann_serial += 1
            if metrics["valid"]:
                kept_obbs.append(obb._data)

        if kept_obbs:
            obbs_valid = ObbTW(torch.stack(kept_obbs))
            writer.write(obbs_valid, img_id, sem_id_to_name=sem_id_to_name)

        pbar.set_postfix_str(
            f"valid={stats['valid3d']} pred={stats['boxer_pred']} rej={stats['rejected']}"
        )

    writer.close()
    if gsam_jsonl is not None:
        gsam_jsonl.close()

    output = copy.deepcopy(source)
    output["annotations"] = annotations_out
    output.setdefault("info", {})
    output["info"]["name"] = f"{output['info'].get('name', dataset_name)} Boxer pseudo"
    output["info"]["pseudo_label_method"] = "BoxerNet"
    output["info"]["box_source"] = args.box_source
    output["info"]["depth_source"] = args.depth_source
    output["info"]["boxer_dataset"] = dataset_name
    output["info"]["boxer_split"] = split
    output["info"]["boxer_original_pseudo_root"] = os.path.abspath(args.original_pseudo_root)
    if original_gsam_info_path is not None:
        output["info"]["boxer_original_gsam_info"] = os.path.abspath(original_gsam_info_path)
    if original_ground_info_path is not None:
        output["info"]["boxer_original_ground_info"] = os.path.abspath(original_ground_info_path)
    output["info"]["boxer_category_prior"] = f"cubercnn.generate_label.llm_generated_prior.{dataset_name}"
    output["info"]["boxer_prior_gate"] = not args.no_prior_gate
    output["info"]["boxer_prior_adjust"] = not args.no_prior_adjust
    output["info"]["boxer_ground_gate"] = not args.no_ground_gate
    output["info"]["boxer_ground_snap"] = not args.no_ground_snap
    output["info"]["boxer_mask_depth_gate"] = bool(args.use_mask_depth_gate)
    output["info"]["boxer_depth_refine"] = bool(args.boxer_refine_with_depth)
    output["info"]["boxer_classwise_quality_gate"] = bool(args.classwise_quality_gate)
    output["info"]["boxer_nms"] = bool(args.boxer_nms)
    output["info"]["boxer_source_json"] = os.path.abspath(args.json_file)
    output["info"]["boxer_csv"] = os.path.abspath(csv_path)
    if args.box_source == "gsam":
        output["info"]["gsam_2dbbs"] = os.path.abspath(gsam_jsonl_path)
    if args.box_source == "external_2d" and args.external_2d_json is not None:
        output["info"]["external_2d_json"] = os.path.abspath(args.external_2d_json)
        output["info"]["external_2d_bbox_format"] = args.external_2d_bbox_format
        output["info"]["external_2d_score_threshold"] = args.external_2d_score_threshold
    if args.box_source == "detany3d":
        output["info"]["detany3d_json"] = os.path.abspath(
            args.detany3d_json or args.external_2d_json
        )
        output["info"]["detany3d_bbox_format"] = args.detany3d_bbox_format
        output["info"]["detany3d_score_threshold"] = args.detany3d_score_threshold

    with open(args.output_json, "w") as f:
        json.dump(output, f)
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    print(f"Wrote Boxer CSV: {csv_path}")
    print(f"Wrote Omni3D JSON: {args.output_json}")
    print(f"Wrote stats: {stats_path}")
    print(
        "Summary: "
        f"input_2d={stats['input_2d']} "
        f"boxer_pred={stats['boxer_pred']} "
        f"valid3d={stats['valid3d']} "
        f"rejected={stats['rejected']}"
    )


if __name__ == "__main__":
    main()
