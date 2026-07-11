#!/usr/bin/env python3
"""Generate Omni3D pseudo labels with GroundedSAM 2D boxes + MoCA3D-Cube.

This is an intentionally clean branch:

    original GroundedSAM cache -> MoCA3D image-plane corners/depth
    -> MoCA3D-Cube adapter -> Omni3D/Cube R-CNN compatible JSON

It does not use pseudo LiDAR, PCA, DFU, Boxer, SOR, or any geometry fallback
unless you add that outside this script.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MOCA_ROOT = REPO_ROOT / "third_party" / "MoCA3D"
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("GroundedSAM -> MoCA3D-Cube -> Omni3D JSON")
    parser.add_argument("--json_file", required=True, help="Input Omni3D JSON split.")
    parser.add_argument("--image_root", default="datasets")
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--output_dir", default=None, help="Optional stats/debug dir.")
    parser.add_argument("--dataset", default=None, help="Dataset name, e.g. SUNRGBD.")
    parser.add_argument("--split", choices=["train", "val", "test"], default=None)
    parser.add_argument(
        "--original_pseudo_root",
        default="pseudo_label",
        help="Root containing original GroundedSAM cache info.pth.",
    )
    parser.add_argument("--moca_root", default=str(DEFAULT_MOCA_ROOT))
    parser.add_argument(
        "--moca_config",
        default=None,
        help="Defaults to third_party/MoCA3D/configs/MoCA_config.yaml.",
    )
    parser.add_argument(
        "--moca_checkpoint",
        default=None,
        help="Defaults to third_party/MoCA3D/checkpoints/moca3d.safetensors.",
    )
    parser.add_argument(
        "--cube_config",
        default=None,
        help="Defaults to third_party/MoCA3D/configs/MoCA_cube_config.yaml.",
    )
    parser.add_argument(
        "--cube_checkpoint",
        default=None,
        help=(
            "MoCA3D-Cube checkpoint. If omitted, the script tries "
            "checkpoints/MoCA3D_Cube/best_iou_inv_cube.pth then best_iou_inv.pth."
        ),
    )
    parser.add_argument(
        "--joint_checkpoint",
        default=None,
        help="Optional joint checkpoint containing both moca_model and cube_model.",
    )
    parser.add_argument("--prefer_ema", action="store_true")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--force_cpu", action="store_true")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_images", type=int, default=None)
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--skip_images", type=int, default=1)
    parser.add_argument("--score_threshold", type=float, default=0.0)
    parser.add_argument("--min_2d_area", type=float, default=4.0)
    parser.add_argument("--min_depth", type=float, default=0.05)
    parser.add_argument("--max_depth", type=float, default=80.0)
    parser.add_argument("--min_dimension", type=float, default=0.01)
    parser.add_argument("--max_dimension", type=float, default=20.0)
    parser.add_argument("--save_invalid", action="store_true")
    parser.add_argument(
        "--precision",
        choices=["float32", "float16", "bfloat16"],
        default="float32",
    )
    parser.add_argument(
        "--strict_moca",
        action="store_true",
        help="Use strict=True when loading MoCA checkpoint.",
    )
    parser.add_argument(
        "--strict_cube",
        action="store_true",
        help="Use strict=True when loading Cube checkpoint.",
    )
    return parser.parse_args()


def infer_dataset_split(json_file: str) -> Tuple[str, str]:
    stem = Path(json_file).stem
    if "_" not in stem:
        return stem, "train"
    dataset, split = stem.rsplit("_", 1)
    return dataset, split


def setup_moca_imports(moca_root: Path):
    if not moca_root.exists():
        raise FileNotFoundError(
            f"MoCA3D root not found: {moca_root}. Clone jeoncwcw/MoCA3D under third_party/MoCA3D."
        )
    sys.path.insert(0, str(moca_root))


def resolve_path(path_like: Optional[str], moca_root: Path) -> Optional[Path]:
    if path_like is None:
        return None
    path = Path(path_like).expanduser()
    if path.is_absolute():
        return path
    for base in (Path.cwd(), REPO_ROOT, moca_root):
        candidate = base / path
        if candidate.exists():
            return candidate
    return moca_root / path


def load_torch_checkpoint(path: Path, device: torch.device):
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    if path.suffix == ".safetensors":
        from safetensors.torch import load_file as load_safetensors

        return load_safetensors(str(path), device=str(device))
    return torch.load(path, map_location=device)


def extract_state_dict(obj):
    if isinstance(obj, dict):
        for key in ("state_dict", "model", "model_state_dict"):
            if key in obj and isinstance(obj[key], dict):
                return obj[key]
        return obj
    raise TypeError(f"Unsupported checkpoint object: {type(obj)}")


def strip_prefixes(state_dict: dict) -> dict:
    return {
        str(k).replace("module.", "").replace("_orig_mod.", ""): v
        for k, v in state_dict.items()
    }


def extract_named_state_dict(obj: dict, keys: Sequence[str]) -> Optional[dict]:
    if not isinstance(obj, dict):
        return None
    for key in keys:
        if key in obj and isinstance(obj[key], dict):
            return strip_prefixes(extract_state_dict(obj[key]))
    return None


def split_prefixed_joint_state_dict(obj: dict) -> Tuple[Optional[dict], Optional[dict]]:
    if not isinstance(obj, dict):
        return None, None
    source = None
    for key in ("state_dict", "model", "model_state_dict", "joint_state_dict", "joint_model"):
        if key in obj and isinstance(obj[key], dict):
            source = obj[key]
            break
    if source is None:
        tensor_items = {k: v for k, v in obj.items() if isinstance(v, torch.Tensor)}
        source = tensor_items or None
    if source is None:
        return None, None

    state = strip_prefixes(source)
    moca_prefixes = ("moca_model.", "moca.", "model.moca_model.", "joint_model.moca_model.")
    cube_prefixes = (
        "cube_model.",
        "bbx3d_model.",
        "bbox3d_model.",
        "cube_head.",
        "bbx3d_head.",
        "joint_model.cube_model.",
        "joint_model.bbx3d_model.",
    )
    moca_state, cube_state = {}, {}
    for key, value in state.items():
        matched = False
        for prefix in moca_prefixes:
            if key.startswith(prefix):
                moca_state[key[len(prefix):]] = value
                matched = True
                break
        if matched:
            continue
        for prefix in cube_prefixes:
            if key.startswith(prefix):
                cube_state[key[len(prefix):]] = value
                break
    return moca_state or None, cube_state or None


def default_cube_checkpoint(moca_root: Path) -> Path:
    ckpt_dir = moca_root / "checkpoints" / "MoCA3D_Cube"
    for name in (
        "best_iou_inv_cube.pth",
        "moca3d_cube_adapter_best_ema.pth",
        "best_iou_inv.pth",
        "moca3d_cube_joint_best_ema.pt",
        "best_iou_inv_joint.pt",
        "latest.pt",
    ):
        path = ckpt_dir / name
        if path.exists():
            return path
    return ckpt_dir / "best_iou_inv_cube.pth"


def default_joint_checkpoint(moca_root: Path) -> Optional[Path]:
    """Mirror MoCA3D tools/evaluate_cube.py: prefer the joint Cube checkpoint."""
    ckpt_dir = moca_root / "checkpoints" / "MoCA3D_Cube"
    for name in ("best_iou_inv_joint.pt", "moca3d_cube_joint_best_ema.pt"):
        path = ckpt_dir / name
        if path.exists():
            return path
    return None


def build_models(args: argparse.Namespace, device: torch.device):
    moca_root = Path(args.moca_root).expanduser().resolve()
    setup_moca_imports(moca_root)

    from omegaconf import OmegaConf
    from models.moca_3d import Moca3DModel
    from models.moca_3d_cube import BBox3DMLP

    moca_config = resolve_path(args.moca_config, moca_root) or (moca_root / "configs" / "MoCA_config.yaml")
    cube_config = resolve_path(args.cube_config, moca_root) or (moca_root / "configs" / "MoCA_cube_config.yaml")
    moca_checkpoint = resolve_path(args.moca_checkpoint, moca_root) or (moca_root / "checkpoints" / "moca3d.safetensors")
    cube_checkpoint = resolve_path(args.cube_checkpoint, moca_root) if args.cube_checkpoint else default_cube_checkpoint(moca_root)
    joint_checkpoint = resolve_path(args.joint_checkpoint, moca_root)
    if (
        joint_checkpoint is None
        and args.moca_checkpoint is None
        and args.cube_checkpoint is None
    ):
        joint_checkpoint = default_joint_checkpoint(moca_root)

    moca_cfg = OmegaConf.load(moca_config)
    cube_cfg = OmegaConf.load(cube_config)
    moca_cfg.device = str(device)
    moca_cfg.feature_mode = False
    moca_cfg.dinov3_checkpoint_path = str(resolve_path(moca_cfg.dinov3_checkpoint_path, moca_root))

    moca_model = Moca3DModel(moca_cfg).to(device).eval()
    cube_model = BBox3DMLP(hidden_dim=int(cube_cfg.get("hidden_dim", 256))).to(device).eval()

    if joint_checkpoint is not None:
        joint_obj = load_torch_checkpoint(joint_checkpoint, device)
        # Default order follows MoCA3D tools/evaluate_cube.py. --prefer_ema is
        # kept as an explicit ablation knob, not the strict-official default.
        moca_keys = (
            ("moca_model_ema", "moca_ema", "moca_model", "moca_state_dict")
            if args.prefer_ema
            else ("moca_model", "moca_state_dict", "moca_model_ema", "moca")
        )
        cube_keys = (
            ("cube_model_ema", "bbx3d_model_ema", "cube_model", "bbx3d_model", "bbox3d_model")
            if args.prefer_ema
            else ("cube_model", "bbx3d_model", "bbox3d_model", "cube_model_ema", "bbx3d_model_ema", "cube")
        )
        moca_state = extract_named_state_dict(joint_obj, moca_keys)
        cube_state = extract_named_state_dict(joint_obj, cube_keys)
        if moca_state is None or cube_state is None:
            split_moca_state, split_cube_state = split_prefixed_joint_state_dict(joint_obj)
            moca_state = moca_state or split_moca_state
            cube_state = cube_state or split_cube_state
        if moca_state is None or cube_state is None:
            raise KeyError(
                "Could not find moca_model/cube_model in joint checkpoint. "
                f"Top-level keys: {sorted(joint_obj.keys()) if isinstance(joint_obj, dict) else type(joint_obj)}"
            )
        moca_missing, moca_unexpected = moca_model.load_state_dict(moca_state, strict=bool(args.strict_moca))
        cube_missing, cube_unexpected = cube_model.load_state_dict(cube_state, strict=bool(args.strict_cube))
        print(f"Loaded joint checkpoint: {joint_checkpoint}")
    else:
        moca_obj = load_torch_checkpoint(moca_checkpoint, device)
        cube_obj = load_torch_checkpoint(cube_checkpoint, device)
        moca_state = strip_prefixes(extract_state_dict(moca_obj))
        cube_state = extract_named_state_dict(
            cube_obj,
            ("cube_model_ema", "cube_model", "bbx3d_model_ema", "bbx3d_model", "bbox3d_model"),
        )
        if cube_state is None:
            cube_state = strip_prefixes(extract_state_dict(cube_obj))
        moca_missing, moca_unexpected = moca_model.load_state_dict(moca_state, strict=bool(args.strict_moca))
        cube_missing, cube_unexpected = cube_model.load_state_dict(cube_state, strict=bool(args.strict_cube))
        print(f"Loaded MoCA checkpoint: {moca_checkpoint}")
        print(f"Loaded Cube checkpoint: {cube_checkpoint}")

    print(f"MoCA missing/unexpected: {len(moca_missing)}/{len(moca_unexpected)}")
    print(f"Cube missing/unexpected: {len(cube_missing)}/{len(cube_unexpected)}")
    return moca_model, cube_model


def load_json(path: str) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def load_gsam_info(root: str, dataset: str, split: str) -> dict:
    path = Path(root) / dataset / split / "info.pth"
    if not path.exists():
        raise FileNotFoundError(f"GroundedSAM info.pth not found: {path}")
    return torch.load(path, map_location="cpu")


def category_maps(data: dict) -> Tuple[Dict[str, int], Dict[int, str]]:
    name_to_id = {}
    id_to_name = {}
    for cat in data.get("categories", []):
        name = str(cat.get("name", ""))
        cid = int(cat.get("id", -1))
        if name:
            name_to_id[name] = cid
            id_to_name[cid] = name
    return name_to_id, id_to_name


def normalize_box_xyxy(box: Sequence[float], width: int, height: int) -> Optional[List[float]]:
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


def collect_boxes(
    image_id: int,
    record: dict,
    width: int,
    height: int,
    name_to_id: Dict[str, int],
    args: argparse.Namespace,
) -> List[dict]:
    if not record:
        return []
    boxes = np.asarray(record.get("boxes", []), dtype=np.float32)
    scores = np.asarray(record.get("conf", []), dtype=np.float32)
    phrases = list(record.get("phrases", []))
    if boxes.ndim != 2 or boxes.shape[1] < 4:
        return []
    out = []
    for idx, raw_box in enumerate(boxes):
        score = float(scores[idx]) if idx < len(scores) else 1.0
        if score < args.score_threshold:
            continue
        name = str(phrases[idx]) if idx < len(phrases) else ""
        if name not in name_to_id:
            continue
        box = normalize_box_xyxy(raw_box, width, height)
        if box is None:
            continue
        area = (box[2] - box[0]) * (box[3] - box[1])
        if area < args.min_2d_area:
            continue
        out.append(
            {
                "image_id": image_id,
                "source_index": idx,
                "bbox_xyxy": box,
                "score": score,
                "category_name": name,
                "category_id": int(name_to_id[name]),
            }
        )
    return out


def letterbox_image(image: Image.Image, image_size: int = 512) -> Tuple[Image.Image, float, int, int, int, int]:
    width, height = image.size
    longest = max(width, height)
    scale = image_size / float(longest)
    new_w, new_h = int(round(width * scale)), int(round(height * scale))
    pad_left = (image_size - new_w) // 2
    pad_top = (image_size - new_h) // 2
    canvas = Image.new("RGB", (image_size, image_size), (0, 0, 0))
    resized = image.resize((new_w, new_h), Image.BILINEAR)
    canvas.paste(resized, (pad_left, pad_top))
    return canvas, scale, pad_left, pad_top, new_w, new_h


def image_to_tensor(image: Image.Image) -> torch.Tensor:
    arr = torch.from_numpy(np.array(image, copy=True)).float() / 255.0
    arr = arr.permute(2, 0, 1)
    mean = arr.new_tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = arr.new_tensor(IMAGENET_STD).view(3, 1, 1)
    return (arr - mean) / std


def transform_boxes_for_moca(
    boxes: Sequence[Sequence[float]],
    scale: float,
    pad_left: int,
    pad_top: int,
    image_size: int = 512,
) -> torch.Tensor:
    arr = torch.as_tensor(boxes, dtype=torch.float32).clone()
    arr[:, [0, 2]] = arr[:, [0, 2]] * float(scale) + float(pad_left)
    arr[:, [1, 3]] = arr[:, [1, 3]] * float(scale) + float(pad_top)
    arr = arr / float(image_size)
    return arr.clamp(0.0, 1.0)


def valid_padding_mask(image_size: int, pad_left: int, pad_top: int, new_w: int, new_h: int) -> torch.Tensor:
    mask = torch.ones((image_size, image_size), dtype=torch.bool)
    mask[pad_top:pad_top + new_h, pad_left:pad_left + new_w] = False
    return mask


def project_points(points: np.ndarray, K: np.ndarray, width: int, height: int) -> List[float]:
    z = np.clip(points[:, 2], 1e-6, None)
    u = K[0, 0] * points[:, 0] / z + K[0, 2]
    v = K[1, 1] * points[:, 1] / z + K[1, 2]
    x1 = float(np.clip(np.min(u), 0, width))
    y1 = float(np.clip(np.min(v), 0, height))
    x2 = float(np.clip(np.max(u), 0, width))
    y2 = float(np.clip(np.max(v), 0, height))
    return [x1, y1, x2, y2]


def bbox_area_xyxy(box: Sequence[float]) -> float:
    if box is None or len(box) < 4:
        return 0.0
    x1, y1, x2, y2 = [float(v) for v in box[:4]]
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def bbox_iou_xyxy(box_a: Sequence[float], box_b: Sequence[float]) -> float:
    if box_a is None or box_b is None or len(box_a) < 4 or len(box_b) < 4:
        return 0.0
    ax1, ay1, ax2, ay2 = [float(v) for v in box_a[:4]]
    bx1, by1, bx2, by2 = [float(v) for v in box_b[:4]]
    inter_x1, inter_y1 = max(ax1, bx1), max(ay1, by1)
    inter_x2, inter_y2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, inter_x2 - inter_x1) * max(0.0, inter_y2 - inter_y1)
    area_a = bbox_area_xyxy(box_a)
    area_b = bbox_area_xyxy(box_b)
    union = area_a + area_b - inter
    return float(inter / union) if union > 0.0 else 0.0


def bbox_from_uv8(uv8: np.ndarray, width: int, height: int) -> List[float]:
    xs = np.asarray(uv8[:, 0], dtype=np.float32)
    ys = np.asarray(uv8[:, 1], dtype=np.float32)
    return [
        float(np.clip(np.nanmin(xs), 0, width)),
        float(np.clip(np.nanmin(ys), 0, height)),
        float(np.clip(np.nanmax(xs), 0, width)),
        float(np.clip(np.nanmax(ys), 0, height)),
    ]


def summarize_values(values: Sequence[float]) -> dict:
    vals = [float(v) for v in values if math.isfinite(float(v))]
    if not vals:
        return {"n": 0}
    vals = sorted(vals)
    n = len(vals)

    def pct(p: float) -> float:
        idx = min(n - 1, max(0, int(round((n - 1) * p))))
        return float(vals[idx])

    return {
        "n": n,
        "min": round(vals[0], 6),
        "p10": round(pct(0.10), 6),
        "p50": round(pct(0.50), 6),
        "p90": round(pct(0.90), 6),
        "p99": round(pct(0.99), 6),
        "max": round(vals[-1], 6),
        "mean": round(float(sum(vals) / n), 6),
    }


def box_to_center_dims_R(corners: np.ndarray) -> Tuple[List[float], List[float], List[List[float]]]:
    center = corners.mean(axis=0)
    x_axis = corners[1] - corners[0]
    y_axis = corners[3] - corners[0]
    z_axis = corners[4] - corners[0]
    dims = np.array([
        np.linalg.norm(x_axis),
        np.linalg.norm(y_axis),
        np.linalg.norm(z_axis),
    ], dtype=np.float32)
    axes = []
    for vec, fallback in (
        (x_axis, np.array([1.0, 0.0, 0.0])),
        (y_axis, np.array([0.0, 1.0, 0.0])),
        (z_axis, np.array([0.0, 0.0, 1.0])),
    ):
        norm = float(np.linalg.norm(vec))
        axes.append(vec / norm if norm > 1e-8 else fallback)
    R = np.stack(axes, axis=1).astype(np.float32)
    if np.linalg.det(R) < 0:
        R[:, 2] *= -1.0
    return center.astype(float).tolist(), dims.astype(float).tolist(), R.astype(float).tolist()


def is_valid_box(center: Sequence[float], dims: Sequence[float], args: argparse.Namespace) -> bool:
    arr_c = np.asarray(center, dtype=np.float32)
    arr_d = np.asarray(dims, dtype=np.float32)
    if not np.isfinite(arr_c).all() or not np.isfinite(arr_d).all():
        return False
    if arr_c[2] < args.min_depth or arr_c[2] > args.max_depth:
        return False
    if (arr_d < args.min_dimension).any() or (arr_d > args.max_dimension).any():
        return False
    return True


def autocast_context(device: torch.device, precision: str):
    if device.type != "cuda" or precision == "float32":
        return nullcontext()
    dtype = torch.float16 if precision == "float16" else torch.bfloat16
    return torch.autocast(device_type="cuda", dtype=dtype)


@torch.no_grad()
def run_moca_on_image(
    moca_model,
    cube_model,
    criterion,
    image_tensor: torch.Tensor,
    padding_mask: torch.Tensor,
    K: torch.Tensor,
    boxes_xyxy: List[List[float]],
    scale: float,
    pad_left: int,
    pad_top: int,
    real_h: int,
    device: torch.device,
    args: argparse.Namespace,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    pred_corners_all = []
    moca_uv_all = []
    moca_depth_all = []

    for start in range(0, len(boxes_xyxy), args.batch_size):
        cur_boxes = boxes_xyxy[start:start + args.batch_size]
        bsz = len(cur_boxes)
        images = image_tensor.unsqueeze(0).repeat(bsz, 1, 1, 1).to(device, non_blocking=True)
        masks = padding_mask.unsqueeze(0).repeat(bsz, 1, 1).to(device, non_blocking=True)
        box_tensor = transform_boxes_for_moca(cur_boxes, scale, pad_left, pad_top).to(device, non_blocking=True)
        K_batch = K.unsqueeze(0).repeat(bsz, 1, 1).to(device, non_blocking=True)

        with autocast_context(device, args.precision):
            moca_outputs = moca_model(
                images_dino=images,
                bbx2d_tight=box_tensor,
                mask=masks,
                return_decoder_feat=True,
            )
            cube_outputs = cube_model(moca_outputs, K_batch)
            pred_boxes = criterion.build_bbox3d(
                cube_outputs["centers"],
                cube_outputs["sizes"],
                cube_outputs["yaws"],
                cube_outputs["ray_x"],
                cube_outputs["ray_z"],
                ray_y=cube_outputs.get("ray_y", None),
            )

        # Convert MoCA image-plane corners/depth back to original image space for auditing.
        uv = moca_outputs["corner coords"].float().detach().clone()
        depths = moca_outputs["sampled depths"].float().detach().clone()
        uv[..., 0] = (uv[..., 0] - float(pad_left)) / float(scale)
        uv[..., 1] = (uv[..., 1] - float(pad_top)) / float(scale)
        virtual_scale = (float(real_h) / (K_batch[:, 1, 1].float() + 1e-6)).unsqueeze(1)
        depths = depths / virtual_scale

        pred_corners_all.append(pred_boxes.float().detach().cpu().numpy())
        moca_uv_all.append(uv.cpu().numpy())
        moca_depth_all.append(depths.cpu().numpy())

    return (
        np.concatenate(pred_corners_all, axis=0),
        np.concatenate(moca_uv_all, axis=0),
        np.concatenate(moca_depth_all, axis=0),
    )


def make_invalid_annotation(base: dict, ann_id: int, dataset_id: int) -> dict:
    return {
        "id": ann_id,
        "image_id": int(base["image_id"]),
        "dataset_id": int(dataset_id),
        "category_name": base["category_name"],
        "category_id": int(base["category_id"]),
        "valid3D": False,
        "bbox2D_tight": base["bbox_xyxy"],
        "bbox2D_trunc": base["bbox_xyxy"],
        "bbox2D_proj": base["bbox_xyxy"],
        "bbox3D_cam": [[0.0, 0.0, 0.0] for _ in range(8)],
        "center_cam": [-1.0, -1.0, -1.0],
        "dimensions": [-1.0, -1.0, -1.0],
        "R_cam": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        "behind_camera": False,
        "visibility": 1.0,
        "truncation": 0.0,
        "segmentation_pts": -1,
        "lidar_pts": -1,
        "depth_error": -1,
        "score": float(base["score"]),
        "source_2d": "original_groundedsam",
        "pseudo_label_method": "MoCA3D-Cube",
    }


def main() -> None:
    args = parse_args()
    dataset_auto, split_auto = infer_dataset_split(args.json_file)
    dataset = args.dataset or dataset_auto
    split = args.split or split_auto

    device = torch.device("cpu" if args.force_cpu or not torch.cuda.is_available() else f"cuda:{args.gpu}")
    print(f"device={device}")
    print(f"dataset={dataset} split={split}")

    data = load_json(args.json_file)
    name_to_id, _ = category_maps(data)
    gsam_info = load_gsam_info(args.original_pseudo_root, dataset, split)
    moca_model, cube_model = build_models(args, device)
    from losses.bbx3d_loss import BBox3DLoss

    criterion = BBox3DLoss().to(device).eval()

    images = list(data.get("images", []))
    if args.start_index > 0 or args.skip_images > 1:
        images = images[args.start_index:: max(1, args.skip_images)]
    if args.max_images is not None:
        images = images[: int(args.max_images)]

    dataset_id = int(data.get("info", {}).get("id", 0))
    annotations = []
    stats = {
        "images": len(images),
        "input_2d": 0,
        "valid3d": 0,
        "invalid3d": 0,
        "skipped_no_gsam": 0,
        "skipped_no_valid_2d": 0,
        "skipped_bad_3d": 0,
    }
    quality = {
        "proj_iou": [],
        "corner_bbox_iou": [],
        "proj_area_ratio": [],
        "max_dim": [],
        "min_dim": [],
        "center_z": [],
    }
    ann_counter = 1

    iterator = tqdm(images, desc="MoCA3D-Cube Omni3D")
    for img in iterator:
        image_id = int(img["id"])
        record = gsam_info.get(image_id, {})
        if not record:
            stats["skipped_no_gsam"] += 1
            continue

        width = int(img["width"])
        height = int(img["height"])
        entries = collect_boxes(image_id, record, width, height, name_to_id, args)
        if not entries:
            stats["skipped_no_valid_2d"] += 1
            continue
        stats["input_2d"] += len(entries)

        image_path = Path(args.image_root) / img["file_path"]
        image = Image.open(image_path).convert("RGB")
        letterboxed, scale, pad_left, pad_top, new_w, new_h = letterbox_image(image, 512)
        image_tensor = image_to_tensor(letterboxed)
        padding_mask = valid_padding_mask(512, pad_left, pad_top, new_w, new_h)
        K = torch.as_tensor(np.asarray(img["K"], dtype=np.float32).reshape(3, 3), dtype=torch.float32)
        K_np = K.numpy()

        boxes_xyxy = [entry["bbox_xyxy"] for entry in entries]
        try:
            pred_boxes, moca_uv, moca_depth = run_moca_on_image(
                moca_model=moca_model,
                cube_model=cube_model,
                criterion=criterion,
                image_tensor=image_tensor,
                padding_mask=padding_mask,
                K=K,
                boxes_xyxy=boxes_xyxy,
                scale=scale,
                pad_left=pad_left,
                pad_top=pad_top,
                real_h=height,
                device=device,
                args=args,
            )
        except RuntimeError as exc:
            print(f"Failed image_id={image_id}: {exc}")
            torch.cuda.empty_cache()
            continue

        for entry, corners, uv8, depth8 in zip(entries, pred_boxes, moca_uv, moca_depth):
            center, dims, R_cam = box_to_center_dims_R(corners)
            valid = is_valid_box(center, dims, args)
            ann_id = dataset_id * 10000000 + ann_counter
            ann_counter += 1
            if not valid:
                stats["skipped_bad_3d"] += 1
                if args.save_invalid:
                    annotations.append(make_invalid_annotation(entry, ann_id, dataset_id))
                    stats["invalid3d"] += 1
                continue

            bbox2d_proj = project_points(corners, K_np, width, height)
            corner_bbox = bbox_from_uv8(uv8, width, height)
            proj_iou = bbox_iou_xyxy(bbox2d_proj, entry["bbox_xyxy"])
            corner_bbox_iou = bbox_iou_xyxy(corner_bbox, entry["bbox_xyxy"])
            tight_area = bbox_area_xyxy(entry["bbox_xyxy"])
            proj_area_ratio = bbox_area_xyxy(bbox2d_proj) / max(tight_area, 1e-6)
            quality["proj_iou"].append(proj_iou)
            quality["corner_bbox_iou"].append(corner_bbox_iou)
            quality["proj_area_ratio"].append(proj_area_ratio)
            quality["max_dim"].append(float(np.max(dims)))
            quality["min_dim"].append(float(np.min(dims)))
            quality["center_z"].append(float(center[2]))
            projected_corners = [
                {"u": float(uv8[j, 0]), "v": float(uv8[j, 1])}
                for j in range(8)
            ]
            ann = {
                "id": ann_id,
                "image_id": image_id,
                "dataset_id": dataset_id,
                "category_name": entry["category_name"],
                "category_id": int(entry["category_id"]),
                "valid3D": True,
                "bbox2D_tight": [float(x) for x in entry["bbox_xyxy"]],
                "bbox2D_trunc": [float(x) for x in entry["bbox_xyxy"]],
                "bbox2D_proj": [float(x) for x in bbox2d_proj],
                "bbox3D_cam": corners.astype(float).tolist(),
                "center_cam": [float(x) for x in center],
                "dimensions": [float(x) for x in dims],
                "R_cam": R_cam,
                "behind_camera": bool(np.any(corners[:, 2] <= 0.0)),
                "visibility": 1.0,
                "truncation": 0.0,
                "segmentation_pts": -1,
                "lidar_pts": -1,
                "depth_error": -1,
                "score": float(entry["score"]),
                "source_2d": "original_groundedsam",
                "source_index": int(entry["source_index"]),
                "pseudo_label_method": "MoCA3D-Cube",
                "moca3d_projected_corners": projected_corners,
                "moca3d_corner_depth": [float(x) for x in depth8.tolist()],
                "moca3d_proj_iou": float(proj_iou),
                "moca3d_corner_bbox_iou": float(corner_bbox_iou),
                "moca3d_proj_area_ratio": float(proj_area_ratio),
            }
            annotations.append(ann)
            stats["valid3d"] += 1
        iterator.set_postfix(valid=stats["valid3d"], pred=stats["input_2d"])

    output = {
        "info": copy.deepcopy(data.get("info", {})),
        "images": copy.deepcopy(data.get("images", [])),
        "categories": copy.deepcopy(data.get("categories", [])),
        "annotations": annotations,
    }
    output.setdefault("info", {})
    output["info"]["pseudo_label_method"] = "GroundedSAM_MoCA3D_Cube"
    output["info"]["pseudo_label_source_json"] = os.path.abspath(args.json_file)
    output["info"]["pseudo_label_2d_cache"] = os.path.abspath(
        str(Path(args.original_pseudo_root) / dataset / split / "info.pth")
    )

    stats["quality"] = {
        key: summarize_values(vals)
        for key, vals in quality.items()
    }
    stats["quality"]["proj_iou_lt_005"] = int(sum(v < 0.05 for v in quality["proj_iou"]))
    stats["quality"]["proj_iou_lt_010"] = int(sum(v < 0.10 for v in quality["proj_iou"]))
    stats["quality"]["proj_iou_ge_035"] = int(sum(v >= 0.35 for v in quality["proj_iou"]))
    stats["quality"]["corner_bbox_iou_ge_050"] = int(sum(v >= 0.50 for v in quality["corner_bbox_iou"]))
    stats["quality"]["proj_area_ratio_gt_5"] = int(sum(v > 5.0 for v in quality["proj_area_ratio"]))
    stats["quality"]["proj_area_ratio_lt_02"] = int(sum(v < 0.2 for v in quality["proj_area_ratio"]))

    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f)
    print(f"Wrote Omni3D JSON: {out_path}")
    print(stats)

    if args.output_dir:
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        stats_path = out_dir / "moca3d_cube_stats.json"
        with open(stats_path, "w") as f:
            json.dump(stats, f, indent=2)
        print(f"Wrote stats: {stats_path}")


if __name__ == "__main__":
    main()
