#!/usr/bin/env python3
"""Extract ROI features for Boxer-Residual-LIFT.

This script is intentionally separate from prepare_lifthead_data.py so heavy
visual encoders such as DINOv2 are run once and cached by annotation id.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List, Mapping

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from tools.lifthead_common import (  # noqa: E402
    extract_roi_feature,
    load_image_rgb,
    roi_feature_names,
    xyxy_from_ann,
)


DEFAULT_DINOV2_CKPT = "/data/ZhaoX/ovmono3d/checkpoints/dinov2_vitb14_pretrain.pth"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cache ROI features for Residual-LIFT.")
    parser.add_argument("--json_file", required=True)
    parser.add_argument("--image_root", default="datasets")
    parser.add_argument("--output", required=True)
    parser.add_argument("--mode", choices=["color_grid", "dinov2_cls"], default="dinov2_cls")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--force_cpu", action="store_true")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--crop_size", type=int, default=224)
    parser.add_argument("--roi_grid_size", type=int, default=4)
    parser.add_argument("--roi_context_scale", type=float, default=1.15)
    parser.add_argument("--dinov2_checkpoint", default=DEFAULT_DINOV2_CKPT)
    parser.add_argument("--max_images", type=int, default=None)
    return parser.parse_args()


def load_json(path: str) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def valid_ann(ann: Mapping) -> bool:
    if not bool(ann.get("valid3D", True)):
        return False
    box = xyxy_from_ann(ann)
    return not np.any(box < 0)


def crop_pil(image_rgb: np.ndarray, ann: Mapping, image: Mapping, context_scale: float) -> Image.Image:
    h_img, w_img = image_rgb.shape[:2]
    width = max(float(image.get("width", w_img)), 1.0)
    height = max(float(image.get("height", h_img)), 1.0)
    box = xyxy_from_ann(ann)
    if np.any(box < 0):
        box = np.array([0.0, 0.0, width, height], dtype=np.float32)
    x1, y1, x2, y2 = [float(x) for x in box]
    bw = max(x2 - x1, 1.0)
    bh = max(y2 - y1, 1.0)
    cx = x1 + 0.5 * bw
    cy = y1 + 0.5 * bh
    scale = max(float(context_scale), 1.0)
    x1 = cx - 0.5 * bw * scale
    x2 = cx + 0.5 * bw * scale
    y1 = cy - 0.5 * bh * scale
    y2 = cy + 0.5 * bh * scale
    ix1 = int(np.floor(np.clip(x1, 0, w_img - 1)))
    iy1 = int(np.floor(np.clip(y1, 0, h_img - 1)))
    ix2 = int(np.ceil(np.clip(x2, ix1 + 1, w_img)))
    iy2 = int(np.ceil(np.clip(y2, iy1 + 1, h_img)))
    crop = image_rgb[iy1:iy2, ix1:ix2, :3]
    if crop.size == 0:
        crop = image_rgb
    return Image.fromarray(crop.astype(np.uint8))


def build_dinov2(device: torch.device, checkpoint: str) -> torch.nn.Module:
    from timm.models import create_model

    model = create_model(
        "vit_base_patch14_dinov2",
        pretrained=False,
        num_classes=0,
        dynamic_img_size=True,
    )
    state = torch.load(checkpoint, map_location="cpu")
    if isinstance(state, dict) and "model" in state and isinstance(state["model"], dict):
        state = state["model"]
    model_keys = set(model.state_dict().keys())
    filtered = {k: v for k, v in state.items() if k in model_keys}
    incompatible = model.load_state_dict(filtered, strict=False)
    unexpected = [k for k in incompatible.unexpected_keys if k != "mask_token"]
    if unexpected or incompatible.missing_keys:
        raise RuntimeError(
            "Failed to load DINOv2 checkpoint into timm vit_base_patch14_dinov2: "
            f"missing={incompatible.missing_keys}, unexpected={unexpected}"
        )
    model.to(device)
    model.eval()
    return model


def normalize_batch(crops: List[Image.Image], crop_size: int) -> torch.Tensor:
    arrays = []
    mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)
    for crop in crops:
        crop = crop.convert("RGB").resize((crop_size, crop_size), Image.BICUBIC)
        arr = torch.from_numpy(np.asarray(crop, dtype=np.float32) / 255.0).permute(2, 0, 1)
        arrays.append((arr - mean) / std)
    return torch.stack(arrays, dim=0)


@torch.no_grad()
def dinov2_features(model: torch.nn.Module, batch: torch.Tensor) -> torch.Tensor:
    out = model.forward_features(batch)
    if isinstance(out, dict):
        for key in ("x_norm_clstoken", "cls_token", "pooled", "features"):
            if key in out:
                feat = out[key]
                break
        else:
            first = next(iter(out.values()))
            feat = first[:, 0] if first.ndim == 3 else first
    elif out.ndim == 3:
        feat = out[:, 0]
    else:
        feat = out
    return F.normalize(feat.float(), dim=-1)


def main() -> None:
    args = parse_args()
    if torch.cuda.is_available() and not args.force_cpu:
        torch.cuda.set_device(args.gpu)
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    data = load_json(args.json_file)
    images = data.get("images", [])
    if args.max_images is not None:
        selected = {int(im["id"]) for im in images[: args.max_images]}
        images = [im for im in images if int(im["id"]) in selected]
    image_by_id = {int(im["id"]): im for im in images}
    anns_by_image: Dict[int, List[Mapping]] = {}
    for ann in data.get("annotations", []):
        image_id = int(ann.get("image_id", -1))
        if image_id not in image_by_id or not valid_ann(ann):
            continue
        anns_by_image.setdefault(image_id, []).append(ann)

    model = None
    feature_names = []
    if args.mode == "dinov2_cls":
        model = build_dinov2(device, args.dinov2_checkpoint)
        feature_names = [f"roi_dinov2_cls_{idx}" for idx in range(768)]
    else:
        feature_names = roi_feature_names("color_grid", args.roi_grid_size)

    ann_ids: List[int] = []
    features: List[torch.Tensor] = []
    pending_ids: List[int] = []
    pending_crops: List[Image.Image] = []

    def flush_dino() -> None:
        if not pending_crops:
            return
        batch = normalize_batch(pending_crops, args.crop_size).to(device)
        feats = dinov2_features(model, batch).cpu()
        features.extend([feat for feat in feats])
        ann_ids.extend(pending_ids)
        pending_crops.clear()
        pending_ids.clear()

    for image_id, image in tqdm(image_by_id.items(), desc="Extracting ROI features"):
        image_rgb = load_image_rgb(args.image_root, image)
        if image_rgb is None:
            continue
        for ann in anns_by_image.get(image_id, []):
            if args.mode == "color_grid":
                feat = extract_roi_feature(
                    image_rgb,
                    ann,
                    image,
                    mode="color_grid",
                    grid_size=args.roi_grid_size,
                    context_scale=args.roi_context_scale,
                )
                ann_ids.append(int(ann["id"]))
                features.append(torch.tensor(feat, dtype=torch.float32))
            else:
                pending_ids.append(int(ann["id"]))
                pending_crops.append(crop_pil(image_rgb, ann, image, args.roi_context_scale))
                if len(pending_crops) >= args.batch_size:
                    flush_dino()
    flush_dino()

    if not features:
        raise RuntimeError("No ROI features were extracted.")
    out = {
        "ann_ids": torch.tensor(ann_ids, dtype=torch.long),
        "features": torch.stack(features, dim=0).float(),
        "feature_names": feature_names,
        "config": {
            "mode": args.mode,
            "json_file": os.path.abspath(args.json_file),
            "image_root": os.path.abspath(args.image_root),
            "crop_size": int(args.crop_size),
            "roi_grid_size": int(args.roi_grid_size),
            "roi_context_scale": float(args.roi_context_scale),
            "dinov2_checkpoint": os.path.abspath(args.dinov2_checkpoint)
            if args.mode == "dinov2_cls"
            else None,
        },
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    torch.save(out, args.output)
    print(f"Wrote ROI feature cache: {args.output}")
    print({"annotations": len(ann_ids), "feature_dim": int(out["features"].shape[1]), "mode": args.mode})


if __name__ == "__main__":
    main()
