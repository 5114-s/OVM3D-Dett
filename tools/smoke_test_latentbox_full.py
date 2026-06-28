#!/usr/bin/env python3
import argparse
import json
import os
import sys

import torch
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.structures import Boxes, Instances
from detectron2.utils.events import EventStorage

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cubercnn.config import get_cfg_defaults
from cubercnn.data.dataset_mapper import DatasetMapper3D
from cubercnn.data.datasets import (
    get_filter_settings_from_cfg,
    load_omni3d_json,
)
from cubercnn.modeling.backbone import build_dla_from_vision_fpn_backbone  # noqa: F401
from cubercnn.modeling.proposal_generator import RPNWithIgnore  # noqa: F401
from cubercnn.modeling.roi_heads import ROIHeads3D_Text  # noqa: F401
from cubercnn.modeling.meta_arch import build_model
from tools.train_net import (
    apply_ema_teacher_targets,
    build_ema_teacher,
    update_ema_teacher,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--real_json", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = get_cfg()
    cfg.set_new_allowed(True)
    get_cfg_defaults(cfg)
    cfg.merge_from_file("configs/Base_Omni3D_SUN.yaml")
    cfg.MODEL.DEVICE = args.device
    cfg.MODEL.WEIGHTS = ""
    cfg.MODEL.WEIGHTS_PRETRAIN = ""
    cfg.VIS_PERIOD = 0
    cfg.INPUT.USE_DEPTH = True
    cfg.INPUT.USE_PSEUDO_MASK = True
    cfg.INPUT.DEPTH_ROOT = "pseudo_label/SUNRGBD"
    cfg.INPUT.PSEUDO_MASK_ROOT = "pseudo_label/SUNRGBD"
    cfg.INPUT.USE_GROUND_MASK = True
    cfg.INPUT.GROUND_MASK_ROOT = "pseudo_label/SUNRGBD"
    cfg.MODEL.ROI_CUBE_HEAD.USE_FACTORIZED_PSEUDO_WEIGHT = True
    cfg.MODEL.ROI_CUBE_HEAD.USE_GEOMETRY_INTERPRETER = True
    if not args.full:
        cfg.MODEL.ROI_CUBE_HEAD.GEOMETRY_NUM_HYPOTHESES = 4
        cfg.MODEL.ROI_CUBE_HEAD.GEOMETRY_HIDDEN_DIM = 64
        cfg.MODEL.ROI_CUBE_HEAD.GEOMETRY_NUM_LAYERS = 1
        cfg.MODEL.ROI_CUBE_HEAD.GEOMETRY_NUM_HEADS = 4
        cfg.MODEL.ROI_CUBE_HEAD.DINOV2_IMAGE_SIZE = 112
        cfg.MODEL.ROI_CUBE_HEAD.DINOV2_OUTPUT_DIM = 32
    cfg.MODEL.ROI_CUBE_HEAD.USE_DIFFERENTIABLE_RENDERER = True
    cfg.MODEL.ROI_CUBE_HEAD.USE_DEPTH_CONSISTENCY_LOSS = True
    cfg.MODEL.ROI_CUBE_HEAD.DEPTH_CONSISTENCY_MODE = "front_surface"
    cfg.MODEL.EMA_TEACHER.ENABLED = True
    cfg.MODEL.EMA_TEACHER.WARMUP_ITERS = 0
    cfg.MODEL.EMA_TEACHER.MIN_SCORE = 0.01
    cfg.MODEL.EMA_TEACHER.MIN_GEOMETRY_SCORE = 0.0
    cfg.freeze()

    model = build_model(cfg, priors=None)
    model.train()
    if args.real_json:
        with open(args.real_json, "r") as handle:
            dataset = json.load(handle)
        category_ids = sorted(int(category["id"]) for category in dataset["categories"])
        MetadataCatalog.get(
            "omni3d_model"
        ).thing_dataset_id_to_contiguous_id = {
            category_id: index
            for index, category_id in enumerate(category_ids)
        }
        rows = load_omni3d_json(
            args.real_json,
            "datasets",
            "SUNRGBD_latentbox_smoke",
            get_filter_settings_from_cfg(cfg),
            filter_empty=True,
        )
        mapper = DatasetMapper3D(cfg, is_train=True)
        mapper.dataset_id_to_unknown_cats = {
            int(rows[0]["dataset_id"]): set(range(38))
        }
        batch = [mapper(rows[0])]
    else:
        height, width = 224, 336
        image = torch.randint(0, 255, (3, height, width), dtype=torch.uint8)
        depth = torch.full((1, height, width), 3.0, dtype=torch.float32)
        instances = Instances((height, width))
        instances.gt_classes = torch.tensor([0], dtype=torch.long)
        instances.gt_boxes = Boxes(torch.tensor([[80.0, 50.0, 250.0, 205.0]]))
        instances.gt_boxes3D = torch.tensor(
            [[165.0, 125.0, 3.0, 1.0, 1.0, 1.0, -0.03, 0.03, 3.0]],
            dtype=torch.float32,
        )
        instances.gt_poses = torch.eye(3).unsqueeze(0)
        instances.gt_unknown_category_mask = torch.zeros((1, 38), dtype=torch.bool)
        instances.gt_pseudo_weight = torch.tensor([0.8])
        for name, value in {
            "xy": 0.7,
            "z": 0.9,
            "dims": 0.8,
            "pose": 0.5,
            "joint": 0.65,
        }.items():
            instances.set(f"gt_pseudo_weight_{name}", torch.tensor([value]))
        mask = torch.zeros((1, 28, 28), dtype=torch.float32)
        mask[:, 3:25, 4:24] = 1.0
        instances.gt_render_masks = mask

        batch = [
            {
                "image": image,
                "depth": depth,
                "height": height,
                "width": width,
                "K": [[500.0, 0.0, 168.0], [0.0, 500.0, 112.0], [0.0, 0.0, 1.0]],
                "instances": instances,
            }
        ]
    text_embeddings = torch.randn(39, 768, device=args.device)
    ema_teacher = build_ema_teacher(model)
    ema_teacher.roi_heads.cube_head.bbox_3D_center_depth.bias.fill_(3.0)
    ema_teacher.roi_heads.cube_head.bbox_3D_uncertainty.bias.fill_(0.1)
    ema_updated, ema_blend = apply_ema_teacher_targets(
        cfg,
        ema_teacher,
        batch,
        text_embeddings,
        iteration=0,
    )
    with EventStorage(0):
        losses = model(batch, text_embeddings)
        total = sum(losses.values())
        total.backward()
    print({name: float(value.detach()) for name, value in losses.items()})
    print(
        {
            "total_loss": float(total.detach()),
            "finite": bool(torch.isfinite(total)),
            "interpreter_grad": float(
                model.roi_heads.geometry_interpreter.delta_head.weight.grad.abs().mean()
            ),
            "dino_projection_grad": float(
                model.roi_heads.dino_encoder.projection[0].weight.grad.abs().mean()
            ),
            "ema_updated": int(ema_updated),
            "ema_blend": float(ema_blend),
            "max_cuda_memory_gb": (
                float(torch.cuda.max_memory_allocated()) / (1024 ** 3)
                if torch.cuda.is_available()
                else 0.0
            ),
            "shape_memory_entries": int(
                model.roi_heads.geometry_interpreter.shape_memory.valid.sum()
            ),
        }
    )
    update_ema_teacher(ema_teacher, model, decay=0.99)


if __name__ == "__main__":
    main()
