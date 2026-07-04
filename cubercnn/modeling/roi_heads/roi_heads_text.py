# Copyright (c) Meta Platforms, Inc. and affiliates
import logging
import numpy as np
import cv2
from typing import Dict, List, Tuple
import torch
from torch import nn
import torch.nn.functional as F
from pytorch3d.transforms.so3 import (
    so3_relative_angle
)
from detectron2.config import configurable
from detectron2.structures import Instances, Boxes, pairwise_iou, pairwise_ioa
from detectron2.layers import ShapeSpec, nonzero_tuple
from detectron2.modeling.proposal_generator.proposal_utils import add_ground_truth_to_proposals
from detectron2.utils.events import get_event_storage
from detectron2.modeling.roi_heads import (
    StandardROIHeads, ROI_HEADS_REGISTRY, select_foreground_proposals,
)
from detectron2.modeling.poolers import ROIPooler
from cubercnn.modeling.roi_heads.cube_head import build_cube_head
from cubercnn.modeling.roi_heads.geometry_interpreter import (
    FrozenDINOv2MultiScale,
    MultiHypothesisGeometryInterpreter,
    SoftCuboidRenderer,
    build_ray_map,
    roi_align_map,
)
from cubercnn.modeling.proposal_generator.rpn import subsample_labels
from cubercnn.modeling.roi_heads.fast_rcnn_text import FastRCNNOutputs_text
from cubercnn import util

logger = logging.getLogger(__name__)

E_CONSTANT = 2.71828183
SQRT_2_CONSTANT = 1.41421356

def depth_to_point_map(depth, Ks_scaled):
    if depth is None:
        return None

    if depth.dim() == 3:
        depth = depth.unsqueeze(1)
    if depth.size(1) != 1:
        depth = depth[:, :1]

    depth = depth.float()
    b, _, h, w = depth.shape
    device = depth.device
    dtype = depth.dtype
    ys, xs = torch.meshgrid(
        torch.arange(h, device=device, dtype=dtype),
        torch.arange(w, device=device, dtype=dtype),
        indexing="ij",
    )
    z = depth[:, 0]
    fx = Ks_scaled[:, 0, 0].view(b, 1, 1).clamp(min=1e-6)
    fy = Ks_scaled[:, 1, 1].view(b, 1, 1).clamp(min=1e-6)
    cx = Ks_scaled[:, 0, 2].view(b, 1, 1)
    cy = Ks_scaled[:, 1, 2].view(b, 1, 1)

    x = (xs.unsqueeze(0) - cx) * z / fx
    y = (ys.unsqueeze(0) - cy) * z / fy
    point_map = torch.stack((x, y, z), dim=1)
    valid = torch.isfinite(point_map).all(dim=1, keepdim=True) & (z.unsqueeze(1) > 0)
    return torch.where(valid, point_map, torch.zeros_like(point_map))


@torch.no_grad()
def estimate_ground_planes(point_map, ground_mask):
    if point_map is None or ground_mask is None:
        return None
    if ground_mask.dim() == 3:
        ground_mask = ground_mask.unsqueeze(1)
    planes = []
    for points_i, mask_i in zip(point_map, ground_mask):
        points = points_i.permute(1, 2, 0).reshape(-1, 3)
        mask = mask_i[0].reshape(-1) > 0.5
        valid = (
            mask
            & torch.isfinite(points).all(dim=1)
            & (points[:, 2] > 0.05)
        )
        points = points[valid]
        if points.shape[0] < 32:
            planes.append(points_i.new_tensor([0.0, 1.0, 0.0, 0.0, 0.0]))
            continue
        if points.shape[0] > 5000:
            indices = torch.linspace(
                0,
                points.shape[0] - 1,
                5000,
                device=points.device,
            ).long()
            points = points[indices]
        center = points.median(dim=0).values
        centered = points - center
        covariance = centered.T @ centered / max(points.shape[0] - 1, 1)
        _, eigenvectors = torch.linalg.eigh(covariance)
        normal = F.normalize(eigenvectors[:, 0], dim=0, eps=1e-6)
        if normal[1] < 0:
            normal = -normal
        offset = -torch.dot(normal, center)
        planes.append(torch.cat((normal, offset[None], points_i.new_tensor([1.0]))))
    return torch.stack(planes)


class DepthAwareROIPooler(nn.Module):
    def __init__(
        self,
        output_size,
        scales,
        sampling_ratio,
        pooler_type,
        out_channels,
        adapter_scale=1.0,
    ):
        super().__init__()
        self.feature_pooler = ROIPooler(
            output_size=output_size,
            scales=scales,
            sampling_ratio=sampling_ratio,
            pooler_type=pooler_type,
        )
        self.point_pooler = ROIPooler(
            output_size=output_size,
            scales=(1.0,),
            sampling_ratio=sampling_ratio,
            pooler_type=pooler_type,
        )
        self.point_adapter = nn.Conv2d(3, out_channels, kernel_size=1)
        nn.init.constant_(self.point_adapter.weight, 0.0)
        nn.init.constant_(self.point_adapter.bias, 0.0)
        self.adapter_scale = float(adapter_scale)

    def forward(self, features, boxes, point_map=None):
        roi_features = self.feature_pooler(features, boxes)
        if point_map is None or sum(len(boxes_i) for boxes_i in boxes) == 0:
            return roi_features
        point_features = self.point_pooler([point_map], boxes)
        return roi_features + self.adapter_scale * self.point_adapter(point_features)


class RegionSegmentationHead(nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_dim=128,
        mask_size=28,
        feature_scale=1.0,
        detach_mask_feature=True,
    ):
        super().__init__()
        self.mask_size = int(mask_size)
        self.feature_scale = float(feature_scale)
        self.detach_mask_feature = bool(detach_mask_feature)
        self.mask_head = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.mask_logits = nn.Conv2d(hidden_dim, 1, kernel_size=1)
        self.feature_adapter = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        nn.init.constant_(self.feature_adapter.weight, 0.0)
        nn.init.constant_(self.feature_adapter.bias, 0.0)

    def forward(self, roi_features):
        region_features = self.mask_head(roi_features)
        mask_logits_low = self.mask_logits(region_features)
        if mask_logits_low.shape[-1] != self.mask_size:
            mask_logits = F.interpolate(
                mask_logits_low,
                size=(self.mask_size, self.mask_size),
                mode="bilinear",
                align_corners=False,
            )
        else:
            mask_logits = mask_logits_low

        mask_prob_low = torch.sigmoid(mask_logits_low)
        if self.detach_mask_feature:
            mask_prob_low = mask_prob_low.detach()
        guided_features = roi_features * mask_prob_low
        enhanced_features = (
            roi_features
            + self.feature_scale * self.feature_adapter(guided_features)
        )
        return enhanced_features, mask_logits, torch.sigmoid(mask_logits)


class ZeroEmbeddingGeometryAdapter(nn.Module):
    """
    DetAny3D-style zero-init geometry adapter.

    It injects metric point-map / camera ray / mask cues into the pooled 3D RoI
    feature. The output projection is zero-initialized, so the first forward pass
    is exactly equivalent to the original CubeHead input.
    """

    def __init__(
        self,
        in_channels,
        hidden_dim=128,
        adapter_scale=1.0,
        gate_init_bias=-2.0,
        use_depth=True,
        use_ray=True,
        use_mask=True,
        detach_geometry=True,
    ):
        super().__init__()
        self.adapter_scale = float(adapter_scale)
        self.use_depth = bool(use_depth)
        self.use_ray = bool(use_ray)
        self.use_mask = bool(use_mask)
        self.detach_geometry = bool(detach_geometry)
        geometry_channels = 0
        if self.use_depth:
            geometry_channels += 3
        if self.use_ray:
            geometry_channels += 3
        if self.use_mask:
            geometry_channels += 1
        geometry_channels = max(1, geometry_channels)
        hidden_dim = max(8, int(hidden_dim))

        num_groups = min(8, hidden_dim)
        while hidden_dim % num_groups != 0 and num_groups > 1:
            num_groups -= 1

        self.geometry_encoder = nn.Sequential(
            nn.Conv2d(geometry_channels, hidden_dim, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups, hidden_dim),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups, hidden_dim),
            nn.GELU(),
        )
        self.delta_projection = nn.Conv2d(hidden_dim, in_channels, kernel_size=1)
        self.gate_projection = nn.Conv2d(hidden_dim, in_channels, kernel_size=1)
        nn.init.constant_(self.delta_projection.weight, 0.0)
        nn.init.constant_(self.delta_projection.bias, 0.0)
        nn.init.constant_(self.gate_projection.weight, 0.0)
        nn.init.constant_(self.gate_projection.bias, float(gate_init_bias))

    @staticmethod
    def _normalize_point_roi(point_roi: torch.Tensor) -> torch.Tensor:
        z = point_roi[:, 2:3]
        valid = torch.isfinite(point_roi).all(dim=1, keepdim=True) & (z > 0.05)
        z_safe = z.clamp(min=0.05)
        x_over_z = (point_roi[:, 0:1] / z_safe).clamp(-3.0, 3.0)
        y_over_z = (point_roi[:, 1:2] / z_safe).clamp(-3.0, 3.0)
        log_z = torch.log(z_safe).clamp(-4.0, 4.0)
        normalized = torch.cat((x_over_z, y_over_z, log_z), dim=1)
        return torch.where(valid, normalized, torch.zeros_like(normalized))

    def forward(self, roi_features, point_roi=None, ray_roi=None, mask_roi=None):
        if roi_features.numel() == 0:
            return roi_features, None
        geometry_parts = []
        if self.use_depth and point_roi is not None:
            geometry_parts.append(self._normalize_point_roi(point_roi.float()))
        if self.use_ray and ray_roi is not None:
            geometry_parts.append(ray_roi.float().clamp(-1.0, 1.0))
        if self.use_mask and mask_roi is not None:
            if mask_roi.dim() == 3:
                mask_roi = mask_roi[:, None]
            if mask_roi.shape[-2:] != roi_features.shape[-2:]:
                mask_roi = F.interpolate(
                    mask_roi.float(),
                    size=roi_features.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
            geometry_parts.append(mask_roi.float().clamp(0.0, 1.0))
        if not geometry_parts:
            return roi_features, None
        geometry = torch.cat(geometry_parts, dim=1)
        if self.detach_geometry:
            geometry = geometry.detach()
        encoded = self.geometry_encoder(geometry)
        gate = torch.sigmoid(self.gate_projection(encoded))
        delta = self.delta_projection(encoded)
        enhanced = roi_features + self.adapter_scale * gate * delta
        return enhanced, gate


def build_roi_heads(cfg, input_shape, priors=None):
    """
    Build ROIHeads defined by `cfg.MODEL.ROI_HEADS.NAME`.
    """
    name = cfg.MODEL.ROI_HEADS.NAME
    return ROI_HEADS_REGISTRY.get(name)(cfg, input_shape, priors=priors)


@ROI_HEADS_REGISTRY.register()
class ROIHeads3D_Text(StandardROIHeads):

    @configurable
    def __init__(
        self,
        *,
        ignore_thresh: float,
        cube_head: nn.Module,
        cube_pooler: nn.Module,
        geometry_interpreter: nn.Module = None,
        dino_encoder: nn.Module = None,
        soft_renderer: nn.Module = None,
        region_segmentation_head: nn.Module = None,
        zem_adapter: nn.Module = None,
        loss_w_3d: float,
        loss_w_xy: float,
        loss_w_z: float,
        loss_w_dims: float,
        loss_w_pose: float,
        loss_w_joint: float,
        use_confidence: float,
        inverse_z_weight: bool,
        z_type: str,
        pose_type: str,
        cluster_bins: int,
        priors = None,
        dims_priors_enabled = None,
        dims_priors_func = None,
        disentangled_loss=None,
        virtual_depth=None,
        virtual_focal=None,
        test_scale=None,
        allocentric_pose=None,
        chamfer_pose=None,
        scale_roi_boxes=None,
        use_depth_roi=None,
        use_pseudo_weight=None,
        use_factorized_pseudo_weight=None,
        use_depth_consistency_loss=None,
        loss_w_depth_consistency=None,
        depth_consistency_min_pixels=None,
        depth_consistency_center_crop=None,
        depth_consistency_mode=None,
        depth_consistency_percentile=None,
        use_projected_corner_depth_aux=None,
        loss_w_projected_corner_2d=None,
        loss_w_projected_corner_depth=None,
        projected_corner_max_loss=None,
        use_region_segmentation_head=None,
        loss_w_region_segmentation=None,
        rsh_use_depth_guidance=None,
        rsh_depth_mask_threshold=None,
        rsh_use_pseudo_weight=None,
        use_zem_adapter=None,
        use_geometry_interpreter=None,
        loss_w_multi_hypothesis=None,
        use_differentiable_renderer=None,
        loss_w_render_silhouette=None,
        loss_w_render_depth=None,
        render_size=None,
        geometry_selection_temperature=None,
        geometry_oracle_weight=None,
        geometry_projection_weight=None,
        geometry_silhouette_weight=None,
        geometry_depth_weight=None,
        geometry_point_weight=None,
        geometry_ground_weight=None,
        loss_w_geometry_closure=None,
        geometry_apply_to_prediction=None,
        geometry_apply_in_inference=None,
        geometry_apply_warmup_iters=None,
        geometry_min_dimension=None,
        geometry_max_dimension=None,
        shape_memory_min_confidence=None,
        shape_memory_max_updates=None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.scale_roi_boxes = scale_roi_boxes
        self.use_depth_roi = bool(use_depth_roi)
        self.use_pseudo_weight = bool(use_pseudo_weight)
        self.use_factorized_pseudo_weight = bool(use_factorized_pseudo_weight)
        self.use_depth_consistency_loss = bool(use_depth_consistency_loss)
        self.loss_w_depth_consistency = float(loss_w_depth_consistency or 0.0)
        self.depth_consistency_min_pixels = int(depth_consistency_min_pixels or 16)
        self.depth_consistency_center_crop = float(depth_consistency_center_crop or 1.0)
        self.depth_consistency_mode = str(depth_consistency_mode or "center")
        self.depth_consistency_percentile = float(
            0.35 if depth_consistency_percentile is None else depth_consistency_percentile
        )
        self.use_projected_corner_depth_aux = bool(use_projected_corner_depth_aux)
        self.loss_w_projected_corner_2d = float(loss_w_projected_corner_2d or 0.0)
        self.loss_w_projected_corner_depth = float(loss_w_projected_corner_depth or 0.0)
        self.projected_corner_max_loss = float(projected_corner_max_loss or 1.0)
        self.region_segmentation_head = region_segmentation_head
        self.use_region_segmentation_head = bool(use_region_segmentation_head)
        self.loss_w_region_segmentation = float(loss_w_region_segmentation or 0.0)
        self.rsh_use_depth_guidance = bool(rsh_use_depth_guidance)
        self.rsh_depth_mask_threshold = float(rsh_depth_mask_threshold or 0.30)
        self.rsh_use_pseudo_weight = bool(rsh_use_pseudo_weight)
        self.use_zem_adapter = bool(use_zem_adapter)
        self.zem_adapter = zem_adapter
        self.use_geometry_interpreter = bool(use_geometry_interpreter)
        self.geometry_interpreter = geometry_interpreter
        self.dino_encoder = dino_encoder
        self.loss_w_multi_hypothesis = float(loss_w_multi_hypothesis or 0.0)
        self.use_differentiable_renderer = bool(use_differentiable_renderer)
        self.soft_renderer = soft_renderer
        self.loss_w_render_silhouette = float(loss_w_render_silhouette or 0.0)
        self.loss_w_render_depth = float(loss_w_render_depth or 0.0)
        self.render_size = int(render_size or 28)
        self.geometry_selection_temperature = float(
            geometry_selection_temperature or 0.20
        )
        self.geometry_oracle_weight = float(geometry_oracle_weight or 0.0)
        self.geometry_projection_weight = float(geometry_projection_weight or 0.0)
        self.geometry_silhouette_weight = float(geometry_silhouette_weight or 0.0)
        self.geometry_depth_weight = float(geometry_depth_weight or 0.0)
        self.geometry_point_weight = float(geometry_point_weight or 0.0)
        self.geometry_ground_weight = float(geometry_ground_weight or 0.0)
        self.loss_w_geometry_closure = float(loss_w_geometry_closure or 0.0)
        self.geometry_apply_to_prediction = bool(geometry_apply_to_prediction)
        self.geometry_apply_in_inference = bool(geometry_apply_in_inference)
        self.geometry_apply_warmup_iters = int(geometry_apply_warmup_iters or 0)
        self.geometry_min_dimension = float(geometry_min_dimension or 0.03)
        self.geometry_max_dimension = float(geometry_max_dimension or 8.0)
        self.shape_memory_min_confidence = float(
            shape_memory_min_confidence or 0.65
        )
        self.shape_memory_max_updates = int(shape_memory_max_updates or 32)

        # rotation settings
        self.allocentric_pose = allocentric_pose
        self.chamfer_pose = chamfer_pose

        # virtual settings
        self.virtual_depth = virtual_depth
        self.virtual_focal = virtual_focal

        # loss weights, <=0 is off
        self.loss_w_3d = loss_w_3d
        self.loss_w_xy = loss_w_xy
        self.loss_w_z = loss_w_z
        self.loss_w_dims = loss_w_dims
        self.loss_w_pose = loss_w_pose
        self.loss_w_joint = loss_w_joint

        # loss modes
        self.disentangled_loss = disentangled_loss
        self.inverse_z_weight = inverse_z_weight

        # misc
        self.test_scale = test_scale
        self.ignore_thresh = ignore_thresh
        
        # related to network outputs
        self.z_type = z_type
        self.pose_type = pose_type
        self.use_confidence = use_confidence

        # related to priors
        self.cluster_bins = cluster_bins
        self.dims_priors_enabled = dims_priors_enabled
        self.dims_priors_func = dims_priors_func

        # if there is no 3D loss, then we don't need any heads. 
        if loss_w_3d > 0:
            
            self.cube_head = cube_head
            self.cube_pooler = cube_pooler
            
            # the dimensions could rely on pre-computed priors
            if self.dims_priors_enabled and priors is not None:
                self.priors_dims_per_cat = nn.Parameter(torch.FloatTensor(priors['priors_dims_per_cat']).unsqueeze(0))
            else:
                self.priors_dims_per_cat = nn.Parameter(torch.ones(1, self.num_classes, 2, 3))

            # Optionally, refactor priors and store them in the network params
            if self.cluster_bins > 1 and priors is not None:

                # the depth could have been clustered based on 2D scales                
                priors_z_scales = torch.stack([torch.FloatTensor(prior[1]) for prior in priors['priors_bins']])
                self.priors_z_scales = nn.Parameter(priors_z_scales)

            else:
                self.priors_z_scales = nn.Parameter(torch.ones(self.num_classes, self.cluster_bins))

            # the depth can be based on priors
            if self.z_type == 'clusters':
                
                assert self.cluster_bins > 1, 'To use z_type of priors, there must be more than 1 cluster bin'
                
                if priors is None:
                    self.priors_z_stats = nn.Parameter(torch.ones(self.num_classes, self.cluster_bins, 2).float())
                else:

                    # stats
                    priors_z_stats = torch.cat([torch.FloatTensor(prior[2]).unsqueeze(0) for prior in priors['priors_bins']])
                    self.priors_z_stats = nn.Parameter(priors_z_stats)

    @classmethod
    def from_config(cls, cfg, input_shape: Dict[str, ShapeSpec], priors=None):
        
        ret = super().from_config(cfg, input_shape)
        
        # pass along priors
        ret["box_predictor"] = FastRCNNOutputs_text(cfg, ret['box_head'].output_shape)
        ret.update(cls._init_cube_head(cfg, input_shape))
        ret["priors"] = priors

        return ret

    @classmethod
    def _init_cube_head(self, cfg, input_shape: Dict[str, ShapeSpec]):
        
        in_features = cfg.MODEL.ROI_HEADS.IN_FEATURES
        pooler_scales = tuple(1.0 / input_shape[k].stride for k in in_features)
        pooler_resolution = cfg.MODEL.ROI_CUBE_HEAD.POOLER_RESOLUTION 
        pooler_sampling_ratio = cfg.MODEL.ROI_CUBE_HEAD.POOLER_SAMPLING_RATIO
        pooler_type = cfg.MODEL.ROI_CUBE_HEAD.POOLER_TYPE

        in_channels = [input_shape[f].channels for f in in_features][0]
        use_depth_roi = bool(cfg.INPUT.USE_DEPTH and cfg.MODEL.ROI_CUBE_HEAD.USE_DEPTH_ROI)
        if use_depth_roi:
            cube_pooler = DepthAwareROIPooler(
                output_size=pooler_resolution,
                scales=pooler_scales,
                sampling_ratio=pooler_sampling_ratio,
                pooler_type=pooler_type,
                out_channels=in_channels,
                adapter_scale=cfg.MODEL.ROI_CUBE_HEAD.DEPTH_ADAPTER_SCALE,
            )
        else:
            cube_pooler = ROIPooler(
                output_size=pooler_resolution,
                scales=pooler_scales,
                sampling_ratio=pooler_sampling_ratio,
                pooler_type=pooler_type,
            )

        shape = ShapeSpec(
            channels=in_channels, width=pooler_resolution, height=pooler_resolution
        )

        cube_head = build_cube_head(cfg, shape)
        use_geometry_interpreter = bool(
            cfg.MODEL.ROI_CUBE_HEAD.USE_GEOMETRY_INTERPRETER
        )
        geometry_interpreter = None
        dino_encoder = None
        soft_renderer = None
        region_segmentation_head = None
        zem_adapter = None
        if cfg.MODEL.ROI_CUBE_HEAD.USE_REGION_SEGMENTATION_HEAD:
            region_segmentation_head = RegionSegmentationHead(
                in_channels=in_channels,
                hidden_dim=cfg.MODEL.ROI_CUBE_HEAD.RSH_HIDDEN_DIM,
                mask_size=cfg.MODEL.ROI_CUBE_HEAD.RSH_MASK_SIZE,
                feature_scale=cfg.MODEL.ROI_CUBE_HEAD.RSH_FEATURE_SCALE,
                detach_mask_feature=cfg.MODEL.ROI_CUBE_HEAD.RSH_DETACH_MASK_FEATURE,
            )
        if cfg.MODEL.ROI_CUBE_HEAD.USE_ZEM_ADAPTER:
            zem_adapter = ZeroEmbeddingGeometryAdapter(
                in_channels=in_channels,
                hidden_dim=cfg.MODEL.ROI_CUBE_HEAD.ZEM_HIDDEN_DIM,
                adapter_scale=cfg.MODEL.ROI_CUBE_HEAD.ZEM_ADAPTER_SCALE,
                gate_init_bias=cfg.MODEL.ROI_CUBE_HEAD.ZEM_GATE_INIT_BIAS,
                use_depth=cfg.MODEL.ROI_CUBE_HEAD.ZEM_USE_DEPTH,
                use_ray=cfg.MODEL.ROI_CUBE_HEAD.ZEM_USE_RAY,
                use_mask=cfg.MODEL.ROI_CUBE_HEAD.ZEM_USE_MASK,
                detach_geometry=cfg.MODEL.ROI_CUBE_HEAD.ZEM_DETACH_GEOMETRY,
            )
        if use_geometry_interpreter:
            dino_encoder = FrozenDINOv2MultiScale(
                checkpoint=cfg.MODEL.ROI_CUBE_HEAD.DINOV2_CHECKPOINT,
                image_size=cfg.MODEL.ROI_CUBE_HEAD.DINOV2_IMAGE_SIZE,
                output_dim=cfg.MODEL.ROI_CUBE_HEAD.DINOV2_OUTPUT_DIM,
                layers=cfg.MODEL.ROI_CUBE_HEAD.DINOV2_LAYERS,
                pixel_mean=cfg.MODEL.PIXEL_MEAN,
                pixel_std=cfg.MODEL.PIXEL_STD,
                input_format=cfg.INPUT.FORMAT,
                chunk_size=cfg.MODEL.ROI_CUBE_HEAD.DINOV2_CHUNK_SIZE,
            )
            geometry_interpreter = MultiHypothesisGeometryInterpreter(
                visual_channels=in_channels,
                dino_channels=cfg.MODEL.ROI_CUBE_HEAD.DINOV2_OUTPUT_DIM,
                hidden_dim=cfg.MODEL.ROI_CUBE_HEAD.GEOMETRY_HIDDEN_DIM,
                num_hypotheses=cfg.MODEL.ROI_CUBE_HEAD.GEOMETRY_NUM_HYPOTHESES,
                num_layers=cfg.MODEL.ROI_CUBE_HEAD.GEOMETRY_NUM_LAYERS,
                num_heads=cfg.MODEL.ROI_CUBE_HEAD.GEOMETRY_NUM_HEADS,
                residual_scale=cfg.MODEL.ROI_CUBE_HEAD.GEOMETRY_RESIDUAL_SCALE,
                shape_memory_capacity=cfg.MODEL.ROI_CUBE_HEAD.SHAPE_MEMORY_CAPACITY,
                shape_memory_topk=cfg.MODEL.ROI_CUBE_HEAD.SHAPE_MEMORY_TOPK,
                shape_memory_momentum=cfg.MODEL.ROI_CUBE_HEAD.SHAPE_MEMORY_MOMENTUM,
                shape_prototype_blend=cfg.MODEL.ROI_CUBE_HEAD.SHAPE_PROTOTYPE_BLEND,
            )
        if cfg.MODEL.ROI_CUBE_HEAD.USE_DIFFERENTIABLE_RENDERER:
            soft_renderer = SoftCuboidRenderer(
                render_size=cfg.MODEL.ROI_CUBE_HEAD.RENDER_SIZE,
                edge_softness=cfg.MODEL.ROI_CUBE_HEAD.RENDER_EDGE_SOFTNESS,
                depth_temperature=cfg.MODEL.ROI_CUBE_HEAD.RENDER_DEPTH_TEMPERATURE,
            )

        return {
            'cube_head': cube_head,
            'cube_pooler': cube_pooler,
            'geometry_interpreter': geometry_interpreter,
            'dino_encoder': dino_encoder,
            'soft_renderer': soft_renderer,
            'region_segmentation_head': region_segmentation_head,
            'zem_adapter': zem_adapter,
            'use_confidence': cfg.MODEL.ROI_CUBE_HEAD.USE_CONFIDENCE,
            'inverse_z_weight': cfg.MODEL.ROI_CUBE_HEAD.INVERSE_Z_WEIGHT,
            'loss_w_3d': cfg.MODEL.ROI_CUBE_HEAD.LOSS_W_3D,
            'loss_w_xy': cfg.MODEL.ROI_CUBE_HEAD.LOSS_W_XY,
            'loss_w_z': cfg.MODEL.ROI_CUBE_HEAD.LOSS_W_Z,
            'loss_w_dims': cfg.MODEL.ROI_CUBE_HEAD.LOSS_W_DIMS,
            'loss_w_pose': cfg.MODEL.ROI_CUBE_HEAD.LOSS_W_POSE,
            'loss_w_joint': cfg.MODEL.ROI_CUBE_HEAD.LOSS_W_JOINT,
            'z_type': cfg.MODEL.ROI_CUBE_HEAD.Z_TYPE,
            'pose_type': cfg.MODEL.ROI_CUBE_HEAD.POSE_TYPE,
            'dims_priors_enabled': cfg.MODEL.ROI_CUBE_HEAD.DIMS_PRIORS_ENABLED,
            'dims_priors_func': cfg.MODEL.ROI_CUBE_HEAD.DIMS_PRIORS_FUNC,
            'disentangled_loss': cfg.MODEL.ROI_CUBE_HEAD.DISENTANGLED_LOSS,
            'virtual_depth': cfg.MODEL.ROI_CUBE_HEAD.VIRTUAL_DEPTH,
            'virtual_focal': cfg.MODEL.ROI_CUBE_HEAD.VIRTUAL_FOCAL,
            'test_scale': cfg.INPUT.MIN_SIZE_TEST,
            'chamfer_pose': cfg.MODEL.ROI_CUBE_HEAD.CHAMFER_POSE,
            'allocentric_pose': cfg.MODEL.ROI_CUBE_HEAD.ALLOCENTRIC_POSE,
            'cluster_bins': cfg.MODEL.ROI_CUBE_HEAD.CLUSTER_BINS,
            'ignore_thresh': cfg.MODEL.RPN.IGNORE_THRESHOLD,
            'scale_roi_boxes': cfg.MODEL.ROI_CUBE_HEAD.SCALE_ROI_BOXES,
            'use_depth_roi': use_depth_roi,
            'use_pseudo_weight': cfg.MODEL.ROI_CUBE_HEAD.USE_PSEUDO_WEIGHT,
            'use_factorized_pseudo_weight': cfg.MODEL.ROI_CUBE_HEAD.USE_FACTORIZED_PSEUDO_WEIGHT,
            'use_depth_consistency_loss': cfg.MODEL.ROI_CUBE_HEAD.USE_DEPTH_CONSISTENCY_LOSS,
            'loss_w_depth_consistency': cfg.MODEL.ROI_CUBE_HEAD.LOSS_W_DEPTH_CONSISTENCY,
            'depth_consistency_min_pixels': cfg.MODEL.ROI_CUBE_HEAD.DEPTH_CONSISTENCY_MIN_PIXELS,
            'depth_consistency_center_crop': cfg.MODEL.ROI_CUBE_HEAD.DEPTH_CONSISTENCY_CENTER_CROP,
            'depth_consistency_mode': cfg.MODEL.ROI_CUBE_HEAD.DEPTH_CONSISTENCY_MODE,
            'depth_consistency_percentile': cfg.MODEL.ROI_CUBE_HEAD.DEPTH_CONSISTENCY_PERCENTILE,
            'use_projected_corner_depth_aux': cfg.MODEL.ROI_CUBE_HEAD.USE_PROJECTED_CORNER_DEPTH_AUX,
            'loss_w_projected_corner_2d': cfg.MODEL.ROI_CUBE_HEAD.LOSS_W_PROJECTED_CORNER_2D,
            'loss_w_projected_corner_depth': cfg.MODEL.ROI_CUBE_HEAD.LOSS_W_PROJECTED_CORNER_DEPTH,
            'projected_corner_max_loss': cfg.MODEL.ROI_CUBE_HEAD.PROJECTED_CORNER_MAX_LOSS,
            'use_region_segmentation_head': cfg.MODEL.ROI_CUBE_HEAD.USE_REGION_SEGMENTATION_HEAD,
            'loss_w_region_segmentation': cfg.MODEL.ROI_CUBE_HEAD.LOSS_W_REGION_SEGMENTATION,
            'rsh_use_depth_guidance': cfg.MODEL.ROI_CUBE_HEAD.RSH_USE_DEPTH_GUIDANCE,
            'rsh_depth_mask_threshold': cfg.MODEL.ROI_CUBE_HEAD.RSH_DEPTH_MASK_THRESHOLD,
            'rsh_use_pseudo_weight': cfg.MODEL.ROI_CUBE_HEAD.RSH_USE_PSEUDO_WEIGHT,
            'use_zem_adapter': cfg.MODEL.ROI_CUBE_HEAD.USE_ZEM_ADAPTER,
            'use_geometry_interpreter': use_geometry_interpreter,
            'loss_w_multi_hypothesis': cfg.MODEL.ROI_CUBE_HEAD.LOSS_W_MULTI_HYPOTHESIS,
            'use_differentiable_renderer': cfg.MODEL.ROI_CUBE_HEAD.USE_DIFFERENTIABLE_RENDERER,
            'loss_w_render_silhouette': cfg.MODEL.ROI_CUBE_HEAD.LOSS_W_RENDER_SILHOUETTE,
            'loss_w_render_depth': cfg.MODEL.ROI_CUBE_HEAD.LOSS_W_RENDER_DEPTH,
            'render_size': cfg.MODEL.ROI_CUBE_HEAD.RENDER_SIZE,
            'geometry_selection_temperature': cfg.MODEL.ROI_CUBE_HEAD.GEOMETRY_SELECTION_TEMPERATURE,
            'geometry_oracle_weight': cfg.MODEL.ROI_CUBE_HEAD.GEOMETRY_ORACLE_WEIGHT,
            'geometry_projection_weight': cfg.MODEL.ROI_CUBE_HEAD.GEOMETRY_PROJECTION_WEIGHT,
            'geometry_silhouette_weight': cfg.MODEL.ROI_CUBE_HEAD.GEOMETRY_SILHOUETTE_WEIGHT,
            'geometry_depth_weight': cfg.MODEL.ROI_CUBE_HEAD.GEOMETRY_DEPTH_WEIGHT,
            'geometry_point_weight': cfg.MODEL.ROI_CUBE_HEAD.GEOMETRY_POINT_WEIGHT,
            'geometry_ground_weight': cfg.MODEL.ROI_CUBE_HEAD.GEOMETRY_GROUND_WEIGHT,
            'loss_w_geometry_closure': cfg.MODEL.ROI_CUBE_HEAD.LOSS_W_GEOMETRY_CLOSURE,
            'geometry_apply_to_prediction': cfg.MODEL.ROI_CUBE_HEAD.GEOMETRY_APPLY_TO_PREDICTION,
            'geometry_apply_in_inference': cfg.MODEL.ROI_CUBE_HEAD.GEOMETRY_APPLY_IN_INFERENCE,
            'geometry_apply_warmup_iters': cfg.MODEL.ROI_CUBE_HEAD.GEOMETRY_APPLY_WARMUP_ITERS,
            'geometry_min_dimension': cfg.MODEL.ROI_CUBE_HEAD.GEOMETRY_MIN_DIMENSION,
            'geometry_max_dimension': cfg.MODEL.ROI_CUBE_HEAD.GEOMETRY_MAX_DIMENSION,
            'shape_memory_min_confidence': cfg.MODEL.ROI_CUBE_HEAD.SHAPE_MEMORY_MIN_CONFIDENCE,
            'shape_memory_max_updates': cfg.MODEL.ROI_CUBE_HEAD.SHAPE_MEMORY_MAX_UPDATES,
        }


    def forward(
        self,
        images,
        features,
        text_embeddings,
        proposals,
        Ks,
        im_scales_ratio,
        targets=None,
        prompt_depth=None,
        prompt_ground=None,
    ):

        im_dims = [image.shape[1:] for image in images]
        image_tensor = images.tensor

        if self.training:
            proposals = self.label_and_sample_proposals(proposals, targets)
        
        del targets

        if self.training:

            losses = self._forward_box(features, text_embeddings, proposals)
            if self.loss_w_3d > 0:
                instances_3d, losses_cube = self._forward_cube(
                    features, proposals, Ks, im_dims, im_scales_ratio,
                    prompt_depth=prompt_depth,
                    prompt_ground=prompt_ground,
                    image_tensor=image_tensor,
                )
                losses.update(losses_cube)

            return instances_3d, losses
        
        else:
            # when oracle is available, by pass the box forward.
            # simulate the predicted instances by creating a new 
            # instance for each passed in image.
            if isinstance(proposals, list) and ~np.any([isinstance(p, Instances) for p in proposals]):
                pred_instances = []
                for proposal, im_dim in zip(proposals, im_dims):
                    
                    pred_instances_i = Instances(im_dim)
                    pred_instances_i.pred_boxes = Boxes(proposal['gt_bbox2D'])
                    pred_instances_i.pred_classes =  proposal['gt_classes']
                    pred_instances_i.scores = torch.ones_like(proposal['gt_classes']).float()
                    if "gt_render_masks" in proposal:
                        pred_instances_i.render_masks = proposal["gt_render_masks"]
                    pred_instances.append(pred_instances_i)
            else:
                pred_instances = self._forward_box(features, text_embeddings, proposals)

            if self.loss_w_3d > 0:
                pred_instances = self._forward_cube(
                    features, pred_instances, Ks, im_dims, im_scales_ratio,
                    prompt_depth=prompt_depth,
                    prompt_ground=prompt_ground,
                    image_tensor=image_tensor,
                )
            return pred_instances, {}
    

    def _forward_box(self, features: Dict[str, torch.Tensor], text_embeddings, proposals: List[Instances]):
        """
        Forward logic of the box prediction branch. If `self.train_on_pred_boxes is True`,
            the function puts predicted boxes in the `proposal_boxes` field of `proposals` argument.

        Args:
            features (dict[str, Tensor]): mapping from feature map names to tensor.
                Same as in :meth:`ROIHeads.forward`.
            proposals (list[Instances]): the per-image object proposals with
                their matching ground truth.
                Each has fields "proposal_boxes", and "objectness_logits",
                "gt_classes", "gt_boxes".

        Returns:
            In training, a dict of losses.
            In inference, a list of `Instances`, the predicted instances.
        """
        features = [features[f] for f in self.box_in_features]
        box_features = self.box_pooler(features, [x.proposal_boxes for x in proposals])
        box_features = self.box_head(box_features)
        predictions = self.box_predictor(box_features, text_embeddings)
        del box_features

        if self.training:
            losses = self.box_predictor.losses(
                predictions, proposals, 
            )
            pred_boxes = self.box_predictor.predict_boxes_for_gt_classes(
                predictions, proposals
            )
            for proposals_per_image, pred_boxes_per_image in zip(proposals, pred_boxes):
                proposals_per_image.pred_boxes = Boxes(pred_boxes_per_image)

            # proposals is modified in-place below, so losses must be computed first.
            if self.train_on_pred_boxes:
                with torch.no_grad():
                    pred_boxes = self.box_predictor.predict_boxes_for_gt_classes(
                        predictions, proposals
                    )
                    for proposals_per_image, pred_boxes_per_image in zip(proposals, pred_boxes):
                        proposals_per_image.proposal_boxes = Boxes(pred_boxes_per_image)
            return losses
        else:
            pred_instances, _ = self.box_predictor.inference(predictions, proposals, )
            return pred_instances

    def l1_loss(self, vals, target):
        return F.smooth_l1_loss(vals, target, reduction='none', beta=0.0)

    def chamfer_loss(self, vals, target):
        B = vals.shape[0]
        xx = vals.view(B, 8, 1, 3)
        yy = target.view(B, 1, 8, 3)
        l1_dist = (xx - yy).abs().sum(-1)
        l1 = (l1_dist.min(1).values.mean(-1) + l1_dist.min(2).values.mean(-1))
        return l1

    def yaw_rotation_matrix(self, yaw):
        cosine = torch.cos(yaw)
        sine = torch.sin(yaw)
        zeros = torch.zeros_like(yaw)
        ones = torch.ones_like(yaw)
        row0 = torch.stack((cosine, zeros, sine), dim=-1)
        row1 = torch.stack((zeros, ones, zeros), dim=-1)
        row2 = torch.stack((-sine, zeros, cosine), dim=-1)
        return torch.stack((row0, row1, row2), dim=-2)

    def apply_geometry_hypotheses(
        self,
        cube_x,
        cube_y,
        cube_z,
        cube_dims,
        cube_pose,
        src_widths,
        src_heights,
        hypothesis_deltas,
        hypothesis_quality,
        hypothesis_log_variance,
        explicit_anchor,
        intrinsics,
        roi_boxes,
        point_roi,
        target_silhouette=None,
        observed_depth_roi=None,
        gt_boxes3D=None,
        gt_poses=None,
        factor_weights=None,
        ground_planes=None,
    ):
        if self.soft_renderer is None:
            raise RuntimeError(
                "Geometry hypothesis closure requires USE_DIFFERENTIABLE_RENDERER=True."
            )
        cube_x3d = cube_z * (
            cube_x - intrinsics[:, 0, 2]
        ) / intrinsics[:, 0, 0]
        cube_y3d = cube_z * (
            cube_y - intrinsics[:, 1, 2]
        ) / intrinsics[:, 1, 1]
        cube_center = torch.stack((cube_x3d, cube_y3d, cube_z), dim=1)

        anchor_confidence = explicit_anchor["confidence"].clamp(0.0, 1.0)
        anchor_center = explicit_anchor["center"].to(cube_center)
        anchor_dims = explicit_anchor["dimensions"].to(cube_dims).clamp(
            min=self.geometry_min_dimension,
            max=self.geometry_max_dimension,
        )
        anchor_yaw = explicit_anchor["yaw"].to(cube_z)
        base_center = (
            anchor_confidence[:, None] * anchor_center
            + (1.0 - anchor_confidence[:, None]) * cube_center
        )
        base_dims = torch.exp(
            anchor_confidence[:, None] * torch.log(anchor_dims)
            + (1.0 - anchor_confidence[:, None])
            * torch.log(cube_dims.clamp(min=0.01))
        )
        anchor_pose = self.yaw_rotation_matrix(anchor_yaw)
        mixed_pose = (
            anchor_confidence[:, None, None] * anchor_pose
            + (1.0 - anchor_confidence[:, None, None]) * cube_pose
        )
        with torch.no_grad():
            u, _, vh = torch.linalg.svd(mixed_pose.detach())
            projected_pose = u @ vh
            negative_determinant = torch.det(projected_pose) < 0
            if negative_determinant.any():
                u = u.clone()
                u[negative_determinant, :, -1] *= -1
                projected_pose = u @ vh
        # Keep the projected rotation in the forward pass while avoiding the
        # unstable SVD backward at repeated or nearly repeated singular values.
        base_pose = mixed_pose + (projected_pose - mixed_pose).detach()
        base_z = base_center[:, 2].clamp(min=0.05)
        base_x = (
            intrinsics[:, 0, 0] * base_center[:, 0] / base_z
            + intrinsics[:, 0, 2]
        )
        base_y = (
            intrinsics[:, 1, 1] * base_center[:, 1] / base_z
            + intrinsics[:, 1, 2]
        )

        delta_xy = hypothesis_deltas[:, :, :2]
        delta_log_z = hypothesis_deltas[:, :, 2]
        delta_log_dims = hypothesis_deltas[:, :, 3:6]
        delta_yaw = hypothesis_deltas[:, :, 6]

        candidate_x = base_x[:, None] + src_widths[:, None] * delta_xy[:, :, 0]
        candidate_y = base_y[:, None] + src_heights[:, None] * delta_xy[:, :, 1]
        candidate_z = base_z[:, None] * torch.exp(delta_log_z.clamp(-0.7, 0.7))
        candidate_dims = base_dims[:, None, :] * torch.exp(
            delta_log_dims.clamp(-0.7, 0.7)
        )
        candidate_dims = candidate_dims.clamp(
            min=self.geometry_min_dimension,
            max=self.geometry_max_dimension,
        )
        yaw_rotation = self.yaw_rotation_matrix(delta_yaw)
        candidate_pose = torch.matmul(base_pose[:, None, :, :], yaw_rotation)
        candidate_x3d = candidate_z * (
            candidate_x - intrinsics[:, None, 0, 2]
        ) / intrinsics[:, None, 0, 0]
        candidate_y3d = candidate_z * (
            candidate_y - intrinsics[:, None, 1, 2]
        ) / intrinsics[:, None, 1, 1]
        candidate_center = torch.stack(
            (candidate_x3d, candidate_y3d, candidate_z),
            dim=-1,
        )

        batch_size, hypotheses = candidate_z.shape
        flat_box = torch.cat(
            (
                candidate_center.reshape(-1, 3),
                candidate_dims.reshape(-1, 3),
            ),
            dim=1,
        )
        flat_pose = candidate_pose.reshape(-1, 3, 3)
        candidate_corners = util.get_cuboid_verts_faces(
            flat_box,
            flat_pose,
        )[0].reshape(batch_size, hypotheses, 8, 3)

        projected = torch.matmul(
            intrinsics[:, None, None, :, :],
            candidate_corners.unsqueeze(-1),
        ).squeeze(-1)
        projected_uv = projected[..., :2] / projected[..., 2:3].clamp(min=1e-4)
        projected_min = projected_uv.min(dim=2).values
        projected_max = projected_uv.max(dim=2).values
        projected_boxes = torch.cat((projected_min, projected_max), dim=-1)
        target_boxes = roi_boxes[:, None, :].expand_as(projected_boxes)
        intersection_min = torch.maximum(projected_boxes[..., :2], target_boxes[..., :2])
        intersection_max = torch.minimum(projected_boxes[..., 2:], target_boxes[..., 2:])
        intersection_size = (intersection_max - intersection_min).clamp(min=0.0)
        intersection = intersection_size.prod(dim=-1)
        projected_area = (
            (projected_boxes[..., 2:] - projected_boxes[..., :2]).clamp(min=0.0)
        ).prod(dim=-1)
        target_area = (
            (target_boxes[..., 2:] - target_boxes[..., :2]).clamp(min=0.0)
        ).prod(dim=-1)
        projection_iou = intersection / (
            projected_area + target_area - intersection
        ).clamp(min=1.0)
        projection_cost = 1.0 - projection_iou

        repeated_intrinsics = intrinsics[:, None].expand(
            -1, hypotheses, -1, -1
        ).reshape(-1, 3, 3)
        repeated_roi_boxes = roi_boxes[:, None].expand(
            -1, hypotheses, -1
        ).reshape(-1, 4)
        rendered_silhouette, rendered_depth = self.soft_renderer(
            candidate_corners.reshape(-1, 8, 3),
            repeated_intrinsics,
            repeated_roi_boxes,
        )
        rendered_silhouette = rendered_silhouette.reshape(
            batch_size, hypotheses, self.render_size, self.render_size
        )
        rendered_depth = rendered_depth.reshape(
            batch_size, hypotheses, self.render_size, self.render_size
        )

        if target_silhouette is not None:
            target_silhouette_expanded = target_silhouette[:, None].expand(
                -1, hypotheses, -1, -1
            )
            silhouette_intersection = (
                rendered_silhouette * target_silhouette_expanded
            ).sum(dim=(2, 3))
            silhouette_denominator = (
                rendered_silhouette.sum(dim=(2, 3))
                + target_silhouette_expanded.sum(dim=(2, 3))
            ).clamp(min=1.0)
            silhouette_cost = 1.0 - (
                2.0 * silhouette_intersection + 1.0
            ) / (silhouette_denominator + 1.0)
        else:
            silhouette_cost = projection_cost

        if observed_depth_roi is not None:
            observed_depth_expanded = observed_depth_roi[:, None].expand(
                -1, hypotheses, -1, -1
            )
            valid_depth = (
                torch.isfinite(observed_depth_expanded)
                & (observed_depth_expanded > 0.05)
            )
            if target_silhouette is not None:
                valid_depth = valid_depth & (target_silhouette_expanded > 0.5)
            depth_residual = F.smooth_l1_loss(
                torch.log(rendered_depth.clamp(0.05, 80.0)),
                torch.log(observed_depth_expanded.clamp(0.05, 80.0)),
                reduction="none",
                beta=0.05,
            )
            depth_weight = valid_depth.float() * rendered_silhouette.detach()
            depth_cost = (
                depth_residual * depth_weight
            ).sum(dim=(2, 3)) / depth_weight.sum(dim=(2, 3)).clamp(min=1.0)
        else:
            depth_cost = projection_cost.new_zeros(projection_cost.shape)

        points = point_roi.permute(0, 2, 3, 1).reshape(batch_size, -1, 3)
        valid_points = torch.isfinite(points).all(dim=2) & (points[:, :, 2] > 0.05)
        difference = points[:, None, :, :] - candidate_center[:, :, None, :]
        local_points = torch.einsum(
            "nkpc,nkcd->nkpd",
            difference,
            candidate_pose,
        )
        half_extent = torch.stack(
            (
                candidate_dims[..., 2],
                candidate_dims[..., 1],
                candidate_dims[..., 0],
            ),
            dim=-1,
        )[:, :, None, :] * 0.5
        softness = (candidate_dims.mean(dim=-1) * 0.05).clamp(min=0.01)
        inside_probability = torch.sigmoid(
            (half_extent - local_points.abs())
            / softness[:, :, None, None]
        ).prod(dim=-1)
        point_support = (
            inside_probability * valid_points[:, None].float()
        ).sum(dim=2) / valid_points.sum(dim=1)[:, None].clamp(min=1.0)
        point_cost = 1.0 - point_support

        if ground_planes is not None and ground_planes[:, 4].bool().any():
            normals = ground_planes[:, :3]
            offsets = ground_planes[:, 3]
            plane_distance = (
                (
                    candidate_corners
                    * normals[:, None, None, :]
                ).sum(dim=-1)
                + offsets[:, None, None]
            ).abs()
            ground_cost_plane = (
                plane_distance.min(dim=2).values
                / candidate_dims[..., 1].clamp(min=0.05)
            ).clamp(max=2.0)
            plane_valid = ground_planes[:, 4].bool()[:, None]
        else:
            ground_cost_plane = projection_cost.new_zeros(projection_cost.shape)
            plane_valid = torch.zeros(
                (batch_size, 1),
                dtype=torch.bool,
                device=projection_cost.device,
            )

        observed_bottom_values = []
        fallback_bottom = candidate_corners[..., 1].max(dim=2).values.detach().mean(dim=1)
        for sample_index in range(batch_size):
            values = points[sample_index, valid_points[sample_index], 1]
            if values.numel() > 0:
                observed_bottom_values.append(torch.quantile(values, 0.90))
            else:
                observed_bottom_values.append(fallback_bottom[sample_index])
        observed_bottom = torch.stack(observed_bottom_values)
        candidate_bottom = candidate_corners[..., 1].max(dim=2).values
        local_ground_cost = (
            (candidate_bottom - observed_bottom[:, None]).abs()
            / candidate_dims[..., 1].clamp(min=0.05)
        ).clamp(max=2.0)
        ground_cost = torch.where(
            plane_valid,
            ground_cost_plane,
            local_ground_cost,
        )

        geometry_cost = (
            self.geometry_projection_weight * projection_cost
            + self.geometry_silhouette_weight * silhouette_cost
            + self.geometry_depth_weight * depth_cost
            + self.geometry_point_weight * point_cost
            + self.geometry_ground_weight * ground_cost
        )

        auxiliary_losses = {}
        oracle_cost = None
        if self.training and gt_boxes3D is not None and gt_poses is not None:
            gt_xy = gt_boxes3D[:, :2]
            gt_z = gt_boxes3D[:, 2].clamp(min=0.05)
            gt_dims = gt_boxes3D[:, 3:6].clamp(min=0.01)
            xy_error = (
                (candidate_x - gt_xy[:, None, 0]).abs()
                / src_widths[:, None].clamp(min=1.0)
                + (candidate_y - gt_xy[:, None, 1]).abs()
                / src_heights[:, None].clamp(min=1.0)
            ) * 0.5
            z_error = (
                torch.log(candidate_z.clamp(min=0.05))
                - torch.log(gt_z[:, None])
            ).abs()
            dims_error = (
                torch.log(candidate_dims.clamp(min=0.01))
                - torch.log(gt_dims[:, None, :])
            ).abs().mean(dim=-1)
            relative_rotation = torch.matmul(
                candidate_pose.transpose(-1, -2),
                gt_poses[:, None, :, :],
            )
            trace = relative_rotation.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
            pose_error = 1.0 - ((trace - 1.0) * 0.5).clamp(-1.0, 1.0)

            if factor_weights is None:
                factor_weights = {
                    name: torch.ones_like(gt_z)
                    for name in ("xy", "z", "dims", "pose")
                }
            oracle_cost = (
                xy_error * factor_weights["xy"][:, None]
                + z_error * factor_weights["z"][:, None]
                + dims_error * factor_weights["dims"][:, None]
                + pose_error * factor_weights["pose"][:, None]
            )
            selection_cost = (
                geometry_cost
                + self.geometry_oracle_weight * oracle_cost
            )
        else:
            selection_cost = geometry_cost
            xy_error = z_error = dims_error = pose_error = None

        geometry_logits = (
            hypothesis_quality - selection_cost
        ) / max(self.geometry_selection_temperature, 1e-4)
        best_index = selection_cost.detach().argmin(dim=1)
        auxiliary_losses["quality"] = F.cross_entropy(
            hypothesis_quality,
            best_index,
            reduction="none",
        )
        if self.training:
            selection_weights = torch.softmax(geometry_logits, dim=1)
        else:
            # At inference the rendered geometry/depth closure is the actual
            # verifier. The learned quality head is auxiliary and can become
            # over-confident on degenerate thin/long boxes, so use the measured
            # cost directly for hard selection.
            selection_weights = F.one_hot(
                best_index,
                num_classes=hypotheses,
            ).to(candidate_x.dtype)

        selected_x = (selection_weights * candidate_x).sum(dim=1)
        selected_y = (selection_weights * candidate_y).sum(dim=1)
        selected_z = (selection_weights * candidate_z).sum(dim=1)
        selected_dims = torch.exp(
            (
                selection_weights[:, :, None]
                * torch.log(candidate_dims.clamp(min=0.01))
            ).sum(dim=1)
        ).clamp(
            min=self.geometry_min_dimension,
            max=self.geometry_max_dimension,
        )
        selected_yaw = (selection_weights * delta_yaw).sum(dim=1)
        selected_pose = torch.matmul(
            base_pose,
            self.yaw_rotation_matrix(selected_yaw),
        )
        selected_quality = (
            selection_weights * hypothesis_quality
        ).sum(dim=1)
        selected_log_variance = (
            selection_weights[:, :, None] * hypothesis_log_variance
        ).sum(dim=1)

        if self.training and xy_error is not None:
            selected_errors = torch.stack(
                (
                    (selection_weights * xy_error).sum(dim=1),
                    (selection_weights * z_error).sum(dim=1),
                    (selection_weights * dims_error).sum(dim=1),
                    (selection_weights * pose_error).sum(dim=1),
                ),
                dim=1,
            )
            auxiliary_losses["uncertainty"] = (
                F.softplus(
                    torch.exp(-selected_log_variance) * selected_errors
                    + selected_log_variance
                )
            ).mean(dim=1)
        auxiliary_losses["closure"] = (
            selection_weights * geometry_cost
        ).sum(dim=1)
        auxiliary_losses["projection"] = (
            selection_weights * projection_cost
        ).sum(dim=1)
        auxiliary_losses["silhouette"] = (
            selection_weights * silhouette_cost
        ).sum(dim=1)
        auxiliary_losses["depth"] = (
            selection_weights * depth_cost
        ).sum(dim=1)
        auxiliary_losses["point"] = (
            selection_weights * point_cost
        ).sum(dim=1)
        auxiliary_losses["ground"] = (
            selection_weights * ground_cost
        ).sum(dim=1)

        selected_geometry_cost = (
            selection_weights * geometry_cost
        ).sum(dim=1)
        geometry_confidence = (
            torch.sigmoid(selected_quality)
            * torch.exp(-F.softplus(selected_log_variance).mean(dim=1))
            * torch.exp(-selected_geometry_cost.detach())
        ).clamp(0.0, 1.0)
        return (
            selected_x,
            selected_y,
            selected_z,
            selected_dims,
            selected_pose,
            geometry_confidence,
            auxiliary_losses,
            selected_geometry_cost,
        )

    # optionally, scale proposals to zoom RoI in (<1.0) our out (>1.0)
    def scale_proposals(self, proposal_boxes):
        if self.scale_roi_boxes > 0:

            proposal_boxes_scaled = []
            for boxes in proposal_boxes:
                centers = boxes.get_centers()
                widths = boxes.tensor[:, 2] - boxes.tensor[:, 0]
                heights = boxes.tensor[:, 2] - boxes.tensor[:, 0]
                x1 = centers[:, 0] - 0.5*widths*self.scale_roi_boxes
                x2 = centers[:, 0] + 0.5*widths*self.scale_roi_boxes
                y1 = centers[:, 1] - 0.5*heights*self.scale_roi_boxes
                y2 = centers[:, 1] + 0.5*heights*self.scale_roi_boxes
                boxes_scaled = Boxes(torch.stack([x1, y1, x2, y2], dim=1))
                proposal_boxes_scaled.append(boxes_scaled)
        else:
            proposal_boxes_scaled = proposal_boxes

        return proposal_boxes_scaled
    
    def _forward_cube(
        self,
        features,
        instances,
        Ks,
        im_current_dims,
        im_scales_ratio,
        prompt_depth=None,
        prompt_ground=None,
        image_tensor=None,
    ):
        features = [features[f] for f in self.in_features]

        # training on foreground
        if self.training:

            losses = {}

            # add up the amount we should normalize the losses by. 
            # this follows the same logic as the BoxHead, where each FG proposal 
            # is able to contribute the same amount of supervision. Technically, 
            # this value doesn't change during training unless the batch size is dynamic.
            self.normalize_factor = max(sum([i.gt_classes.numel() for i in instances]), 1.0)

            # The loss is only defined on positive proposals
            proposals, _ = select_foreground_proposals(instances, self.num_classes)
            proposal_boxes = [x.proposal_boxes for x in proposals]
            pred_boxes = [x.pred_boxes for x in proposals]
            depth_target_boxes = [
                x.gt_boxes if x.has("gt_boxes") else x.proposal_boxes
                for x in proposals
            ]

            box_classes = (torch.cat([p.gt_classes for p in proposals], dim=0) if len(proposals) else torch.empty(0))
            gt_boxes3D = torch.cat([p.gt_boxes3D for p in proposals], dim=0,)
            gt_poses = torch.cat([p.gt_poses for p in proposals], dim=0,)
            if len(proposals) and proposals[0].has("gt_render_masks"):
                gt_render_masks = torch.cat(
                    [p.gt_render_masks for p in proposals],
                    dim=0,
                ).to(gt_boxes3D.device)
            else:
                gt_render_masks = None
            if len(proposals) and proposals[0].has("gt_pseudo_weight"):
                gt_pseudo_weight = torch.cat([p.gt_pseudo_weight for p in proposals], dim=0).to(gt_boxes3D.device)
            else:
                gt_pseudo_weight = torch.ones_like(box_classes, dtype=torch.float32, device=gt_boxes3D.device)
            gt_factor_weights = {}
            for factor_name in ("xy", "z", "dims", "pose", "joint"):
                field_name = f"gt_pseudo_weight_{factor_name}"
                if len(proposals) and proposals[0].has(field_name):
                    gt_factor_weights[factor_name] = torch.cat(
                        [p.get(field_name) for p in proposals],
                        dim=0,
                    ).to(gt_boxes3D.device)
                else:
                    gt_factor_weights[factor_name] = gt_pseudo_weight
            if len(proposals) and proposals[0].has("gt_pag_score"):
                gt_pag_score = torch.cat(
                    [p.gt_pag_score for p in proposals],
                    dim=0,
                ).to(gt_boxes3D.device).clamp(0.05, 1.0)
            else:
                gt_pag_score = torch.ones_like(
                    gt_pseudo_weight,
                    dtype=torch.float32,
                    device=gt_boxes3D.device,
                )
            if len(proposals) and proposals[0].has("gt_projected_corner_depth_score"):
                gt_projected_corner_depth_score = torch.cat(
                    [p.gt_projected_corner_depth_score for p in proposals],
                    dim=0,
                ).to(gt_boxes3D.device).clamp(0.05, 1.0)
            else:
                gt_projected_corner_depth_score = torch.ones_like(
                    gt_pseudo_weight,
                    dtype=torch.float32,
                    device=gt_boxes3D.device,
                )
            gt_corner_aux_quality = torch.minimum(
                gt_pag_score,
                gt_projected_corner_depth_score,
            ).clamp(0.05, 1.0)
            assert len(gt_poses) == len(gt_boxes3D) == len(box_classes)

        # eval on all instances
        else:
            proposals = instances
            pred_boxes = [x.pred_boxes for x in instances]
            proposal_boxes = pred_boxes
            depth_target_boxes = None
            box_classes = torch.cat([x.pred_classes for x in instances])
            if len(proposals) and proposals[0].has("render_masks"):
                proposal_render_masks = torch.cat(
                    [proposal.render_masks for proposal in proposals],
                    dim=0,
                ).to(box_classes.device)
            else:
                proposal_render_masks = None

        proposal_boxes_scaled = self.scale_proposals(proposal_boxes)

        # forward features
        point_map = None
        ray_map = None
        Ks_scaled_per_image = torch.stack([
            Ks[i] / im_scales_ratio[i] for i in range(len(Ks))
        ]).to(features[0].device)
        Ks_scaled_per_image[:, -1, -1] = 1
        needs_geometry_maps = (
            self.use_depth_roi
            or self.use_zem_adapter
            or self.use_geometry_interpreter
            or self.use_differentiable_renderer
        )
        if needs_geometry_maps and prompt_depth is not None:
            point_map = depth_to_point_map(prompt_depth.tensor.to(features[0].device), Ks_scaled_per_image)
            ray_map = build_ray_map(
                prompt_depth.tensor.to(features[0].device),
                Ks_scaled_per_image,
            )
        ground_planes_per_image = estimate_ground_planes(
            point_map,
            prompt_ground.tensor.to(features[0].device)
            if prompt_ground is not None
            else None,
        )

        if self.use_depth_roi:
            cube_feature_map = self.cube_pooler(features, proposal_boxes_scaled, point_map)
        else:
            cube_feature_map = self.cube_pooler(features, proposal_boxes_scaled)

        rsh_mask_logits = None
        rsh_mask_prob = None
        rsh_depth_masks_flat = None
        if (
            self.use_region_segmentation_head
            and self.region_segmentation_head is not None
        ):
            cube_feature_map, rsh_mask_logits, rsh_mask_prob = (
                self.region_segmentation_head(cube_feature_map)
            )
            if self.training and gt_render_masks is not None:
                rsh_targets = F.interpolate(
                    gt_render_masks[:, None].float(),
                    size=rsh_mask_logits.shape[-2:],
                    mode="nearest",
                )
                rsh_loss = F.binary_cross_entropy_with_logits(
                    rsh_mask_logits,
                    rsh_targets,
                    reduction="none",
                )
                rsh_pixel_weight = 1.0 + 2.0 * rsh_targets
                rsh_loss = (
                    rsh_loss * rsh_pixel_weight
                ).mean(dim=(1, 2, 3))
                rsh_valid = rsh_targets.sum(dim=(1, 2, 3)) > 4.0
                if gt_pseudo_weight.numel() == rsh_loss.numel():
                    rsh_loss = rsh_loss * gt_pseudo_weight.to(rsh_loss.device).clamp(0.25, 1.0)
                if rsh_valid.any():
                    losses["Cube/loss_region_seg"] = (
                        self.safely_reduce_losses(rsh_loss[rsh_valid])
                        * self.loss_w_region_segmentation
                        * self.loss_w_3d
                    )

                with torch.no_grad():
                    rsh_prob_detached = rsh_mask_prob.detach()
                    intersection = (
                        rsh_prob_detached * rsh_targets
                    ).sum(dim=(1, 2, 3))
                    denominator = (
                        rsh_prob_detached.sum(dim=(1, 2, 3))
                        + rsh_targets.sum(dim=(1, 2, 3))
                    ).clamp(min=1.0)
                    rsh_quality = (
                        (2.0 * intersection + 1.0)
                        / (denominator + 1.0)
                    ).clamp(0.05, 1.0)
                    rsh_quality = torch.where(
                        rsh_valid,
                        rsh_quality,
                        torch.ones_like(rsh_quality),
                    )
                    get_event_storage().put_scalar(
                        "Cube/rsh_dice",
                        rsh_quality[rsh_valid].mean().item() if rsh_valid.any() else 1.0,
                        smoothing_hint=False,
                    )
                    if self.rsh_use_pseudo_weight:
                        rsh_factor = (0.5 + 0.5 * rsh_quality).to(gt_boxes3D.device)
                        for factor_name in ("z", "dims", "joint"):
                            gt_factor_weights[factor_name] = (
                                gt_factor_weights[factor_name] * rsh_factor
                            ).clamp(0.05, 1.0)

                if self.rsh_use_depth_guidance:
                    rsh_depth_masks_flat = rsh_targets[:, 0].detach()
            elif (not self.training) and rsh_mask_prob is not None:
                if proposal_render_masks is None:
                    proposal_render_masks = rsh_mask_prob[:, 0].detach()
                if self.rsh_use_depth_guidance:
                    rsh_depth_masks_flat = rsh_mask_prob[:, 0].detach()

        zem_gate = None
        if self.use_zem_adapter and self.zem_adapter is not None:
            roi_size = cube_feature_map.shape[-1]
            if point_map is not None:
                zem_point_roi = roi_align_map(
                    point_map,
                    proposal_boxes_scaled,
                    roi_size,
                    spatial_scale=1.0,
                )
            else:
                zem_point_roi = None
            if ray_map is not None:
                zem_ray_roi = roi_align_map(
                    ray_map,
                    proposal_boxes_scaled,
                    roi_size,
                    spatial_scale=1.0,
                )
            else:
                zem_ray_roi = None
            if self.training and gt_render_masks is not None:
                zem_mask_roi = gt_render_masks
            elif (not self.training) and proposal_render_masks is not None:
                zem_mask_roi = proposal_render_masks
            elif zem_point_roi is not None:
                zem_mask_roi = (zem_point_roi[:, 2] > 0.05).float()
            else:
                zem_mask_roi = None
            cube_feature_map, zem_gate = self.zem_adapter(
                cube_feature_map,
                point_roi=zem_point_roi,
                ray_roi=zem_ray_roi,
                mask_roi=zem_mask_roi,
            )
            if zem_gate is not None:
                get_event_storage().put_scalar(
                    "Cube/zem_gate_mean",
                    float(zem_gate.detach().mean().item()),
                    smoothing_hint=False,
                )
        cube_features = cube_feature_map.flatten(1)

        n = cube_features.shape[0]
        
        # nothing to do..
        if n == 0:
            return instances if not self.training else (instances, {})

        num_boxes_per_image = [len(i) for i in proposals]

        hypothesis_deltas = None
        hypothesis_quality = None
        hypothesis_log_variance = None
        explicit_geometry_anchor = None
        dino_descriptor = None
        prototype_confidence = None
        point_roi = None
        if self.use_geometry_interpreter:
            if image_tensor is None:
                raise RuntimeError("Geometry interpreter requires the detector image tensor.")
            if point_map is None or ray_map is None:
                spatial_shape = image_tensor.shape[-2:]
                point_map = image_tensor.new_zeros(
                    (image_tensor.shape[0], 3, *spatial_shape)
                )
                ray_map = build_ray_map(
                    image_tensor.new_ones(
                        (image_tensor.shape[0], 1, *spatial_shape)
                    ),
                    Ks_scaled_per_image,
                )
            dino_map, dino_spatial_scale = self.dino_encoder(image_tensor)
            roi_size = cube_feature_map.shape[-1]
            dino_roi = roi_align_map(
                dino_map,
                proposal_boxes_scaled,
                roi_size,
                spatial_scale=dino_spatial_scale,
            )
            point_roi = roi_align_map(
                point_map,
                proposal_boxes_scaled,
                roi_size,
                spatial_scale=1.0,
            )
            ray_roi = roi_align_map(
                ray_map,
                proposal_boxes_scaled,
                roi_size,
                spatial_scale=1.0,
            )
            (
                hypothesis_deltas,
                hypothesis_quality,
                hypothesis_log_variance,
                explicit_geometry_anchor,
                dino_descriptor,
                prototype_confidence,
            ) = self.geometry_interpreter(
                cube_feature_map,
                dino_roi,
                point_roi,
                ray_roi,
            )

        # scale the intrinsics according to the ratio the image has been scaled. 
        # this means the projections at the current scale are in sync.
        Ks_scaled_per_box = torch.cat([
            (Ks[i]/im_scales_ratio[i]).unsqueeze(0).repeat([num, 1, 1]) 
            for (i, num) in enumerate(num_boxes_per_image)
        ]).to(cube_features.device)
        Ks_scaled_per_box[:, -1, -1] = 1
        ground_planes_per_box = (
            torch.cat(
                [
                    ground_planes_per_image[image_index : image_index + 1].repeat(
                        num_boxes, 1
                    )
                    for image_index, num_boxes in enumerate(num_boxes_per_image)
                ],
                dim=0,
            )
            if ground_planes_per_image is not None
            else None
        )

        focal_lengths_per_box = torch.cat([
            (Ks[i][1, 1]).unsqueeze(0).repeat([num]) 
            for (i, num) in enumerate(num_boxes_per_image)
        ]).to(cube_features.device)

        im_ratios_per_box = torch.cat([
            torch.FloatTensor([im_scales_ratio[i]]).repeat(num) 
            for (i, num) in enumerate(num_boxes_per_image)
        ]).to(cube_features.device)

        # scaling factor for Network resolution -> Original
        im_scales_per_box = torch.cat([
            torch.FloatTensor([im_current_dims[i][0]]).repeat(num) 
            for (i, num) in enumerate(num_boxes_per_image)
        ]).to(cube_features.device)

        im_scales_original_per_box = im_scales_per_box * im_ratios_per_box

        if self.virtual_depth:
                
            virtual_to_real = util.compute_virtual_scale_from_focal_spaces(
                focal_lengths_per_box, im_scales_original_per_box, 
                self.virtual_focal, im_scales_per_box
            )
            real_to_virtual = 1 / virtual_to_real

        else:
            real_to_virtual = virtual_to_real = 1.0

        # 2D boxes are needed to apply deltas
        src_boxes = torch.cat([box_per_im.tensor for box_per_im in proposal_boxes], dim=0)
        src_widths = src_boxes[:, 2] - src_boxes[:, 0]
        src_heights = src_boxes[:, 3] - src_boxes[:, 1]
        src_scales = (src_heights**2 + src_widths**2).sqrt()
        src_ctr_x = src_boxes[:, 0] + 0.5 * src_widths
        src_ctr_y = src_boxes[:, 1] + 0.5 * src_heights

        # For some methods, we need the predicted 2D box,
        # e.g., the differentiable tensors from the 2D box head. 
        pred_src_boxes = torch.cat([box_per_im.tensor for box_per_im in pred_boxes], dim=0)
        pred_widths = pred_src_boxes[:, 2] - pred_src_boxes[:, 0]
        pred_heights = pred_src_boxes[:, 3] - pred_src_boxes[:, 1]
        pred_src_x = (pred_src_boxes[:, 2] + pred_src_boxes[:, 0]) * 0.5
        pred_src_y = (pred_src_boxes[:, 3] + pred_src_boxes[:, 1]) * 0.5
        
        # forward predictions
        cube_2d_deltas, cube_z, cube_dims, cube_pose, cube_uncert = self.cube_head(cube_features)
        
        # simple indexing re-used commonly for selection purposes
        fg_inds = torch.arange(n)

        # Z when clusters are used
        if cube_z is not None and self.cluster_bins > 1:
        
            # compute closest bin assignments per batch per category (batch x n_category)
            scales_diff = (self.priors_z_scales.detach().T.unsqueeze(0) - src_scales.unsqueeze(1).unsqueeze(2)).abs()
            
            # assign the correct scale prediction.
            # (the others are not used / thrown away)
            assignments = scales_diff.argmin(1)

            # select FG, category, and correct cluster
            cube_z = cube_z[fg_inds, :, box_classes, :][fg_inds, assignments[fg_inds, box_classes]]

        elif cube_z is not None:

            # if z is available, collect the per-category predictions.
            cube_z = cube_z[fg_inds, box_classes, :]
            
        cube_dims = cube_dims[fg_inds, box_classes, :]
        cube_pose = cube_pose[fg_inds, box_classes, :, :]

        if self.use_confidence:
            
            # if uncertainty is available, collect the per-category predictions.
            cube_uncert = cube_uncert[fg_inds, box_classes]
        
        cube_2d_deltas = cube_2d_deltas[fg_inds, box_classes, :]
        
        # apply our predicted deltas based on src boxes.
        cube_x = src_ctr_x + src_widths * cube_2d_deltas[:, 0]
        cube_y = src_ctr_y + src_heights * cube_2d_deltas[:, 1]
        
        cube_xy = torch.cat((cube_x.unsqueeze(1), cube_y.unsqueeze(1)), dim=1)

        cube_dims_norm = cube_dims

        if self.dims_priors_enabled:

            # gather prior dimensions
            prior_dims = self.priors_dims_per_cat.detach().repeat([n, 1, 1, 1])[fg_inds, box_classes]
            prior_dims_mean = prior_dims[:, 0, :]
            prior_dims_std = prior_dims[:, 1, :]

            if self.dims_priors_func == 'sigmoid':
                prior_dims_min = (prior_dims_mean - 3*prior_dims_std).clip(0.0)
                prior_dims_max = (prior_dims_mean + 3*prior_dims_std)
                cube_dims = util.scaled_sigmoid(cube_dims_norm, min=prior_dims_min, max=prior_dims_max)
            elif self.dims_priors_func == 'exp':
                cube_dims = torch.exp(cube_dims_norm.clip(max=5)) * prior_dims_mean

        else:
            # no priors are used
            cube_dims = torch.exp(cube_dims_norm.clip(max=5))
        
        if self.allocentric_pose:
            # To compare with GTs, we need the pose to be egocentric, not allocentric
            cube_pose_allocentric = cube_pose
            cube_pose = util.R_from_allocentric(Ks_scaled_per_box, cube_pose, u=cube_x.detach(), v=cube_y.detach())
        cube_z = cube_z.squeeze()
        
        if self.z_type =='sigmoid':    
            cube_z_norm = torch.sigmoid(cube_z)
            cube_z = cube_z_norm * 100

        elif self.z_type == 'log':
            cube_z_norm = cube_z
            cube_z = torch.exp(cube_z)

        elif self.z_type == 'clusters':
            
            # gather the mean depth, same operation as above, for a n x c result
            z_means = self.priors_z_stats[:, :, 0].T.unsqueeze(0).repeat([n, 1, 1])
            z_means = torch.gather(z_means, 1, assignments.unsqueeze(1)).squeeze(1)

            # gather the std depth, same operation as above, for a n x c result
            z_stds = self.priors_z_stats[:, :, 1].T.unsqueeze(0).repeat([n, 1, 1])
            z_stds = torch.gather(z_stds, 1, assignments.unsqueeze(1)).squeeze(1)

            # do not learn these, they are static
            z_means = z_means.detach()
            z_stds = z_stds.detach()

            z_means = z_means[fg_inds, box_classes]
            z_stds = z_stds[fg_inds, box_classes]

            z_mins = (z_means - 3*z_stds).clip(0)
            z_maxs = (z_means + 3*z_stds)

            cube_z_norm = cube_z
            cube_z = util.scaled_sigmoid(cube_z, min=z_mins, max=z_maxs)

        if self.virtual_depth:
            cube_z = (cube_z * virtual_to_real)

        geometry_confidence = None
        geometry_cost = None
        closure_boxes = depth_target_boxes if self.training else proposal_boxes
        closure_roi_boxes = torch.cat(
            [boxes.tensor for boxes in closure_boxes],
            dim=0,
        ).to(cube_z.device)
        closure_observed_depth = None
        if prompt_depth is not None:
            closure_observed_depth = roi_align_map(
                prompt_depth.tensor.to(cube_z.device),
                closure_boxes,
                self.render_size,
                spatial_scale=1.0,
            )[:, 0]
        closure_target_silhouette = None
        if self.training and gt_render_masks is not None:
            closure_target_silhouette = F.interpolate(
                gt_render_masks[:, None].float(),
                size=(self.render_size, self.render_size),
                mode="nearest",
            )[:, 0]
        elif not self.training and proposal_render_masks is not None:
            closure_target_silhouette = F.interpolate(
                proposal_render_masks[:, None].float(),
                size=(self.render_size, self.render_size),
                mode="nearest",
            )[:, 0]

        if (
            self.training
            and hypothesis_deltas is not None
            and dino_descriptor is not None
        ):
            memory_confidence = gt_factor_weights["dims"]
            if memory_confidence.numel() > self.shape_memory_max_updates:
                memory_indices = memory_confidence.topk(
                    self.shape_memory_max_updates
                ).indices
            else:
                memory_indices = torch.arange(
                    memory_confidence.numel(),
                    device=memory_confidence.device,
                )
            self.geometry_interpreter.update_shape_memory(
                dino_descriptor[memory_indices],
                gt_boxes3D[memory_indices, 3:6],
                memory_confidence[memory_indices],
                self.shape_memory_min_confidence,
            )

        if (
            not self.training
            and hypothesis_deltas is not None
            and self.geometry_apply_in_inference
        ):
            (
                cube_x,
                cube_y,
                cube_z,
                cube_dims,
                cube_pose,
                geometry_confidence,
                _,
                geometry_cost,
            ) = self.apply_geometry_hypotheses(
                cube_x,
                cube_y,
                cube_z,
                cube_dims,
                cube_pose,
                src_widths,
                src_heights,
                hypothesis_deltas,
                hypothesis_quality,
                hypothesis_log_variance,
                explicit_geometry_anchor,
                Ks_scaled_per_box,
                closure_roi_boxes,
                point_roi,
                target_silhouette=closure_target_silhouette,
                observed_depth_roi=closure_observed_depth,
                ground_planes=ground_planes_per_box,
            )

        if self.training:
            prefix = 'Cube/'
            if hypothesis_deltas is not None:
                (
                    geometry_cube_x,
                    geometry_cube_y,
                    geometry_cube_z,
                    geometry_cube_dims,
                    geometry_cube_pose,
                    geometry_confidence,
                    hypothesis_losses,
                    geometry_cost,
                ) = self.apply_geometry_hypotheses(
                    cube_x,
                    cube_y,
                    cube_z,
                    cube_dims,
                    cube_pose,
                    src_widths,
                    src_heights,
                    hypothesis_deltas,
                    hypothesis_quality,
                    hypothesis_log_variance,
                    explicit_geometry_anchor,
                    Ks_scaled_per_box,
                    closure_roi_boxes,
                    point_roi,
                    target_silhouette=closure_target_silhouette,
                    observed_depth_roi=closure_observed_depth,
                    ground_planes=ground_planes_per_box,
                    gt_boxes3D=gt_boxes3D,
                    gt_poses=gt_poses,
                    factor_weights=gt_factor_weights,
                )
                geometry_apply_ready = True
                if self.geometry_apply_warmup_iters > 0:
                    try:
                        geometry_apply_ready = (
                            int(get_event_storage().iter)
                            >= self.geometry_apply_warmup_iters
                        )
                    except Exception:
                        geometry_apply_ready = False
                if self.geometry_apply_to_prediction and geometry_apply_ready:
                    cube_x = geometry_cube_x
                    cube_y = geometry_cube_y
                    cube_z = geometry_cube_z
                    cube_dims = geometry_cube_dims
                    cube_pose = geometry_cube_pose
                get_event_storage().put_scalar(
                    prefix + "geometry_apply_ready",
                    float(geometry_apply_ready),
                    smoothing_hint=False,
                )
                if self.loss_w_multi_hypothesis > 0:
                    losses[prefix + 'loss_hypothesis_quality'] = (
                        self.safely_reduce_losses(hypothesis_losses["quality"])
                        * self.loss_w_multi_hypothesis
                        * self.loss_w_3d
                    )
                    losses[prefix + 'loss_hypothesis_uncertainty'] = (
                        self.safely_reduce_losses(hypothesis_losses["uncertainty"])
                        * self.loss_w_multi_hypothesis
                        * self.loss_w_3d
                    )
                if self.loss_w_geometry_closure > 0:
                    losses[prefix + 'loss_geometry_closure'] = (
                        self.safely_reduce_losses(hypothesis_losses["closure"])
                        * self.loss_w_geometry_closure
                        * self.loss_w_3d
                    )
                storage = get_event_storage()
                for component_name in (
                    "projection",
                    "silhouette",
                    "depth",
                    "point",
                    "ground",
                ):
                    storage.put_scalar(
                        prefix + "closure_" + component_name,
                        hypothesis_losses[component_name].mean().item(),
                        smoothing_hint=False,
                    )

            storage = get_event_storage()

            # Pull off necessary GT information
            # let lowercase->2D and uppercase->3D
            # [x, y, Z, W, H, L] 
            gt_2d = gt_boxes3D[:, :2]
            gt_z = gt_boxes3D[:, 2]
            gt_dims = gt_boxes3D[:, 3:6]

            # this box may have been mirrored and scaled so
            # we need to recompute XYZ in 3D by backprojecting.
            gt_x3d = gt_z * (gt_2d[:, 0] - Ks_scaled_per_box[:, 0, 2])/Ks_scaled_per_box[:, 0, 0]
            gt_y3d = gt_z * (gt_2d[:, 1] - Ks_scaled_per_box[:, 1, 2])/Ks_scaled_per_box[:, 1, 1]
            gt_3d = torch.stack((gt_x3d, gt_y3d, gt_z)).T

            # put together the GT boxes
            gt_box3d = torch.cat((gt_3d, gt_dims), dim=1)

            # These are the corners which will be the target for all losses!!
            gt_corners = util.get_cuboid_verts_faces(gt_box3d, gt_poses)[0]

            if (
                self.use_differentiable_renderer
                and self.soft_renderer is not None
                and gt_render_masks is not None
                and prompt_depth is not None
            ):
                pred_x3d_render = cube_z * (
                    cube_x - Ks_scaled_per_box[:, 0, 2]
                ) / Ks_scaled_per_box[:, 0, 0]
                pred_y3d_render = cube_z * (
                    cube_y - Ks_scaled_per_box[:, 1, 2]
                ) / Ks_scaled_per_box[:, 1, 1]
                pred_box3d_render = torch.cat(
                    (
                        torch.stack(
                            (pred_x3d_render, pred_y3d_render, cube_z),
                            dim=1,
                        ),
                        cube_dims,
                    ),
                    dim=1,
                )
                pred_corners_render = util.get_cuboid_verts_faces(
                    pred_box3d_render,
                    cube_pose,
                )[0]
                render_boxes = torch.cat(
                    [boxes.tensor for boxes in depth_target_boxes],
                    dim=0,
                ).to(cube_z.device)
                rendered_silhouette, rendered_depth = self.soft_renderer(
                    pred_corners_render,
                    Ks_scaled_per_box,
                    render_boxes,
                )
                target_silhouette = F.interpolate(
                    gt_render_masks[:, None].float(),
                    size=(self.render_size, self.render_size),
                    mode="nearest",
                )[:, 0]
                silhouette_intersection = (
                    rendered_silhouette * target_silhouette
                ).sum(dim=(1, 2))
                silhouette_denominator = (
                    rendered_silhouette.sum(dim=(1, 2))
                    + target_silhouette.sum(dim=(1, 2))
                ).clamp(min=1.0)
                loss_render_silhouette = (
                    1.0
                    - (2.0 * silhouette_intersection + 1.0)
                    / (silhouette_denominator + 1.0)
                )
                if self.use_factorized_pseudo_weight:
                    loss_render_silhouette = (
                        loss_render_silhouette
                        * gt_factor_weights["dims"].to(loss_render_silhouette.device)
                    )
                losses[prefix + "loss_render_silhouette"] = (
                    self.safely_reduce_losses(loss_render_silhouette)
                    * self.loss_w_render_silhouette
                    * self.loss_w_3d
                )

                observed_depth_roi = roi_align_map(
                    prompt_depth.tensor.to(cube_z.device),
                    depth_target_boxes,
                    self.render_size,
                    spatial_scale=1.0,
                )[:, 0]
                valid_depth = (
                    torch.isfinite(observed_depth_roi)
                    & (observed_depth_roi > 0.05)
                    & (target_silhouette > 0.5)
                )
                depth_residual = F.smooth_l1_loss(
                    torch.log(rendered_depth.clamp(0.05, 80.0)),
                    torch.log(observed_depth_roi.clamp(0.05, 80.0)),
                    reduction="none",
                    beta=0.05,
                )
                depth_weight = (
                    valid_depth.float() * rendered_silhouette.detach()
                )
                depth_count = depth_weight.sum(dim=(1, 2)).clamp(min=1.0)
                loss_render_depth = (
                    depth_residual * depth_weight
                ).sum(dim=(1, 2)) / depth_count
                if self.use_factorized_pseudo_weight:
                    loss_render_depth = (
                        loss_render_depth
                        * gt_factor_weights["z"].to(loss_render_depth.device)
                    )
                losses[prefix + "loss_render_depth"] = (
                    self.safely_reduce_losses(loss_render_depth)
                    * self.loss_w_render_depth
                    * self.loss_w_3d
                )
                storage.put_scalar(
                    prefix + "render_valid_depth",
                    valid_depth.float().mean().item(),
                    smoothing_hint=False,
                )

            # project GT corners
            gt_proj_boxes = torch.bmm(Ks_scaled_per_box, gt_corners.transpose(1,2))
            gt_proj_boxes /= gt_proj_boxes[:, -1, :].clone().unsqueeze(1)

            gt_proj_x1 = gt_proj_boxes[:, 0, :].min(1)[0]
            gt_proj_y1 = gt_proj_boxes[:, 1, :].min(1)[0]
            gt_proj_x2 = gt_proj_boxes[:, 0, :].max(1)[0]
            gt_proj_y2 = gt_proj_boxes[:, 1, :].max(1)[0]

            gt_widths = gt_proj_x2 - gt_proj_x1
            gt_heights = gt_proj_y2 - gt_proj_y1
            gt_x = gt_proj_x1 + 0.5 * gt_widths
            gt_y = gt_proj_y1 + 0.5 * gt_heights

            gt_proj_boxes = torch.stack((gt_proj_x1, gt_proj_y1, gt_proj_x2, gt_proj_y2), dim=1)
            
            if self.disentangled_loss:
                '''
                Disentangled loss compares each varaible group to the 
                cuboid corners, which is generally more robust to hyperparams.
                '''
                    
                # compute disentangled Z corners
                cube_dis_x3d_from_z = cube_z * (gt_2d[:, 0] - Ks_scaled_per_box[:, 0, 2])/Ks_scaled_per_box[:, 0, 0]
                cube_dis_y3d_from_z = cube_z * (gt_2d[:, 1] - Ks_scaled_per_box[:, 1, 2])/Ks_scaled_per_box[:, 1, 1]
                cube_dis_z = torch.cat((torch.stack((cube_dis_x3d_from_z, cube_dis_y3d_from_z, cube_z)).T, gt_dims), dim=1)
                dis_z_corners = util.get_cuboid_verts_faces(cube_dis_z, gt_poses)[0]
                
                # compute disentangled XY corners
                cube_dis_x3d = gt_z * (cube_x - Ks_scaled_per_box[:, 0, 2])/Ks_scaled_per_box[:, 0, 0]
                cube_dis_y3d = gt_z * (cube_y - Ks_scaled_per_box[:, 1, 2])/Ks_scaled_per_box[:, 1, 1]
                cube_dis_XY = torch.cat((torch.stack((cube_dis_x3d, cube_dis_y3d, gt_z)).T, gt_dims), dim=1)
                dis_XY_corners = util.get_cuboid_verts_faces(cube_dis_XY, gt_poses)[0]
                loss_xy = self.l1_loss(dis_XY_corners, gt_corners).contiguous().view(n, -1).mean(dim=1)
                    
                # Pose
                dis_pose_corners = util.get_cuboid_verts_faces(gt_box3d, cube_pose)[0]
                
                # Dims
                dis_dims_corners = util.get_cuboid_verts_faces(torch.cat((gt_3d, cube_dims), dim=1), gt_poses)[0]

                # Loss dims
                loss_dims = self.l1_loss(dis_dims_corners, gt_corners).contiguous().view(n, -1).mean(dim=1)

                # Loss z
                loss_z = self.l1_loss(dis_z_corners, gt_corners).contiguous().view(n, -1).mean(dim=1)
    
                # Rotation uses chamfer or l1 like others
                if self.chamfer_pose:
                    loss_pose = self.chamfer_loss(dis_pose_corners, gt_corners)

                else:
                    loss_pose = self.l1_loss(dis_pose_corners, gt_corners).contiguous().view(n, -1).mean(dim=1)
                
            # Non-disentangled training losses
            else:
                '''
                These loss functions are fairly arbitrarily designed. 
                Generally, they are in some normalized space but there
                are many alternative implementations for most functions.
                '''

                # XY
                gt_deltas = (gt_2d.clone() - torch.cat((src_ctr_x.unsqueeze(1), src_ctr_y.unsqueeze(1)), dim=1)) \
                            / torch.cat((src_widths.unsqueeze(1), src_heights.unsqueeze(1)), dim=1)
                
                loss_xy = self.l1_loss(cube_2d_deltas, gt_deltas).mean(1) 

                # Dims
                if self.dims_priors_enabled:
                    cube_dims_gt_normspace = torch.log(gt_dims/prior_dims)
                    loss_dims = self.l1_loss(cube_dims_norm, cube_dims_gt_normspace).mean(1) 

                else:
                    loss_dims = self.l1_loss(cube_dims_norm, torch.log(gt_dims)).mean(1)
                
                # Pose
                try:
                    if self.allocentric_pose:
                        gt_poses_allocentric = util.R_to_allocentric(Ks_scaled_per_box, gt_poses, u=cube_x.detach(), v=cube_y.detach())
                        loss_pose = 1-so3_relative_angle(cube_pose_allocentric, gt_poses_allocentric, eps=0.1, cos_angle=True)
                    else:
                        loss_pose = 1-so3_relative_angle(cube_pose, gt_poses, eps=0.1, cos_angle=True)
                
                # Can fail with bad EPS values/instability
                except:
                    loss_pose = None

                if self.z_type == 'direct':
                    loss_z = self.l1_loss(cube_z, gt_z)

                elif self.z_type == 'sigmoid':
                    loss_z = self.l1_loss(cube_z_norm, (gt_z * real_to_virtual / 100).clip(0, 1))
                    
                elif self.z_type == 'log':
                    loss_z = self.l1_loss(cube_z_norm, torch.log((gt_z * real_to_virtual).clip(0.01)))

                elif self.z_type == 'clusters':
                    loss_z = self.l1_loss(cube_z_norm, (((gt_z * real_to_virtual) - z_means)/(z_stds)))
            
            total_3D_loss_for_reporting = loss_dims*self.loss_w_dims

            if not loss_pose is None:
                total_3D_loss_for_reporting += loss_pose*self.loss_w_pose

            if not cube_2d_deltas is None:
                total_3D_loss_for_reporting += loss_xy*self.loss_w_xy

            if not loss_z is None:
                total_3D_loss_for_reporting += loss_z*self.loss_w_z
            
            # reporting does not need gradients
            total_3D_loss_for_reporting = total_3D_loss_for_reporting.detach()

            if self.loss_w_joint > 0:
                '''
                If we are using joint [entangled] loss, then we also need to pair all 
                predictions together and compute a chamfer or l1 loss vs. cube corners.
                '''
                
                cube_dis_x3d_from_z = cube_z * (cube_x - Ks_scaled_per_box[:, 0, 2])/Ks_scaled_per_box[:, 0, 0]
                cube_dis_y3d_from_z = cube_z * (cube_y - Ks_scaled_per_box[:, 1, 2])/Ks_scaled_per_box[:, 1, 1]
                cube_dis_z = torch.cat((torch.stack((cube_dis_x3d_from_z, cube_dis_y3d_from_z, cube_z)).T, cube_dims), dim=1)
                dis_z_corners_joint = util.get_cuboid_verts_faces(cube_dis_z, cube_pose)[0]
                
                if self.chamfer_pose and self.disentangled_loss:
                    loss_joint = self.chamfer_loss(dis_z_corners_joint, gt_corners)

                else:
                    loss_joint = self.l1_loss(dis_z_corners_joint, gt_corners).contiguous().view(n, -1).mean(dim=1)

                valid_joint = loss_joint < np.inf
                total_3D_loss_for_reporting += (loss_joint*self.loss_w_joint).detach()

            # compute errors for tracking purposes
            z_error = (cube_z - gt_z).detach().abs()
            dims_error = (cube_dims - gt_dims).detach().abs()
            xy_error = (cube_xy - gt_2d).detach().abs()

            storage.put_scalar(prefix + 'z_error', z_error.mean().item(), smoothing_hint=False)
            storage.put_scalar(prefix + 'dims_error', dims_error.mean().item(), smoothing_hint=False)
            storage.put_scalar(prefix + 'xy_error', xy_error.mean().item(), smoothing_hint=False)
            storage.put_scalar(prefix + 'z_close', (z_error<0.20).float().mean().item(), smoothing_hint=False)
            
            storage.put_scalar(prefix + 'total_3D_loss', self.loss_w_3d * self.safely_reduce_losses(total_3D_loss_for_reporting), smoothing_hint=False)

            if self.inverse_z_weight:
                '''
                Weights all losses to prioritize close up boxes.
                '''

                gt_z = gt_boxes3D[:, 2]

                inverse_z_w = 1/torch.log(gt_z.clip(E_CONSTANT))
                
                loss_dims *= inverse_z_w

                # scale based on log, but clip at e
                if not cube_2d_deltas is None:
                    loss_xy *= inverse_z_w
                
                if loss_z is not None:
                    loss_z *= inverse_z_w

                if loss_pose is not None:
                    loss_pose *= inverse_z_w
    
                if self.loss_w_joint > 0:
                    loss_joint *= inverse_z_w

            if self.use_confidence > 0:
                
                uncert_sf = SQRT_2_CONSTANT * torch.exp(-cube_uncert)
                
                loss_dims *= uncert_sf

                if not cube_2d_deltas is None:
                    loss_xy *= uncert_sf

                if not loss_z is None:
                    loss_z *= uncert_sf

                if loss_pose is not None:
                    loss_pose *= uncert_sf
    
                if self.loss_w_joint > 0:
                    loss_joint *= uncert_sf

                losses.update({prefix + 'uncert': self.use_confidence*self.safely_reduce_losses(cube_uncert.clone())})
                storage.put_scalar(prefix + 'conf', torch.exp(-cube_uncert).mean().item(), smoothing_hint=False)

            if self.use_pseudo_weight and gt_pseudo_weight.numel() > 0:
                gt_pseudo_weight = gt_pseudo_weight.to(loss_dims.device).clamp(0.05, 1.0)
                loss_dims = loss_dims * gt_pseudo_weight

                if not cube_2d_deltas is None:
                    loss_xy = loss_xy * gt_pseudo_weight

                if loss_z is not None:
                    loss_z = loss_z * gt_pseudo_weight

                if loss_pose is not None:
                    loss_pose = loss_pose * gt_pseudo_weight

                if self.loss_w_joint > 0:
                    loss_joint = loss_joint * gt_pseudo_weight

                storage.put_scalar(prefix + 'pseudo_weight', gt_pseudo_weight.mean().item(), smoothing_hint=False)

            if self.use_factorized_pseudo_weight and gt_factor_weights["dims"].numel() > 0:
                factor_weights = {
                    name: values.to(loss_dims.device).clamp(0.05, 1.0)
                    for name, values in gt_factor_weights.items()
                }
                loss_dims = loss_dims * factor_weights["dims"]
                if cube_2d_deltas is not None:
                    loss_xy = loss_xy * factor_weights["xy"]
                if loss_z is not None:
                    loss_z = loss_z * factor_weights["z"]
                if loss_pose is not None:
                    loss_pose = loss_pose * factor_weights["pose"]
                if self.loss_w_joint > 0:
                    loss_joint = loss_joint * factor_weights["joint"]
                for factor_name, factor_weight in factor_weights.items():
                    storage.put_scalar(
                        prefix + f'pseudo_weight_{factor_name}',
                        factor_weight.mean().item(),
                        smoothing_hint=False,
                    )

            if (
                self.use_projected_corner_depth_aux
                and (
                    self.loss_w_projected_corner_2d > 0
                    or self.loss_w_projected_corner_depth > 0
                )
            ):
                pred_x3d_corner = cube_z * (
                    cube_x - Ks_scaled_per_box[:, 0, 2]
                ) / Ks_scaled_per_box[:, 0, 0]
                pred_y3d_corner = cube_z * (
                    cube_y - Ks_scaled_per_box[:, 1, 2]
                ) / Ks_scaled_per_box[:, 1, 1]
                pred_box3d_corner = torch.cat(
                    (
                        torch.stack(
                            (pred_x3d_corner, pred_y3d_corner, cube_z),
                            dim=1,
                        ),
                        cube_dims,
                    ),
                    dim=1,
                )
                pred_corners_aux = util.get_cuboid_verts_faces(
                    pred_box3d_corner,
                    cube_pose,
                )[0]
                pred_corner_depth = pred_corners_aux[:, :, 2]
                gt_corner_depth = gt_corners[:, :, 2]
                valid_corners = (
                    torch.isfinite(pred_corners_aux).all(dim=2)
                    & torch.isfinite(gt_corners).all(dim=2)
                    & (pred_corner_depth > 0.05)
                    & (gt_corner_depth > 0.05)
                )
                valid_counts = valid_corners.float().sum(dim=1)
                valid_boxes = valid_counts >= 4.0

                if valid_boxes.any():
                    pred_proj_corners = torch.bmm(
                        Ks_scaled_per_box,
                        pred_corners_aux.transpose(1, 2),
                    ).transpose(1, 2)
                    gt_proj_corners = torch.bmm(
                        Ks_scaled_per_box,
                        gt_corners.transpose(1, 2),
                    ).transpose(1, 2)
                    pred_uv = (
                        pred_proj_corners[:, :, :2]
                        / pred_proj_corners[:, :, 2:3].clamp(min=1e-4)
                    )
                    gt_uv = (
                        gt_proj_corners[:, :, :2]
                        / gt_proj_corners[:, :, 2:3].clamp(min=1e-4)
                    )
                    corner_scale = src_scales.clamp(min=1.0).view(-1, 1, 1)
                    corner_valid_weight = valid_corners.float()
                    corner_2d_residual = F.smooth_l1_loss(
                        pred_uv / corner_scale,
                        gt_uv / corner_scale,
                        reduction="none",
                        beta=0.02,
                    ).sum(dim=2)
                    corner_2d_residual = torch.where(
                        valid_corners,
                        corner_2d_residual,
                        torch.zeros_like(corner_2d_residual),
                    )
                    loss_projected_corner_2d = (
                        corner_2d_residual * corner_valid_weight
                    ).sum(dim=1) / valid_counts.clamp(min=1.0)
                    loss_projected_corner_2d = loss_projected_corner_2d.clamp(
                        max=self.projected_corner_max_loss
                    )

                    corner_depth_residual = F.smooth_l1_loss(
                        torch.log(pred_corner_depth.clamp(0.05, 80.0)),
                        torch.log(gt_corner_depth.clamp(0.05, 80.0)),
                        reduction="none",
                        beta=0.05,
                    )
                    corner_depth_residual = torch.where(
                        valid_corners,
                        corner_depth_residual,
                        torch.zeros_like(corner_depth_residual),
                    )
                    loss_projected_corner_depth = (
                        corner_depth_residual * corner_valid_weight
                    ).sum(dim=1) / valid_counts.clamp(min=1.0)
                    loss_projected_corner_depth = (
                        loss_projected_corner_depth.clamp(
                            max=self.projected_corner_max_loss
                        )
                    )

                    if self.use_factorized_pseudo_weight:
                        corner_2d_weight = gt_factor_weights["joint"].to(
                            cube_z.device
                        ).clamp(0.05, 1.0)
                        corner_depth_weight = torch.minimum(
                            gt_factor_weights["z"].to(cube_z.device),
                            gt_factor_weights["dims"].to(cube_z.device),
                        ).clamp(0.05, 1.0)
                    elif self.use_pseudo_weight and gt_pseudo_weight.numel() == n:
                        corner_2d_weight = gt_pseudo_weight.to(cube_z.device).clamp(
                            0.05,
                            1.0,
                        )
                        corner_depth_weight = corner_2d_weight
                    else:
                        corner_2d_weight = torch.ones_like(cube_z)
                        corner_depth_weight = torch.ones_like(cube_z)

                    corner_aux_quality = gt_corner_aux_quality.to(cube_z.device)
                    corner_2d_weight = (
                        corner_2d_weight * corner_aux_quality
                    ).clamp(0.05, 1.0)
                    corner_depth_weight = (
                        corner_depth_weight * corner_aux_quality
                    ).clamp(0.05, 1.0)

                    if self.loss_w_projected_corner_2d > 0:
                        loss_2d_valid = (
                            loss_projected_corner_2d[valid_boxes]
                            * corner_2d_weight[valid_boxes]
                        )
                        losses[prefix + "loss_projected_corner_2d"] = (
                            self.safely_reduce_losses(loss_2d_valid)
                            * self.loss_w_projected_corner_2d
                            * self.loss_w_3d
                        )

                    if self.loss_w_projected_corner_depth > 0:
                        loss_depth_valid = (
                            loss_projected_corner_depth[valid_boxes]
                            * corner_depth_weight[valid_boxes]
                        )
                        losses[prefix + "loss_projected_corner_depth"] = (
                            self.safely_reduce_losses(loss_depth_valid)
                            * self.loss_w_projected_corner_depth
                            * self.loss_w_3d
                        )

                    storage.put_scalar(
                        prefix + "projected_corner_valid",
                        valid_boxes.float().mean().item(),
                        smoothing_hint=False,
                    )
                    storage.put_scalar(
                        prefix + "projected_corner_2d_raw",
                        loss_projected_corner_2d[valid_boxes].mean().item(),
                        smoothing_hint=False,
                    )
                    storage.put_scalar(
                        prefix + "projected_corner_depth_raw",
                        loss_projected_corner_depth[valid_boxes].mean().item(),
                        smoothing_hint=False,
                    )
                    storage.put_scalar(
                        prefix + "projected_corner_aux_quality",
                        corner_aux_quality[valid_boxes].mean().item(),
                        smoothing_hint=False,
                    )

            if (
                self.use_depth_consistency_loss
                and self.loss_w_depth_consistency > 0
                and prompt_depth is not None
                and loss_z is not None
            ):
                depth_masks_per_image = None
                if (
                    self.rsh_use_depth_guidance
                    and rsh_depth_masks_flat is not None
                    and rsh_depth_masks_flat.shape[0] == cube_z.numel()
                ):
                    depth_masks_per_image = list(
                        rsh_depth_masks_flat.split(num_boxes_per_image)
                    )
                depth_targets, depth_valid = self.depth_targets_from_boxes(
                    prompt_depth.tensor.to(cube_z.device),
                    depth_target_boxes,
                    device=cube_z.device,
                    masks_per_image=depth_masks_per_image,
                )
                if depth_valid.numel() == cube_z.numel() and depth_valid.any():
                    depth_targets = depth_targets[depth_valid].clamp(0.05, 80.0)
                    if self.depth_consistency_mode == "front_surface":
                        pred_x3d = cube_z * (
                            cube_x - Ks_scaled_per_box[:, 0, 2]
                        ) / Ks_scaled_per_box[:, 0, 0]
                        pred_y3d = cube_z * (
                            cube_y - Ks_scaled_per_box[:, 1, 2]
                        ) / Ks_scaled_per_box[:, 1, 1]
                        pred_boxes_3d = torch.cat(
                            (
                                torch.stack((pred_x3d, pred_y3d, cube_z)).T,
                                cube_dims,
                            ),
                            dim=1,
                        )
                        pred_corners = util.get_cuboid_verts_faces(
                            pred_boxes_3d,
                            cube_pose,
                        )[0]
                        pred_depth_all = pred_corners[:, :, 2].min(dim=1)[0]
                    else:
                        pred_depth_all = cube_z
                    pred_depth = pred_depth_all[depth_valid].clamp(0.05, 80.0)
                    loss_depth_consistency = F.smooth_l1_loss(
                        torch.log(pred_depth),
                        torch.log(depth_targets),
                        reduction='none',
                        beta=0.05,
                    )
                    if gt_pseudo_weight.numel() == cube_z.numel():
                        depth_weight = gt_pseudo_weight.to(cube_z.device)[depth_valid].clamp(0.25, 1.0)
                        loss_depth_consistency = loss_depth_consistency * (0.5 + 0.5 * depth_weight)
                    losses.update({
                        prefix + 'loss_depth_consistency': (
                            self.safely_reduce_losses(loss_depth_consistency)
                            * self.loss_w_depth_consistency
                            * self.loss_w_3d
                        )
                    })
                    storage.put_scalar(
                        prefix + 'depth_consistency_valid',
                        depth_valid.float().mean().item(),
                        smoothing_hint=False,
                    )

            # store per batch loss stats temporarily
            self.batch_losses = [batch_losses.mean().item() for batch_losses in total_3D_loss_for_reporting.split(num_boxes_per_image)]
            
            if self.loss_w_dims > 0:
                losses.update({
                    prefix + 'loss_dims': self.safely_reduce_losses(loss_dims) * self.loss_w_dims * self.loss_w_3d,
                })

            if not cube_2d_deltas is None:
                losses.update({
                    prefix + 'loss_xy': self.safely_reduce_losses(loss_xy) * self.loss_w_xy * self.loss_w_3d,
                })

            if not loss_z is None:
                losses.update({
                    prefix + 'loss_z': self.safely_reduce_losses(loss_z) * self.loss_w_z * self.loss_w_3d,
                })

            if loss_pose is not None:
                
                losses.update({
                    prefix + 'loss_pose': self.safely_reduce_losses(loss_pose) * self.loss_w_pose * self.loss_w_3d, 
                })

            if self.loss_w_joint > 0:
                if valid_joint.any():
                    losses.update({prefix + 'loss_joint': self.safely_reduce_losses(loss_joint[valid_joint]) * self.loss_w_joint * self.loss_w_3d})

            
        '''
        Inference
        '''
        if len(cube_z.shape) == 0:
            cube_z = cube_z.unsqueeze(0)

        # inference
        cube_x3d = cube_z * (cube_x - Ks_scaled_per_box[:, 0, 2])/Ks_scaled_per_box[:, 0, 0]
        cube_y3d = cube_z * (cube_y - Ks_scaled_per_box[:, 1, 2])/Ks_scaled_per_box[:, 1, 1]
        cube_3D = torch.cat((torch.stack((cube_x3d, cube_y3d, cube_z)).T, cube_dims, cube_xy*im_ratios_per_box.unsqueeze(1)), dim=1)

        if self.use_confidence:
            cube_conf = torch.exp(-cube_uncert)
            cube_3D = torch.cat((cube_3D, cube_conf.unsqueeze(1)), dim=1)

        # convert the predictions to intances per image
        cube_3D = cube_3D.split(num_boxes_per_image)
        cube_pose = cube_pose.split(num_boxes_per_image)
        box_classes = box_classes.split(num_boxes_per_image)
        geometry_confidence_split = (
            geometry_confidence.split(num_boxes_per_image)
            if geometry_confidence is not None
            else [None] * len(num_boxes_per_image)
        )
        geometry_cost_split = (
            geometry_cost.split(num_boxes_per_image)
            if geometry_cost is not None
            else [None] * len(num_boxes_per_image)
        )
        rsh_mask_split = (
            rsh_mask_prob[:, 0].detach().split(num_boxes_per_image)
            if rsh_mask_prob is not None
            else [None] * len(num_boxes_per_image)
        )
        
        pred_instances = None
        
        pred_instances = instances if not self.training else \
            [Instances(image_size) for image_size in im_current_dims]

        for cube_3D_i, cube_pose_i, geometry_confidence_i, geometry_cost_i, rsh_mask_i, instances_i, K, im_dim, im_scale_ratio, box_classes_i, pred_boxes_i in \
            zip(cube_3D, cube_pose, geometry_confidence_split, geometry_cost_split, rsh_mask_split, pred_instances, Ks, im_current_dims, im_scales_ratio, box_classes, pred_boxes):
            
            # merge scores if they already exist
            if hasattr(instances_i, 'scores'):
                instances_i.scores = (instances_i.scores * cube_3D_i[:, -1])**(1/2)
            
            # assign scores if none are present
            else:
                instances_i.scores = cube_3D_i[:, -1]
            
            # assign box classes if none exist
            if not hasattr(instances_i, 'pred_classes'):
                instances_i.pred_classes = box_classes_i

            # assign predicted boxes if none exist    
            if not hasattr(instances_i, 'pred_boxes'):
                instances_i.pred_boxes = pred_boxes_i

            instances_i.pred_bbox3D = util.get_cuboid_verts_faces(cube_3D_i[:, :6], cube_pose_i)[0]
            instances_i.pred_center_cam = cube_3D_i[:, :3]
            instances_i.pred_center_2D = cube_3D_i[:, 6:8]
            instances_i.pred_dimensions = cube_3D_i[:, 3:6]
            instances_i.pred_pose = cube_pose_i
            if geometry_confidence_i is not None:
                instances_i.pred_geometry_confidence = geometry_confidence_i
            if geometry_cost_i is not None:
                instances_i.pred_geometry_score = torch.exp(
                    -geometry_cost_i.detach()
                ).clamp(0.0, 1.0)
            if rsh_mask_i is not None:
                instances_i.pred_region_masks = rsh_mask_i

        if self.training:
            return pred_instances, losses
        else:
            return pred_instances

    def _sample_proposals(
        self, matched_idxs: torch.Tensor, matched_labels: torch.Tensor, gt_classes: torch.Tensor, matched_ious=None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Based on the matching between N proposals and M groundtruth,
        sample the proposals and set their classification labels.
        Args:
            matched_idxs (Tensor): a vector of length N, each is the best-matched
                gt index in [0, M) for each proposal.
            matched_labels (Tensor): a vector of length N, the matcher's label
                (one of cfg.MODEL.ROI_HEADS.IOU_LABELS) for each proposal.
            gt_classes (Tensor): a vector of length M.
        Returns:
            Tensor: a vector of indices of sampled proposals. Each is in [0, N).
            Tensor: a vector of the same length, the classification label for
                each sampled proposal. Each sample is labeled as either a category in
                [0, num_classes) or the background (num_classes).
        """
        has_gt = gt_classes.numel() > 0
        # Get the corresponding GT for each proposal
        if has_gt:
            gt_classes = gt_classes[matched_idxs]
            # Label unmatched proposals (0 label from matcher) as background (label=num_classes)
            gt_classes[matched_labels == 0] = self.num_classes
            # Label ignore proposals (-1 label)
            gt_classes[matched_labels == -1] = -1
        else:
            gt_classes = torch.zeros_like(matched_idxs) + self.num_classes

        sampled_fg_idxs, sampled_bg_idxs = subsample_labels(
            gt_classes, self.batch_size_per_image, self.positive_fraction, self.num_classes, matched_ious=matched_ious
        )

        sampled_idxs = torch.cat([sampled_fg_idxs, sampled_bg_idxs], dim=0)
        return sampled_idxs, gt_classes[sampled_idxs]
    
    @torch.no_grad()
    def label_and_sample_proposals(self, proposals: List[Instances], targets: List[Instances]) -> List[Instances]:
        
        #separate valid and ignore gts
        targets_ign = [target[target.gt_classes < 0] for target in targets]
        targets = [target[target.gt_classes >= 0] for target in targets]
        
        if self.proposal_append_gt:
            proposals = add_ground_truth_to_proposals(targets, proposals)

        proposals_with_gt = []

        num_fg_samples = []
        num_bg_samples = []

        for proposals_per_image, targets_per_image, targets_ign_per_image in zip(proposals, targets, targets_ign):
            
            has_gt = len(targets_per_image) > 0
            
            match_quality_matrix = pairwise_iou(targets_per_image.gt_boxes, proposals_per_image.proposal_boxes)
            matched_idxs, matched_labels = self.proposal_matcher(match_quality_matrix)
            
            try:
                if len(targets_ign_per_image) > 0:

                    # compute the quality matrix, only on subset of background
                    background_inds = (matched_labels == 0).nonzero().squeeze()

                    # determine the boxes inside ignore regions with sufficient threshold
                    if background_inds.numel() > 1:
                        match_quality_matrix_ign = pairwise_ioa(targets_ign_per_image.gt_boxes, proposals_per_image.proposal_boxes[background_inds])
                        matched_labels[background_inds[match_quality_matrix_ign.max(0)[0] >= self.ignore_thresh]] = -1
                    
                        del match_quality_matrix_ign
            except:
                pass
            
            gt_arange = torch.arange(match_quality_matrix.shape[1]).to(matched_idxs.device)
            matched_ious = match_quality_matrix[matched_idxs, gt_arange]
            sampled_idxs, gt_classes = self._sample_proposals(matched_idxs, matched_labels, targets_per_image.gt_classes, matched_ious=matched_ious)

            # Set target attributes of the sampled proposals:
            proposals_per_image = proposals_per_image[sampled_idxs]
            proposals_per_image.gt_classes = gt_classes

            if has_gt:
                sampled_targets = matched_idxs[sampled_idxs]
                # We index all the attributes of targets that start with "gt_"
                # and have not been added to proposals yet (="gt_classes").
                # NOTE: here the indexing waste some compute, because heads
                # like masks, keypoints, etc, will filter the proposals again,
                # (by foreground/background, or number of keypoints in the image, etc)
                # so we essentially index the data twice.
                for (trg_name, trg_value) in targets_per_image.get_fields().items():
                    if trg_name.startswith("gt_") and not proposals_per_image.has(trg_name):
                        proposals_per_image.set(trg_name, trg_value[sampled_targets])
            

            num_bg_samples.append((gt_classes == self.num_classes).sum().item())
            num_fg_samples.append(gt_classes.numel() - num_bg_samples[-1])
            proposals_with_gt.append(proposals_per_image)

        # Log the number of fg/bg samples that are selected for training ROI heads
        storage = get_event_storage()
        storage.put_scalar("roi_head/num_fg_samples", np.mean(num_fg_samples))
        storage.put_scalar("roi_head/num_bg_samples", np.mean(num_bg_samples))

        return proposals_with_gt

    def depth_targets_from_boxes(
        self,
        depth,
        boxes_per_image,
        device=None,
        masks_per_image=None,
    ):
        if depth is None or boxes_per_image is None:
            empty = torch.empty(0, device=device)
            return empty, empty.bool()

        if depth.dim() == 3:
            depth = depth.unsqueeze(1)
        if depth.size(1) != 1:
            depth = depth[:, :1]

        depth = depth.float()
        device = device or depth.device
        targets = []
        valid = []
        crop_ratio = min(max(self.depth_consistency_center_crop, 0.05), 1.0)

        with torch.no_grad():
            for img_idx, boxes in enumerate(boxes_per_image):
                if img_idx >= depth.shape[0] or len(boxes) == 0:
                    continue

                depth_i = depth[img_idx, 0]
                height, width = depth_i.shape[-2:]

                masks_i = None
                if masks_per_image is not None and img_idx < len(masks_per_image):
                    masks_i = masks_per_image[img_idx].to(depth_i.device)

                for box_idx, box in enumerate(boxes.tensor.to(depth_i.device)):
                    x1, y1, x2, y2 = box.unbind()

                    if crop_ratio < 1.0:
                        cx = 0.5 * (x1 + x2)
                        cy = 0.5 * (y1 + y2)
                        bw = (x2 - x1).clamp(min=1.0) * crop_ratio
                        bh = (y2 - y1).clamp(min=1.0) * crop_ratio
                        x1 = cx - 0.5 * bw
                        x2 = cx + 0.5 * bw
                        y1 = cy - 0.5 * bh
                        y2 = cy + 0.5 * bh

                    xi1 = int(torch.floor(x1).clamp(0, max(width - 1, 0)).item())
                    yi1 = int(torch.floor(y1).clamp(0, max(height - 1, 0)).item())
                    xi2 = int(torch.ceil(x2).clamp(1, width).item())
                    yi2 = int(torch.ceil(y2).clamp(1, height).item())

                    if xi2 <= xi1 or yi2 <= yi1:
                        targets.append(depth_i.new_tensor(0.0))
                        valid.append(False)
                        continue

                    patch = depth_i[yi1:yi2, xi1:xi2]
                    full_values = patch[torch.isfinite(patch) & (patch > 0.05)]
                    values = full_values
                    if masks_i is not None and box_idx < masks_i.shape[0]:
                        mask = masks_i[box_idx].float()
                        if mask.dim() == 3:
                            mask = mask[0]
                        mask = F.interpolate(
                            mask[None, None],
                            size=patch.shape[-2:],
                            mode="bilinear",
                            align_corners=False,
                        )[0, 0]
                        mask_valid = mask > self.rsh_depth_mask_threshold
                        masked_values = patch[
                            mask_valid
                            & torch.isfinite(patch)
                            & (patch > 0.05)
                        ]
                        if masked_values.numel() >= self.depth_consistency_min_pixels:
                            values = masked_values

                    if values.numel() < self.depth_consistency_min_pixels:
                        targets.append(depth_i.new_tensor(0.0))
                        valid.append(False)
                    else:
                        if self.depth_consistency_mode == "front_surface":
                            quantile = min(
                                max(self.depth_consistency_percentile, 0.05),
                                0.50,
                            )
                            targets.append(torch.quantile(values, quantile))
                        else:
                            targets.append(values.median())
                        valid.append(True)

        if len(targets) == 0:
            empty = torch.empty(0, device=device)
            return empty, empty.bool()

        return (
            torch.stack(targets).to(device),
            torch.tensor(valid, dtype=torch.bool, device=device),
        )


    def safely_reduce_losses(self, loss):

        valid = (~(loss.isinf())) & (~(loss.isnan()))

        if valid.any():
            return loss[valid].mean()
        else:
            # no valid losses, simply zero out
            return loss.mean()*0.0
