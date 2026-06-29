# Copyright (c) Meta Platforms, Inc. and affiliates
from detectron2.utils.registry import Registry
from typing import Dict
from detectron2.layers import ShapeSpec
from torch import nn
import torch
import numpy as np
import fvcore.nn.weight_init as weight_init
from detectron2.utils.events import get_event_storage

from pytorch3d.transforms.rotation_conversions import _copysign
from pytorch3d.transforms import (
    rotation_6d_to_matrix, 
    euler_angles_to_matrix, 
    quaternion_to_matrix
)

ROI_CUBE_HEAD_REGISTRY = Registry("ROI_CUBE_HEAD")

@ROI_CUBE_HEAD_REGISTRY.register()
class CubeHead(nn.Module):

    def __init__(self, cfg, input_shape: Dict[str, ShapeSpec]):
        super().__init__()

        #-------------------------------------------
        # Settings
        #-------------------------------------------
        self.num_classes        = cfg.MODEL.ROI_HEADS.NUM_CLASSES
        self.use_conf           = cfg.MODEL.ROI_CUBE_HEAD.USE_CONFIDENCE
        self.z_type             = cfg.MODEL.ROI_CUBE_HEAD.Z_TYPE
        self.pose_type          = cfg.MODEL.ROI_CUBE_HEAD.POSE_TYPE
        self.cluster_bins       = cfg.MODEL.ROI_CUBE_HEAD.CLUSTER_BINS
        self.shared_fc          = cfg.MODEL.ROI_CUBE_HEAD.SHARED_FC
        self.use_zero_init_residual = cfg.MODEL.ROI_CUBE_HEAD.USE_ZERO_INIT_RESIDUAL
        self.residual_scale_xy = cfg.MODEL.ROI_CUBE_HEAD.RESIDUAL_SCALE_XY
        self.residual_scale_z = cfg.MODEL.ROI_CUBE_HEAD.RESIDUAL_SCALE_Z
        self.residual_scale_dims = cfg.MODEL.ROI_CUBE_HEAD.RESIDUAL_SCALE_DIMS
        self.residual_scale_pose = cfg.MODEL.ROI_CUBE_HEAD.RESIDUAL_SCALE_POSE
        self.use_cop_gs = cfg.MODEL.ROI_CUBE_HEAD.USE_COP_GS
        self.cop_gs_hidden_dim = cfg.MODEL.ROI_CUBE_HEAD.COP_GS_HIDDEN_DIM
        self.cop_gs_gate_init_bias = cfg.MODEL.ROI_CUBE_HEAD.COP_GS_GATE_INIT_BIAS
        self.cop_gs_scale_xy = cfg.MODEL.ROI_CUBE_HEAD.COP_GS_SCALE_XY
        self.cop_gs_scale_z = cfg.MODEL.ROI_CUBE_HEAD.COP_GS_SCALE_Z
        self.cop_gs_scale_dims = cfg.MODEL.ROI_CUBE_HEAD.COP_GS_SCALE_DIMS
        self.cop_gs_scale_pose = cfg.MODEL.ROI_CUBE_HEAD.COP_GS_SCALE_POSE

        #-------------------------------------------
        # Feature generator
        #-------------------------------------------

        num_conv = cfg.MODEL.ROI_CUBE_HEAD.NUM_CONV
        conv_dim = cfg.MODEL.ROI_CUBE_HEAD.CONV_DIM
        num_fc = cfg.MODEL.ROI_CUBE_HEAD.NUM_FC
        fc_dim = cfg.MODEL.ROI_CUBE_HEAD.FC_DIM

        conv_dims = [conv_dim] * num_conv
        fc_dims = [fc_dim] * num_fc

        assert len(conv_dims) + len(fc_dims) > 0

        self._output_size = (input_shape.channels, input_shape.height, input_shape.width)

        if self.shared_fc:
            self.feature_generator = nn.Sequential()
        else:
            self.feature_generator_XY = nn.Sequential()
            self.feature_generator_dims = nn.Sequential()
            self.feature_generator_pose = nn.Sequential()
            self.feature_generator_Z = nn.Sequential()

            if self.use_conf:
                self.feature_generator_conf = nn.Sequential()

        # create fully connected layers for Cube Head
        for k, fc_dim in enumerate(fc_dims):
            
            fc_dim_in = int(np.prod(self._output_size))
            
            self._output_size = fc_dim

            if self.shared_fc:
                fc = nn.Linear(fc_dim_in, fc_dim)
                weight_init.c2_xavier_fill(fc)
                self.feature_generator.add_module("fc{}".format(k + 1), fc)
                self.feature_generator.add_module("fc_relu{}".format(k + 1), nn.ReLU())
            
            else:
                
                fc = nn.Linear(fc_dim_in, fc_dim)
                weight_init.c2_xavier_fill(fc)
                self.feature_generator_dims.add_module("fc{}".format(k + 1), fc)
                self.feature_generator_dims.add_module("fc_relu{}".format(k + 1), nn.ReLU())

                fc = nn.Linear(fc_dim_in, fc_dim)
                weight_init.c2_xavier_fill(fc)
                self.feature_generator_XY.add_module("fc{}".format(k + 1), fc)
                self.feature_generator_XY.add_module("fc_relu{}".format(k + 1), nn.ReLU())

                fc = nn.Linear(fc_dim_in, fc_dim)
                weight_init.c2_xavier_fill(fc)
                self.feature_generator_pose.add_module("fc{}".format(k + 1), fc)
                self.feature_generator_pose.add_module("fc_relu{}".format(k + 1), nn.ReLU())

                fc = nn.Linear(fc_dim_in, fc_dim)
                weight_init.c2_xavier_fill(fc)
                self.feature_generator_Z.add_module("fc{}".format(k + 1), fc)
                self.feature_generator_Z.add_module("fc_relu{}".format(k + 1), nn.ReLU())

                if self.use_conf:
                    fc = nn.Linear(fc_dim_in, fc_dim)
                    weight_init.c2_xavier_fill(fc)
                    self.feature_generator_conf.add_module("fc{}".format(k + 1), fc)
                    self.feature_generator_conf.add_module("fc_relu{}".format(k + 1), nn.ReLU())

        #-------------------------------------------
        # 3D outputs
        #-------------------------------------------
        
        # Dimensions in meters (width, height, length)
        self.bbox_3D_dims = nn.Linear(self._output_size, self.num_classes*3)
        nn.init.normal_(self.bbox_3D_dims.weight, std=0.001)
        nn.init.constant_(self.bbox_3D_dims.bias, 0)

        cluster_bins = self.cluster_bins if self.cluster_bins > 1 else 1

        # XY
        self.bbox_3D_center_deltas = nn.Linear(self._output_size, self.num_classes*2)
        nn.init.normal_(self.bbox_3D_center_deltas.weight, std=0.001)
        nn.init.constant_(self.bbox_3D_center_deltas.bias, 0)

        # Pose
        if self.pose_type == '6d':
            pose_dim = 6
            self.bbox_3D_pose = nn.Linear(self._output_size, self.num_classes*6)

        elif self.pose_type == 'quaternion':
            pose_dim = 4
            self.bbox_3D_pose = nn.Linear(self._output_size, self.num_classes*4)

        elif self.pose_type == 'euler':
            pose_dim = 3
            self.bbox_3D_pose = nn.Linear(self._output_size, self.num_classes*3)

        else:
            raise ValueError('Cuboid pose type {} is not recognized'.format(self.pose_type))
        
        nn.init.normal_(self.bbox_3D_pose.weight, std=0.001)
        nn.init.constant_(self.bbox_3D_pose.bias, 0)

        # Z 
        self.bbox_3D_center_depth = nn.Linear(self._output_size, self.num_classes*cluster_bins)
        nn.init.normal_(self.bbox_3D_center_depth.weight, std=0.001)
        nn.init.constant_(self.bbox_3D_center_depth.bias, 0)

        # Optionally, box confidence
        if self.use_conf:
            self.bbox_3D_uncertainty = nn.Linear(self._output_size, self.num_classes*1)
            nn.init.normal_(self.bbox_3D_uncertainty.weight, std=0.001)
            nn.init.constant_(self.bbox_3D_uncertainty.bias, 5)

        if self.use_zero_init_residual:
            self.res_bbox_3D_center_deltas = nn.Linear(self._output_size, self.num_classes*2)
            self.res_bbox_3D_dims = nn.Linear(self._output_size, self.num_classes*3)
            self.res_bbox_3D_pose = nn.Linear(self._output_size, self.num_classes*pose_dim)
            self.res_bbox_3D_center_depth = nn.Linear(self._output_size, self.num_classes*cluster_bins)
            for layer in [
                self.res_bbox_3D_center_deltas,
                self.res_bbox_3D_dims,
                self.res_bbox_3D_pose,
                self.res_bbox_3D_center_depth,
            ]:
                nn.init.constant_(layer.weight, 0.0)
                nn.init.constant_(layer.bias, 0.0)

        if self.use_cop_gs:
            hidden_dim = int(self.cop_gs_hidden_dim)
            if hidden_dim <= 0:
                hidden_dim = int(self._output_size)

            self.cop_xy_context = self._make_cop_context(self._output_size, hidden_dim)
            self.cop_z_context = self._make_cop_context(self._output_size + hidden_dim, hidden_dim)
            self.cop_dims_context = self._make_cop_context(self._output_size + 2 * hidden_dim, hidden_dim)
            self.cop_pose_context = self._make_cop_context(self._output_size + 3 * hidden_dim, hidden_dim)

            self.cop_xy_residual = nn.Linear(hidden_dim, self.num_classes * 2)
            self.cop_z_residual = nn.Linear(hidden_dim, self.num_classes * cluster_bins)
            self.cop_dims_residual = nn.Linear(hidden_dim, self.num_classes * 3)
            self.cop_pose_residual = nn.Linear(hidden_dim, self.num_classes * pose_dim)

            self.cop_xy_gate = nn.Linear(hidden_dim, self.num_classes * 2)
            self.cop_z_gate = nn.Linear(hidden_dim, self.num_classes * cluster_bins)
            self.cop_dims_gate = nn.Linear(hidden_dim, self.num_classes * 3)
            self.cop_pose_gate = nn.Linear(hidden_dim, self.num_classes * pose_dim)

            for layer in [
                self.cop_xy_residual,
                self.cop_z_residual,
                self.cop_dims_residual,
                self.cop_pose_residual,
            ]:
                nn.init.constant_(layer.weight, 0.0)
                nn.init.constant_(layer.bias, 0.0)

            for layer in [
                self.cop_xy_gate,
                self.cop_z_gate,
                self.cop_dims_gate,
                self.cop_pose_gate,
            ]:
                nn.init.constant_(layer.weight, 0.0)
                nn.init.constant_(layer.bias, float(self.cop_gs_gate_init_bias))

    def _make_cop_context(self, input_dim, hidden_dim):
        layer = nn.Linear(input_dim, hidden_dim)
        weight_init.c2_xavier_fill(layer)
        return nn.Sequential(
            layer,
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
        )

    def _apply_cop_gs(
        self,
        features_xy,
        features_z,
        features_dims,
        features_pose,
        box_2d_deltas,
        box_z,
        box_dims,
        box_pose,
    ):
        ctx_xy = self.cop_xy_context(features_xy)
        ctx_z = self.cop_z_context(torch.cat([features_z, ctx_xy], dim=1))
        ctx_dims = self.cop_dims_context(torch.cat([features_dims, ctx_xy, ctx_z], dim=1))
        ctx_pose = self.cop_pose_context(torch.cat([features_pose, ctx_xy, ctx_z, ctx_dims], dim=1))

        gate_xy = torch.sigmoid(self.cop_xy_gate(ctx_xy))
        gate_z = torch.sigmoid(self.cop_z_gate(ctx_z))
        gate_dims = torch.sigmoid(self.cop_dims_gate(ctx_dims))
        gate_pose = torch.sigmoid(self.cop_pose_gate(ctx_pose))

        box_2d_deltas = (
            box_2d_deltas
            + float(self.cop_gs_scale_xy) * gate_xy * self.cop_xy_residual(ctx_xy)
        )
        box_z = (
            box_z
            + float(self.cop_gs_scale_z) * gate_z * self.cop_z_residual(ctx_z)
        )
        box_dims = (
            box_dims
            + float(self.cop_gs_scale_dims) * gate_dims * self.cop_dims_residual(ctx_dims)
        )
        box_pose = (
            box_pose
            + float(self.cop_gs_scale_pose) * gate_pose * self.cop_pose_residual(ctx_pose)
        )

        if self.training:
            try:
                storage = get_event_storage()
                storage.put_scalar("Cube/cop_gs_gate_xy", gate_xy.mean().item(), smoothing_hint=False)
                storage.put_scalar("Cube/cop_gs_gate_z", gate_z.mean().item(), smoothing_hint=False)
                storage.put_scalar("Cube/cop_gs_gate_dims", gate_dims.mean().item(), smoothing_hint=False)
                storage.put_scalar("Cube/cop_gs_gate_pose", gate_pose.mean().item(), smoothing_hint=False)
            except AssertionError:
                pass

        return box_2d_deltas, box_z, box_dims, box_pose

    def forward(self, x):
    
        n = x.shape[0]
        
        box_z = None
        box_uncert = None
        box_2d_deltas = None

        if self.shared_fc:
            features = self.feature_generator(x)
            box_2d_deltas = self.bbox_3D_center_deltas(features)
            box_dims = self.bbox_3D_dims(features)
            box_pose = self.bbox_3D_pose(features)
            box_z = self.bbox_3D_center_depth(features)

            if self.use_zero_init_residual:
                box_2d_deltas = box_2d_deltas + self.residual_scale_xy * self.res_bbox_3D_center_deltas(features)
                box_dims = box_dims + self.residual_scale_dims * self.res_bbox_3D_dims(features)
                box_pose = box_pose + self.residual_scale_pose * self.res_bbox_3D_pose(features)
                box_z = box_z + self.residual_scale_z * self.res_bbox_3D_center_depth(features)

            if self.use_cop_gs:
                box_2d_deltas, box_z, box_dims, box_pose = self._apply_cop_gs(
                    features,
                    features,
                    features,
                    features,
                    box_2d_deltas,
                    box_z,
                    box_dims,
                    box_pose,
                )

            if self.use_conf:
                box_uncert = self.bbox_3D_uncertainty(features).clip(0.01)
        else:

            features_xy = self.feature_generator_XY(x)
            features_dims = self.feature_generator_dims(x)
            features_pose = self.feature_generator_pose(x)
            features_z = self.feature_generator_Z(x)

            box_2d_deltas = self.bbox_3D_center_deltas(features_xy)
            box_dims = self.bbox_3D_dims(features_dims)
            box_pose = self.bbox_3D_pose(features_pose)
            box_z = self.bbox_3D_center_depth(features_z)

            if self.use_zero_init_residual:
                box_2d_deltas = box_2d_deltas + self.residual_scale_xy * self.res_bbox_3D_center_deltas(features_xy)
                box_dims = box_dims + self.residual_scale_dims * self.res_bbox_3D_dims(features_dims)
                box_pose = box_pose + self.residual_scale_pose * self.res_bbox_3D_pose(features_pose)
                box_z = box_z + self.residual_scale_z * self.res_bbox_3D_center_depth(features_z)

            if self.use_cop_gs:
                box_2d_deltas, box_z, box_dims, box_pose = self._apply_cop_gs(
                    features_xy,
                    features_z,
                    features_dims,
                    features_pose,
                    box_2d_deltas,
                    box_z,
                    box_dims,
                    box_pose,
                )

            if self.use_conf:
                box_uncert = self.bbox_3D_uncertainty(self.feature_generator_conf(x)).clip(0.01)

        # Pose
        if self.pose_type == '6d':
            box_pose = rotation_6d_to_matrix(box_pose.view(-1, 6))

        elif self.pose_type == 'quaternion':
            quats = box_pose.view(-1, 4)
            quats_scales = (quats * quats).sum(1)
            quats = quats / _copysign(torch.sqrt(quats_scales), quats[:, 0])[:, None]
            box_pose = quaternion_to_matrix(quats)

        elif self.pose_type == 'euler':
            box_pose = euler_angles_to_matrix(box_pose.view(-1, 3), 'XYZ')

        box_2d_deltas = box_2d_deltas.view(n, self.num_classes, 2)
        box_dims = box_dims.view(n, self.num_classes, 3)
        box_pose = box_pose.view(n, self.num_classes, 3, 3)

        if self.cluster_bins > 1:
            box_z = box_z.view(n, self.cluster_bins, self.num_classes, -1)

        else:
            box_z = box_z.view(n, self.num_classes, -1)
            
        return box_2d_deltas, box_z, box_dims, box_pose, box_uncert


def build_cube_head(cfg, input_shape: Dict[str, ShapeSpec]):
    name = cfg.MODEL.ROI_CUBE_HEAD.NAME
    return ROI_CUBE_HEAD_REGISTRY.get(name)(cfg, input_shape)
