import math
from typing import List, Optional, Sequence, Tuple

import torch
from torch import nn
import torch.nn.functional as F
from torchvision.ops import roi_align


def boxes_to_roi_tensor(boxes: Sequence, device, dtype) -> torch.Tensor:
    rois = []
    for image_index, boxes_i in enumerate(boxes):
        tensor = boxes_i.tensor if hasattr(boxes_i, "tensor") else boxes_i
        if tensor.numel() == 0:
            continue
        batch_column = torch.full(
            (tensor.shape[0], 1),
            float(image_index),
            device=device,
            dtype=dtype,
        )
        rois.append(torch.cat((batch_column, tensor.to(device=device, dtype=dtype)), dim=1))
    if not rois:
        return torch.empty((0, 5), device=device, dtype=dtype)
    return torch.cat(rois, dim=0)


def roi_align_map(
    feature_map: Optional[torch.Tensor],
    boxes: Sequence,
    output_size: int,
    spatial_scale: float = 1.0,
) -> Optional[torch.Tensor]:
    if feature_map is None:
        return None
    rois = boxes_to_roi_tensor(boxes, feature_map.device, feature_map.dtype)
    if rois.numel() == 0:
        return feature_map.new_empty((0, feature_map.shape[1], output_size, output_size))
    return roi_align(
        feature_map,
        rois,
        output_size=(output_size, output_size),
        spatial_scale=float(spatial_scale),
        sampling_ratio=2,
        aligned=True,
    )


def build_ray_map(depth: torch.Tensor, intrinsics: torch.Tensor) -> torch.Tensor:
    if depth.dim() == 3:
        depth = depth.unsqueeze(1)
    batch, _, height, width = depth.shape
    dtype = depth.dtype
    device = depth.device
    ys, xs = torch.meshgrid(
        torch.arange(height, device=device, dtype=dtype),
        torch.arange(width, device=device, dtype=dtype),
        indexing="ij",
    )
    fx = intrinsics[:, 0, 0].view(batch, 1, 1).clamp(min=1e-6)
    fy = intrinsics[:, 1, 1].view(batch, 1, 1).clamp(min=1e-6)
    cx = intrinsics[:, 0, 2].view(batch, 1, 1)
    cy = intrinsics[:, 1, 2].view(batch, 1, 1)
    x = (xs.unsqueeze(0) - cx) / fx
    y = (ys.unsqueeze(0) - cy) / fy
    rays = torch.stack((x, y, torch.ones_like(x)), dim=1)
    return F.normalize(rays, dim=1, eps=1e-6)


@torch.no_grad()
def online_pca_anchor(point_roi: torch.Tensor) -> dict:
    centers = []
    dimensions = []
    yaws = []
    confidences = []
    valid_ratios = []
    anisotropies = []
    for points_map in point_roi:
        points = points_map.permute(1, 2, 0).reshape(-1, 3)
        valid = torch.isfinite(points).all(dim=1) & (points[:, 2] > 0.05)
        valid_ratio = valid.float().mean()
        points = points[valid]
        if points.shape[0] < 6:
            centers.append(points_map.new_tensor([0.0, 0.0, 1.0]))
            dimensions.append(points_map.new_tensor([0.5, 0.5, 0.5]))
            yaws.append(points_map.new_tensor(0.0))
            confidences.append(points_map.new_tensor(0.0))
            valid_ratios.append(valid_ratio)
            anisotropies.append(points_map.new_tensor(0.0))
            continue

        depth = points[:, 2]
        median = depth.median()
        mad = (depth - median).abs().median().clamp(min=0.03)
        depth_keep = (depth - median).abs() <= 3.5 * mad
        filtered = points[depth_keep]
        if filtered.shape[0] < 6:
            filtered = points

        center_robust = filtered.median(dim=0).values
        xz = filtered[:, [0, 2]] - center_robust[[0, 2]]
        covariance = xz.T @ xz / max(filtered.shape[0] - 1, 1)
        eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
        principal = eigenvectors[:, -1]
        yaw = torch.atan2(principal[1], principal[0])
        cosine = torch.cos(yaw)
        sine = torch.sin(yaw)
        rotation = points_map.new_tensor(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        )
        rotation[0, 0] = cosine
        rotation[0, 2] = sine
        rotation[2, 0] = -sine
        rotation[2, 2] = cosine
        local = (filtered - center_robust) @ rotation
        low = torch.quantile(local, 0.05, dim=0)
        high = torch.quantile(local, 0.95, dim=0)
        local_center = 0.5 * (low + high)
        center = center_robust + local_center @ rotation.T
        extent = (high - low).clamp(min=0.03)
        dims_whl = torch.stack((extent[2], extent[1], extent[0]))
        anisotropy = (
            (eigenvalues[-1] - eigenvalues[0])
            / eigenvalues.sum().clamp(min=1e-5)
        ).clamp(0.0, 1.0)
        point_confidence = min(filtered.shape[0] / 32.0, 1.0)
        confidence = (
            0.45 * valid_ratio.clamp(0.0, 1.0)
            + 0.35 * point_confidence
            + 0.20 * anisotropy
        ).clamp(0.0, 1.0)
        centers.append(center)
        dimensions.append(dims_whl)
        yaws.append(yaw)
        confidences.append(confidence)
        valid_ratios.append(valid_ratio)
        anisotropies.append(anisotropy)

    return {
        "center": torch.stack(centers),
        "dimensions": torch.stack(dimensions),
        "yaw": torch.stack(yaws),
        "confidence": torch.stack(confidences),
        "valid_ratio": torch.stack(valid_ratios),
        "anisotropy": torch.stack(anisotropies),
    }


class CrossInstanceShapeMemory(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        capacity: int = 2048,
        topk: int = 8,
        momentum: float = 0.9,
    ):
        super().__init__()
        self.capacity = max(int(capacity), 1)
        self.topk = max(int(topk), 1)
        self.momentum = float(momentum)
        self.register_buffer("keys", torch.zeros(self.capacity, feature_dim))
        self.register_buffer("log_dimensions", torch.zeros(self.capacity, 3))
        self.register_buffer("confidence", torch.zeros(self.capacity))
        self.register_buffer("valid", torch.zeros(self.capacity, dtype=torch.bool))
        self.register_buffer("write_pointer", torch.zeros((), dtype=torch.long))

    @torch.no_grad()
    def retrieve(self, descriptors: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        descriptors = F.normalize(descriptors.float(), dim=-1, eps=1e-6)
        valid_indices = torch.nonzero(self.valid, as_tuple=False).flatten()
        if valid_indices.numel() == 0:
            return (
                descriptors.new_zeros((descriptors.shape[0], 3)),
                descriptors.new_zeros(descriptors.shape[0]),
            )
        keys = F.normalize(self.keys[valid_indices].float(), dim=-1, eps=1e-6)
        similarities = descriptors @ keys.T
        count = min(self.topk, valid_indices.numel())
        top_similarity, top_local = similarities.topk(count, dim=1)
        top_indices = valid_indices[top_local]
        weights = torch.softmax(top_similarity * 10.0, dim=1)
        prototype = (
            weights[:, :, None] * self.log_dimensions[top_indices]
        ).sum(dim=1)
        prototype_confidence = (
            weights
            * self.confidence[top_indices]
            * ((top_similarity + 1.0) * 0.5).clamp(0.0, 1.0)
        ).sum(dim=1)
        return prototype, prototype_confidence.clamp(0.0, 1.0)

    @torch.no_grad()
    def update(
        self,
        descriptors: torch.Tensor,
        dimensions: torch.Tensor,
        confidence: torch.Tensor,
        minimum_confidence: float,
    ):
        descriptors = F.normalize(descriptors.detach().float(), dim=-1, eps=1e-6)
        log_dimensions = torch.log(dimensions.detach().float().clamp(min=0.01))
        confidence = confidence.detach().float().clamp(0.0, 1.0)
        keep = (
            torch.isfinite(descriptors).all(dim=1)
            & torch.isfinite(log_dimensions).all(dim=1)
            & (confidence >= float(minimum_confidence))
        )
        for descriptor, log_dimension, score in zip(
            descriptors[keep],
            log_dimensions[keep],
            confidence[keep],
        ):
            valid_indices = torch.nonzero(self.valid, as_tuple=False).flatten()
            target_index = None
            if valid_indices.numel() > 0:
                similarities = F.cosine_similarity(
                    descriptor[None],
                    self.keys[valid_indices],
                    dim=1,
                )
                best_similarity, best_local = similarities.max(dim=0)
                if float(best_similarity) > 0.97:
                    target_index = int(valid_indices[best_local])
            if target_index is None:
                target_index = int(self.write_pointer)
                self.write_pointer.copy_(
                    (self.write_pointer + 1) % self.capacity
                )
                self.keys[target_index].copy_(descriptor)
                self.log_dimensions[target_index].copy_(log_dimension)
                self.confidence[target_index].copy_(score)
                self.valid[target_index] = True
            else:
                momentum = self.momentum
                self.keys[target_index].mul_(momentum).add_(
                    descriptor,
                    alpha=1.0 - momentum,
                )
                self.keys[target_index].copy_(
                    F.normalize(self.keys[target_index], dim=0, eps=1e-6)
                )
                self.log_dimensions[target_index].mul_(momentum).add_(
                    log_dimension,
                    alpha=1.0 - momentum,
                )
                self.confidence[target_index].mul_(momentum).add_(
                    score,
                    alpha=1.0 - momentum,
                )


class FrozenDINOv2MultiScale(nn.Module):
    def __init__(
        self,
        checkpoint: str,
        image_size: int = 336,
        output_dim: int = 128,
        layers: Sequence[int] = (2, 5, 8, 11),
        pixel_mean: Sequence[float] = (103.530, 116.280, 123.675),
        pixel_std: Sequence[float] = (57.375, 57.120, 58.395),
        input_format: str = "BGR",
        chunk_size: int = 2,
    ):
        super().__init__()
        from timm.models import create_model

        self.model = create_model(
            "vit_base_patch14_dinov2",
            pretrained=False,
            num_classes=0,
            dynamic_img_size=True,
        )
        state = torch.load(checkpoint, map_location="cpu")
        if isinstance(state, dict) and isinstance(state.get("model"), dict):
            state = state["model"]
        model_keys = set(self.model.state_dict())
        filtered = {key: value for key, value in state.items() if key in model_keys}
        incompatible = self.model.load_state_dict(filtered, strict=False)
        unexpected = [key for key in incompatible.unexpected_keys if key != "mask_token"]
        if incompatible.missing_keys or unexpected:
            raise RuntimeError(
                "DINOv2 checkpoint mismatch: "
                f"missing={incompatible.missing_keys}, unexpected={unexpected}"
            )
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.model.eval()

        self.layers = tuple(int(layer) for layer in layers)
        self.image_size = int(image_size)
        self.patch_size = 14
        self.output_dim = int(output_dim)
        self.input_format = str(input_format).upper()
        self.chunk_size = max(int(chunk_size), 1)
        self.register_buffer(
            "detector_mean",
            torch.tensor(pixel_mean, dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "detector_std",
            torch.tensor(pixel_std, dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "dino_mean",
            torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "dino_std",
            torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )
        self.projection = nn.Sequential(
            nn.Conv2d(768 * len(self.layers), self.output_dim, kernel_size=1, bias=False),
            nn.GroupNorm(8, self.output_dim),
            nn.GELU(),
        )

    def train(self, mode: bool = True):
        super().train(mode)
        self.model.eval()
        return self

    def forward(self, detector_images: torch.Tensor) -> Tuple[torch.Tensor, float]:
        raw = detector_images * self.detector_std + self.detector_mean
        raw = raw.clamp(0.0, 255.0) / 255.0
        if self.input_format == "BGR":
            raw = raw[:, [2, 1, 0]]

        height, width = raw.shape[-2:]
        scale = float(self.image_size) / max(height, width)
        resized_h = max(self.patch_size, int(round(height * scale / self.patch_size)) * self.patch_size)
        resized_w = max(self.patch_size, int(round(width * scale / self.patch_size)) * self.patch_size)
        actual_scale = resized_w / float(width)
        raw = F.interpolate(
            raw,
            size=(resized_h, resized_w),
            mode="bilinear",
            align_corners=False,
        )
        normalized = (raw - self.dino_mean) / self.dino_std
        projected_chunks = []
        for start in range(0, normalized.shape[0], self.chunk_size):
            chunk = normalized[start : start + self.chunk_size]
            with torch.no_grad(), torch.cuda.amp.autocast(
                enabled=chunk.is_cuda,
                dtype=torch.float16,
            ):
                feature_list = self.model.get_intermediate_layers(
                    chunk,
                    n=list(self.layers),
                    reshape=True,
                    norm=True,
                )
            projected_chunks.append(
                self.projection(
                    torch.cat(
                        [feature.float() for feature in feature_list],
                        dim=1,
                    )
                )
            )
        features = torch.cat(projected_chunks, dim=0)
        spatial_scale = actual_scale / self.patch_size
        return features, spatial_scale


class MultiHypothesisGeometryInterpreter(nn.Module):
    def __init__(
        self,
        visual_channels: int,
        dino_channels: int,
        hidden_dim: int = 256,
        num_hypotheses: int = 8,
        num_layers: int = 3,
        num_heads: int = 8,
        residual_scale: float = 0.25,
        shape_memory_capacity: int = 2048,
        shape_memory_topk: int = 8,
        shape_memory_momentum: float = 0.9,
        shape_prototype_blend: float = 0.25,
    ):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.num_hypotheses = max(int(num_hypotheses), 2)
        self.residual_scale = float(residual_scale)
        self.shape_prototype_blend = float(shape_prototype_blend)

        self.visual_projection = nn.Conv2d(visual_channels, hidden_dim, kernel_size=1)
        self.dino_projection = nn.Conv2d(dino_channels, hidden_dim, kernel_size=1)
        self.point_projection = nn.Sequential(
            nn.Conv2d(3, hidden_dim, kernel_size=1),
            nn.GroupNorm(8, hidden_dim),
            nn.GELU(),
        )
        self.ray_projection = nn.Sequential(
            nn.Conv2d(3, hidden_dim, kernel_size=1),
            nn.GroupNorm(8, hidden_dim),
            nn.GELU(),
        )
        self.modality_embedding = nn.Parameter(torch.zeros(4, hidden_dim))
        nn.init.normal_(self.modality_embedding, std=0.02)

        self.anchor_encoder = nn.Sequential(
            nn.Linear(15, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.hypothesis_queries = nn.Parameter(
            torch.randn(self.num_hypotheses - 1, hidden_dim) * 0.02
        )
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.delta_head = nn.Linear(hidden_dim, 7)
        self.quality_head = nn.Linear(hidden_dim, 1)
        self.log_variance_head = nn.Linear(hidden_dim, 4)
        nn.init.normal_(self.delta_head.weight, std=0.002)
        nn.init.constant_(self.delta_head.bias, 0.0)
        nn.init.constant_(self.quality_head.bias, 0.0)
        nn.init.constant_(self.log_variance_head.bias, -1.0)
        self.shape_memory = CrossInstanceShapeMemory(
            feature_dim=dino_channels,
            capacity=shape_memory_capacity,
            topk=shape_memory_topk,
            momentum=shape_memory_momentum,
        )

    def _memory_tokens(
        self,
        visual_roi: torch.Tensor,
        dino_roi: torch.Tensor,
        point_roi: torch.Tensor,
        ray_roi: torch.Tensor,
    ) -> torch.Tensor:
        maps = [
            self.visual_projection(visual_roi),
            self.dino_projection(dino_roi),
            self.point_projection(point_roi),
            self.ray_projection(ray_roi),
        ]
        tokens = []
        for index, feature in enumerate(maps):
            token = feature.flatten(2).transpose(1, 2)
            tokens.append(token + self.modality_embedding[index].view(1, 1, -1))
        return torch.cat(tokens, dim=1)

    def _anchor_features(
        self,
        explicit_anchor: dict,
        ray_roi: torch.Tensor,
        prototype_confidence: torch.Tensor,
    ) -> torch.Tensor:
        ray_mean = ray_roi.mean(dim=(2, 3))
        anchor = torch.cat(
            (
                explicit_anchor["center"],
                torch.log(explicit_anchor["dimensions"].clamp(min=0.01)),
                torch.sin(explicit_anchor["yaw"])[:, None],
                torch.cos(explicit_anchor["yaw"])[:, None],
                explicit_anchor["confidence"][:, None],
                explicit_anchor["valid_ratio"][:, None],
                explicit_anchor["anisotropy"][:, None],
                ray_mean,
                prototype_confidence[:, None],
            ),
            dim=1,
        )
        return self.anchor_encoder(anchor)

    def forward(
        self,
        visual_roi: torch.Tensor,
        dino_roi: torch.Tensor,
        point_roi: torch.Tensor,
        ray_roi: torch.Tensor,
    ):
        memory = self._memory_tokens(visual_roi, dino_roi, point_roi, ray_roi)
        dino_descriptor = F.normalize(
            dino_roi.mean(dim=(2, 3)).float(),
            dim=-1,
            eps=1e-6,
        )
        explicit_anchor = online_pca_anchor(point_roi)
        prototype_log_dimensions, prototype_confidence = self.shape_memory.retrieve(
            dino_descriptor
        )
        prototype_blend = (
            prototype_confidence * self.shape_prototype_blend
        ).clamp(0.0, self.shape_prototype_blend)
        explicit_anchor["dimensions"] = torch.exp(
            (1.0 - prototype_blend[:, None])
            * torch.log(explicit_anchor["dimensions"].clamp(min=0.01))
            + prototype_blend[:, None] * prototype_log_dimensions
        )
        anchor = self._anchor_features(
            explicit_anchor,
            ray_roi,
            prototype_confidence,
        )
        learned_queries = anchor[:, None, :] + self.hypothesis_queries[None, :, :]
        decoded = self.decoder(learned_queries, memory)

        learned_delta = torch.tanh(self.delta_head(decoded)) * self.residual_scale
        zero_delta = torch.zeros(
            (decoded.shape[0], 1, learned_delta.shape[-1]),
            device=decoded.device,
            dtype=decoded.dtype,
        )
        deltas = torch.cat((zero_delta, learned_delta), dim=1)

        anchor_quality = torch.zeros(
            (decoded.shape[0], 1),
            device=decoded.device,
            dtype=decoded.dtype,
        )
        quality_logits = torch.cat(
            (anchor_quality, self.quality_head(decoded).squeeze(-1)),
            dim=1,
        )
        anchor_log_variance = torch.zeros(
            (decoded.shape[0], 1, 4),
            device=decoded.device,
            dtype=decoded.dtype,
        )
        log_variance = torch.cat(
            (anchor_log_variance, self.log_variance_head(decoded).clamp(-4.0, 4.0)),
            dim=1,
        )
        return (
            deltas,
            quality_logits,
            log_variance,
            explicit_anchor,
            dino_descriptor,
            prototype_confidence,
        )

    @torch.no_grad()
    def update_shape_memory(
        self,
        descriptors: torch.Tensor,
        dimensions: torch.Tensor,
        confidence: torch.Tensor,
        minimum_confidence: float,
    ):
        self.shape_memory.update(
            descriptors,
            dimensions,
            confidence,
            minimum_confidence,
        )


class SoftCuboidRenderer(nn.Module):
    _TRIANGLES = (
        (0, 1, 2), (0, 2, 3),
        (4, 6, 5), (4, 7, 6),
        (0, 4, 5), (0, 5, 1),
        (1, 5, 6), (1, 6, 2),
        (2, 6, 7), (2, 7, 3),
        (3, 7, 4), (3, 4, 0),
    )

    def __init__(
        self,
        render_size: int = 28,
        edge_softness: float = 0.02,
        depth_temperature: float = 0.08,
    ):
        super().__init__()
        self.render_size = int(render_size)
        self.edge_softness = float(edge_softness)
        self.depth_temperature = float(depth_temperature)

    def forward(
        self,
        corners_cam: torch.Tensor,
        intrinsics: torch.Tensor,
        roi_boxes_xyxy: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        z = corners_cam[:, :, 2].clamp(min=1e-4)
        projected = torch.bmm(intrinsics, corners_cam.transpose(1, 2)).transpose(1, 2)
        uv = projected[:, :, :2] / projected[:, :, 2:3].clamp(min=1e-4)

        x1, y1, x2, y2 = roi_boxes_xyxy.unbind(dim=1)
        width = (x2 - x1).clamp(min=1.0)
        height = (y2 - y1).clamp(min=1.0)
        uv_local = uv.clone()
        uv_local[:, :, 0] = (
            (uv[:, :, 0] - x1[:, None]) / width[:, None] * (self.render_size - 1)
        )
        uv_local[:, :, 1] = (
            (uv[:, :, 1] - y1[:, None]) / height[:, None] * (self.render_size - 1)
        )

        grid_y, grid_x = torch.meshgrid(
            torch.arange(self.render_size, device=uv.device, dtype=uv.dtype),
            torch.arange(self.render_size, device=uv.device, dtype=uv.dtype),
            indexing="ij",
        )
        points = torch.stack((grid_x, grid_y), dim=-1).view(1, 1, -1, 2)

        occupancies = []
        depths = []
        for triangle in self._TRIANGLES:
            vertex = uv_local[:, triangle, :]
            vertex_z = z[:, triangle]
            a = vertex[:, 0:1, None, :]
            b = vertex[:, 1:2, None, :]
            c = vertex[:, 2:3, None, :]
            p = points
            denominator = (
                (b[..., 1] - c[..., 1]) * (a[..., 0] - c[..., 0])
                + (c[..., 0] - b[..., 0]) * (a[..., 1] - c[..., 1])
            )
            denominator = torch.where(
                denominator.abs() < 1e-5,
                torch.full_like(denominator, 1e-5),
                denominator,
            )
            weight_a = (
                (b[..., 1] - c[..., 1]) * (p[..., 0] - c[..., 0])
                + (c[..., 0] - b[..., 0]) * (p[..., 1] - c[..., 1])
            ) / denominator
            weight_b = (
                (c[..., 1] - a[..., 1]) * (p[..., 0] - c[..., 0])
                + (a[..., 0] - c[..., 0]) * (p[..., 1] - c[..., 1])
            ) / denominator
            weight_c = 1.0 - weight_a - weight_b
            barycentric = torch.stack((weight_a, weight_b, weight_c), dim=-1)
            inside = torch.sigmoid(
                barycentric.min(dim=-1).values / max(self.edge_softness, 1e-4)
            ).squeeze(1)
            triangle_depth = (
                barycentric.squeeze(1) * vertex_z[:, None, :]
            ).sum(dim=-1)
            occupancies.append(inside)
            depths.append(triangle_depth)

        occupancy = torch.stack(occupancies, dim=1).clamp(1e-5, 1.0 - 1e-5)
        triangle_depth = torch.stack(depths, dim=1).clamp(min=1e-4)
        silhouette = 1.0 - torch.prod(1.0 - occupancy, dim=1)
        depth_logits = (
            torch.log(occupancy)
            - triangle_depth / max(self.depth_temperature, 1e-4)
        )
        depth_weights = torch.softmax(depth_logits, dim=1)
        rendered_depth = (depth_weights * triangle_depth).sum(dim=1)
        shape = (-1, self.render_size, self.render_size)
        return silhouette.view(*shape), rendered_depth.view(*shape)
