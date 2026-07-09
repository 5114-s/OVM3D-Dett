# Copyright (c) Meta Platforms, Inc. and affiliates
import copy
import os
import torch
import numpy as np
import cv2
import torch.nn.functional as F
from detectron2.structures import BoxMode, Keypoints
from detectron2.data import detection_utils
from detectron2.data import transforms as T
from detectron2.data import (
    DatasetMapper
)
from detectron2.structures import (
    Boxes,
    BoxMode,
    Instances,
)

class DatasetMapper3D(DatasetMapper):

    def __init__(self, cfg, is_train=True):
        super().__init__(cfg, is_train=is_train)
        self.use_depth = bool(getattr(cfg.INPUT, "USE_DEPTH", False))
        self.depth_root = str(getattr(cfg.INPUT, "DEPTH_ROOT", ""))
        self.depth_allow_sensor_fallback = bool(
            getattr(cfg.INPUT, "DEPTH_ALLOW_SENSOR_FALLBACK", True)
        )
        self.use_pseudo_mask = bool(getattr(cfg.INPUT, "USE_PSEUDO_MASK", False))
        self.pseudo_mask_root = str(getattr(cfg.INPUT, "PSEUDO_MASK_ROOT", ""))
        self.pseudo_mask_size = int(getattr(cfg.INPUT, "PSEUDO_MASK_SIZE", 28))
        self.pseudo_mask_match_iou_threshold = float(
            getattr(cfg.INPUT, "PSEUDO_MASK_MATCH_IOU_THRESHOLD", 0.05)
        )
        self.pseudo_mask_allow_reuse = bool(
            getattr(cfg.INPUT, "PSEUDO_MASK_ALLOW_REUSE", False)
        )
        self.use_ground_mask = bool(getattr(cfg.INPUT, "USE_GROUND_MASK", False))
        self.ground_mask_root = str(getattr(cfg.INPUT, "GROUND_MASK_ROOT", ""))
        self.distributional_num_candidates = int(
            getattr(cfg.MODEL.ROI_CUBE_HEAD, "DISTRIBUTIONAL_NUM_CANDIDATES", 8)
        )

    def __call__(self, dataset_dict):
        
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
        
        image = detection_utils.read_image(dataset_dict["file_name"], format=self.image_format)
        detection_utils.check_image_size(dataset_dict, image)
        depth = self._load_depth(dataset_dict, image.shape[:2]) if self.use_depth else None
        pseudo_masks = (
            self._load_pseudo_masks(dataset_dict, image.shape[:2])
            if self.use_pseudo_mask
            else None
        )
        ground_mask = (
            self._load_ground_mask(dataset_dict, image.shape[:2])
            if self.use_ground_mask
            else None
        )

        aug_input = T.AugInput(image)
        transforms = self.augmentations(aug_input)
        image = aug_input.image
        if depth is not None:
            depth = transforms.apply_image(depth[:, :, None])[:, :, 0]
            depth = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
            depth = np.maximum(depth, 0.0).astype(np.float32)
        if pseudo_masks is not None:
            transformed_masks = []
            for pseudo_mask in pseudo_masks:
                transformed_masks.append(
                    transforms.apply_segmentation(pseudo_mask.astype(np.uint8)) > 0
                )
            pseudo_masks = np.asarray(transformed_masks, dtype=np.uint8)
        if ground_mask is not None:
            ground_mask = (
                transforms.apply_segmentation(ground_mask.astype(np.uint8)) > 0
            ).astype(np.float32)

        image_shape = image.shape[:2]  # h, w

        # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
        # Therefore it's important to use torch.Tensor.
        dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))
        if depth is not None:
            dataset_dict["depth"] = torch.as_tensor(np.ascontiguousarray(depth[None, :, :]))
        if ground_mask is not None:
            dataset_dict["ground_mask"] = torch.as_tensor(
                np.ascontiguousarray(ground_mask[None, :, :])
            )

        # no need for additoinal processing at inference
        if not self.is_train:
            return dataset_dict

        if "annotations" in dataset_dict:

            dataset_id = dataset_dict['dataset_id']
            K = np.array(dataset_dict['K'])

            unknown_categories = self.dataset_id_to_unknown_cats[dataset_id]

            # transform and pop off annotations
            annos = [
                transform_instance_annotations(obj, transforms, K=K)
                for obj in dataset_dict.pop("annotations") if obj.get("iscrowd", 0) == 0
            ]
            if pseudo_masks is not None:
                self._attach_render_masks(annos, pseudo_masks, image_shape)

            # convert to instance format
            instances = annotations_to_instances(
                annos,
                image_shape,
                unknown_categories,
                distributional_num_candidates=self.distributional_num_candidates,
            )
            dataset_dict["instances"] = detection_utils.filter_empty_instances(instances)

        return dataset_dict

    def _attach_render_masks(self, annos, pseudo_masks, image_shape):
        height, width = image_shape
        matched_indices = self._match_pseudo_masks_to_annos(
            annos, pseudo_masks, image_shape
        )
        for annotation, matched_index in zip(annos, matched_indices):
            mask_index = int(annotation.get("pseudo_mask_index", -1))
            if not (0 <= mask_index < len(pseudo_masks)):
                mask_index = matched_index

            if 0 <= mask_index < len(pseudo_masks):
                mask = pseudo_masks[mask_index]
            else:
                mask = np.zeros((height, width), dtype=np.uint8)

            box = BoxMode.convert(
                annotation["bbox"],
                annotation["bbox_mode"],
                BoxMode.XYXY_ABS,
            )
            x1, y1, x2, y2 = [float(value) for value in box]
            ix1 = int(np.floor(np.clip(x1, 0, max(width - 1, 0))))
            iy1 = int(np.floor(np.clip(y1, 0, max(height - 1, 0))))
            ix2 = int(np.ceil(np.clip(x2, ix1 + 1, width)))
            iy2 = int(np.ceil(np.clip(y2, iy1 + 1, height)))
            crop = mask[iy1:iy2, ix1:ix2]
            if crop.size == 0:
                crop = np.zeros((1, 1), dtype=np.uint8)
            crop_tensor = torch.as_tensor(crop[None, None].astype(np.float32))
            resized = F.interpolate(
                crop_tensor,
                size=(self.pseudo_mask_size, self.pseudo_mask_size),
                mode="nearest",
            )[0, 0]
            annotation["render_mask"] = resized.numpy().astype(np.float32)

    def _match_pseudo_masks_to_annos(self, annos, pseudo_masks, image_shape):
        if pseudo_masks is None or len(pseudo_masks) == 0 or len(annos) == 0:
            return [-1 for _ in annos]

        mask_boxes = self._mask_boxes_xyxy(pseudo_masks, image_shape)
        if len(mask_boxes) == 0:
            return [-1 for _ in annos]

        height, width = image_shape
        anno_boxes = []
        for annotation in annos:
            box = BoxMode.convert(
                annotation["bbox"],
                annotation["bbox_mode"],
                BoxMode.XYXY_ABS,
            )
            x1, y1, x2, y2 = [float(value) for value in box]
            anno_boxes.append(
                [
                    np.clip(x1, 0.0, float(width)),
                    np.clip(y1, 0.0, float(height)),
                    np.clip(x2, 0.0, float(width)),
                    np.clip(y2, 0.0, float(height)),
                ]
            )
        anno_boxes = np.asarray(anno_boxes, dtype=np.float32)

        ious = self._pairwise_iou_xyxy(anno_boxes, mask_boxes)
        matched = [-1 for _ in annos]
        used_masks = set()
        for anno_idx in np.argsort(-ious.max(axis=1)):
            for mask_idx in np.argsort(-ious[anno_idx]):
                mask_idx = int(mask_idx)
                if not self.pseudo_mask_allow_reuse and mask_idx in used_masks:
                    continue
                if ious[anno_idx, mask_idx] < self.pseudo_mask_match_iou_threshold:
                    break
                matched[int(anno_idx)] = mask_idx
                used_masks.add(mask_idx)
                break
        return matched

    def _mask_boxes_xyxy(self, masks, image_shape):
        height, width = image_shape
        boxes = []
        for mask in masks:
            ys, xs = np.nonzero(mask > 0)
            if len(xs) == 0 or len(ys) == 0:
                boxes.append([0.0, 0.0, 0.0, 0.0])
                continue
            x1 = float(np.clip(xs.min(), 0, max(width - 1, 0)))
            y1 = float(np.clip(ys.min(), 0, max(height - 1, 0)))
            x2 = float(np.clip(xs.max() + 1, 0, width))
            y2 = float(np.clip(ys.max() + 1, 0, height))
            boxes.append([x1, y1, x2, y2])
        return np.asarray(boxes, dtype=np.float32)

    def _pairwise_iou_xyxy(self, boxes1, boxes2):
        if len(boxes1) == 0 or len(boxes2) == 0:
            return np.zeros((len(boxes1), len(boxes2)), dtype=np.float32)
        lt = np.maximum(boxes1[:, None, :2], boxes2[None, :, :2])
        rb = np.minimum(boxes1[:, None, 2:], boxes2[None, :, 2:])
        wh = np.maximum(rb - lt, 0.0)
        inter = wh[:, :, 0] * wh[:, :, 1]
        area1 = np.maximum(boxes1[:, 2] - boxes1[:, 0], 0.0) * np.maximum(
            boxes1[:, 3] - boxes1[:, 1], 0.0
        )
        area2 = np.maximum(boxes2[:, 2] - boxes2[:, 0], 0.0) * np.maximum(
            boxes2[:, 3] - boxes2[:, 1], 0.0
        )
        union = area1[:, None] + area2[None, :] - inter
        return inter / np.maximum(union, 1e-6)

    def _load_depth(self, dataset_dict, image_shape):
        candidates = []
        for key in ("depth_file_name", "depth_path", "depth_file_path"):
            path = dataset_dict.get(key)
            if path:
                candidates.append(path)

        image_id = dataset_dict.get("image_id")
        if image_id is not None and self.depth_root:
            image_id = int(image_id)
            root = self.depth_root
            candidates.extend([
                os.path.join(root, f"{image_id}.npy"),
                os.path.join(root, "depth", f"{image_id}.npy"),
                os.path.join(root, "train", "depth", f"{image_id}.npy"),
                os.path.join(root, "val", "depth", f"{image_id}.npy"),
                os.path.join(root, "test", "depth", f"{image_id}.npy"),
                os.path.join(root, "SUNRGBD", "train", "depth", f"{image_id}.npy"),
                os.path.join(root, "SUNRGBD", "val", "depth", f"{image_id}.npy"),
                os.path.join(root, "SUNRGBD", "test", "depth", f"{image_id}.npy"),
            ])

        file_name = dataset_dict.get("file_name", "")
        if self.depth_allow_sensor_fallback and "/image/" in file_name:
            candidates.extend([
                file_name.replace("/image/", "/depth/").replace(".jpg", ".png"),
                file_name.replace("/image/", "/depth_bfx/").replace(".jpg", ".png"),
            ])

        for path in candidates:
            if not path:
                continue
            path = os.path.expanduser(path)
            if not os.path.isabs(path):
                path = os.path.abspath(path)
            if not os.path.exists(path):
                continue

            depth = self._read_depth_path(path)
            if depth is None:
                continue
            if depth.ndim != 2:
                continue
            if depth.shape[:2] != tuple(image_shape):
                depth = cv2.resize(
                    depth,
                    (int(image_shape[1]), int(image_shape[0])),
                    interpolation=cv2.INTER_LINEAR,
                )
            return depth.astype(np.float32)

        return None

    def _load_pseudo_masks(self, dataset_dict, image_shape):
        image_id = dataset_dict.get("image_id")
        if image_id is None or not self.pseudo_mask_root:
            return None
        image_id = int(image_id)
        root = self.pseudo_mask_root
        candidates = [
            os.path.join(root, "mask", f"{image_id}.npy"),
            os.path.join(root, "train", "mask", f"{image_id}.npy"),
            os.path.join(root, "val", "mask", f"{image_id}.npy"),
            os.path.join(root, "SUNRGBD", "train", "mask", f"{image_id}.npy"),
            os.path.join(root, "SUNRGBD", "val", "mask", f"{image_id}.npy"),
        ]
        for path in candidates:
            path = os.path.abspath(os.path.expanduser(path))
            if not os.path.exists(path):
                continue
            masks = np.load(path)
            if masks.ndim == 4 and masks.shape[1] == 1:
                masks = masks[:, 0]
            if masks.ndim == 2:
                masks = masks[None]
            if masks.shape[-2:] != tuple(image_shape):
                masks = np.asarray(
                    [
                        cv2.resize(
                            mask.astype(np.uint8),
                            (int(image_shape[1]), int(image_shape[0])),
                            interpolation=cv2.INTER_NEAREST,
                        )
                        for mask in masks
                    ]
                )
            return masks
        return None

    def _load_ground_mask(self, dataset_dict, image_shape):
        image_id = dataset_dict.get("image_id")
        if image_id is None or not self.ground_mask_root:
            return None
        image_id = int(image_id)
        root = self.ground_mask_root
        candidates = [
            os.path.join(root, "ground_mask", f"{image_id}.npy"),
            os.path.join(root, "train", "ground_mask", f"{image_id}.npy"),
            os.path.join(root, "val", "ground_mask", f"{image_id}.npy"),
            os.path.join(root, "SUNRGBD", "train", "ground_mask", f"{image_id}.npy"),
            os.path.join(root, "SUNRGBD", "val", "ground_mask", f"{image_id}.npy"),
        ]
        for path in candidates:
            path = os.path.abspath(os.path.expanduser(path))
            if not os.path.exists(path):
                continue
            mask = np.squeeze(np.load(path))
            if mask.ndim > 2:
                mask = mask.reshape(-1, mask.shape[-2], mask.shape[-1]).any(axis=0)
            if mask.shape != tuple(image_shape):
                mask = cv2.resize(
                    mask.astype(np.uint8),
                    (int(image_shape[1]), int(image_shape[0])),
                    interpolation=cv2.INTER_NEAREST,
                )
            return mask > 0
        return None

    def _read_depth_path(self, path):
        if path.lower().endswith(".npy"):
            depth = np.load(path).astype(np.float32)
        else:
            depth = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            if depth is None:
                return None
            depth = depth.astype(np.float32) / 8000.0
        return np.squeeze(depth)

'''
Cached for mirroring annotations
'''
_M1 = np.array([
    [1, 0, 0], 
    [0, -1, 0],
    [0, 0, -1]
])
_M2 = np.array([
    [-1.,  0.,  0.],
    [ 0., -1.,  0.],
    [ 0.,  0.,  1.]
])


def transform_instance_annotations(annotation, transforms, *, K):
    
    if isinstance(transforms, (tuple, list)):
        transforms = T.TransformList(transforms)
    
    # bbox is 1d (per-instance bounding box)
    bbox = BoxMode.convert(annotation["bbox"], annotation["bbox_mode"], BoxMode.XYXY_ABS)
    bbox = transforms.apply_box(np.array([bbox]))[0]
    
    annotation["bbox"] = bbox
    annotation["bbox_mode"] = BoxMode.XYXY_ABS

    if annotation['center_cam'][2] != 0:

        # project the 3D box annotation XYZ_3D to screen 
        point3D = annotation['center_cam']
        point2D = K @ np.array(point3D)
        point2D[:2] = point2D[:2] / point2D[-1]
        annotation["center_cam_proj"] = point2D.tolist()

        # apply coords transforms to 2D box
        annotation["center_cam_proj"][0:2] = transforms.apply_coords(
            point2D[np.newaxis][:, :2]
        )[0].tolist()

        keypoints = (K @ np.array(annotation["bbox3D_cam"]).T).T
        keypoints[:, 0] /= keypoints[:, -1]
        keypoints[:, 1] /= keypoints[:, -1]
        
        if annotation['ignore']:
            # all keypoints marked as not visible 
            # 0 - unknown, 1 - not visible, 2 visible
            keypoints[:, 2] = 1
        else:
            
            valid_keypoints = keypoints[:, 2] > 0

            # 0 - unknown, 1 - not visible, 2 visible
            keypoints[:, 2] = 2
            keypoints[valid_keypoints, 2] = 2

        # in place
        transforms.apply_coords(keypoints[:, :2])
        annotation["keypoints"] = keypoints.tolist()

        # manually apply mirror for pose
        for transform in transforms:

            # horrizontal flip?
            if isinstance(transform, T.HFlipTransform):

                pose = _M1 @ np.array(annotation["pose"]) @ _M2
                annotation["pose"] = pose.tolist()
                annotation["R_cam"] = pose.tolist()

        transform_distributional_candidates(annotation, transforms, K)

    return annotation


def transform_distributional_candidates(annotation, transforms, K):
    raw_candidates = annotation.get("distributional_box_candidates", None)
    if raw_candidates is None:
        raw_candidates = annotation.get("latent_box_candidates", None)
    if not isinstance(raw_candidates, list):
        return

    transformed = []
    for candidate in raw_candidates:
        if not isinstance(candidate, dict):
            continue
        candidate = copy.deepcopy(candidate)
        center = candidate.get("center_cam", None)
        if center is not None:
            try:
                point3d = np.asarray(center, dtype=np.float32).reshape(3)
                if np.all(np.isfinite(point3d)) and point3d[2] > 0:
                    point2d = K @ point3d
                    point2d[:2] = point2d[:2] / max(float(point2d[-1]), 1e-6)
                    point2d[:2] = transforms.apply_coords(
                        point2d[np.newaxis, :2]
                    )[0]
                    candidate["center_cam_proj"] = point2d[:2].tolist()
            except Exception:
                pass

        pose = candidate.get("R_cam", candidate.get("pose", None))
        if pose is not None:
            try:
                pose = np.asarray(pose, dtype=np.float32).reshape(3, 3)
                for transform in transforms:
                    if isinstance(transform, T.HFlipTransform):
                        pose = _M1 @ pose @ _M2
                candidate["R_cam"] = pose.tolist()
                candidate["pose"] = pose.tolist()
            except Exception:
                pass
        transformed.append(candidate)

    annotation["distributional_box_candidates"] = transformed


def _distributional_candidates_to_tensors(anno, max_candidates):
    candidates = anno.get("distributional_box_candidates", None)
    if candidates is None:
        candidates = anno.get("latent_box_candidates", None)
    if not isinstance(candidates, list):
        candidates = []

    boxes = np.zeros((max_candidates, 6), dtype=np.float32)
    poses = np.zeros((max_candidates, 3, 3), dtype=np.float32)
    weights = np.zeros((max_candidates,), dtype=np.float32)
    valid = np.zeros((max_candidates,), dtype=np.bool_)

    out_index = 0
    for candidate in candidates:
        if out_index >= max_candidates:
            break
        if not isinstance(candidate, dict):
            continue
        center = candidate.get("center_cam", None)
        center_proj = candidate.get("center_cam_proj", None)
        dims = candidate.get("dimensions", None)
        pose = candidate.get("R_cam", candidate.get("pose", None))
        if center is None or center_proj is None or dims is None or pose is None:
            continue
        try:
            center = np.asarray(center, dtype=np.float32).reshape(3)
            center_proj = np.asarray(center_proj, dtype=np.float32).reshape(2)
            dims = np.asarray(dims, dtype=np.float32).reshape(3)
            pose = np.asarray(pose, dtype=np.float32).reshape(3, 3)
        except Exception:
            continue
        if (
            not np.all(np.isfinite(center))
            or not np.all(np.isfinite(center_proj))
            or not np.all(np.isfinite(dims))
            or not np.all(np.isfinite(pose))
            or center[2] <= 0.05
            or np.any(dims <= 0.0)
        ):
            continue
        weight = candidate.get("posterior", candidate.get("score", candidate.get("weight", 1.0)))
        try:
            weight = float(weight)
        except Exception:
            weight = 1.0
        boxes[out_index] = np.asarray(
            [center_proj[0], center_proj[1], center[2], dims[0], dims[1], dims[2]],
            dtype=np.float32,
        )
        poses[out_index] = pose
        weights[out_index] = max(weight, 0.0)
        valid[out_index] = True
        out_index += 1

    return boxes, poses, weights, valid


def annotations_to_instances(
    annos,
    image_size,
    unknown_categories,
    distributional_num_candidates=8,
):

    # init
    target = Instances(image_size)
    
    # add classes, 2D boxes, 3D boxes and poses
    target.gt_classes = torch.tensor([int(obj["category_id"]) for obj in annos], dtype=torch.int64)
    target.gt_boxes = Boxes([BoxMode.convert(obj["bbox"], obj["bbox_mode"], BoxMode.XYXY_ABS) for obj in annos])
    target.gt_boxes3D = torch.FloatTensor([anno['center_cam_proj'] + anno['dimensions'] + anno['center_cam'] for anno in annos])
    target.gt_poses = torch.FloatTensor([anno['pose'] for anno in annos])
    if len(annos) > 0:
        weights = [float(anno.get("pseudo_weight", 1.0)) for anno in annos]
        target.gt_pseudo_weight = torch.FloatTensor(weights).clamp(0.05, 1.0)
        factor_names = ("xy", "z", "dims", "pose", "joint")
        for factor_name in factor_names:
            factor_weights = [
                float(anno.get(f"pseudo_weight_{factor_name}", anno.get("pseudo_weight", 1.0)))
                for anno in annos
            ]
            target.set(
                f"gt_pseudo_weight_{factor_name}",
                torch.FloatTensor(factor_weights).clamp(0.0, 1.0),
            )
        render_masks = [
            torch.as_tensor(
                anno.get("render_mask", np.zeros((28, 28), dtype=np.float32)),
                dtype=torch.float32,
            )
            for anno in annos
        ]
        target.gt_render_masks = torch.stack(render_masks, dim=0)
        target.gt_pag_score = torch.FloatTensor([
            float(anno.get("pag_score", 1.0)) for anno in annos
        ]).clamp(0.05, 1.0)
        corner_depth_scores = []
        for anno in annos:
            corner_score = anno.get("moca3d_projected_corner_depth_score", 1.0)
            if isinstance(corner_score, bool):
                corner_score = anno.get("pag_score", 1.0) if corner_score else 0.05
            corner_depth_scores.append(float(corner_score))
        target.gt_projected_corner_depth_score = torch.FloatTensor(corner_depth_scores).clamp(0.05, 1.0)
        candidate_boxes = []
        candidate_poses = []
        candidate_weights = []
        candidate_valid = []
        for anno in annos:
            boxes_i, poses_i, weights_i, valid_i = _distributional_candidates_to_tensors(
                anno,
                int(distributional_num_candidates),
            )
            candidate_boxes.append(torch.from_numpy(boxes_i))
            candidate_poses.append(torch.from_numpy(poses_i))
            candidate_weights.append(torch.from_numpy(weights_i))
            candidate_valid.append(torch.from_numpy(valid_i))
        target.gt_distributional_candidate_boxes3D = torch.stack(candidate_boxes, dim=0)
        target.gt_distributional_candidate_poses = torch.stack(candidate_poses, dim=0)
        target.gt_distributional_candidate_weights = torch.stack(candidate_weights, dim=0)
        target.gt_distributional_candidate_valid = torch.stack(candidate_valid, dim=0)
    else:
        target.gt_pseudo_weight = torch.FloatTensor([])
        for factor_name in ("xy", "z", "dims", "pose", "joint"):
            target.set(f"gt_pseudo_weight_{factor_name}", torch.FloatTensor([]))
        target.gt_render_masks = torch.empty((0, 28, 28), dtype=torch.float32)
        target.gt_pag_score = torch.FloatTensor([])
        target.gt_projected_corner_depth_score = torch.FloatTensor([])
        target.gt_distributional_candidate_boxes3D = torch.empty(
            (0, int(distributional_num_candidates), 6),
            dtype=torch.float32,
        )
        target.gt_distributional_candidate_poses = torch.empty(
            (0, int(distributional_num_candidates), 3, 3),
            dtype=torch.float32,
        )
        target.gt_distributional_candidate_weights = torch.empty(
            (0, int(distributional_num_candidates)),
            dtype=torch.float32,
        )
        target.gt_distributional_candidate_valid = torch.empty(
            (0, int(distributional_num_candidates)),
            dtype=torch.bool,
        )
    
    n = len(target.gt_classes)

    # do keypoints?
    target.gt_keypoints = Keypoints(torch.FloatTensor([anno['keypoints'] for anno in annos]))

    gt_unknown_category_mask = torch.zeros(max(unknown_categories)+1, dtype=bool)
    gt_unknown_category_mask[torch.tensor(list(unknown_categories))] = True

    # include available category indices as tensor with GTs
    target.gt_unknown_category_mask = gt_unknown_category_mask.unsqueeze(0).repeat([n, 1])

    return target
