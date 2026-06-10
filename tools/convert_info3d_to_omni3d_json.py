"""
将 pseudo_label_gsam_v2 的 info_3d.pth 转换为 Omni3D JSON 格式
与 SUNRGBD_train.json (Omni3D_pl-1) 格式完全一致

用法:
    python tools/convert_info3d_to_omni3d_json.py \
        --pseudo_pth /extra/ZhaoX/pseudo_label_gsam_v2/train/info_3d.pth \
        --source_json /data/ZhaoX/OVM3D-Dett/datasets/Omni3D_pl-1/SUNRGBD_train.json \
        --output /extra/ZhaoX/pseudo_label_gsam_v2/SUNRGBD_train.json \
        --split train

    python tools/convert_info3d_to_omni3d_json.py \
        --pseudo_pth /extra/ZhaoX/pseudo_label_gsam_v2/val/info_3d.pth \
        --source_json /data/ZhaoX/OVM3D-Dett/datasets/Omni3D_pl-1/SUNRGBD_val.json \
        --output /extra/ZhaoX/pseudo_label_gsam_v2/SUNRGBD_val.json \
        --split val
"""

import argparse
from collections import Counter
import json
import os

import numpy as np
import torch
from tqdm import tqdm


# ============================================================
# SUNRGBD 类别表 (与 Omni3D_pl-1 保持一致)
# ============================================================
COCO_CATEGORIES = [
    {"id": 11, "name": "bicycle", "supercategory": "object"},
    {"id": 14, "name": "books", "supercategory": "object"},
    {"id": 15, "name": "bottle", "supercategory": "object"},
    {"id": 18, "name": "chair", "supercategory": "furniture"},
    {"id": 19, "name": "cup", "supercategory": "object"},
    {"id": 20, "name": "laptop", "supercategory": "electronics"},
    {"id": 21, "name": "shoes", "supercategory": "object"},
    {"id": 22, "name": "towel", "supercategory": "object"},
    {"id": 23, "name": "blinds", "supercategory": "window"},
    {"id": 24, "name": "window", "supercategory": "furniture"},
    {"id": 25, "name": "lamp", "supercategory": "furniture"},
    {"id": 26, "name": "shelves", "supercategory": "furniture"},
    {"id": 27, "name": "mirror", "supercategory": "furniture"},
    {"id": 28, "name": "sink", "supercategory": "furniture"},
    {"id": 29, "name": "cabinet", "supercategory": "furniture"},
    {"id": 30, "name": "bathtub", "supercategory": "furniture"},
    {"id": 31, "name": "door", "supercategory": "furniture"},
    {"id": 32, "name": "toilet", "supercategory": "furniture"},
    {"id": 33, "name": "desk", "supercategory": "furniture"},
    {"id": 34, "name": "box", "supercategory": "object"},
    {"id": 35, "name": "bookcase", "supercategory": "furniture"},
    {"id": 36, "name": "picture", "supercategory": "object"},
    {"id": 37, "name": "table", "supercategory": "furniture"},
    {"id": 38, "name": "counter", "supercategory": "furniture"},
    {"id": 39, "name": "bed", "supercategory": "furniture"},
    {"id": 40, "name": "night stand", "supercategory": "furniture"},
    {"id": 42, "name": "pillow", "supercategory": "object"},
    {"id": 43, "name": "sofa", "supercategory": "furniture"},
    {"id": 44, "name": "television", "supercategory": "electronics"},
    {"id": 45, "name": "floor mat", "supercategory": "object"},
    {"id": 46, "name": "curtain", "supercategory": "furniture"},
    {"id": 47, "name": "clothes", "supercategory": "object"},
    {"id": 48, "name": "stationery", "supercategory": "object"},
    {"id": 49, "name": "refrigerator", "supercategory": "appliance"},
    {"id": 52, "name": "bin", "supercategory": "object"},
    {"id": 53, "name": "stove", "supercategory": "appliance"},
    {"id": 57, "name": "oven", "supercategory": "appliance"},
    {"id": 61, "name": "machine", "supercategory": "appliance"},
]

CATEGORY_NAME_TO_ID = {cat["name"]: cat["id"] for cat in COCO_CATEGORIES}

# SUNRGBD priors are [X width, Y height, Z depth]. Omni3D/CubeRCNN stores
# dimensions as [Z width, Y height, X length].
SUNRGBD_ORIGINAL_PRIORS = {
    "bicycle": [0.5, 1.0, 1.5],
    "books": [0.2, 0.1, 0.3],
    "bottle": [0.1, 0.3, 0.1],
    "chair": [0.5, 1.0, 0.5],
    "cup": [0.1, 0.1, 0.1],
    "laptop": [0.3, 0.1, 0.4],
    "shoes": [0.2, 0.1, 0.3],
    "towel": [0.2, 0.1, 0.3],
    "blinds": [0.1, 1.0, 1.5],
    "window": [0.1, 1.0, 1.5],
    "lamp": [0.3, 0.6, 0.3],
    "shelves": [0.3, 1.5, 1.5],
    "mirror": [0.1, 1.0, 0.5],
    "sink": [0.5, 0.2, 0.8],
    "cabinet": [0.5, 1.5, 1.0],
    "bathtub": [0.8, 0.5, 1.5],
    "door": [0.1, 2.0, 1.0],
    "toilet": [0.4, 0.8, 0.5],
    "desk": [0.6, 0.8, 1.2],
    "box": [0.5, 0.5, 0.5],
    "bookcase": [0.3, 2.0, 1.0],
    "picture": [0.1, 0.5, 0.5],
    "table": [0.8, 0.8, 1.5],
    "counter": [0.6, 1.0, 1.5],
    "bed": [1.5, 0.5, 2.0],
    "night stand": [0.4, 0.5, 0.5],
    "pillow": [0.3, 0.3, 0.5],
    "sofa": [1.0, 1.0, 2.0],
    "television": [1.0, 0.5, 0.1],
    "floor mat": [1.0, 0.1, 1.5],
    "curtain": [0.1, 1.5, 1.0],
    "clothes": [0.5, 1.0, 0.5],
    "stationery": [0.3, 0.3, 0.3],
    "refrigerator": [0.8, 1.5, 0.8],
    "bin": [0.5, 0.5, 0.5],
    "stove": [0.6, 0.8, 0.8],
    "oven": [0.6, 0.8, 0.8],
    "machine": [0.8, 1.0, 1.0],
}


def to_list(value):
    if hasattr(value, "tolist"):
        return value.tolist()
    return value


def as_float_list(value):
    return [float(v) for v in to_list(value)]


def as_float_matrix(value):
    return [[float(v) for v in row] for row in to_list(value)]


def convert_2d_box_to_xyxy_pixels(box, width, height):
    """
    Convert GroundingDINO boxes to Omni3D's expected XYXY pixel format.

    The pseudo-label pipeline stores normalized [cx, cy, w, h]. Older helper
    scripts accidentally treated those values as [x1, y1, x2, y2], which makes
    most boxes have x2 <= x1 or y2 <= y1 after scaling.
    """
    vals = as_float_list(box)
    if len(vals) != 4:
        raise ValueError(f"Expected 4 values for 2D box, got {len(vals)}")

    if max(abs(v) for v in vals) <= 2.0:
        cx, cy, bw, bh = vals
        x1 = (cx - bw / 2.0) * width
        y1 = (cy - bh / 2.0) * height
        x2 = (cx + bw / 2.0) * width
        y2 = (cy + bh / 2.0) * height
    else:
        x1, y1, x2, y2 = vals
        if x2 <= x1 or y2 <= y1:
            x2 = x1 + max(0.0, x2)
            y2 = y1 + max(0.0, y2)

    x1 = min(max(float(x1), 0.0), float(width - 1))
    y1 = min(max(float(y1), 0.0), float(height - 1))
    x2 = min(max(float(x2), 0.0), float(width - 1))
    y2 = min(max(float(y2), 0.0), float(height - 1))
    return [x1, y1, x2, y2]


def box_area(box):
    return max(0.0, float(box[2] - box[0])) * max(0.0, float(box[3] - box[1]))


def box_iou_xyxy(box_a, box_b):
    x1 = max(float(box_a[0]), float(box_b[0]))
    y1 = max(float(box_a[1]), float(box_b[1]))
    x2 = min(float(box_a[2]), float(box_b[2]))
    y2 = min(float(box_a[3]), float(box_b[3]))
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = box_area(box_a) + box_area(box_b) - inter
    return inter / max(union, 1e-9)


def clip_box_xyxy(box, width, height):
    return [
        min(max(float(box[0]), 0.0), float(width - 1)),
        min(max(float(box[1]), 0.0), float(height - 1)),
        min(max(float(box[2]), 0.0), float(width - 1)),
        min(max(float(box[3]), 0.0), float(height - 1)),
    ]


def project_box3d_to_2d(bbox3d_cam, K, width, height):
    pts = np.asarray(bbox3d_cam, dtype=np.float64)
    if pts.shape != (8, 3) or not np.all(np.isfinite(pts)):
        return None
    proj = (np.asarray(K, dtype=np.float64) @ pts.T).T
    if np.any(np.abs(proj[:, 2]) < 1e-6):
        return None
    xy = proj[:, :2] / proj[:, 2:3]
    if not np.all(np.isfinite(xy)):
        return None

    raw_box = [
        float(xy[:, 0].min()),
        float(xy[:, 1].min()),
        float(xy[:, 0].max()),
        float(xy[:, 1].max()),
    ]
    clipped_box = clip_box_xyxy(raw_box, width, height)
    if clipped_box[2] <= clipped_box[0] or clipped_box[3] <= clipped_box[1]:
        return None
    return raw_box, clipped_box


def projection_alignment_metrics(projected_box, target_box):
    iou = box_iou_xyxy(projected_box, target_box)

    proj_ctr = np.array([
        (projected_box[0] + projected_box[2]) / 2.0,
        (projected_box[1] + projected_box[3]) / 2.0,
    ])
    target_ctr = np.array([
        (target_box[0] + target_box[2]) / 2.0,
        (target_box[1] + target_box[3]) / 2.0,
    ])
    target_w = max(float(target_box[2] - target_box[0]), 1e-6)
    target_h = max(float(target_box[3] - target_box[1]), 1e-6)
    target_diag = max((target_w * target_w + target_h * target_h) ** 0.5, 1e-6)
    center_norm = float(np.linalg.norm(proj_ctr - target_ctr) / target_diag)
    area_ratio = box_area(projected_box) / max(box_area(target_box), 1e-9)
    return iou, center_norm, area_ratio


def build_3d_assignments(phrases, boxes_px, dimensions, boxes3d, K, width, height):
    """
    Match 2D detections to 3D candidates inside each image and class.

    The pseudo pipeline can produce phrase/box arrays and 3D arrays in different
    orders. Training then pairs the j-th 2D ROI with the j-th 3D target, which
    creates impossible supervision. We fix that by assigning each 2D detection
    to the same-class 3D candidate whose projected cuboid best matches it.
    """
    candidates_by_cat = {}
    detections_by_cat = {}

    for idx, cat_name in enumerate(phrases):
        detections_by_cat.setdefault(cat_name, []).append(idx)

        dim_j = to_list(dimensions[idx])
        dim_val = dim_j[0] if isinstance(dim_j, (list, np.ndarray)) else float(dim_j)
        if dim_val == -1:
            continue

        projection = project_box3d_to_2d(to_list(boxes3d[idx]), K, width, height)
        if projection is None:
            continue

        candidates_by_cat.setdefault(cat_name, []).append((idx, projection[1]))

    assignments = {}
    for cat_name, det_indices in detections_by_cat.items():
        candidates = candidates_by_cat.get(cat_name, [])
        if not candidates:
            continue

        pairs = []
        for det_idx in det_indices:
            target_box = boxes_px[det_idx]
            for cand_idx, projected_box in candidates:
                iou, center_norm, area_ratio = projection_alignment_metrics(projected_box, target_box)
                area_penalty = abs(np.log(max(area_ratio, 1e-6)))
                score = iou - 0.05 * center_norm - 0.01 * area_penalty
                pairs.append((score, iou, center_norm, det_idx, cand_idx))

        pairs.sort(reverse=True, key=lambda item: item[0])
        used_detections = set()
        used_candidates = set()
        for _, _, _, det_idx, cand_idx in pairs:
            if det_idx in used_detections or cand_idx in used_candidates:
                continue
            assignments[det_idx] = cand_idx
            used_detections.add(det_idx)
            used_candidates.add(cand_idx)

    return assignments


def validate_3d_geometry(
    category_name,
    center_cam,
    dimensions,
    bbox3d_cam,
    K,
    target_box2d,
    image_width,
    image_height,
    min_corner_z,
    max_projected_abs,
    max_axis_prior_ratio,
    max_volume_prior_ratio,
    max_abs_dim,
    min_project_iou,
    max_project_center_norm,
    min_project_area_ratio,
    max_project_area_ratio,
):
    pts = np.asarray(bbox3d_cam, dtype=np.float64)
    dims_zyx = np.asarray(dimensions, dtype=np.float64)
    center = np.asarray(center_cam, dtype=np.float64)

    if pts.shape != (8, 3):
        return False, "bad_bbox3d_shape"
    if not np.all(np.isfinite(pts)) or not np.all(np.isfinite(dims_zyx)) or not np.all(np.isfinite(center)):
        return False, "nonfinite_3d"
    if center[2] <= 0 or np.any(dims_zyx <= 0):
        return False, "bad_center_or_dim"
    if pts[:, 2].min() <= min_corner_z:
        return False, "corner_too_close"
    if dims_zyx.max() > max_abs_dim:
        return False, "dim_abs_too_large"

    proj = (np.asarray(K, dtype=np.float64) @ pts.T).T
    if np.any(np.abs(proj[:, 2]) < 1e-6):
        return False, "project_depth_zero"
    proj_xy = proj[:, :2] / proj[:, 2:3]
    if not np.all(np.isfinite(proj_xy)):
        return False, "project_nonfinite"
    if np.max(np.abs(proj_xy)) > max_projected_abs:
        return False, "project_too_large"

    projection = project_box3d_to_2d(pts, K, image_width, image_height)
    if projection is None:
        return False, "project_clipped_bad"
    _, projected_box = projection
    iou, center_norm, area_ratio = projection_alignment_metrics(projected_box, target_box2d)
    if iou < min_project_iou:
        return False, "project_iou"
    if center_norm > max_project_center_norm:
        return False, "project_center"
    if area_ratio < min_project_area_ratio:
        return False, "project_area_small"
    if area_ratio > max_project_area_ratio:
        return False, "project_area_large"

    prior_xyz = np.asarray(SUNRGBD_ORIGINAL_PRIORS.get(category_name, [0.5, 0.5, 0.5]), dtype=np.float64)
    dims_xyz = dims_zyx[[2, 1, 0]]
    axis_ratio = dims_xyz / np.maximum(prior_xyz, 1e-6)
    volume_ratio = float(np.prod(dims_xyz) / max(float(np.prod(prior_xyz)), 1e-6))
    if np.max(axis_ratio) > max_axis_prior_ratio:
        return False, "axis_prior_ratio"
    if volume_ratio > max_volume_prior_ratio:
        return False, "volume_prior_ratio"

    return True, "valid"


# ============================================================
# 主转换逻辑
# ============================================================
def convert_info3d_to_omni3d_json(
    pseudo_pth_path: str,
    source_json_path: str,
    output_path: str,
    split: str = "Train",
    min_corner_z: float = 0.5,
    max_projected_abs: float = 5000.0,
    max_axis_prior_ratio: float = 6.0,
    max_volume_prior_ratio: float = 30.0,
    max_abs_dim: float = 8.0,
    min_project_iou: float = 0.03,
    max_project_center_norm: float = 1.0,
    min_project_area_ratio: float = 0.05,
    max_project_area_ratio: float = 20.0,
):
    print(f"Loading pseudo labels: {pseudo_pth_path}")
    info = torch.load(pseudo_pth_path, map_location="cpu", weights_only=False)
    print(f"  -> {len(info)} images")

    print(f"Loading source JSON: {source_json_path}")
    with open(source_json_path, "r") as f:
        source = json.load(f)
    print(f"  -> {len(source['images'])} images, {len(source['annotations'])} annotations")

    pseudo_img_ids = set(info.keys())

    # 只保留有伪标签的图片
    filtered_images = [
        img for img in source["images"] if img["id"] in pseudo_img_ids
    ]
    print(f"Filtered images (with pseudo labels): {len(filtered_images)}")

    # 构建新 info (沿用 source 的 info，但调整 split)
    new_info = dict(source["info"])
    new_info["split"] = split
    new_info["name"] = f"SUNRGBD {split}"

    # 类别表沿用 Omni3D 标准表
    categories = COCO_CATEGORIES

    # 生成 annotations
    annotations = []
    ann_id = 1
    stats = {
        "total": 0,
        "valid_3d": 0,
        "invalid_3d": 0,
        "unknown_cat": 0,
        "invalid_depth": 0,
        "invalid_2d": 0,
        "unmatched_3d": 0,
    }
    invalid_reasons = Counter()

    for img in tqdm(filtered_images, desc="Converting"):
        im_id = img["id"]
        img_data = info[im_id]
        W, H = img["width"], img["height"]

        phrases = img_data["phrases"]
        boxes = img_data["boxes"]
        center_cam = img_data["center_cam"]
        dimensions = img_data["dimensions"]
        R_cam = img_data["R_cam"]
        boxes3d = img_data["boxes3d"]

        num_objs = len(phrases)
        stats["total"] += num_objs

        boxes_px = [convert_2d_box_to_xyxy_pixels(boxes[j], W, H) for j in range(num_objs)]
        matched_3d_indices = build_3d_assignments(
            phrases=phrases,
            boxes_px=boxes_px,
            dimensions=dimensions,
            boxes3d=boxes3d,
            K=img["K"],
            width=W,
            height=H,
        )

        for j in range(num_objs):
            cat_name = phrases[j]
            cat_id = CATEGORY_NAME_TO_ID.get(cat_name, 0)
            if cat_id == 0:
                stats["unknown_cat"] += 1

            # ---- 2D bbox ----
            bbox_px = boxes_px[j]
            has_valid_2d = bbox_px[2] > bbox_px[0] and bbox_px[3] > bbox_px[1]
            if not has_valid_2d:
                stats["invalid_2d"] += 1

            # ---- 3D params ----
            matched_idx = matched_3d_indices.get(j)
            if matched_idx is None:
                is_valid_3d = False
                invalid_reasons["unmatched_3d"] += 1
                stats["unmatched_3d"] += 1
                dim_j = [-1.0, -1.0, -1.0]
            else:
                dim_j = dimensions[matched_idx]
            dim_j = to_list(dim_j)
            dim_val = dim_j[0] if isinstance(dim_j, (list, np.ndarray)) else float(dim_j)
            is_valid_3d = (matched_idx is not None and dim_val != -1)

            if is_valid_3d:
                cc = as_float_list(center_cam[matched_idx])
                dm = as_float_list(dim_j)
                rc = as_float_matrix(R_cam[matched_idx])
                bbox3d_cam = as_float_matrix(boxes3d[matched_idx])

                is_valid_3d, reason = validate_3d_geometry(
                    category_name=cat_name,
                    center_cam=cc,
                    dimensions=dm,
                    bbox3d_cam=bbox3d_cam,
                    K=img["K"],
                    target_box2d=bbox_px,
                    image_width=W,
                    image_height=H,
                    min_corner_z=min_corner_z,
                    max_projected_abs=max_projected_abs,
                    max_axis_prior_ratio=max_axis_prior_ratio,
                    max_volume_prior_ratio=max_volume_prior_ratio,
                    max_abs_dim=max_abs_dim,
                    min_project_iou=min_project_iou,
                    max_project_center_norm=max_project_center_norm,
                    min_project_area_ratio=min_project_area_ratio,
                    max_project_area_ratio=max_project_area_ratio,
                )
                if not is_valid_3d:
                    is_valid_3d = False
                    invalid_reasons[reason] += 1
                    if reason in {"bad_center_or_dim", "corner_too_close"}:
                        stats["invalid_depth"] += 1
                else:
                    stats["valid_3d"] += 1
            else:
                if matched_idx is not None:
                    invalid_reasons["missing_3d"] += 1
                cc = [-1.0, -1.0, -1.0]
                dm = [-1.0, -1.0, -1.0]
                rc = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]

            if not is_valid_3d:
                stats["invalid_3d"] += 1
                cc = [-1.0, -1.0, -1.0]
                dm = [-1.0, -1.0, -1.0]
                rc = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
                bbox3d_cam = [[-1.0, -1.0, -1.0]] * 8
                bbox2d_proj = bbox_px
                bbox2d_trunc = bbox_px
            else:
                bbox2d_proj, bbox2d_trunc = project_box3d_to_2d(bbox3d_cam, img["K"], W, H)

            ann = {
                "id": int(new_info["id"] * 10000000 + ann_id),
                "image_id": int(im_id),
                "dataset_id": new_info["id"],
                "category_name": str(cat_name),
                "category_id": int(cat_id),
                "valid3D": bool(is_valid_3d),
                "bbox2D_tight": bbox_px,
                "bbox2D_trunc": bbox2d_trunc,
                "bbox2D_proj": bbox2d_proj,
                "bbox3D_cam": bbox3d_cam,
                "center_cam": cc,
                "dimensions": dm,
                "R_cam": rc,
                "behind_camera": False,
                "visibility": 1.0,
                "truncation": 0.0,
                "segmentation_pts": -1,
                "lidar_pts": -1,
                "depth_error": -1,
            }
            annotations.append(ann)
            ann_id += 1

    # 构建输出
    output = {
        "info": new_info,
        "images": filtered_images,
        "categories": categories,
        "annotations": annotations,
    }

    # 保存
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f)

    # 统计报告
    print(f"\n{'='*50}")
    print(f"  Conversion complete: {output_path}")
    print(f"{'='*50}")
    print(f"  Images:       {len(filtered_images)}")
    print(f"  Annotations:  {len(annotations)}")
    print(f"  Total objs:   {stats['total']}")
    print(f"  Valid 3D:     {stats['valid_3d']} ({stats['valid_3d']/max(1,stats['total'])*100:.1f}%)")
    print(f"  Invalid 3D:   {stats['invalid_3d']} ({stats['invalid_3d']/max(1,stats['total'])*100:.1f}%)")
    print(f"  Invalid depth:{stats['invalid_depth']}")
    print(f"  Invalid 2D:   {stats['invalid_2d']}")
    print(f"  Unmatched 3D: {stats['unmatched_3d']}")
    print(f"  Unknown cat:  {stats['unknown_cat']}")
    print(f"  Invalid reasons: {dict(invalid_reasons.most_common())}")
    print(f"{'='*50}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert info_3d.pth to Omni3D JSON")
    parser.add_argument("--pseudo_pth", type=str, required=True,
                        help="Path to info_3d.pth from pseudo label generation")
    parser.add_argument("--source_json", type=str, required=True,
                        help="Path to original Omni3D JSON (e.g. SUNRGBD_train.json)")
    parser.add_argument("--output", type=str, required=True,
                        help="Output JSON path")
    parser.add_argument("--split", type=str, default="Train",
                        help="Split name: Train or Val")
    parser.add_argument("--min_corner_z", type=float, default=0.5,
                        help="Minimum allowed depth for all 3D box corners")
    parser.add_argument("--max_projected_abs", type=float, default=5000.0,
                        help="Maximum absolute projected 3D corner coordinate")
    parser.add_argument("--max_axis_prior_ratio", type=float, default=6.0,
                        help="Maximum per-axis dimension/prior ratio")
    parser.add_argument("--max_volume_prior_ratio", type=float, default=30.0,
                        help="Maximum volume/prior-volume ratio")
    parser.add_argument("--max_abs_dim", type=float, default=8.0,
                        help="Maximum absolute dimension in meters")
    parser.add_argument("--min_project_iou", type=float, default=0.03,
                        help="Minimum IoU between projected 3D box and detector 2D box")
    parser.add_argument("--max_project_center_norm", type=float, default=1.0,
                        help="Maximum projected 3D/2D center offset normalized by 2D-box diagonal")
    parser.add_argument("--min_project_area_ratio", type=float, default=0.05,
                        help="Minimum projected-3D-box area divided by detector-2D-box area")
    parser.add_argument("--max_project_area_ratio", type=float, default=20.0,
                        help="Maximum projected-3D-box area divided by detector-2D-box area")
    args = parser.parse_args()

    convert_info3d_to_omni3d_json(
        pseudo_pth_path=args.pseudo_pth,
        source_json_path=args.source_json,
        output_path=args.output,
        split=args.split,
        min_corner_z=args.min_corner_z,
        max_projected_abs=args.max_projected_abs,
        max_axis_prior_ratio=args.max_axis_prior_ratio,
        max_volume_prior_ratio=args.max_volume_prior_ratio,
        max_abs_dim=args.max_abs_dim,
        min_project_iou=args.min_project_iou,
        max_project_center_norm=args.max_project_center_norm,
        min_project_area_ratio=args.min_project_area_ratio,
        max_project_area_ratio=args.max_project_area_ratio,
    )
