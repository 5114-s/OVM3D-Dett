#!/usr/bin/env python3
import argparse
import copy
import json
import math
import os
from collections import defaultdict

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description="Conservatively blend OVM3D teacher predictions into geometry anchors."
    )
    parser.add_argument("--source_json", required=True)
    parser.add_argument("--teacher_json", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--min_score", type=float, default=0.25)
    parser.add_argument("--min_iou", type=float, default=0.30)
    parser.add_argument("--max_blend", type=float, default=0.35)
    parser.add_argument("--max_center_distance", type=float, default=1.25)
    parser.add_argument("--max_dimension_ratio", type=float, default=4.0)
    return parser.parse_args()


def xyxy_from_annotation(annotation):
    for key in ("bbox2D_tight", "bbox2D_trunc", "bbox2D_proj"):
        box = annotation.get(key)
        if box is not None and len(box) == 4 and float(box[0]) >= 0:
            return np.asarray(box, dtype=np.float64)
    return None


def xyxy_from_prediction(prediction):
    x, y, width, height = [float(v) for v in prediction["bbox"]]
    return np.asarray([x, y, x + width, y + height], dtype=np.float64)


def bbox_iou(box_a, box_b):
    x1 = max(float(box_a[0]), float(box_b[0]))
    y1 = max(float(box_a[1]), float(box_b[1]))
    x2 = min(float(box_a[2]), float(box_b[2]))
    y2 = min(float(box_a[3]), float(box_b[3]))
    intersection = max(x2 - x1, 0.0) * max(y2 - y1, 0.0)
    area_a = max(float(box_a[2] - box_a[0]), 0.0) * max(
        float(box_a[3] - box_a[1]), 0.0
    )
    area_b = max(float(box_b[2] - box_b[0]), 0.0) * max(
        float(box_b[3] - box_b[1]), 0.0
    )
    union = area_a + area_b - intersection
    return float(intersection / union) if union > 0 else 0.0


def project_rotation(matrix):
    u, _, vh = np.linalg.svd(np.asarray(matrix, dtype=np.float64).reshape(3, 3))
    rotation = u @ vh
    if np.linalg.det(rotation) < 0:
        u[:, -1] *= -1
        rotation = u @ vh
    return rotation


def box_vertices(center, dimensions, rotation):
    center = np.asarray(center, dtype=np.float64).reshape(3)
    width, height, length = np.asarray(dimensions, dtype=np.float64).reshape(3)
    local = np.asarray(
        [
            [-length / 2, -height / 2, -width / 2],
            [length / 2, -height / 2, -width / 2],
            [length / 2, height / 2, -width / 2],
            [-length / 2, height / 2, -width / 2],
            [-length / 2, -height / 2, width / 2],
            [length / 2, -height / 2, width / 2],
            [length / 2, height / 2, width / 2],
            [-length / 2, height / 2, width / 2],
        ],
        dtype=np.float64,
    )
    return local @ np.asarray(rotation, dtype=np.float64).reshape(3, 3).T + center


def valid_geometry(center, dimensions, rotation):
    return (
        np.all(np.isfinite(center))
        and np.all(np.isfinite(dimensions))
        and np.all(np.isfinite(rotation))
        and float(center[2]) > 0.05
        and np.all(dimensions > 0.01)
    )


def main():
    args = parse_args()
    with open(args.source_json, "r") as handle:
        source = json.load(handle)
    with open(args.teacher_json, "r") as handle:
        teacher_predictions = json.load(handle)

    prediction_index = defaultdict(list)
    for prediction in teacher_predictions:
        if float(prediction.get("score", 0.0)) < float(args.min_score):
            continue
        key = (int(prediction["image_id"]), int(prediction["category_id"]))
        prediction_index[key].append((prediction, xyxy_from_prediction(prediction)))

    output = copy.deepcopy(source)
    matched = 0
    rejected_geometry = 0
    used_predictions = set()

    for annotation in output.get("annotations", []):
        if not bool(annotation.get("valid3D", True)):
            continue
        source_box = xyxy_from_annotation(annotation)
        if source_box is None:
            continue
        key = (int(annotation["image_id"]), int(annotation["category_id"]))
        best = None
        for prediction, prediction_box in prediction_index.get(key, []):
            prediction_id = id(prediction)
            if prediction_id in used_predictions:
                continue
            overlap = bbox_iou(source_box, prediction_box)
            if best is None or overlap > best[0]:
                best = (overlap, prediction, prediction_id)
        if best is None or best[0] < float(args.min_iou):
            continue

        overlap, prediction, prediction_id = best
        center_source = np.asarray(annotation["center_cam"], dtype=np.float64)
        dims_source = np.asarray(annotation["dimensions"], dtype=np.float64)
        rotation_source = np.asarray(annotation["R_cam"], dtype=np.float64)
        center_teacher = np.asarray(prediction["center_cam"], dtype=np.float64)
        dims_teacher = np.asarray(prediction["dimensions"], dtype=np.float64)
        rotation_teacher = np.asarray(prediction["pose"], dtype=np.float64)
        if not valid_geometry(center_teacher, dims_teacher, rotation_teacher):
            rejected_geometry += 1
            continue

        center_distance = float(np.linalg.norm(center_teacher - center_source))
        dimension_ratio = float(
            np.max(
                np.maximum(
                    dims_teacher / np.maximum(dims_source, 1e-4),
                    dims_source / np.maximum(dims_teacher, 1e-4),
                )
            )
        )
        if (
            center_distance > float(args.max_center_distance)
            or dimension_ratio > float(args.max_dimension_ratio)
        ):
            rejected_geometry += 1
            continue

        source_confidence = float(annotation.get("pseudo_weight_joint", 0.5))
        teacher_score = float(prediction["score"])
        blend = float(args.max_blend) * math.sqrt(
            max(overlap * teacher_score * source_confidence, 0.0)
        )
        blend = float(np.clip(blend, 0.0, float(args.max_blend)))
        center = (1.0 - blend) * center_source + blend * center_teacher
        dimensions = np.exp(
            (1.0 - blend) * np.log(np.maximum(dims_source, 1e-4))
            + blend * np.log(np.maximum(dims_teacher, 1e-4))
        )
        rotation = project_rotation(
            (1.0 - blend) * rotation_source + blend * rotation_teacher
        )

        annotation["center_cam"] = [float(v) for v in center]
        annotation["dimensions"] = [float(v) for v in dimensions]
        annotation["R_cam"] = [[float(v) for v in row] for row in rotation]
        annotation["bbox3D_cam"] = [
            [float(v) for v in row]
            for row in box_vertices(center, dimensions, rotation)
        ]
        annotation["teacher_anchor_matched"] = True
        annotation["teacher_anchor_iou"] = float(overlap)
        annotation["teacher_anchor_score"] = teacher_score
        annotation["teacher_anchor_blend"] = blend
        annotation["teacher_anchor_center_distance"] = center_distance
        annotation["teacher_anchor_dimension_ratio"] = dimension_ratio
        used_predictions.add(prediction_id)
        matched += 1

    output.setdefault("info", {})
    output["info"]["teacher_anchor_source"] = os.path.abspath(args.teacher_json)
    output["info"]["teacher_anchor_matched"] = matched
    output["info"]["teacher_anchor_rejected_geometry"] = rejected_geometry
    output["info"]["teacher_anchor_max_blend"] = float(args.max_blend)

    os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
    with open(args.output_json, "w") as handle:
        json.dump(output, handle)
    print(f"Wrote {args.output_json}")
    print(
        {
            "annotations": len(output.get("annotations", [])),
            "teacher_matched": matched,
            "teacher_rejected_geometry": rejected_geometry,
        }
    )


if __name__ == "__main__":
    main()
