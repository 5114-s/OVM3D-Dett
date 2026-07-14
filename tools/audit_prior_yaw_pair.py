#!/usr/bin/env python3
"""Verify that a paired yaw dataset changes only valid 3D geometry."""

import argparse
import json
from pathlib import Path

import numpy as np


GEOMETRY_FIELDS = {"bbox3D_cam", "center_cam", "dimensions", "R_cam"}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-dir", required=True)
    parser.add_argument("--yaw-dir", required=True)
    parser.add_argument("--dataset", default="SUNRGBD")
    parser.add_argument("--output", required=True)
    parser.add_argument("--dimension-tolerance", type=float, default=0.02)
    return parser.parse_args()


def load_json(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def audit_split(base, yaw, tolerance):
    for key in ("info", "images", "categories"):
        if base.get(key) != yaw.get(key):
            raise AssertionError(f"top-level {key} differs")
    base_annotations = base.get("annotations", [])
    yaw_annotations = yaw.get("annotations", [])
    if len(base_annotations) != len(yaw_annotations):
        raise AssertionError("annotation counts differ")

    counts = {
        "annotations": len(base_annotations),
        "changed_geometry": 0,
        "changed_vertices": 0,
        "changed_center": 0,
        "changed_dimensions": 0,
        "changed_rotation": 0,
        "unchanged": 0,
        "non_geometry_mismatches": 0,
        "invalid_changed_geometry": 0,
        "changed_dimension_mismatches": 0,
    }
    for base_ann, yaw_ann in zip(base_annotations, yaw_annotations):
        if base_ann.get("id") != yaw_ann.get("id"):
            raise AssertionError("annotation order/id differs")
        keys = set(base_ann) | set(yaw_ann)
        if any(
            base_ann.get(key) != yaw_ann.get(key)
            for key in keys - GEOMETRY_FIELDS
        ):
            counts["non_geometry_mismatches"] += 1
            continue
        if not any(
            base_ann.get(key) != yaw_ann.get(key) for key in GEOMETRY_FIELDS
        ):
            counts["unchanged"] += 1
            continue

        counts["changed_geometry"] += 1
        if base_ann.get("bbox3D_cam") != yaw_ann.get("bbox3D_cam"):
            counts["changed_vertices"] += 1
        if base_ann.get("center_cam") != yaw_ann.get("center_cam"):
            counts["changed_center"] += 1
        if base_ann.get("dimensions") != yaw_ann.get("dimensions"):
            counts["changed_dimensions"] += 1
        if base_ann.get("R_cam") != yaw_ann.get("R_cam"):
            counts["changed_rotation"] += 1
        vertices = np.asarray(yaw_ann.get("bbox3D_cam"), dtype=np.float64)
        center = np.asarray(yaw_ann.get("center_cam"), dtype=np.float64)
        dimensions = np.asarray(yaw_ann.get("dimensions"), dtype=np.float64)
        rotation = np.asarray(yaw_ann.get("R_cam"), dtype=np.float64)
        valid = (
            vertices.shape == (8, 3)
            and center.shape == (3,)
            and dimensions.shape == (3,)
            and rotation.shape == (3, 3)
            and np.all(np.isfinite(vertices))
            and np.all(np.isfinite(center))
            and np.all(np.isfinite(dimensions))
            and np.all(np.isfinite(rotation))
            and np.all(dimensions > 0)
        )
        if not valid:
            counts["invalid_changed_geometry"] += 1
            continue

        local_vertices = (vertices - center[None, :]) @ rotation
        derived_whl = np.ptp(local_vertices, axis=0)[[2, 1, 0]]
        if not np.allclose(
            dimensions, derived_whl, atol=float(tolerance), rtol=0.0
        ):
            counts["changed_dimension_mismatches"] += 1
    return counts


def main():
    args = parse_args()
    report = {"dataset": args.dataset, "splits": {}}
    total = None
    for split in ("train", "val"):
        base = load_json(Path(args.base_dir) / f"{args.dataset}_{split}.json")
        yaw = load_json(Path(args.yaw_dir) / f"{args.dataset}_{split}.json")
        counts = audit_split(base, yaw, args.dimension_tolerance)
        report["splits"][split] = counts
        if total is None:
            total = {key: 0 for key in counts}
        for key, value in counts.items():
            total[key] += int(value)
    report["total"] = total

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    print(json.dumps(report, indent=2, ensure_ascii=False))

    if total["non_geometry_mismatches"]:
        raise AssertionError("non-geometry annotation fields changed")
    if total["invalid_changed_geometry"]:
        raise AssertionError("invalid geometry found among changed labels")
    if total["changed_dimension_mismatches"]:
        raise AssertionError("changed yaw labels have inconsistent dimensions")


if __name__ == "__main__":
    main()
