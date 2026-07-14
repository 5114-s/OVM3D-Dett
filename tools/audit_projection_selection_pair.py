#!/usr/bin/env python3
"""Verify that a paired projection-selection dataset is a single-variable edit."""

import argparse
import json
from pathlib import Path

import numpy as np


GEOMETRY_FIELDS = {"bbox3D_cam", "center_cam", "dimensions"}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-dir", required=True)
    parser.add_argument("--projection-dir", required=True)
    parser.add_argument("--dataset", default="SUNRGBD")
    parser.add_argument("--output", required=True)
    parser.add_argument("--dimension-tolerance", type=float, default=0.01)
    return parser.parse_args()


def load_json(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def main():
    args = parse_args()
    report = {"dataset": args.dataset, "splits": {}}
    totals = {
        "annotations": 0,
        "changed_geometry": 0,
        "unchanged": 0,
        "non_geometry_mismatches": 0,
        "invalid_changed_geometry": 0,
        "changed_dimension_mismatches": 0,
    }

    for split in ("train", "val"):
        base_path = Path(args.base_dir) / f"{args.dataset}_{split}.json"
        projection_path = (
            Path(args.projection_dir) / f"{args.dataset}_{split}.json"
        )
        base = load_json(base_path)
        projection = load_json(projection_path)
        for key in ("info", "images", "categories"):
            if base.get(key) != projection.get(key):
                raise AssertionError(f"{split}: top-level {key} differs")

        base_annotations = base.get("annotations", [])
        projection_annotations = projection.get("annotations", [])
        if len(base_annotations) != len(projection_annotations):
            raise AssertionError(f"{split}: annotation counts differ")

        split_counts = {
            "annotations": len(base_annotations),
            "changed_geometry": 0,
            "unchanged": 0,
            "non_geometry_mismatches": 0,
            "invalid_changed_geometry": 0,
            "changed_dimension_mismatches": 0,
        }
        for base_ann, projection_ann in zip(
            base_annotations, projection_annotations
        ):
            if base_ann.get("id") != projection_ann.get("id"):
                raise AssertionError(f"{split}: annotation order/id differs")
            keys = set(base_ann) | set(projection_ann)
            non_geometry_changed = [
                key
                for key in keys - GEOMETRY_FIELDS
                if base_ann.get(key) != projection_ann.get(key)
            ]
            if non_geometry_changed:
                split_counts["non_geometry_mismatches"] += 1
                continue

            geometry_changed = any(
                base_ann.get(key) != projection_ann.get(key)
                for key in GEOMETRY_FIELDS
            )
            if not geometry_changed:
                split_counts["unchanged"] += 1
                continue
            split_counts["changed_geometry"] += 1

            vertices = np.asarray(
                projection_ann.get("bbox3D_cam"), dtype=np.float64
            )
            center = np.asarray(
                projection_ann.get("center_cam"), dtype=np.float64
            )
            dimensions = np.asarray(
                projection_ann.get("dimensions"), dtype=np.float64
            )
            rotation = np.asarray(
                projection_ann.get("R_cam"), dtype=np.float64
            )
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
                split_counts["invalid_changed_geometry"] += 1
                continue

            local_vertices = (vertices - center[None, :]) @ rotation
            extents_xyz = np.ptp(local_vertices, axis=0)
            derived_whl = extents_xyz[[2, 1, 0]]
            if not np.allclose(
                dimensions,
                derived_whl,
                atol=float(args.dimension_tolerance),
                rtol=0.0,
            ):
                split_counts["changed_dimension_mismatches"] += 1

        report["splits"][split] = split_counts
        for key in totals:
            totals[key] += int(split_counts[key])

    report["total"] = totals
    if totals["non_geometry_mismatches"] != 0:
        raise AssertionError("non-geometry annotation fields changed")
    if totals["invalid_changed_geometry"] != 0:
        raise AssertionError("invalid geometry found among changed annotations")
    if totals["changed_dimension_mismatches"] != 0:
        raise AssertionError("changed candidate dimensions are inconsistent")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

