#!/usr/bin/env python3
"""Repair only width/length-inconsistent dimensions in Omni3D pseudo labels.

The selected cuboid vertices, center, and rotation already describe the chosen
candidate.  This tool derives ``dimensions`` from those fields and updates an
annotation only when its old dimensions are a pure width/length swap.  Every
other dataset field is verified to remain semantically identical.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
from typing import Any

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a dimensions-only candidate-consistency ablation."
    )
    parser.add_argument(
        "--source_dir",
        default="datasets/Omni3D_pl-1",
        help="Directory containing the original SUNRGBD pseudo-label JSONs.",
    )
    parser.add_argument(
        "--output_dir",
        default="datasets/Omni3D_pl-candidate-consistency-only",
        help="New directory; source JSONs are never modified.",
    )
    parser.add_argument(
        "--dataset",
        default="SUNRGBD",
        help="Dataset filename prefix.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "val"],
        help="Dataset splits to repair.",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.01,
        help="Maximum dimension/corner disagreement considered numerical noise (m).",
    )
    return parser.parse_args()


def dimensions_from_cuboid(annotation: dict[str, Any]) -> np.ndarray | None:
    vertices = np.asarray(annotation.get("bbox3D_cam", []), dtype=np.float64)
    center = np.asarray(annotation.get("center_cam", []), dtype=np.float64)
    rotation = np.asarray(annotation.get("R_cam", []), dtype=np.float64)
    if vertices.shape != (8, 3) or center.shape != (3,) or rotation.shape != (3, 3):
        return None
    if not (
        np.all(np.isfinite(vertices))
        and np.all(np.isfinite(center))
        and np.all(np.isfinite(rotation))
    ):
        return None

    # OVM3D stores cuboid dimensions as [width, height, length] = [dz, dy, dx].
    local_extents_xyz = np.ptp((vertices - center) @ rotation, axis=0)
    dimensions = local_extents_xyz[[2, 1, 0]]
    if not np.all(np.isfinite(dimensions)) or np.any(dimensions <= 0.0):
        return None
    return dimensions


def repair_dataset(data: dict[str, Any], tolerance: float) -> tuple[dict[str, Any], dict[str, Any]]:
    repaired = copy.deepcopy(data)
    stats: dict[str, Any] = {
        "annotations": len(data.get("annotations", [])),
        "valid_geometry": 0,
        "already_consistent": 0,
        "repaired_width_length_swaps": 0,
        "invalid_geometry_skipped": 0,
        "non_swap_mismatches": 0,
        "repaired_annotation_ids": [],
    }

    for source_ann, output_ann in zip(
        data.get("annotations", []), repaired.get("annotations", [])
    ):
        old_dimensions = np.asarray(source_ann.get("dimensions", []), dtype=np.float64)
        expected_dimensions = dimensions_from_cuboid(source_ann)
        if (
            old_dimensions.shape != (3,)
            or expected_dimensions is None
            or not np.all(np.isfinite(old_dimensions))
            or np.any(old_dimensions <= 0.0)
        ):
            stats["invalid_geometry_skipped"] += 1
            continue

        stats["valid_geometry"] += 1
        max_error = float(np.max(np.abs(old_dimensions - expected_dimensions)))
        if max_error <= tolerance:
            stats["already_consistent"] += 1
            continue

        swapped_dimensions = old_dimensions[[2, 1, 0]]
        if not np.allclose(
            swapped_dimensions,
            expected_dimensions,
            atol=tolerance,
            rtol=0.0,
        ):
            stats["non_swap_mismatches"] += 1
            continue

        output_ann["dimensions"] = [float(value) for value in expected_dimensions]
        stats["repaired_width_length_swaps"] += 1
        stats["repaired_annotation_ids"].append(int(source_ann["id"]))

    return repaired, stats


def assert_only_expected_dimensions_changed(
    source: dict[str, Any],
    output: dict[str, Any],
    repaired_ids: set[int],
    tolerance: float,
) -> None:
    for top_level_key in ("info", "images", "categories"):
        if source.get(top_level_key) != output.get(top_level_key):
            raise AssertionError(f"Unexpected change in top-level field: {top_level_key}")

    source_annotations = source.get("annotations", [])
    output_annotations = output.get("annotations", [])
    if len(source_annotations) != len(output_annotations):
        raise AssertionError("Annotation count changed")

    changed_ids: set[int] = set()
    remaining_mismatches = 0
    for source_ann, output_ann in zip(source_annotations, output_annotations):
        ann_id = int(source_ann["id"])
        if source_ann.keys() != output_ann.keys():
            raise AssertionError(f"Annotation keys changed for id={ann_id}")

        for key in source_ann:
            if key == "dimensions":
                continue
            if source_ann[key] != output_ann[key]:
                raise AssertionError(f"Unexpected {key} change for id={ann_id}")

        if source_ann["dimensions"] != output_ann["dimensions"]:
            changed_ids.add(ann_id)

        expected_dimensions = dimensions_from_cuboid(output_ann)
        output_dimensions = np.asarray(output_ann.get("dimensions", []), dtype=np.float64)
        if (
            expected_dimensions is not None
            and output_dimensions.shape == (3,)
            and np.all(output_dimensions > 0.0)
            and np.max(np.abs(output_dimensions - expected_dimensions)) > tolerance
        ):
            remaining_mismatches += 1

    if changed_ids != repaired_ids:
        raise AssertionError(
            "Changed annotation ids do not match the planned repair set: "
            f"changed={len(changed_ids)}, planned={len(repaired_ids)}"
        )
    if remaining_mismatches:
        raise AssertionError(
            f"Found {remaining_mismatches} remaining vertex/dimension mismatches"
        )


def atomic_json_dump(data: dict[str, Any], output_path: Path) -> None:
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    with temporary_path.open("w") as handle:
        json.dump(data, handle)
    os.replace(temporary_path, output_path)


def main() -> None:
    args = parse_args()
    source_dir = Path(args.source_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    report: dict[str, Any] = {
        "source_dir": str(source_dir.resolve()),
        "output_dir": str(output_dir.resolve()),
        "dataset": args.dataset,
        "tolerance_m": float(args.tolerance),
        "splits": {},
    }

    for split in args.splits:
        filename = f"{args.dataset}_{split}.json"
        source_path = source_dir / filename
        output_path = output_dir / filename
        with source_path.open("r") as handle:
            source_data = json.load(handle)

        output_data, stats = repair_dataset(source_data, args.tolerance)
        repaired_ids = set(stats.pop("repaired_annotation_ids"))
        if stats["non_swap_mismatches"]:
            raise RuntimeError(
                f"{filename}: found {stats['non_swap_mismatches']} non-swap mismatches; "
                "refusing to modify an impure set"
            )

        assert_only_expected_dimensions_changed(
            source_data, output_data, repaired_ids, args.tolerance
        )
        atomic_json_dump(output_data, output_path)
        stats["output_file"] = str(output_path.resolve())
        report["splits"][split] = stats
        print(
            f"{split}: repaired {stats['repaired_width_length_swaps']} / "
            f"{stats['valid_geometry']} valid annotations"
        )

    report_path = output_dir / "candidate_consistency_report.json"
    atomic_json_dump(report, report_path)
    total_repaired = sum(
        split_stats["repaired_width_length_swaps"]
        for split_stats in report["splits"].values()
    )
    print(f"total repaired: {total_repaired}")
    print(f"report: {report_path}")


if __name__ == "__main__":
    main()
