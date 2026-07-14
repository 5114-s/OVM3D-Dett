#!/usr/bin/env python
"""CPU smoke tests for depth-aware projective candidate selection."""

import copy
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cubercnn.generate_label.projection_selection import (
    get_projection_selection_cfg,
    score_projected_candidate,
    select_projected_candidate,
)
from cubercnn.generate_label.util import convert_box_vertices


def candidate(center, dimensions, inside_ratio=0.9):
    center = np.asarray(center, dtype=np.float64)
    dimensions = np.asarray(dimensions, dtype=np.float64)
    vertices = convert_box_vertices(
        center[0],
        center[1],
        center[2],
        dimensions[0],
        dimensions[1],
        dimensions[2],
        0.0,
    )
    return {
        "vertices_cam": vertices,
        "center_cam": center,
        "R_cam": np.eye(3, dtype=np.float64),
        "dimensions_xyz": dimensions,
        "inside_ratio": float(inside_ratio),
    }


def main():
    height, width = 96, 128
    K = np.array(
        [[100.0, 0.0, 64.0], [0.0, 100.0, 48.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    raw_mask = np.zeros((height, width), dtype=np.uint8)
    core_mask = np.zeros_like(raw_mask)
    # The z=4 front face of a 2m cube centered at z=5 projects here.
    cv2.rectangle(raw_mask, (38, 22), (90, 74), 1, thickness=-1)
    cv2.rectangle(core_mask, (42, 26), (86, 70), 1, thickness=-1)
    depth = np.zeros((height, width), dtype=np.float64)
    depth[core_mask > 0] = 4.0

    good = candidate([0.0, 0.0, 5.0], [2.0, 2.0, 2.0])
    bad = candidate([1.25, 0.0, 6.0], [2.0, 2.0, 2.0])
    cfg = get_projection_selection_cfg()
    cfg.update(
        {
            "enabled": True,
            "switch_margin": 0.01,
            "min_score": 0.05,
            "max_inside_drop": 0.05,
        }
    )

    good_score = score_projected_candidate(
        good, raw_mask, core_mask, depth, K, cfg
    )
    bad_score = score_projected_candidate(
        bad, raw_mask, core_mask, depth, K, cfg
    )
    assert good_score["valid"] and bad_score["valid"]
    assert good_score["score"] > bad_score["score"], (good_score, bad_score)
    assert good_score["depth_error"] < bad_score["depth_error"]

    selected, metric = select_projected_candidate(
        [bad, good], 0, raw_mask, core_mask, depth, K, cfg
    )
    assert selected == 1, metric
    assert metric["switched"] and metric["reason"] == "projection_improved"

    conservative_cfg = copy.deepcopy(cfg)
    conservative_cfg["switch_margin"] = 2.0
    selected, metric = select_projected_candidate(
        [bad, good], 0, raw_mask, core_mask, depth, K, conservative_cfg
    )
    assert selected == 0, metric
    assert not metric["switched"] and metric["reason"] == "kept_original"

    disabled_cfg = copy.deepcopy(cfg)
    disabled_cfg["enabled"] = False
    selected, metric = select_projected_candidate(
        [bad, good], 0, raw_mask, core_mask, depth, K, disabled_cfg
    )
    assert selected == 0 and metric["reason"] == "disabled"

    print("projection-selection smoke test: PASS")
    print(
        "scores:",
        {
            "bad": round(float(bad_score["score"]), 4),
            "good": round(float(good_score["score"]), 4),
        },
    )


if __name__ == "__main__":
    main()
