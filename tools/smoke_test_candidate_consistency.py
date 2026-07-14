#!/usr/bin/env python3
"""Verify the candidate-consistency ablation without touching dataset caches."""

import os
import sys
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cubercnn.generate_label import process_indoor, process_outdoor


POINTS = np.array(
    [
        [-0.10, -0.10, 2.00],
        [0.10, 0.10, 2.10],
        [-0.05, 0.05, 2.20],
        [0.08, -0.08, 2.15],
    ],
    dtype=np.float32,
)
PRIOR = np.array([1.0, 1.0, 2.0], dtype=np.float32)


def _candidate_boxes(*_args):
    # The unchanged objective below selects the first candidate [dx=2, dz=1],
    # while the released implementation exports dimensions left over from the
    # final candidate [dx=1, dz=2].
    return [
        [-1.0, 1.0, -0.5, 0.5],
        [-0.5, 0.5, -1.0, 1.0],
    ]


def _ray_score(wl, *_args):
    return 0.0 if float(wl[0]) < float(wl[1]) else 100.0


@contextmanager
def _fix_enabled(enabled):
    key = "OVM3D_FIX_CANDIDATE_CONSISTENCY"
    previous = os.environ.get(key)
    os.environ[key] = "1" if enabled else "0"
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = previous


def _run(module, enabled):
    with _fix_enabled(enabled), patch.object(
        module, "generate_possible_bboxs", side_effect=_candidate_boxes
    ), patch.object(module, "calc_inside_ratio", return_value=1.0), patch.object(
        module, "calc_dis_ray_tracing", side_effect=_ray_score
    ):
        if module is process_outdoor:
            return module.estimate_bbox(POINTS.copy(), PRIOR, "car", None)
        return module.estimate_bbox(POINTS.copy(), PRIOR, None)


def _assert_module(module):
    baseline = _run(module, enabled=False)
    fixed = _run(module, enabled=True)

    baseline_vertices, baseline_center, baseline_dims, baseline_rotation = (
        baseline[0][0],
        baseline[1][0],
        np.asarray(baseline[2][0], dtype=np.float32),
        baseline[3][0],
    )
    fixed_vertices, fixed_center, fixed_dims, fixed_rotation = (
        fixed[0][0],
        fixed[1][0],
        np.asarray(fixed[2][0], dtype=np.float32),
        fixed[3][0],
    )

    # The ablation must not change candidate generation, selection, center, or
    # rotation.  It only exports dimensions belonging to the selected box.
    np.testing.assert_allclose(fixed_vertices, baseline_vertices)
    np.testing.assert_allclose(fixed_center, baseline_center)
    np.testing.assert_allclose(fixed_rotation, baseline_rotation)
    np.testing.assert_allclose(baseline_dims, [2.0, 1.0, 1.0])
    np.testing.assert_allclose(fixed_dims, [1.0, 1.0, 2.0])

    local_vertices = (fixed_vertices - fixed_center) @ fixed_rotation
    local_extents = np.ptp(local_vertices, axis=0)
    dimensions_from_vertices = local_extents[[2, 1, 0]]
    np.testing.assert_allclose(fixed_dims, dimensions_from_vertices, atol=2e-3)


def _assert_all_infinite_ray_fallback(module):
    """Sparse masks must still select a consistent original candidate."""
    with _fix_enabled(True), patch.object(
        module, "generate_possible_bboxs", side_effect=_candidate_boxes
    ), patch.object(
        module, "calc_inside_ratio", side_effect=[0.8, 0.2]
    ), patch.object(
        module, "calc_dis_ray_tracing", return_value=float("inf")
    ):
        if module is process_outdoor:
            result = module.estimate_bbox(POINTS.copy(), PRIOR, "car", None)
        else:
            result = module.estimate_bbox(POINTS.copy(), PRIOR, None)

    vertices = np.asarray(result[0][0], dtype=np.float32)
    center = np.asarray(result[1][0], dtype=np.float32)
    dimensions = np.asarray(result[2][0], dtype=np.float32)
    rotation = np.asarray(result[3][0], dtype=np.float32)
    local_extents = np.ptp((vertices - center) @ rotation, axis=0)
    np.testing.assert_allclose(dimensions, local_extents[[2, 1, 0]], atol=2e-3)
    np.testing.assert_allclose(dimensions, [1.0, 1.0, 2.0])


def main():
    _assert_module(process_indoor)
    _assert_module(process_outdoor)
    _assert_all_infinite_ray_fallback(process_indoor)
    _assert_all_infinite_ray_fallback(process_outdoor)
    print(
        "candidate-consistency smoke test: PASS "
        "(indoor + outdoor + all-inf fallback)"
    )


if __name__ == "__main__":
    main()
