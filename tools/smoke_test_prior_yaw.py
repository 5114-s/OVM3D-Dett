#!/usr/bin/env python
"""CPU smoke tests for prior-conditioned dual-axis yaw selection."""

import copy
import math
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cubercnn.generate_label.yaw_disambiguation import (
    get_prior_yaw_cfg,
    select_prior_conditioned_yaw,
)


def planar_strip(span_x, span_z, count=240):
    rng = np.random.RandomState(7)
    return np.stack(
        [
            rng.uniform(-span_x / 2.0, span_x / 2.0, count),
            rng.uniform(-0.4, 0.4, count),
            rng.uniform(-span_z / 2.0, span_z / 2.0, count),
        ],
        axis=1,
    )


def main():
    prior = np.array([0.8, 1.0, 2.0], dtype=np.float64)  # w, h, l
    cfg = get_prior_yaw_cfg()
    cfg.update({"enabled": True})

    # A visible side spans the physical length: released PCA axis is correct.
    selected, side_metric = select_prior_conditioned_yaw(
        planar_strip(2.0, 0.08), prior, 0.0, cfg
    )
    assert math.isclose(selected, 0.0, abs_tol=1e-12), side_metric
    assert not side_metric["switched"]

    # A front/rear surface spans width: PCA calls it length, so swap axes.
    selected, rear_metric = select_prior_conditioned_yaw(
        planar_strip(0.8, 0.08), prior, 0.0, cfg
    )
    assert math.isclose(selected, math.pi / 2.0, abs_tol=1e-12), rear_metric
    assert rear_metric["switched"]
    assert rear_metric["reason"] == "prior_axis_improved"
    assert rear_metric["alternate_score"] > rear_metric["original_score"]

    # Near-square priors do not contain reliable length/width information.
    selected, square_metric = select_prior_conditioned_yaw(
        planar_strip(0.8, 0.08), [1.0, 1.0, 1.05], 0.0, cfg
    )
    assert math.isclose(selected, 0.0, abs_tol=1e-12), square_metric
    assert square_metric["reason"] == "near_square_prior"

    conservative = copy.deepcopy(cfg)
    conservative["switch_margin"] = 2.0
    selected, conservative_metric = select_prior_conditioned_yaw(
        planar_strip(0.8, 0.08), prior, 0.0, conservative
    )
    assert math.isclose(selected, 0.0, abs_tol=1e-12), conservative_metric
    assert conservative_metric["reason"] == "kept_original"

    disabled = copy.deepcopy(cfg)
    disabled["enabled"] = False
    selected, disabled_metric = select_prior_conditioned_yaw(
        planar_strip(0.8, 0.08), prior, 0.37, disabled
    )
    assert math.isclose(selected, 0.37, abs_tol=1e-12)
    assert disabled_metric["reason"] == "disabled"

    # SCC is an unoriented rectangle criterion: the two axis assignments must
    # have equal quality (up to floating-point error), hence the physical prior.
    assert math.isclose(
        rear_metric["original_scc_quality"],
        rear_metric["alternate_scc_quality"],
        rel_tol=0.0,
        abs_tol=1e-12,
    )

    print("prior-conditioned yaw smoke test: PASS")
    print(
        "rear scores:",
        {
            "original": round(rear_metric["original_score"], 4),
            "alternate": round(rear_metric["alternate_score"], 4),
            "scc_quality": round(rear_metric["scc_quality"], 4),
        },
    )


if __name__ == "__main__":
    main()
