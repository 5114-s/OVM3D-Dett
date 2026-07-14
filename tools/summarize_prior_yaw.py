#!/usr/bin/env python3
"""Summarize prior-yaw decisions stored in generated info_3d files."""

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--work-root", required=True)
    parser.add_argument("--dataset", default="SUNRGBD")
    parser.add_argument("--output", required=True)
    parser.add_argument("--splits", nargs="+", default=["train", "val"])
    return parser.parse_args()


def summarize_split(info):
    decisions = []
    for image_id, image_info in info.items():
        if isinstance(image_id, str) and image_id.startswith("_"):
            continue
        if not isinstance(image_info, dict):
            continue
        decisions.extend(image_info.get("prior_yaw_stats", []))

    reasons = Counter(str(item.get("reason", "missing")) for item in decisions)
    eligible = [item for item in decisions if item.get("eligible", False)]
    switched = [item for item in eligible if item.get("switched", False)]
    improvements = [
        float(item["score_improvement"])
        for item in switched
        if np.isfinite(item.get("score_improvement", np.nan))
    ]
    eigen_ratios = [
        float(item["pca_eigen_ratio"])
        for item in switched
        if np.isfinite(item.get("pca_eigen_ratio", np.nan))
    ]
    return {
        "instances": len(decisions),
        "eligible": len(eligible),
        "switched": len(switched),
        "switch_rate_among_eligible": (
            float(len(switched) / len(eligible)) if eligible else 0.0
        ),
        "reasons": dict(sorted(reasons.items())),
        "switched_improvement_median": (
            float(np.median(improvements)) if improvements else None
        ),
        "switched_pca_eigen_ratio_median": (
            float(np.median(eigen_ratios)) if eigen_ratios else None
        ),
    }


def main():
    args = parse_args()
    report = {"dataset": args.dataset, "splits": {}}
    for split in args.splits:
        path = Path(args.work_root) / args.dataset / split / "info_3d.pth"
        info = torch.load(path, map_location="cpu")
        report["splits"][split] = summarize_split(info)

    total_instances = sum(v["instances"] for v in report["splits"].values())
    total_eligible = sum(v["eligible"] for v in report["splits"].values())
    total_switched = sum(v["switched"] for v in report["splits"].values())
    report["total"] = {
        "instances": total_instances,
        "eligible": total_eligible,
        "switched": total_switched,
        "switch_rate_among_eligible": (
            float(total_switched / total_eligible) if total_eligible else 0.0
        ),
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
