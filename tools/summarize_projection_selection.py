#!/usr/bin/env python3
"""Summarize candidate-switch statistics stored in generated info_3d files."""

import argparse
import json
from pathlib import Path

import torch


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--work-root", required=True)
    parser.add_argument("--dataset", default="SUNRGBD")
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    report = {"dataset": args.dataset, "splits": {}}
    total = {
        "instances": 0,
        "eligible": 0,
        "switched": 0,
        "kept_original": 0,
        "invalid": 0,
    }
    for split in ("train", "val"):
        path = Path(args.work_root) / args.dataset / split / "info_3d.pth"
        info = torch.load(path, map_location="cpu")
        summary = dict(info.get("_projection_selection_summary", {}))
        report["splits"][split] = summary
        for key in total:
            total[key] += int(summary.get(key, 0))

    total["switch_rate_among_eligible"] = (
        float(total["switched"] / total["eligible"])
        if total["eligible"] > 0
        else 0.0
    )
    report["total"] = total
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

