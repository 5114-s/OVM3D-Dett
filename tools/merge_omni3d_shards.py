#!/usr/bin/env python3
import argparse
import json
import os


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    merged = None
    annotations = []
    seen_image_ids = set()
    images = []

    for path in args.inputs:
        with open(path, "r") as f:
            data = json.load(f)
        if merged is None:
            merged = data
        for img in data.get("images", []):
            image_id = img.get("id")
            if image_id not in seen_image_ids:
                images.append(img)
                seen_image_ids.add(image_id)
        annotations.extend(data.get("annotations", []))

    if merged is None:
        raise RuntimeError("No input JSONs loaded")

    for ann_id, ann in enumerate(annotations, 1):
        ann["id"] = ann_id

    merged["images"] = images
    merged["annotations"] = annotations
    merged.setdefault("info", {})
    merged["info"]["merged_from"] = args.inputs

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(merged, f)

    valid3d = sum(1 for ann in annotations if ann.get("valid3D", False))
    print(f"Wrote {args.output}")
    print(f"images={len(images)} annotations={len(annotations)} valid3D={valid3d}")


if __name__ == "__main__":
    main()
