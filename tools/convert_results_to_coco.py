"""
将 results.json 转换为标准 COCO 格式

COCO 格式结构:
{
    "images": [...],      # 图片信息
    "annotations": [...],  # 标注信息
    "categories": [...]   # 类别信息
}
"""

import json
import os
from pathlib import Path

# SUNRGBD 38 类别的类别映射
SUNRGBD_CATEGORIES = {
    'bicycle': 11, 'books': 14, 'bottle': 15, 'chair': 18, 'cup': 19,
    'laptop': 20, 'shoes': 21, 'towel': 22, 'blinds': 23, 'window': 24,
    'lamp': 25, 'shelves': 26, 'mirror': 27, 'sink': 28, 'cabinet': 29,
    'bathtub': 30, 'door': 31, 'toilet': 32, 'desk': 33, 'box': 34,
    'bookcase': 35, 'picture': 36, 'table': 37, 'counter': 38, 'bed': 39,
    'night stand': 40, 'pillow': 42, 'sofa': 43, 'television': 44,
    'floor mat': 45, 'curtain': 46, 'clothes': 47, 'stationery': 48,
    'refrigerator': 49, 'bin': 52, 'stove': 53, 'oven': 57, 'machine': 61
}

# COCO categories 格式
COCO_CATEGORIES = [
    {"id": 11, "name": "bicycle", "supercategory": "object"},
    {"id": 14, "name": "books", "supercategory": "object"},
    {"id": 15, "name": "bottle", "supercategory": "object"},
    {"id": 18, "name": "chair", "supercategory": "furniture"},
    {"id": 19, "name": "cup", "supercategory": "object"},
    {"id": 20, "name": "laptop", "supercategory": "electronics"},
    {"id": 21, "name": "shoes", "supercategory": "object"},
    {"id": 22, "name": "towel", "supercategory": "object"},
    {"id": 23, "name": "blinds", "supercategory": "window"},
    {"id": 24, "name": "window", "supercategory": "furniture"},
    {"id": 25, "name": "lamp", "supercategory": "furniture"},
    {"id": 26, "name": "shelves", "supercategory": "furniture"},
    {"id": 27, "name": "mirror", "supercategory": "furniture"},
    {"id": 28, "name": "sink", "supercategory": "furniture"},
    {"id": 29, "name": "cabinet", "supercategory": "furniture"},
    {"id": 30, "name": "bathtub", "supercategory": "furniture"},
    {"id": 31, "name": "door", "supercategory": "furniture"},
    {"id": 32, "name": "toilet", "supercategory": "furniture"},
    {"id": 33, "name": "desk", "supercategory": "furniture"},
    {"id": 34, "name": "box", "supercategory": "object"},
    {"id": 35, "name": "bookcase", "supercategory": "furniture"},
    {"id": 36, "name": "picture", "supercategory": "object"},
    {"id": 37, "name": "table", "supercategory": "furniture"},
    {"id": 38, "name": "counter", "supercategory": "furniture"},
    {"id": 39, "name": "bed", "supercategory": "furniture"},
    {"id": 40, "name": "night stand", "supercategory": "furniture"},
    {"id": 42, "name": "pillow", "supercategory": "object"},
    {"id": 43, "name": "sofa", "supercategory": "furniture"},
    {"id": 44, "name": "television", "supercategory": "electronics"},
    {"id": 45, "name": "floor mat", "supercategory": "object"},
    {"id": 46, "name": "curtain", "supercategory": "furniture"},
    {"id": 47, "name": "clothes", "supercategory": "object"},
    {"id": 48, "name": "stationery", "supercategory": "object"},
    {"id": 49, "name": "refrigerator", "supercategory": "appliance"},
    {"id": 52, "name": "bin", "supercategory": "object"},
    {"id": 53, "name": "stove", "supercategory": "appliance"},
    {"id": 57, "name": "oven", "supercategory": "appliance"},
    {"id": 61, "name": "machine", "supercategory": "appliance"},
]


def convert_results_to_coco(results_json_path: str, output_path: str = None):
    """
    将 results.json 转换为 COCO 格式

    Args:
        results_json_path: results.json 文件路径
        output_path: 输出文件路径
    """
    print(f"📂 Loading results from: {results_json_path}")

    with open(results_json_path, 'r') as f:
        results = json.load(f)

    print(f"✅ Loaded {len(results)} images")

    # 构建 COCO 格式
    coco_data = {
        "images": [],
        "annotations": [],
        "categories": COCO_CATEGORIES
    }

    annotation_id = 1

    for image_id_str, data in results.items():
        image_id = int(image_id_str)

        # 图片信息 (使用 image_id 作为 id)
        image_info = {
            "id": image_id,
            "file_name": f"{image_id}.jpg",  # 假设图片格式
            "width": 640,  # 需要根据实际情况调整
            "height": 480,  # 需要根据实际情况调整
        }
        coco_data["images"].append(image_info)

        # 标注信息
        boxes = data.get("boxes", [])
        phrases = data.get("phrases", [])
        confs = data.get("conf", [])

        for i, (box, phrase, conf) in enumerate(zip(boxes, phrases, confs)):
            # 解析置信度 (可能是 [[0.98]] 或 [0.98] 格式)
            if isinstance(conf, list):
                if len(conf) == 1:
                    conf = conf[0]
                if isinstance(conf, list):
                    conf = conf[0] if conf else 0.0

            score = float(conf)

            # 获取类别 ID
            category_id = SUNRGBD_CATEGORIES.get(phrase, -1)
            if category_id == -1:
                print(f"⚠️ Unknown category: '{phrase}', skipping")
                continue

            # COCO bbox 格式: [x, y, width, height]
            x1, y1, x2, y2 = box
            bbox = [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]

            # COCO segmentation 格式 (这里用 bbox 作为简化)
            # 如果有 mask，可以添加 polygon 格式
            segmentation = [[float(x) for x in box]]

            annotation = {
                "id": annotation_id,
                "image_id": image_id,
                "category_id": category_id,
                "bbox": bbox,
                "area": bbox[2] * bbox[3],
                "iscrowd": 0,
                "score": score,
                "phrases": phrase,
                # 额外 3D 信息 (非标准 COCO 字段)
                "center_cam": data["objects"][i].get("center_cam", [-1, -1, -1]),
                "dimensions": data["objects"][i].get("dimensions", [-1, -1, -1]),
                "vertices": data["objects"][i].get("vertices", []),
                "bbox2d_proj": box,
            }

            coco_data["annotations"].append(annotation)
            annotation_id += 1

    # 设置输出路径
    if output_path is None:
        output_path = str(Path(results_json_path).parent / "coco_format.json")

    # 保存
    print(f"💾 Saving COCO format to: {output_path}")
    with open(output_path, 'w') as f:
        json.dump(coco_data, f, indent=2)

    print(f"\n✅ Conversion complete!")
    print(f"   - Images: {len(coco_data['images'])}")
    print(f"   - Annotations: {len(coco_data['annotations'])}")
    print(f"   - Categories: {len(coco_data['categories'])}")

    return coco_data


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Convert results.json to COCO format")
    parser.add_argument("--input", type=str,
                        default="/data/ZhaoX/OVM3D-Dett/pseudo_label_gsam_val/results.json",
                        help="Input results.json file")
    parser.add_argument("--output", type=str, default=None,
                        help="Output COCO JSON file")

    args = parser.parse_args()

    convert_results_to_coco(args.input, args.output)
