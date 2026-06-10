import argparse
import json
import os

import torch


def transform_to_coco_format(all_results: dict, output_dir: str, dataset_name: str):
    thing_classes_dict = {
        "SUNRGBD": {
            'chair': 18, 'door': 31, 'table': 37, 'shelves': 26, 'kitchen pan': 51,
            'bin': 52, 'counter': 38, 'cabinet': 29, 'stove': 53, 'sink': 28,
            'books': 14, 'refrigerator': 49, 'microwave': 54, 'bottle': 15,
            'plates': 55, 'bowl': 56, 'oven': 57, 'vase': 58, 'faucet': 59,
            'towel': 22, 'tissues': 60, 'machine': 61, 'printer': 62, 'desk': 33,
            'monitor': 63, 'podium': 64, 'bookcase': 35, 'dresser': 41, 'cart': 65,
            'projector': 66, 'electronics': 67, 'computer': 68, 'box': 34,
            'picture': 36, 'laptop': 20, 'pillow': 42, 'bed': 39,
            'air conditioner': 69, 'lamp': 25, 'night stand': 40, 'board': 50,
            'sofa': 43, 'coffee maker': 71, 'toaster': 72, 'potted plant': 73,
            'stationery': 48, 'painting': 74, 'bag': 75, 'tray': 76, 'cup': 19,
            'drawers': 70, 'keyboard': 77, 'shoes': 21, 'bicycle': 11,
            'blanket': 78, 'television': 44, 'rack': 79, 'mirror': 27,
            'clothes': 47, 'phone': 80, 'mouse': 81, 'person': 7,
            'fire extinguisher': 82, 'toys': 83, 'ladder': 84, 'fan': 85,
            'toilet': 32, 'bathtub': 30, 'glass': 86, 'clock': 87,
            'toilet paper': 88, 'closet': 89, 'curtain': 46, 'window': 24,
            'fume hood': 90, 'utensils': 91, 'floor mat': 45, 'soundsystem': 92,
            'fire place': 93, 'shower curtain': 94, 'blinds': 23,
            'remote': 95, 'pen': 96
        }
    }

    thing_classes = thing_classes_dict.get(dataset_name, {})
    annotations = []
    num = 1
    dataset_id = 1

    for image_id, result in all_results.items():
        phrases = result['phrases']
        scores = result['conf']
        boxes = result['boxes']
        boxes3d = result['boxes3d']
        center_cam = result['center_cam']
        dimensions = result['dimensions']
        R_cam = result['R_cam']

        for j in range(len(phrases)):
            category_name = phrases[j]
            if category_name not in thing_classes:
                continue

            def extract_float(val):
                if isinstance(val, list):
                    if len(val) == 1:
                        val = val[0]
                    if isinstance(val, list):
                        return float(val[0]) if val else 0.0
                return float(val)

            score_val = extract_float(scores[j])
            obj = {
                'id': dataset_id * 10000000 + num,
                'image_id': image_id,
                'dataset_id': dataset_id,
                'category_name': category_name,
                'category_id': thing_classes[category_name],
                'valid3D': True,
                'bbox2D_tight': boxes[j].tolist() if hasattr(boxes[j], 'tolist') else boxes[j],
                'bbox2D_trunc': boxes[j].tolist() if hasattr(boxes[j], 'tolist') else boxes[j],
                'bbox2D_proj': boxes[j].tolist() if hasattr(boxes[j], 'tolist') else boxes[j],
                'bbox3D_cam': boxes3d[j].tolist() if hasattr(boxes3d[j], 'tolist') else boxes3d[j],
                'center_cam': center_cam[j].tolist() if hasattr(center_cam[j], 'tolist') else center_cam[j],
                'dimensions': [float(x) for x in dimensions[j]],
                'R_cam': R_cam[j].tolist() if hasattr(R_cam[j], 'tolist') else R_cam[j],
                'behind_camera': False,
                'visibility': 1.0,
                'truncation': 0.0,
                'segmentation_pts': -1,
                'lidar_pts': -1,
                'depth_error': -1,
                'score': score_val
            }
            annotations.append(obj)
            num += 1

    coco_output_dir = os.path.join(output_dir, 'coco_format')
    os.makedirs(coco_output_dir, exist_ok=True)
    coco_data = {
        'annotations': annotations,
        'dataset_name': dataset_name,
        'num_images': len(all_results),
        'num_objects': len(annotations)
    }

    coco_path = os.path.join(coco_output_dir, f'{dataset_name}_pseudo_labels.json')
    with open(coco_path, 'w') as f:
        json.dump(coco_data, f, indent=2, default=lambda x: x.tolist() if hasattr(x, 'tolist') else list(x))

    print(coco_path)
    print(f'num_images={len(all_results)}')
    print(f'num_objects={len(annotations)}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_pth', required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--dataset', default='SUNRGBD')
    args = parser.parse_args()

    all_results = torch.load(args.input_pth, map_location='cpu', weights_only=False)
    transform_to_coco_format(all_results, args.output_dir, args.dataset)
