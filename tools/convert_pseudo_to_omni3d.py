"""
将伪标签结果转换为 Omni3D JSON 格式
适配用户的数据路径结构
"""

import json
import torch
import numpy as np
from tqdm import tqdm
import os
import argparse

# SUNRGBD 类别映射
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


def convert_to_omni3d_format(source_json_path, info_pth_path, output_path, dataset_name='SUNRGBD', mode='val'):
    """
    将伪标签转换为 Omni3D JSON 格式
    
    Args:
        source_json_path: 原始 JSON 文件路径 (Omni3D_pl)
        info_pth_path: info_3d.pth 文件路径
        output_path: 输出 JSON 文件路径
        dataset_name: 数据集名称
        mode: train 或 val
    """
    print(f"📂 Loading source JSON: {source_json_path}")
    with open(source_json_path, 'r') as f:
        source_data = json.load(f)
    
    print(f"📦 Loading pseudo labels: {info_pth_path}")
    info = torch.load(info_pth_path, map_location='cpu', weights_only=False)
    
    # 创建 image_id 到 image 信息的映射
    image_id_to_info = {img['id']: img for img in source_data['images']}
    
    # 统计
    stats = {
        'total_images': 0,
        'total_objects': 0,
        'valid_3d': 0,
        'invalid_3d': 0,
        'unknown_category': 0
    }
    
    annotations = []
    ann_id = 1
    
    # 使用 source JSON 中的 info
    dataset_id = source_data['info']['id']
    
    print(f"\n🔄 Converting {len(source_data['images'])} images...")
    
    for img_info in tqdm(source_data['images']):
        im_id = img_info['id']
        stats['total_images'] += 1
        
        if im_id not in info:
            continue
        
        img_data = info[im_id]
        phrases = img_data['phrases']
        scores = img_data['conf']
        boxes = img_data['boxes']
        boxes3d = img_data['boxes3d']
        center_cam = img_data['center_cam']
        dimensions = img_data['dimensions']
        R_cam = img_data['R_cam']
        
        for j in range(len(phrases)):
            cat_name = phrases[j]
            stats['total_objects'] += 1
            
            # 检查 dimensions 是否有效
            dim = dimensions[j]
            if isinstance(dim, np.ndarray):
                dim = dim.tolist()
            elif hasattr(dim, 'tolist'):
                dim = dim.tolist()
            
            dim_val = dim[0] if isinstance(dim, (list, np.ndarray)) else dim
            is_valid_3d = (dim_val != -1)
            
            if is_valid_3d:
                stats['valid_3d'] += 1
            else:
                stats['invalid_3d'] += 1
            
            # 获取类别 ID
            if cat_name in SUNRGBD_CATEGORIES:
                cat_id = SUNRGBD_CATEGORIES[cat_name]
            else:
                stats['unknown_category'] += 1
                cat_id = 0  # 默认值
            
            # 转换 boxes
            bbox = boxes[j]
            if hasattr(bbox, 'tolist'):
                bbox = bbox.tolist()
            
            # 转换 boxes3d
            bbox3d = boxes3d[j]
            if hasattr(bbox3d, 'tolist'):
                bbox3d = bbox3d.tolist()
            
            # 转换 center_cam
            center = center_cam[j]
            if hasattr(center, 'tolist'):
                center = center.tolist()
            
            # 转换 dimensions
            if hasattr(dim, 'tolist'):
                dim = dim.tolist()
            
            # 转换 R_cam
            rot = R_cam[j]
            if hasattr(rot, 'tolist'):
                rot = rot.tolist()
            
            # 获取 score
            score = scores[j]
            if hasattr(score, 'item'):
                score = score.item()
            
            obj = {
                'id': int(dataset_id * 10000000 + ann_id),
                'image_id': int(im_id),
                'dataset_id': int(dataset_id),
                'category_name': str(cat_name),
                'category_id': int(cat_id),
                'valid3D': bool(is_valid_3d),
                'bbox2D_tight': [-1, -1, -1, -1],
                'bbox2D_trunc': [-1, -1, -1, -1],
                'bbox2D_proj': [float(x) for x in bbox],
                'bbox3D_cam': [[float(x) for x in row] for row in bbox3d],
                'center_cam': [float(x) for x in center],
                'dimensions': [float(x) for x in dim],
                'R_cam': [[float(x) for x in row] for row in rot],
                'behind_camera': False,
                'visibility': 1.0,
                'truncation': 0.0,
                'segmentation_pts': -1,
                'lidar_pts': -1,
                'depth_error': -1,
                'score': float(score)
            }
            
            annotations.append(obj)
            ann_id += 1
    
    # 构建输出数据
    output_data = {
        'info': source_data['info'],
        'images': source_data['images'],
        'categories': COCO_CATEGORIES,
        'annotations': annotations
    }
    
    # 保存
    print(f"\n💾 Saving to: {output_path}")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(output_data, f)
    
    # 打印统计
    print("\n" + "=" * 50)
    print("📊 转换统计")
    print("=" * 50)
    print(f"总图片数: {stats['total_images']}")
    print(f"总物体数: {stats['total_objects']}")
    print(f"有效 3D: {stats['valid_3d']} ({stats['valid_3d']/max(1,stats['total_objects'])*100:.1f}%)")
    print(f"无效 3D: {stats['invalid_3d']} ({stats['invalid_3d']/max(1,stats['total_objects'])*100:.1f}%)")
    print(f"未知类别: {stats['unknown_category']}")
    print(f"标注总数: {len(annotations)}")
    print("=" * 50)
    
    return output_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert pseudo labels to Omni3D format")
    parser.add_argument('--source_json', type=str, required=True,
                        help='Source JSON file (Omni3D_pl)')
    parser.add_argument('--info_pth', type=str, required=True,
                        help='info_3d.pth file from pseudo label generation')
    parser.add_argument('--output', type=str, required=True,
                        help='Output JSON file path')
    parser.add_argument('--dataset_name', type=str, default='SUNRGBD',
                        help='Dataset name')
    parser.add_argument('--mode', type=str, default='val',
                        help='train or val')
    
    args = parser.parse_args()
    
    convert_to_omni3d_format(
        source_json_path=args.source_json,
        info_pth_path=args.info_pth,
        output_path=args.output,
        dataset_name=args.dataset_name,
        mode=args.mode
    )
