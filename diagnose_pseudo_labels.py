#!/usr/bin/env python3
"""
诊断脚本：对比 Omni3D 伪标签的 3D 数值分布
比较 Omni3D_pl 和 Omni3D_pl-sam3d 两个版本的差异
"""

import json
import numpy as np
from pathlib import Path
from collections import defaultdict

def analyze_annotations(annotations, dataset_name, sample_limit=500):
    """分析单个数据集中的 3D 标注"""
    
    stats = {
        'count': 0,
        'center_x': [], 'center_y': [], 'center_z': [],
        'dim_d': [], 'dim_w': [], 'dim_h': [],
        'rotation_matrices': [],
        'category_counts': defaultdict(int),
        'problematic': []
    }
    
    for i, ann in enumerate(annotations[:sample_limit]):
        stats['count'] += 1
        
        # 提取 3D 信息
        center = ann.get('center_cam', ann.get('center_3d', []))
        dims = ann.get('dimensions', ann.get('dimensions_3d', []))
        R = ann.get('R_cam', ann.get('rotation', []))
        
        category = ann.get('category_name', ann.get('category', 'unknown'))
        stats['category_counts'][category] += 1
        
        # 记录数值
        if len(center) == 3:
            stats['center_x'].append(center[0])
            stats['center_y'].append(center[1])
            stats['center_z'].append(center[2])
        
        if len(dims) == 3:
            stats['dim_d'].append(dims[0])
            stats['dim_w'].append(dims[1])
            stats['dim_h'].append(dims[2])
        
        # 检测异常值
        problems = []
        
        # 1. 检查中心坐标是否合理（室内场景通常在 [-10, 10] 米范围）
        if len(center) == 3:
            if any(abs(c) > 20 for c in center):
                problems.append(f"center_outlier: {center}")
            if center[2] < 0:  # Z 通常为正（深度）
                problems.append(f"negative_depth: z={center[2]}")
        
        # 2. 检查尺寸是否合理
        if len(dims) == 3:
            if any(d < 0.01 or d > 10 for d in dims):
                problems.append(f"dim_outlier: {dims}")
            # 检查是否与类别先验差异过大
            if dims[0] > 5 or dims[1] > 5 or dims[2] > 5:
                problems.append(f"dim_too_large: {dims}")
        
        # 3. 检查旋转矩阵是否有效
        if isinstance(R, list) and len(R) == 9:
            R_arr = np.array(R).reshape(3, 3)
            det = np.linalg.det(R_arr)
            if abs(det - 1.0) > 0.1:
                problems.append(f"rotation_invalid_det: {det}")
        
        # 4. 检查是否有无效值（-1）
        if len(center) == 3 and -1 in center:
            problems.append("center_has_invalid_value")
        if len(dims) == 3 and -1 in dims:
            problems.append("dims_has_invalid_value")
        
        if problems:
            stats['problematic'].append({
                'id': ann.get('id', i),
                'category': category,
                'problems': problems,
                'center': center,
                'dims': dims
            })
    
    return stats


def print_stats(name, stats):
    """打印统计信息"""
    print(f"\n{'='*60}")
    print(f"Dataset: {name}")
    print(f"{'='*60}")
    print(f"Total annotations analyzed: {stats['count']}")
    
    if stats['center_x']:
        print(f"\n3D Center Statistics (meters):")
        print(f"  X: min={min(stats['center_x']):.3f}, max={max(stats['center_x']):.3f}, "
              f"mean={np.mean(stats['center_x']):.3f}, std={np.std(stats['center_x']):.3f}")
        print(f"  Y: min={min(stats['center_y']):.3f}, max={max(stats['center_y']):.3f}, "
              f"mean={np.mean(stats['center_y']):.3f}, std={np.std(stats['center_y']):.3f}")
        print(f"  Z: min={min(stats['center_z']):.3f}, max={max(stats['center_z']):.3f}, "
              f"mean={np.mean(stats['center_z']):.3f}, std={np.std(stats['center_z']):.3f}")
    
    if stats['dim_d']:
        print(f"\n3D Dimensions Statistics (meters):")
        print(f"  D (depth): min={min(stats['dim_d']):.3f}, max={max(stats['dim_d']):.3f}, "
              f"mean={np.mean(stats['dim_d']):.3f}, std={np.std(stats['dim_d']):.3f}")
        print(f"  W (width): min={min(stats['dim_w']):.3f}, max={max(stats['dim_w']):.3f}, "
              f"mean={np.mean(stats['dim_w']):.3f}, std={np.std(stats['dim_w']):.3f}")
        print(f"  H (height): min={min(stats['dim_h']):.3f}, max={max(stats['dim_h']):.3f}, "
              f"mean={np.mean(stats['dim_h']):.3f}, std={np.std(stats['dim_h']):.3f}")
    
    print(f"\nCategory Distribution:")
    for cat, count in sorted(stats['category_counts'].items(), key=lambda x: -x[1])[:10]:
        print(f"  {cat}: {count}")
    
    print(f"\nProblematic Annotations: {len(stats['problematic'])}")
    if stats['problematic']:
        print("\nSample problematic annotations:")
        for p in stats['problematic'][:5]:
            print(f"  ID={p['id']}, Category={p['category']}")
            print(f"    Problems: {p['problems']}")
            print(f"    Center: {p['center']}, Dims: {p['dims']}")


def compare_datasets(stats1, stats2, name1, name2):
    """对比两个数据集的统计差异"""
    print(f"\n{'='*60}")
    print(f"COMPARISON: {name1} vs {name2}")
    print(f"{'='*60}")
    
    # 对比类别分布
    all_cats = set(stats1['category_counts'].keys()) | set(stats2['category_counts'].keys())
    
    print(f"\nCategory count differences:")
    for cat in sorted(all_cats):
        c1 = stats1['category_counts'].get(cat, 0)
        c2 = stats2['category_counts'].get(cat, 0)
        if c1 != c2:
            print(f"  {cat}: {name1}={c1}, {name2}={c2} (diff={c2-c1})")
    
    # 对比尺寸均值
    if stats1['dim_d'] and stats2['dim_d']:
        print(f"\nDimension mean comparison:")
        for name, stats in [(name1, stats1), (name2, stats2)]:
            print(f"  {name}: D={np.mean(stats['dim_d']):.3f}, W={np.mean(stats['dim_w']):.3f}, H={np.mean(stats['dim_h']):.3f}")
        
        # 计算比例
        ratio_d = np.mean(stats2['dim_d']) / (np.mean(stats1['dim_d']) + 1e-6)
        ratio_w = np.mean(stats2['dim_w']) / (np.mean(stats1['dim_w']) + 1e-6)
        ratio_h = np.mean(stats2['dim_h']) / (np.mean(stats1['dim_h']) + 1e-6)
        print(f"  Ratio ({name2}/{name1}): D={ratio_d:.2f}x, W={ratio_w:.2f}x, H={ratio_h:.2f}x")
        
        if ratio_d > 2 or ratio_w > 2 or ratio_h > 2 or ratio_d < 0.5 or ratio_w < 0.5 or ratio_h < 0.5:
            print("  ⚠️  WARNING: Significant dimension difference detected!")


def main():
    base_path = Path("/data/ZhaoX/OVM3D-Dett/datasets")
    
    datasets = [
        ("Omni3D_pl/SUNRGBD_train", "Original (Omni3D_pl)"),
        ("Omni3D_pl-sam3d/SUNRGBD_train", "New (Omni3D_pl-sam3d)"),
    ]
    
    all_stats = {}
    
    for rel_path, name in datasets:
        json_path = base_path / rel_path
        train_path = json_path.with_name(json_path.name.replace('_val', '_train'))
        
        print(f"\n\nLoading {train_path}...")
        
        if not train_path.exists():
            print(f"  File not found: {train_path}")
            continue
        
        with open(train_path, 'r') as f:
            data = json.load(f)
        
        # 提取 annotations
        if 'annotations' in data:
            annotations = data['annotations']
        elif isinstance(data, list):
            annotations = data
        else:
            annotations = data.get('data', data.get('images', []))
        
        print(f"  Found {len(annotations)} total annotations")
        
        stats = analyze_annotations(annotations, name, sample_limit=1000)
        all_stats[name] = stats
        print_stats(name, stats)
    
    # 对比
    if len(all_stats) == 2:
        names = list(all_stats.keys())
        compare_datasets(all_stats[names[0]], all_stats[names[1]], names[0], names[1])


if __name__ == "__main__":
    main()
