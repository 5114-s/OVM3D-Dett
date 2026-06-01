#!/usr/bin/env python3
"""
Omni3D 伪标签 JSON 文件分析脚本
分析两个伪标签 JSON 文件的 3D 数值结构差异
"""

import json
import sys
from pathlib import Path

def analyze_omni3d_file(filepath, max_annotations=200):
    """分析 Omni3D JSON 文件的 3D 结构"""
    output = []
    
    def log(msg):
        print(msg)
        output.append(msg)
    
    log(f"\n{'='*60}")
    log(f"分析文件: {filepath}")
    log(f"{'='*60}")
    
    if not Path(filepath).exists():
        log(f"文件不存在: {filepath}")
        return None, output
    
    file_size = Path(filepath).stat().st_size
    log(f"文件大小: {file_size / (1024*1024):.2f} MB")
    
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    # 基本结构分析
    log(f"\n顶层字段: {list(data.keys())}")
    
    images_count = len(data.get('images', []))
    annotations_count = len(data.get('annotations', []))
    log(f"images 数量: {images_count}")
    log(f"annotations 数量: {annotations_count}")
    
    # 提取 annotation 的 3D 信息
    annotations = data.get('annotations', [])[:max_annotations]
    
    centers = []
    dimensions = []
    
    for ann in annotations:
        if 'center_cam' in ann:
            centers.append(ann['center_cam'])
        if 'dimensions' in ann:
            dimensions.append(ann['dimensions'])
    
    log(f"\n分析前 {len(annotations)} 个 annotations 的 3D 信息:")
    
    # 分析 center_cam (X, Y, Z)
    if centers:
        x_vals = [c[0] for c in centers]
        y_vals = [c[1] for c in centers]
        z_vals = [c[2] for c in centers]
        
        log(f"\n--- center_cam (3D 中心坐标) ---")
        log(f"X: min={min(x_vals):.4f}, max={max(x_vals):.4f}, mean={sum(x_vals)/len(x_vals):.4f}")
        log(f"Y: min={min(y_vals):.4f}, max={max(y_vals):.4f}, mean={sum(y_vals)/len(y_vals):.4f}")
        log(f"Z: min={min(z_vals):.4f}, max={max(z_vals):.4f}, mean={sum(z_vals)/len(z_vals):.4f}")
    else:
        log("\n未找到 center_cam 字段")
        x_vals, y_vals, z_vals = [], [], []
    
    # 分析 dimensions (depth, width, height)
    if dimensions:
        depth_vals = [d[0] for d in dimensions]
        width_vals = [d[1] for d in dimensions]
        height_vals = [d[2] for d in dimensions]
        
        log(f"\n--- dimensions (3D 尺寸) ---")
        log(f"Depth:  min={min(depth_vals):.4f}, max={max(depth_vals):.4f}, mean={sum(depth_vals)/len(depth_vals):.4f}")
        log(f"Width:  min={min(width_vals):.4f}, max={max(width_vals):.4f}, mean={sum(width_vals)/len(width_vals):.4f}")
        log(f"Height: min={min(height_vals):.4f}, max={max(height_vals):.4f}, mean={sum(height_vals)/len(height_vals):.4f}")
    else:
        log("\n未找到 dimensions 字段")
        depth_vals, width_vals, height_vals = [], [], []
    
    # 检查异常值
    log(f"\n--- 异常值检测 ---")
    coord_issues = 0
    size_issues = 0
    
    for i, ann in enumerate(annotations):
        if 'center_cam' in ann:
            c = ann['center_cam']
            if any(abs(v) > 20 for v in c):
                coord_issues += 1
                if coord_issues <= 3:
                    log(f"  异常坐标 #{i}: center_cam = {c}")
        
        if 'dimensions' in ann:
            d = ann['dimensions']
            if any(v < 0.01 or v > 10 for v in d):
                size_issues += 1
                if size_issues <= 3:
                    log(f"  异常尺寸 #{i}: dimensions = {d}")
    
    if coord_issues == 0:
        log("  所有坐标在合理范围内 [-20, 20] 米")
    else:
        log(f"  发现 {coord_issues} 个异常坐标 (超出 [-20, 20] 米)")
    
    if size_issues == 0:
        log("  所有尺寸在合理范围内 [0.01, 10] 米")
    else:
        log(f"  发现 {size_issues} 个异常尺寸 (超出 [0.01, 10] 米)")
    
    return {
        'images_count': images_count,
        'annotations_count': annotations_count,
        'x_vals': x_vals,
        'y_vals': y_vals,
        'z_vals': z_vals,
        'depth_vals': depth_vals,
        'width_vals': width_vals,
        'height_vals': height_vals,
        'coord_issues': coord_issues,
        'size_issues': size_issues
    }, output

def main():
    file_fixed = "/data/ZhaoX/OVM3D-Dett/pseudo_labels_coco_fixed.json"
    output_file = "/data/ZhaoX/OVM3D-Dett/pseudo_label_comparison.txt"
    
    all_output = []
    
    def log(msg):
        all_output.append(msg)
    
    log("=" * 70)
    log("Omni3D 伪标签 JSON 文件 3D 结构分析报告")
    log("=" * 70)
    
    # 分析 fixed 文件 (truncated 文件无法解析)
    log("\n注意: pseudo_labels_coco.json 是截断文件，无法解析")
    log("只分析完整的 pseudo_labels_coco_fixed.json\n")
    
    result, out = analyze_omni3d_file(file_fixed)
    all_output.extend(out)
    
    # 对比分析 (single file analysis)
    log(f"\n{'='*60}")
    log("分析总结")
    log(f"{'='*60}")
    
    if result:
        log(f"\n| 指标              | pseudo_labels_coco_fixed |")
        log(f"|-------------------|-------------------------|")
        
        # 数量
        log(f"| annotations 数量 | {result['annotations_count']:<24} |")
        
        # 中心点统计
        if result['x_vals']:
            log(f"| X 坐标均值       | {sum(result['x_vals'])/len(result['x_vals']):<24.4f} |")
            log(f"| Y 坐标均值       | {sum(result['y_vals'])/len(result['y_vals']):<24.4f} |")
            log(f"| Z 坐标均值       | {sum(result['z_vals'])/len(result['z_vals']):<24.4f} |")
        
        # 尺寸统计
        if result['depth_vals']:
            log(f"| Depth 均值      | {sum(result['depth_vals'])/len(result['depth_vals']):<24.4f} |")
            log(f"| Width 均值       | {sum(result['width_vals'])/len(result['width_vals']):<24.4f} |")
            log(f"| Height 均值      | {sum(result['height_vals'])/len(result['height_vals']):<24.4f} |")
        
        log(f"\n异常值检测:")
        log(f"| 坐标异常数量     | {result['coord_issues']:<24} |")
        log(f"| 尺寸异常数量     | {result['size_issues']:<24} |")
    
    # 样本提取
    log(f"\n{'='*60}")
    log("前 5 个 annotation 样本")
    log(f"{'='*60}")
    
    log(f"\n--- pseudo_labels_coco_fixed ---")
    with open(file_fixed, 'r') as f:
        data = json.load(f)
    for i, ann in enumerate(data.get('annotations', [])[:5]):
        log(f"\n样本 {i+1}:")
        log(f"  category_id: {ann.get('category_id', 'N/A')}")
        if 'center_cam' in ann:
            log(f"  center_cam: {ann['center_cam']}")
        if 'dimensions' in ann:
            log(f"  dimensions: {ann['dimensions']}")
        if 'R_cam' in ann:
            r = ann['R_cam']
            log(f"  R_cam[0]: [{r[0]}, {r[1]}, {r[2]}]")
    
    # 保存到文件
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(all_output))
    
    log(f"\n{'='*60}")
    log(f"分析结果已保存到: {output_file}")
    log(f"{'='*60}")

if __name__ == "__main__":
    main()
