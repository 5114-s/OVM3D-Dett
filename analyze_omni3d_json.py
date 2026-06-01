#!/usr/bin/env python3
"""
分析 Omni3D 伪标签 JSON 文件的 3D 数值结构
比较两个版本的差异并检测异常值
"""

import json
import ijson
import numpy as np
from pathlib import Path
from collections import defaultdict

# 文件路径
FILE1 = "/data/ZhaoX/OVM3D-Dett/datasets/Omni3D_pl/SUNRGBD_train.json"
FILE2 = "/data/ZhaoX/OVM3D-Dett/datasets/Omni3D_pl-sam3d/SUNRGBD_train.json"
OUTPUT_FILE = "/data/ZhaoX/OVM3D-Dett/pseudo_label_comparison.txt"

# 异常值检测范围
COORD_RANGE = (-20, 20)  # 米
DIM_RANGE = (0.01, 10)   # 米

def load_annotations_streaming(filepath, max_count=200):
    """使用 ijson 流式加载 annotations"""
    annotations = []
    try:
        with open(filepath, 'r') as f:
            # 先获取顶层键
            parser = ijson.parse(f)
            keys = set()
            for prefix, event, value in parser:
                if event == 'map_key':
                    keys.add(value)
            
        # 重新打开并流式读取 annotations
        with open(filepath, 'r') as f:
            # 读取 annotations 数组
            for i, item in enumerate(f.objects):
                if i >= max_count:
                    break
                if 'annotations' in item:
                    annotations.extend(item['annotations'][:max_count])
                    if len(annotations) >= max_count:
                        annotations = annotations[:max_count]
                        break
    except Exception as e:
        print(f"Streaming failed for {filepath}: {e}")
        # 回退到普通加载方式
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
                if 'annotations' in data:
                    annotations = data['annotations'][:max_count]
                elif isinstance(data, list):
                    annotations = data[:max_count]
        except json.JSONDecodeError:
            print(f"Failed to load {filepath}")
            return None
    return annotations

def load_annotations_chunked(filepath, chunk_size=10000):
    """分块加载方式（后备方案）"""
    annotations = []
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
            if 'annotations' in data:
                annotations = data['annotations'][:200]
            elif isinstance(data, list):
                annotations = data[:200]
    except:
        return None
    return annotations

def analyze_3d_properties(annotations, name):
    """分析 3D 属性"""
    if not annotations:
        return None
    
    result = {
        'name': name,
        'count': len(annotations),
        'center_cam': {'x': [], 'y': [], 'z': []},
        'dimensions': {'depth': [], 'width': [], 'height': []},
        'anomalies': {
            'coord_out_of_range': 0,
            'dim_out_of_range': 0,
            'missing_fields': 0,
            'negative_dims': 0
        }
    }
    
    for ann in annotations:
        # 提取 center_cam
        center_cam = ann.get('center_cam', [])
        if len(center_cam) >= 3:
            result['center_cam']['x'].append(center_cam[0])
            result['center_cam']['y'].append(center_cam[1])
            result['center_cam']['z'].append(center_cam[2])
            
            # 坐标异常检测
            if not (COORD_RANGE[0] <= center_cam[0] <= COORD_RANGE[1] and
                    COORD_RANGE[0] <= center_cam[1] <= COORD_RANGE[1] and
                    COORD_RANGE[0] <= center_cam[2] <= COORD_RANGE[1]):
                result['anomalies']['coord_out_of_range'] += 1
        else:
            result['anomalies']['missing_fields'] += 1
        
        # 提取 dimensions (depth, width, height)
        dims = ann.get('dimensions', [])
        if len(dims) >= 3:
            result['dimensions']['depth'].append(dims[0])
            result['dimensions']['width'].append(dims[1])
            result['dimensions']['height'].append(dims[2])
            
            # 尺寸异常检测
            for d in dims:
                if d <= 0:
                    result['anomalies']['negative_dims'] += 1
                    break
            if not (DIM_RANGE[0] <= dims[0] <= DIM_RANGE[1] and
                    DIM_RANGE[0] <= dims[1] <= DIM_RANGE[1] and
                    DIM_RANGE[0] <= dims[2] <= DIM_RANGE[1]):
                result['anomalies']['dim_out_of_range'] += 1
        else:
            result['anomalies']['missing_fields'] += 1
    
    # 计算统计量
    def compute_stats(values):
        if not values:
            return {'min': 0, 'max': 0, 'mean': 0, 'std': 0}
        arr = np.array(values)
        return {
            'min': float(np.min(arr)),
            'max': float(np.max(arr)),
            'mean': float(np.mean(arr)),
            'std': float(np.std(arr))
        }
    
    result['center_cam_stats'] = {k: compute_stats(v) for k, v in result['center_cam'].items()}
    result['dimensions_stats'] = {k: compute_stats(v) for k, v in result['dimensions'].items()}
    
    return result

def format_stats(stats_dict, indent=2):
    """格式化统计输出"""
    lines = []
    spacer = " " * indent
    
    for name, data in stats_dict.items():
        if isinstance(data, dict) and 'min' in data:
            lines.append(f"{spacer}{name}:")
            lines.append(f"{spacer}  Range: [{data['min']:.4f}, {data['max']:.4f}]")
            lines.append(f"{spacer}  Mean: {data['mean']:.4f}, Std: {data['std']:.4f}")
    return "\n".join(lines)

def compare_analyses(analysis1, analysis2):
    """对比两个分析结果"""
    if not analysis1 or not analysis2:
        return "无法对比（缺少数据）"
    
    lines = []
    lines.append("\n" + "="*60)
    lines.append("两个版本的差异对比")
    lines.append("="*60)
    
    # 样本数量差异
    lines.append(f"\n样本数量: v1={analysis1['count']}, v2={analysis2['count']}")
    
    # Center_cam 差异
    lines.append("\n--- Center_Cam 差异 ---")
    for axis in ['x', 'y', 'z']:
        v1_mean = analysis1['center_cam_stats'][axis]['mean']
        v2_mean = analysis2['center_cam_stats'][axis]['mean']
        v1_std = analysis1['center_cam_stats'][axis]['std']
        v2_std = analysis2['center_cam_stats'][axis]['std']
        diff_mean = v2_mean - v1_mean
        diff_std = v2_std - v1_std
        lines.append(f"  {axis}-axis: mean Δ={diff_mean:.4f}, std Δ={diff_std:.4f}")
        lines.append(f"    v1: mean={v1_mean:.4f}±{v1_std:.4f}")
        lines.append(f"    v2: mean={v2_mean:.4f}±{v2_std:.4f}")
    
    # Dimensions 差异
    lines.append("\n--- Dimensions 差异 ---")
    for dim in ['depth', 'width', 'height']:
        v1_mean = analysis1['dimensions_stats'][dim]['mean']
        v2_mean = analysis2['dimensions_stats'][dim]['mean']
        v1_std = analysis1['dimensions_stats'][dim]['std']
        v2_std = analysis2['dimensions_stats'][dim]['std']
        diff_mean = v2_mean - v1_mean
        diff_std = v2_std - v1_std
        pct_change = (diff_mean / v1_mean * 100) if v1_mean != 0 else 0
        lines.append(f"  {dim}: mean Δ={diff_mean:.4f} ({pct_change:+.2f}%), std Δ={diff_std:.4f}")
        lines.append(f"    v1: mean={v1_mean:.4f}±{v1_std:.4f}")
        lines.append(f"    v2: mean={v2_mean:.4f}±{v2_std:.4f}")
    
    # 异常值对比
    lines.append("\n--- 异常值对比 ---")
    for key in analysis1['anomalies']:
        v1_count = analysis1['anomalies'][key]
        v2_count = analysis2['anomalies'][key]
        lines.append(f"  {key}: v1={v1_count} ({v1_count/analysis1['count']*100:.1f}%), v2={v2_count} ({v2_count/analysis2['count']*100:.1f}%)")
    
    return "\n".join(lines)

def extract_sample_annotations(filepath, n=5):
    """提取前 n 个样本的原始数据"""
    annotations = load_annotations_chunked(filepath, n)
    if not annotations:
        return []
    return annotations[:n]

def main():
    print("="*60)
    print("Omni3D 伪标签 JSON 文件 3D 数值结构分析")
    print("="*60)
    
    output_lines = []
    output_lines.append("="*60)
    output_lines.append("Omni3D 伪标签 JSON 文件 3D 数值结构分析报告")
    output_lines.append("="*60)
    output_lines.append(f"\n分析时间: 2026-05-31")
    output_lines.append(f"异常值检测范围:")
    output_lines.append(f"  - 坐标范围: {COORD_RANGE} 米")
    output_lines.append(f"  - 尺寸范围: {DIM_RANGE} 米")
    
    # 文件信息
    for filepath, label in [(FILE1, "v1"), (FILE2, "v2")]:
        path = Path(filepath)
        if path.exists():
            size_mb = path.stat().st_size / (1024 * 1024)
            output_lines.append(f"\n文件 {label}: {filepath}")
            output_lines.append(f"  大小: {size_mb:.2f} MB")
    
    # 加载并分析
    print("\n正在加载文件...")
    
    annotations1 = load_annotations_chunked(FILE1, 200)
    annotations2 = load_annotations_chunked(FILE2, 200)
    
    print(f"v1 annotations loaded: {len(annotations1) if annotations1 else 0}")
    print(f"v2 annotations loaded: {len(annotations2) if annotations2 else 0}")
    
    # 分析
    analysis1 = analyze_3d_properties(annotations1, "v1 (Omni3D_pl)")
    analysis2 = analyze_3d_properties(annotations2, "v2 (Omni3D_pl-sam3d)")
    
    # 输出每个文件的详细分析
    for analysis, label, filepath in [
        (analysis1, "v1 (Omni3D_pl)", FILE1),
        (analysis2, "v2 (Omni3D_pl-sam3d)", FILE2)
    ]:
        if analysis:
            section = []
            section.append(f"\n{'='*60}")
            section.append(f"文件 {label}")
            section.append(f"路径: {filepath}")
            section.append(f"{'='*60}")
            section.append(f"\n样本数量: {analysis['count']}")
            
            section.append("\n--- Center_Cam (3D 中心坐标) ---")
            section.append(format_stats(analysis['center_cam_stats']))
            
            section.append("\n--- Dimensions (深度/宽度/高度) ---")
            section.append(format_stats(analysis['dimensions_stats']))
            
            section.append("\n--- 异常值统计 ---")
            total_samples = analysis['count']
            for key, count in analysis['anomalies'].items():
                pct = count / total_samples * 100 if total_samples > 0 else 0
                section.append(f"  {key}: {count} ({pct:.1f}%)")
            
            text = "\n".join(section)
            print(text)
            output_lines.append(text)
    
    # 提取样本对比
    print("\n提取前 5 个样本进行详细对比...")
    samples1 = extract_sample_annotations(FILE1, 5)
    samples2 = extract_sample_annotations(FILE2, 5)
    
    sample_section = []
    sample_section.append("\n" + "="*60)
    sample_section.append("前 5 个样本的 3D 信息对比")
    sample_section.append("="*60)
    
    for i in range(min(5, len(samples1), len(samples2))):
        ann1 = samples1[i]
        ann2 = samples2[i]
        
        sample_section.append(f"\n--- 样本 {i+1} ---")
        
        # 标注 ID
        sample_section.append(f"  ID: v1={ann1.get('id', 'N/A')}, v2={ann2.get('id', 'N/A')}")
        
        # Center_cam
        c1 = ann1.get('center_cam', [])
        c2 = ann2.get('center_cam', [])
        sample_section.append(f"  center_cam:")
        sample_section.append(f"    v1: {c1}")
        sample_section.append(f"    v2: {c2}")
        if c1 and c2:
            diff = [c2[j] - c1[j] if j < len(c1) and j < len(c2) else 0 for j in range(3)]
            sample_section.append(f"    Δ:  [{diff[0]:.4f}, {diff[1]:.4f}, {diff[2]:.4f}]")
        
        # Dimensions
        d1 = ann1.get('dimensions', [])
        d2 = ann2.get('dimensions', [])
        sample_section.append(f"  dimensions (D/W/H):")
        sample_section.append(f"    v1: {d1}")
        sample_section.append(f"    v2: {d2}")
        if d1 and d2:
            diff = [d2[j] - d1[j] if j < len(d1) and j < len(d2) else 0 for j in range(3)]
            sample_section.append(f"    Δ:  [{diff[0]:.4f}, {diff[1]:.4f}, {diff[2]:.4f}]")
        
        # R_cam (旋转矩阵 - 前两行)
        r1 = ann1.get('R_cam', [])
        r2 = ann2.get('R_cam', [])
        if r1 and r2:
            sample_section.append(f"  R_cam (行展开):")
            sample_section.append(f"    v1: {r1[:6]}... (前6元素)")
            sample_section.append(f"    v2: {r2[:6]}... (前6元素)")
    
    text = "\n".join(sample_section)
    print(text)
    output_lines.append(text)
    
    # 对比分析
    comparison = compare_analyses(analysis1, analysis2)
    print(comparison)
    output_lines.append(comparison)
    
    # 写入输出文件
    final_output = "\n".join(output_lines)
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        f.write(final_output)
    
    print(f"\n\n分析结果已保存到: {OUTPUT_FILE}")
    print("="*60)

if __name__ == "__main__":
    main()
