#!/usr/bin/env python3
"""
对比 Omni3D 伪标签的 3D 数值
比较 Omni3D_pl (原版) vs Omni3D_pl-sam3d (新版本)
"""

import json
import numpy as np
from pathlib import Path

def analyze_file(filepath, max_samples=500):
    """分析单个文件"""
    print(f"\n{'='*60}")
    print(f"分析: {filepath}")
    print(f"{'='*60}")
    
    if not Path(filepath).exists():
        print(f"文件不存在!")
        return None
    
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    annotations = data.get('annotations', [])
    print(f"总 annotations 数量: {len(annotations)}")
    
    centers_x, centers_y, centers_z = [], [], []
    dims_d, dims_w, dims_h = [], [], []
    categories = []
    problems = []
    
    for i, ann in enumerate(annotations[:max_samples]):
        # 3D 中心
        center = ann.get('center_cam', ann.get('center_3d', []))
        if len(center) == 3:
            centers_x.append(center[0])
            centers_y.append(center[1])
            centers_z.append(center[2])
            
            # 检测异常
            if any(abs(v) > 50 for v in center):
                problems.append(f"ann[{i}]: 坐标异常 {center}")
        
        # 3D 尺寸
        dims = ann.get('dimensions', ann.get('dimensions_3d', []))
        if len(dims) == 3:
            dims_d.append(dims[0])
            dims_w.append(dims[1])
            dims_h.append(dims[2])
            
            # 检测异常
            if any(v < 0.001 or v > 20 for v in dims):
                problems.append(f"ann[{i}]: 尺寸异常 {dims}")
        
        # 类别
        cat = ann.get('category_name', ann.get('category_id', '?'))
        categories.append(cat)
    
    print(f"\n--- 3D 中心坐标 (单位: 米) ---")
    if centers_x:
        print(f"X: min={min(centers_x):.3f}, max={max(centers_x):.3f}, mean={np.mean(centers_x):.3f}, std={np.std(centers_x):.3f}")
        print(f"Y: min={min(centers_y):.3f}, max={max(centers_y):.3f}, mean={np.mean(centers_y):.3f}, std={np.std(centers_y):.3f}")
        print(f"Z: min={min(centers_z):.3f}, max={max(centers_z):.3f}, mean={np.mean(centers_z):.3f}, std={np.std(centers_z):.3f}")
    
    print(f"\n--- 3D 尺寸 (单位: 米) ---")
    if dims_d:
        print(f"D(depth):  min={min(dims_d):.4f}, max={max(dims_d):.4f}, mean={np.mean(dims_d):.4f}, std={np.std(dims_d):.4f}")
        print(f"W(width):  min={min(dims_w):.4f}, max={max(dims_w):.4f}, mean={np.mean(dims_w):.4f}, std={np.std(dims_w):.4f}")
        print(f"H(height): min={min(dims_h):.4f}, max={max(dims_h):.4f}, mean={np.mean(dims_h):.4f}, std={np.std(dims_h):.4f}")
    
    print(f"\n--- 异常检测 ---")
    print(f"问题数量: {len(problems)}")
    for p in problems[:5]:
        print(f"  {p}")
    
    # 打印样本
    print(f"\n--- 前3个样本 ---")
    for i in range(min(3, len(annotations))):
        ann = annotations[i]
        center = ann.get('center_cam', [])
        dims = ann.get('dimensions', [])
        cat = ann.get('category_name', '?')
        print(f"样本{i+1} [{cat}]: center={center}, dims={dims}")
    
    return {
        'cx': centers_x, 'cy': centers_y, 'cz': centers_z,
        'dd': dims_d, 'dw': dims_w, 'dh': dims_h,
        'problems': problems,
        'count': len(annotations)
    }

def main():
    base = Path("/data/ZhaoX/OVM3D-Dett/datasets")
    
    files = [
        base / "Omni3D_pl/SUNRGBD_train.json",
        base / "Omni3D_pl-sam3d/SUNRGBD_train.json",
    ]
    
    results = []
    for f in files:
        r = analyze_file(str(f))
        results.append(r)
    
    # 对比
    print(f"\n{'='*70}")
    print("对比分析")
    print(f"{'='*70}")
    
    if all(r for r in results):
        r1, r2 = results
        
        print(f"\n| 指标         | Omni3D_pl (原版) | Omni3D_pl-sam3d (新) | 比率     |")
        print(f"|--------------|------------------|---------------------|----------|")
        
        # 数量
        print(f"| 样本数       | {r1['count']:<17} | {r2['count']:<20} |          |")
        
        # 中心坐标
        if r1['cx'] and r2['cx']:
            mcx1, mcx2 = np.mean(r1['cx']), np.mean(r2['cx'])
            mcy1, mcy2 = np.mean(r1['cy']), np.mean(r2['cy'])
            mcz1, mcz2 = np.mean(r1['cz']), np.mean(r2['cz'])
            print(f"| X 中心均值   | {mcx1:<17.4f} | {mcx2:<20.4f} | {mcx2/mcx1:.2f}x  |")
            print(f"| Y 中心均值   | {mcy1:<17.4f} | {mcy2:<20.4f} | {mcy2/mcy1:.2f}x  |")
            print(f"| Z 中心均值   | {mcz1:<17.4f} | {mcz2:<20.4f} | {mcz2/mcz1:.2f}x  |")
        
        # 尺寸
        if r1['dd'] and r2['dd']:
            md1, md2 = np.mean(r1['dd']), np.mean(r2['dd'])
            mw1, mw2 = np.mean(r1['dw']), np.mean(r2['dw'])
            mh1, mh2 = np.mean(r1['dh']), np.mean(r2['dh'])
            print(f"| Depth 均值   | {md1:<17.4f} | {md2:<20.4f} | {md2/md1:.2f}x  |")
            print(f"| Width 均值   | {mw1:<17.4f} | {mw2:<20.4f} | {mw2/mw1:.2f}x  |")
            print(f"| Height 均值  | {mh1:<17.4f} | {mh2:<20.4f} | {mh2/mh1:.2f}x  |")
        
        # 问题数量
        print(f"| 异常数量     | {len(r1['problems']):<17} | {len(r2['problems']):<20} |          |")
        
        # 检测关键差异
        print(f"\n{'='*70}")
        print("关键发现")
        print(f"{'='*70}")
        
        if abs(mcx2/mcx1 - 1.0) > 0.5:
            print(f"⚠️ X 中心差异巨大: {mcx1:.3f} vs {mcx2:.3f} ({mcx2/mcx1:.1f}x)")
        if abs(mcz2/mcz1 - 1.0) > 0.5:
            print(f"⚠️ Z 深度差异巨大: {mcz1:.3f} vs {mcz2:.3f} ({mcz2/mcz1:.1f}x)")
        if abs(md2/md1 - 1.0) > 0.5:
            print(f"⚠️ Depth 尺寸差异: {md1:.3f} vs {md2:.3f} ({md2/md1:.1f}x)")
        if len(r2['problems']) > len(r1['problems']) * 2:
            print(f"⚠️ 新版本有更多异常值!")

if __name__ == "__main__":
    main()
