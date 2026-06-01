#!/usr/bin/env python3
"""
诊断脚本：对比原版 Omni3D 标注 vs 你生成的伪标签
找出 AP3D=0 的根本原因
"""

import torch
import json
import numpy as np
from pathlib import Path

def analyze_original_annotations(json_path, max_samples=100):
    """分析原版 Omni3D 标注"""
    print(f"\n{'='*70}")
    print(f"分析原版 Omni3D 标注: {json_path}")
    print(f"{'='*70}")
    
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    annotations = data.get('annotations', [])[:max_samples]
    
    centers_x, centers_y, centers_z = [], [], []
    dims_d, dims_w, dims_h = [], [], []
    samples = []
    
    for ann in annotations:
        center = ann.get('center_cam', [])
        dims = ann.get('dimensions', [])
        
        if len(center) == 3 and len(dims) == 3:
            centers_x.append(center[0])
            centers_y.append(center[1])
            centers_z.append(center[2])
            dims_d.append(dims[0])
            dims_w.append(dims[1])
            dims_h.append(dims[2])
            
            samples.append({
                'category': ann.get('category_name', ann.get('category_id', '?')),
                'center': center,
                'dims': dims,
                'R_cam': ann.get('R_cam', [])
            })
    
    print(f"\n--- 3D 中心坐标 (米) ---")
    print(f"X: min={min(centers_x):.3f}, max={max(centers_x):.3f}, mean={np.mean(centers_x):.3f}")
    print(f"Y: min={min(centers_y):.3f}, max={max(centers_y):.3f}, mean={np.mean(centers_y):.3f}")
    print(f"Z: min={min(centers_z):.3f}, max={max(centers_z):.3f}, mean={np.mean(centers_z):.3f}")
    
    print(f"\n--- 3D 尺寸 (米) [dz, dy, dx] ---")
    print(f"D(depth): min={min(dims_d):.3f}, max={max(dims_d):.3f}, mean={np.mean(dims_d):.3f}")
    print(f"H(height): min={min(dims_h):.3f}, max={max(dims_h):.3f}, mean={np.mean(dims_h):.3f}")
    print(f"W(width): min={min(dims_w):.3f}, max={max(dims_w):.3f}, mean={np.mean(dims_w):.3f}")
    
    print(f"\n--- 前5个样本 ---")
    for s in samples[:5]:
        print(f"  {s['category']}: center={s['center']}, dims={s['dims']}")
    
    return {
        'cx': centers_x, 'cy': centers_y, 'cz': centers_z,
        'dd': dims_d, 'dw': dims_w, 'dh': dims_h,
        'samples': samples
    }

def analyze_pseudo_labels(info_path, max_samples=100):
    """分析你生成的伪标签"""
    print(f"\n{'='*70}")
    print(f"分析伪标签: {info_path}")
    print(f"{'='*70}")
    
    data = torch.load(info_path, map_location='cpu')
    
    centers_x, centers_y, centers_z = [], [], []
    dims_d, dims_w, dims_h = [], [], []
    samples = []
    
    for i, (img_id, info) in enumerate(list(data.items())[:max_samples]):
        phrases = info.get('phrases', [])
        dims_list = info.get('dimensions', [])
        centers = info.get('center_cam', [])
        
        for j, (center, dims) in enumerate(zip(centers, dims_list)):
            if not isinstance(center, (list, np.ndarray)) or len(center) != 3:
                continue
            if not isinstance(dims, list) or len(dims) != 3:
                continue
            if dims[0] < 0 or center[0] < -100:  # 跳过无效值
                continue
            
            centers_x.append(center[0])
            centers_y.append(center[1])
            centers_z.append(center[2])
            dims_d.append(dims[0])
            dims_w.append(dims[1])
            dims_h.append(dims[2])
            
            cat = phrases[j] if j < len(phrases) else '?'
            samples.append({
                'img_id': img_id,
                'category': cat,
                'center': center,
                'dims': dims
            })
    
    if not centers_x:
        print("没有找到有效的 3D 数据!")
        return None
    
    print(f"\n--- 3D 中心坐标 (米) ---")
    print(f"X: min={min(centers_x):.3f}, max={max(centers_x):.3f}, mean={np.mean(centers_x):.3f}")
    print(f"Y: min={min(centers_y):.3f}, max={max(centers_y):.3f}, mean={np.mean(centers_y):.3f}")
    print(f"Z: min={min(centers_z):.3f}, max={max(centers_z):.3f}, mean={np.mean(centers_z):.3f}")
    
    print(f"\n--- 3D 尺寸 (米) [dz, dy, dx] ---")
    print(f"D(depth): min={min(dims_d):.3f}, max={max(dims_d):.3f}, mean={np.mean(dims_d):.3f}")
    print(f"H(height): min={min(dims_h):.3f}, max={max(dims_h):.3f}, mean={np.mean(dims_h):.3f}")
    print(f"W(width): min={min(dims_w):.3f}, max={max(dims_w):.3f}, mean={np.mean(dims_w):.3f}")
    
    print(f"\n--- 前5个样本 ---")
    for s in samples[:5]:
        print(f"  [{s['category']}] img={s['img_id']}: center={s['center']}, dims={s['dims']}")
    
    return {
        'cx': centers_x, 'cy': centers_y, 'cz': centers_z,
        'dd': dims_d, 'dw': dims_w, 'dh': dims_h,
        'samples': samples
    }

def diagnose_differences(original, pseudo):
    """诊断两者差异"""
    print(f"\n{'='*70}")
    print("诊断结果")
    print(f"{'='*70}")
    
    if not original or not pseudo:
        print("无法对比，数据不足")
        return
    
    # 中心点差异
    orig_cx, pseudo_cx = np.mean(original['cx']), np.mean(pseudo['cx'])
    orig_cy, pseudo_cy = np.mean(original['cy']), np.mean(pseudo['cy'])
    orig_cz, pseudo_cz = np.mean(original['cz']), np.mean(pseudo['cz'])
    
    print(f"\n--- 中心点差异 ---")
    print(f"X: 原版={orig_cx:.3f}, 伪标签={pseudo_cx:.3f}, 差={pseudo_cx-orig_cx:.3f}")
    print(f"Y: 原版={orig_cy:.3f}, 伪标签={pseudo_cy:.3f}, 差={pseudo_cy-orig_cy:.3f}")
    print(f"Z: 原版={orig_cz:.3f}, 伪标签={pseudo_cz:.3f}, 差={pseudo_cz-orig_cz:.3f}")
    
    # 检测问题
    issues = []
    
    # 问题1: 中心点是否在合理范围
    if abs(pseudo_cx) > 10:
        issues.append(f"❌ X 中心偏移过大: {pseudo_cx:.3f} 米")
    if abs(pseudo_cy) > 10:
        issues.append(f"❌ Y 中心偏移过大: {pseudo_cy:.3f} 米")
    if pseudo_cz < 0:
        issues.append(f"❌ Z 深度为负: {pseudo_cz:.3f} 米")
    
    # 问题2: 尺寸是否合理
    orig_d = np.mean(original['dd'])
    pseudo_d = np.mean(pseudo['dd'])
    
    print(f"\n--- 尺寸差异 ---")
    print(f"Depth: 原版={orig_d:.3f}, 伪标签={pseudo_d:.3f}, 比率={pseudo_d/(orig_d+0.001):.3f}x")
    
    if pseudo_d / (orig_d + 0.001) > 2 or pseudo_d / (orig_d + 0.001) < 0.5:
        issues.append(f"❌ Depth 尺寸差异过大: {pseudo_d/orig_d:.2f}x")
    
    # 问题3: 坐标系偏移
    offset = np.sqrt((pseudo_cx - orig_cx)**2 + (pseudo_cy - orig_cy)**2 + (pseudo_cz - orig_cz)**2)
    print(f"\n3D 中心点总体偏移: {offset:.3f} 米")
    
    if offset > 5:
        issues.append(f"❌ 中心点总体偏移过大: {offset:.3f} 米 - 可能存在坐标系不匹配")
    elif offset > 1:
        issues.append(f"⚠️ 中心点存在偏移: {offset:.3f} 米")
    else:
        issues.append(f"✅ 中心点位置正常")
    
    # 输出结论
    print(f"\n{'='*70}")
    print("问题诊断结论")
    print(f"{'='*70}")
    
    if issues:
        for issue in issues:
            print(f"  {issue}")
    else:
        print("  ✅ 未发现明显问题")
    
    # 具体建议
    print(f"\n{'='*70}")
    print("改进建议")
    print(f"{'='*70}")
    
    if offset > 5:
        print("  1. 检查坐标系转换 - FastSAM3D 输出的坐标系可能与 Omni3D 不一致")
        print("  2. 检查相机内参 K 是否正确传递")
        print("  3. 检查 pointmap 的深度单位（应该是米）")
    if pseudo_d / (orig_d + 0.001) > 2:
        print("  4. 检查尺度转换 - Mesh 的尺度可能需要归一化")
    if abs(pseudo_cy) > 10:
        print("  5. 检查 Y 轴方向 - 可能需要翻转或添加偏移")

def main():
    # 原版 Omni3D 标注
    original_json = "/data/ZhaoX/OVM3D-Dett/datasets/Omni3D_pl/SUNRGBD_train.json"
    
    # 你生成的伪标签
    pseudo_info = "/extra/ZhaoX/pseudo_label_gsam_train/info_3d.pth"
    
    # 分析
    original = analyze_original_annotations(original_json, max_samples=500)
    pseudo = analyze_pseudo_labels(pseudo_info, max_samples=500)
    
    # 诊断
    diagnose_differences(original, pseudo)

if __name__ == "__main__":
    main()
