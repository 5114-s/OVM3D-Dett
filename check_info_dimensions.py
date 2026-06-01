#!/usr/bin/env python3
"""检查 info3d 文件中的 dimensions 维度顺序"""

import torch
import os

def check_info_file(filepath):
    """检查 info3d 文件"""
    print(f"\n{'='*60}")
    print(f"检查文件: {filepath}")
    print(f"{'='*60}")
    
    if not os.path.exists(filepath):
        print(f"文件不存在!")
        return
    
    # 加载
    data = torch.load(filepath, map_location='cpu')
    
    # 统计信息
    total_objects = 0
    all_dims = []
    sample_dims = []
    
    for img_id, info in list(data.items())[:5]:  # 只看前5张图
        if 'dimensions' not in info:
            continue
        
        for i, dims in enumerate(info['dimensions']):
            if isinstance(dims, list) and len(dims) == 3:
                all_dims.append(dims)
                if total_objects < 10:  # 保存前10个样本
                    sample_dims.append({
                        'img_id': img_id,
                        'index': i,
                        'dims': dims,
                        'dims_sorted': sorted(dims)
                    })
            total_objects += 1
    
    print(f"\n总物体数量: {total_objects}")
    print(f"总 dimensions 样本数: {len(all_dims)}")
    
    if all_dims:
        import numpy as np
        dims_arr = np.array(all_dims)
        
        print(f"\n--- Dimensions 统计分析 ---")
        print(f"第0维 (应该是 depth): min={dims_arr[:,0].min():.4f}, max={dims_arr[:,0].max():.4f}, mean={dims_arr[:,0].mean():.4f}")
        print(f"第1维 (应该是 height): min={dims_arr[:,1].min():.4f}, max={dims_arr[:,1].max():.4f}, mean={dims_arr[:,1].mean():.4f}")
        print(f"第2维 (应该是 width): min={dims_arr[:,2].min():.4f}, max={dims_arr[:,2].max():.4f}, mean={dims_arr[:,2].mean():.4f}")
        
        # 检查顺序
        print(f"\n--- 维度顺序分析 ---")
        dim0_is_depth = dims_arr[:,0].mean() > dims_arr[:,2].mean()
        dim2_is_smallest = dims_arr[:,2].mean() < dims_arr[:,0].mean() and dims_arr[:,2].mean() < dims_arr[:,1].mean()
        
        print(f"dim0 ({dims_arr[:,0].mean():.4f}) > dim2 ({dims_arr[:,2].mean():.4f}): {dim0_is_depth}")
        
        # Omni3D 正确格式: [dz, dy, dx]，其中 dz > dx
        # 所以 dim0 应该 > dim2（如果 dim0 是 depth）
        if dim0_is_depth:
            print("✅ 维度顺序可能是正确的 [dz, dy, dx]")
        else:
            print("❌ 维度顺序可能是错误的 [dx, dy, dz]")
        
        print(f"\n--- 前10个样本 ---")
        for s in sample_dims[:10]:
            print(f"  img_id={s['img_id']}, idx={s['index']}: dims={s['dims']}, sorted={s['dims_sorted']}")

def main():
    # 检查多个可能的文件
    files = [
        "/extra/ZhaoX/pseudo_label_gsam_train/info_3d.pth",
        "/extra/ZhaoX/pseudo_label_gsam_train/info_3d_0.pth",
        "/extra/ZhaoX/pseudo_label_gsam_train/info_3d_1.pth",
    ]
    
    for f in files:
        if os.path.exists(f):
            check_info_file(f)

if __name__ == "__main__":
    main()
