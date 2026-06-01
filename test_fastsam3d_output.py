#!/usr/bin/env python3
"""
测试脚本：验证 FastSAM3D 输出是否包含 scale/shift
"""

import sys
import os
import torch
import numpy as np

# 添加项目路径
sys.path.insert(0, '/data/ZhaoX/OVM3D-Dett')

def test_fastsam3d_output():
    """测试 FastSAM3D 输出格式"""
    from Fast_SAM3D.sam3d_objects.pipeline.inference_pipeline_pointmap import InferencePipelinePointMap
    from Fast_SAM3D.sam3d_objects.pipeline.depth_models.unidepth import UniDepth
    from Fast_SAM3D.sam3d_objects.model.backbone.generator.SLATI.sa_transformer import SATransformer
    
    print("=" * 60)
    print("测试 FastSAM3D 输出")
    print("=" * 60)
    
    # 检查配置文件
    import yaml
    config_path = "/data/ZhaoX/OVM3D-Dett/Fast-SAM3D/configs/pointmap_inference.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    print(f"Config keys: {list(config.keys())}")
    
    # 检查推理管道是否保存了 scale/shift
    pipeline_file = "/data/ZhaoX/OVM3D-Dett/Fast-SAM3D/sam3d_objects/pipeline/inference_pipeline_pointmap.py"
    with open(pipeline_file, 'r') as f:
        content = f.read()
    
    # 检查是否包含我们添加的字段
    if 'depth_physical' in content:
        print("✅ depth_physical 已添加到输出")
    if 'intrinsics_physical' in content:
        print("✅ intrinsics_physical 已添加到输出")
    
    # 检查是否有 scale/shift 相关的代码
    if 'scene_scale' in content or 'pointmap_scale' in content:
        print("✅ scale/shift 相关代码存在")
        
        # 查找具体在哪里使用
        import re
        scale_pattern = r'(scene_scale|pointmap_scale|scene_shift|pointmap_shift)'
        matches = re.findall(scale_pattern, content)
        print(f"   找到 {len(matches)} 处 scale/shift 引用")
    
    return True

def check_ply_depth():
    """检查现有 PLY 文件的深度"""
    from plyfile import PlyData
    
    print("\n" + "=" * 60)
    print("检查现有 PLY 文件")
    print("=" * 60)
    
    # 查找 PLY 文件
    ply_dir = "/extra/ZhaoX/pseudo_label_gsam_train/0000001/meshes"
    if not os.path.exists(ply_dir):
        print(f"目录不存在: {ply_dir}")
        return
    
    ply_files = [f for f in os.listdir(ply_dir) if f.endswith('.ply')][:2]
    
    for ply_file in ply_files:
        ply_path = os.path.join(ply_dir, ply_file)
        print(f"\n--- {ply_file} ---")
        
        try:
            plydata = PlyData.read(ply_path)
            vertex = plydata['vertex']
            
            x = np.array(vertex['x'])
            y = np.array(vertex['y'])
            z = np.array(vertex['z'])
            
            print(f"Z 范围: [{z.min():.4f}, {z.max():.4f}], 均值: {z.mean():.4f}")
            
            # 检查是否是归一化坐标
            if z.max() <= 1.0:
                print("  ⚠️ Z 在 [0, 1] 范围内 - 归一化坐标")
                print("  ⚠️ 需要反归一化才能得到物理深度")
        except Exception as e:
            print(f"错误: {e}")

def main():
    print("FastSAM3D 输出检查工具")
    print("=" * 60)
    
    test_fastsam3d_output()
    check_ply_depth()
    
    print("\n" + "=" * 60)
    print("总结")
    print("=" * 60)
    print("""
修复方案：
1. 已修改 inference_pipeline_pointmap.py 添加 depth_physical 输出
2. 已修改 generate_pseudo_label_gsam.py 尝试获取 scale/shift
3. 如果 scale/shift 不可用，使用物理深度图计算缩放因子

下一步：
1. 重新运行伪标签生成脚本
2. 检查生成的 center_cam Z 值是否在合理范围（1-5米）
""")

if __name__ == "__main__":
    main()
