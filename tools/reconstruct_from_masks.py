"""
简化版: 从已有掩码进行 3D 重建和边界框提取
================================================

当 Step 1 (分割) 已经完成后，使用此脚本进行后续处理。

输入格式:
    input_dir/
    ├── images/
    │   ├── scene_001.png
    │   └── scene_002.png
    └── masks/
        ├── scene_001/
        │   ├── 0.png  (chair mask)
        │   └── 1.png  (table mask)
        └── scene_002/
            └── 0.png  (floor mask)
"""

import os
import sys
import json
import time
import numpy as np
import cv2
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import argparse

# 添加路径
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / 'Fast-SAM3D'))


def load_fastsam3d(config_path: str = 'Fast-SAM3D/checkpoints/hf/pipeline.yaml'):
    """加载 Fast-SAM3D 模型"""
    from omegaconf import OmegaConf
    from notebook.inference import Inference
    
    print(f"📦 Loading Fast-SAM3D from {config_path}...")
    
    config = OmegaConf.load(config_path)
    config.workspace_dir = os.path.dirname(os.path.abspath(config_path))
    
    # 使用加速配置
    if os.path.exists(os.path.join(config.workspace_dir, 'ss_generator_faster.yaml')):
        config['ss_generator_config_path'] = "ss_generator_faster.yaml"
    if os.path.exists(os.path.join(config.workspace_dir, 'slat_generator_faster.yaml')):
        config['slat_generator_config_path'] = "slat_generator_faster.yaml"
    
    class Args:
        ss_cache_stride = 3
        ss_warmup = 2
        ss_order = 1
        ss_momentum_beta = 0.5
        slat_thresh = 1.5
        slat_warmup = 3
        slat_carving_ratio = 0.1
        mesh_spectral_threshold_low = 0.5
        mesh_spectral_threshold_high = 0.7
        enable_acceleration = True
    
    inference = Inference(config, compile=False, args=Args())
    print("✅ Fast-SAM3D loaded!")
    return inference


def extract_bbox_from_pointcloud(points: np.ndarray) -> dict:
    """从点云提取 3D 边界框"""
    # 计算中心
    center = points.mean(axis=0)
    
    # 去中心化
    centered = points - center
    
    # PCA 分析
    cov = np.cov(centered.T)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    
    # 排序特征向量
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    # 确保右手坐标系
    if np.linalg.det(eigenvectors) < 0:
        eigenvectors[:, 0] *= -1
    
    # 旋转后的范围
    rotated = centered @ eigenvectors
    ranges = rotated.max(axis=0) - rotated.min(axis=0)
    
    # 计算 yaw 角
    yaw = np.arctan2(eigenvectors[0, 0], eigenvectors[2, 0])
    
    return {
        'center': center.tolist(),
        'dimensions': ranges.tolist(),  # [w, h, l]
        'rotation_matrix': eigenvectors.tolist(),
        'yaw': float(yaw),
        'eigenvalues': eigenvalues.tolist()
    }


def save_ply_from_gs(gs_model, output_path: str):
    """保存高斯溅射为 PLY 文件"""
    from plyfile import PlyData, PlyElement
    
    xyz = gs_model._xyz.detach().cpu().numpy()
    f_dc = gs_model._features_dc.detach().contiguous().cpu().numpy()
    
    SH_C0 = 0.28209479177387814
    rgb = 0.5 + (SH_C0 * f_dc)
    rgb = np.clip(rgb, 0, 1) * 255
    rgb = rgb.astype(np.uint8).squeeze(1)
    
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
             ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    
    elements = np.empty(xyz.shape[0], dtype=dtype)
    elements['x'], elements['y'], elements['z'] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    elements['red'], elements['green'], elements['blue'] = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    PlyData([PlyElement.describe(elements, 'vertex')]).write(output_path)


def process_scene(image_path: str, 
                  mask_dir: str, 
                  output_dir: str,
                  inference,
                  scene_name: str = None) -> dict:
    """处理单个场景的所有物体"""
    if scene_name is None:
        scene_name = Path(image_path).stem
    
    # 读取图像
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(image)
    
    # 获取掩码文件
    mask_files = sorted([f for f in os.listdir(mask_dir) if f.endswith('.png')])
    
    scene_results = {
        'scene_name': scene_name,
        'image_path': image_path,
        'objects': []
    }
    
    scene_output_dir = os.path.join(output_dir, scene_name)
    mesh_dir = os.path.join(scene_output_dir, 'meshes')
    os.makedirs(mesh_dir, exist_ok=True)
    
    for i, mask_file in enumerate(mask_files):
        print(f"  🏗️  Reconstructing object {i+1}/{len(mask_files)}: {mask_file}")
        
        # 读取掩码
        mask = cv2.imread(os.path.join(mask_dir, mask_file), cv2.IMREAD_GRAYSCALE)
        mask = (mask > 127).astype(np.uint8)
        mask_pil = Image.fromarray(mask * 255)
        
        # 3D 重建
        try:
            start_time = time.time()
            output = inference(pil_image, mask_pil, seed=42)
            reconstruct_time = time.time() - start_time
            
            # 保存 PLY
            obj_name = mask_file.replace('.png', '')
            ply_path = os.path.join(mesh_dir, f"{obj_name}.ply")
            save_ply_from_gs(output['gs'], ply_path)
            
            # 保存 GLB
            glb_path = os.path.join(mesh_dir, f"{obj_name}.glb")
            output['glb'].export(glb_path)
            
            # 提取边界框
            gs_xyz = output['gs']._xyz.detach().cpu().numpy()
            bbox = extract_bbox_from_pointcloud(gs_xyz)
            
            # 获取掩码面积（用于过滤小物体）
            mask_area = mask.sum()
            
            obj_result = {
                'object_id': i,
                'mask_file': mask_file,
                'ply_path': ply_path,
                'glb_path': glb_path,
                'mask_area': int(mask_area),
                'num_points': len(gs_xyz),
                'bbox_3d': {
                    'center': bbox['center'],
                    'dimensions': bbox['dimensions'],
                    'yaw': bbox['yaw'],
                    'rotation_matrix': bbox['rotation_matrix']
                },
                'reconstruction_time': reconstruct_time
            }
            
            print(f"    ✅ Center: [{bbox['center'][0]:.3f}, {bbox['center'][1]:.3f}, {bbox['center'][2]:.3f}]")
            print(f"    ✅ Size: [{bbox['dimensions'][0]:.3f}, {bbox['dimensions'][1]:.3f}, {bbox['dimensions'][2]:.3f}]")
            
        except Exception as e:
            print(f"    ❌ Failed: {e}")
            obj_result = {
                'object_id': i,
                'mask_file': mask_file,
                'error': str(e)
            }
        
        scene_results['objects'].append(obj_result)
    
    return scene_results


def main():
    parser = argparse.ArgumentParser(description="3D Reconstruction from Masks")
    
    parser.add_argument('--image_dir', type=str, required=True, 
                       help='Directory containing input images')
    parser.add_argument('--mask_base_dir', type=str, required=True,
                       help='Base directory containing mask subdirectories')
    parser.add_argument('--output_dir', type=str, default='./3d_output',
                       help='Output directory')
    parser.add_argument('--config', type=str, 
                       default='Fast-SAM3D/checkpoints/hf/pipeline.yaml',
                       help='Fast-SAM3D config path')
    parser.add_argument('--gpu', type=int, default=0, help='GPU ID')
    parser.add_argument('--scene_list', type=str, default=None,
                       help='Path to JSON file listing scenes to process')
    
    args = parser.parse_args()
    
    # 设置 GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    
    # 加载模型
    inference = load_fastsam3d(args.config)
    
    # 获取场景列表
    if args.scene_list and os.path.exists(args.scene_list):
        with open(args.scene_list, 'r') as f:
            scene_names = json.load(f)
    else:
        image_files = [f for f in os.listdir(args.image_dir) if f.endswith(('.png', '.jpg'))]
        scene_names = [Path(f).stem for f in image_files]
    
    print(f"\n📁 Processing {len(scene_names)} scenes...")
    print("=" * 60)
    
    all_results = []
    
    for scene_name in tqdm(scene_names, desc="Processing scenes"):
        # 查找图像
        image_path = None
        for ext in ['.png', '.jpg', '.jpeg']:
            potential_path = os.path.join(args.image_dir, f"{scene_name}{ext}")
            if os.path.exists(potential_path):
                image_path = potential_path
                break
        
        if image_path is None:
            print(f"\n⚠️  Image not found for scene: {scene_name}")
            continue
        
        # 查找掩码目录
        mask_dir = os.path.join(args.mask_base_dir, scene_name)
        if not os.path.exists(mask_dir):
            print(f"\n⚠️  Mask directory not found: {mask_dir}")
            continue
        
        # 处理场景
        print(f"\n🎬 Scene: {scene_name}")
        scene_result = process_scene(
            image_path=image_path,
            mask_dir=mask_dir,
            output_dir=args.output_dir,
            inference=inference,
            scene_name=scene_name
        )
        all_results.append(scene_result)
    
    # 保存汇总结果
    output_json = os.path.join(args.output_dir, 'all_results.json')
    with open(output_json, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print("\n" + "=" * 60)
    print(f"✅ Processing complete!")
    print(f"📊 Results saved to: {output_json}")
    print(f"📁 3D meshes saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
