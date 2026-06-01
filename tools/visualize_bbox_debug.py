#!/usr/bin/env python3
"""
3D 边界框可视化诊断工具
对比：点云 vs Fast-SAM3D 提取的边界框
"""

import os
import sys
import json
import numpy as np
import open3d as o3d
import argparse

def load_point_cloud_from_depth(image_path, intrinsics=None):
    """从 RGB+深度图生成点云"""
    from PIL import Image
    import torch
    
    # 读取图像
    img = Image.open(image_path.replace('_rgb.png', '_rgb.png'))
    img = np.array(img) / 255.0
    
    # 尝试读取深度图
    depth_path = image_path.replace('_rgb.png', '_depth.png')
    if os.path.exists(depth_path):
        depth = np.array(Image.open(depth_path)).astype(np.float32) / 1000.0  # mm to m
    else:
        print(f"    ⚠️  No depth file: {depth_path}")
        return None
    
    H, W = depth.shape
    
    # 默认 intrinsics (如果未提供)
    if intrinsics is None:
        fx = fy = W * 1.2  # 近似
        cx, cy = W / 2, H / 2
    else:
        fx, fy = intrinsics[0, 0], intrinsics[1, 1]
        cx, cy = intrinsics[0, 2], intrinsics[1, 2]
    
    # 生成点云
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    z = depth
    x = (u - cx) / fx * z
    y = (v - cy) / fy * z
    
    points = np.stack([x, y, z], axis=-1).reshape(-1, 3)
    
    # 过滤无效点
    valid = (points[:, 2] > 0) & (points[:, 2] < 10) & np.isfinite(points).all(axis=1)
    return points[valid]


def load_pseudo_label(json_path):
    """加载伪标签"""
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data


def create_bbox_mesh(center, dimensions, yaw=0):
    """从边界框参数创建 mesh"""
    from cubercnn.generate_label.util import convert_box_vertices
    
    dx, dy, dz = dimensions
    cx, cy, cz = center
    
    vertices = convert_box_vertices(cx, cy, cz, dx, dy, dz, yaw)
    
    # 创建线框框
    lines = [
        [0, 1], [1, 2], [2, 3], [3, 0],  # bottom
        [4, 5], [5, 6], [6, 7], [7, 4],  # top
        [0, 4], [1, 5], [2, 6], [3, 7],  # verticals
    ]
    
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(vertices)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    
    return line_set


def create_bbox_colored_mesh(center, dimensions, yaw=0, color=[1, 0, 0]):
    """创建带颜色的边界框"""
    bbox = create_bbox_mesh(center, dimensions, yaw)
    colors = [color for _ in range(len(bbox.lines))]
    bbox.colors = o3d.utility.Vector3dVector(colors)
    return bbox


def visualize_single_sample(image_path, pseudo_label_path, output_dir):
    """可视化单个样本"""
    print(f"\n{'='*60}")
    print(f"Visualizing: {os.path.basename(image_path)}")
    print(f"{'='*60}")
    
    # 1. 加载点云
    print(f"    📦 Loading point cloud from image...")
    points = load_point_cloud_from_depth(image_path)
    
    if points is None:
        print(f"    ❌ Failed to load point cloud")
        return False
    
    print(f"    ✅ Point cloud: {len(points)} points")
    print(f"    📍 Point cloud range: X=[{points[:,0].min():.2f}, {points[:,0].max():.2f}], "
          f"Y=[{points[:,1].min():.2f}, {points[:,1].max():.2f}], "
          f"Z=[{points[:,2].min():.2f}, {points[:,2].max():.2f}]")
    
    # 2. 加载伪标签
    print(f"    📦 Loading pseudo labels...")
    pseudo_labels = load_pseudo_label(pseudo_label_path)
    
    if 'annotations' not in pseudo_labels or len(pseudo_labels['annotations']) == 0:
        print(f"    ⚠️  No annotations in pseudo label file")
        return False
    
    # 3. 创建 Open3D 可视化
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=f"BBox Debug: {os.path.basename(image_path)}", width=1280, height=720)
    
    # 3.1 添加点云
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    # 点云着色为灰色
    pcd.paint_uniform_color([0.7, 0.7, 0.7])
    vis.add_geometry(pcd)
    
    # 3.2 添加伪标签边界框
    for i, ann in enumerate(pseudo_labels['annotations']):
        if 'center_cam' not in ann or 'dimensions' not in ann:
            continue
        
        center = ann['center_cam']
        dimensions = ann['dimensions']
        yaw = ann.get('yaw', 0)
        label = ann.get('category_name', f'obj_{i}')
        
        print(f"    📦 BBox {i}: {label}")
        print(f"        Center: [{center[0]:.3f}, {center[1]:.3f}, {center[2]:.3f}]")
        print(f"        Size:   [{dimensions[0]:.3f}, {dimensions[1]:.3f}, {dimensions[2]:.3f}]")
        print(f"        Yaw:    {yaw:.3f}")
        
        # 检查边界框是否合理
        z_in_range = 0 < center[2] < 10
        size_reasonable = all(0.01 < d < 5 for d in dimensions)
        
        if not z_in_range:
            print(f"        ⚠️  WARNING: Z depth out of reasonable range!")
        if not size_reasonable:
            print(f"        ⚠️  WARNING: Size seems unreasonable!")
        
        # 创建边界框
        bbox = create_bbox_colored_mesh(center, dimensions, yaw, color=[1, 0, 0])  # 红色
        vis.add_geometry(bbox)
    
    # 4. 保存截图
    vis.poll_events()
    vis.update_renderer()
    
    output_path = os.path.join(output_dir, f"debug_{os.path.basename(image_path)}.png")
    vis.capture_screen_image(output_path)
    print(f"    📸 Saved screenshot: {output_path}")
    
    vis.destroy_window()
    
    return True


def main():
    parser = argparse.ArgumentParser(description="Visualize 3D BBox vs Point Cloud")
    parser.add_argument("--image_dir", type=str, required=True, help="SUNRGBD image directory")
    parser.add_argument("--pseudo_label_dir", type=str, required=True, help="Pseudo label JSON directory")
    parser.add_argument("--num_samples", type=int, default=5, help="Number of samples to visualize")
    parser.add_argument("--output_dir", type=str, default="./bbox_debug", help="Output directory")
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 找到伪标签文件
    pseudo_files = sorted([f for f in os.listdir(args.pseudo_label_dir) if f.endswith('.json')])[:args.num_samples]
    
    print(f"Found {len(pseudo_files)} pseudo label files")
    
    for pf in pseudo_files:
        image_name = pf.replace('.json', '.png')
        
        # 尝试在多个位置找图片
        possible_paths = [
            os.path.join(args.image_dir, image_name),
            os.path.join(args.image_dir, image_name.replace('_rgb.png', '_rgb.png')),
        ]
        
        image_path = None
        for p in possible_paths:
            if os.path.exists(p):
                image_path = p
                break
        
        if image_path is None:
            # 尝试找对应的图片目录
            sample_id = pf.replace('.json', '')
            for root, dirs, files in os.walk(args.image_dir):
                if sample_id in root:
                    for f in files:
                        if f.endswith('.png') and 'rgb' in f.lower():
                            image_path = os.path.join(root, f)
                            break
        
        if image_path is None:
            print(f"    ❌ Image not found for {pf}")
            continue
        
        try:
            visualize_single_sample(image_path, os.path.join(args.pseudo_label_dir, pf), args.output_dir)
        except Exception as e:
            print(f"    ❌ Error: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
