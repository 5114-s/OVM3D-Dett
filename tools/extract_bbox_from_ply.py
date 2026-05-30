#!/usr/bin/env python3
"""
从现有的 PLY 文件重新提取边界框 (使用修复后的坐标系转换)
支持并行处理加速
"""

import os
import sys
import glob
import numpy as np
import torch
from multiprocessing import Pool, cpu_count
from functools import partial

sys.path.insert(0, '/data/ZhaoX/OVM3D-Dett')

from cubercnn.generate_label.util import (
    rotation_matrix_from_vectors, rotate_y, convert_box_vertices,
    point_to_plane_distance, generate_possible_bboxs
)
from cubercnn.generate_label.raytrace import calc_inside_ratio
from plyfile import PlyData
from sklearn.decomposition import PCA
from sklearn.linear_model import RANSACRegressor


def fit_ground_plane_from_points(points):
    if len(points) < 100:
        return None
    X = points[:, [0, 2]]
    y = points[:, 1]
    try:
        model = RANSACRegressor(random_state=42, max_trials=100)
        model.fit(X, y)
        a, b = model.coef_
        c = -1
        d = model.intercept_
        normal = np.array([a, b, c])
        normal = normal / np.linalg.norm(normal)
        if normal[1] > 0:
            normal = -normal
            d = -d
        return np.array([normal[0], normal[1], normal[2], d])
    except:
        return None


def extract_bbox_from_ply(ply_path, prior=None):
    plydata = PlyData.read(ply_path)
    vertex = plydata['vertex']
    
    x = np.array(vertex['x'])
    y = np.array(vertex['y'])
    z = np.array(vertex['z'])
    points = np.stack([x, y, z], axis=1)
    
    # ========== 坐标系转换: OpenGL -> Camera ==========
    points[:, 0] *= -1
    points[:, 2] *= -1
    
    if points.shape[0] > 500:
        indices = np.random.choice(points.shape[0], 500, replace=False)
        points = points[indices]
    
    w, h, l = prior if prior else [0.5, 0.5, 0.5]
    
    ground_equ = fit_ground_plane_from_points(points)
    
    if ground_equ is not None:
        dot_product = np.dot([0, -1, 0], ground_equ[:3])
        if dot_product <= 0:
            ground_equ = -ground_equ
        new_ground_equ = np.array([0, -1, 0, point_to_plane_distance(ground_equ, 0, 0, 0)])
        rotation_matrix = rotation_matrix_from_vectors([0, -1, 0], ground_equ[:3])
    else:
        rotation_matrix = np.eye(3)
    
    rotated_pc = np.dot(points, rotation_matrix)
    
    pca = PCA(2)
    pca.fit(rotated_pc[:, [0, 2]])
    yaw_vec = pca.components_[0, :]
    yaw = np.arctan2(yaw_vec[1], yaw_vec[0])
    
    rotated_pc_2 = rotate_y(yaw) @ rotated_pc.T
    rotated_pc_2 = rotated_pc_2.T
    
    x_min, x_max = rotated_pc_2[:, 0].min(), rotated_pc_2[:, 0].max()
    y_min, y_max = rotated_pc_2[:, 1].min(), rotated_pc_2[:, 1].max()
    z_min, z_max = rotated_pc_2[:, 2].min(), rotated_pc_2[:, 2].max()
    
    dx, dy, dz = x_max - x_min, y_max - y_min, z_max - z_min
    cx, cy, cz = (x_min + x_max) / 2, (y_min + y_max) / 2, (z_min + z_max) / 2
    
    if dy < h * 0.5:
        dy = h
        if ground_equ is not None:
            cdis = point_to_plane_distance(new_ground_equ, cx, cy, cz)
            if cdis - dy / 2 < 0.5:
                cy += cdis - dy / 2
    
    if (l * 0.5 <= dx and w * 0.5 <= dz) or (l * 0.5 <= dz and w * 0.5 <= dx):
        vertives = convert_box_vertices(cx, cy, cz, dx, dy, dz, 0).astype(np.float16)
        vertives = np.dot(rotate_y(-yaw), vertives.T).T
        vertives = np.dot(vertives, rotation_matrix.T)
        center = vertives.mean(axis=0)
        dimensions = [dz, dy, dx]
        R_cam = rotation_matrix @ rotate_y(-yaw)
    else:
        possible_bboxs = generate_possible_bboxs(cx, cz, dx, dz, w, l)
        min_loss, best_vertives = float('inf'), None
        best_dims = None
        
        for possible_bbox in possible_bboxs:
            x_min, x_max, z_min, z_max = possible_bbox
            dx_box, dz_box = x_max - x_min, z_max - z_min
            cx_box, cz_box = (x_min + x_max) / 2, (z_min + z_max) / 2
            
            inside_ratio = calc_inside_ratio(rotated_pc_2.T, x_min, x_max, z_min, z_max)
            vertives = convert_box_vertices(cx_box, cy, cz_box, dx_box, dy, dz_box, 0).astype(np.float16)
            vertives = np.dot(rotate_y(-yaw), vertives.T).T
            vertives = np.dot(vertives, rotation_matrix.T)
            
            loss = -inside_ratio
            
            if loss < min_loss:
                min_loss = loss
                best_vertives = vertives.copy()
                best_dims = [dz_box, dy, dx_box]
        
        if best_vertives is not None:
            vertives = best_vertives
            center = vertives.mean(axis=0)
            dimensions = best_dims
            R_cam = rotation_matrix @ rotate_y(-yaw)
        else:
            vertives = convert_box_vertices(cx, cy, cz, dx, dy, dz, 0).astype(np.float16)
            vertives = np.dot(rotate_y(-yaw), vertives.T).T
            vertives = np.dot(vertives, rotation_matrix.T)
            center = vertives.mean(axis=0)
            dimensions = [dz, dy, dx]
            R_cam = rotation_matrix @ rotate_y(-yaw)
    
    return {
        'vertices': vertives.tolist(),
        'center_cam': center.tolist(),
        'dimensions': dimensions,
        'R_cam': R_cam.tolist()
    }


def process_single_ply(args):
    """处理单个 PLY 文件"""
    ply_path, category_name, prior = args
    
    try:
        bbox = extract_bbox_from_ply(ply_path, prior)
        return {
            'success': True,
            'vertices': np.array(bbox['vertices'], dtype=np.float16),
            'center_cam': np.array(bbox['center_cam']),
            'dimensions': bbox['dimensions'],
            'R_cam': np.array(bbox['R_cam']),
            'category': category_name
        }
    except Exception as e:
        return {
            'success': False,
            'vertices': np.full((8, 3), -1, dtype=np.float16),
            'center_cam': np.array([-1, -1, -1]),
            'dimensions': [-1, -1, -1],
            'R_cam': np.eye(3, dtype=np.float32),
            'category': category_name
        }


def process_image_dir(args):
    """处理单个图像目录"""
    dir_name, pseudo_label_dir, category_prior = args
    
    dir_path = os.path.join(pseudo_label_dir, dir_name)
    if not os.path.isdir(dir_path):
        return None
    
    mesh_dir = os.path.join(dir_path, 'meshes')
    if not os.path.exists(mesh_dir):
        return None
    
    ply_files = glob.glob(os.path.join(mesh_dir, '*.ply'))
    if not ply_files:
        return None
    
    # 准备任务参数
    tasks = []
    labels = []
    
    for ply_path in ply_files:
        basename = os.path.basename(ply_path)
        label = os.path.splitext(basename)[0]
        
        parts = label.rsplit('_', 1)
        category_name = parts[0] if len(parts) == 2 and parts[1].isdigit() else label
        
        labels.append(category_name)
        prior = category_prior.get(category_name, [0.5, 0.5, 0.5])
        tasks.append((ply_path, category_name, prior))
    
    # 串行处理 (避免 multiprocessing 冲突)
    results = [process_single_ply(t) for t in tasks]
    
    boxes3d_list = [r['vertices'] for r in results]
    center_cam_list = [r['center_cam'] for r in results]
    dimensions_list = [r['dimensions'] for r in results]
    R_cam_list = [r['R_cam'] for r in results]
    
    return {
        'image_id': dir_name,
        'boxes3d': boxes3d_list,
        'center_cam': center_cam_list,
        'dimensions': dimensions_list,
        'R_cam': R_cam_list,
        'phrases': labels,
        'conf': [1.0] * len(labels),
        'boxes': [[0, 0, 100, 100]] * len(labels)
    }


def main():
    import time
    from cubercnn.generate_label.llm_generated_prior import SUNRGBD
    
    print("=" * 60)
    print("从现有 PLY 重新提取边界框 (修复坐标系后)")
    print("=" * 60)
    
    pseudo_label_dir = '/extra/ZhaoX_pseudo_label_gsam/'
    output_file = os.path.join(pseudo_label_dir, 'info_3d_fixed.pth')
    
    category_prior = SUNRGBD
    
    # 收集所有图像目录
    image_dirs = []
    for dir_name in sorted(os.listdir(pseudo_label_dir)):
        dir_path = os.path.join(pseudo_label_dir, dir_name)
        if os.path.isdir(dir_path):
            mesh_dir = os.path.join(dir_path, 'meshes')
            if os.path.exists(mesh_dir) and glob.glob(os.path.join(mesh_dir, '*.ply')):
                image_dirs.append(dir_name)
    
    print(f"找到 {len(image_dirs)} 个图像目录")
    
    start_time = time.time()
    all_results = {}
    processed = 0
    
    for i, dir_name in enumerate(image_dirs):
        result = process_image_dir((dir_name, pseudo_label_dir, category_prior))
        
        if result:
            all_results[dir_name] = result
            processed += 1
        
        if (i + 1) % 100 == 0:
            elapsed = time.time() - start_time
            print(f"  [{elapsed:.1f}s] Processed {i + 1}/{len(image_dirs)} images...")
    
    # 保存
    torch.save(all_results, output_file)
    
    elapsed = time.time() - start_time
    
    # Z 值统计
    z_values = []
    for result in all_results.values():
        for center in result['center_cam']:
            if center[0] != -1:
                z_values.append(center[2])
    
    z_values = np.array(z_values)
    
    print(f"\n" + "=" * 60)
    print("处理完成!")
    print("=" * 60)
    print(f"  耗时: {elapsed:.1f} 秒")
    print(f"  处理: {processed} 张图片")
    print(f"\nZ 值统计 (修复后):")
    print(f"  数量: {len(z_values)}")
    print(f"  均值: {z_values.mean():.3f}m")
    print(f"  范围: {z_values.min():.3f} ~ {z_values.max():.3f}m")
    print(f"\n结果保存到: {output_file}")


if __name__ == '__main__':
    main()
