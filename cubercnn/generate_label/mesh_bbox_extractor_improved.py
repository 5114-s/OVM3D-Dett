"""
改进版: 从 Fast-SAM3D Mesh 提取 3D 边界框
============================================

与原版 process_indoor.py 的 estimate_bbox 逻辑一致，但输入是 Fast-SAM3D 的 3D Mesh

关键改进:
1. 添加地面平面检测和约束
2. 添加射线追踪优化（当尺寸不合理时）
3. 使用类别先验约束尺寸

使用方式:
    python -m cubercnn.generate_label.mesh_bbox_extractor \
        --mesh output/meshes/chair_0.ply \
        --category chair \
        --ground_plane "0.0, -1.0, 0.0, 0.5"
"""

import os
import sys
import numpy as np
import torch
import argparse
import json
from pathlib import Path

# 导入原版工具函数
from cubercnn.generate_label.util import (
    rotation_matrix_from_vectors, rotate_y, convert_box_vertices,
    point_to_plane_distance, generate_possible_bboxs
)
from cubercnn.generate_label.raytrace import calc_inside_ratio, calc_dis_ray_tracing


def load_ply_points(filepath):
    """从 PLY 文件加载点云"""
    from plyfile import PlyData
    
    plydata = PlyData.read(filepath)
    vertex = plydata['vertex']
    
    x = np.array(vertex['x'])
    y = np.array(vertex['y'])
    z = np.array(vertex['z'])
    
    return np.stack([x, y, z], axis=1)


def fit_ground_plane(points):
    """
    从点云拟合地面平面
    
    Args:
        points: Nx3 点云
    
    Returns:
        np.array: [a, b, c, d] 平面方程 ax + by + cz + d = 0
    """
    from sklearn.decomposition import PCA
    
    # 使用 PCA 找主平面
    pca = PCA(2)
    pca.fit(points)
    
    # 法向量是特征值最小的方向
    normal = pca.components_[1]  # 第二小的特征向量是平面的法向量
    
    # 确保法向量朝上 (y > 0)
    if normal[1] < 0:
        normal = -normal
    
    # 计算 d
    d = -np.dot(normal, points.mean(axis=0))
    
    return np.array([normal[0], normal[1], normal[2], d])


def estimate_bbox_from_mesh(in_pc, prior, ground_equ=None):
    """
    从 Fast-SAM3D Mesh 提取 3D 边界框
    
    与原版 process_indoor.py 的 estimate_bbox 逻辑一致
    
    Args:
        in_pc: Nx3 点云数据
        prior: [w, h, l] 类别先验尺寸
        ground_equ: 地面平面参数 [a, b, c, d]，如果为 None 则自动检测
    
    Returns:
        tuple: (vertices_list, center_cam_list, dimension_list, R_cam_list)
    """
    # 下采样
    if in_pc.shape[0] > 500:
        rand_ind = np.random.randint(0, in_pc.shape[0], 500)
        in_pc = in_pc[rand_ind]
    
    w, h, l = prior
    
    # ========== 地面约束 ==========
    if ground_equ is None:
        # 自动检测地面
        try:
            ground_equ = fit_ground_plane(in_pc)
        except:
            ground_equ = None
    
    if ground_equ is not None:
        dot_product = np.dot([0, -1, 0], ground_equ[:3])
        if dot_product <= 0:
            ground_equ = -ground_equ
        new_ground_equ = np.array([0, -1, 0, point_to_plane_distance(ground_equ, 0, 0, 0)])
        rotation_matrix = rotation_matrix_from_vectors([0, -1, 0], ground_equ[:3])
    else:
        rotation_matrix = np.eye(3)
    
    # 旋转点云到地面平行
    rotated_pc = np.dot(in_pc, rotation_matrix)
    
    # ========== PCA 确定 Yaw ==========
    from sklearn.decomposition import PCA
    pca = PCA(2)
    pca.fit(rotated_pc[:, [0, 2]])  # 在 XZ 平面做 PCA
    yaw_vec = pca.components_[0, :]
    yaw = np.arctan2(yaw_vec[1], yaw_vec[0])
    
    # 旋转点云对齐 x 轴
    rotated_pc_2 = rotate_y(yaw) @ rotated_pc.T
    rotated_pc_2 = rotated_pc_2.T  # 转回 Nx3
    
    # 计算边界
    x_min, x_max = rotated_pc_2[:, 0].min(), rotated_pc_2[:, 0].max()
    y_min, y_max = rotated_pc_2[:, 1].min(), rotated_pc_2[:, 1].max()
    z_min, z_max = rotated_pc_2[:, 2].min(), rotated_pc_2[:, 2].max()
    
    dx, dy, dz = x_max - x_min, y_max - y_min, z_max - z_min
    cx, cy, cz = (x_min + x_max) / 2, (y_min + y_max) / 2, (z_min + z_max) / 2
    
    # ========== 高度约束 ==========
    if dy < h * 0.5:
        dy = h
        if ground_equ is not None:
            cdis = point_to_plane_distance(new_ground_equ, cx, cy, cz)
            if cdis - dy / 2 < 0.5:
                cy += cdis - dy / 2
    
    vertives_list, center_cam_list, dimension_list, R_cam_list = [], [], [], []
    
    # ========== 边界框生成 ==========
    # 如果尺寸在合理范围内，直接使用
    if (l * 0.5 <= dx and w * 0.5 <= dz) or (l * 0.5 <= dz and w * 0.5 <= dx):
        vertives = convert_box_vertices(cx, cy, cz, dx, dy, dz, 0).astype(np.float16)
        vertives = np.dot(rotate_y(-yaw), vertives.T).T
        vertives = np.dot(vertives, rotation_matrix.T)
        vertives_list.append(vertives)
        center_cam = vertives.mean(0)
        dimension = [dz, dy, dx]  # Omni3D 格式: [dz, dy, dx]
        R_cam = rotation_matrix @ rotate_y(-yaw)
        center_cam_list.append(center_cam)
        dimension_list.append(dimension)
        R_cam_list.append(R_cam)
    else:
        # ========== 射线追踪优化（当尺寸不合理时）==========
        possible_bboxs = generate_possible_bboxs(cx, cz, dx, dz, w, l)
        min_loss, min_vertives = float('inf'), None
        
        for possible_bbox in possible_bboxs:
            x_min, x_max, z_min, z_max = possible_bbox
            dx_box, dz_box = x_max - x_min, z_max - z_min
            cx_box, cz_box = (x_min + x_max) / 2, (z_min + z_max) / 2
            
            # 计算 inside ratio
            inside_ratio = calc_inside_ratio(rotated_pc_2.T, x_min, x_max, z_min, z_max)
            
            # 生成顶点
            vertives = convert_box_vertices(cx_box, cy, cz_box, dx_box, dy, dz_box, 0).astype(np.float16)
            vertives = np.dot(rotate_y(-yaw), vertives.T).T
            new_cx, new_cz = vertives[:, 0].mean(), vertives[:, 2].mean()
            
            # 射线追踪损失
            pc_tensor = torch.from_numpy(rotated_pc).float()
            loss_ray_tracing = calc_dis_ray_tracing(
                torch.Tensor([dz_box, dx_box]),
                torch.Tensor([yaw]),
                pc_tensor[:, [0, 2]],
                torch.Tensor([new_cx, new_cz])
            )
            loss_inside_ratio = 1 - inside_ratio
            
            # 总损失
            loss = loss_ray_tracing + 5 * loss_inside_ratio
            
            if loss < min_loss:
                min_loss = loss
                min_vertives = vertives
        
        # 旋转回原坐标系
        min_vertives = np.dot(min_vertives, rotation_matrix.T)
        vertives_list.append(min_vertives)
        center_cam = min_vertives.mean(0)
        dimension = [dz_box, dy, dx_box]
        R_cam = rotation_matrix @ rotate_y(-yaw)
        center_cam_list.append(center_cam)
        dimension_list.append(dimension)
        R_cam_list.append(R_cam)
    
    return vertives_list, center_cam_list, dimension_list, R_cam_list


def extract_bbox_from_ply_improved(ply_path, category_prior=None, ground_equ=None):
    """
    改进版: 从 PLY 提取 3D 边界框
    
    与原版 process_indoor.py 逻辑完全一致
    
    Args:
        ply_path: PLY 文件路径
        category_prior: 类别先验 [w, h, l]
        ground_equ: 地面平面参数 [a, b, c, d]
    
    Returns:
        dict: {
            'center_cam': [cx, cy, cz],
            'dimensions': [dz, dy, dx],
            'R_cam': 3x3 rotation matrix,
            'yaw': float,
            'vertices': 8x3
        }
    """
    # 加载点云
    points = load_ply_points(ply_path)
    
    # 默认先验
    if category_prior is None:
        category_prior = [0.5, 0.5, 0.5]
    
    # 估计边界框
    vertives_list, center_list, dim_list, R_list = estimate_bbox_from_mesh(
        points, category_prior, ground_equ
    )
    
    if len(vertives_list) == 0:
        return None
    
    return {
        'center_cam': center_list[0].tolist(),
        'dimensions': dim_list[0],
        'R_cam': R_list[0].tolist(),
        'yaw': float(np.arctan2(R_list[0][0, 0], R_list[0][2, 0])),
        'vertices': vertives_list[0].tolist()
    }


# ========== CLI 接口 ==========

def main():
    parser = argparse.ArgumentParser(description="Extract 3D BBox from Mesh (Improved)")
    
    parser.add_argument('--mesh', type=str, required=True, help='Path to PLY file')
    parser.add_argument('--output', type=str, default=None, help='Output JSON path')
    parser.add_argument('--category', type=str, default=None, help='Object category for prior')
    parser.add_argument('--prior', type=str, default=None, 
                       help='Prior dimensions as "w,h,l" (e.g., "0.5,0.8,0.5")')
    parser.add_argument('--ground_plane', type=str, default=None,
                       help='Ground plane as "a,b,c,d" (e.g., "0,-1,0,0.5")')
    parser.add_argument('--detect_ground', action='store_true',
                       help='Auto-detect ground plane from point cloud')
    
    args = parser.parse_args()
    
    # 解析先验尺寸
    category_prior = None
    if args.prior:
        category_prior = [float(x) for x in args.prior.split(',')]
    elif args.category:
        from cubercnn.generate_label import llm_generated_prior
        category_prior = llm_generated_prior.get('SUNRGBD', {}).get(args.category)
        if category_prior:
            print(f"Using prior from category: {category_prior}")
    
    # 解析地面平面
    ground_equ = None
    if args.ground_plane:
        ground_equ = np.array([float(x) for x in args.ground_plane.split(',')])
    elif args.detect_ground:
        print("Detecting ground plane...")
        points = load_ply_points(args.mesh)
        ground_equ = fit_ground_plane(points)
        print(f"Detected ground plane: {ground_equ}")
    
    # 提取边界框
    result = extract_bbox_from_ply_improved(args.mesh, category_prior, ground_equ)
    
    if result is None:
        print("❌ Failed to extract bbox")
        return
    
    # 打印结果
    print("\n" + "=" * 50)
    print("3D Bounding Box Result (Improved)")
    print("=" * 50)
    print(f"Center: {result['center_cam']}")
    print(f"Dimensions [dz, dy, dx]: {result['dimensions']}")
    print(f"Yaw: {result['yaw']:.4f} rad ({np.degrees(result['yaw']):.2f} deg)")
    print(f"Rotation Matrix:")
    print(np.array(result['R_cam']))
    
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
