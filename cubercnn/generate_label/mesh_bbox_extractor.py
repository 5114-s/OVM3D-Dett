"""
Fast-SAM3D Mesh 到 3D 边界框的提取模块
==========================================

此模块用于集成 Fast-SAM3D 的 3D Mesh 输出到现有的 OVM3D-Det 流程中。

与现有 process_indoor.py 的区别:
- 现有: 从点云 (mask * depth) 拟合边界框
- 新增: 从 Fast-SAM3D 生成的 3D Mesh 直接提取边界框 (更精确)

使用方式:
    from cubercnn.generate_label.mesh_bbox_extractor import MeshBBoxExtractor
    
    extractor = MeshBBoxExtractor()
    bbox_result = extractor.extract_from_ply("object.ply")
"""

import os
import numpy as np
import torch
from sklearn.decomposition import PCA
import json


def load_ply_points(filepath):
    """
    从 PLY 文件加载点云数据
    
    Args:
        filepath: PLY 文件路径
    
    Returns:
        np.ndarray: Nx3 点云数据
    """
    from plyfile import PlyData
    
    plydata = PlyData.read(filepath)
    vertex = plydata['vertex']
    
    x = np.array(vertex['x'])
    y = np.array(vertex['y'])
    z = np.array(vertex['z'])
    
    points = np.stack([x, y, z], axis=1)
    return points


def extract_bbox_from_points(points, use_pca=True, prior=None):
    """
    从点云提取 3D 边界框
    
    与现有 process_indoor.py 中的 estimate_bbox 类似，但更加模块化
    
    Args:
        points: Nx3 点云数据
        use_pca: 是否使用 PCA 估计 yaw
        prior: 类别先验尺寸 [w, h, l]，用于约束边界框尺寸
    
    Returns:
        dict: {
            'center': [cx, cy, cz],
            'dimensions': [w, h, l],
            'yaw': float (弧度),
            'rotation_matrix': 3x3 rotation,
            'vertices': 8x3 边界框顶点
        }
    """
    # 下采样（如果点太多）
    if points.shape[0] > 2000:
        indices = np.random.choice(points.shape[0], 2000, replace=False)
        points = points[indices]
    
    # 计算中心
    center = points.mean(axis=0)
    
    if use_pca:
        # 去中心化
        centered = points - center
        
        # PCA 分析
        cov = np.cov(centered.T)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        
        # 按特征值降序排列
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        # 确保右手坐标系
        if np.linalg.det(eigenvectors) < 0:
            eigenvectors[:, 0] *= -1
        
        # 旋转后的点云
        rotated = centered @ eigenvectors
        
        # 计算各轴的范围
        ranges = rotated.max(axis=0) - rotated.min(axis=0)
        
        # 分配长宽高（按轴排序）
        dim_idx = np.argsort(ranges)
        w, h, l = ranges[dim_idx[0]], ranges[dim_idx[1]], ranges[dim_idx[2]]
        
        # 计算 yaw (绕 y 轴的旋转)
        # eigenvectors[:, 0] 是 x 轴方向，eigenvectors[:, 2] 是 z 轴方向
        yaw = np.arctan2(eigenvectors[0, 0], eigenvectors[2, 0])
        
        # 旋转矩阵 = eigenvectors (点云坐标系到世界坐标系)
        R_cam = eigenvectors.copy()
    else:
        # 不使用 PCA，直接计算边界框
        min_pt = points.min(axis=0)
        max_pt = points.max(axis=0)
        
        w = max_pt[0] - min_pt[0]
        h = max_pt[1] - min_pt[1]
        l = max_pt[2] - min_pt[2]
        
        ranges = np.array([w, h, l])
        yaw = 0.0
        R_cam = np.eye(3)
    
    # 如果提供了先验尺寸，使用先验约束
    if prior is not None:
        w_prior, h_prior, l_prior = prior
        
        # 计算当前估计的尺寸
        current_dims = sorted([w, h, l])
        prior_dims = sorted([w_prior, h_prior, l_prior])
        
        # 如果差异太大，使用先验
        for i in range(3):
            if abs(current_dims[i] - prior_dims[i]) / prior_dims[i] > 0.5:
                if i == 0:
                    w = w_prior if w < h_prior else h_prior
                    h = h_prior if w < h_prior else w_prior
                elif i == 1:
                    l = l_prior
        
        # 重新分配尺寸
        dims = sorted([w, h, l])
        dims[0] = w_prior if w < h_prior else h_prior
        dims[1] = h_prior if w < h_prior else w_prior
        dims[2] = l_prior
        w, h, l = dims
    
    # 生成边界框顶点 (8个顶点)
    vertices = generate_box_vertices(center, [w, h, l], R_cam)
    
    return {
        'center': center.tolist(),
        'dimensions': [w, h, l],
        'yaw': float(yaw),
        'rotation_matrix': R_cam.tolist(),
        'vertices': vertices.tolist() if isinstance(vertices, np.ndarray) else vertices
    }


def generate_box_vertices(center, dimensions, rotation_matrix=None):
    """
    生成边界框的 8 个顶点
    
    Args:
        center: [cx, cy, cz] 中心点
        dimensions: [w, h, l] 宽、高、长
        rotation_matrix: 3x3 旋转矩阵
    
    Returns:
        np.ndarray: 8x3 顶点坐标
    """
    cx, cy, cz = center
    w, h, l = dimensions
    
    # 局部坐标系下的 8 个顶点
    local_vertices = np.array([
        [-w/2, -h/2, -l/2],
        [w/2, -h/2, -l/2],
        [w/2, h/2, -l/2],
        [-w/2, h/2, -l/2],
        [-w/2, -h/2, l/2],
        [w/2, -h/2, l/2],
        [w/2, h/2, l/2],
        [-w/2, h/2, l/2]
    ])
    
    if rotation_matrix is not None:
        # 应用旋转
        rotated = local_vertices @ rotation_matrix.T
        return rotated + center
    else:
        return local_vertices + center


def compute_pose_from_mesh(mesh_path, prior=None):
    """
    从 3D Mesh 文件计算位姿
    
    与 process_indoor.py 中的 estimate_bbox 功能相同，但输入是 Mesh 文件
    
    Args:
        mesh_path: PLY 或 OBJ 文件路径
        prior: 类别先验尺寸
    
    Returns:
        dict: {
            'center_cam': [cx, cy, cz],
            'dimensions': [dz, dy, dx],  # 与 Omni3D 格式一致
            'R_cam': 3x3 旋转矩阵,
            'yaw': float,
            'vertices': 8x3 边界框顶点
        }
    """
    # 加载点云
    if mesh_path.endswith('.ply'):
        points = load_ply_points(mesh_path)
    elif mesh_path.endswith('.obj'):
        points = load_obj_points(mesh_path)
    else:
        raise ValueError(f"Unsupported mesh format: {mesh_path}")
    
    # 提取边界框
    bbox = extract_bbox_from_points(points, prior=prior)
    
    # 转换为 Omni3D 格式
    # 注意: Omni3D 中 dimensions 是 [dz, dy, dx]，即深度、高度、宽度
    dimensions_omni = [bbox['dimensions'][2], bbox['dimensions'][1], bbox['dimensions'][0]]
    
    return {
        'center_cam': bbox['center'],
        'dimensions': dimensions_omni,  # [dz, dy, dx]
        'R_cam': bbox['rotation_matrix'],
        'yaw': bbox['yaw'],
        'vertices': bbox['vertices']
    }


def load_obj_points(filepath):
    """
    从 OBJ 文件加载点云数据
    
    Args:
        filepath: OBJ 文件路径
    
    Returns:
        np.ndarray: Nx3 点云数据
    """
    points = []
    with open(filepath, 'r') as f:
        for line in f:
            if line.startswith('v '):
                parts = line.strip().split()
                if len(parts) >= 4:
                    points.append([float(parts[1]), float(parts[2]), float(parts[3])])
    return np.array(points) if points else np.array([])


class MeshBBoxExtractor:
    """
    从 3D Mesh 提取 3D 边界框的类
    
    与现有的 process_indoor.py 配合使用，
    可以使用 Fast-SAM3D 生成的 Mesh 来获得更精确的边界框
    """
    
    def __init__(self):
        self.priors = {}  # 类别先验尺寸
    
    def load_priors(self, dataset='SUNRGBD'):
        """加载类别的先验尺寸"""
        # 从 llm_generated_prior 加载
        from cubercnn.generate_label import llm_generated_prior
        
        if dataset in llm_generated_prior:
            self.priors = llm_generated_prior[dataset]
            print(f"Loaded {len(self.priors)} category priors for {dataset}")
    
    def extract_from_ply(self, ply_path, category=None):
        """从 PLY 文件提取边界框"""
        points = load_ply_points(ply_path)
        prior = self.priors.get(category) if category else None
        return extract_bbox_from_points(points, prior=prior)
    
    def extract_from_directory(self, mesh_dir, category_map=None):
        """
        从目录批量提取边界框
        
        Args:
            mesh_dir: 包含 PLY 文件的目录
            category_map: dict, 文件名到类别的映射
        
        Returns:
            dict: {filename: bbox_result}
        """
        results = {}
        
        for filename in os.listdir(mesh_dir):
            if filename.endswith('.ply'):
                category = category_map.get(filename) if category_map else None
                ply_path = os.path.join(mesh_dir, filename)
                results[filename] = self.extract_from_ply(ply_path, category)
        
        return results


def integrate_with_process_indoor(mesh_dir, mask_dir, depth_dir, info_path, output_path, category_map=None):
    """
    与现有 process_indoor.py 集成的函数
    
    使用 Fast-SAM3D 生成的 Mesh 来改进边界框估计
    
    Args:
        mesh_dir: Fast-SAM3D 输出的 PLY 文件目录
        mask_dir: 原始掩码目录
        depth_dir: 深度图目录
        info_path: info_3d.pth 路径
        output_path: 输出路径
        category_map: dict, image_id 到类别的映射
    
    Returns:
        更新后的 info dict
    """
    import torch
    
    # 加载现有的 info
    info = torch.load(info_path)
    
    # 初始化提取器
    extractor = MeshBBoxExtractor()
    
    # 遍历每个图像
    for im_id, im_info in info.items():
        if not im_info or 'phrases' not in im_info:
            continue
        
        # 查找对应的 mesh 文件
        mesh_files = []
        for phrase in im_info['phrases']:
            clean_phrase = phrase.lower().replace(' ', '_')
            for ext in ['.ply', '.glb']:
                potential_file = f"{clean_phrase}{ext}"
                mesh_path = os.path.join(mesh_dir, potential_file)
                if os.path.exists(mesh_path):
                    mesh_files.append((phrase, mesh_path))
        
        # 如果找到对应的 mesh，使用 mesh 计算边界框
        if mesh_files:
            for i, (phrase, mesh_path) in enumerate(mesh_files):
                try:
                    # 使用 mesh 提取边界框
                    bbox = compute_pose_from_mesh(mesh_path)
                    
                    # 更新 info
                    if 'boxes3d_mesh' not in im_info:
                        im_info['boxes3d_mesh'] = []
                        im_info['center_cam_mesh'] = []
                        im_info['dimensions_mesh'] = []
                        im_info['R_cam_mesh'] = []
                    
                    im_info['boxes3d_mesh'].append(bbox['vertices'])
                    im_info['center_cam_mesh'].append(bbox['center_cam'])
                    im_info['dimensions_mesh'].append(bbox['dimensions'])
                    im_info['R_cam_mesh'].append(bbox['R_cam'])
                    
                    print(f"Updated {im_id}: {phrase} from mesh")
                    
                except Exception as e:
                    print(f"Failed to process {mesh_path}: {e}")
    
    # 保存结果
    torch.save(info, output_path)
    print(f"Saved updated info to {output_path}")
    
    return info


# CLI 接口
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Extract 3D BBox from Mesh")
    parser.add_argument('--mesh', type=str, required=True, help='Path to mesh file (.ply or .obj)')
    parser.add_argument('--output', type=str, default=None, help='Output JSON path')
    parser.add_argument('--category', type=str, default=None, help='Object category for prior')
    
    args = parser.parse_args()
    
    result = compute_pose_from_mesh(args.mesh)
    
    print("\n" + "=" * 50)
    print("3D Bounding Box Result")
    print("=" * 50)
    print(f"Center: {result['center_cam']}")
    print(f"Dimensions [dz, dy, dx]: {result['dimensions']}")
    print(f"Yaw: {result['yaw']:.4f} rad ({np.degrees(result['yaw']):.2f} deg)")
    print(f"Rotation Matrix:\n{np.array(result['R_cam'])}")
    
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"\nSaved to {args.output}")
