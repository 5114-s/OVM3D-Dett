"""
修改版的 process_indoor.py，支持 Fast-SAM3D Mesh 输入
=======================================================

主要修改:
1. 添加 --mesh_dir 参数，支持从 Fast-SAM3D 的 3D Mesh 提取边界框
2. 添加 --use_mesh 选项，优先使用 Mesh 而非点云
3. 保留原有点云方法作为备选

使用方式:
    python -m cubercnn.generate_label.process_indoor_mesh \
        --dataset SUNRGBD_train \
        --input_folder pseudo_label/SUNRGBD/train \
        --output_folder output/SUNRGBD/train \
        --mesh_dir Fast-SAM3D_output/meshes \
        --use_mesh
"""

import cv2
import numpy as np
import torch
import argparse
import os
from tqdm import tqdm
from sklearn.decomposition import PCA

# 导入现有工具
from cubercnn.generate_label.util import (
    create_uv_depth, project_image_to_cam, erode_mask,
    adaptive_erode_mask, extract_ground, rotation_matrix_from_vectors,
    rotate_y, convert_box_vertices, point_to_plane_distance,
    generate_possible_bboxs, estimate_bbox
)
from cubercnn.generate_label.raytrace import calc_inside_ratio
from cubercnn.generate_label.mesh_bbox_extractor import (
    compute_pose_from_mesh, load_ply_points, extract_bbox_from_points
)


def process_ground(info_ground, im_id, depth, input_folder, K):
    """处理地面平面检测（与原版相同）"""
    if im_id not in info_ground or not info_ground[im_id]:
        return False, None

    ground_mask = np.load(f'{input_folder}/ground_mask/{im_id}.npy')
    ground_mask = erode_mask(ground_mask.astype(float), 4, 4)
    ground_mask = ground_mask[np.argmax(info_ground[im_id]['conf'])]
    ground_depth = depth * ground_mask.squeeze()

    uv_depth = create_uv_depth(ground_depth)
    pseudo_lidar_ground = project_image_to_cam(uv_depth, np.array(K))

    if pseudo_lidar_ground.shape[0] > 10:
        ground_equ = extract_ground(pseudo_lidar_ground)
        return True, ground_equ
    return False, None


def load_mesh_for_object(mesh_dir, phrase, object_index=None):
    """
    从 Fast-SAM3D 输出目录加载对应的 Mesh 文件
    
    Args:
        mesh_dir: Fast-SAM3D 输出的 meshes 目录
        phrase: 物体类别名称
        object_index: 可选，物体索引
    
    Returns:
        str: Mesh 文件路径，或 None
    """
    clean_phrase = phrase.lower().replace(' ', '_')
    
    # 可能的文件名格式
    possible_names = [
        f"{clean_phrase}_{object_index}.ply" if object_index is not None else None,
        f"{clean_phrase}.ply",
        f"{clean_phrase}_{object_index}.glb" if object_index is not None else None,
        f"{clean_phrase}.glb",
        f"mask_{object_index}.ply" if object_index is not None else None,
        f"object_{object_index}.ply" if object_index is not None else None,
    ]
    
    possible_names = [n for n in possible_names if n is not None]
    
    for name in possible_names:
        path = os.path.join(mesh_dir, name)
        if os.path.exists(path):
            return path
    
    return None


def process_instances_from_mesh(mesh_dir, info_i, category_prior, ground_equ, has_ground):
    """
    从 Fast-SAM3D Mesh 提取 3D 边界框
    
    Args:
        mesh_dir: Fast-SAM3D Mesh 目录
        info_i: 当前图像的信息 dict
        category_prior: 类别先验尺寸
        ground_equ: 地面平面参数
        has_ground: 是否有地面
    
    Returns:
        list: boxes3d, center_cam_list, dimension_list, R_cam_list
    """
    boxes3d = []
    center_cam_list = []
    dimension_list = []
    R_cam_list = []
    
    phrases = info_i.get('phrases', [])
    
    for mask_ind, category_name in enumerate(phrases):
        # 尝试加载对应的 Mesh
        mesh_path = load_mesh_for_object(mesh_dir, category_name, mask_ind)
        
        if mesh_path is None:
            print(f"  ⚠️ Mesh not found for {category_name} (mask {mask_ind}), skipping")
            # 使用默认值
            boxes3d.append(np.full((8, 3), -1))
            center_cam_list.append(-1 * np.ones(3))
            dimension_list.append([-1, -1, -1])
            R_cam_list.append(-1 * np.ones((3, 3)))
            continue
        
        try:
            # 获取先验尺寸
            prior = np.array(category_prior.get(category_name, [0.5, 0.5, 0.5]))
            
            # 从 Mesh 计算边界框
            bbox_result = compute_pose_from_mesh(mesh_path, prior=prior.tolist())
            
            # 转换格式
            vertices = np.array(bbox_result['vertices'])
            boxes3d.append(vertices.astype(np.float16))
            center_cam_list.append(np.array(bbox_result['center_cam']))
            dimension_list.append(bbox_result['dimensions'])
            R_cam_list.append(np.array(bbox_result['R_cam']))
            
            print(f"  ✅ {category_name}: center={bbox_result['center_cam'][:2]}, dims={bbox_result['dimensions']}")
            
        except Exception as e:
            print(f"  ❌ Failed to process {category_name}: {e}")
            boxes3d.append(np.full((8, 3), -1))
            center_cam_list.append(-1 * np.ones(3))
            dimension_list.append([-1, -1, -1])
            R_cam_list.append(-1 * np.ones((3, 3)))
    
    return boxes3d, center_cam_list, dimension_list, R_cam_list


def process_instances_from_pointcloud(mask, depth, K, info_i, cat_prior, has_ground, ground_equ):
    """
    从点云提取 3D 边界框（原有方法）
    """
    boxes3d = []
    center_cam_list = []
    dimension_list = []
    R_cam_list = []

    for mask_ind, cur_mask in enumerate(mask):
        if cur_mask.sum() < 10:
            boxes3d.append(np.full((8, 3), -1))
            center_cam_list.append(-1 * np.ones(3))
            dimension_list.append([-1, -1, -1])
            R_cam_list.append(-1 * np.ones((3, 3)))
            continue

        cur_depth = depth * cur_mask.squeeze(0)
        uv_depth = create_uv_depth(cur_depth)
        pseudo_lidar = project_image_to_cam(uv_depth, np.array(K))

        category_name = info_i["phrases"][mask_ind]
        prior = np.array(cat_prior[category_name])
        bbox_params = estimate_bbox(pseudo_lidar, prior, ground_equ if has_ground else None)

        boxes3d.extend(bbox_params[0])
        center_cam_list.extend(bbox_params[1])
        dimension_list.extend(bbox_params[2])
        R_cam_list.extend(bbox_params[3])

    return boxes3d, center_cam_list, dimension_list, R_cam_list


def process_indoor_mesh(dataset, cat_prior, input_folder, output_folder, mesh_dir=None, use_mesh=True):
    """
    改进版的 process_indoor，支持从 Fast-SAM3D Mesh 提取边界框
    
    Args:
        dataset: 数据集对象
        cat_prior: 类别先验
        input_folder: 输入目录 (包含 mask, depth, info.pth)
        output_folder: 输出目录
        mesh_dir: Fast-SAM3D Mesh 目录（可选）
        use_mesh: 是否优先使用 Mesh（如果可用）
    """
    info = torch.load(os.path.join(input_folder, 'info.pth'))
    info_ground = torch.load(os.path.join(input_folder, 'info_ground.pth'))

    for idx in tqdm(range(len(dataset._dataset))):
        im_id = dataset._dataset[idx]['image_id']
        if im_id not in info or not info[im_id]:
            continue

        depth = np.load(f'{input_folder}/depth/{im_id}.npy')
        mask = np.load(f'{input_folder}/mask/{im_id}.npy')
        mask = adaptive_erode_mask(mask.astype(float), 12, 2, 6, 2)
        K = dataset._dataset[idx]['K']

        # 地面检测
        has_ground, ground_equ = process_ground(info_ground, im_id, depth, input_folder, K)

        # 选择处理方法
        if use_mesh and mesh_dir is not None:
            # 尝试使用 Mesh
            scene_name = str(im_id)
            scene_mesh_dir = os.path.join(mesh_dir, scene_name)
            
            if os.path.exists(scene_mesh_dir):
                boxes3d, center_cam_list, dimension_list, R_cam_list = \
                    process_instances_from_mesh(
                        scene_mesh_dir, info[im_id], cat_prior, ground_equ, has_ground
                    )
            else:
                # 回退到点云方法
                print(f"Mesh dir not found: {scene_mesh_dir}, using point cloud method")
                boxes3d, center_cam_list, dimension_list, R_cam_list = \
                    process_instances_from_pointcloud(
                        mask, depth, K, info[im_id], cat_prior, has_ground, ground_equ
                    )
        else:
            # 使用原有的点云方法
            boxes3d, center_cam_list, dimension_list, R_cam_list = \
                process_instances_from_pointcloud(
                    mask, depth, K, info[im_id], cat_prior, has_ground, ground_equ
                )

        # 更新 info
        info[im_id].update({
            'boxes3d': boxes3d,
            'center_cam': center_cam_list,
            'dimensions': dimension_list,
            'R_cam': R_cam_list
        })

    # 保存结果
    torch.save(info, os.path.join(input_folder, 'info_3d.pth'))


# CLI 入口
def main():
    parser = argparse.ArgumentParser(description="Process Indoor 3D BBox with Mesh Support")
    parser.add_argument('--dataset', type=str, required=True, help='Dataset name (e.g., SUNRGBD_train)')
    parser.add_argument('--input_folder', type=str, required=True, help='Input folder path')
    parser.add_argument('--output_folder', type=str, required=True, help='Output folder path')
    parser.add_argument('--mesh_dir', type=str, default=None, help='Fast-SAM3D Mesh directory')
    parser.add_argument('--use_mesh', action='store_true', help='Use Mesh if available')
    parser.add_argument('--no_mesh', action='store_true', help='Force use point cloud method')
    
    args = parser.parse_args()
    
    # 解析数据集名称
    dataset_name, mode = args.dataset.rsplit('_', 1)
    
    # 加载类别先验
    from cubercnn.generate_label import llm_generated_prior
    cat_prior = llm_generated_prior.get(dataset_name, {})
    
    # 加载数据集
    from detectron2.data import build_detection_test_loader
    from detectron2.config import get_cfg
    from cubercnn.data import simple_register, get_filter_settings_from_cfg
    
    cfg = get_cfg()
    simple_register(args.dataset, get_filter_settings_from_cfg(cfg))
    data_loader = build_detection_test_loader(cfg, args.dataset)
    
    # 处理
    use_mesh = args.use_mesh and not args.no_mesh and args.mesh_dir is not None
    
    if use_mesh:
        print(f"Using Mesh from: {args.mesh_dir}")
    
    process_indoor_mesh(
        data_loader.dataset, 
        cat_prior, 
        args.input_folder, 
        args.output_folder,
        mesh_dir=args.mesh_dir,
        use_mesh=use_mesh
    )


if __name__ == "__main__":
    main()
