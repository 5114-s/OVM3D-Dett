#!/usr/bin/env python3
"""
诊断脚本：分析 Fast-SAM3D 点云生成过程中每一步的问题
"""
import os
import sys
import numpy as np
import torch

# 添加 Fast-SAM3D 路径
sys.path.insert(0, '/data/ZhaoX/OVM3D-Dett/Fast-SAM3D')

from omegaconf import OmegaConf
from sam3d_objects.pipeline.inference_pipeline_pointmap import InferencePipelinePointMap
from sam3d_objects.pipeline.depth_models.unidepth import UniDepth

def diagnose_pointmap_issue(image_path=None, image_array=None, mask_array=None):
    """
    诊断点云生成过程中的问题
    
    Args:
        image_path: 图像文件路径
        image_array: numpy 数组格式的图像 (H, W, 3)
        mask_array: numpy 数组格式的 mask (H, W)
    """
    print("=" * 60)
    print("开始诊断点云生成过程...")
    print("=" * 60)
    
    # 1. 加载配置
    config_path = "/data/ZhaoX/OVM3D-Dett/Fast-SAM3D/checkpoints/hf/pipeline_unidepth.yaml"
    config = OmegaConf.load(config_path)
    
    # 2. 加载图像
    if image_path:
        import cv2
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    elif image_array is not None:
        image = image_array
    else:
        raise ValueError("必须提供 image_path 或 image_array")
    
    print(f"\n[Step 1] 加载图像")
    print(f"  - 图像形状: {image.shape}")
    print(f"  - 图像 dtype: {image.dtype}")
    print(f"  - 图像值范围: [{image.min()}, {image.max()}]")
    
    # 3. 加载/准备 mask
    if mask_array is not None:
        mask = mask_array
        if mask.max() <= 1.0:
            mask = (mask * 255).astype(np.uint8)
    else:
        # 创建全 1 mask
        mask = np.ones((image.shape[0], image.shape[1]), dtype=np.uint8) * 255
    
    print(f"\n[Step 2] Mask 信息")
    print(f"  - Mask 形状: {mask.shape}")
    print(f"  - Mask 值范围: [{mask.min()}, {mask.max()}]")
    print(f"  - Mask 非零像素数: {np.count_nonzero(mask)}")
    print(f"  - Mask 非零比例: {np.count_nonzero(mask) / mask.size:.4f}")
    
    # 4. 合并图像和 mask
    rgba_image = np.concatenate([image[..., :3], mask[..., None]], axis=-1)
    print(f"\n[Step 3] 合并 RGBA 图像")
    print(f"  - RGBA 形状: {rgba_image.shape}")
    print(f"  - RGBA dtype: {rgba_image.dtype}")
    
    # 5. 加载并运行 UniDepth
    print(f"\n[Step 4] 运行 UniDepth 深度估计...")
    
    # 加载图像为 tensor
    loaded_image = rgba_image.astype(np.float32) / 255.0
    loaded_image = torch.from_numpy(loaded_image)
    loaded_mask = loaded_image[..., -1]
    loaded_image_t = loaded_image.permute(2, 0, 1).contiguous()[:3]
    
    print(f"  - loaded_image_t 形状: {loaded_image_t.shape}")
    print(f"  - loaded_mask 形状: {loaded_mask.shape}")
    print(f"  - loaded_mask 值范围: [{loaded_mask.min():.3f}, {loaded_mask.max():.3f}]")
    print(f"  - loaded_mask 非零比例: {(loaded_mask > 0.5).sum().item() / loaded_mask.numel():.4f}")
    
    # 初始化 UniDepth
    unidepth_model = UniDepth(
        model=OmegaConf.create({
            "_target_": "unidepth.models.UniDepthV1.from_pretrained",
            "pretrained_model_name_or_path": "/data/ZhaoX/OVM3D-Dett/weights/unidepth_local"
        })
    )
    unidepth_model = unidepth_model.to(device="cuda:0")
    unidepth_model.eval()
    
    # 运行深度模型
    with torch.no_grad():
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            output = unidepth_model(loaded_image_t.cuda())
    
    pointmaps = output["pointmaps"]
    print(f"\n[Step 5] UniDepth 输出")
    print(f"  - pointmaps 类型: {type(pointmaps)}")
    print(f"  - pointmaps 形状: {pointmaps.shape}")
    print(f"  - pointmaps dtype: {pointmaps.dtype}")
    print(f"  - pointmaps 设备: {pointmaps.device}")
    
    # 6. 分析 pointmaps 内容
    print(f"\n[Step 6] Pointmaps 统计分析")
    print(f"  - 所有值是否为 NaN: {torch.isnan(pointmaps).all().item()}")
    print(f"  - 所有值是否为 Inf: {torch.isinf(pointmaps).all().item()}")
    print(f"  - NaN 数量: {torch.isnan(pointmaps).sum().item()}")
    print(f"  - Inf 数量: {torch.isinf(pointmaps).sum().item()}")
    print(f"  - 有效值数量: {(~torch.isnan(pointmaps) & ~torch.isinf(pointmaps)).sum().item()}")
    
    if pointmaps.numel() > 0:
        valid_mask = ~torch.isnan(pointmaps) & ~torch.isinf(pointmaps)
        if valid_mask.any():
            valid_values = pointmaps[valid_mask]
            print(f"  - 有效值范围: [{valid_values.min():.4f}, {valid_values.max():.4f}]")
            print(f"  - 有效值均值: {valid_values.mean():.4f}")
            print(f"  - 有效值中位数: {valid_values.median():.4f}")
    
    # 7. 检查 pointmaps 维度
    print(f"\n[Step 7] Pointmaps 维度检查")
    print(f"  - pointmaps.dim(): {pointmaps.dim()}")
    
    if pointmaps.dim() == 4:  # [B, H, W, 3]
        B, H, W, C = pointmaps.shape
        print(f"  - 批量大小 B: {B}")
        print(f"  - 高度 H: {H}")
        print(f"  - 宽度 W: {W}")
        print(f"  - 通道数 C: {C}")
        
        # 检查每层
        for i in range(min(3, C)):
            layer = pointmaps[..., i]
            valid = ~torch.isnan(layer) & ~torch.isinf(layer)
            if valid.any():
                print(f"  - 通道 {i} 有效值比例: {valid.sum().item() / layer.numel():.4f}")
                print(f"  - 通道 {i} 范围: [{layer[valid].min():.4f}, {layer[valid].max():.4f}]")
    else:
        print(f"  - 非标准格式，可能是 [B, N, 3]")
    
    # 8. 检查 transform_points 后的情况
    print(f"\n[Step 8] Camera Transform 检查")
    try:
        from pytorch3d.transforms import Transform3d
        from sam3d_objects.pipeline.depth_models.utils import camera_to_pytorch3d_camera
        
        camera_convention_transform = (
            Transform3d()
            .rotate(camera_to_pytorch3d_camera(device="cuda:0").rotation)
            .to("cuda:0")
        )
        
        if pointmaps.dim() == 4:
            B, H, W, C = pointmaps.shape
            pointmaps_flat = pointmaps.reshape(B, H * W, C)
        else:
            pointmaps_flat = pointmaps
        
        print(f"  - pointmaps_flat 形状: {pointmaps_flat.shape}")
        
        points_tensor = camera_convention_transform.transform_points(pointmaps_flat)
        print(f"  - transform_points 后形状: {points_tensor.shape}")
        print(f"  - transform 后 NaN 数量: {torch.isnan(points_tensor).sum().item()}")
        print(f"  - transform 后 Inf 数量: {torch.isinf(points_tensor).sum().item()}")
        
        valid = ~torch.isnan(points_tensor) & ~torch.isinf(points_tensor)
        if valid.all():
            print(f"  - 所有 transform 值都有效")
        else:
            print(f"  - 无效值比例: {(~valid).sum().item() / points_tensor.numel():.4f}")
        
    except Exception as e:
        print(f"  - Transform 检查失败: {e}")
    
    # 9. 检查与 mask 交集
    print(f"\n[Step 9] Mask 与 Pointmap 交集检查")
    try:
        points_tensor_final = points_tensor.cpu()
        if pointmaps.dim() == 4:
            points_tensor_final = points_tensor_final.reshape(H, W, C)
        
        # 调整 mask 大小
        mask_bool = (loaded_mask > 0.5).cpu()
        
        print(f"  - Mask 有效像素数: {mask_bool.sum().item()}")
        print(f"  - Pointmap 形状: {points_tensor_final.shape}")
        print(f"  - Mask 形状: {mask_bool.shape}")
        
        if mask_bool.shape != points_tensor_final.shape[:2]:
            print(f"  - 形状不匹配，跳过交集检查")
        else:
            points_masked = points_tensor_final[mask_bool.bool()]
            print(f"  - Mask 区域内的点云形状: {points_masked.shape}")
            
            if points_masked.numel() > 0:
                valid_pts = ~torch.isnan(points_masked).any(dim=-1)
                print(f"  - Mask 区域内有效点数: {valid_pts.sum().item()}")
                print(f"  - Mask 区域内有效点比例: {valid_pts.sum().item() / len(points_masked):.4f}")
                
                if valid_pts.any():
                    valid_points = points_masked[valid_pts]
                    print(f"  - 有效点 X 范围: [{valid_points[:, 0].min():.4f}, {valid_points[:, 0].max():.4f}]")
                    print(f"  - 有效点 Y 范围: [{valid_points[:, 1].min():.4f}, {valid_points[:, 1].max():.4f}]")
                    print(f"  - 有效点 Z 范围: [{valid_points[:, 2].min():.4f}, {valid_points[:, 2].max():.4f}]")
    
    except Exception as e:
        print(f"  - Mask 交集检查失败: {e}")
    
    print("\n" + "=" * 60)
    print("诊断完成")
    print("=" * 60)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="诊断点云生成问题")
    parser.add_argument("--image", type=str, help="图像文件路径")
    args = parser.parse_args()
    
    if args.image:
        diagnose_pointmap_issue(image_path=args.image)
    else:
        # 使用测试图像
        print("请提供 --image 参数指定图像路径")
        print("\n示例:")
        print("  python diagnose_pointmap.py --image /path/to/image.jpg")
