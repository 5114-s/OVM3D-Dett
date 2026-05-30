"""
OVM3D-Det 完整流程集成
========================

Step 1: Grounded-SAM (Grounding DINO + SAM) - 图像分割
Step 2: Fast-SAM3D - 3D 重建
Step 3: 从 Mesh 提取 3D 边界框

使用方式:
    python tools/ovm3d_pipeline.py \
        --image_path examples/image.png \
        --text_prompt "chair table floor" \
        --output_dir ./output \
        --gpu_id 0
"""

import sys
import os
import argparse
import torch
import numpy as np
import cv2
from PIL import Image
from pathlib import Path
import time

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent.absolute()

# ============================================================
# 【修复1】环境变量配置 (必须在导入 SAM3D 模块之前设置)
# ============================================================
# CUDA 环境
os.environ["CUDA_HOME"] = os.environ.get("CONDA_PREFIX", "")
os.environ["LODIST_SKIP_INIT"] = "true"
os.environ["LIDRA_SKIP_INIT"] = "true"
# PyTorch 缓存目录
os.environ['TORCH_HOME'] = str(PROJECT_ROOT / 'Fast-SAM3D' / 'checkpoints' / 'torch-cache')

# ============================================================
# 【修复2】sys.path 配置 (确保 Fast-SAM3D 在路径中)
# ============================================================
sys.path.insert(0, str(PROJECT_ROOT / 'third_party' / 'Grounded-Segment-Anything'))
sys.path.insert(0, str(PROJECT_ROOT / 'Fast-SAM3D'))
sys.path.insert(0, str(PROJECT_ROOT / 'Fast-SAM3D' / 'notebook'))

# ============================================================
# Step 1: Grounded-SAM 分割模块
# ============================================================
class GroundedSAMSegmentor:
    """Step 1: 使用 Grounded-SAM 进行图像分割"""
    
    def __init__(self, device='cuda'):
        self.device = device
        self.grounding_dino_model = None
        self.sam_model = None
        self.sam_predictor = None
    
    def load_models(self, 
                    grounded_sam_checkpoint: str = None,
                    grounding_dino_checkpoint: str = None,
                    sam_model_type: str = "vit_h"):
        """加载 Grounding DINO 和 SAM 模型"""
        from segment_anything import build_sam, SamPredictor
        from GroundingDINO.groundingdino.util.inference import load_model
        
        print("📦 Loading Grounding DINO model...")
        self.grounding_dino_model = load_model(
            config_path='third_party/Grounded-Segment-Anything/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py',
            grounded_checkpoint=grounding_dino_checkpoint or '',
            device=self.device
        )
        
        print("📦 Loading SAM model...")
        sam_checkpoint = grounded_sam_checkpoint or ''
        self.sam_model = build_sam(checkpoint=sam_checkpoint, model_type=sam_model_type)
        self.sam_model.to(device=self.device)
        self.sam_predictor = SamPredictor(self.sam_model)
        
        print("✅ Grounded-SAM models loaded successfully!")
    
    @torch.no_grad()
    def segment(self, image: np.ndarray, text_prompt: str, box_threshold: float = 0.3, text_threshold: float = 0.25):
        """
        使用文本提示进行分割
        
        Args:
            image: RGB 图像 (H, W, 3)
            text_prompt: 文本提示，如 "chair table floor"
            box_threshold: 框检测阈值
            text_threshold: 文本匹配阈值
        
        Returns:
            dict: {
                'boxes': List of bounding boxes [x1, y1, x2, y2],
                'labels': List of class labels,
                'masks': List of segmentation masks,
                'scores': List of confidence scores
            }
        """
        from GroundingDINO.groundingdino.util.inference import load_model, predict
        from segment_anything import build_sam, SamPredictor
        
        # Step 1: 使用 Grounding DINO 检测
        print(f"🔍 Detecting objects with text: '{text_prompt}'")
        boxes, logits, phrases = predict(
            model=self.grounding_dino_model,
            image=image,
            caption=text_prompt,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            device=self.device
        )
        
        # Step 2: 使用 SAM 生成精确掩码
        print(f"🎯 Generating masks for {len(boxes)} detected objects...")
        self.sam_predictor.set_image(image)
        
        transformed_boxes = self.sam_predictor.transform.apply_boxes_torch(
            boxes, image.shape[:2]
        ).to(self.device)
        
        masks, scores, _ = self.sam_predictor.predict_torch(
            points=None,
            boxes=transformed_boxes,
            multimask_output=False
        )
        
        # 转换结果
        results = {
            'boxes': boxes.cpu().numpy(),
            'labels': phrases,
            'masks': masks.cpu().numpy(),
            'scores': scores.cpu().numpy()
        }
        
        print(f"✅ Detected {len(boxes)} objects")
        return results
    
    def save_masks(self, masks: np.ndarray, labels: list, output_dir: str, prefix: str = "mask"):
        """保存掩码为单独的文件"""
        os.makedirs(output_dir, exist_ok=True)
        
        for i, (mask, label) in enumerate(zip(masks, labels)):
            # 清理标签名
            clean_label = label.lower().replace(' ', '_')
            mask_path = os.path.join(output_dir, f"{prefix}_{i}_{clean_label}.png")
            
            # 保存掩码
            mask_image = (mask[0].astype(np.uint8) * 255)
            cv2.imwrite(mask_path, mask_image)
            
        print(f"💾 Saved {len(masks)} masks to {output_dir}")


# ============================================================
# Step 2: Fast-SAM3D 3D 重建模块 (与官方代码一致版)
# ============================================================
class FastSAM3DReconstructor:
    """Step 2: 使用 Fast-SAM3D 进行单图像 3D 重建"""
    
    def __init__(self, device='cuda'):
        self.device = device
        self.inference = None
        self.config = None
        self._args = None  # 保存 args 引用
    
    def load_model(self, 
                   pipeline_config: str = 'Fast-SAM3D/checkpoints/hf/pipeline_unidepth.yaml',
                   enable_acceleration: bool = True):
        """
        加载 Fast-SAM3D 模型
        
        【修复】与官方 infer.py 代码保持一致:
        1. 使用 argparse.Namespace 模拟官方参数传递
        2. 正确设置所有加速参数
        3. 在 instantiate() 之前修改配置文件路径
        """
        from omegaconf import OmegaConf
        from notebook.inference import Inference
        from argparse import Namespace
        import argparse
        
        print(f"📦 Loading Fast-SAM3D model from {pipeline_config}...")
        
        # 【修复5】确保 pipeline_config 是绝对路径
        if not os.path.isabs(pipeline_config):
            pipeline_config = str(PROJECT_ROOT / pipeline_config)
        
        config = OmegaConf.load(pipeline_config)
        config.workspace_dir = os.path.dirname(pipeline_config)
        
        # 【修复5】在 instantiate() 之前修改配置
        if enable_acceleration:
            config['ss_generator_config_path'] = "ss_generator_faster.yaml"
            config['slat_generator_config_path'] = "slat_generator_faster.yaml"
        
        # 【修复3】使用 argparse.Namespace 与官方保持一致
        parser = argparse.ArgumentParser()
        # SS Generator 加速参数
        parser.add_argument('--ss_cache_stride', type=int, default=3)
        parser.add_argument('--ss_warmup', type=int, default=2)
        parser.add_argument('--ss_order', type=int, default=1)
        parser.add_argument('--ss_momentum_beta', type=float, default=0.5)
        # SLaT Generator 加速参数
        parser.add_argument('--slat_thresh', type=float, default=1.5)
        parser.add_argument('--slat_warmup', type=int, default=3)
        parser.add_argument('--slat_carving_ratio', type=float, default=0.1)
        # Mesh 加速参数
        parser.add_argument('--mesh_spectral_threshold_low', type=float, default=0.5)
        parser.add_argument('--mesh_spectral_threshold_high', type=float, default=0.7)
        # 加速开关
        parser.add_argument('--enable_ss_cache', action='store_true', default=enable_acceleration)
        parser.add_argument('--enable_slat_carving', action='store_true', default=enable_acceleration)
        parser.add_argument('--enable_mesh_aggregation', action='store_true', default=enable_acceleration)
        parser.add_argument('--enable_acceleration', action='store_true', default=enable_acceleration)
        
        args = parser.parse_args([])
        
        # 【修复3】当 enable_acceleration=True 时，确保所有加速选项开启
        if enable_acceleration:
            args.enable_ss_cache = True
            args.enable_slat_carving = True
            args.enable_mesh_aggregation = True
            args.enable_acceleration = True
        
        self._args = args
        
        # 【修复3】创建 Inference 并显式传递参数
        self.inference = Inference(config, compile=False, args=args)
        
        # 【修复3】显式调用 get_params() 确保参数生效
        if hasattr(self.inference, 'get_params'):
            self.inference.get_params(args)
        
        self.config = config
        print("✅ Fast-SAM3D model loaded successfully!")
        print(f"   ✅ SS Cache: {args.enable_ss_cache}, Stride: {args.ss_cache_stride}")
        print(f"   ✅ SLAT Carving: {args.enable_slat_carving}, Ratio: {args.slat_carving_ratio}")
        print(f"   ✅ Mesh Aggregation: {args.enable_mesh_aggregation}")
    
    def reconstruct(self, 
                   image: np.ndarray, 
                   mask: np.ndarray,
                   mask_path: str = None,
                   seed: int = 42) -> dict:
        """
        对单个物体进行 3D 重建
        
        【修复】添加 HFER 计算，与官方代码一致
        
        Args:
            image: 原始 RGB 图像
            mask: 物体掩码 (H, W)
            mask_path: 掩码文件路径（用于计算 HFER）
            seed: 随机种子
        
        Returns:
            dict: {
                'gs': 高斯溅射模型,
                'mesh': 3D Mesh,
                'glb': GLB 格式,
                'depth': 深度图,
                'pose': 位姿 (R, t, s)
            }
        """
        from notebook.inference import load_image, load_single_mask
        from fft.fft2d import calculate_hfer_robust
        import tempfile
        
        print(f"🏗️ Starting 3D reconstruction...")
        start_time = time.time()
        
        # 准备输入
        pil_image = Image.fromarray(image)
        
        # 转换为 mask 格式
        if len(mask.shape) == 3:
            mask = mask[:, :, 0]
        mask_pil = Image.fromarray((mask * 255).astype(np.uint8))
        
        # 【修复2】计算 HFER (高频能量比)
        if mask_path is None:
            # 如果没有提供 mask_path，创建临时文件
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
                mask_pil.save(tmp.name)
                tmp_mask_path = tmp.name
            try:
                hfer = calculate_hfer_robust(tmp_mask_path)
            finally:
                os.unlink(tmp_mask_path)
        else:
            hfer = calculate_hfer_robust(mask_path)
        
        # 【修复2】传递 HFER 给 pipeline
        if hasattr(self.inference, 'get_hfer'):
            self.inference.get_hfer(hfer)
            print(f"   📊 HFER calculated: {hfer:.4f}")
        
        # 推理
        output = self.inference(pil_image, mask_pil, seed=seed)
        
        elapsed = time.time() - start_time
        print(f"✅ 3D reconstruction completed in {elapsed:.2f}s")
        
        return output
    
    def save_ply(self, gs_model, output_path: str):
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
        
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        PlyData([PlyElement.describe(elements, 'vertex')]).write(output_path)
        print(f"💾 Saved PLY to {output_path}")


# ============================================================
# Step 3: 从 Mesh 提取 3D 边界框
# ============================================================
class MeshToBBoxExtractor:
    """Step 3: 从 3D Mesh 提取 3D 边界框"""
    
    def __init__(self):
        pass
    
    def extract_from_mesh(self, mesh_path: str, mesh_type: str = 'obj') -> dict:
        """
        从 3D Mesh 文件提取 3D 边界框
        
        Args:
            mesh_path: Mesh 文件路径 (.obj, .ply)
            mesh_type: 'obj' 或 'ply'
        
        Returns:
            dict: {
                'center': [cx, cy, cz],
                'dimensions': [w, h, l],
                'rotation': 3x3 rotation matrix,
                'yaw': yaw angle in radians,
                'bbox_vertices': 8 vertices of the bbox
            }
        """
        import trimesh
        
        # 加载 mesh
        if mesh_type == 'obj':
            mesh = trimesh.load_mesh(mesh_path, process=False)
        elif mesh_type == 'ply':
            mesh = trimesh.load(mesh_path, process=False)
        else:
            raise ValueError(f"Unsupported mesh type: {mesh_type}")
        
        # 获取顶点
        vertices = np.array(mesh.vertices)
        
        # 计算中心
        center = vertices.mean(axis=0)
        
        # 计算 PCA 确定主方向
        centered = vertices - center
        cov = np.cov(centered.T)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        
        # 排序特征向量
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        # 确保右手坐标系
        if np.linalg.det(eigenvectors) < 0:
            eigenvectors[:, 0] *= -1
        
        # 计算旋转后范围的边界
        rotated = centered @ eigenvectors
        ranges = rotated.max(axis=0) - rotated.min(axis=0)
        
        # 排序尺寸 (宽, 高, 长)
        dim_idx = np.argsort(ranges)
        w, h, l = ranges[dim_idx[0]], ranges[dim_idx[1]], ranges[dim_idx[2]]
        
        # 计算 yaw 角 (绕 y 轴)
        yaw = np.arctan2(eigenvectors[0, 0], eigenvectors[2, 0])
        
        # 生成边界框顶点
        bbox_vertices = self._generate_bbox_vertices(center, [w, h, l], eigenvectors)
        
        return {
            'center': center.tolist(),
            'dimensions': [w, h, l],
            'rotation': eigenvectors.tolist(),
            'yaw': float(yaw),
            'bbox_vertices': bbox_vertices.tolist(),
            'eigenvalues': eigenvalues.tolist()
        }
    
    def _generate_bbox_vertices(self, center: np.ndarray, dimensions: list, rotation: np.ndarray) -> np.ndarray:
        """生成边界框的 8 个顶点"""
        w, h, l = dimensions
        cx, cy, cz = center
        
        # 局部坐标系的 8 个顶点
        local_vertices = np.array([
            [-w/2, -h/2, -l/2], [w/2, -h/2, -l/2], [w/2, h/2, -l/2], [-w/2, h/2, -l/2],
            [-w/2, -h/2, l/2], [w/2, -h/2, l/2], [w/2, h/2, l/2], [-w/2, h/2, l/2]
        ])
        
        # 旋转和平移
        return (local_vertices @ rotation.T) + center
    
    def extract_from_pointcloud(self, points: np.ndarray) -> dict:
        """
        从点云提取 3D 边界框 (备选方法)
        
        Args:
            points: Nx3 点云
        
        Returns:
            dict: 边界框参数
        """
        # 计算中心
        center = points.mean(axis=0)
        
        # 去中心化
        centered = points - center
        
        # PCA
        cov = np.cov(centered.T)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        # 旋转后的范围
        rotated = centered @ eigenvectors
        ranges = rotated.max(axis=0) - rotated.min(axis=0)
        
        # yaw
        yaw = np.arctan2(eigenvectors[0, 0], eigenvectors[2, 0])
        
        return {
            'center': center.tolist(),
            'dimensions': ranges.tolist(),
            'rotation': eigenvectors.tolist(),
            'yaw': float(yaw)
        }


# ============================================================
# 完整流程
# ============================================================
class OVM3DPipeline:
    """完整的 OVM3D-Det 流程"""
    
    def __init__(self, device='cuda', gpu_id=0):
        self.device = f"cuda:{gpu_id}" if torch.cuda.is_available() else 'cpu'
        self.segmentor = GroundedSAMSegmentor(device=self.device)
        self.reconstructor = FastSAM3DReconstructor(device=self.device)
        self.bbox_extractor = MeshToBBoxExtractor()
    
    def setup(self, 
              grounded_sam_checkpoint: str = None,
              grounding_dino_checkpoint: str = None,
              fastsam3d_config: str = 'Fast-SAM3D/checkpoints/hf/pipeline_unidepth.yaml',
              enable_acceleration: bool = True):
        """
        初始化所有模型
        
        Args:
            fastsam3d_config: Fast-SAM3D 配置文件（默认使用 UniDepth 版本）
            enable_acceleration: 是否启用加速
        """
        print("=" * 60)
        print("🚀 Setting up OVM3D-Det Pipeline")
        print("=" * 60)
        
        # Step 1: 加载分割模型
        print("\n[Step 1/3] Loading Grounded-SAM...")
        try:
            self.segmentor.load_models(
                grounded_sam_checkpoint=grounded_sam_checkpoint,
                grounding_dino_checkpoint=grounding_dino_checkpoint
            )
        except Exception as e:
            print(f"⚠️ Failed to load Grounded-SAM: {e}")
            print("   Step 1 will use pre-existing masks if available")
        
        # Step 2: 加载 3D 重建模型
        print("\n[Step 2/3] Loading Fast-SAM3D (with UniDepth)...")
        try:
            self.reconstructor.load_model(
                pipeline_config=fastsam3d_config,
                enable_acceleration=enable_acceleration
            )
        except Exception as e:
            print(f"⚠️ Failed to load Fast-SAM3D: {e}")
            print("   Step 2 will be skipped")
        
        # Step 3: 初始化边界框提取器
        print("\n[Step 3/3] Initializing BBox extractor...")
        print("✅ Pipeline setup complete!")
        print("=" * 60)
    
    def run(self, 
            image_path: str,
            text_prompt: str,
            output_dir: str,
            mask_dir: str = None,
            save_meshes: bool = True,
            save_visualizations: bool = True) -> dict:
        """
        运行完整流程
        
        Args:
            image_path: 输入图像路径
            text_prompt: 文本提示
            output_dir: 输出目录
            mask_dir: 可选，预存在的掩码目录
            save_meshes: 是否保存 3D Mesh
            save_visualizations: 是否保存可视化结果
        
        Returns:
            dict: {
                'segmentation': Step 1 结果,
                'reconstruction': Step 2 结果,
                'bbox': Step 3 结果
            }
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # 读取图像
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        results = {
            'image_path': image_path,
            'text_prompt': text_prompt,
            'objects': []
        }
        
        # ============ Step 1: 分割 ============
        print("\n" + "=" * 60)
        print("[Step 1/3] Image Segmentation with Grounded-SAM")
        print("=" * 60)
        
        if mask_dir and os.path.exists(mask_dir):
            # 使用预存在的掩码
            print(f"📂 Using existing masks from {mask_dir}")
            mask_files = sorted([f for f in os.listdir(mask_dir) if f.endswith('.png')])
            
            segment_results = {'masks': [], 'labels': [], 'boxes': []}
            for mf in mask_files:
                mask = cv2.imread(os.path.join(mask_dir, mf), cv2.IMREAD_GRAYSCALE)
                mask = (mask > 127).astype(np.uint8)
                segment_results['masks'].append(mask)
                segment_results['labels'].append(mf.replace('.png', ''))
                # 估算边界框
                ys, xs = np.where(mask > 0)
                if len(xs) > 0 and len(ys) > 0:
                    segment_results['boxes'].append([xs.min(), ys.min(), xs.max(), ys.max()])
                else:
                    segment_results['boxes'].append([0, 0, 0, 0])
        else:
            # 使用 Grounded-SAM 分割
            segment_results = self.segmentor.segment(image, text_prompt)
            # 保存掩码
            mask_out_dir = os.path.join(output_dir, 'masks')
            self.segmentor.save_masks(segment_results['masks'], segment_results['labels'], mask_out_dir)
        
        results['segmentation'] = segment_results
        
        # ============ Step 2: 3D 重建 ============
        print("\n" + "=" * 60)
        print("[Step 2/3] 3D Reconstruction with Fast-SAM3D")
        print("=" * 60)
        
        mesh_dir = os.path.join(output_dir, 'meshes')
        os.makedirs(mesh_dir, exist_ok=True)
        
        reconstruction_results = []
        
        for i, (mask, label) in enumerate(zip(segment_results['masks'], segment_results['labels'])):
            print(f"\n🏗️ Reconstructing object {i+1}/{len(segment_results['masks'])}: {label}")
            
            try:
                # 3D 重建
                output = self.reconstructor.reconstruct(image, mask)
                
                if save_meshes:
                    # 保存 PLY
                    ply_path = os.path.join(mesh_dir, f"{label}_object_{i}.ply")
                    self.reconstructor.save_ply(output['gs'], ply_path)
                    
                    # 保存 GLB
                    glb_path = os.path.join(mesh_dir, f"{label}_object_{i}.glb")
                    output['glb'].export(glb_path)
                
                reconstruction_results.append({
                    'object_id': i,
                    'label': label,
                    'gs_model': output['gs'],
                    'mesh': output.get('mesh'),
                    'depth': output.get('depth'),
                    'pose': output.get('pose')
                })
                
            except Exception as e:
                print(f"⚠️ Failed to reconstruct {label}: {e}")
                reconstruction_results.append({
                    'object_id': i,
                    'label': label,
                    'error': str(e)
                })
        
        results['reconstruction'] = reconstruction_results
        
        # ============ Step 3: 边界框提取 ============
        print("\n" + "=" * 60)
        print("[Step 3/3] Extracting 3D Bounding Boxes from Meshes")
        print("=" * 60)
        
        bbox_results = []
        
        for i, (rec_result, mask, label) in enumerate(zip(reconstruction_results, 
                                                            segment_results['masks'],
                                                            segment_results['labels'])):
            print(f"\n📦 Extracting BBox for {label}...")
            
            # 尝试从点云提取
            if 'gs_model' in rec_result:
                try:
                    # 从高斯模型获取点云
                    gs_xyz = rec_result['gs_model']._xyz.detach().cpu().numpy()
                    bbox_info = self.bbox_extractor.extract_from_pointcloud(gs_xyz)
                except Exception as e:
                    print(f"⚠️ Failed to extract from GS: {e}")
                    bbox_info = {'error': str(e)}
            else:
                # 尝试从掩码和深度图提取点云
                bbox_info = {'error': 'No 3D model available'}
            
            bbox_results.append({
                'object_id': i,
                'label': label,
                'bbox': bbox_info,
                'mask': mask
            })
            
            print(f"   Center: {bbox_info.get('center', 'N/A')}")
            print(f"   Dimensions: {bbox_info.get('dimensions', 'N/A')}")
        
        results['bbox'] = bbox_results
        
        # 保存结果
        import json
        json_path = os.path.join(output_dir, 'pipeline_results.json')
        
        # 转换 numpy 数组为列表以便 JSON 序列化
        def convert_to_serializable(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(i) for i in obj]
            return obj
        
        serializable_results = convert_to_serializable(results)
        with open(json_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        print(f"\n✅ Results saved to {json_path}")
        print("=" * 60)
        
        return results


# ============================================================
# 命令行接口
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="OVM3D-Det Complete Pipeline")
    
    # 输入输出
    parser.add_argument('--image_path', type=str, required=True, help='Input image path')
    parser.add_argument('--text_prompt', type=str, default='chair table floor', help='Text prompt for segmentation')
    parser.add_argument('--output_dir', type=str, default='./ovm3d_output', help='Output directory')
    parser.add_argument('--mask_dir', type=str, default=None, help='Pre-existing masks directory')
    
    # 模型路径
    parser.add_argument('--grounded_sam_checkpoint', type=str, default=None, help='SAM checkpoint path')
    parser.add_argument('--grounding_dino_checkpoint', type=str, default=None, help='Grounding DINO checkpoint path')
    # 【修改】默认使用 UniDepth 版本配置文件
    parser.add_argument('--fastsam3d_config', type=str, 
                       default='Fast-SAM3D/checkpoints/hf/pipeline_unidepth.yaml', 
                       help='Fast-SAM3D config (default: pipeline_unidepth.yaml)')
    
    # 【新增】加速选项
    parser.add_argument('--enable_acceleration', action='store_true', default=True,
                       help='Enable Fast-SAM3D acceleration (default: True)')
    parser.add_argument('--no_acceleration', action='store_true',
                       help='Disable Fast-SAM3D acceleration')
    
    # 参数
    parser.add_argument('--gpu_id', type=int, default=0, help='GPU ID')
    parser.add_argument('--box_threshold', type=float, default=0.3, help='Box detection threshold')
    parser.add_argument('--text_threshold', type=float, default=0.25, help='Text matching threshold')
    
    args = parser.parse_args()
    
    # 【修改】处理加速选项
    enable_acceleration = args.enable_acceleration and not args.no_acceleration
    
    # 创建并运行流程
    pipeline = OVM3DPipeline(gpu_id=args.gpu_id)
    pipeline.setup(
        grounded_sam_checkpoint=args.grounded_sam_checkpoint,
        grounding_dino_checkpoint=args.grounding_dino_checkpoint,
        fastsam3d_config=args.fastsam3d_config,
        enable_acceleration=enable_acceleration
    )
    
    results = pipeline.run(
        image_path=args.image_path,
        text_prompt=args.text_prompt,
        output_dir=args.output_dir,
        mask_dir=args.mask_dir
    )
    
    print("\n🎉 Pipeline completed!")


if __name__ == "__main__":
    main()
