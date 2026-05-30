# SUNRGBD 38 类别的文本提示（与原版格式一致，用 ". " 分割）
SUNRGBD_38_CLASSES = "bicycle. books. bottle. chair. cup. laptop. shoes. towel. blinds. window. lamp. shelves. mirror. sink. cabinet. bathtub. door. toilet. desk. box. bookcase. picture. table. counter. bed. night stand. pillow. sofa. television. floor mat. curtain. clothes. stationery. refrigerator. bin. stove. oven. machine"

# 与原版 OVM3D-Det 一致的 SUNRGBD 38 类先验尺寸 (顺序: [X=宽, Y=高, Z=深])
# 来源: /data/ZhaoX/OVM3D-Dettt/cubercnn/generate_label/priors.py
SUNRGBD_ORIGINAL_PRIORS = {
    'bicycle':       [0.5, 1.0, 1.5],
    'books':         [0.2, 0.1, 0.3],
    'bottle':        [0.1, 0.3, 0.1],
    'chair':         [0.5, 1.0, 0.5],
    'cup':           [0.1, 0.1, 0.1],
    'laptop':        [0.3, 0.1, 0.4],
    'shoes':         [0.2, 0.1, 0.3],
    'towel':         [0.2, 0.1, 0.3],
    'blinds':        [0.1, 1.0, 1.5],
    'window':        [0.1, 1.0, 1.5],
    'lamp':          [0.3, 0.6, 0.3],
    'shelves':       [0.3, 1.5, 1.5],
    'mirror':        [0.1, 1.0, 0.5],
    'sink':          [0.5, 0.2, 0.8],
    'cabinet':       [0.5, 1.5, 1.0],
    'bathtub':       [0.8, 0.5, 1.5],
    'door':          [0.1, 2.0, 1.0],
    'toilet':        [0.4, 0.8, 0.5],
    'desk':          [0.6, 0.8, 1.2],
    'box':           [0.5, 0.5, 0.5],
    'bookcase':      [0.3, 2.0, 1.0],
    'picture':       [0.1, 0.5, 0.5],
    'table':         [0.8, 0.8, 1.5],
    'counter':       [0.6, 1.0, 1.5],
    'bed':           [1.5, 0.5, 2.0],
    'night stand':   [0.4, 0.5, 0.5],
    'pillow':        [0.3, 0.3, 0.5],
    'sofa':          [1.0, 1.0, 2.0],
    'television':    [1.0, 0.5, 0.1],
    'floor mat':     [1.0, 0.1, 1.5],
    'curtain':       [0.1, 1.5, 1.0],
    'clothes':       [0.5, 1.0, 0.5],
    'stationery':    [0.3, 0.3, 0.3],
    'refrigerator':  [0.8, 1.5, 0.8],
    'bin':           [0.5, 0.5, 0.5],
    'stove':         [0.6, 0.8, 0.8],
    'oven':          [0.6, 0.8, 0.8],
    'machine':       [0.8, 1.0, 1.0],
}

# 与原版 grounded_sam_detect.py 一致的配置
ORIGINAL_BOX_THRESHOLD = 0.2
ORIGINAL_TEXT_THRESHOLD = 0.2
ORIGINAL_IOU_THRESHOLD = 0.9

"""
Grounding-SAM 2 + Fast-SAM3D 完整流程
=====================================

生成伪标签的完整流程：

Step 1: Grounding-SAM 2 (分割)
    输入: 图像 + SUNRGBD 38 类文本提示
    输出: 掩码 (mask_0.png, mask_1.png, ...)

Step 2: Fast-SAM3D (3D 重建)
    输入: 图像 + 掩码
    输出: 3D Mesh (.ply, .glb)

Step 3: 边界框提取 (与原版 process_indoor.py 逻辑一致)
    输入: 3D Mesh
    输出: 伪标签 (center_cam, dimensions, R_cam, vertices)

Step 4: 转换为 COCO 格式 (可选)
    输入: 伪标签
    输出: COCO JSON 格式

# 使用方法:
# SUNRGBD 数据集 (从 JSON 文件读取)
python tools/generate_pseudo_label_gsam.py \
    --json_file datasets/Omni3D_pl/SUNRGBD_train.json \
    --image_root datasets \
    --dataset SUNRGBD \
    --output_dir pseudo_label_gsam \
    --gpu 0

# 其他数据集 (从目录读取)
python tools/generate_pseudo_label_gsam.py \
    --image_dir /path/to/images \
    --dataset SUNRGBD \
    --output_dir pseudo_label_gsam \
    --gpu 0

# Step 1-4 (包含 COCO 转换)
python tools/generate_pseudo_label_gsam.py \
    --json_file datasets/Omni3D_pl/SUNRGBD_train.json \
    --image_root datasets \
    --dataset SUNRGBD \
    --output_dir pseudo_label_gsam \
    --transform_to_coco \
    --gpu 0
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from tqdm import tqdm
import numpy as np
import cv2
import torch

# 添加路径 - 使用 cwd() 确保在多进程环境下也能正确解析
import os
_project_root = Path(__file__).resolve().parent.parent
# 确保 PROJECT_ROOT 是绝对路径
PROJECT_ROOT = Path(os.path.abspath(_project_root))
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / 'third_party' / 'Grounded-SAM-2'))
sys.path.insert(0, str(PROJECT_ROOT / 'Fast-SAM3D'))


# ============================================================
# Step 1: Grounding-SAM 2 分割
# ============================================================

def generate_substrings(input_string):
    """生成所有可能的子字符串（与原版 grounded_sam_detect.py 一致）"""
    input_list = input_string.split()
    result = []
    n = len(input_list)
    for i in range(n):
        for j in range(i, n):
            result.append(' '.join(input_list[i:j+1]))
    return result


def calculate_iou(mask1, mask2):
    """计算两个掩码的 IoU（与原版一致）"""
    intersection = np.logical_and(mask1, mask2)
    union = np.logical_or(mask1, mask2)
    iou = np.sum(intersection) / np.sum(union) if np.sum(union) > 0 else 0
    return iou


def remove_duplicate_masks(masks, confs, iou_threshold=0.9):
    """
    移除重复的掩码（与原版 grounded_sam_detect.py 一致）
    基于 IoU 阈值，保留置信度最高的掩码
    """
    if len(masks) == 0:
        return []
    
    seen_indices = [0]
    for i in range(1, len(masks)):
        flag = 0
        for ind in seen_indices:
            iou = calculate_iou(masks[i], masks[ind])
            if iou > iou_threshold:
                flag = 1
                seen_ind = ind
                break
        if flag == 0:
            seen_indices.append(i)
        else:
            if confs[i] > confs[seen_ind]:
                seen_indices.remove(seen_ind)
                seen_indices.append(i)
    return seen_indices


def load_grounding_sam2_models(
    grounding_dino_config: str = None,
    grounding_dino_checkpoint: str = None,
    sam_checkpoint: str = None,
    use_large_gdino: bool = False
):
    """
    加载 Grounding-SAM 2 模型

    Args:
        grounding_dino_config: Grounding DINO 配置文件路径
        grounding_dino_checkpoint: Grounding DINO 权重文件路径
        sam_checkpoint: SAM 2 权重文件路径
        use_large_gdino: 是否使用大版本 Grounding DINO (Swin-B)
                          原版使用: groundingdino_swinb_cogcoor.pth

    Returns:
        tuple: (grounding_dino_model, sam2_predictor)
    """
    import os
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    from grounding_dino.groundingdino.util.inference import load_model as load_gdino_model

    # ========== Grounding DINO 配置 ==========
    print("📦 Loading Grounding DINO...")

    if use_large_gdino:
        # 使用大版本 Swin-B (与原版 grounded_sam_detect.py 一致)
        if grounding_dino_config is None:
            grounding_dino_config = str((PROJECT_ROOT / 'third_party' / 'Grounded-SAM-2' /
                                       'grounding_dino' / 'groundingdino' / 'config' / 'GroundingDINO_SwinB_cfg.py').resolve())

        if grounding_dino_checkpoint is None:
            grounding_dino_checkpoint = str((PROJECT_ROOT / 'third_party' / 'Grounded-SAM-2' /
                                            'gdino_checkpoints' / 'groundingdino_swinb_cogcoor.pth').resolve())

        # 如果大版本权重不存在，尝试从外部目录链接
        if not os.path.exists(grounding_dino_checkpoint):
            alt_path = '/extra/OVM3D-Det-1/OVM3D-Det-1/Grounded-SAM-2/checkpoints/groundingdino_swinb_cogcoor.pth'
            if os.path.exists(alt_path):
                grounding_dino_checkpoint = alt_path
                print(f"   使用外部权重: {alt_path}")

        print(f"   模型: Grounding DINO Swin-B (大版本)")
    else:
        # 使用小版本 Swin-T
        if grounding_dino_config is None:
            grounding_dino_config = str((PROJECT_ROOT / 'third_party' / 'Grounded-SAM-2' /
                                       'grounding_dino' / 'groundingdino' / 'config' / 'GroundingDINO_SwinT_OGC.py').resolve())

        if grounding_dino_checkpoint is None:
            grounding_dino_checkpoint = str((PROJECT_ROOT / 'third_party' / 'Grounded-SAM-2' /
                                            'gdino_checkpoints' / 'groundingdino_swint_ogc.pth').resolve())
        print(f"   模型: Grounding DINO Swin-T (小版本)")

    print(f"   Config: {grounding_dino_config}")
    print(f"   Checkpoint: {grounding_dino_checkpoint}")

    grounding_dino_model = load_gdino_model(
        model_config_path=grounding_dino_config,
        model_checkpoint_path=grounding_dino_checkpoint,
        device='cuda'
    )
    
    print("📦 Loading SAM 2...")
    if sam_checkpoint is None:
        sam_checkpoint = str((PROJECT_ROOT / 'third_party' / 'Grounded-SAM-2' / 
                            'checkpoints' / 'sam2.1_hiera_large.pt').resolve())
    
    # SAM2 config path (relative to Grounded-SAM-2/sam2 directory for Hydra)
    sam2_config = 'configs/sam2.1/sam2.1_hiera_l.yaml'
    
    # 切换到 Grounded-SAM-2/sam2 目录让 Hydra 能找到配置
    sam2_dir = str(PROJECT_ROOT / 'third_party' / 'Grounded-SAM-2' / 'sam2')
    original_cwd = os.getcwd()
    try:
        os.chdir(sam2_dir)
        sam2_model = build_sam2(
            config_file=sam2_config,
            ckpt_path=sam_checkpoint,
            device='cuda'
        )
        sam2_predictor = SAM2ImagePredictor(sam2_model)
    finally:
        os.chdir(original_cwd)
    
    print("✅ Grounding-SAM 2 models loaded!")
    return grounding_dino_model, sam2_predictor


def segment_with_grounding_sam2(
    image_source: np.ndarray,
    image_transformed: torch.Tensor,
    text_prompt: str,
    grounding_dino_model,
    sam2_predictor,
    box_threshold: float = 0.2,
    text_threshold: float = 0.2,
    iou_threshold: float = 0.9,
    valid_categories: list = None
) -> dict:
    """
    使用 Grounding-SAM 2 进行分割（与原版 grounded_sam_detect.py 一致）

    Args:
        image_source: 原始 RGB 图像 (H, W, 3), 值在 [0, 255]
        image_transformed: 归一化后的图像张量 (C, H, W)
        text_prompt: 文本提示，如 "bicycle. books. bottle. chair..."
        grounding_dino_model: Grounding DINO 模型
        sam2_predictor: SAM 2 预测器
        box_threshold: 框检测阈值（原版默认 0.2）
        text_threshold: 文本匹配阈值（原版默认 0.2）
        iou_threshold: 掩码去重的 IoU 阈值（原版默认 0.9）
        valid_categories: 有效的类别列表，用于过滤

    Returns:
        dict: {
            'boxes': 边界框列表,
            'labels': 类别名称列表,
            'masks': 掩码列表,
            'scores': 置信度列表
        }
    """
    from grounding_dino.groundingdino.util.inference import predict as gdino_predict
    from torchvision.ops import box_convert

    # 解析有效类别列表（从 text_prompt 中提取）
    if valid_categories is None:
        valid_categories = [cat.strip() for cat in text_prompt.split('.') if cat.strip()]

    # Step 1: Grounding DINO 检测
    # 注意：Grounded-SAM-2 的 predict 只返回 3 个值 (boxes, confidences, phrases)
    # 我们需要手动提取特征用于子字符串匹配
    print(f"🔍 Detecting with box_threshold={box_threshold}, text_threshold={text_threshold}")
    
    # 使用 Grounded-SAM-2 的 predict
    boxes, confidences, phrases = gdino_predict(
        model=grounding_dino_model,
        image=image_transformed,
        caption=text_prompt,
        box_threshold=box_threshold,
        text_threshold=text_threshold,
        device='cuda'
    )
    
    if len(boxes) == 0:
        print("⚠️ No objects detected!")
        return {'boxes': [], 'labels': [], 'masks': [], 'scores': []}

    print(f"📦 Initial detections: {len(boxes)}")

    # Step 2: 使用置信度和短语列表（简化版）
    # 直接使用 Grounding DINO 返回的 boxes 和 phrases
    boxes_list = boxes.cpu().numpy()
    conf_list = confidences.cpu().numpy().tolist()
    phrases_list = phrases
    
    # 过滤出在有效类别中的检测
    if valid_categories:
        filtered_boxes = []
        filtered_confs = []
        filtered_phrases = []
        for box, conf, phrase in zip(boxes_list, conf_list, phrases_list):
            if phrase.lower() in [c.lower() for c in valid_categories]:
                filtered_boxes.append(box)
                filtered_confs.append(conf)
                filtered_phrases.append(phrase)
        boxes_list = filtered_boxes
        conf_list = filtered_confs
        phrases_list = filtered_phrases

    if len(boxes_list) == 0:
        print("⚠️ No valid detections after filtering!")
        return {'boxes': [], 'labels': [], 'masks': [], 'scores': []}

    print(f"🔤 After category filtering: {len(boxes_list)} detections")
    if phrases_list:
        from collections import Counter
        phrase_counts = Counter(phrases_list)
        print(f"   Detected categories: {dict(phrase_counts)}")

    # Step 3: SAM 2 生成掩码
    print(f"🎯 Generating masks for {len(boxes_list)} objects...")
    h, w = image_source.shape[:2]

    # 转换 boxes 到 xyxy 像素格式
    boxes_xyxy = torch.tensor(boxes_list) * torch.tensor([w, h, w, h])

    # 设置图像
    sam2_predictor.set_image(image_source)

    # 生成掩码
    # 注意：boxes_xyxy 是像素坐标，SAM 2 默认 normalize_coords=True，需要设为 False
    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        masks, _, _ = sam2_predictor.predict(
            point_coords=None,
            point_labels=None,
            box=boxes_xyxy.numpy(),
            multimask_output=False,
            normalize_coords=False  # 传入的是像素坐标，不需要归一化
        )

    # 处理掩码格式
    if masks.ndim == 4:
        masks = masks.squeeze(1)
    
    # Step 4: 移除重复掩码
    # 确保 masks 是 numpy 数组
    if hasattr(masks, 'cpu'):
        masks_np = masks.cpu().numpy()
    else:
        masks_np = np.array(masks)
    masks_np = masks_np.astype(bool)
    indices = remove_duplicate_masks(masks_np, conf_list, iou_threshold=iou_threshold)

    boxes_list = [boxes_list[i] for i in indices]
    conf_list = [conf_list[i] for i in indices]
    phrases_list = [phrases_list[i] for i in indices]
    masks_np = [masks_np[i] for i in indices]

    print(f"✅ After deduplication: {len(phrases_list)} unique detections")

    return {
        'boxes': boxes_list,
        'labels': phrases_list,
        'masks': masks_np,
        'scores': conf_list
    }


# ============================================================
# Step 2: Fast-SAM3D 3D 重建
# ============================================================

def load_fastsam3d(
    config_path: str = None,
    compile: bool = False
):
    """
    加载 Fast-SAM3D 模型
    
    Returns:
        Inference pipeline
    """
    from omegaconf import OmegaConf
    
    # 直接导入 inference 模块（notebook 目录没有 __init__.py）
    import importlib.util
    import sys
    
    notebook_dir = PROJECT_ROOT / 'Fast-SAM3D' / 'notebook'
    inference_file = notebook_dir / 'inference.py'
    
    # 添加 Fast-SAM3D 根目录到 sys.path
    fastsam_root = str(PROJECT_ROOT / 'Fast-SAM3D')
    if fastsam_root not in sys.path:
        sys.path.insert(0, fastsam_root)
    
    # 直接加载模块
    spec = importlib.util.spec_from_file_location("inference", inference_file)
    inference_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(inference_module)
    Inference = inference_module.Inference
    
    if config_path is None:
        # 使用 UniDepth 配置（如果存在）或默认 MoGe 配置
        unidepth_config = str(PROJECT_ROOT / 'Fast-SAM3D' / 'checkpoints' / 'hf' / 'pipeline_unidepth.yaml')
        if os.path.exists(unidepth_config):
            config_path = unidepth_config
            print(f"📦 使用 UniDepth 深度估计配置")
        else:
            config_path = str(PROJECT_ROOT / 'Fast-SAM3D' / 'checkpoints' / 'hf' / 'pipeline.yaml')
    
    print(f"📦 Loading Fast-SAM3D from {config_path}...")
    
    config = OmegaConf.load(config_path)
    config.workspace_dir = str(Path(config_path).parent)
    
    # 使用加速配置
    config['ss_generator_config_path'] = "ss_generator_faster.yaml"
    config['slat_generator_config_path'] = "slat_generator_faster.yaml"
    
    # 使用 argparse 与官方代码保持一致
    import argparse
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
    parser.add_argument('--enable_ss_cache', action='store_true', default=True)
    parser.add_argument('--enable_slat_carving', action='store_true', default=True)
    parser.add_argument('--enable_mesh_aggregation', action='store_true', default=False)
    parser.add_argument('--enable_acceleration', action='store_true', default=True)
    
    args = parser.parse_args([])
    
    inference = Inference(config, compile=compile, args=args)
    
    # 【修复】显式调用 get_params() 确保参数生效（与官方代码一致）
    if hasattr(inference, 'get_params'):
        inference.get_params(args)
    
    print("✅ Fast-SAM3D loaded!")
    print(f"   ✅ SS Cache: {args.enable_ss_cache}, Stride: {args.ss_cache_stride}")
    print(f"   ✅ SLAT Carving: {args.enable_slat_carving}, Ratio: {args.slat_carving_ratio}")
    print(f"   ✅ Mesh Aggregation: {args.enable_mesh_aggregation}")
    return inference


def reconstruct_3d(
    image: np.ndarray,
    mask: np.ndarray,
    fastsam_inference,
    seed: int = 42,
    intrinsics: np.ndarray = None
) -> dict:
    """
    使用 Fast-SAM3D 进行 3D 重建

    Args:
        image: RGB 图像
        mask: 二值掩码 (H, W)，值在 [0, 1] 范围
        fastsam_inference: Fast-SAM3D 推理器
        seed: 随机种子
        intrinsics: 相机内参矩阵 (3x3)，如果为 None 则使用默认内参

    Returns:
        dict: {
            'gs': 高斯溅射模型,
            'glb': GLB 格式,
            'ply_path': PLY 文件路径
        }
    """
    from PIL import Image

    # 转换图像格式 - 确保是 numpy 数组
    if isinstance(image, Image.Image):
        image = np.array(image)
    if isinstance(mask, Image.Image):
        mask = np.array(mask)

    # 确保 mask 是 [0, 1] 范围（不是 [0, 255]）
    if mask.max() > 1.0:
        # 如果是 [0, 255] 范围，转换为 [0, 1]
        mask = mask.astype(np.float32) / 255.0
    else:
        mask = mask.astype(np.float32)

    # 【新增】计算 HFER (高频能量比) - 与官方代码一致
    from fft.fft2d import calculate_hfer_robust
    import tempfile
    mask_pil = Image.fromarray((mask * 255).astype(np.uint8))
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
        mask_pil.save(tmp.name)
        tmp_mask_path = tmp.name
    try:
        hfer = calculate_hfer_robust(tmp_mask_path)
        # 传递给 pipeline
        if hasattr(fastsam_inference, 'get_hfer'):
            fastsam_inference.get_hfer(hfer)
        print(f"   📊 HFER: {hfer:.4f}")
    finally:
        os.unlink(tmp_mask_path)

    # 推理 - 传入 intrinsics
    start_time = time.time()
    print(f"   🏗️  Running Fast-SAM3D...")
    output = fastsam_inference(image, mask, seed=seed, intrinsics=intrinsics)
    print(f"   ✅ Fast-SAM3D output keys: {list(output.keys()) if output else 'None'}")
    elapsed = time.time() - start_time

    print(f"   ⏱️  Reconstruction time: {elapsed:.2f}s")

    return output


def save_gs_as_ply(gs_model, output_path: str):
    """保存高斯溅射为 PLY"""
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


# ============================================================
# Step 3: 从 Mesh 提取 3D 边界框
# ============================================================

# ============================================================
# Step 3: 从 Mesh 提取 3D 边界框 (与原版 process_indoor.py 一致)
# ============================================================

def fit_ground_plane_from_points(points):
    """从点云拟合地面平面"""
    from sklearn.decomposition import PCA
    
    pca = PCA(2)
    pca.fit(points)
    normal = pca.components_[1]
    
    if normal[1] < 0:
        normal = -normal
    
    d = -np.dot(normal, points.mean(axis=0))
    return np.array([normal[0], normal[1], normal[2], d])


def extract_bbox_from_ply(ply_path: str, prior: list = None, ground_equ: np.ndarray = None) -> dict:
    """
    从 PLY 点云提取 3D 边界框 (与原版 process_indoor.py 的 estimate_bbox 逻辑一致)
    
    关键特性:
    1. 地面平面检测和约束
    2. PCA 在 XZ 平面估计 yaw
    3. 射线追踪 + inside_ratio 优化（当尺寸不合理时）
    4. 类别先验约束
    """
    from plyfile import PlyData
    from cubercnn.generate_label.util import (
        rotation_matrix_from_vectors, rotate_y, convert_box_vertices,
        point_to_plane_distance, generate_possible_bboxs
    )
    from cubercnn.generate_label.raytrace import calc_inside_ratio, calc_dis_ray_tracing
    
    # 加载点云
    plydata = PlyData.read(ply_path)
    vertex = plydata['vertex']
    
    x = np.array(vertex['x'])
    y = np.array(vertex['y'])
    z = np.array(vertex['z'])
    points = np.stack([x, y, z], axis=1)
    
    # 下采样
    if points.shape[0] > 500:
        indices = np.random.choice(points.shape[0], 500, replace=False)
        points = points[indices]
    
    w, h, l = prior if prior else [0.5, 0.5, 0.5]
    
    # ========== 地面约束 ==========
    if ground_equ is None:
        try:
            ground_equ = fit_ground_plane_from_points(points)
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
    
    rotated_pc = np.dot(points, rotation_matrix)
    
    # ========== PCA 确定 Yaw (在 XZ 平面) ==========
    from sklearn.decomposition import PCA
    pca = PCA(2)
    pca.fit(rotated_pc[:, [0, 2]])
    yaw_vec = pca.components_[0, :]
    yaw = np.arctan2(yaw_vec[1], yaw_vec[0])
    
    # 旋转点云对齐 x 轴
    rotated_pc_2 = rotate_y(yaw) @ rotated_pc.T
    rotated_pc_2 = rotated_pc_2.T
    
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
    
    # ========== 边界框生成 ==========
    # 如果尺寸在合理范围内，直接使用
    if (l * 0.5 <= dx and w * 0.5 <= dz) or (l * 0.5 <= dz and w * 0.5 <= dx):
        vertives = convert_box_vertices(cx, cy, cz, dx, dy, dz, 0).astype(np.float16)
        vertives = np.dot(rotate_y(-yaw), vertives.T).T
        vertives = np.dot(vertives, rotation_matrix.T)
        center = vertives.mean(axis=0)
        dimensions = [dx, dy, dz]  # Omni3D 标准格式: [width, height, length]
        R_cam = rotation_matrix @ rotate_y(-yaw)
    else:
        # ========== 射线追踪优化 ==========
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
            new_cx, new_cz = vertives[:, 0].mean(), vertives[:, 2].mean()
            
            pc_tensor = torch.from_numpy(rotated_pc).float()
            loss_ray_tracing = calc_dis_ray_tracing(
                torch.Tensor([dz_box, dx_box]),
                torch.Tensor([yaw]),
                pc_tensor[:, [0, 2]],
                torch.Tensor([new_cx, new_cz])
            )
            loss_inside_ratio = 1 - inside_ratio
            loss = loss_ray_tracing + 5 * loss_inside_ratio
            
            if loss < min_loss:
                min_loss = loss
                best_vertives = vertives
                best_dims = (dz_box, dx_box)

        if best_vertives is None or best_dims is None:
            # fallback: use the raw bounds bbox (in rotated_pc_2 frame)
            dz_fallback, dx_fallback = float(dz), float(dx)
            vertives = convert_box_vertices(cx, cy, cz, dx_fallback, dy, dz_fallback, 0).astype(np.float16)
            vertives = np.dot(rotate_y(-yaw), vertives.T).T
            vertives = np.dot(vertives, rotation_matrix.T)
            center = vertives.mean(axis=0)
            dimensions = [dx_fallback, dy, dz_fallback]  # Omni3D 标准格式
            R_cam = rotation_matrix @ rotate_y(-yaw)
        else:
            dz_box, dx_box = best_dims
            vertives = np.dot(best_vertives, rotation_matrix.T)
            center = vertives.mean(axis=0)
            dimensions = [dx_box, dy, dz_box]  # Omni3D 标准格式
            R_cam = rotation_matrix @ rotate_y(-yaw)
    
    return {
        'center_cam': center.tolist(),
        'dimensions': dimensions,  # [dx, dy, dz] Omni3D 标准格式: [width, height, length]
        'R_cam': R_cam.tolist(),
        'yaw': float(yaw),
        'vertices': vertives.tolist() if isinstance(vertives, np.ndarray) else vertives
    }


# ============================================================
# 主流程
# ============================================================

def process_single_image(
    image_path: str,
    text_prompt: str,
    grounding_dino_model,
    sam_predictor,
    fastsam_inference,
    category_prior: dict,
    output_dir: str,
    box_threshold: float = 0.3,
    text_threshold: float = 0.25,
    intrinsics: np.ndarray = None
) -> dict:
    """
    处理单张图像，生成伪标签
    
    Returns:
        dict: {
            'image_id': 图像 ID,
            'boxes3d': 边界框顶点列表,
            'center_cam': 中心列表,
            'dimensions': 尺寸列表,
            'R_cam': 旋转矩阵列表,
            'phrases': 类别名称列表,
            'conf': 置信度列表,
            'boxes': 2D 框列表
        }
    """
    # 读取图像
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    image_name = Path(image_path).stem
    image_id = Path(image_path).stem  # 或使用文件 ID
    
    # 创建输出目录
    scene_output_dir = os.path.join(output_dir, image_name)
    mask_dir = os.path.join(scene_output_dir, 'masks')
    mesh_dir = os.path.join(scene_output_dir, 'meshes')
    os.makedirs(mask_dir, exist_ok=True)
    os.makedirs(mesh_dir, exist_ok=True)
    
    # ============ Step 1: 分割 ============
    print(f"\n[Step 1/3] Segmenting {image_name}...")
    
    # 使用 Grounding-SAM 2 的图像加载方式
    from grounding_dino.groundingdino.util.inference import load_image
    image_source, image_transformed = load_image(image_path)
    
    # 原始图像用于 SAM 2 和 Fast-SAM3D (H, W, 3) 格式, 值在 [0, 255]
    original_image = cv2.imread(image_path)
    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    
    seg_results = segment_with_grounding_sam2(
        image_source, image_transformed, text_prompt,
        grounding_dino_model, sam_predictor,
        box_threshold, text_threshold
    )
    
    if len(seg_results['masks']) == 0:
        return None
    
    # 保存掩码
    boxes3d_list = []
    center_cam_list = []
    dimensions_list = []
    R_cam_list = []
    bbox2d_list = []
    
    # ============ Step 2 & 3: 重建 + 边界框 ============
    print(f"\n[Step 2/3] 3D Reconstruction + [Step 3/3] BBox Extraction...")
    
    for i, (mask, label, box) in enumerate(zip(
        seg_results['masks'], seg_results['labels'], seg_results['boxes']
    )):
        print(f"\n  Object {i+1}/{len(seg_results['masks'])}: {label}")
        
        # 检查掩码大小 - 跳过太小的掩码
        mask_pixels = np.sum(mask > 0.5)
        mask_height, mask_width = mask.shape
        mask_area_ratio = mask_pixels / (mask_height * mask_width)
        
        # 最小掩码阈值：至少 100 像素且占比 > 0.001
        if mask_pixels < 100 or mask_area_ratio < 0.001:
            print(f"    ⚠️  Mask too small ({mask_pixels} pixels, ratio={mask_area_ratio:.4f}), skipping Fast-SAM3D")
            boxes3d_list.append(np.full((8, 3), -1, dtype=np.float16))
            center_cam_list.append(np.array([-1, -1, -1]))
            dimensions_list.append([-1, -1, -1])
            R_cam_list.append(np.eye(3, dtype=np.float32))
            bbox2d_list.append(box.tolist())
            continue
        
        # 保存掩码
        mask_image = (mask.astype(np.uint8) * 255)
        
        # ============ 自适应腐蚀（与原版 process_indoor.py 一致）============
        # 从 mask_image (H, W) 创建与原版兼容的格式 (N, 1, H, W)
        from cubercnn.generate_label.util import adaptive_erode_mask as gsam_adaptive_erode
        
        mask_for_erode = np.zeros((1, 1, mask_image.shape[0], mask_image.shape[1]), dtype=np.float32)
        mask_for_erode[0, 0] = (mask_image > 127).astype(np.float32)
        
        # 原版参数：k_vertical=12, k_vertical_min=2, k_horizontal=6, k_horizontal_min=2
        eroded_mask = gsam_adaptive_erode(mask_for_erode, 12, 2, 6, 2)
        eroded_mask_image = (eroded_mask[0, 0] > 0.5).astype(np.uint8) * 255
        
        # 如果侵蚀后掩码太小，使用原始掩码
        eroded_pixels = np.sum(eroded_mask_image > 0)
        if eroded_pixels < 50:
            print(f"    ⚠️  Eroded mask too small ({eroded_pixels} pixels), using original mask")
            eroded_mask_for_fastsam = (mask_image > 0).astype(np.uint8)
            mask_filename = f"{label.replace(' ', '_')}_{i}.png"
            cv2.imwrite(os.path.join(mask_dir, mask_filename), mask_image)
        else:
            eroded_mask_for_fastsam = (eroded_mask_image > 0).astype(np.uint8)
            mask_filename = f"{label.replace(' ', '_')}_{i}.png"
            cv2.imwrite(os.path.join(mask_dir, mask_filename), eroded_mask_image)
            print(f"    🔧 Mask erosion: {mask_pixels} -> {eroded_pixels} pixels")
        
        # 获取先验尺寸
        prior = category_prior.get(label, [0.5, 0.5, 0.5])
        if not (isinstance(prior, (list, tuple)) and len(prior) == 3):
            print(
                f"    ⚠️  Bad prior for label='{label}': {prior} (type={type(prior)}), "
                "fallback to default [0.5, 0.5, 0.5]"
            )
            prior = [0.5, 0.5, 0.5]
        
        # ============ Fast-SAM3D 重建 ============
        try:
            # 检查掩码是否有效（使用侵蚀或原始掩码）
            valid_pixels = np.sum(eroded_mask_for_fastsam > 0)
            
            # 只有当有效像素小于阈值时才跳过
            if valid_pixels < 100:
                print(f"    ⚠️  Mask too small ({valid_pixels} pixels), skipping Fast-SAM3D")
                boxes3d_list.append(np.full((8, 3), -1, dtype=np.float16))
                center_cam_list.append(np.array([-1, -1, -1]))
                dimensions_list.append([-1, -1, -1])
                R_cam_list.append(np.eye(3, dtype=np.float32))
                bbox2d_list.append(box.tolist())
                continue
            
            print(f"    🏗️  Running Fast-SAM3D with intrinsics...")
            output = reconstruct_3d(original_image, eroded_mask_for_fastsam, fastsam_inference, intrinsics=intrinsics)
            
            # 检查输出是否有效
            if output is None:
                raise ValueError("Fast-SAM3D returned None")
            if 'gs' not in output:
                raise ValueError(f"Fast-SAM3D output missing 'gs' key. Available keys: {list(output.keys())}")
            if 'glb' not in output or output['glb'] is None:
                raise ValueError("Fast-SAM3D output missing 'glb'")
        
            # 保存 Mesh
            mesh_filename = f"{label.replace(' ', '_')}_{i}.ply"
            mesh_path = os.path.join(mesh_dir, mesh_filename)
            save_gs_as_ply(output['gs'], mesh_path)
            
            glb_filename = f"{label.replace(' ', '_')}_{i}.glb"
            output['glb'].export(os.path.join(mesh_dir, glb_filename))
            
            # ============ 边界框提取 ============
            print(f"    📦 Extracting BBox...")
            bbox = extract_bbox_from_ply(mesh_path, prior)
            
            boxes3d_list.append(np.array(bbox['vertices'], dtype=np.float16))
            center_cam_list.append(np.array(bbox['center_cam']))
            dimensions_list.append(bbox['dimensions'])
            R_cam_list.append(np.array(bbox['R_cam']))
            bbox2d_list.append(box.tolist())
            
            print(f"    ✅ Center: [{bbox['center_cam'][0]:.3f}, {bbox['center_cam'][1]:.3f}, {bbox['center_cam'][2]:.3f}]")
            print(f"    ✅ Size: [{bbox['dimensions'][0]:.3f}, {bbox['dimensions'][1]:.3f}, {bbox['dimensions'][2]:.3f}]")
            
        except Exception as e:
            import traceback
            print(f"    ❌ Failed: {e}")
            print("    🔎 Traceback (for debugging):")
            print(traceback.format_exc())
            # 使用默认值
            boxes3d_list.append(np.full((8, 3), -1, dtype=np.float16))
            center_cam_list.append(np.array([-1, -1, -1]))
            dimensions_list.append([-1, -1, -1])
            R_cam_list.append(np.eye(3, dtype=np.float32))
            bbox2d_list.append(box.tolist())
    
    return {
        'image_id': image_id,
        'boxes3d': boxes3d_list,
        'center_cam': center_cam_list,
        'dimensions': dimensions_list,
        'R_cam': R_cam_list,
        'phrases': seg_results['labels'],
        'conf': list(seg_results['scores']) if isinstance(seg_results['scores'], (list, tuple)) else seg_results['scores'].tolist(),
        'boxes': bbox2d_list
    }


def process_images_on_gpu(gpu_id, image_data_subset, args, category_prior, output_suffix='', resume=False):
    """
    在单个 GPU 上处理一组图片
    
    Args:
        resume: 如果为 True，从 checkpoint 恢复，跳过已处理的图片
    """
    import torch
    
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    
    # 确定输出文件名
    output_file = os.path.join(args.get('output_dir'), f'info_3d{output_suffix}.pth')
    
    # 加载已处理的结果（断点续传）
    processed_results = {}
    start_idx = 0
    
    if resume and os.path.exists(output_file):
        try:
            processed_results = torch.load(output_file, map_location='cpu')
            processed_ids = set(processed_results.keys())
            # 找到第一个未处理的图片索引
            for i, img_info in enumerate(image_data_subset):
                if img_info['id'] not in processed_ids:
                    start_idx = i
                    break
            else:
                # 所有图片都已处理
                start_idx = len(image_data_subset)
            
            print(f"\n{'='*60}")
            print(f"🔄 GPU {gpu_id}: 恢复进度，已完成 {len(processed_results)} 张图片")
            print(f"   跳过前 {start_idx} 张图片，从第 {start_idx + 1} 张继续")
            print(f"{'='*60}")
        except Exception as e:
            print(f"⚠️ 加载 checkpoint 失败: {e}，从头开始")
            processed_results = {}
            start_idx = 0
    
    print(f"\n{'='*60}")
    print(f"🖥️  GPU {gpu_id}: Processing {len(image_data_subset)} images")
    if start_idx > 0:
        print(f"   (从第 {start_idx + 1} 张继续，还剩 {len(image_data_subset) - start_idx} 张)")
    print(f"{'='*60}")
    
    # 加载模型
    grounding_dino_model, sam_predictor = load_grounding_sam2_models(
        grounding_dino_checkpoint=args.get('grounding_dino_checkpoint'),
        sam_checkpoint=args.get('sam_checkpoint'),
        use_large_gdino=args.get('use_large_gdino', False)
    )
    
    fastsam_inference = load_fastsam3d(config_path=args.get('fastsam3d_config'))
    
    # 处理图片（从 start_idx 开始）
    all_results = processed_results.copy()
    # 用于追踪的子集（只显示未处理的部分）
    remaining_images = image_data_subset[start_idx:]
    # 定期保存 checkpoint 的间隔（每处理 50 张保存一次）
    checkpoint_interval = args.get('checkpoint_interval', 50)
    processed_since_checkpoint = 0
    
    for img_info in tqdm(remaining_images, desc=f"GPU {gpu_id}"):
        image_id = img_info['id']
        image_path = img_info['path']
        
        try:
            # 获取 intrinsics (K 矩阵) 并根据图像尺寸调整
            intrinsics = None
            if 'K' in img_info and img_info['K'] is not None:
                K = img_info['K']
                if isinstance(K, list) and len(K) == 3:
                    import numpy as np
                    intrinsics = np.array(K, dtype=np.float32)
                    # 调整 K 矩阵以适应 resize 后的图像尺寸
                    orig_width = img_info.get('width', None)
                    orig_height = img_info.get('height', None)
                    
                    # 读取实际图像尺寸
                    if os.path.exists(image_path):
                        import cv2
                        actual_img = cv2.imread(image_path)
                        if actual_img is not None:
                            actual_height, actual_width = actual_img.shape[:2]
                            if orig_width and orig_height:
                                scale_x = actual_width / orig_width
                                scale_y = actual_height / orig_height
                                # 调整 fx, cx (第0行) 和 fy, cy (第1行)
                                intrinsics[0, :] *= scale_x
                                intrinsics[1, :] *= scale_y
            
            result = process_single_image(
                image_path=image_path,
                text_prompt=args.get('text_prompt'),
                grounding_dino_model=grounding_dino_model,
                sam_predictor=sam_predictor,
                fastsam_inference=fastsam_inference,
                category_prior=category_prior,
                output_dir=args.get('output_dir'),
                box_threshold=args.get('box_threshold'),
                text_threshold=args.get('text_threshold'),
                intrinsics=intrinsics
            )
            
            if result is not None:
                result['image_id'] = image_id
                all_results[image_id] = result
                result['_metadata'] = {
                    'width': img_info.get('width'),
                    'height': img_info.get('height'),
                    'K': img_info.get('K'),
                    'dataset_id': img_info.get('dataset_id')
                }
            
            processed_since_checkpoint += 1
            
            # 定期保存 checkpoint
            if processed_since_checkpoint >= checkpoint_interval:
                torch.save(all_results, output_file)
                print(f"\n💾 GPU {gpu_id}: Checkpoint saved ({len(all_results)} images)")
                processed_since_checkpoint = 0
                
        except Exception as e:
            import traceback
            print(f"\n❌ Failed to process {image_path}: {e}")
            print("🔎 Full traceback:")
            print(traceback.format_exc())
    
    # 保存该 GPU 的结果
    torch.save(all_results, output_file)
    print(f"✅ GPU {gpu_id}: Saved {len(all_results)} results to {output_file}")
    
    return all_results


def main():
    parser = argparse.ArgumentParser(description="Generate Pseudo Labels with Grounding-SAM 2 + Fast-SAM3D")
    
    # 输入输出
    parser.add_argument('--json_file', type=str, default=None, help='JSON file containing image paths (for SUNRGBD)')
    parser.add_argument('--image_root', type=str, default='', help='Root directory for images (used with --json_file)')
    parser.add_argument('--image_dir', type=str, default=None, help='Directory containing images (alternative to --json_file)')
    parser.add_argument('--image_list', type=str, default=None, help='List of image files to process')
    parser.add_argument('--text_prompt', type=str, default=None, help='Text prompt for segmentation (if None, use SUNRGBD 38 classes for SUNRGBD dataset)')
    parser.add_argument('--output_dir', type=str, default='pseudo_label_gsam', help='Output directory')
    parser.add_argument('--max_images', type=int, default=None, help='Maximum number of images to process')
    
    # 模型路径
    parser.add_argument('--grounding_dino_checkpoint', type=str, default=None, help='Grounding DINO checkpoint')
    parser.add_argument('--sam_checkpoint', type=str, default=None, help='SAM checkpoint')
    parser.add_argument('--fastsam3d_config', type=str, default=None, help='Fast-SAM3D config path')
    parser.add_argument('--use_large_gdino', action='store_true',
                       help='使用大版本 Grounding DINO Swin-B (与原版一致，需下载 groundingdino_swinb_cogcoor.pth)')

    # 参数
    parser.add_argument('--gpu', type=str, default='0', help='GPU IDs (comma-separated, e.g., "0,1")')
    parser.add_argument('--box_threshold', type=float, default=0.2, help='Box detection threshold (原版默认 0.2)')
    parser.add_argument('--text_threshold', type=float, default=0.2, help='Text matching threshold (原版默认 0.2)')
    parser.add_argument('--dataset', type=str, default='SUNRGBD', help='Dataset name for category priors')
    parser.add_argument('--transform_to_coco', action='store_true', help='Convert to COCO format (Step 4)')
    parser.add_argument('--resume', action='store_true', help='Resume from checkpoint (skip already processed images)')
    parser.add_argument('--checkpoint_interval', type=int, default=50, help='Save checkpoint every N images')
    parser.add_argument('--parallel_per_image', action='store_true', help='Process multiple images in parallel on multiple GPUs')
    parser.add_argument('--only_bbox', action='store_true', help='Only run Step 3 (bbox extraction) from existing PLY files. Skips Step 1/2.')
    parser.add_argument('--mesh_dir_suffix', type=str, default='meshes', help='Suffix for mesh directory (default: meshes)')

    args = parser.parse_args()
    
    # 解析 GPU IDs
    gpu_ids = [int(x.strip()) for x in args.gpu.split(',')]
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu.replace(' ', '')
    
    # 加载类别先验 (使用与原版 OVM3D-Det 一致的先验)
    from cubercnn.generate_label.llm_generated_prior import KITTI, ARKitScenes, nuScenes
    dataset_prior_map = {'SUNRGBD': SUNRGBD_ORIGINAL_PRIORS, 'KITTI': KITTI, 'ARKitScenes': ARKitScenes, 'nuScenes': nuScenes}
    category_prior = dataset_prior_map.get(args.dataset, {})
    print(f"📊 Loaded {len(category_prior)} category priors for {args.dataset} (using original Omni3D priors)")
    
    # 固定 SUNRGBD 的 38 类别
    if args.dataset == 'SUNRGBD':
        args.text_prompt = SUNRGBD_38_CLASSES
        print(f"🎯 Using fixed SUNRGBD 38 classes for detection")
    elif args.text_prompt is None:
        args.text_prompt = 'chair table floor shelf cabinet'
        print(f"⚠️ No text prompt provided, using default: {args.text_prompt}")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 获取图像列表
    image_data = []
    
    if args.json_file and os.path.exists(args.json_file):
        print(f"📁 Loading images from JSON: {args.json_file}")
        with open(args.json_file, 'r') as f:
            data = json.load(f)
        
        for img_info in data.get('images', []):
            file_path = img_info['file_path']
            full_path = os.path.join(args.image_root, file_path)
            
            if os.path.exists(full_path):
                image_data.append({
                    'id': img_info.get('id', os.path.splitext(os.path.basename(file_path))[0]),
                    'path': full_path,
                    'width': img_info.get('width'),
                    'height': img_info.get('height'),
                    'K': img_info.get('K'),
                    'dataset_id': img_info.get('dataset_id')
                })
            else:
                print(f"⚠️ Image not found: {full_path}")
        
        print(f"✅ Found {len(image_data)} valid images from JSON")
        
    elif args.image_list and os.path.exists(args.image_list):
        with open(args.image_list, 'r') as f:
            image_files = [line.strip() for line in f if line.strip()]
        image_data = [{'id': f, 'path': os.path.join(args.image_dir or '', f)} for f in image_files]
        
    elif args.image_dir:
        image_files = [f for f in os.listdir(args.image_dir) 
                     if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        image_data = [{'id': f, 'path': os.path.join(args.image_dir, f)} for f in image_files]
    else:
        print("❌ Please specify --json_file or --image_dir")
        return
    
    # ========== Step 3/4: 只从现有 PLY 提取边界框 ==========
    if args.only_bbox:
        print("\n" + "="*60)
        print("🔄 Step 3/4: 从现有 PLY 文件提取边界框 (使用修复后的坐标系)")
        print("="*60)
        
        import torch
        import glob
        from cubercnn.generate_label.llm_generated_prior import SUNRGBD, KITTI, ARKitScenes, nuScenes
        dataset_prior_map = {'SUNRGBD': SUNRGBD, 'KITTI': KITTI, 'ARKitScenes': ARKitScenes, 'nuScenes': nuScenes}
        category_prior = dataset_prior_map.get(args.dataset, {})
        
        all_results = {}
        processed = 0
        skipped = 0
        
        for img_info in image_data:
            img_id = img_info['id']
            
            # 构建图像 ID (去掉扩展名)
            img_id_clean = os.path.splitext(img_id)[0]
            
            # 查找对应的 mesh 目录
            # 可能的目录格式: image_id/meshes 或 image_id-meshes
            mesh_base_dir = os.path.join(args.output_dir, img_id_clean)
            mesh_dir = os.path.join(mesh_base_dir, args.mesh_dir_suffix)
            
            if not os.path.exists(mesh_dir):
                # 尝试其他可能的路径格式
                for possible_dir in os.listdir(args.output_dir):
                    if possible_dir.startswith(img_id_clean) and os.path.isdir(os.path.join(args.output_dir, possible_dir)):
                        mesh_dir = os.path.join(args.output_dir, possible_dir, args.mesh_dir_suffix)
                        if os.path.exists(mesh_dir):
                            break
                else:
                    print(f"⚠️ Mesh directory not found for {img_id}")
                    skipped += 1
                    continue
            
            # 查找所有 PLY 文件
            ply_files = glob.glob(os.path.join(mesh_dir, '*.ply'))
            
            if not ply_files:
                print(f"⚠️ No PLY files in {mesh_dir}")
                skipped += 1
                continue
            
            boxes3d_list = []
            center_cam_list = []
            dimensions_list = []
            R_cam_list = []
            bbox2d_list = []
            labels = []
            
            for ply_path in ply_files:
                # 从文件名提取标签
                basename = os.path.basename(ply_path)
                label = os.path.splitext(basename)[0]
                # 提取类别名 (去掉下划线和数字后缀)
                parts = label.rsplit('_', 1)
                if len(parts) == 2 and parts[1].isdigit():
                    category_name = parts[0]
                else:
                    category_name = label
                
                labels.append(category_name)
                
                # 获取类别先验
                prior = category_prior.get(category_name, [0.5, 0.5, 0.5])
                
                try:
                    # 提取边界框 (现在会应用坐标系转换)
                    bbox = extract_bbox_from_ply(ply_path, prior)
                    
                    boxes3d_list.append(np.array(bbox['vertices'], dtype=np.float16))
                    center_cam_list.append(np.array(bbox['center_cam']))
                    dimensions_list.append(bbox['dimensions'])
                    R_cam_list.append(np.array(bbox['R_cam']))
                    
                    # 打印提取的边界框信息
                    cx, cy, cz = bbox['center_cam']
                    dx, dy, dz = bbox['dimensions']
                    print(f"  {category_name}: Center=[{cx:.3f}, {cy:.3f}, {cz:.3f}], Size=[{dx:.3f}, {dy:.3f}, {dz:.3f}]")
                    
                except Exception as e:
                    print(f"  ❌ Failed to extract bbox from {ply_path}: {e}")
                    boxes3d_list.append(np.full((8, 3), -1, dtype=np.float16))
                    center_cam_list.append(np.array([-1, -1, -1]))
                    dimensions_list.append([-1, -1, -1])
                    R_cam_list.append(np.eye(3, dtype=np.float32))
            
            all_results[img_id] = {
                'image_id': img_id,
                'boxes3d': boxes3d_list,
                'center_cam': center_cam_list,
                'dimensions': dimensions_list,
                'R_cam': R_cam_list,
                'phrases': labels,
                'conf': [1.0] * len(labels),
                'boxes': bbox2d_list if bbox2d_list else [[0, 0, 100, 100]] * len(labels)
            }
            processed += 1
            
            if processed % 100 == 0:
                print(f"  Processed {processed} images...")
        
        # 保存结果
        output_file = os.path.join(args.output_dir, 'info_3d.pth')
        torch.save(all_results, output_file)
        print(f"\n✅ Step 3/4 完成!")
        print(f"   处理: {processed} 张图片")
        print(f"   跳过: {skipped} 张图片")
        print(f"   结果保存到: {output_file}")
        
        # 如果需要转换 COCO 格式
        if args.transform_to_coco:
            from detectron2.data import DatasetCatalog, MetadataCatalog
            from cubercnn.util import category
            from detectron2.structures import BoxMode
            
            # 添加虚拟的 DatasetCatalog
            def get_gsa_dict():
                return all_results
            
            DatasetCatalog.register('gsam_pseudo', get_gsa_dict)
            MetadataCatalog.get('gsam_pseudo').set(thing_classes=list(category_prior.keys()))
            
            print("\n🔄 转换为 COCO 格式...")
            from detectron2.utils.file_io import PathManager
            import json as json_module
            
            output_coco = os.path.join(args.output_dir, 'pseudo_labels_coco.json')
            
            # 简单转换
            coco_format = {
                'images': [{'id': k, 'file_name': v['image_id']} for k, v in all_results.items()],
                'annotations': [],
                'categories': [{'id': i+1, 'name': name} for i, name in enumerate(category_prior.keys())]
            }
            
            ann_id = 1
            for img_id, result in all_results.items():
                for i, (box, phrase, center, dims) in enumerate(zip(
                    result['boxes'], result['phrases'], result['center_cam'], result['dimensions']
                )):
                    if center[0] == -1:  # 跳过无效的框
                        continue
                    coco_format['annotations'].append({
                        'id': ann_id,
                        'image_id': img_id,
                        'category_id': list(category_prior.keys()).index(phrase) + 1 if phrase in category_prior else 0,
                        'bbox': box,
                        'center_cam': center,
                        'dimensions': dims,
                        'area': dims[0] * dims[1] * dims[2],
                        'iscrowd': 0
                    })
                    ann_id += 1
            
            with open(output_coco, 'w') as f:
                json_module.dump(coco_format, f)
            print(f"✅ COCO 格式保存到: {output_coco}")
        
        return  # 结束 Step 3/4
        
    elif args.image_list and os.path.exists(args.image_list):
        with open(args.image_list, 'r') as f:
            image_files = [line.strip() for line in f if line.strip()]
        image_data = [{'id': f, 'path': os.path.join(args.image_dir or '', f)} for f in image_files]
        
    elif args.image_dir:
        image_files = [f for f in os.listdir(args.image_dir) 
                     if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        image_data = [{'id': f, 'path': os.path.join(args.image_dir, f)} for f in image_files]
    else:
        print("❌ Please specify --json_file or --image_dir")
        return
    
    # 限制处理数量
    if args.max_images:
        image_data = image_data[:args.max_images]
    
    print(f"\n🖥️  Using GPUs: {gpu_ids}")
    print(f"📊 Total images: {len(image_data)}")
    print(f"📊 Images per GPU: ~{len(image_data) // len(gpu_ids) + 1}")
    
    # 多卡处理
    if len(gpu_ids) > 1:
        import multiprocessing as mp
        
        # 将图片分配到各个 GPU
        chunks = []
        chunk_size = len(image_data) // len(gpu_ids)
        for i, gpu_id in enumerate(gpu_ids):
            start_idx = i * chunk_size
            end_idx = start_idx + chunk_size if i < len(gpu_ids) - 1 else len(image_data)
            # 只传递基本类型，避免 pickle 问题
            chunk_args = {
                'json_file': args.json_file,
                'image_root': args.image_root,
                'image_dir': args.image_dir,
                'image_list': args.image_list,
                'text_prompt': args.text_prompt,
                'output_dir': args.output_dir,
                'grounding_dino_checkpoint': args.grounding_dino_checkpoint,
                'sam_checkpoint': args.sam_checkpoint,
                'fastsam3d_config': args.fastsam3d_config,
                'use_large_gdino': args.use_large_gdino,
                'box_threshold': args.box_threshold,
                'text_threshold': args.text_threshold,
                'dataset': args.dataset,
                'transform_to_coco': args.transform_to_coco,
                'gpu': str(gpu_id),
                'resume': args.resume,
                'checkpoint_interval': args.checkpoint_interval,
            }
            chunks.append((gpu_id, image_data[start_idx:end_idx], chunk_args, category_prior, f'_gpu{gpu_id}', args.resume))
        
        print(f"\n{'='*60}")
        print("🚀 Starting Multi-GPU Processing...")
        print(f"{'='*60}")
        
        # 使用 multiprocessing 并行处理
        with mp.Pool(processes=len(gpu_ids)) as pool:
            results_list = pool.starmap(process_images_on_gpu, chunks)
        
        # 合并所有结果
        all_results = {}
        for results in results_list:
            all_results.update(results)
        
        print(f"\n{'='*60}")
        print("🔄 Merging Results from All GPUs...")
        print(f"{'='*60}")
        
        # 合并 pth 文件
        import torch
        merged_info = {}
        for gpu_id in gpu_ids:
            pth_path = os.path.join(args.output_dir, f'info_3d_gpu{gpu_id}.pth')
            if os.path.exists(pth_path):
                gpu_results = torch.load(pth_path, map_location='cpu')
                merged_info.update(gpu_results)
        
        # 保存合并后的结果
        torch.save(merged_info, os.path.join(args.output_dir, 'info_3d.pth'))
        print(f"✅ Merged {len(merged_info)} images from {len(gpu_ids)} GPUs")
        
        # 保存 JSON 格式
        json_results = {}
        for image_id, result in merged_info.items():
            json_results[image_id] = {
                'boxes': result['boxes'],
                'phrases': result['phrases'],
                'conf': result['conf'],
                'objects': []
            }
            for i in range(len(result['phrases'])):
                json_results[image_id]['objects'].append({
                    'label': result['phrases'][i],
                    'score': result['conf'][i],
                    'bbox2d': result['boxes'][i],
                    'center_cam': result['center_cam'][i],
                    'dimensions': result['dimensions'][i],
                    'vertices': result['boxes3d'][i].tolist() if hasattr(result['boxes3d'][i], 'tolist') else result['boxes3d'][i]
                })
        
        with open(os.path.join(args.output_dir, 'results.json'), 'w') as f:
            json.dump(json_results, f, indent=2, default=lambda x: x.tolist() if hasattr(x, 'tolist') else list(x))
        
        print(f"\n✅ Done! Processed {len(all_results)} images")
        print(f"📁 Results saved to: {args.output_dir}/")
        
    else:
        # 单卡处理 (原始逻辑)
        gpu_id = gpu_ids[0]
        
        # 将 Namespace 转换为字典以便使用 .get() 方法
        args_dict = vars(args)
        
        all_results = process_images_on_gpu(
            gpu_id, image_data, args_dict, category_prior, output_suffix='', resume=args.resume
        )
        
        # 转换为 JSON 格式
        json_results = {}
        for image_id, result in all_results.items():
            metadata = result.pop('_metadata', {})
            json_results[image_id] = {
                'boxes': result['boxes'],
                'phrases': result['phrases'],
                'conf': result['conf'],
                'objects': []
            }
            for i in range(len(result['phrases'])):
                json_results[image_id]['objects'].append({
                    'label': result['phrases'][i],
                    'score': result['conf'][i],
                    'bbox2d': result['boxes'][i],
                    'center_cam': result['center_cam'][i],
                    'dimensions': result['dimensions'][i],
                    'vertices': result['boxes3d'][i].tolist() if hasattr(result['boxes3d'][i], 'tolist') else result['boxes3d'][i]
                })
            
            # 恢复 metadata
            result['_metadata'] = metadata
        
        with open(os.path.join(args.output_dir, 'results.json'), 'w') as f:
            json.dump(json_results, f, indent=2, default=lambda x: x.tolist() if hasattr(x, 'tolist') else list(x))
        
        print(f"\n✅ Done! Processed {len(all_results)} images")
        print(f"📁 Results saved to: {args.output_dir}/")
    
    # ============ Step 4: 转换为 COCO 格式 ============
    if args.transform_to_coco:
        print("\n" + "=" * 60)
        print("[Step 4/4] Converting to COCO Format")
        print("=" * 60)
        
        transform_to_coco_format(
            all_results=all_results,
            output_dir=args.output_dir,
            dataset_name=args.dataset
        )


def transform_to_coco_format(all_results: dict, output_dir: str, dataset_name: str):
    """
    Step 4: 将伪标签转换为 COCO 格式
    
    与 tools/transform_to_coco.py 逻辑一致
    """
    import torch
    
    # 类别映射
    thing_classes_dict = {
        "SUNRGBD": {
            'chair': 18, 'door': 31, 'table': 37, 'shelves': 26, 'kitchen pan': 51,
            'bin': 52, 'counter': 38, 'cabinet': 29, 'stove': 53, 'sink': 28,
            'books': 14, 'refrigerator': 49, 'microwave': 54, 'bottle': 15,
            'plates': 55, 'bowl': 56, 'oven': 57, 'vase': 58, 'faucet': 59,
            'towel': 22, 'tissues': 60, 'machine': 61, 'printer': 62, 'desk': 33,
            'monitor': 63, 'podium': 64, 'bookcase': 35, 'dresser': 41, 'cart': 65,
            'projector': 66, 'electronics': 67, 'computer': 68, 'box': 34,
            'picture': 36, 'laptop': 20, 'pillow': 42, 'bed': 39,
            'air conditioner': 69, 'lamp': 25, 'night stand': 40, 'board': 50,
            'sofa': 43, 'coffee maker': 71, 'toaster': 72, 'potted plant': 73,
            'stationery': 48, 'painting': 74, 'bag': 75, 'tray': 76, 'cup': 19,
            'drawers': 70, 'keyboard': 77, 'shoes': 21, 'bicycle': 11,
            'blanket': 78, 'television': 44, 'rack': 79, 'mirror': 27,
            'clothes': 47, 'phone': 80, 'mouse': 81, 'person': 7,
            'fire extinguisher': 82, 'toys': 83, 'ladder': 84, 'fan': 85,
            'toilet': 32, 'bathtub': 30, 'glass': 86, 'clock': 87,
            'toilet paper': 88, 'closet': 89, 'curtain': 46, 'window': 24,
            'fume hood': 90, 'utensils': 91, 'floor mat': 45, 'soundsystem': 92,
            'fire place': 93, 'shower curtain': 94, 'blinds': 23,
            'remote': 95, 'pen': 96
        }
    }
    
    thing_classes = thing_classes_dict.get(dataset_name, {})
    
    # 创建 COCO 格式的 annotations
    annotations = []
    num = 1
    dataset_id = 1  # 可以根据数据集调整
    
    for image_id, result in all_results.items():
        phrases = result['phrases']
        scores = result['conf']
        boxes = result['boxes']
        boxes3d = result['boxes3d']
        center_cam = result['center_cam']
        dimensions = result['dimensions']
        R_cam = result['R_cam']
        
        for j in range(len(phrases)):
            category_name = phrases[j]
            
            # 跳过未知类别
            if category_name not in thing_classes:
                print(f"⚠️ Unknown category: {category_name}, skipping")
                continue
            
            # 处理嵌套列表（如 [[0.96]] 或 [0.96]）
            def extract_float(val):
                if isinstance(val, list):
                    if len(val) == 1:
                        val = val[0]
                    if isinstance(val, list):
                        return float(val[0]) if val else 0.0
                return float(val)
            
            score_val = extract_float(scores[j])
            
            obj = {
                'id': dataset_id * 10000000 + num,
                'image_id': image_id,
                'dataset_id': dataset_id,
                'category_name': category_name,
                'category_id': thing_classes[category_name],
                'valid3D': True,
                'bbox2D_tight': boxes[j].tolist() if hasattr(boxes[j], 'tolist') else boxes[j],  # 使用 2D 检测框
                'bbox2D_trunc': boxes[j].tolist() if hasattr(boxes[j], 'tolist') else boxes[j],  # 使用 2D 检测框
                'bbox2D_proj': boxes[j].tolist() if hasattr(boxes[j], 'tolist') else boxes[j],
                'bbox3D_cam': boxes3d[j].tolist() if hasattr(boxes3d[j], 'tolist') else boxes3d[j],
                'center_cam': center_cam[j].tolist() if hasattr(center_cam[j], 'tolist') else center_cam[j],
                'dimensions': [float(x) for x in dimensions[j]],
                'R_cam': R_cam[j].tolist() if hasattr(R_cam[j], 'tolist') else R_cam[j],
                'behind_camera': False,
                'visibility': 1.0,
                'truncation': 0.0,
                'segmentation_pts': -1,
                'lidar_pts': -1,
                'depth_error': -1,
                'score': score_val
            }
            
            annotations.append(obj)
            num += 1
    
    # 保存 COCO 格式
    coco_output_dir = os.path.join(output_dir, 'coco_format')
    os.makedirs(coco_output_dir, exist_ok=True)
    
    coco_data = {
        'annotations': annotations,
        'dataset_name': dataset_name,
        'num_images': len(all_results),
        'num_objects': len(annotations)
    }
    
    coco_path = os.path.join(coco_output_dir, f'{dataset_name}_pseudo_labels.json')
    with open(coco_path, 'w') as f:
        json.dump(coco_data, f, indent=2, default=lambda x: x.tolist() if hasattr(x, 'tolist') else list(x))
    
    print(f"\n✅ COCO format saved to: {coco_path}")
    print(f"   - Images: {len(all_results)}")
    print(f"   - Objects: {len(annotations)}")


if __name__ == "__main__":
    main()
