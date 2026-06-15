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

# Training-safe 3D gate. A low projected IoU means the reconstructed cuboid is
# not in the same camera-space geometry as the 2D evidence, so it must remain
# 2D-only instead of becoming a noisy valid3D label.
PSEUDO3D_MIN_PROJECTION_IOU = 0.15
PSEUDO3D_MAX_CENTER_NORM = 0.50
PSEUDO3D_MIN_AREA_RATIO = 0.20
PSEUDO3D_MAX_AREA_RATIO = 5.00
PSEUDO3D_MIN_POINT_SUPPORT = 0.25
RENDER_ALIGN_SCALE = 0.5
RENDER_ALIGN_MAX_FACES = 4000
RENDER_ALIGN_MIN_SILHOUETTE_IOU = 0.10
RENDER_ALIGN_MIN_BBOX_IOU = 0.15
RENDER_ALIGN_MIN_DEPTH_PIXELS = 25
RENDER_ALIGN_MAX_REL_DEPTH_ERROR = 0.45
RENDER_ALIGN_STRICT_MIN_PROJECTION_IOU = 0.35
RENDER_ALIGN_STRICT_MIN_SILHOUETTE_IOU = 0.08
RENDER_ALIGN_STRICT_MAX_REL_DEPTH_ERROR = 0.35
RENDER_ALIGN_REQUIRE_STRICT = False
RENDER_PRIOR_MIN_RATIO = 0.15
RENDER_PRIOR_MAX_RATIO = 4.0
RENDER_USE_ALIGNED_MESH_OBB = True
RENDER_SILHOUETTE_REFINE = True
RENDER_SILHOUETTE_AREA_SCALE_MIN = 0.60
RENDER_SILHOUETTE_AREA_SCALE_MAX = 2.00
RENDER_SILHOUETTE_MIN_RENDER_PIXELS = 8
RENDER_SILHOUETTE_MIN_SHIFT_PIXELS = 1.0
RENDER_SILHOUETTE_MAX_SHIFT_NORM = 0.15
DEPTH_EDGE_REL_THRESH = 0.08
DEPTH_EDGE_MIN_KEEP_RATIO = 0.35
RENDER_PNP_MIN_MATCHES = 12
RENDER_PNP_MIN_INLIERS = 8
RENDER_PNP_MIN_INLIER_RATIO = 0.35
RENDER_PNP_MAX_CONTOUR_DIST = 8.0
RENDER_PNP_MAX_REPROJ_ERROR = 8.0
RENDER_PNP_MAX_POINTS = 220
RENDER_PNP_MAX_CENTER_SHIFT_NORM = 0.75
RENDER_PNP_MIN_UPRIGHT_DOT = 0.60
RENDER_PNP_MATCHER = 'auto'
MAST3R_ROOT = '/data/ZhaoX/LabelAny3D-main/LabelAny3D-main/external/mast3r'
MAST3R_MODEL_NAME = 'naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric'
MAST3R_IMAGE_SIZE = 512
MAST3R_SUBSAMPLE = 8
MAST3R_CROP_PAD = 1.25
RENDER_YAW_OFFSETS_DEG = [-30.0, -15.0, 0.0, 15.0, 30.0]
RENDER_SCALE_MULTIPLIERS = [0.75, 1.0, 1.25]
FASTSAM3D_MIN_MASK_PIXELS = 100
FASTSAM3D_MIN_MASK_AREA_RATIO = 0.001

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
import gc
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

    # 提取 pointmap_scale 和 pointmap_shift（用于反归一化）
    # 这些参数用于将归一化的坐标转换回物理单位
    if output is not None:
        # 检查是否有 scale/shift 参数
        output['scene_scale'] = output.get('scale', None)
        output['scene_shift'] = output.get('shift', None)
        
        # 检查是否有原始 pointmap（物理坐标）
        if 'pointmap' in output:
            print(f"   📏 Pointmap shape: {output['pointmap'].shape}")
            
            # pointmap 中间可能包含物理深度信息
            # 尝试获取物理深度
            if hasattr(output['pointmap'], 'cpu'):
                pm = output['pointmap'].cpu().numpy()
            else:
                pm = output['pointmap']
            
            if isinstance(pm, np.ndarray) and pm.ndim >= 3:
                # 假设格式是 [H, W, 3] 或 [B, H, W, 3]
                if pm.shape[-1] == 3:
                    # 计算物理深度统计
                    valid_mask = pm[..., 2] > 0
                    if valid_mask.any():
                        depth_stats = {
                            'mean': float(pm[..., 2][valid_mask].mean()),
                            'min': float(pm[..., 2][valid_mask].min()),
                            'max': float(pm[..., 2][valid_mask].max())
                        }
                        output['depth_stats'] = depth_stats
                        print(f"   📏 Physical depth stats: mean={depth_stats['mean']:.3f}m, range=[{depth_stats['min']:.3f}, {depth_stats['max']:.3f}]")

    return output


def save_gs_as_ply(gs_model, output_path: str, denormalize: bool = False, 
                   scale: torch.Tensor = None, shift: torch.Tensor = None):
    """
    保存高斯溅射为 PLY
    
    Args:
        gs_model: 高斯溅射模型
        output_path: 输出路径
        denormalize: 是否反归一化回物理单位
        scale: 点云的 scale 参数（用于反归一化）
        shift: 点云的 shift 参数（用于反归一化）
    """
    from plyfile import PlyData, PlyElement
    
    xyz = gs_model._xyz.detach().cpu().numpy()
    f_dc = gs_model._features_dc.detach().contiguous().cpu().numpy()
    
    # 如果需要反归一化
    if denormalize and scale is not None and shift is not None:
        # 将 scale 和 shift 转换到正确设备
        if isinstance(scale, torch.Tensor):
            scale_np = scale.detach().cpu().numpy()
        else:
            scale_np = np.array(scale)
        if isinstance(shift, torch.Tensor):
            shift_np = shift.detach().cpu().numpy()
        else:
            shift_np = np.array(shift)
        
        # 反归一化公式: point_physical = point_normalized * scale + shift
        # 需要先计算逆变换
        xyz = xyz * scale_np + shift_np
    
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


# ============================================================
# 原版 Cubercnn 工具函数（用于 ray tracing 优化）
# ============================================================

def rotate_y_cubercnn(angle):
    """旋转 Y 轴的旋转矩阵"""
    rotmat = np.zeros((3, 3))
    rotmat[1, 1] = 1
    cosval = np.cos(angle)
    sinval = np.sin(angle)
    rotmat[0, 0] = cosval
    rotmat[0, 2] = sinval
    rotmat[2, 0] = -sinval
    rotmat[2, 2] = cosval
    return rotmat


def calc_inside_ratio(pc, x_min, x_max, z_min, z_max):
    """计算点云在 bbox 内的比例"""
    inside = (pc[0, :] > x_min) & (pc[0, :] < x_max) & \
             (pc[2, :] > z_min) & (pc[2, :] < z_max)
    inside_ratio = np.sum(inside) / pc.shape[1]
    return inside_ratio


def calc_dis_ray_tracing_cubercnn(wl, Ry, points, bev_box_center):
    """计算 ray tracing 损失"""
    import torch
    init_theta = torch.atan(wl[0] / wl[1])
    length = torch.sqrt(wl[0] ** 2 + wl[1] ** 2) / 2
    
    corners = [
        (length * torch.cos(init_theta + Ry) + bev_box_center[0],
         length * torch.sin(init_theta + Ry) + bev_box_center[1]),
        (length * torch.cos(np.pi - init_theta + Ry) + bev_box_center[0],
         length * torch.sin(np.pi - init_theta + Ry) + bev_box_center[1]),
        (length * torch.cos(np.pi + init_theta + Ry) + bev_box_center[0],
         length * torch.sin(np.pi + init_theta + Ry) + bev_box_center[1]),
        (length * torch.cos(-init_theta + Ry) + bev_box_center[0],
         length * torch.sin(-init_theta + Ry) + bev_box_center[1])
    ]
    
    if Ry == np.pi/2:
        Ry -= 1e-4
    if Ry == 0:
        Ry += 1e-4
        
    k1, k2 = torch.tan(Ry), torch.tan(Ry + np.pi / 2)
    b11 = corners[0][1] - k1 * corners[0][0]
    b12 = corners[2][1] - k1 * corners[2][0]
    b21 = corners[0][1] - k2 * corners[0][0]
    b22 = corners[2][1] - k2 * corners[2][0]

    line0 = [k1, -1, b11]
    line1 = [k2, -1, b22]
    line2 = [k1, -1, b12]
    line3 = [k2, -1, b21]

    points = points.clone()
    points[points[:, 0] == 0, 0] = 1e-4

    slope_x = points[:, 1] / points[:, 0]
    intersect0 = torch.stack([line0[2] / (slope_x - line0[0]),
                              line0[2]*slope_x / (slope_x - line0[0])], dim=1)
    intersect1 = torch.stack([line1[2] / (slope_x - line1[0]),
                              line1[2]*slope_x / (slope_x - line1[0])], dim=1)
    intersect2 = torch.stack([line2[2] / (slope_x - line2[0]),
                              line2[2]*slope_x / (slope_x - line2[0])], dim=1)
    intersect3 = torch.stack([line3[2] / (slope_x - line3[0]),
                              line3[2]*slope_x / (slope_x - line3[0])], dim=1)

    dis0 = torch.abs(intersect0[:, 0] - points[:, 0]) + torch.abs(intersect0[:, 1] - points[:, 1])
    dis1 = torch.abs(intersect1[:, 0] - points[:, 0]) + torch.abs(intersect1[:, 1] - points[:, 1])
    dis2 = torch.abs(intersect2[:, 0] - points[:, 0]) + torch.abs(intersect2[:, 1] - points[:, 1])
    dis3 = torch.abs(intersect3[:, 0] - points[:, 0]) + torch.abs(intersect3[:, 1] - points[:, 1])

    dis_inter2center0 = torch.sqrt(intersect0[:, 0]**2 + intersect0[:, 1]**2)
    dis_inter2center1 = torch.sqrt(intersect1[:, 0]**2 + intersect1[:, 1]**2)
    dis_inter2center2 = torch.sqrt(intersect2[:, 0]**2 + intersect2[:, 1]**2)
    dis_inter2center3 = torch.sqrt(intersect3[:, 0]**2 + intersect3[:, 1]**2)

    intersect0 = torch.round(intersect0*1e4)
    intersect1 = torch.round(intersect1*1e4)
    intersect2 = torch.round(intersect2*1e4)
    intersect3 = torch.round(intersect3*1e4)

    dis0_in_box_edge = ((intersect0[:, 0] > torch.round(min(corners[0][0], corners[1][0])*1e4)) &
                        (intersect0[:, 0] < torch.round(max(corners[0][0], corners[1][0])*1e4))) | \
                       ((intersect0[:, 1] > torch.round(min(corners[0][1], corners[1][1])*1e4)) &
                        (intersect0[:, 1] < torch.round(max(corners[0][1], corners[1][1])*1e4)))
    dis1_in_box_edge = ((intersect1[:, 0] > torch.round(min(corners[1][0], corners[2][0])*1e4)) &
                        (intersect1[:, 0] < torch.round(max(corners[1][0], corners[2][0])*1e4))) | \
                       ((intersect1[:, 1] > torch.round(min(corners[1][1], corners[2][1])*1e4)) &
                        (intersect1[:, 1] < torch.round(max(corners[1][1], corners[2][1])*1e4)))
    dis2_in_box_edge = ((intersect2[:, 0] > torch.round(min(corners[2][0], corners[3][0])*1e4)) &
                        (intersect2[:, 0] < torch.round(max(corners[2][0], corners[3][0])*1e4))) | \
                       ((intersect2[:, 1] > torch.round(min(corners[2][1], corners[3][1])*1e4)) &
                        (intersect2[:, 1] < torch.round(max(corners[2][1], corners[3][1])*1e4)))
    dis3_in_box_edge = ((intersect3[:, 0] > torch.round(min(corners[3][0], corners[0][0])*1e4)) &
                        (intersect3[:, 0] < torch.round(max(corners[3][0], corners[0][0])*1e4))) | \
                       ((intersect3[:, 1] > torch.round(min(corners[3][1], corners[0][1])*1e4)) &
                        (intersect3[:, 1] < torch.round(max(corners[3][1], corners[0][1])*1e4)))

    dis_in_mul = torch.stack([dis0_in_box_edge, dis1_in_box_edge,
                              dis2_in_box_edge, dis3_in_box_edge], dim=1)
    dis_inter2cen = torch.stack([dis_inter2center0, dis_inter2center1,
                                 dis_inter2center2, dis_inter2center3], dim=1)
    dis_all = torch.stack([dis0, dis1, dis2, dis3], dim=1)

    dis_in = (torch.sum(dis_in_mul, dim=1) == 2).type(torch.bool)
    if torch.sum(dis_in.int()) < 3:
        return 0

    dis_in = dis_in.squeeze(0)
    dis_in_mul = dis_in_mul[:,:,dis_in]
    dis_inter2cen = dis_inter2cen[:,:,dis_in]
    dis_all = dis_all[:,:,dis_in]

    z_buffer_ind = torch.argmin(dis_inter2cen[dis_in_mul].view(2, -1), dim=0)
    z_buffer_ind_gather = torch.stack([~z_buffer_ind.byte(), z_buffer_ind.byte()],
                                      dim=1).type(torch.bool)
    dis = (dis_all[dis_in_mul].view(2, -1).permute(1,0))[z_buffer_ind_gather]

    dis_mean = torch.mean(dis)
    return dis_mean.item()


def generate_possible_bboxs(cx, cz, dx, dz, w, l):
    """生成所有可能的 bounding box proposals"""
    import math
    
    def scale_rectangle(rectangle, length_left, length_right, origin_vertex):
        scaled_rectangles = [None] * 4
        scaled_rectangles[origin_vertex] = rectangle[origin_vertex]
        origin_x, origin_y = rectangle[origin_vertex]

        x, y = rectangle[(origin_vertex-1)%4]
        relative_x, relative_y = x - origin_x, y - origin_y
        scale_left = length_left / np.linalg.norm([relative_x, relative_y])
        scaled_rectangles[(origin_vertex-1)%4] = (origin_x + relative_x * scale_left,
                                                  origin_y + relative_y * scale_left)

        x, y = rectangle[(origin_vertex+1)%4]
        relative_x, relative_y = x - origin_x, y - origin_y
        scale_right = length_right / np.linalg.norm([relative_x, relative_y])
        scaled_rectangles[(origin_vertex+1)%4] = (origin_x + relative_x * scale_right,
                                                  origin_y + relative_y * scale_right)

        x, y = rectangle[(origin_vertex+2)%4]
        x_scaled = origin_x + relative_x * scale_right + (x - origin_x) / np.linalg.norm([relative_x, relative_y]) * scale_left
        y_scaled = origin_y + relative_y * scale_right + (y - origin_y) / np.linalg.norm([relative_x, relative_y]) * scale_left
        scaled_rectangles[(origin_vertex+2)%4] = (x_scaled, y_scaled)
        return scaled_rectangles
    
    init_theta, length = math.atan(dz / dx), math.sqrt(dx ** 2 + dz ** 2) / 2
    
    corners = [
        (length * math.cos(init_theta) + cx, length * math.sin(init_theta) + cz),
        (length * math.cos(np.pi - init_theta) + cx, length * math.sin(np.pi - init_theta) + cz),
        (length * math.cos(np.pi + init_theta) + cx, length * math.sin(np.pi + init_theta) + cz),
        (length * math.cos(-init_theta) + cx, length * math.sin(-init_theta) + cz),
    ]
    
    scaled_rectangles = []
    for i in range(4):
        scaled_rectangles.append(scale_rectangle(corners, w, l, i))
        scaled_rectangles.append(scale_rectangle(corners, l, w, i))
    
    transform_scaled_rectangles = []
    for rect in scaled_rectangles:
        corners_arr = np.array(rect)
        x_min, x_max = corners_arr[:,0].min(), corners_arr[:,0].max()
        z_min, z_max = corners_arr[:,1].min(), corners_arr[:,1].max()
        transform_scaled_rectangles.append([x_min, x_max, z_min, z_max])
    return transform_scaled_rectangles


def convert_box_vertices(cx, cy, cz, l, w, h, yaw):
    """将 bbox 参数转换为 8 个顶点坐标"""
    import math
    
    local_corners = np.array([
        [-l / 2, -w / 2, -h / 2],
        [l / 2, -w / 2, -h / 2],
        [l / 2, w / 2, -h / 2],
        [-l / 2, w / 2, -h / 2],
        [-l / 2, -w / 2, h / 2],
        [l / 2, -w / 2, h / 2],
        [l / 2, w / 2, h / 2],
        [-l / 2, w / 2, h / 2]
    ])
    
    rotation_matrix = np.array([
        [math.cos(yaw), 0, math.sin(yaw)],
        [0, 1, 0],
        [-math.sin(yaw), 0, math.cos(yaw)]
    ])
    
    rotated_corners = np.dot(local_corners, rotation_matrix.T)
    global_corners = rotated_corners + np.array([cx, cy, cz])
    return global_corners


def convert_box_vertices_pose(center, dims_xyz, R_cam):
    """Convert an object-frame cuboid to camera-space vertices with full pose."""
    center = np.asarray(center, dtype=np.float64).reshape(3)
    dims_xyz = np.maximum(np.asarray(dims_xyz, dtype=np.float64).reshape(3), 1e-6)
    R_cam = np.asarray(R_cam, dtype=np.float64).reshape(3, 3)
    dx, dy, dz = [float(v) for v in dims_xyz]
    local_corners = np.array([
        [-dx / 2, -dy / 2, -dz / 2],
        [dx / 2, -dy / 2, -dz / 2],
        [dx / 2, dy / 2, -dz / 2],
        [-dx / 2, dy / 2, -dz / 2],
        [-dx / 2, -dy / 2, dz / 2],
        [dx / 2, -dy / 2, dz / 2],
        [dx / 2, dy / 2, dz / 2],
        [-dx / 2, dy / 2, dz / 2],
    ], dtype=np.float64)
    return local_corners @ R_cam.T + center[None, :]


def yaw_from_rotation_matrix(R_cam):
    R_cam = np.asarray(R_cam, dtype=np.float64).reshape(3, 3)
    yaw = np.arctan2(R_cam[0, 2] - R_cam[2, 0], R_cam[0, 0] + R_cam[2, 2])
    return normalize_yaw(yaw)


# ============================================================
# 新流程：伪点云 + 3D重建 融合
# Center: 来自伪点云质心 (原版 process_indoor.py 流程)
# Size/Yaw: 来自 3D mesh (FastSAM3D)
# ============================================================

# ============================================================
# 独立 UniDepth 推理（完全不经过 Fast-SAM3D）
# ============================================================

_UNIDEPTH_MODEL = None
_UNIDEPTH_MODEL_PATH = "lpiccinelli/unidepth-v1-vitl14"

def load_unidepth_model():
    """加载 UniDepth 模型（仅加载一次）"""
    global _UNIDEPTH_MODEL
    if _UNIDEPTH_MODEL is None:
        import sys
        unidepth_path = '/data/ZhaoX/OVM3D-Dett/third_party/UniDepth'
        if unidepth_path not in sys.path:
            sys.path.insert(0, unidepth_path)
        from unidepth.models import UniDepthV1
        print(f"    📦 Loading independent UniDepth model...")
        _UNIDEPTH_MODEL = UniDepthV1.from_pretrained(_UNIDEPTH_MODEL_PATH)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        _UNIDEPTH_MODEL = _UNIDEPTH_MODEL.to(device)
        _UNIDEPTH_MODEL.eval()
        print(f"    ✅ Independent UniDepth model loaded!")
    return _UNIDEPTH_MODEL


def run_independent_unidepth(image_rgb, intrinsics):
    """
    独立调用 UniDepth，完全不经过 Fast-SAM3D
    
    与原版 run_unidepth.py 流程一致:
        1. 加载 UniDepthV1.from_pretrained("lpiccinelli/unidepth-v1-vitl14")
        2. model.infer(rgb, intrinsics)
        3. 返回 depth (单位：米)
    
    Args:
        image_rgb: RGB 图像 (H, W, 3) numpy array, 值范围 [0, 255]
        intrinsics: 相机内参 (3, 3)
    
    Returns:
        depth: 深度图 (H, W), 单位：米
    """
    model = load_unidepth_model()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 确保输入是 numpy 数组
    if isinstance(image_rgb, torch.Tensor):
        image_rgb = image_rgb.cpu().numpy()
    
    # 确保是 (H, W, 3) 格式
    if image_rgb.ndim == 3 and image_rgb.shape[0] == 3:
        image_rgb = np.transpose(image_rgb, (1, 2, 0))  # (3, H, W) -> (H, W, 3)
    
    # 确保值范围是 [0, 255] 且类型是 uint8
    if image_rgb.dtype != np.uint8:
        image_rgb = (image_rgb * 255).astype(np.uint8) if image_rgb.max() <= 1.0 else image_rgb.astype(np.uint8)
    
    # 转换为 torch tensor: (H, W, 3) -> (1, 3, H, W)
    # 使用 torch.tensor 而不是 torch.from_numpy，避免 contiguous 问题
    image_tensor = torch.tensor(image_rgb.copy(), dtype=torch.float32).permute(2, 0, 1).unsqueeze(0) / 255.0
    image_tensor = image_tensor.to(device)
    
    # 处理内参格式 - UniDepth 需要 torch.Tensor
    if isinstance(intrinsics, torch.Tensor):
        K = intrinsics.detach().float()
    else:
        K = torch.from_numpy(np.array(intrinsics).reshape(3, 3).astype(np.float32))
    K = K.to(device)
    
    # 推理 - UniDepth.infer 会自动添加 batch 维度
    with torch.no_grad():
        output = model.infer(image_tensor.squeeze(0), K)  # (1, 3, H, W) -> (3, H, W)
    
    # 提取深度图
    depth = output["depth"]
    if isinstance(depth, torch.Tensor):
        depth = depth.squeeze(0).squeeze(0).cpu().numpy()  # (1, 1, H, W) -> (H, W)
    else:
        depth = depth.squeeze(0).squeeze(0)
    
    return depth  # 单位：米


def create_uv_depth(depth, mask=None):
    """与原版 process_indoor.py 一致的 UV-Depth 点云生成"""
    if mask is not None:
        depth = depth * mask
    x, y = np.meshgrid(
        np.linspace(0, depth.shape[1] - 1, depth.shape[1]),
        np.linspace(0, depth.shape[0] - 1, depth.shape[0])
    )
    uv_depth = np.stack((x, y, depth), axis=-1)
    uv_depth = uv_depth.reshape(-1, 3)
    return uv_depth[uv_depth[:, 2] != 0]


def project_image_to_cam(uv_depth, K):
    """
    与原版 util.py 一致的投影函数
    Input: nx3 first two channels are uv, 3rd channel is depth in camera coord.
    Output: nx3 points in camera coord.
    """
    c_u = K[0, 2]
    c_v = K[1, 2]
    f_u = K[0, 0]
    f_v = K[1, 1]

    n = uv_depth.shape[0]
    x = ((uv_depth[:, 0] - c_u) * uv_depth[:, 2]) / f_u
    y = ((uv_depth[:, 1] - c_v) * uv_depth[:, 2]) / f_v
    pts_3d_rect = np.zeros((n, 3))
    pts_3d_rect[:, 0] = x
    pts_3d_rect[:, 1] = y
    pts_3d_rect[:, 2] = uv_depth[:, 2]
    return pts_3d_rect


def remove_depth_edges_from_mask(depth_map, mask, rel_thresh=None, min_keep_ratio=None):
    """Remove depth discontinuities inside a mask, with fallback handled by caller."""
    if rel_thresh is None:
        rel_thresh = DEPTH_EDGE_REL_THRESH
    if min_keep_ratio is None:
        min_keep_ratio = DEPTH_EDGE_MIN_KEEP_RATIO

    depth = np.asarray(depth_map).astype(np.float32)
    mask = np.asarray(mask).astype(bool)
    valid = mask & np.isfinite(depth) & (depth > 0.05)
    if int(valid.sum()) < 20:
        return mask.astype(np.float32)

    log_depth = np.zeros_like(depth, dtype=np.float32)
    log_depth[valid] = np.log(np.maximum(depth[valid], 1e-3))
    grad_x = cv2.Sobel(log_depth, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(log_depth, cv2.CV_32F, 0, 1, ksize=3)
    grad = np.sqrt(grad_x * grad_x + grad_y * grad_y)

    edge = (grad > float(rel_thresh)) & valid
    edge = cv2.dilate(edge.astype(np.uint8), np.ones((3, 3), dtype=np.uint8), iterations=1).astype(bool)
    cleaned = valid & (~edge)

    keep_ratio = float(cleaned.sum() / max(float(valid.sum()), 1.0))
    if int(cleaned.sum()) < 20 or keep_ratio < float(min_keep_ratio):
        return mask.astype(np.float32)
    return cleaned.astype(np.float32)


def extract_pointcloud_from_depth(depth_map, mask, intrinsics):
    """
    从深度图和掩码提取点云质心（原版流程）
    
    与 process_indoor.py 中的流程完全一致:
        depth * mask → create_uv_depth → project_image_to_cam → 质心
    
    Args:
        depth_map: 深度图 (H, W) 或 (1, H, W)，单位：米
        mask: 二值掩码 (H, W), 值在 [0, 1]
        intrinsics: 相机内参 (3, 3)
    
    Returns:
        pseudo_lidar: 相机坐标系下的伪点云 (N, 3)
        centroid: 点云质心 (3,)
    """
    # 处理深度图格式
    if isinstance(depth_map, torch.Tensor):
        depth_map = depth_map.detach().cpu().numpy()
    
    # 处理 batch 维度
    if depth_map.ndim == 3:
        depth_map = depth_map[0]  # 取第一张
    if depth_map.ndim > 2:
        depth_map = depth_map.squeeze()
    
    H, W = depth_map.shape[:2]
    
    # 确保掩码形状匹配
    if isinstance(mask, torch.Tensor):
        mask = mask.detach().cpu().numpy()
    
    if mask.shape != (H, W):
        import cv2
        mask_resized = cv2.resize(mask.astype(np.float32), 
                                  (W, H), 
                                  interpolation=cv2.INTER_NEAREST)
    else:
        mask_resized = mask.astype(np.float32)
    
    # 确保掩码是 2D
    if mask_resized.ndim == 3:
        mask_resized = mask_resized.squeeze()
    
    # 确保内参格式
    if isinstance(intrinsics, torch.Tensor):
        intrinsics = intrinsics.detach().cpu().numpy()
    K = np.array(intrinsics).reshape(3, 3)
    
    # ========== 原版流程开始 ==========
    # 1. 应用掩码到深度图
    stable_mask = remove_depth_edges_from_mask(depth_map, mask_resized)
    cur_depth = depth_map * stable_mask
    
    # 2. 创建 UV-Depth 点云（原版函数）
    uv_depth = create_uv_depth(cur_depth)
    
    # 3. 投影到相机坐标系（原版函数）
    pseudo_lidar = project_image_to_cam(uv_depth, K)
    # ========== 原版流程结束 ==========
    
    # 过滤无效点
    valid_mask = np.all(np.isfinite(pseudo_lidar), axis=1) & (pseudo_lidar[:, 2] > 0.01)
    pseudo_lidar = pseudo_lidar[valid_mask]
    
    if len(pseudo_lidar) == 0:
        return None, None
    
    # 计算质心（原版流程）
    centroid = pseudo_lidar.mean(axis=0)
    
    return pseudo_lidar, centroid


def extract_mask_pointcloud(pointmap, mask, intrinsics):
    """
    从场景点云中提取指定掩码区域的点云
    
    Args:
        pointmap: Fast-SAM3D 输出的场景点云 [H, W, 3] 或 [B, H, W, 3]
        mask: 二值掩码 (H, W), 值在 [0, 1]
        intrinsics: 相机内参 (3, 3)
    
    Returns:
        mask_pointcloud: 掩码区域的点云 (N, 3)
        centroid: 点云质心 (3,)
    """
    # 确保点云格式正确
    if hasattr(pointmap, 'cpu'):
        pointmap = pointmap.cpu().numpy()
    
    # 处理 batch 维度
    if pointmap.ndim == 4:  # [B, H, W, 3]
        pointmap = pointmap[0]  # 取第一张
    
    # 确保 mask 格式正确
    if isinstance(mask, torch.Tensor):
        mask = mask.cpu().numpy()
    
    # 确保 mask 形状匹配
    if mask.shape != pointmap.shape[:2]:
        import cv2
        mask_resized = cv2.resize(mask.astype(np.float32), 
                                  (pointmap.shape[1], pointmap.shape[0]), 
                                  interpolation=cv2.INTER_NEAREST)
    else:
        mask_resized = mask.astype(np.float32)
    
    # 展平掩码获取有效点
    mask_flat = mask_resized.flatten() > 0.5
    points_flat = pointmap.reshape(-1, 3)
    
    # 获取掩码区域的点
    masked_points = points_flat[mask_flat]
    
    # 过滤无效点 (深度 <= 0)
    valid_mask = np.all(np.isfinite(masked_points), axis=1) & (np.linalg.norm(masked_points, axis=1) > 0.01)
    masked_points = masked_points[valid_mask]
    
    if len(masked_points) == 0:
        return None, None
    
    # 计算质心
    centroid = masked_points.mean(axis=0)
    
    return masked_points, centroid


def _as_numpy_array(value):
    """Convert torch/list values to a detached numpy array without changing shape."""
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def cleanup_cuda_memory():
    """Release per-object temporary tensors after heavy SAM3D/render work."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        if hasattr(torch.cuda, 'ipc_collect'):
            try:
                torch.cuda.ipc_collect()
            except Exception:
                pass


def is_cuda_oom_error(error):
    msg = str(error).lower()
    return 'cuda out of memory' in msg or 'out of memory' in msg


def category_prior_dimension_bounds(prior_xyz):
    """Return conservative [min,max] metric-size bounds from the class prior."""
    prior_xyz = np.asarray(prior_xyz if prior_xyz is not None else [0.5, 0.5, 0.5], dtype=np.float64)
    prior_xyz = np.maximum(prior_xyz.reshape(3), 1e-3)
    min_dims = np.maximum(prior_xyz * float(RENDER_PRIOR_MIN_RATIO), np.array([0.02, 0.02, 0.02]))
    max_dims = np.maximum(prior_xyz * float(RENDER_PRIOR_MAX_RATIO), np.array([0.25, 0.25, 0.25]))
    return min_dims, max_dims


def parse_float_list(value, default_values):
    if value is None:
        return list(default_values)
    if isinstance(value, (list, tuple)):
        return [float(v) for v in value]
    values = []
    for item in str(value).replace(';', ',').split(','):
        item = item.strip()
        if not item:
            continue
        values.append(float(item))
    return values if values else list(default_values)


def sanitize_scene_points(points, max_points=6000, z_min=0.05):
    """Keep metric camera-space points that are stable enough for box fitting."""
    points = _as_numpy_array(points)
    if points is None:
        return None
    points = points.reshape(-1, 3).astype(np.float64)
    valid = np.all(np.isfinite(points), axis=1) & (points[:, 2] > z_min)
    points = points[valid]
    if points.shape[0] < 10:
        return None

    lo = np.percentile(points, 1.0, axis=0)
    hi = np.percentile(points, 99.0, axis=0)
    keep = np.all((points >= lo) & (points <= hi), axis=1)
    clipped = points[keep]
    if clipped.shape[0] >= 10:
        points = clipped

    if points.shape[0] > max_points:
        step = int(np.ceil(points.shape[0] / max_points))
        points = points[::step][:max_points]
    return points


def robust_pointcloud_center(points):
    """Line A center: mean of a trimmed UniDepth mask point cloud."""
    points = sanitize_scene_points(points)
    if points is None:
        return None

    lo = np.percentile(points, 5.0, axis=0)
    hi = np.percentile(points, 95.0, axis=0)
    keep = np.all((points >= lo) & (points <= hi), axis=1)
    trimmed = points[keep]
    if trimmed.shape[0] >= 10:
        points = trimmed
    return points.mean(axis=0)


def normalize_yaw(yaw):
    """Normalize yaw modulo pi because 3D boxes are unchanged by a 180 degree flip."""
    yaw = float((yaw + np.pi) % (2.0 * np.pi) - np.pi)
    if yaw > np.pi / 2:
        yaw -= np.pi
    if yaw < -np.pi / 2:
        yaw += np.pi
    return float(yaw)


def estimate_box_yaw_from_scene_points(points):
    """Estimate local-to-camera box yaw from XZ PCA of metric scene points."""
    points = sanitize_scene_points(points)
    if points is None or points.shape[0] < 10:
        return None

    xz = points[:, [0, 2]]
    xz = xz - xz.mean(axis=0, keepdims=True)
    cov = xz.T @ xz / max(xz.shape[0] - 1, 1)
    try:
        eigvals, eigvecs = np.linalg.eigh(cov)
    except np.linalg.LinAlgError:
        return None
    major = eigvecs[:, int(np.argmax(eigvals))]
    align_yaw = np.arctan2(major[1], major[0])
    return normalize_yaw(-align_yaw)


def quaternion_wxyz_to_matrix(quat):
    """Fast-SAM3D/PyTorch3D stores quaternions as wxyz."""
    quat = _as_numpy_array(quat)
    if quat is None:
        return None
    quat = np.squeeze(quat).astype(np.float64)
    if quat.shape[0] != 4 or not np.all(np.isfinite(quat)):
        return None
    norm = np.linalg.norm(quat)
    if norm < 1e-8:
        return None
    w, x, y, z = quat / norm
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)


def yaw_from_fastsam_rotation(rotation):
    """Extract a yaw candidate from Fast-SAM3D local-to-camera rotation."""
    R = quaternion_wxyz_to_matrix(rotation)
    if R is None:
        return None
    yaw = np.arctan2(R[0, 2] - R[2, 0], R[0, 0] + R[2, 2])
    return normalize_yaw(yaw)


def extract_mesh_vertices_from_output(output):
    """Get Fast-SAM3D mesh vertices in its relative/local frame."""
    candidates = []

    glb = output.get('glb', None) if isinstance(output, dict) else None
    if glb is not None:
        try:
            verts = np.asarray(glb.vertices)
            if verts.size:
                candidates.append(verts)
        except Exception:
            pass

    mesh = output.get('mesh', None) if isinstance(output, dict) else None
    if mesh is not None:
        try:
            if isinstance(mesh, (list, tuple)):
                mesh = mesh[0] if len(mesh) > 0 else None
            if mesh is not None and hasattr(mesh, 'vertices'):
                verts = mesh.vertices
                candidates.append(_as_numpy_array(verts))
            elif isinstance(mesh, np.ndarray):
                candidates.append(mesh)
        except Exception:
            pass

    gs = output.get('gs', None) if isinstance(output, dict) else None
    if gs is not None and hasattr(gs, '_xyz'):
        try:
            candidates.append(_as_numpy_array(gs._xyz))
        except Exception:
            pass

    for verts in candidates:
        if verts is None:
            continue
        verts = np.asarray(verts).reshape(-1, 3).astype(np.float64)
        finite = np.all(np.isfinite(verts), axis=1)
        verts = verts[finite]
        if verts.shape[0] >= 8:
            return verts
    return None


def _clean_mesh_geometry(vertices, faces=None):
    vertices = _as_numpy_array(vertices)
    if vertices is None:
        return None, None
    vertices = np.asarray(vertices).reshape(-1, 3).astype(np.float64)
    finite = np.all(np.isfinite(vertices), axis=1)
    remap = None
    if not np.all(finite):
        remap = np.full((vertices.shape[0],), -1, dtype=np.int64)
        remap[finite] = np.arange(int(finite.sum()), dtype=np.int64)
    vertices = vertices[finite]
    if vertices.shape[0] < 8:
        return None, None

    clean_faces = None
    if faces is not None:
        faces_arr = _as_numpy_array(faces)
        if faces_arr is not None:
            faces_arr = np.asarray(faces_arr)
            if faces_arr.ndim == 1:
                if faces_arr.size % 3 == 0:
                    faces_arr = faces_arr.reshape(-1, 3)
                else:
                    faces_arr = None
            elif faces_arr.ndim >= 2:
                faces_arr = faces_arr.reshape(-1, faces_arr.shape[-1])

            if faces_arr is not None and faces_arr.shape[1] >= 3:
                faces_arr = faces_arr.astype(np.int64)
                if faces_arr.shape[1] > 3:
                    triangles = []
                    for k in range(1, faces_arr.shape[1] - 1):
                        triangles.append(faces_arr[:, [0, k, k + 1]])
                    faces_arr = np.concatenate(triangles, axis=0)
                if remap is not None:
                    in_range = np.all((faces_arr >= 0) & (faces_arr < remap.shape[0]), axis=1)
                    faces_arr = faces_arr[in_range]
                    if faces_arr.size:
                        faces_arr = remap[faces_arr]
                valid = (
                    np.all(faces_arr >= 0, axis=1) &
                    np.all(faces_arr < vertices.shape[0], axis=1) &
                    (faces_arr[:, 0] != faces_arr[:, 1]) &
                    (faces_arr[:, 1] != faces_arr[:, 2]) &
                    (faces_arr[:, 0] != faces_arr[:, 2])
                )
                faces_arr = faces_arr[valid]
                if faces_arr.shape[0] >= 4:
                    clean_faces = faces_arr

    return vertices, clean_faces


def _mesh_geometry_candidates_from_object(obj):
    candidates = []
    if obj is None:
        return candidates

    if isinstance(obj, np.ndarray):
        vertices, faces = _clean_mesh_geometry(obj, None)
        if vertices is not None:
            candidates.append((vertices, faces))
        return candidates

    if hasattr(obj, 'geometry'):
        try:
            geometries = list(obj.geometry.values())
        except Exception:
            geometries = []
        all_vertices, all_faces = [], []
        offset = 0
        for geom in geometries:
            vertices, faces = _clean_mesh_geometry(
                getattr(geom, 'vertices', None),
                getattr(geom, 'faces', None),
            )
            if vertices is None:
                continue
            all_vertices.append(vertices)
            if faces is not None:
                all_faces.append(faces + offset)
            offset += vertices.shape[0]
        if all_vertices:
            vertices = np.concatenate(all_vertices, axis=0)
            faces = np.concatenate(all_faces, axis=0) if all_faces else None
            candidates.append((vertices, faces))

    if hasattr(obj, 'vertices'):
        vertices, faces = _clean_mesh_geometry(
            getattr(obj, 'vertices', None),
            getattr(obj, 'faces', None),
        )
        if vertices is not None:
            candidates.append((vertices, faces))

    return candidates


def extract_mesh_geometry_from_output(output):
    """Get renderable Fast-SAM3D mesh geometry in its relative/local frame."""
    if not isinstance(output, dict):
        return None, None

    candidates = []
    candidates.extend(_mesh_geometry_candidates_from_object(output.get('glb', None)))

    mesh = output.get('mesh', None)
    if isinstance(mesh, (list, tuple)):
        for item in mesh:
            candidates.extend(_mesh_geometry_candidates_from_object(item))
    else:
        candidates.extend(_mesh_geometry_candidates_from_object(mesh))

    gs = output.get('gs', None)
    if gs is not None and hasattr(gs, '_xyz'):
        vertices, faces = _clean_mesh_geometry(_as_numpy_array(gs._xyz), None)
        if vertices is not None:
            candidates.append((vertices, faces))

    for vertices, faces in candidates:
        if vertices is not None and faces is not None:
            return vertices, faces
    for vertices, faces in candidates:
        if vertices is not None:
            return vertices, faces
    return None, None


def canonicalize_mesh_vertices(mesh_vertices):
    """Center the relative mesh before applying camera-space yaw/scale/translation."""
    vertices = _as_numpy_array(mesh_vertices)
    if vertices is None:
        return None, None, None
    vertices = np.asarray(vertices).reshape(-1, 3).astype(np.float64)
    vertices = vertices[np.all(np.isfinite(vertices), axis=1)]
    if vertices.shape[0] < 8:
        return None, None, None
    lo = np.percentile(vertices, 1.0, axis=0)
    hi = np.percentile(vertices, 99.0, axis=0)
    mesh_center = (lo + hi) / 2.0
    mesh_extent = np.maximum(hi - lo, 1e-4)
    return vertices - mesh_center[None, :], mesh_center, mesh_extent


def transform_mesh_vertices_to_camera(mesh_vertices, center, scale, yaw):
    local_vertices, _, _ = canonicalize_mesh_vertices(mesh_vertices)
    if local_vertices is None:
        return None
    center = np.asarray(center, dtype=np.float64).reshape(3)
    R_yaw = rotate_y_cubercnn(yaw)
    return local_vertices @ R_yaw.T * float(scale) + center[None, :]


def scale_camera_matrix(K, scale_x, scale_y=None):
    if scale_y is None:
        scale_y = scale_x
    K_scaled = np.asarray(K, dtype=np.float64).reshape(3, 3).copy()
    K_scaled[0, 0] *= scale_x
    K_scaled[0, 2] *= scale_x
    K_scaled[1, 1] *= scale_y
    K_scaled[1, 2] *= scale_y
    return K_scaled


def prepare_render_reference(mask, real_depth, image_shape, render_scale):
    H, W = image_shape[:2]
    render_scale = float(render_scale)
    Hr = max(1, int(round(H * render_scale)))
    Wr = max(1, int(round(W * render_scale)))

    mask_np = _as_numpy_array(mask)
    if mask_np is None:
        return None, None, None, None
    mask_np = np.squeeze(mask_np).astype(np.float32)
    if mask_np.shape != (H, W):
        mask_np = cv2.resize(mask_np, (W, H), interpolation=cv2.INTER_NEAREST)
    mask_small = cv2.resize(mask_np, (Wr, Hr), interpolation=cv2.INTER_NEAREST) > 0.5

    depth_small = None
    if real_depth is not None:
        depth_np = _as_numpy_array(real_depth)
        if depth_np is not None:
            depth_np = np.squeeze(depth_np).astype(np.float32)
            if depth_np.shape != (H, W):
                depth_np = cv2.resize(depth_np, (W, H), interpolation=cv2.INTER_NEAREST)
            depth_small = cv2.resize(depth_np, (Wr, Hr), interpolation=cv2.INTER_NEAREST)

    return mask_small, depth_small, Hr, Wr


def render_mesh_depth_mask(vertices_cam, faces, K, height, width, max_faces=None):
    """Small non-differentiable z-buffer renderer for pseudo-label verification."""
    vertices, faces = _clean_mesh_geometry(vertices_cam, faces)
    if vertices is None or faces is None:
        return None, None

    if max_faces is None:
        max_faces = RENDER_ALIGN_MAX_FACES
    if faces.shape[0] > max_faces:
        sample_idx = np.linspace(0, faces.shape[0] - 1, max_faces).astype(np.int64)
        faces = faces[sample_idx]

    K = np.asarray(K, dtype=np.float64).reshape(3, 3)
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    depth = np.full((height, width), np.inf, dtype=np.float32)
    mask = np.zeros((height, width), dtype=bool)

    tri_vertices = vertices[faces]
    for tri in tri_vertices:
        z = tri[:, 2]
        if np.any(z <= 0.05) or not np.all(np.isfinite(z)):
            continue

        u = fx * tri[:, 0] / z + cx
        v = fy * tri[:, 1] / z + cy
        if not (np.all(np.isfinite(u)) and np.all(np.isfinite(v))):
            continue

        xmin = max(int(np.floor(np.min(u))), 0)
        xmax = min(int(np.ceil(np.max(u))), width - 1)
        ymin = max(int(np.floor(np.min(v))), 0)
        ymax = min(int(np.ceil(np.max(v))), height - 1)
        if xmax < xmin or ymax < ymin:
            continue

        x0, x1, x2 = u
        y0, y1, y2 = v
        denom = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
        if abs(denom) < 1e-8:
            continue

        yy, xx = np.mgrid[ymin:ymax + 1, xmin:xmax + 1]
        px = xx.astype(np.float64) + 0.5
        py = yy.astype(np.float64) + 0.5
        w0 = ((y1 - y2) * (px - x2) + (x2 - x1) * (py - y2)) / denom
        w1 = ((y2 - y0) * (px - x2) + (x0 - x2) * (py - y2)) / denom
        w2 = 1.0 - w0 - w1
        inside = (w0 >= -1e-5) & (w1 >= -1e-5) & (w2 >= -1e-5)
        if not np.any(inside):
            continue

        tri_depth = w0 * z[0] + w1 * z[1] + w2 * z[2]
        patch_depth = depth[ymin:ymax + 1, xmin:xmax + 1]
        update = inside & (tri_depth > 0.05) & (tri_depth < patch_depth)
        if not np.any(update):
            continue

        patch_depth[update] = tri_depth[update].astype(np.float32)
        depth[ymin:ymax + 1, xmin:xmax + 1] = patch_depth
        mask_patch = mask[ymin:ymax + 1, xmin:xmax + 1]
        mask_patch[update] = True
        mask[ymin:ymax + 1, xmin:xmax + 1] = mask_patch

    depth[~np.isfinite(depth)] = 0.0
    if int(mask.sum()) == 0:
        return depth, mask
    return depth, mask


def render_mesh_depth_mask_with_object_points(
    object_vertices,
    vertices_cam,
    faces,
    K,
    height,
    width,
    max_faces=None,
):
    """Render depth/mask and keep the object-frame 3D point at each visible pixel."""
    object_vertices = _as_numpy_array(object_vertices)
    vertices_cam = _as_numpy_array(vertices_cam)
    faces = _as_numpy_array(faces)
    if object_vertices is None or vertices_cam is None or faces is None:
        return None, None, None

    object_vertices = np.asarray(object_vertices).reshape(-1, 3).astype(np.float64)
    vertices_cam = np.asarray(vertices_cam).reshape(-1, 3).astype(np.float64)
    if object_vertices.shape != vertices_cam.shape or object_vertices.shape[0] < 8:
        return None, None, None

    faces = np.asarray(faces)
    if faces.ndim == 1:
        if faces.size % 3 != 0:
            return None, None, None
        faces = faces.reshape(-1, 3)
    elif faces.ndim >= 2:
        faces = faces.reshape(-1, faces.shape[-1])
    if faces.shape[1] < 3:
        return None, None, None
    faces = faces.astype(np.int64)
    if faces.shape[1] > 3:
        triangles = []
        for k in range(1, faces.shape[1] - 1):
            triangles.append(faces[:, [0, k, k + 1]])
        faces = np.concatenate(triangles, axis=0)

    finite = np.all(np.isfinite(object_vertices), axis=1) & np.all(np.isfinite(vertices_cam), axis=1)
    if not np.all(finite):
        remap = np.full((finite.shape[0],), -1, dtype=np.int64)
        remap[finite] = np.arange(int(finite.sum()), dtype=np.int64)
        in_range = np.all((faces >= 0) & (faces < remap.shape[0]), axis=1)
        faces = faces[in_range]
        if faces.size == 0:
            return None, None, None
        faces = remap[faces]
        object_vertices = object_vertices[finite]
        vertices_cam = vertices_cam[finite]

    valid = (
        np.all(faces >= 0, axis=1) &
        np.all(faces < vertices_cam.shape[0], axis=1) &
        (faces[:, 0] != faces[:, 1]) &
        (faces[:, 1] != faces[:, 2]) &
        (faces[:, 0] != faces[:, 2])
    )
    faces = faces[valid]
    if faces.shape[0] < 4:
        return None, None, None

    if max_faces is None:
        max_faces = RENDER_ALIGN_MAX_FACES
    if faces.shape[0] > max_faces:
        sample_idx = np.linspace(0, faces.shape[0] - 1, max_faces).astype(np.int64)
        faces = faces[sample_idx]

    K = np.asarray(K, dtype=np.float64).reshape(3, 3)
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    depth = np.full((height, width), np.inf, dtype=np.float32)
    mask = np.zeros((height, width), dtype=bool)
    object_point_map = np.zeros((height, width, 3), dtype=np.float32)

    for face in faces:
        tri_cam = vertices_cam[face]
        tri_obj = object_vertices[face]
        z = tri_cam[:, 2]
        if np.any(z <= 0.05) or not np.all(np.isfinite(z)):
            continue

        u = fx * tri_cam[:, 0] / z + cx
        v = fy * tri_cam[:, 1] / z + cy
        if not (np.all(np.isfinite(u)) and np.all(np.isfinite(v))):
            continue

        xmin = max(int(np.floor(np.min(u))), 0)
        xmax = min(int(np.ceil(np.max(u))), width - 1)
        ymin = max(int(np.floor(np.min(v))), 0)
        ymax = min(int(np.ceil(np.max(v))), height - 1)
        if xmax < xmin or ymax < ymin:
            continue

        x0, x1, x2 = u
        y0, y1, y2 = v
        denom = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
        if abs(denom) < 1e-8:
            continue

        yy, xx = np.mgrid[ymin:ymax + 1, xmin:xmax + 1]
        px = xx.astype(np.float64) + 0.5
        py = yy.astype(np.float64) + 0.5
        w0 = ((y1 - y2) * (px - x2) + (x2 - x1) * (py - y2)) / denom
        w1 = ((y2 - y0) * (px - x2) + (x0 - x2) * (py - y2)) / denom
        w2 = 1.0 - w0 - w1
        inside = (w0 >= -1e-5) & (w1 >= -1e-5) & (w2 >= -1e-5)
        if not np.any(inside):
            continue

        tri_depth = w0 * z[0] + w1 * z[1] + w2 * z[2]
        patch_depth = depth[ymin:ymax + 1, xmin:xmax + 1]
        update = inside & (tri_depth > 0.05) & (tri_depth < patch_depth)
        if not np.any(update):
            continue

        interp_obj = (
            w0[..., None] * tri_obj[0][None, None, :] +
            w1[..., None] * tri_obj[1][None, None, :] +
            w2[..., None] * tri_obj[2][None, None, :]
        )
        patch_depth[update] = tri_depth[update].astype(np.float32)
        depth[ymin:ymax + 1, xmin:xmax + 1] = patch_depth

        mask_patch = mask[ymin:ymax + 1, xmin:xmax + 1]
        mask_patch[update] = True
        mask[ymin:ymax + 1, xmin:xmax + 1] = mask_patch

        object_patch = object_point_map[ymin:ymax + 1, xmin:xmax + 1]
        object_patch[update] = interp_obj[update].astype(np.float32)
        object_point_map[ymin:ymax + 1, xmin:xmax + 1] = object_patch

    depth[~np.isfinite(depth)] = 0.0
    return depth, mask, object_point_map


def extract_mask_boundary_points(mask, max_points=RENDER_PNP_MAX_POINTS):
    mask = np.asarray(mask).astype(bool)
    if mask.ndim != 2 or int(mask.sum()) == 0:
        return np.zeros((0, 2), dtype=np.float32)

    kernel = np.ones((3, 3), dtype=np.uint8)
    eroded = cv2.erode(mask.astype(np.uint8), kernel, iterations=1).astype(bool)
    boundary = mask & (~eroded)
    ys, xs = np.where(boundary)
    if xs.size == 0:
        ys, xs = np.where(mask)
    if xs.size == 0:
        return np.zeros((0, 2), dtype=np.float32)

    pts = np.stack([xs, ys], axis=1).astype(np.float32)
    if pts.shape[0] > max_points:
        idx = np.linspace(0, pts.shape[0] - 1, max_points).astype(np.int64)
        pts = pts[idx]
    return pts


def match_render_to_target_boundary(
    render_mask,
    target_mask,
    object_point_map,
    max_dist=RENDER_PNP_MAX_CONTOUR_DIST,
    max_points=RENDER_PNP_MAX_POINTS,
):
    render_pts = extract_mask_boundary_points(render_mask, max_points=max_points)
    target_pts = extract_mask_boundary_points(target_mask, max_points=max_points)
    if render_pts.shape[0] == 0 or target_pts.shape[0] == 0:
        return None, None, {'pnp_matches': 0, 'pnp_match_reason': 'empty_boundary'}

    render_xy = np.round(render_pts).astype(np.int64)
    h, w = object_point_map.shape[:2]
    in_bounds = (
        (render_xy[:, 0] >= 0) & (render_xy[:, 0] < w) &
        (render_xy[:, 1] >= 0) & (render_xy[:, 1] < h)
    )
    render_pts = render_pts[in_bounds]
    render_xy = render_xy[in_bounds]
    if render_pts.shape[0] == 0:
        return None, None, {'pnp_matches': 0, 'pnp_match_reason': 'render_out_of_bounds'}

    object_points = object_point_map[render_xy[:, 1], render_xy[:, 0]].astype(np.float64)
    valid_obj = np.all(np.isfinite(object_points), axis=1)
    render_pts = render_pts[valid_obj]
    object_points = object_points[valid_obj]
    if object_points.shape[0] < RENDER_PNP_MIN_MATCHES:
        return None, None, {
            'pnp_matches': int(object_points.shape[0]),
            'pnp_match_reason': 'few_render_points',
        }

    diff = render_pts[:, None, :] - target_pts[None, :, :]
    dists = np.linalg.norm(diff, axis=2)
    nearest = np.argmin(dists, axis=1)
    nearest_dist = dists[np.arange(dists.shape[0]), nearest]
    keep = nearest_dist <= float(max_dist)
    if int(keep.sum()) < RENDER_PNP_MIN_MATCHES:
        return None, None, {
            'pnp_matches': int(keep.sum()),
            'pnp_match_reason': 'few_close_boundary_matches',
            'pnp_boundary_median_dist': float(np.median(nearest_dist)) if nearest_dist.size else float('inf'),
        }

    matched_object = object_points[keep].astype(np.float32)
    matched_image = target_pts[nearest[keep]].astype(np.float32)
    return matched_object, matched_image, {
        'pnp_match_method': 'boundary_nearest',
        'pnp_matches': int(matched_object.shape[0]),
        'pnp_boundary_median_dist': float(np.median(nearest_dist[keep])),
    }


def mask_edges_uint8(mask):
    mask = np.asarray(mask).astype(bool)
    if mask.ndim != 2 or int(mask.sum()) == 0:
        return np.zeros(mask.shape[:2], dtype=np.uint8)
    kernel = np.ones((3, 3), dtype=np.uint8)
    clean = cv2.morphologyEx(mask.astype(np.uint8) * 255, cv2.MORPH_CLOSE, kernel, iterations=1)
    return cv2.Canny(clean, 50, 150)


def render_depth_feature_image(render_depth, render_mask):
    render_mask = np.asarray(render_mask).astype(bool)
    render_depth = np.asarray(render_depth).astype(np.float32)
    image = np.zeros(render_mask.shape, dtype=np.uint8)
    valid = render_mask & np.isfinite(render_depth) & (render_depth > 0.05)
    if int(valid.sum()) > 0:
        vals = render_depth[valid].astype(np.float32)
        lo, hi = np.percentile(vals, [2.0, 98.0])
        if hi <= lo:
            hi = lo + 1e-3
        norm = np.clip((render_depth - lo) / (hi - lo), 0.0, 1.0)
        image[valid] = (255.0 * (1.0 - norm[valid])).astype(np.uint8)
    edges = mask_edges_uint8(render_mask)
    return np.maximum(image, edges)


def target_feature_image(target_image, target_mask):
    target_mask = np.asarray(target_mask).astype(bool)
    if target_image is not None:
        image = _as_numpy_array(target_image)
        if image is not None and image.ndim == 3:
            if image.shape[2] == 3:
                gray = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_RGB2GRAY)
            else:
                gray = np.squeeze(image).astype(np.uint8)
        elif image is not None and image.ndim == 2:
            gray = image.astype(np.uint8)
        else:
            gray = np.zeros(target_mask.shape, dtype=np.uint8)
    else:
        gray = np.zeros(target_mask.shape, dtype=np.uint8)

    if gray.shape != target_mask.shape:
        gray = cv2.resize(gray, (target_mask.shape[1], target_mask.shape[0]), interpolation=cv2.INTER_AREA)
    gray = cv2.equalizeHist(gray)
    edges = mask_edges_uint8(target_mask)
    gray = np.maximum((gray * target_mask.astype(np.uint8) * 0.65).astype(np.uint8), edges)
    return gray


def match_render_to_target_opencv_features(
    render_depth,
    render_mask,
    target_mask,
    object_point_map,
    target_image=None,
    backend='sift',
    max_points=RENDER_PNP_MAX_POINTS,
):
    if render_depth is None:
        return None, None, {
            'pnp_match_method': backend,
            'pnp_matches': 0,
            'pnp_match_reason': 'missing_render_depth',
        }

    render_gray = render_depth_feature_image(render_depth, render_mask)
    target_gray = target_feature_image(target_image, target_mask)
    target_valid = cv2.dilate(target_mask.astype(np.uint8), np.ones((5, 5), dtype=np.uint8), iterations=1).astype(bool)

    backend = str(backend).lower()
    try:
        if backend in ('sift', 'opencv'):
            if not hasattr(cv2, 'SIFT_create'):
                return None, None, {
                    'pnp_match_method': 'sift',
                    'pnp_matches': 0,
                    'pnp_match_reason': 'sift_unavailable',
                }
            detector = cv2.SIFT_create(nfeatures=max(int(max_points) * 4, 400))
            norm_type = cv2.NORM_L2
        elif backend == 'akaze':
            detector = cv2.AKAZE_create()
            norm_type = cv2.NORM_HAMMING
        else:
            return None, None, {
                'pnp_match_method': backend,
                'pnp_matches': 0,
                'pnp_match_reason': 'unsupported_opencv_feature_backend',
            }

        kp_render, des_render = detector.detectAndCompute(render_gray, None)
        kp_target, des_target = detector.detectAndCompute(target_gray, None)
        if des_render is None or des_target is None or not kp_render or not kp_target:
            return None, None, {
                'pnp_match_method': backend,
                'pnp_matches': 0,
                'pnp_match_reason': 'empty_feature_descriptors',
            }

        matcher = cv2.BFMatcher(norm_type, crossCheck=False)
        knn = matcher.knnMatch(des_render, des_target, k=2)
        good = []
        for pair in knn:
            if len(pair) < 2:
                continue
            first, second = pair
            if first.distance < 0.75 * second.distance:
                good.append(first)
        good = sorted(good, key=lambda m: m.distance)[:int(max_points)]

        object_points = []
        image_points = []
        h, w = object_point_map.shape[:2]
        for match in good:
            rx, ry = kp_render[match.queryIdx].pt
            tx, ty = kp_target[match.trainIdx].pt
            ix, iy = int(round(rx)), int(round(ry))
            tx_i, ty_i = int(round(tx)), int(round(ty))
            if ix < 0 or ix >= w or iy < 0 or iy >= h:
                continue
            if tx_i < 0 or tx_i >= w or ty_i < 0 or ty_i >= h:
                continue
            if not render_mask[iy, ix] or not target_valid[ty_i, tx_i]:
                continue
            obj = object_point_map[iy, ix]
            if not np.all(np.isfinite(obj)):
                continue
            object_points.append(obj)
            image_points.append([tx, ty])

        if len(object_points) < RENDER_PNP_MIN_MATCHES:
            return None, None, {
                'pnp_match_method': backend,
                'pnp_matches': int(len(object_points)),
                'pnp_match_reason': 'few_feature_matches',
            }

        return (
            np.asarray(object_points, dtype=np.float32),
            np.asarray(image_points, dtype=np.float32),
            {
                'pnp_match_method': backend,
                'pnp_matches': int(len(object_points)),
                'pnp_feature_median_distance': float(np.median([m.distance for m in good])) if good else float('inf'),
            },
        )
    except cv2.error as error:
        return None, None, {
            'pnp_match_method': backend,
            'pnp_matches': 0,
            'pnp_match_reason': f'feature_cv2_error:{error}',
        }


def match_render_to_target_orb_edges(
    render_mask,
    target_mask,
    object_point_map,
    max_points=RENDER_PNP_MAX_POINTS,
):
    """Feature-style edge matching between rendered silhouette and real mask."""
    try:
        render_edges = mask_edges_uint8(render_mask)
        target_edges = mask_edges_uint8(target_mask)
        if int((render_edges > 0).sum()) == 0 or int((target_edges > 0).sum()) == 0:
            return None, None, {'pnp_matches': 0, 'pnp_match_reason': 'empty_orb_edges'}

        orb = cv2.ORB_create(
            nfeatures=max(int(max_points) * 3, 300),
            edgeThreshold=3,
            patchSize=15,
            fastThreshold=5,
        )
        kp_render, des_render = orb.detectAndCompute(render_edges, None)
        kp_target, des_target = orb.detectAndCompute(target_edges, None)
        if des_render is None or des_target is None or not kp_render or not kp_target:
            return None, None, {'pnp_matches': 0, 'pnp_match_reason': 'empty_orb_descriptors'}

        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = matcher.match(des_render, des_target)
        if not matches:
            return None, None, {'pnp_matches': 0, 'pnp_match_reason': 'empty_orb_matches'}

        matches = sorted(matches, key=lambda m: m.distance)
        distances = np.array([m.distance for m in matches], dtype=np.float64)
        dist_keep = distances <= max(float(np.percentile(distances, 70.0)), 25.0)
        matches = [m for m, keep in zip(matches, dist_keep) if keep]
        matches = matches[:int(max_points)]

        object_points = []
        image_points = []
        h, w = object_point_map.shape[:2]
        for match in matches:
            rx, ry = kp_render[match.queryIdx].pt
            tx, ty = kp_target[match.trainIdx].pt
            ix, iy = int(round(rx)), int(round(ry))
            if ix < 0 or ix >= w or iy < 0 or iy >= h:
                continue
            if not render_mask[iy, ix]:
                continue
            obj = object_point_map[iy, ix]
            if not np.all(np.isfinite(obj)):
                continue
            object_points.append(obj)
            image_points.append([tx, ty])

        if len(object_points) < RENDER_PNP_MIN_MATCHES:
            return None, None, {
                'pnp_match_method': 'orb_edge',
                'pnp_matches': int(len(object_points)),
                'pnp_match_reason': 'few_orb_edge_matches',
            }

        return (
            np.asarray(object_points, dtype=np.float32),
            np.asarray(image_points, dtype=np.float32),
            {
                'pnp_match_method': 'orb_edge',
                'pnp_matches': int(len(object_points)),
                'pnp_orb_median_distance': float(np.median([m.distance for m in matches])) if matches else float('inf'),
            },
        )
    except cv2.error as error:
        return None, None, {
            'pnp_match_method': 'orb_edge',
            'pnp_matches': 0,
            'pnp_match_reason': f'orb_cv2_error:{error}',
        }


_MAST3R_MODEL_CACHE = {}


def _ensure_mast3r_imports():
    root = str(MAST3R_ROOT or '').strip()
    if root:
        root_path = Path(root).expanduser().resolve()
        if not root_path.exists():
            return None, None, None, f'mast3r_root_missing:{root_path}'
        root = str(root_path)
        if root not in sys.path:
            sys.path.insert(0, root)
        dust3r_root = str(root_path / 'dust3r')
        if (root_path / 'dust3r').exists() and dust3r_root not in sys.path:
            sys.path.insert(0, dust3r_root)

    try:
        import mast3r.utils.path_to_dust3r  # noqa: F401
        from mast3r.model import AsymmetricMASt3R
        from mast3r.fast_nn import fast_reciprocal_NNs
        from dust3r.inference import inference
        return AsymmetricMASt3R, fast_reciprocal_NNs, inference, None
    except Exception as error:
        return None, None, None, f'mast3r_import_error:{type(error).__name__}:{error}'


def _get_mast3r_model():
    AsymmetricMASt3R, fast_reciprocal_NNs, inference, reason = _ensure_mast3r_imports()
    if reason is not None:
        return None, None, None, None, reason

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_name = str(MAST3R_MODEL_NAME or '').strip()
    if not model_name:
        return None, None, None, None, 'mast3r_model_name_empty'

    model_source = _resolve_mast3r_model_source(model_name)
    key = (str(MAST3R_ROOT), str(model_source), device)
    model = _MAST3R_MODEL_CACHE.get(key)
    if model is None:
        old_offline = os.environ.get('HF_HUB_OFFLINE')
        try:
            os.environ['HF_HUB_OFFLINE'] = '1'
            print(f"📦 Loading MASt3R dense matcher on {device}: {model_source}")
            model = AsymmetricMASt3R.from_pretrained(str(model_source)).to(device)
            model.eval()
            _MAST3R_MODEL_CACHE[key] = model
        except Exception as error:
            cleanup_cuda_memory()
            return None, None, None, None, f'mast3r_model_load_error:{type(error).__name__}:{error}'
        finally:
            if old_offline is None:
                os.environ.pop('HF_HUB_OFFLINE', None)
            else:
                os.environ['HF_HUB_OFFLINE'] = old_offline

    return model, fast_reciprocal_NNs, inference, device, None


def _resolve_mast3r_model_source(model_name):
    model_path = Path(str(model_name)).expanduser()
    if model_path.exists():
        return model_path.resolve()

    if '/' not in str(model_name):
        return model_name

    cache_home = Path(os.environ.get('HF_HOME', Path.home() / '.cache' / 'huggingface'))
    hub_root = cache_home / 'hub'
    cache_name = 'models--' + str(model_name).replace('/', '--')
    model_cache = hub_root / cache_name
    refs_main = model_cache / 'refs' / 'main'
    if refs_main.exists():
        revision = refs_main.read_text().strip()
        snapshot = model_cache / 'snapshots' / revision
        if (snapshot / 'config.json').exists():
            return snapshot.resolve()

    snapshots = model_cache / 'snapshots'
    if snapshots.exists():
        for snapshot in sorted(snapshots.iterdir(), reverse=True):
            if (snapshot / 'config.json').exists():
                return snapshot.resolve()
    return model_name


def _mast3r_view_from_rgb(image_rgb, idx=0):
    image_rgb = np.asarray(image_rgb, dtype=np.uint8)
    tensor = torch.from_numpy(image_rgb.astype(np.float32) / 255.0).permute(2, 0, 1)
    tensor = (tensor - 0.5) / 0.5
    h, w = image_rgb.shape[:2]
    return {
        'img': tensor[None],
        'true_shape': np.int32([[h, w]]),
        'idx': int(idx),
        'instance': str(idx),
    }


def _masked_target_rgb_for_mast3r(target_image, target_mask):
    target_mask = np.asarray(target_mask).astype(bool)
    if target_image is not None:
        image = _as_numpy_array(target_image)
        if image is not None and image.ndim == 3 and image.shape[2] >= 3:
            rgb = image[..., :3].astype(np.uint8)
        elif image is not None and image.ndim == 2:
            gray = image.astype(np.uint8)
            rgb = np.repeat(gray[..., None], 3, axis=2)
        else:
            gray = target_feature_image(None, target_mask)
            rgb = np.repeat(gray[..., None], 3, axis=2)
    else:
        gray = target_feature_image(None, target_mask)
        rgb = np.repeat(gray[..., None], 3, axis=2)

    if rgb.shape[:2] != target_mask.shape:
        rgb = cv2.resize(rgb, (target_mask.shape[1], target_mask.shape[0]), interpolation=cv2.INTER_AREA)

    out = np.full_like(rgb, 255, dtype=np.uint8)
    out[target_mask] = rgb[target_mask]
    edges = mask_edges_uint8(target_mask)
    out[edges > 0] = np.array([0, 0, 0], dtype=np.uint8)
    return out


def _render_rgb_for_mast3r(render_depth, render_mask):
    render_mask = np.asarray(render_mask).astype(bool)
    feature = render_depth_feature_image(render_depth, render_mask)
    rgb = np.full((render_mask.shape[0], render_mask.shape[1], 3), 255, dtype=np.uint8)
    rgb[render_mask] = np.repeat(feature[..., None], 3, axis=2)[render_mask]
    edges = mask_edges_uint8(render_mask)
    rgb[edges > 0] = np.array([0, 0, 0], dtype=np.uint8)
    return rgb


def _square_crop_for_mast3r(image_rgb, mask, image_size=None, pad=None):
    if image_size is None:
        image_size = int(MAST3R_IMAGE_SIZE)
    if pad is None:
        pad = float(MAST3R_CROP_PAD)

    image_rgb = np.asarray(image_rgb, dtype=np.uint8)
    mask = np.asarray(mask).astype(bool)
    ys, xs = np.where(mask)
    if xs.size == 0 or ys.size == 0:
        return None, None

    h, w = mask.shape[:2]
    x1, x2 = float(xs.min()), float(xs.max() + 1)
    y1, y2 = float(ys.min()), float(ys.max() + 1)
    side = max(x2 - x1, y2 - y1, 8.0) * max(float(pad), 1.0)
    side_i = max(int(np.ceil(side)), 8)
    cx = (x1 + x2) * 0.5
    cy = (y1 + y2) * 0.5
    x0 = int(np.floor(cx - side_i * 0.5))
    y0 = int(np.floor(cy - side_i * 0.5))

    canvas = np.full((side_i, side_i, 3), 255, dtype=np.uint8)
    src_x1 = max(x0, 0)
    src_y1 = max(y0, 0)
    src_x2 = min(x0 + side_i, w)
    src_y2 = min(y0 + side_i, h)
    if src_x2 <= src_x1 or src_y2 <= src_y1:
        return None, None

    dst_x1 = src_x1 - x0
    dst_y1 = src_y1 - y0
    dst_x2 = dst_x1 + (src_x2 - src_x1)
    dst_y2 = dst_y1 + (src_y2 - src_y1)
    canvas[dst_y1:dst_y2, dst_x1:dst_x2] = image_rgb[src_y1:src_y2, src_x1:src_x2]
    crop = cv2.resize(canvas, (int(image_size), int(image_size)), interpolation=cv2.INTER_AREA)
    scale = float(image_size) / float(side_i)
    params = {'x0': float(x0), 'y0': float(y0), 'scale': float(scale), 'side': float(side_i)}
    return crop, params


def _mast3r_points_to_canvas(points, crop_params):
    points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    scale = max(float(crop_params['scale']), 1e-8)
    mapped = np.empty_like(points, dtype=np.float64)
    mapped[:, 0] = float(crop_params['x0']) + points[:, 0] / scale
    mapped[:, 1] = float(crop_params['y0']) + points[:, 1] / scale
    return mapped


def match_render_to_target_mast3r(
    render_depth,
    render_mask,
    target_mask,
    object_point_map,
    target_image=None,
    max_points=RENDER_PNP_MAX_POINTS,
):
    if render_depth is None:
        return None, None, {
            'pnp_match_method': 'mast3r_dense',
            'pnp_matches': 0,
            'pnp_match_reason': 'missing_render_depth',
        }

    model, fast_reciprocal_NNs, inference, device, reason = _get_mast3r_model()
    if reason is not None:
        return None, None, {
            'pnp_match_method': 'mast3r_dense',
            'pnp_matches': 0,
            'pnp_match_reason': reason,
        }

    render_mask = np.asarray(render_mask).astype(bool)
    target_mask = np.asarray(target_mask).astype(bool)
    if int(render_mask.sum()) == 0 or int(target_mask.sum()) == 0:
        return None, None, {
            'pnp_match_method': 'mast3r_dense',
            'pnp_matches': 0,
            'pnp_match_reason': 'empty_mask_for_mast3r',
        }

    target_rgb = _masked_target_rgb_for_mast3r(target_image, target_mask)
    render_rgb = _render_rgb_for_mast3r(render_depth, render_mask)
    target_crop, target_params = _square_crop_for_mast3r(target_rgb, target_mask)
    render_crop, render_params = _square_crop_for_mast3r(render_rgb, render_mask)
    if target_crop is None or render_crop is None:
        return None, None, {
            'pnp_match_method': 'mast3r_dense',
            'pnp_matches': 0,
            'pnp_match_reason': 'bad_mast3r_crop',
        }

    try:
        view_target = _mast3r_view_from_rgb(target_crop, idx=0)
        view_render = _mast3r_view_from_rgb(render_crop, idx=1)
        with torch.inference_mode():
            output = inference([(view_target, view_render)], model, device, batch_size=1, verbose=False)

        view1, pred1 = output['view1'], output['pred1']
        view2, pred2 = output['view2'], output['pred2']
        desc1 = pred1['desc'].squeeze(0).detach()
        desc2 = pred2['desc'].squeeze(0).detach()
        matches_target, matches_render = fast_reciprocal_NNs(
            desc1,
            desc2,
            subsample_or_initxy1=int(MAST3R_SUBSAMPLE),
            device=device,
            dist='dot',
            block_size=2**13,
        )
    except Exception as error:
        cleanup_cuda_memory()
        return None, None, {
            'pnp_match_method': 'mast3r_dense',
            'pnp_matches': 0,
            'pnp_match_reason': f'mast3r_inference_error:{type(error).__name__}:{error}',
        }

    matches_target = _as_numpy_array(matches_target).reshape(-1, 2).astype(np.float64)
    matches_render = _as_numpy_array(matches_render).reshape(-1, 2).astype(np.float64)
    raw_matches = int(min(matches_target.shape[0], matches_render.shape[0]))
    if raw_matches == 0:
        return None, None, {
            'pnp_match_method': 'mast3r_dense',
            'pnp_matches': 0,
            'pnp_match_reason': 'empty_mast3r_matches',
        }

    h0, w0 = view1['true_shape'][0]
    h1, w1 = view2['true_shape'][0]
    valid_target_crop = (
        (matches_target[:, 0] >= 3) & (matches_target[:, 0] < int(w0) - 3) &
        (matches_target[:, 1] >= 3) & (matches_target[:, 1] < int(h0) - 3)
    )
    valid_render_crop = (
        (matches_render[:, 0] >= 3) & (matches_render[:, 0] < int(w1) - 3) &
        (matches_render[:, 1] >= 3) & (matches_render[:, 1] < int(h1) - 3)
    )
    valid = valid_target_crop & valid_render_crop
    matches_target = matches_target[valid]
    matches_render = matches_render[valid]
    if matches_target.shape[0] == 0:
        return None, None, {
            'pnp_match_method': 'mast3r_dense',
            'pnp_matches': 0,
            'pnp_match_reason': 'mast3r_matches_only_on_border',
            'pnp_mast3r_raw_matches': raw_matches,
        }

    target_xy = _mast3r_points_to_canvas(matches_target, target_params)
    render_xy = _mast3r_points_to_canvas(matches_render, render_params)

    h, w = object_point_map.shape[:2]
    target_valid_mask = cv2.dilate(
        target_mask.astype(np.uint8),
        np.ones((5, 5), dtype=np.uint8),
        iterations=1,
    ).astype(bool)

    target_round = np.round(target_xy).astype(np.int64)
    render_round = np.round(render_xy).astype(np.int64)
    in_bounds = (
        (render_round[:, 0] >= 0) & (render_round[:, 0] < w) &
        (render_round[:, 1] >= 0) & (render_round[:, 1] < h) &
        (target_round[:, 0] >= 0) & (target_round[:, 0] < w) &
        (target_round[:, 1] >= 0) & (target_round[:, 1] < h)
    )
    target_xy = target_xy[in_bounds]
    render_round = render_round[in_bounds]
    target_round = target_round[in_bounds]
    if target_xy.shape[0] == 0:
        return None, None, {
            'pnp_match_method': 'mast3r_dense',
            'pnp_matches': 0,
            'pnp_match_reason': 'mast3r_matches_out_of_bounds',
            'pnp_mast3r_raw_matches': raw_matches,
        }

    keep = (
        render_mask[render_round[:, 1], render_round[:, 0]] &
        target_valid_mask[target_round[:, 1], target_round[:, 0]]
    )
    object_points = object_point_map[render_round[:, 1], render_round[:, 0]].astype(np.float64)
    keep &= np.all(np.isfinite(object_points), axis=1)
    keep &= np.linalg.norm(object_points, axis=1) > 1e-8
    object_points = object_points[keep]
    image_points = target_xy[keep]

    if object_points.shape[0] > int(max_points):
        keep_idx = np.linspace(0, object_points.shape[0] - 1, int(max_points)).astype(np.int64)
        object_points = object_points[keep_idx]
        image_points = image_points[keep_idx]

    if object_points.shape[0] < RENDER_PNP_MIN_MATCHES:
        return None, None, {
            'pnp_match_method': 'mast3r_dense',
            'pnp_matches': int(object_points.shape[0]),
            'pnp_match_reason': 'few_mast3r_dense_matches',
            'pnp_mast3r_raw_matches': raw_matches,
            'pnp_mast3r_after_mask_matches': int(object_points.shape[0]),
        }

    return (
        object_points.astype(np.float32),
        image_points.astype(np.float32),
        {
            'pnp_match_method': 'mast3r_dense',
            'pnp_matches': int(object_points.shape[0]),
            'pnp_mast3r_raw_matches': raw_matches,
            'pnp_mast3r_after_mask_matches': int(object_points.shape[0]),
            'pnp_mast3r_image_size': int(MAST3R_IMAGE_SIZE),
            'pnp_mast3r_subsample': int(MAST3R_SUBSAMPLE),
        },
    )


def match_render_to_target_external_stub(backend):
    import importlib.util
    installed = importlib.util.find_spec(backend) is not None
    return None, None, {
        'pnp_match_method': backend,
        'pnp_matches': 0,
        'pnp_match_reason': 'external_matcher_not_wired' if installed else f'{backend}_not_installed',
    }


def match_render_to_target_points(render_depth, render_mask, target_mask, object_point_map, target_image=None):
    matcher = str(RENDER_PNP_MATCHER).lower()
    if matcher == 'mast3r':
        return match_render_to_target_mast3r(
            render_depth=render_depth,
            render_mask=render_mask,
            target_mask=target_mask,
            object_point_map=object_point_map,
            target_image=target_image,
        )
    if matcher in ('lightglue', 'loftr'):
        return match_render_to_target_external_stub(matcher)

    if matcher in ('auto', 'sift', 'opencv'):
        object_points, image_points, metrics = match_render_to_target_opencv_features(
            render_depth=render_depth,
            render_mask=render_mask,
            target_mask=target_mask,
            object_point_map=object_point_map,
            target_image=target_image,
            backend='sift',
        )
        if object_points is not None and image_points is not None:
            return object_points, image_points, metrics
        if matcher in ('sift', 'opencv'):
            return object_points, image_points, metrics

    if matcher in ('auto', 'akaze'):
        object_points, image_points, metrics = match_render_to_target_opencv_features(
            render_depth=render_depth,
            render_mask=render_mask,
            target_mask=target_mask,
            object_point_map=object_point_map,
            target_image=target_image,
            backend='akaze',
        )
        if object_points is not None and image_points is not None:
            return object_points, image_points, metrics
        if matcher == 'akaze':
            return object_points, image_points, metrics

    if matcher in ('auto', 'orb', 'orb_edge'):
        object_points, image_points, metrics = match_render_to_target_orb_edges(
            render_mask=render_mask,
            target_mask=target_mask,
            object_point_map=object_point_map,
        )
        if object_points is not None and image_points is not None:
            return object_points, image_points, metrics
        if matcher in ('orb', 'orb_edge'):
            return object_points, image_points, metrics

    object_points, image_points, metrics = match_render_to_target_boundary(
        render_mask=render_mask,
        target_mask=target_mask,
        object_point_map=object_point_map,
    )
    metrics.setdefault('pnp_match_method', 'boundary_nearest')
    return object_points, image_points, metrics


def solve_render_pnp_from_boundaries(
    object_vertices,
    vertices_cam_initial,
    faces,
    K_render,
    target_mask_render,
    target_image_render=None,
):
    render_depth, render_mask, object_point_map = render_mesh_depth_mask_with_object_points(
        object_vertices=object_vertices,
        vertices_cam=vertices_cam_initial,
        faces=faces,
        K=K_render,
        height=target_mask_render.shape[0],
        width=target_mask_render.shape[1],
    )
    if render_mask is None or object_point_map is None or int(render_mask.sum()) == 0:
        return None

    object_points, image_points, match_metrics = match_render_to_target_points(
        render_depth=render_depth,
        render_mask=render_mask,
        target_mask=target_mask_render,
        object_point_map=object_point_map,
        target_image=target_image_render,
    )
    if object_points is None or image_points is None:
        return None
    if object_points.shape[0] < RENDER_PNP_MIN_MATCHES:
        return None

    K_render = np.asarray(K_render, dtype=np.float64).reshape(3, 3)
    try:
        ok, rvec, tvec, inliers = cv2.solvePnPRansac(
            objectPoints=object_points.astype(np.float32),
            imagePoints=image_points.astype(np.float32),
            cameraMatrix=K_render,
            distCoeffs=None,
            iterationsCount=100,
            reprojectionError=float(RENDER_PNP_MAX_REPROJ_ERROR),
            confidence=0.98,
            flags=cv2.SOLVEPNP_EPNP,
        )
    except cv2.error:
        return None
    if not ok or inliers is None:
        return None

    inliers = inliers.reshape(-1)
    inlier_count = int(inliers.size)
    inlier_ratio = float(inlier_count / max(float(object_points.shape[0]), 1.0))
    if inlier_count < RENDER_PNP_MIN_INLIERS or inlier_ratio < RENDER_PNP_MIN_INLIER_RATIO:
        return None

    inlier_object = object_points[inliers].astype(np.float64)
    inlier_image = image_points[inliers].astype(np.float64)
    if hasattr(cv2, 'solvePnPRefineLM') and inlier_count >= 6:
        try:
            rvec, tvec = cv2.solvePnPRefineLM(
                objectPoints=inlier_object,
                imagePoints=inlier_image,
                cameraMatrix=K_render,
                distCoeffs=None,
                rvec=rvec,
                tvec=tvec,
            )
        except cv2.error:
            pass

    R_cam, _ = cv2.Rodrigues(rvec)
    t_cam = np.asarray(tvec, dtype=np.float64).reshape(3)
    if not (np.all(np.isfinite(R_cam)) and np.all(np.isfinite(t_cam))) or t_cam[2] <= 0.05:
        return None

    projected, _ = cv2.projectPoints(inlier_object, rvec, tvec, K_render, None)
    projected = projected.reshape(-1, 2)
    reproj_error = float(np.median(np.linalg.norm(projected - inlier_image, axis=1)))

    metrics = {
        **match_metrics,
        'pnp_inliers': inlier_count,
        'pnp_inlier_ratio': inlier_ratio,
        'pnp_reproj_error': reproj_error,
        'pnp_initial_render_pixels': int(render_mask.sum()),
    }
    return {'R_cam': R_cam.astype(np.float64), 't_cam': t_cam, 'metrics': metrics}


def robust_extent(points, lower=3.0, upper=97.0):
    points = _as_numpy_array(points)
    if points is None:
        return None
    points = points.reshape(-1, 3).astype(np.float64)
    points = points[np.all(np.isfinite(points), axis=1)]
    if points.shape[0] < 4:
        return None
    lo = np.percentile(points, lower, axis=0)
    hi = np.percentile(points, upper, axis=0)
    return np.maximum(hi - lo, 1e-4)


def overlap_depth_scale_from_metrics(metrics, min_pixels=None, min_ratio=0.40, max_ratio=2.50):
    """LabelAny3D-style global depth scale from rendered/real overlap medians."""
    if min_pixels is None:
        min_pixels = RENDER_ALIGN_MIN_DEPTH_PIXELS
    if int(metrics.get('depth_pixels', 0)) < int(min_pixels):
        return None
    real_median = float(metrics.get('real_depth_median', float('nan')))
    render_median = float(metrics.get('render_depth_median', float('nan')))
    if not (np.isfinite(real_median) and np.isfinite(render_median)):
        return None
    if real_median <= 0.05 or render_median <= 0.05:
        return None
    ratio = real_median / render_median
    if not np.isfinite(ratio) or ratio < float(min_ratio) or ratio > float(max_ratio):
        return None
    if abs(np.log(ratio)) < 0.03:
        return None
    return float(ratio)


def robust_pose_bbox_from_object_vertices(object_vertices, R_cam, t_cam, min_dims, max_dims):
    """
    Estimate a cuboid from aligned mesh points, similar to LabelAny3D's
    aligned-mesh-sample -> robust OBB step.
    """
    object_vertices = _as_numpy_array(object_vertices)
    if object_vertices is None:
        return None
    object_vertices = np.asarray(object_vertices).reshape(-1, 3).astype(np.float64)
    object_vertices = object_vertices[np.all(np.isfinite(object_vertices), axis=1)]
    if object_vertices.shape[0] < 8:
        return None

    lo = np.percentile(object_vertices, 2.0, axis=0)
    hi = np.percentile(object_vertices, 98.0, axis=0)
    dims_xyz = np.maximum(hi - lo, 1e-4)
    if np.any(dims_xyz < min_dims) or np.any(dims_xyz > max_dims):
        return None

    local_center = (lo + hi) / 2.0
    R_cam = np.asarray(R_cam, dtype=np.float64).reshape(3, 3)
    t_cam = np.asarray(t_cam, dtype=np.float64).reshape(3)
    center_cam = local_center @ R_cam.T + t_cam
    vertices = convert_box_vertices_pose(center_cam, dims_xyz, R_cam).astype(np.float32)
    return center_cam, dims_xyz, vertices


def convert_detection_box_to_xyxy_pixels(box, width, height):
    """GroundingDINO gives normalized cxcywh; tolerate already-pixel boxes too."""
    vals = _as_numpy_array(box).reshape(-1).astype(np.float64).tolist()
    if len(vals) != 4:
        return None
    if max(abs(v) for v in vals) <= 2.0:
        cx, cy, bw, bh = vals
        x1 = (cx - bw / 2.0) * width
        y1 = (cy - bh / 2.0) * height
        x2 = (cx + bw / 2.0) * width
        y2 = (cy + bh / 2.0) * height
    else:
        x1, y1, x2, y2 = vals
        if x2 <= x1 or y2 <= y1:
            x2 = x1 + max(0.0, x2)
            y2 = y1 + max(0.0, y2)
    return [
        min(max(float(x1), 0.0), float(width - 1)),
        min(max(float(y1), 0.0), float(height - 1)),
        min(max(float(x2), 0.0), float(width - 1)),
        min(max(float(y2), 0.0), float(height - 1)),
    ]


def mask_to_xyxy_pixels(mask):
    mask = _as_numpy_array(mask)
    if mask is None:
        return None
    if mask.ndim == 3:
        mask = np.squeeze(mask)
    ys, xs = np.where(mask > 0.5)
    if xs.size == 0 or ys.size == 0:
        return None
    return [float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())]


def box_area_xyxy(box):
    return max(0.0, float(box[2] - box[0])) * max(0.0, float(box[3] - box[1]))


def box_iou_xyxy(box_a, box_b):
    x1 = max(float(box_a[0]), float(box_b[0]))
    y1 = max(float(box_a[1]), float(box_b[1]))
    x2 = min(float(box_a[2]), float(box_b[2]))
    y2 = min(float(box_a[3]), float(box_b[3]))
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = box_area_xyxy(box_a) + box_area_xyxy(box_b) - inter
    return inter / max(union, 1e-9)


def mask_iou_binary(mask_a, mask_b):
    if mask_a is None or mask_b is None:
        return 0.0
    mask_a = np.asarray(mask_a).astype(bool)
    mask_b = np.asarray(mask_b).astype(bool)
    if mask_a.shape != mask_b.shape:
        return 0.0
    inter = np.logical_and(mask_a, mask_b).sum()
    union = np.logical_or(mask_a, mask_b).sum()
    return float(inter / max(float(union), 1.0))


def compute_render_alignment_metrics(render_depth, render_mask, target_mask, real_depth, target_box=None):
    if render_mask is None or target_mask is None:
        return {'silhouette_iou': 0.0, 'bbox_iou': 0.0, 'depth_pixels': 0, 'rel_depth_error': float('inf')}

    render_mask = np.asarray(render_mask).astype(bool)
    target_mask = np.asarray(target_mask).astype(bool)
    silhouette_iou = mask_iou_binary(render_mask, target_mask)

    render_box = mask_to_xyxy_pixels(render_mask)
    mask_box = mask_to_xyxy_pixels(target_mask)
    bbox_targets = [b for b in [target_box, mask_box] if b is not None and b[2] > b[0] and b[3] > b[1]]
    bbox_iou = max([box_iou_xyxy(render_box, b) for b in bbox_targets], default=0.0) if render_box is not None else 0.0

    depth_pixels = 0
    rel_depth_error = float('inf')
    depth_bias = float('inf')
    real_depth_median = float('nan')
    render_depth_median = float('nan')
    if real_depth is not None and render_depth is not None:
        real_depth = np.asarray(real_depth).astype(np.float32)
        render_depth = np.asarray(render_depth).astype(np.float32)
        if real_depth.shape == render_depth.shape:
            valid = (
                render_mask &
                target_mask &
                np.isfinite(real_depth) &
                np.isfinite(render_depth) &
                (real_depth > 0.05) &
                (render_depth > 0.05)
            )
            depth_pixels = int(valid.sum())
            if depth_pixels > 0:
                real_vals = real_depth[valid].astype(np.float64)
                render_vals = render_depth[valid].astype(np.float64)
                rel = (render_vals - real_vals) / np.maximum(real_vals, 1e-3)
                rel_depth_error = float(np.median(np.abs(rel)))
                depth_bias = float(np.median(rel))
                real_depth_median = float(np.median(real_vals))
                render_depth_median = float(np.median(render_vals))

    return {
        'silhouette_iou': float(silhouette_iou),
        'bbox_iou': float(bbox_iou),
        'depth_pixels': int(depth_pixels),
        'rel_depth_error': float(rel_depth_error),
        'depth_bias': float(depth_bias),
        'real_depth_median': float(real_depth_median),
        'render_depth_median': float(render_depth_median),
        'render_pixels': int(render_mask.sum()),
        'target_pixels': int(target_mask.sum()),
    }


def mask_centroid_pixels(mask):
    mask = np.asarray(mask).astype(bool)
    if mask.ndim != 2 or int(mask.sum()) == 0:
        return None
    ys, xs = np.where(mask)
    if xs.size == 0:
        return None
    return np.array([float(xs.mean()), float(ys.mean())], dtype=np.float64)


def silhouette_area_scale_from_metrics(metrics):
    render_pixels = int(metrics.get('render_pixels', 0))
    target_pixels = int(metrics.get('target_pixels', 0))
    if render_pixels < int(RENDER_SILHOUETTE_MIN_RENDER_PIXELS) or target_pixels <= 0:
        return None
    ratio = float(target_pixels) / max(float(render_pixels), 1.0)
    if not np.isfinite(ratio) or ratio <= 0:
        return None
    area_scale = float(np.sqrt(ratio))
    area_scale = float(np.clip(
        area_scale,
        float(RENDER_SILHOUETTE_AREA_SCALE_MIN),
        float(RENDER_SILHOUETTE_AREA_SCALE_MAX),
    ))
    if abs(np.log(area_scale)) < 0.03:
        return None
    return area_scale


def silhouette_centroid_translation(render_mask, target_mask, K_render, depth_z):
    render_ctr = mask_centroid_pixels(render_mask)
    target_ctr = mask_centroid_pixels(target_mask)
    if render_ctr is None or target_ctr is None:
        return None, None

    delta_px = target_ctr - render_ctr
    pixel_norm = float(np.linalg.norm(delta_px))
    if pixel_norm < float(RENDER_SILHOUETTE_MIN_SHIFT_PIXELS):
        return None, None

    K_render = np.asarray(K_render, dtype=np.float64).reshape(3, 3)
    fx = max(abs(float(K_render[0, 0])), 1e-6)
    fy = max(abs(float(K_render[1, 1])), 1e-6)
    z = max(float(depth_z), 1e-3)
    delta_cam = np.array([
        float(delta_px[0]) * z / fx,
        float(delta_px[1]) * z / fy,
        0.0,
    ], dtype=np.float64)

    shift_norm = float(np.linalg.norm(delta_cam[:2]) / max(z, 1e-3))
    max_shift_norm = float(RENDER_SILHOUETTE_MAX_SHIFT_NORM)
    if shift_norm > max_shift_norm:
        delta_cam *= max_shift_norm / max(shift_norm, 1e-8)
        shift_norm = max_shift_norm

    return delta_cam, {
        'silhouette_shift_px': delta_px.astype(float).tolist(),
        'silhouette_shift_px_norm': pixel_norm,
        'silhouette_shift_norm': shift_norm,
    }


def render_alignment_score(metrics):
    silhouette_iou = float(metrics.get('silhouette_iou', 0.0))
    bbox_iou = float(metrics.get('bbox_iou', 0.0))
    rel_depth_error = float(metrics.get('rel_depth_error', float('inf')))
    depth_pixels = int(metrics.get('depth_pixels', 0))
    depth_term = 0.0
    if depth_pixels >= RENDER_ALIGN_MIN_DEPTH_PIXELS and np.isfinite(rel_depth_error):
        depth_term = max(0.0, 1.0 - rel_depth_error / RENDER_ALIGN_MAX_REL_DEPTH_ERROR)
    return float(0.60 * silhouette_iou + 0.25 * bbox_iou + 0.15 * depth_term)


def passes_render_alignment_gate(metrics):
    silhouette_iou = float(metrics.get('silhouette_iou', 0.0))
    bbox_iou = float(metrics.get('bbox_iou', 0.0))
    rel_depth_error = float(metrics.get('rel_depth_error', float('inf')))
    depth_pixels = int(metrics.get('depth_pixels', 0))

    if silhouette_iou < RENDER_ALIGN_MIN_SILHOUETTE_IOU and bbox_iou < RENDER_ALIGN_MIN_BBOX_IOU:
        return False
    if depth_pixels >= RENDER_ALIGN_MIN_DEPTH_PIXELS:
        if not np.isfinite(rel_depth_error) or rel_depth_error > RENDER_ALIGN_MAX_REL_DEPTH_ERROR:
            return False
    elif silhouette_iou < max(RENDER_ALIGN_MIN_SILHOUETTE_IOU * 1.5, 0.16):
        return False
    return True


def passes_strict_render_alignment_gate(metrics, projection_iou, point_support):
    silhouette_iou = float(metrics.get('silhouette_iou', 0.0))
    rel_depth_error = float(metrics.get('rel_depth_error', float('inf')))
    depth_pixels = int(metrics.get('depth_pixels', 0))
    if float(projection_iou) < float(RENDER_ALIGN_STRICT_MIN_PROJECTION_IOU):
        return False
    if silhouette_iou < float(RENDER_ALIGN_STRICT_MIN_SILHOUETTE_IOU):
        return False
    if float(point_support) < float(PSEUDO3D_MIN_POINT_SUPPORT):
        return False
    if depth_pixels < int(RENDER_ALIGN_MIN_DEPTH_PIXELS):
        return False
    if not np.isfinite(rel_depth_error) or rel_depth_error > float(RENDER_ALIGN_STRICT_MAX_REL_DEPTH_ERROR):
        return False
    return True


def depth_refined_scale_from_render(metrics, center_z, current_scale):
    depth_pixels = int(metrics.get('depth_pixels', 0))
    if depth_pixels < RENDER_ALIGN_MIN_DEPTH_PIXELS:
        return None
    real_median = float(metrics.get('real_depth_median', float('nan')))
    render_median = float(metrics.get('render_depth_median', float('nan')))
    if not (np.isfinite(real_median) and np.isfinite(render_median)):
        return None

    numerator = real_median - float(center_z)
    denominator = render_median - float(center_z)
    if abs(denominator) < 0.03:
        return None
    ratio = numerator / denominator
    if not np.isfinite(ratio) or ratio < 0.50 or ratio > 2.00:
        return None
    refined = float(current_scale) * float(ratio)
    if not np.isfinite(refined) or refined <= 1e-5:
        return None
    if abs(np.log(refined / max(float(current_scale), 1e-8))) < 0.03:
        return None
    return refined


def project_vertices_to_box(vertices, K, width, height):
    pts = _as_numpy_array(vertices).reshape(-1, 3).astype(np.float64)
    if pts.shape != (8, 3) or not np.all(np.isfinite(pts)):
        return None
    if np.min(pts[:, 2]) <= 0.05:
        return None
    proj = (np.asarray(K, dtype=np.float64).reshape(3, 3) @ pts.T).T
    if np.any(np.abs(proj[:, 2]) < 1e-6):
        return None
    xy = proj[:, :2] / proj[:, 2:3]
    if not np.all(np.isfinite(xy)):
        return None
    raw = [float(xy[:, 0].min()), float(xy[:, 1].min()), float(xy[:, 0].max()), float(xy[:, 1].max())]
    clipped = [
        min(max(raw[0], 0.0), float(width - 1)),
        min(max(raw[1], 0.0), float(height - 1)),
        min(max(raw[2], 0.0), float(width - 1)),
        min(max(raw[3], 0.0), float(height - 1)),
    ]
    if clipped[2] <= clipped[0] or clipped[3] <= clipped[1]:
        return None
    return raw, clipped


def projection_alignment_score(projected_box, target_box):
    iou = box_iou_xyxy(projected_box, target_box)
    proj_ctr = np.array([(projected_box[0] + projected_box[2]) / 2.0,
                         (projected_box[1] + projected_box[3]) / 2.0])
    target_ctr = np.array([(target_box[0] + target_box[2]) / 2.0,
                           (target_box[1] + target_box[3]) / 2.0])
    target_w = max(float(target_box[2] - target_box[0]), 1e-6)
    target_h = max(float(target_box[3] - target_box[1]), 1e-6)
    target_diag = max((target_w * target_w + target_h * target_h) ** 0.5, 1e-6)
    center_norm = float(np.linalg.norm(proj_ctr - target_ctr) / target_diag)
    area_ratio = box_area_xyxy(projected_box) / max(box_area_xyxy(target_box), 1e-9)
    area_penalty = abs(np.log(max(area_ratio, 1e-6)))
    return iou - 0.10 * center_norm - 0.02 * area_penalty, iou, center_norm, area_ratio


def passes_pseudo3d_projection_gate(iou, center_norm, area_ratio, point_support=None):
    vals = [iou, center_norm, area_ratio]
    if point_support is not None:
        vals.append(point_support)
    if not all(np.isfinite(float(v)) for v in vals):
        return False
    if iou < PSEUDO3D_MIN_PROJECTION_IOU:
        return False
    if center_norm > PSEUDO3D_MAX_CENTER_NORM:
        return False
    if not (PSEUDO3D_MIN_AREA_RATIO <= area_ratio <= PSEUDO3D_MAX_AREA_RATIO):
        return False
    if point_support is not None and point_support < PSEUDO3D_MIN_POINT_SUPPORT:
        return False
    return True


def box_point_support_ratio(scene_points, center, dims_xyz, yaw):
    points = sanitize_scene_points(scene_points)
    if points is None or points.shape[0] == 0:
        return 0.0
    center = np.asarray(center, dtype=np.float64).reshape(3)
    dims_xyz = np.maximum(np.asarray(dims_xyz, dtype=np.float64).reshape(3), 1e-6)
    local = (points - center) @ rotate_y_cubercnn(yaw)
    half = dims_xyz / 2.0
    inside = np.all(np.abs(local) <= half[None, :] * 1.05, axis=1)
    return float(np.mean(inside))


def box_point_support_ratio_pose(scene_points, center, dims_xyz, R_cam):
    points = sanitize_scene_points(scene_points)
    if points is None or points.shape[0] == 0:
        return 0.0
    center = np.asarray(center, dtype=np.float64).reshape(3)
    dims_xyz = np.maximum(np.asarray(dims_xyz, dtype=np.float64).reshape(3), 1e-6)
    R_cam = np.asarray(R_cam, dtype=np.float64).reshape(3, 3)
    local = (points - center) @ R_cam
    half = dims_xyz / 2.0
    inside = np.all(np.abs(local) <= half[None, :] * 1.05, axis=1)
    return float(np.mean(inside))


def unique_yaw_candidates(candidates):
    unique = []
    for yaw in candidates:
        if yaw is None or not np.isfinite(yaw):
            continue
        yaw = normalize_yaw(yaw)
        if all(abs(normalize_yaw(yaw - old)) > np.deg2rad(5.0) for old in unique):
            unique.append(yaw)
    return unique


def expand_yaw_candidates_with_render_offsets(candidates):
    base_yaws = unique_yaw_candidates(candidates)
    offsets = [np.deg2rad(float(v)) for v in RENDER_YAW_OFFSETS_DEG]
    expanded = []
    for yaw in base_yaws:
        for offset in offsets:
            expanded.append(yaw + offset)
    return unique_yaw_candidates(expanded)


def align_fastsam3d_mesh_to_scene_points(
    scene_points,
    mesh_vertices,
    prior_xyz,
    K,
    image_shape,
    detection_box,
    mask,
    fastsam_rotation=None,
):
    """
    Put Fast-SAM3D's relative shape into the real camera frame.

    The metric anchor is always the UniDepth mask point cloud. Fast-SAM3D only
    contributes relative shape and yaw candidates; the final cuboid must project
    back onto the 2D detection/mask before it is accepted.
    """
    H, W = image_shape[:2]
    if K is None:
        return None, {'reason': 'missing_K'}

    scene_points = sanitize_scene_points(scene_points)
    if scene_points is None or scene_points.shape[0] < 20:
        return None, {'reason': 'few_scene_points', 'num_points': 0 if scene_points is None else scene_points.shape[0]}

    center = robust_pointcloud_center(scene_points)
    if center is None or center[2] <= 0.05:
        return None, {'reason': 'bad_center'}

    prior_xyz = np.asarray(prior_xyz if prior_xyz is not None else [0.5, 0.5, 0.5], dtype=np.float64)
    prior_xyz = np.maximum(prior_xyz.reshape(3), 1e-3)

    det_box = convert_detection_box_to_xyxy_pixels(detection_box, W, H)
    mask_box = mask_to_xyxy_pixels(mask)
    target_boxes = [b for b in [mask_box, det_box] if b is not None and b[2] > b[0] and b[3] > b[1]]
    if not target_boxes:
        return None, {'reason': 'bad_2d_target'}

    mesh_extent = robust_extent(mesh_vertices, lower=1.0, upper=99.0)
    if mesh_extent is not None:
        mesh_extent = np.maximum(mesh_extent, 1e-4)

    scene_yaw = estimate_box_yaw_from_scene_points(scene_points)
    fastsam_yaw = yaw_from_fastsam_rotation(fastsam_rotation)
    yaw_candidates = expand_yaw_candidates_with_render_offsets([
        fastsam_yaw,
        scene_yaw,
        0.0,
        np.pi / 2.0,
        None if scene_yaw is None else scene_yaw + np.pi / 2.0,
        None if fastsam_yaw is None else fastsam_yaw + np.pi / 2.0,
    ])

    best = None
    for yaw in yaw_candidates:
        R_yaw = rotate_y_cubercnn(yaw)
        local_scene = (scene_points - center) @ R_yaw
        scene_extent = robust_extent(local_scene, lower=3.0, upper=97.0)
        if scene_extent is None:
            continue

        scene_extent = np.maximum(scene_extent, prior_xyz * np.array([0.20, 0.15, 0.20]))

        if mesh_extent is not None:
            valid_axes = mesh_extent > 1e-4
            sim3_scale = float(np.median(scene_extent[valid_axes] / mesh_extent[valid_axes]))
            mesh_dims = mesh_extent * max(sim3_scale, 1e-4)
            # Mesh supplies shape; scene points supply metric scale. Blend lightly
            # toward the real point cloud so partially reconstructed meshes do not
            # collapse an axis.
            dims_xyz = 0.70 * mesh_dims + 0.30 * scene_extent
        else:
            sim3_scale = None
            dims_xyz = scene_extent

        min_dims, max_dims = category_prior_dimension_bounds(prior_xyz)
        dims_xyz = np.clip(dims_xyz, min_dims, max_dims)

        dx, dy, dz = [float(v) for v in dims_xyz]
        vertices = convert_box_vertices(center[0], center[1], center[2], dx, dy, dz, yaw).astype(np.float32)
        projection = project_vertices_to_box(vertices, K, W, H)
        if projection is None:
            continue

        _, clipped_box = projection
        target_scores = [projection_alignment_score(clipped_box, target) for target in target_boxes]
        score, iou, center_norm, area_ratio = max(target_scores, key=lambda item: item[0])
        point_support = box_point_support_ratio(scene_points, center, dims_xyz, yaw)

        passes_projection = passes_pseudo3d_projection_gate(
            iou=iou,
            center_norm=center_norm,
            area_ratio=area_ratio,
            point_support=point_support,
        )

        candidate = {
            'score': float(score),
            'passes_projection': bool(passes_projection),
            'bbox': {
                'center_cam': center.astype(np.float32).tolist(),
                'dimensions': [float(dx), float(dy), float(dz)],
                'R_cam': R_yaw.astype(np.float32).tolist(),
                'yaw': float(yaw),
                'vertices': vertices.astype(np.float32).tolist(),
            },
            'metrics': {
                'iou': float(iou),
                'center_norm': float(center_norm),
                'area_ratio': float(area_ratio),
                'point_support': float(point_support),
                'sim3_scale': None if sim3_scale is None else float(sim3_scale),
                'scene_extent_xyz': scene_extent.astype(float).tolist(),
                'mesh_extent_xyz': None if mesh_extent is None else mesh_extent.astype(float).tolist(),
                'prior_min_dims_xyz': min_dims.astype(float).tolist(),
                'prior_max_dims_xyz': max_dims.astype(float).tolist(),
            },
        }

        if best is None or candidate['score'] > best['score']:
            best = candidate

    if best is None:
        return None, {'reason': 'no_projectable_candidate', 'num_points': int(scene_points.shape[0])}
    if not best['passes_projection']:
        return None, {'reason': 'projection_gate', **best['metrics']}
    return best['bbox'], {'reason': 'valid', **best['metrics']}


def align_fastsam3d_mesh_with_render_back(
    scene_points,
    mesh_vertices,
    mesh_faces,
    prior_xyz,
    K,
    image_shape,
    detection_box,
    mask,
    real_depth,
    image_rgb=None,
    fastsam_rotation=None,
):
    """
    Reconstruction-mainline alignment:

    RGB/mask/K/metric depth gives a real camera-space point cloud anchor. The
    Fast-SAM3D mesh provides only relative shape. We try a compact set of
    yaw/scale hypotheses, render the mesh back into the input image, and accept
    a 3D box only if the rendered silhouette/depth closes the loop with the
    2D mask and metric depth.
    """
    H, W = image_shape[:2]
    if K is None:
        return None, {'reason': 'missing_K'}

    scene_points = sanitize_scene_points(scene_points)
    if scene_points is None or scene_points.shape[0] < 20:
        return None, {'reason': 'few_scene_points', 'num_points': 0 if scene_points is None else scene_points.shape[0]}

    center = robust_pointcloud_center(scene_points)
    if center is None or center[2] <= 0.05:
        return None, {'reason': 'bad_center'}

    mesh_vertices, mesh_faces = _clean_mesh_geometry(mesh_vertices, mesh_faces)
    if mesh_vertices is None:
        return None, {'reason': 'missing_mesh_vertices'}
    if mesh_faces is None:
        return None, {'reason': 'missing_render_faces', 'num_vertices': int(mesh_vertices.shape[0])}

    local_mesh, _, mesh_extent = canonicalize_mesh_vertices(mesh_vertices)
    if local_mesh is None or mesh_extent is None:
        return None, {'reason': 'bad_mesh_extent'}
    mesh_extent = np.maximum(mesh_extent, 1e-4)

    prior_xyz = np.asarray(prior_xyz if prior_xyz is not None else [0.5, 0.5, 0.5], dtype=np.float64)
    prior_xyz = np.maximum(prior_xyz.reshape(3), 1e-3)

    det_box = convert_detection_box_to_xyxy_pixels(detection_box, W, H)
    mask_box = mask_to_xyxy_pixels(mask)
    target_boxes = [b for b in [mask_box, det_box] if b is not None and b[2] > b[0] and b[3] > b[1]]
    if not target_boxes:
        return None, {'reason': 'bad_2d_target'}

    target_mask_render, real_depth_render, Hr, Wr = prepare_render_reference(
        mask=mask,
        real_depth=real_depth,
        image_shape=image_shape,
        render_scale=RENDER_ALIGN_SCALE,
    )
    if target_mask_render is None or int(target_mask_render.sum()) == 0:
        return None, {'reason': 'bad_render_target_mask'}

    target_image_render = None
    if image_rgb is not None:
        target_image = _as_numpy_array(image_rgb)
        if target_image is not None and target_image.ndim == 3:
            target_image_render = cv2.resize(target_image.astype(np.uint8), (Wr, Hr), interpolation=cv2.INTER_AREA)

    sx = float(Wr) / float(max(W, 1))
    sy = float(Hr) / float(max(H, 1))
    K_render = scale_camera_matrix(K, sx, sy)
    det_box_render = None
    if det_box is not None:
        det_box_render = [
            float(det_box[0]) * sx,
            float(det_box[1]) * sy,
            float(det_box[2]) * sx,
            float(det_box[3]) * sy,
        ]

    scene_yaw = estimate_box_yaw_from_scene_points(scene_points)
    fastsam_yaw = yaw_from_fastsam_rotation(fastsam_rotation)
    yaw_candidates = expand_yaw_candidates_with_render_offsets([
        fastsam_yaw,
        scene_yaw,
        0.0,
        np.pi / 2.0,
        None if scene_yaw is None else scene_yaw + np.pi / 2.0,
        None if fastsam_yaw is None else fastsam_yaw + np.pi / 2.0,
    ])

    min_dims, max_dims = category_prior_dimension_bounds(prior_xyz)

    best = None
    evaluated = 0

    def add_scale_candidate(values, scale_value):
        if scale_value is None or not np.isfinite(scale_value) or scale_value <= 1e-5:
            return
        for old in values:
            if abs(np.log(float(scale_value) / max(float(old), 1e-8))) < 0.04:
                return
        values.append(float(scale_value))

    def choose_better_candidate(base_candidate, new_candidate):
        if new_candidate is None:
            return base_candidate
        if base_candidate is None:
            return new_candidate
        if new_candidate['passes_all'] and not base_candidate['passes_all']:
            return new_candidate
        if new_candidate['passes_all'] == base_candidate['passes_all'] and new_candidate['score'] > base_candidate['score']:
            return new_candidate
        return base_candidate

    def evaluate_pose_candidate(R_cam, t_cam, scale, source, extra_metrics=None, allow_silhouette_refine=True):
        nonlocal evaluated
        scale = float(scale)
        rough_dims_xyz = mesh_extent * scale
        if np.any(rough_dims_xyz < min_dims) or np.any(rough_dims_xyz > max_dims):
            return None

        R_cam = np.asarray(R_cam, dtype=np.float64).reshape(3, 3)
        t_cam = np.asarray(t_cam, dtype=np.float64).reshape(3)
        if not (np.all(np.isfinite(R_cam)) and np.all(np.isfinite(t_cam))) or t_cam[2] <= 0.05:
            return None

        object_vertices = local_mesh * scale
        vertices_cam = object_vertices @ R_cam.T + t_cam[None, :]
        render_depth, render_mask = render_mesh_depth_mask(
            vertices_cam=vertices_cam,
            faces=mesh_faces,
            K=K_render,
            height=Hr,
            width=Wr,
        )
        evaluated += 1
        if render_mask is None or int(render_mask.sum()) == 0:
            return None

        bbox_center = t_cam
        dims_xyz = rough_dims_xyz
        box_vertices = convert_box_vertices_pose(bbox_center, dims_xyz, R_cam).astype(np.float32)
        if RENDER_USE_ALIGNED_MESH_OBB:
            mesh_bbox = robust_pose_bbox_from_object_vertices(
                object_vertices=object_vertices,
                R_cam=R_cam,
                t_cam=t_cam,
                min_dims=min_dims,
                max_dims=max_dims,
            )
            if mesh_bbox is not None:
                bbox_center, dims_xyz, box_vertices = mesh_bbox

        metrics = compute_render_alignment_metrics(
            render_depth=render_depth,
            render_mask=render_mask,
            target_mask=target_mask_render,
            real_depth=real_depth_render,
            target_box=det_box_render,
        )

        projection = project_vertices_to_box(box_vertices, K, W, H)
        if projection is None:
            return None

        _, clipped_box = projection
        target_scores = [projection_alignment_score(clipped_box, target) for target in target_boxes]
        proj_score, proj_iou, proj_center_norm, proj_area_ratio = max(target_scores, key=lambda item: item[0])
        point_support = box_point_support_ratio_pose(scene_points, bbox_center, dims_xyz, R_cam)
        passes_projection = passes_pseudo3d_projection_gate(
            iou=proj_iou,
            center_norm=proj_center_norm,
            area_ratio=proj_area_ratio,
            point_support=point_support,
        )
        passes_render = passes_render_alignment_gate(metrics)
        passes_strict = passes_strict_render_alignment_gate(
            metrics=metrics,
            projection_iou=proj_iou,
            point_support=point_support,
        )

        render_score = render_alignment_score(metrics)
        pnp_inlier_ratio = 0.0
        if extra_metrics is not None:
            pnp_inlier_ratio = float(extra_metrics.get('pnp_inlier_ratio', 0.0))
        score = (
            render_score +
            0.20 * float(proj_iou) +
            0.10 * float(point_support) -
            0.05 * float(proj_center_norm) -
            0.02 * abs(np.log(max(float(proj_area_ratio), 1e-6))) +
            0.03 * pnp_inlier_ratio
        )

        yaw = yaw_from_rotation_matrix(R_cam)
        metrics_out = {
            **metrics,
            'projection_iou': float(proj_iou),
            'projection_center_norm': float(proj_center_norm),
            'projection_area_ratio': float(proj_area_ratio),
            'point_support': float(point_support),
            'scale': float(scale),
            'yaw': float(yaw),
            'scale_source': source,
            'pnp_t_cam': t_cam.astype(float).tolist(),
            'bbox_center_shift_norm': float(np.linalg.norm(bbox_center - t_cam) / max(float(t_cam[2]), 1e-3)),
            'aligned_mesh_obb': bool(RENDER_USE_ALIGNED_MESH_OBB),
            'passes_projection_gate': bool(passes_projection),
            'passes_render_gate': bool(passes_render),
            'passes_strict_render_back': bool(passes_strict),
            'quality_tier': 'render_back_strict' if passes_strict else 'render_back_loose',
            'mesh_extent_xyz': mesh_extent.astype(float).tolist(),
            'dims_xyz': dims_xyz.astype(float).tolist(),
            'prior_min_dims_xyz': min_dims.astype(float).tolist(),
            'prior_max_dims_xyz': max_dims.astype(float).tolist(),
        }
        if extra_metrics is not None:
            metrics_out.update(extra_metrics)

        dx, dy, dz = [float(v) for v in np.asarray(dims_xyz, dtype=np.float64).reshape(3)]
        result = {
            'score': float(score),
            'passes_all': bool(passes_projection and passes_render and (passes_strict or not RENDER_ALIGN_REQUIRE_STRICT)),
            'bbox': {
                'center_cam': bbox_center.astype(np.float32).tolist(),
                'dimensions': [float(dx), float(dy), float(dz)],
                'R_cam': R_cam.astype(np.float32).tolist(),
                'yaw': float(yaw),
                'vertices': box_vertices.astype(np.float32).tolist(),
            },
            'metrics': metrics_out,
        }

        if allow_silhouette_refine and bool(RENDER_SILHOUETTE_REFINE):
            base_extra = dict(extra_metrics or {})
            area_scale = silhouette_area_scale_from_metrics(metrics)
            shift_delta, shift_metrics = silhouette_centroid_translation(
                render_mask=render_mask,
                target_mask=target_mask_render,
                K_render=K_render,
                depth_z=t_cam[2],
            )

            refined_best = result
            if area_scale is not None:
                refined_best = choose_better_candidate(
                    refined_best,
                    evaluate_pose_candidate(
                        R_cam=R_cam,
                        t_cam=t_cam,
                        scale=scale * area_scale,
                        source=f'{source}_silhouette_area',
                        extra_metrics={
                            **base_extra,
                            'silhouette_refine': 'area',
                            'silhouette_area_scale': float(area_scale),
                            'silhouette_refine_from_score': float(result['score']),
                            'silhouette_refine_from_iou': float(metrics.get('silhouette_iou', 0.0)),
                        },
                        allow_silhouette_refine=False,
                    ),
                )

            if shift_delta is not None:
                refined_best = choose_better_candidate(
                    refined_best,
                    evaluate_pose_candidate(
                        R_cam=R_cam,
                        t_cam=t_cam + shift_delta,
                        scale=scale,
                        source=f'{source}_silhouette_shift',
                        extra_metrics={
                            **base_extra,
                            'silhouette_refine': 'shift',
                            **shift_metrics,
                            'silhouette_refine_from_score': float(result['score']),
                            'silhouette_refine_from_iou': float(metrics.get('silhouette_iou', 0.0)),
                        },
                        allow_silhouette_refine=False,
                    ),
                )

            if area_scale is not None and shift_delta is not None:
                refined_best = choose_better_candidate(
                    refined_best,
                    evaluate_pose_candidate(
                        R_cam=R_cam,
                        t_cam=t_cam + shift_delta,
                        scale=scale * area_scale,
                        source=f'{source}_silhouette_area_shift',
                        extra_metrics={
                            **base_extra,
                            'silhouette_refine': 'area_shift',
                            'silhouette_area_scale': float(area_scale),
                            **shift_metrics,
                            'silhouette_refine_from_score': float(result['score']),
                            'silhouette_refine_from_iou': float(metrics.get('silhouette_iou', 0.0)),
                        },
                        allow_silhouette_refine=False,
                    ),
                )
            return refined_best

        return result

    def evaluate_candidate(yaw, scale, source):
        return evaluate_pose_candidate(
            R_cam=rotate_y_cubercnn(yaw),
            t_cam=center,
            scale=scale,
            source=source,
            extra_metrics={'pose_source': 'yaw_scale', 'initial_yaw': float(yaw)},
        )

    def should_try_render_pnp(candidate):
        if candidate is None:
            return False
        metrics = candidate.get('metrics', {})
        return (
            float(metrics.get('silhouette_iou', 0.0)) >= 0.03 or
            float(metrics.get('bbox_iou', 0.0)) >= 0.05
        )

    def evaluate_pnp_candidate(yaw, scale, source):
        scale = float(scale)
        dims_xyz = mesh_extent * scale
        if np.any(dims_xyz < min_dims) or np.any(dims_xyz > max_dims):
            return None

        R_init = rotate_y_cubercnn(yaw)
        object_vertices = local_mesh * scale
        vertices_cam_initial = object_vertices @ R_init.T + center[None, :]
        pnp_result = solve_render_pnp_from_boundaries(
            object_vertices=object_vertices,
            vertices_cam_initial=vertices_cam_initial,
            faces=mesh_faces,
            K_render=K_render,
            target_mask_render=target_mask_render,
            target_image_render=target_image_render,
        )
        if pnp_result is None:
            return None

        R_pnp = pnp_result['R_cam']
        t_pnp = pnp_result['t_cam']
        center_shift_norm = float(np.linalg.norm(t_pnp - center) / max(float(center[2]), 1e-3))
        upright_dot = float(abs(np.dot(R_pnp[:, 1], np.array([0.0, 1.0, 0.0], dtype=np.float64))))
        if center_shift_norm > RENDER_PNP_MAX_CENTER_SHIFT_NORM:
            return None
        if upright_dot < RENDER_PNP_MIN_UPRIGHT_DOT:
            return None

        extra_metrics = {
            **pnp_result['metrics'],
            'pose_source': 'render_pnp',
            'initial_yaw': float(yaw),
            'pnp_center_shift_norm': center_shift_norm,
            'pnp_upright_dot': upright_dot,
        }
        base_candidate = evaluate_pose_candidate(
            R_cam=R_pnp,
            t_cam=t_pnp,
            scale=scale,
            source=source,
            extra_metrics=extra_metrics,
        )
        if base_candidate is None:
            return None

        overlap_scale = overlap_depth_scale_from_metrics(base_candidate['metrics'])
        if overlap_scale is None:
            return base_candidate

        scaled_candidate = evaluate_pose_candidate(
            R_cam=R_pnp,
            t_cam=t_pnp * overlap_scale,
            scale=scale * overlap_scale,
            source=f'{source}_overlap_depth_scaled',
            extra_metrics={
                **extra_metrics,
                'overlap_depth_scale': float(overlap_scale),
                'overlap_depth_scaled_from': source,
            },
        )
        if scaled_candidate is None:
            return base_candidate
        if scaled_candidate['passes_all'] and not base_candidate['passes_all']:
            return scaled_candidate
        if scaled_candidate['passes_all'] == base_candidate['passes_all'] and scaled_candidate['score'] > base_candidate['score']:
            return scaled_candidate
        return base_candidate

    for yaw in yaw_candidates:
        R_yaw = rotate_y_cubercnn(yaw)
        local_scene = (scene_points - center) @ R_yaw
        scene_extent = robust_extent(local_scene, lower=3.0, upper=97.0)
        if scene_extent is None:
            continue
        scene_extent = np.maximum(scene_extent, prior_xyz * np.array([0.10, 0.10, 0.10]))

        valid_axes = mesh_extent > 1e-4
        sim3_scale = float(np.median(scene_extent[valid_axes] / mesh_extent[valid_axes]))
        prior_scale = float(np.median(prior_xyz[valid_axes] / mesh_extent[valid_axes]))

        scale_candidates = []
        for base in [sim3_scale, prior_scale]:
            for multiplier in RENDER_SCALE_MULTIPLIERS:
                add_scale_candidate(scale_candidates, base * multiplier)

        candidates_for_yaw = []
        for scale in scale_candidates:
            candidate = evaluate_candidate(yaw, scale, 'scene_or_prior')
            if candidate is None:
                continue
            candidates_for_yaw.append(candidate)
            if should_try_render_pnp(candidate):
                pnp_candidate = evaluate_pnp_candidate(yaw, scale, 'scene_or_prior_pnp')
                if pnp_candidate is not None:
                    candidates_for_yaw.append(pnp_candidate)

            refined_scale = depth_refined_scale_from_render(candidate['metrics'], center[2], scale)
            if refined_scale is not None:
                refined_candidate = evaluate_candidate(yaw, refined_scale, 'render_depth_refined')
                if refined_candidate is not None:
                    candidates_for_yaw.append(refined_candidate)
                    if should_try_render_pnp(refined_candidate):
                        refined_pnp_candidate = evaluate_pnp_candidate(
                            yaw,
                            refined_scale,
                            'render_depth_refined_pnp',
                        )
                        if refined_pnp_candidate is not None:
                            candidates_for_yaw.append(refined_pnp_candidate)

        for candidate in candidates_for_yaw:
            if best is None:
                best = candidate
            elif candidate['passes_all'] and not best['passes_all']:
                best = candidate
            elif candidate['passes_all'] == best['passes_all'] and candidate['score'] > best['score']:
                best = candidate

    if best is None:
        return None, {
            'reason': 'no_render_candidate',
            'num_points': int(scene_points.shape[0]),
            'num_faces': int(mesh_faces.shape[0]),
            'evaluated': int(evaluated),
        }
    if not best['passes_all']:
        return None, {'reason': 'render_back_gate', **best['metrics'], 'evaluated': int(evaluated)}
    reason = 'valid_render_back_strict' if best['metrics'].get('passes_strict_render_back') else 'valid_render_back'
    return best['bbox'], {'reason': reason, **best['metrics'], 'evaluated': int(evaluated)}


def compute_pointcloud_center_from_depth(image, mask, intrinsics):
    """
    使用 UniDepth/深度图计算掩码区域的伪点云质心
    
    Args:
        image: RGB 图像 (H, W, 3)
        mask: 二值掩码 (H, W), 值在 [0, 1]
        intrinsics: 相机内参 (3, 3)
    
    Returns:
        centroid: 点云质心 (3,) 或 None
    """
    # 尝试从 Fast-SAM3D 的 scene pointmap 获取物理点云
    # 如果已经有 scene pointmap，直接从中提取
    # 这在 reconstruct_3d 函数中通过 output['pointmap'] 提供
    pass


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
    # 始终使用点云实际边界计算中心
    vertives = convert_box_vertices(cx, cy, cz, dx, dy, dz, 0).astype(np.float16)
    vertives = np.dot(rotate_y(-yaw), vertives.T).T
    vertives = np.dot(vertives, rotation_matrix.T)
    # 从正确旋转后的顶点计算中心（保持一致性）
    center = vertives.mean(axis=0)
    dimensions = [dx, dy, dz]  # Omni3D 标准格式: [width, height, length]
    R_cam = rotation_matrix @ rotate_y(-yaw)
    
    # 如果尺寸太离谱，尝试射线追踪优化，但保持点云中心不变
    if not (l * 0.5 <= dx and w * 0.5 <= dz) and not (l * 0.5 <= dz and w * 0.5 <= dx):
        # 尺寸不合理，只调整尺寸，不改变中心
        possible_bboxs = generate_possible_bboxs(cx, cz, dx, dz, w, l)
        min_loss, best_dims = float('inf'), None
        
        for possible_bbox in possible_bboxs:
            x_min, x_max, z_min, z_max = possible_bbox
            dx_box, dz_box = x_max - x_min, z_max - z_min
            
            inside_ratio = calc_inside_ratio(rotated_pc_2.T, x_min, x_max, z_min, z_max)
            loss_inside_ratio = 1 - inside_ratio
            
            if loss_inside_ratio < min_loss:
                min_loss = loss_inside_ratio
                best_dims = (dz_box, dx_box)
        
        if best_dims is not None:
            dz_box, dx_box = best_dims
            vertives = convert_box_vertices(cx, cy, cz, dx_box, dy, dz_box, 0).astype(np.float16)
            vertives = np.dot(rotate_y(-yaw), vertives.T).T
            vertives = np.dot(vertives, rotation_matrix.T)
            dimensions = [dx_box, dy, dz_box]
            # 从顶点重新计算中心（保持一致性）
            center = vertives.mean(axis=0)
    
    return {
        'center_cam': center.tolist(),
        'dimensions': dimensions,  # [dx, dy, dz] Omni3D 标准格式: [width, height, length]
        'R_cam': R_cam.tolist(),
        'yaw': float(yaw),
        'vertices': vertives.tolist() if isinstance(vertives, np.ndarray) else vertives
    }


def extract_bbox_from_points(points: np.ndarray, prior: list = None, ground_equ: np.ndarray = None) -> dict:
    """
    从点云数组提取 3D 边界框（不依赖PLY文件）
    
    关键特性:
    1. 地面平面检测和约束
    2. PCA 在 XZ 平面估计 yaw
    3. 射线追踪 + inside_ratio 优化（当尺寸不合理时）
    4. 类别先验约束
    
    Args:
        points: Nx3 点云数组（已经是物理单位）
        prior: 类别先验尺寸 [width, height, depth]
        ground_equ: 地面平面方程
    """
    from sklearn.decomposition import PCA
    from cubercnn.generate_label.util import (
        rotation_matrix_from_vectors, rotate_y, convert_box_vertices,
        point_to_plane_distance, generate_possible_bboxs
    )
    from cubercnn.generate_label.raytrace import calc_inside_ratio, calc_dis_ray_tracing
    
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
    # 始终使用点云实际边界计算中心
    vertives = convert_box_vertices(cx, cy, cz, dx, dy, dz, 0).astype(np.float16)
    vertives = np.dot(rotate_y(-yaw), vertives.T).T
    vertives = np.dot(vertives, rotation_matrix.T)
    # 从正确旋转后的顶点计算中心（保持一致性）
    center = vertives.mean(axis=0)
    dimensions = [dx, dy, dz]  # Omni3D 标准格式: [width, height, length]
    R_cam = rotation_matrix @ rotate_y(-yaw)
    
    # 如果尺寸太离谱，尝试射线追踪优化，但保持点云中心不变
    if not (l * 0.5 <= dx and w * 0.5 <= dz) and not (l * 0.5 <= dz and w * 0.5 <= dx):
        # 尺寸不合理，只调整尺寸，不改变中心
        possible_bboxs = generate_possible_bboxs(cx, cz, dx, dz, w, l)
        min_loss, best_dims = float('inf'), None
        
        for possible_bbox in possible_bboxs:
            x_min, x_max, z_min, z_max = possible_bbox
            dx_box, dz_box = x_max - x_min, z_max - z_min
            
            inside_ratio = calc_inside_ratio(rotated_pc_2.T, x_min, x_max, z_min, z_max)
            loss_inside_ratio = 1 - inside_ratio
            
            if loss_inside_ratio < min_loss:
                min_loss = loss_inside_ratio
                best_dims = (dz_box, dx_box)
        
        if best_dims is not None:
            dz_box, dx_box = best_dims
            vertives = convert_box_vertices(cx, cy, cz, dx_box, dy, dz_box, 0).astype(np.float16)
            vertives = np.dot(rotate_y(-yaw), vertives.T).T
            vertives = np.dot(vertives, rotation_matrix.T)
            dimensions = [dx_box, dy, dz_box]
            # 从顶点重新计算中心（保持一致性）
            center = vertives.mean(axis=0)
    
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
    intrinsics: np.ndarray = None,
    alignment_mode: str = 'render_back_then_sim3',
    clear_cuda_cache_per_object: bool = True,
    min_mask_pixels: int = FASTSAM3D_MIN_MASK_PIXELS,
    min_mask_area_ratio: float = FASTSAM3D_MIN_MASK_AREA_RATIO,
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
    depth_dir = os.path.join(scene_output_dir, 'depth')
    os.makedirs(mask_dir, exist_ok=True)
    os.makedirs(mesh_dir, exist_ok=True)
    os.makedirs(depth_dir, exist_ok=True)
    
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

    # One UniDepth prediction per image. Every object below uses this same metric
    # depth map plus the original K, so Line A and Line B share one camera frame.
    depth_independent_image = None
    if intrinsics is not None:
        try:
            print(f"    📷 Line A: Running independent UniDepth once for image...")
            depth_independent_image = run_independent_unidepth(original_image, intrinsics)
            print(
                f"    📷 Line A: Image depth stats: "
                f"mean={depth_independent_image.mean():.3f}m, "
                f"range=[{depth_independent_image.min():.3f}, {depth_independent_image.max():.3f}]"
            )
            np.save(
                os.path.join(depth_dir, f'{image_name}_independent.npy'),
                depth_independent_image.astype(np.float32),
            )
        except Exception as e:
            print(f"    ⚠️  Line A: image-level UniDepth failed: {e}")
    else:
        print(f"    ⚠️  Line A: intrinsics missing, cannot run original-K UniDepth anchoring")
    
    # 保存掩码
    boxes3d_list = []
    center_cam_list = []
    dimensions_list = []
    R_cam_list = []
    bbox2d_list = []
    alignment_info_list = []

    def maybe_cleanup_cuda():
        if clear_cuda_cache_per_object:
            cleanup_cuda_memory()
    
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
        
        if mask_pixels < int(min_mask_pixels) or mask_area_ratio < float(min_mask_area_ratio):
            print(f"    ⚠️  Mask too small ({mask_pixels} pixels, ratio={mask_area_ratio:.4f}), skipping Fast-SAM3D")
            boxes3d_list.append(np.full((8, 3), -1, dtype=np.float16))
            center_cam_list.append(np.array([-1, -1, -1]))
            dimensions_list.append([-1, -1, -1])
            R_cam_list.append(np.eye(3, dtype=np.float32))
            bbox2d_list.append(box.tolist())
            alignment_info_list.append({
                'mode': alignment_mode,
                'reason': 'mask_too_small',
                'mask_pixels': int(mask_pixels),
                'mask_area_ratio': float(mask_area_ratio),
            })
            maybe_cleanup_cuda()
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
            if valid_pixels < int(min_mask_pixels):
                print(f"    ⚠️  Mask too small ({valid_pixels} pixels), skipping Fast-SAM3D")
                boxes3d_list.append(np.full((8, 3), -1, dtype=np.float16))
                center_cam_list.append(np.array([-1, -1, -1]))
                dimensions_list.append([-1, -1, -1])
                R_cam_list.append(np.eye(3, dtype=np.float32))
                bbox2d_list.append(box.tolist())
                alignment_info_list.append({
                    'mode': alignment_mode,
                    'reason': 'eroded_mask_too_small',
                    'mask_pixels': int(valid_pixels),
                })
                maybe_cleanup_cuda()
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
            
            # 获取 scale 和 shift 用于反归一化
            # FastSAM3D 输出中包含 scale/shift 用于 SSI 归一化
            # 注意: 输出键是 'scale' 和 'translation'，不是 'shift'
            scene_scale = output.get('scale', None)
            scene_shift = output.get('translation', output.get('shift', None))
            
            if scene_scale is not None:
                print(f"    📏 Scale available: {scene_scale}")
            if scene_shift is not None:
                print(f"    📏 Translation available: {scene_shift}")
            
            # 保存 Mesh（使用原始归一化坐标，因为需要计算相对位置）
            mesh_filename = f"{label.replace(' ', '_')}_{i}.ply"
            mesh_path = os.path.join(mesh_dir, mesh_filename)
            save_gs_as_ply(output['gs'], mesh_path)
            
            glb_filename = f"{label.replace(' ', '_')}_{i}.glb"
            output['glb'].export(os.path.join(mesh_dir, glb_filename))
            
            # ============ 新流程：伪点云 + 3D重建 融合 ============
            # Line A: UniDepth → 伪点云 → 质心 (显式分离)
            # Line B: FastSAM3D → 3D mesh → 尺寸、朝向
            # ================================================================
            
            print(f"    📦 Fusion Strategy (Two-Line Explicit Separation):")
            print(f"       Line A: 独立 UniDepth → 伪点云 → 质心 (完全不经过 Fast-SAM3D)")
            print(f"       Line B: FastSAM3D mesh → size + yaw")
            
            # 1. 优先使用图像级独立 UniDepth（完全不经过 Fast-SAM3D）
            print(f"    📷 Line A: Using independent UniDepth point cloud...")
            depth_independent = None
            center_from_pc = None  # 初始化变量，避免 UnboundLocalError
            pseudo_lidar = None
            try:
                # 使用同一张图的 UniDepth；若图像级推理失败，再做一次 object-local fallback。
                if depth_independent_image is not None:
                    depth_independent = depth_independent_image
                else:
                    depth_independent = run_independent_unidepth(original_image, intrinsics)
                    depth_save_path = os.path.join(depth_dir, f'{label.replace(" ", "_")}_{i}_independent.npy')
                    np.save(depth_save_path, depth_independent.astype(np.float32))
                    print(f"    💾 Saved independent depth fallback: {depth_save_path}")
                
                # 使用原版流程计算点云质心
                pseudo_lidar, mask_centroid = extract_pointcloud_from_depth(
                    depth_independent, eroded_mask_for_fastsam.astype(np.float32), intrinsics
                )
                
                robust_center = robust_pointcloud_center(pseudo_lidar)
                if robust_center is not None:
                    center_from_pc = robust_center
                    print(f"    ✅ Line A: Center from 独立UniDepth 伪点云: {center_from_pc}")
                    print(f"       (valid points: {len(pseudo_lidar) if pseudo_lidar is not None else 0})")
                elif mask_centroid is not None:
                    center_from_pc = mask_centroid
                    print(f"    ✅ Line A: Center from 独立UniDepth 伪点云(mean fallback): {center_from_pc}")
                else:
                    print(f"    ⚠️  Line A: Failed to extract from 独立depth, trying Fast-SAM3D depth_physical")
            except Exception as e:
                print(f"    ⚠️  Line A: 独立 UniDepth 失败: {e}")
                print(f"    ⚠️  Line A: Falling back to Fast-SAM3D depth_physical...")
            
            # 2. 如果独立 UniDepth 失败，使用 Fast-SAM3D 的 depth_physical
            if center_from_pc is None:
                depth_physical = output.get('depth_physical', None)
                
                if depth_physical is not None:
                    print(f"    📷 Line A (fallback): Using Fast-SAM3D depth_physical")
                    pseudo_lidar, mask_centroid = extract_pointcloud_from_depth(
                        depth_physical, eroded_mask_for_fastsam.astype(np.float32), intrinsics
                    )
                    
                    robust_center = robust_pointcloud_center(pseudo_lidar)
                    if robust_center is not None:
                        center_from_pc = robust_center
                        print(f"    ✅ Line A: Center from Fast-SAM3D 伪点云: {center_from_pc}")
                        print(f"       (valid points: {len(pseudo_lidar) if pseudo_lidar is not None else 0})")
                    elif mask_centroid is not None:
                        center_from_pc = mask_centroid
                        print(f"    ✅ Line A: Center from Fast-SAM3D 伪点云(mean fallback): {center_from_pc}")
                    else:
                        print(f"    ⚠️  Line A: Failed to extract from depth_physical")
                else:
                    print(f"    ⚠️  Line A: No depth_physical available, trying scene pointmap")
            
            # 3. 如果 depth_physical 也失败，尝试从 scene_pointmap 提取
            if center_from_pc is None:
                scene_pointmap = output.get('pointmap', None)  # 场景级点云 (H, W, 3)
                
                if scene_pointmap is not None:
                    print(f"    📷 Line A (fallback 2): Using scene pointmap")
                    mask_points, mask_centroid = extract_mask_pointcloud(
                        scene_pointmap, eroded_mask_for_fastsam.astype(np.float32), intrinsics
                    )
                    
                    robust_center = robust_pointcloud_center(mask_points)
                    if robust_center is not None:
                        pseudo_lidar = mask_points
                        center_from_pc = robust_center
                        print(f"    ✅ Line A: Center from pointmap: {center_from_pc}")
                    elif mask_centroid is not None:
                        pseudo_lidar = mask_points
                        center_from_pc = mask_centroid
                        print(f"    ✅ Line A: Center from pointmap(mean fallback): {center_from_pc}")
                    else:
                        print(f"    ⚠️  Line A: Failed to extract from pointmap")
                else:
                    print(f"    ⚠️  Line A: No scene_pointmap available")
            
            # 如果点云提取失败，回退到从 mesh 点云提取
            if center_from_pc is None:
                # 读取归一化的点云
                from plyfile import PlyData
                plydata = PlyData.read(mesh_path)
                vertex = plydata['vertex']
                xyz_normalized = np.stack([
                    np.array(vertex['x']),
                    np.array(vertex['y']),
                    np.array(vertex['z'])
                ], axis=1)
                
                # 获取 scale 和 shift
                scene_scale_local = output.get('scale', None)
                scene_shift_local = output.get('translation', output.get('shift', None))
                
                if scene_scale_local is not None and scene_shift_local is not None:
                    # 转换到 numpy
                    if isinstance(scene_scale_local, torch.Tensor):
                        scene_scale_local = scene_scale_local.detach().cpu().numpy()
                    if isinstance(scene_shift_local, torch.Tensor):
                        scene_shift_local = scene_shift_local.detach().cpu().numpy()
                    
                    # 处理 batch 维度
                    if scene_scale_local.ndim > 1:
                        scene_scale_local = scene_scale_local[0]
                    if scene_shift_local.ndim > 1:
                        scene_shift_local = scene_shift_local[0]
                    
                    # 反归一化
                    if np.all(np.isfinite(scene_scale_local)) and np.all(np.isfinite(scene_shift_local)):
                        xyz_physical = xyz_normalized * scene_scale_local + scene_shift_local
                        center_from_pc = xyz_physical.mean(axis=0)
                        print(f"    ✅ Center from mesh point cloud (fallback): {center_from_pc}")
            
            # 最终中心
            if center_from_pc is not None:
                final_center = center_from_pc
            else:
                # 最后的 fallback: 使用 Fast-SAM3D 的 translation
                fastsam_translation = output.get('translation', None)
                if fastsam_translation is not None:
                    if isinstance(fastsam_translation, torch.Tensor):
                        fastsam_translation = fastsam_translation.detach().cpu().numpy()
                    if fastsam_translation.ndim > 1:
                        fastsam_translation = fastsam_translation[0]
                    final_center = fastsam_translation
                    print(f"    ⚠️  Using Fast-SAM3D translation as final center: {final_center}")
                else:
                    final_center = np.array([0.0, 0.0, 0.0])
                    print(f"    ❌ No center available, using origin")
            
            # ================================================================
            # 2. 从 Fast-SAM3D 的 scale/mesh 计算边界框尺寸
            # ================================================================
            
            # 获取 bbox_size (用于 fallback)
            bbox_size = output.get('scale', None)
            
            # 从 Mesh 顶点计算真正的边界框尺寸
            # Fast-SAM3D 生成的 mesh 包含完整的 3D 形状
            mesh_vertices = None
            if 'mesh' in output and output['mesh'] is not None:
                mesh = output['mesh']
                try:
                    # output['mesh'] 是 list[MeshExtractResult]，取第一个
                    if isinstance(mesh, (list, tuple)):
                        mesh = mesh[0] if len(mesh) > 0 else None
                    if mesh is not None and hasattr(mesh, 'vertices'):
                        verts = mesh.vertices
                        if hasattr(verts, 'numpy'):
                            mesh_vertices = verts.detach().cpu().numpy()
                        elif isinstance(verts, np.ndarray):
                            mesh_vertices = verts
                        # DEBUG: print raw mesh vertices range (before any transformation)
                        if mesh_vertices is not None:
                            vmin_raw = mesh_vertices.min(axis=0)
                            vmax_raw = mesh_vertices.max(axis=0)
                            print(f"    🔍 [DEBUG] RAW mesh vertices range: [{mesh_vertices.min():.4f}, {mesh_vertices.max():.4f}]")
                            print(f"    🔍 [DEBUG] RAW per-axis min={vmin_raw}, max={vmax_raw}")
                    elif isinstance(mesh, np.ndarray):
                        mesh_vertices = mesh
                except Exception as e:
                    print(f"    ⚠️  Failed to extract mesh vertices: {e}")
            
            # 如果 MeshExtractResult 顶点不可用，尝试从 glb (trimesh) 中提取
            # glb 经过了 to_glb 的 z-up→y-up 坐标变换，范围在 [-0.5, 0.5]
            # SLAT space = glb space（都在同一个归一化体素空间）
            glb_mesh_vertices = None
            glb = output.get('glb', None)
            if glb is not None:
                try:
                    glb_verts = np.asarray(glb.vertices)
                    if glb_verts.size > 0:
                        glb_mesh_vertices = glb_verts
                        print(f"    🔍 [DEBUG] glb vertices: [{glb_verts.min():.4f}, {glb_verts.max():.4f}]")
                except Exception as e:
                    print(f"    ⚠️  Failed to extract glb vertices: {e}")

            # =========================================================
            # New camera-frame alignment path:
            #   UniDepth + original K -> P_scene -> Line A center
            #   Fast-SAM3D mesh -> rendered views
            #   render-back silhouette/depth/projection gate
            #   aligned mesh -> 3D OBB -> JSON
            # =========================================================
            aligned_mesh_vertices, aligned_mesh_faces = extract_mesh_geometry_from_output(output)
            if aligned_mesh_vertices is None:
                aligned_mesh_vertices = glb_mesh_vertices if glb_mesh_vertices is not None else mesh_vertices
                aligned_mesh_faces = None
            if aligned_mesh_vertices is None:
                aligned_mesh_vertices = extract_mesh_vertices_from_output(output)

            depth_for_render = depth_independent if depth_independent is not None else output.get('depth_physical', None)
            aligned_bbox = None
            align_info = {'reason': 'alignment_not_run'}
            alignment_attempts = []

            if alignment_mode in ('render_back', 'render_back_then_sim3'):
                aligned_bbox, align_info = align_fastsam3d_mesh_with_render_back(
                    scene_points=pseudo_lidar,
                    mesh_vertices=aligned_mesh_vertices,
                    mesh_faces=aligned_mesh_faces,
                    prior_xyz=prior,
                    K=intrinsics,
                    image_shape=original_image.shape,
                    detection_box=box,
                    mask=mask,
                    real_depth=depth_for_render,
                    image_rgb=original_image,
                    fastsam_rotation=output.get('rotation', None),
                )
                alignment_attempts.append({'mode': 'render_back', **align_info})

            if aligned_bbox is None and alignment_mode in ('sim3', 'render_back_then_sim3'):
                sim3_bbox, sim3_info = align_fastsam3d_mesh_to_scene_points(
                    scene_points=pseudo_lidar,
                    mesh_vertices=aligned_mesh_vertices,
                    prior_xyz=prior,
                    K=intrinsics,
                    image_shape=original_image.shape,
                    detection_box=box,
                    mask=mask,
                    fastsam_rotation=output.get('rotation', None),
                )
                alignment_attempts.append({'mode': 'sim3', **sim3_info})
                if sim3_bbox is not None:
                    aligned_bbox = sim3_bbox
                    align_info = sim3_info
                    align_info['fallback_from_render_back'] = alignment_mode == 'render_back_then_sim3'

            if alignment_mode != 'legacy':
                align_record = {
                    'mode': alignment_mode,
                    **align_info,
                    'attempts': alignment_attempts,
                }
                if aligned_bbox is None:
                    print(f"    ⚠️  {alignment_mode} alignment rejected 3D label: {align_record}")
                    boxes3d_list.append(np.full((8, 3), -1, dtype=np.float16))
                    center_cam_list.append(np.array([-1, -1, -1]))
                    dimensions_list.append([-1, -1, -1])
                    R_cam_list.append(np.eye(3, dtype=np.float32))
                    bbox2d_list.append(box.tolist())
                    alignment_info_list.append(align_record)
                    output = None
                    pseudo_lidar = None
                    aligned_mesh_vertices = None
                    aligned_mesh_faces = None
                    depth_independent = None
                    maybe_cleanup_cuda()
                    continue

                boxes3d_list.append(np.array(aligned_bbox['vertices'], dtype=np.float16))
                center_cam_list.append(np.array(aligned_bbox['center_cam'], dtype=np.float32))
                dimensions_list.append(aligned_bbox['dimensions'])
                R_cam_list.append(np.array(aligned_bbox['R_cam'], dtype=np.float32))
                bbox2d_list.append(box.tolist())
                alignment_info_list.append(align_record)

                print(f"    ✅ {alignment_mode} aligned: {align_record}")
                print(
                    f"    ✅ Center: [{aligned_bbox['center_cam'][0]:.3f}, "
                    f"{aligned_bbox['center_cam'][1]:.3f}, {aligned_bbox['center_cam'][2]:.3f}]"
                )
                print(
                    f"    ✅ Size [X,Y,Z]: [{aligned_bbox['dimensions'][0]:.3f}, "
                    f"{aligned_bbox['dimensions'][1]:.3f}, {aligned_bbox['dimensions'][2]:.3f}]"
                )
                output = None
                pseudo_lidar = None
                aligned_mesh_vertices = None
                aligned_mesh_faces = None
                depth_independent = None
                maybe_cleanup_cuda()
                continue
            
            # =========================================================
            # 获取尺度因子：使用 pose decoder 输出的 instance scale
            # - output["scale"]: 每个实例自己的尺度（来自 pose decoder，已经在物理单位）
            # - output["translation"]: 实例位置
            # - output["rotation"]: 实例旋转 (quaternion wxyz)
            # - output["pointmap_scale/shift"]: pointmap SSI 归一化参数（不要用于 mesh）
            # =========================================================
            instance_scale_np = None
            instance_translation_np = None
            instance_rotation_np = None
            
            instance_scale_raw = output.get('scale', None)
            if instance_scale_raw is not None:
                if isinstance(instance_scale_raw, torch.Tensor):
                    instance_scale_raw = instance_scale_raw.detach().cpu().numpy()
                instance_scale_raw = np.squeeze(instance_scale_raw)
                # pose decoder 返回的 scale 是 (3,) 向量，取平均值作为统一尺度
                if instance_scale_raw.ndim > 0:
                    instance_scale_np = float(np.mean(np.abs(instance_scale_raw)))
                else:
                    instance_scale_np = float(np.abs(instance_scale_raw))
            
            instance_translation_raw = output.get('translation', None)
            if instance_translation_raw is not None:
                if isinstance(instance_translation_raw, torch.Tensor):
                    instance_translation_raw = instance_translation_raw.detach().cpu().numpy()
                instance_translation_np = np.squeeze(instance_translation_raw)
            
            instance_rotation_raw = output.get('rotation', None)
            if instance_rotation_raw is not None:
                if isinstance(instance_rotation_raw, torch.Tensor):
                    instance_rotation_raw = instance_rotation_raw.detach().cpu().numpy()
                instance_rotation_np = np.squeeze(instance_rotation_raw)
            
            # pointmap_scale/shift 仅用于记录，不用于 mesh 尺寸计算
            scene_scale_np = output.get('pointmap_scale', None)
            scene_shift_np = output.get('pointmap_shift', None)
            if scene_scale_np is not None:
                if isinstance(scene_scale_np, torch.Tensor):
                    scene_scale_np = scene_scale_np.detach().cpu().numpy()
                scene_scale_np = np.squeeze(scene_scale_np)
            if scene_shift_np is not None:
                if isinstance(scene_shift_np, torch.Tensor):
                    scene_shift_np = scene_shift_np.detach().cpu().numpy()
                scene_shift_np = np.squeeze(scene_shift_np)
            
            # 如果没有 mesh（包括 MeshExtractResult 和 glb），退回到使用 instance scale
            if glb_mesh_vertices is None and mesh_vertices is None:
                if instance_scale_np is not None and instance_scale_np > 0:
                    dx = dy = dz = instance_scale_np
                    print(f"    📏 Using instance scale (no mesh): {dx:.3f}")
                elif bbox_size is not None:
                    if isinstance(bbox_size, torch.Tensor):
                        bbox_size_np = bbox_size.detach().cpu().numpy()
                    else:
                        bbox_size_np = bbox_size
                    if bbox_size_np.ndim > 1:
                        bbox_size_np = bbox_size_np[0]
                    scale_val = np.abs(bbox_size_np[0]) if len(bbox_size_np) > 0 else 1.0
                else:
                    scale_val = 1.0
                dx = dy = dz = scale_val
                print(f"    📏 Using Fast-SAM3D scale (no mesh): {scale_val:.3f}")
            else:
                # =========================================================
                # 从 mesh 顶点计算物理尺寸（简化为相对比例法）
                #
                # 问题背景：SLAT 生成的 mesh 在归一化 voxel 空间，正确的物理换算系数未知。
                # 解决方案：计算 mesh 的"相对形状比例"（各轴占最大轴的比例），再用 instance_scale 换算。
                # 这样即使不知道绝对尺度，形状比例也是正确的。
                #
                # 公式：
                #   mesh_AABB_voxel = mesh.max - mesh.min          # voxel 空间 AABB
                #   max_axis = mesh_AABB_voxel.max()               # 最大轴
                #   relative_shape = mesh_AABB_voxel / max_axis     # 相对形状比例 (各轴 ∈ [0, 1])
                #   physical_size = relative_shape * instance_scale  # 物理尺寸
                # =========================================================
                active_verts = glb_mesh_vertices if glb_mesh_vertices is not None else mesh_vertices
                
                if active_verts.ndim == 3:
                    active_verts = active_verts.reshape(-1, 3)
                
                # DEBUG: 打印中间量
                print(f"    🔍 [DEBUG] instance_scale={instance_scale_np:.4f}")
                print(f"    🔍 [DEBUG] active_verts range: [{active_verts.min():.4f}, {active_verts.max():.4f}]")
                
                if instance_scale_np is not None and instance_scale_np > 0:
                    # 计算 mesh 在 voxel 空间的 AABB
                    mesh_min_vox = active_verts.min(axis=0)
                    mesh_max_vox = active_verts.max(axis=0)
                    mesh_aabb_vox = mesh_max_vox - mesh_min_vox  # voxel 空间 AABB
                    max_axis = max(mesh_aabb_vox.max(), 1e-6)    # 最大轴
                    
                    # 相对形状比例
                    relative_shape = mesh_aabb_vox / max_axis  # (dx_r, dy_r, dz_r)
                    physical_size = relative_shape * instance_scale_np
                    
                    dx, dy, dz = physical_size[0], physical_size[1], physical_size[2]
                    print(f"    🔍 [DEBUG] voxel_AABB: {mesh_aabb_vox}, max_axis: {max_axis:.4f}")
                    print(f"    🔍 [DEBUG] relative_shape: {relative_shape}")
                    print(f"    🔍 [DEBUG] physical_size: [{dx:.3f}, {dy:.3f}, {dz:.3f}]")
                else:
                    # instance_scale 不可用，退而用原始 voxel AABB（无物理单位）
                    mesh_min_vox = active_verts.min(axis=0)
                    mesh_max_vox = active_verts.max(axis=0)
                    dx = max(0.01, mesh_max_vox[0] - mesh_min_vox[0])
                    dy = max(0.01, mesh_max_vox[1] - mesh_min_vox[1])
                    dz = max(0.01, mesh_max_vox[2] - mesh_min_vox[2])
                    print(f"    ⚠️  No instance_scale, using voxel AABB (no physical units)")
                
                print(f"    📏 Mesh physical size: [{dx:.3f}, {dy:.3f}, {dz:.3f}]")
            
            print(f"    📏 Raw size: [{dx:.3f}, {dy:.3f}, {dz:.3f}]")
            
            # 3. 用 prior 进行合理性校验和调整
            if prior is not None and any(p > 0 for p in prior):
                prior_dims = np.array(prior)  # [w=X宽, h=Y高, l=Z深] (camera 坐标系)
                prior_volume = prior_dims[0] * prior_dims[1] * prior_dims[2]
                
                # 计算 prior 的等效均匀尺度（prior 体积的立方根）
                prior_equivalent_scale = (prior_volume) ** (1.0/3.0)
                
                # 坐标系说明：
                # - glb mesh: y-up 坐标系 (X=right, Y=up, Z=depth)
                # - prior: camera 坐标系 (X=right, Y=down, Z=depth)
                # - 因此 physical_size = [dx=X宽, dy=Y(up), dz=Z深]
                # - prior_dims = [prior[0]=X宽, prior[1]=Y(down), prior[2]=Z深]
                # - Y 轴方向相反，不能直接比较绝对值，需要考虑方向
                
                mesh_dims = np.array([dx, dy, dz])
                
                # 对于 X 和 Z 轴（方向一致），直接比较比例
                # 对于 Y 轴，因为 glb 是 y-up，camera 是 y-down，所以：
                #   camera_y_height = -glb_y_range (取负值使其向下为正)
                #   或者直接用 prior 的 Y 高度作为参考
                
                # X, Z 轴比例（直接比较）
                axis_ratios_xz = np.array([mesh_dims[0] / (prior_dims[0] + 1e-6),  # X: dx/w
                                           mesh_dims[2] / (prior_dims[2] + 1e-6)])  # Z: dz/l
                
                # Y 轴比例：mesh 的 Y 跨度 / prior 的 Y 高度
                # 由于 Y 轴方向相反，我们只检查 mesh Y 跨度是否在合理范围内
                y_ratio = mesh_dims[1] / (prior_dims[1] + 1e-6)  # dy/h
                
                # 综合判断：
                # 1. 极端形状：X 或 Z 轴 < prior 的 10%，或 Y 轴 < 10%
                # 2. 合理：X 和 Z 轴在 [0.2, 5.0] 范围内，且 Y 轴 >= 10% 且 < 3
                # 3. 尺度偏差：其他情况
                
                extreme_xz = np.any(axis_ratios_xz < 0.1)  # X 或 Z 轴极端
                extreme_y = y_ratio < 0.1  # Y 轴极端（小于 prior 的 10%）
                extreme_shape = extreme_xz or extreme_y
                
                healthy_xz = np.all(axis_ratios_xz >= 0.2) and np.all(axis_ratios_xz <= 5.0)
                healthy_y = y_ratio >= 0.1 and y_ratio < 3.0  # Y 轴在合理范围内
                all_axes_close = healthy_xz and healthy_y and not extreme_shape
                
                print(f"    🔍 Axis ratios: XZ={axis_ratios_xz.round(2).tolist()}, Y={y_ratio:.2f}")
                
                # 初始化标志
                yaw_from_ray_tracing = False
                yaw = 0.0
                
                if extreme_shape:
                    # 极端形状：mesh 某轴接近 0 或 prior 的 <10%
                    # 完全按照原版 Cubercnn 的 estimate_bbox 流程
                    print(f"    ⚠️  Extreme shape (XZ={axis_ratios_xz.round(2).tolist()}, Y={y_ratio:.2f}), running cubercnn estimate_bbox...")
                    
                    # 检查是否有可用的伪点云
                    if pseudo_lidar is not None and len(pseudo_lidar) > 10:
                        try:
                            from sklearn.decomposition import PCA
                            
                            # ========== 原版流程 Step 1: 采样点云 ==========
                            if pseudo_lidar.shape[0] > 500:
                                rand_ind = np.random.randint(0, pseudo_lidar.shape[0], 500)
                                pc = pseudo_lidar[rand_ind]
                            else:
                                pc = pseudo_lidar
                            
                            w, h, l = float(prior_dims[0]), float(prior_dims[1]), float(prior_dims[2])
                            
                            # ========== 原版流程 Step 2: PCA 确定 yaw ==========
                            pca = PCA(2)
                            pca.fit(pc[:, [0, 2]])  # XZ 平面
                            yaw_vec = pca.components_[0, :]
                            yaw_rt = float(np.arctan2(yaw_vec[1], yaw_vec[0]))
                            
                            # ========== 原版流程 Step 3: yaw 对齐 ==========
                            rot_y_mat = rotate_y_cubercnn(yaw_rt)
                            rotated_pc = pc @ rot_y_mat.T  # 第一次旋转到 yaw=0 方向
                            
                            # ========== 原版流程 Step 4: 计算 AABB ==========
                            x_min, x_max = rotated_pc[:, 0].min(), rotated_pc[:, 0].max()
                            y_min, y_max = rotated_pc[:, 1].min(), rotated_pc[:, 1].max()
                            z_min, z_max = rotated_pc[:, 2].min(), rotated_pc[:, 2].max()
                            
                            dx_rt, dy_rt, dz_rt = x_max - x_min, y_max - y_min, z_max - z_min
                            cx_rt, cy_rt, cz_rt = (x_min + x_max) / 2, (y_min + y_max) / 2, (z_min + z_max) / 2
                            
                            # ========== 原版流程 Step 5: 高度处理 ==========
                            if dy_rt < h * 0.5:
                                dy_rt = h
                            
                            # ========== 原版流程 Step 6: 判断合理范围 ==========
                            if (l * 0.5 <= dx_rt and w * 0.5 <= dz_rt) or (l * 0.5 <= dz_rt and w * 0.5 <= dx_rt):
                                # 合理范围：直接使用 AABB，与原版一致
                                dx, dy, dz = dx_rt, dy_rt, dz_rt
                                
                                # 原版：用 convert_box_vertices 生成顶点，然后逆旋转
                                vertives_rt = convert_box_vertices(cx_rt, cy_rt, cz_rt, dx, dy, dz, 0).astype(np.float16)
                                vertives_rt = np.dot(rotate_y_cubercnn(-yaw_rt), vertives_rt.T).T  # 逆 yaw 旋转
                                vertives_rt = np.dot(vertives_rt, rot_y_mat.T)  # 逆第一次旋转
                                
                                # 原版：用顶点重新计算 center
                                final_center_from_rt = vertives_rt.mean(axis=0)
                                print(f"    ✅ Cubercnn: size OK, using AABB [{dz:.3f}, {dy:.3f}, {dx:.3f}]")
                                print(f"    ✅ Cubercnn: center from vertices: {final_center_from_rt}")
                            else:
                                # 不合理范围：使用 ray tracing 优化，与原版一致
                                possible_bboxs = generate_possible_bboxs(cx_rt, cz_rt, dx_rt, dz_rt, w, l)
                                min_loss, best_vertives = float('inf'), None
                                
                                for bbox in possible_bboxs:
                                    x_min_b, x_max_b, z_min_b, z_max_b = bbox
                                    dx_b, dz_b = x_max_b - x_min_b, z_max_b - z_min_b
                                    cx_b, cz_b = (x_min_b + x_max_b) / 2, (z_min_b + z_max_b) / 2
                                    
                                    # 原版：计算 inside_ratio（在 rotated_pc 上，即 yaw 对齐后的点云）
                                    inside_ratio = calc_inside_ratio(rotated_pc.T, x_min_b, x_max_b, z_min_b, z_max_b)
                                    
                                    # 原版：生成顶点
                                    vertives_b = convert_box_vertices(cx_b, cy_rt, cz_b, dx_b, dy_rt, dz_b, 0).astype(np.float16)
                                    vertives_b = np.dot(rotate_y_cubercnn(-yaw_rt), vertives_b.T).T  # 逆 yaw 旋转
                                    
                                    # 原版：用顶点重新计算 center
                                    new_cx, new_cz = vertives_b[:, 0].mean(), vertives_b[:, 2].mean()
                                    
                                    # 原版：计算 ray tracing loss
                                    pc_tensor = torch.from_numpy(rotated_pc).float()
                                    loss_ray = calc_dis_ray_tracing_cubercnn(
                                        torch.Tensor([dz_b, dx_b]), torch.Tensor([yaw_rt]),
                                        pc_tensor[:, [0, 2]], torch.Tensor([new_cx, new_cz])
                                    )
                                    loss = loss_ray + 5 * (1 - inside_ratio)
                                    
                                    if loss < min_loss:
                                        min_loss = loss
                                        best_vertives = vertives_b.copy()
                                        best_dx, best_dz = dx_b, dz_b
                                
                                # 原版：逆第一次旋转
                                best_vertives = np.dot(best_vertives, rot_y_mat.T)
                                
                                dx, dy, dz = best_dx, dy_rt, best_dz
                                print(f"    ✅ Cubercnn: best bbox [{dz:.3f}, {dy:.3f}, {dx:.3f}], loss={min_loss:.3f}")
                            
                            # 最终 center 使用 ray tracing 结果
                            yaw = yaw_rt
                            yaw_from_ray_tracing = True
                            
                            # 原版：用顶点重新计算 center
                            if 'best_vertives' in locals() or 'vertives_rt' in locals():
                                final_center = vertives_rt.mean(axis=0) if 'vertives_rt' in locals() else best_vertives.mean(axis=0)
                                print(f"    ✅ Final center from ray tracing: {final_center}")
                            
                            volume_ratio = np.prod([dx, dy, dz]) / (prior_dims[0] * prior_dims[1] * prior_dims[2])
                            
                        except Exception as e:
                            # Ray tracing 失败，回退到 prior
                            import traceback
                            print(f"    ⚠️  Cubercnn estimate_bbox failed ({e}), using prior")
                            print(f"    Traceback: {traceback.format_exc()}")
                            dx, dy, dz = float(prior_dims[0]), float(prior_dims[1]), float(prior_dims[2])
                            volume_ratio = 1.0
                    else:
                        # 没有伪点云，回退到 prior
                        print(f"    ⚠️  No pseudo_lidar available, using prior")
                        dx, dy, dz = float(prior_dims[0]), float(prior_dims[1]), float(prior_dims[2])
                        volume_ratio = 1.0
                elif all_axes_close:
                    # 合理：X 和 Z 轴与 prior 接近，Y 轴在合理范围内
                    # 使用 prior 等效尺度保持物理大小一致
                    volume_ratio = np.prod(axis_ratios_xz) * y_ratio
                    print(f"    📏 Size healthy (XZ={axis_ratios_xz.round(2).tolist()}, Y={y_ratio:.2f}), keeping mesh size")
                else:
                    # 尺度偏差：保留 mesh 形状比例，但用 prior 等效尺度调整整体大小
                    relative_shape = mesh_dims / max(mesh_dims.max(), 1e-6)
                    dx, dy, dz = relative_shape[0] * prior_equivalent_scale, \
                                  relative_shape[1] * prior_equivalent_scale, \
                                  relative_shape[2] * prior_equivalent_scale
                    volume_ratio = np.prod(relative_shape)
                    print(f"    📏 Scale adjusted (XZ={axis_ratios_xz.round(2).tolist()}, Y={y_ratio:.2f}), shape preserved")
            
            # 原版 dimension 顺序: [dz, dy, dx] (KITTI format: width, height, length)
            # dx = x轴长度, dy = y轴长度, dz = z轴长度
            dimensions = [float(dz), float(dy), float(dx)]
            
            # 确保尺寸为正数
            if dx <= 0 or dy <= 0 or dz <= 0:
                print(f"    ⚠️  Negative size detected [{dx:.3f}, {dy:.3f}, {dz:.3f}], using prior")
                if prior is not None:
                    dimensions = [float(p) for p in prior]
                    dx, dy, dz = dimensions
            
            # 4. 提取旋转
            # - 如果是极端形状：yaw 已由 ray tracing 计算
            # - 否则：使用 Fast-SAM3D 的 rotation 输出
            R_cam = np.eye(3, dtype=np.float32)
            if yaw_from_ray_tracing:
                # 使用 ray tracing 计算的 yaw 构建 R_cam
                rot_y = rotate_y_cubercnn(yaw)
                R_cam = rot_y
                print(f"    📐 Ray tracing: using yaw from point cloud: {yaw:.3f} rad ({np.degrees(yaw):.1f}°)")
            else:
                rotation = output.get('rotation', None)
                if rotation is not None:
                    if isinstance(rotation, torch.Tensor):
                        rotation = rotation.detach().cpu().numpy()
                    if rotation.ndim > 1:
                        rotation = rotation[0]
                    try:
                        from scipy.transform import Rotation as R_scipy
                        r = R_scipy.from_quat(rotation)
                        R_cam = r.as_matrix().astype(np.float32)
                        yaw = np.arctan2(R_cam[0, 2] - R_cam[2, 0], R_cam[0, 0] + R_cam[2, 2]) if R_cam[0, 0] != 0 or R_cam[2, 2] != 0 else 0
                        print(f"    📐 Line B: Fast-SAM3D yaw: {yaw:.3f} rad ({np.degrees(yaw):.1f}°)")
                    except:
                        print(f"    📐 Line B: Using default yaw (Fast-SAM3D rotation failed)")
                else:
                    print(f"    📐 Line B: No Fast-SAM3D rotation, using default yaw")
            
            # 生成 8 个顶点
            from cubercnn.generate_label.util import convert_box_vertices
            vertives = convert_box_vertices(
                final_center[0], final_center[1], final_center[2],
                dx, dy, dz, yaw
            ).astype(np.float16)
            
            bbox = {
                'center_cam': final_center.tolist() if hasattr(final_center, 'tolist') else list(final_center),
                'dimensions': dimensions,
                'R_cam': R_cam.tolist(),
                'yaw': float(yaw),
                'vertices': vertives.tolist()
            }
            print(f"    ✅ Final BBox: center={final_center}, size=[{dx:.3f}, {dy:.3f}, {dz:.3f}]")
            
            boxes3d_list.append(np.array(bbox['vertices'], dtype=np.float16))
            center_cam_list.append(np.array(bbox['center_cam']))
            dimensions_list.append(bbox['dimensions'])
            R_cam_list.append(np.array(bbox['R_cam']))
            bbox2d_list.append(box.tolist())
            alignment_info_list.append({
                'mode': alignment_mode,
                'reason': 'legacy_bbox',
            })
            
            print(f"    ✅ Center: [{bbox['center_cam'][0]:.3f}, {bbox['center_cam'][1]:.3f}, {bbox['center_cam'][2]:.3f}]")
            print(f"    ✅ Size: [{bbox['dimensions'][0]:.3f}, {bbox['dimensions'][1]:.3f}, {bbox['dimensions'][2]:.3f}]")
            output = None
            pseudo_lidar = None
            depth_independent = None
            maybe_cleanup_cuda()
            
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
            alignment_info_list.append({
                'mode': alignment_mode,
                'reason': 'exception',
                'error': str(e),
                'cuda_oom': bool(is_cuda_oom_error(e)),
            })
            output = None
            pseudo_lidar = None
            depth_independent = None
            maybe_cleanup_cuda()
    
    return {
        'image_id': image_id,
        'boxes3d': boxes3d_list,
        'center_cam': center_cam_list,
        'dimensions': dimensions_list,
        'R_cam': R_cam_list,
        'phrases': seg_results['labels'],
        'conf': list(seg_results['scores']) if isinstance(seg_results['scores'], (list, tuple)) else seg_results['scores'].tolist(),
        'boxes': bbox2d_list,
        'alignment_info': alignment_info_list
    }


def process_images_on_gpu(gpu_id, image_data_subset, args, category_prior, output_suffix='', resume=False):
    """
    在单个 GPU 上处理一组图片
    
    Args:
        resume: 如果为 True，从 checkpoint 恢复，跳过已处理的图片
    """
    import torch

    global PSEUDO3D_MIN_PROJECTION_IOU, PSEUDO3D_MIN_POINT_SUPPORT
    global RENDER_ALIGN_SCALE, RENDER_ALIGN_MAX_FACES, RENDER_ALIGN_MIN_SILHOUETTE_IOU
    global RENDER_ALIGN_MIN_BBOX_IOU, RENDER_ALIGN_MAX_REL_DEPTH_ERROR
    global RENDER_ALIGN_STRICT_MIN_PROJECTION_IOU, RENDER_ALIGN_STRICT_MIN_SILHOUETTE_IOU
    global RENDER_ALIGN_STRICT_MAX_REL_DEPTH_ERROR, RENDER_ALIGN_REQUIRE_STRICT
    global RENDER_PRIOR_MIN_RATIO, RENDER_PRIOR_MAX_RATIO, RENDER_PNP_MATCHER
    global MAST3R_ROOT, MAST3R_MODEL_NAME, MAST3R_IMAGE_SIZE, MAST3R_SUBSAMPLE, MAST3R_CROP_PAD
    global RENDER_YAW_OFFSETS_DEG, RENDER_SCALE_MULTIPLIERS
    global RENDER_USE_ALIGNED_MESH_OBB, RENDER_SILHOUETTE_REFINE
    global RENDER_SILHOUETTE_AREA_SCALE_MIN, RENDER_SILHOUETTE_AREA_SCALE_MAX
    global RENDER_SILHOUETTE_MIN_RENDER_PIXELS, RENDER_SILHOUETTE_MIN_SHIFT_PIXELS
    global RENDER_SILHOUETTE_MAX_SHIFT_NORM, DEPTH_EDGE_REL_THRESH, DEPTH_EDGE_MIN_KEEP_RATIO
    PSEUDO3D_MIN_PROJECTION_IOU = args.get('pseudo3d_min_projection_iou', PSEUDO3D_MIN_PROJECTION_IOU)
    PSEUDO3D_MIN_POINT_SUPPORT = args.get('pseudo3d_min_point_support', PSEUDO3D_MIN_POINT_SUPPORT)
    RENDER_ALIGN_SCALE = args.get('render_align_scale', RENDER_ALIGN_SCALE)
    RENDER_ALIGN_MAX_FACES = args.get('render_align_max_faces', RENDER_ALIGN_MAX_FACES)
    RENDER_ALIGN_MIN_SILHOUETTE_IOU = args.get(
        'render_align_min_silhouette_iou', RENDER_ALIGN_MIN_SILHOUETTE_IOU
    )
    RENDER_ALIGN_MIN_BBOX_IOU = args.get('render_align_min_bbox_iou', RENDER_ALIGN_MIN_BBOX_IOU)
    RENDER_ALIGN_MAX_REL_DEPTH_ERROR = args.get(
        'render_align_max_rel_depth_error', RENDER_ALIGN_MAX_REL_DEPTH_ERROR
    )
    RENDER_ALIGN_STRICT_MIN_PROJECTION_IOU = args.get(
        'render_align_strict_min_projection_iou', RENDER_ALIGN_STRICT_MIN_PROJECTION_IOU
    )
    RENDER_ALIGN_STRICT_MIN_SILHOUETTE_IOU = args.get(
        'render_align_strict_min_silhouette_iou', RENDER_ALIGN_STRICT_MIN_SILHOUETTE_IOU
    )
    RENDER_ALIGN_STRICT_MAX_REL_DEPTH_ERROR = args.get(
        'render_align_strict_max_rel_depth_error', RENDER_ALIGN_STRICT_MAX_REL_DEPTH_ERROR
    )
    RENDER_ALIGN_REQUIRE_STRICT = args.get('render_align_require_strict', RENDER_ALIGN_REQUIRE_STRICT)
    RENDER_PRIOR_MIN_RATIO = args.get('render_prior_min_ratio', RENDER_PRIOR_MIN_RATIO)
    RENDER_PRIOR_MAX_RATIO = args.get('render_prior_max_ratio', RENDER_PRIOR_MAX_RATIO)
    RENDER_PNP_MATCHER = args.get('render_pnp_matcher', RENDER_PNP_MATCHER)
    MAST3R_ROOT = args.get('mast3r_root', MAST3R_ROOT)
    MAST3R_MODEL_NAME = args.get('mast3r_model_name', MAST3R_MODEL_NAME)
    MAST3R_IMAGE_SIZE = int(args.get('mast3r_image_size', MAST3R_IMAGE_SIZE))
    MAST3R_SUBSAMPLE = int(args.get('mast3r_subsample', MAST3R_SUBSAMPLE))
    MAST3R_CROP_PAD = float(args.get('mast3r_crop_pad', MAST3R_CROP_PAD))
    RENDER_YAW_OFFSETS_DEG = parse_float_list(
        args.get('render_yaw_offsets_deg', None),
        RENDER_YAW_OFFSETS_DEG,
    )
    RENDER_SCALE_MULTIPLIERS = parse_float_list(
        args.get('render_scale_multipliers', None),
        RENDER_SCALE_MULTIPLIERS,
    )
    RENDER_USE_ALIGNED_MESH_OBB = not args.get('no_aligned_mesh_obb', False)
    RENDER_SILHOUETTE_REFINE = not args.get('no_silhouette_refine', False)
    RENDER_SILHOUETTE_AREA_SCALE_MIN = args.get(
        'silhouette_area_scale_min', RENDER_SILHOUETTE_AREA_SCALE_MIN
    )
    RENDER_SILHOUETTE_AREA_SCALE_MAX = args.get(
        'silhouette_area_scale_max', RENDER_SILHOUETTE_AREA_SCALE_MAX
    )
    RENDER_SILHOUETTE_MIN_RENDER_PIXELS = args.get(
        'silhouette_min_render_pixels', RENDER_SILHOUETTE_MIN_RENDER_PIXELS
    )
    RENDER_SILHOUETTE_MIN_SHIFT_PIXELS = args.get(
        'silhouette_min_shift_pixels', RENDER_SILHOUETTE_MIN_SHIFT_PIXELS
    )
    RENDER_SILHOUETTE_MAX_SHIFT_NORM = args.get(
        'silhouette_max_shift_norm', RENDER_SILHOUETTE_MAX_SHIFT_NORM
    )
    DEPTH_EDGE_REL_THRESH = args.get('depth_edge_rel_thresh', DEPTH_EDGE_REL_THRESH)
    DEPTH_EDGE_MIN_KEEP_RATIO = args.get('depth_edge_min_keep_ratio', DEPTH_EDGE_MIN_KEEP_RATIO)
    
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
            # 获取 intrinsics (K 矩阵) - 直接使用 JSON 中的值，不需要调整
            # JSON 中的 K 已经是针对该图像的正确内参
            intrinsics = None
            if 'K' in img_info and img_info['K'] is not None:
                K = img_info['K']
                if isinstance(K, list) and len(K) == 3:
                    import numpy as np
                    intrinsics = np.array(K, dtype=np.float32)
            
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
                intrinsics=intrinsics,
                alignment_mode=args.get('fastsam3d_alignment_mode', 'render_back_then_sim3'),
                clear_cuda_cache_per_object=not args.get('no_clear_cuda_cache_per_object', False),
                min_mask_pixels=args.get('fastsam3d_min_mask_pixels', FASTSAM3D_MIN_MASK_PIXELS),
                min_mask_area_ratio=args.get('fastsam3d_min_mask_area_ratio', FASTSAM3D_MIN_MASK_AREA_RATIO),
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
    global PSEUDO3D_MIN_PROJECTION_IOU, PSEUDO3D_MIN_POINT_SUPPORT
    global RENDER_ALIGN_SCALE, RENDER_ALIGN_MAX_FACES, RENDER_ALIGN_MIN_SILHOUETTE_IOU
    global RENDER_ALIGN_MIN_BBOX_IOU, RENDER_ALIGN_MAX_REL_DEPTH_ERROR
    global RENDER_ALIGN_STRICT_MIN_PROJECTION_IOU, RENDER_ALIGN_STRICT_MIN_SILHOUETTE_IOU
    global RENDER_ALIGN_STRICT_MAX_REL_DEPTH_ERROR, RENDER_ALIGN_REQUIRE_STRICT
    global RENDER_PRIOR_MIN_RATIO, RENDER_PRIOR_MAX_RATIO, RENDER_PNP_MATCHER
    global MAST3R_ROOT, MAST3R_MODEL_NAME, MAST3R_IMAGE_SIZE, MAST3R_SUBSAMPLE, MAST3R_CROP_PAD
    global RENDER_YAW_OFFSETS_DEG, RENDER_SCALE_MULTIPLIERS
    global RENDER_USE_ALIGNED_MESH_OBB, RENDER_SILHOUETTE_REFINE
    global RENDER_SILHOUETTE_AREA_SCALE_MIN, RENDER_SILHOUETTE_AREA_SCALE_MAX
    global RENDER_SILHOUETTE_MIN_RENDER_PIXELS, RENDER_SILHOUETTE_MIN_SHIFT_PIXELS
    global RENDER_SILHOUETTE_MAX_SHIFT_NORM, DEPTH_EDGE_REL_THRESH, DEPTH_EDGE_MIN_KEEP_RATIO

    parser = argparse.ArgumentParser(description="Generate Pseudo Labels with Grounding-SAM 2 + Fast-SAM3D")
    
    # 输入输出
    parser.add_argument('--json_file', type=str, default=None, help='JSON file containing image paths (for SUNRGBD)')
    parser.add_argument('--image_root', type=str, default='', help='Root directory for images (used with --json_file)')
    parser.add_argument('--image_dir', type=str, default=None, help='Directory containing images (alternative to --json_file)')
    parser.add_argument('--image_list', type=str, default=None, help='List of image files to process')
    parser.add_argument('--text_prompt', type=str, default=None, help='Text prompt for segmentation (if None, use SUNRGBD 38 classes for SUNRGBD dataset)')
    parser.add_argument('--split', type=str, default='train', choices=['train', 'val'], help='Dataset split (train or val)')
    parser.add_argument('--output_dir', type=str, default='/extra/ZhaoX/pseudo_label_gsam', help='Output directory (base path)')
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
    parser.add_argument(
        '--fastsam3d_alignment_mode',
        type=str,
        default='render_back_then_sim3',
        choices=['render_back', 'sim3', 'render_back_then_sim3', 'legacy'],
        help='How to put Fast-SAM3D mesh into metric camera space. '
             'render_back: require silhouette/depth closure; '
             'sim3: align relative mesh to UniDepth mask cloud; '
             'render_back_then_sim3: use render-back first, then Sim(3) fallback; '
             'legacy: old size/yaw branch.'
    )
    parser.add_argument('--pseudo3d_min_projection_iou', type=float, default=PSEUDO3D_MIN_PROJECTION_IOU,
                        help='Minimum projected 3D box IoU with the 2D mask/box gate')
    parser.add_argument('--pseudo3d_min_point_support', type=float, default=PSEUDO3D_MIN_POINT_SUPPORT,
                        help='Minimum UniDepth mask point ratio inside the candidate 3D box')
    parser.add_argument('--render_align_scale', type=float, default=RENDER_ALIGN_SCALE,
                        help='Downscale factor for render-back silhouette/depth verification')
    parser.add_argument('--render_align_max_faces', type=int, default=RENDER_ALIGN_MAX_FACES,
                        help='Maximum mesh faces used by the lightweight render-back z-buffer')
    parser.add_argument('--render_align_min_silhouette_iou', type=float, default=RENDER_ALIGN_MIN_SILHOUETTE_IOU,
                        help='Minimum rendered mesh silhouette IoU with the target mask')
    parser.add_argument('--render_align_min_bbox_iou', type=float, default=RENDER_ALIGN_MIN_BBOX_IOU,
                        help='Minimum rendered mesh bbox IoU with the target mask/detection box')
    parser.add_argument('--render_align_max_rel_depth_error', type=float, default=RENDER_ALIGN_MAX_REL_DEPTH_ERROR,
                        help='Maximum median relative depth error for render-back depth closure')
    parser.add_argument('--render_align_strict_min_projection_iou', type=float, default=RENDER_ALIGN_STRICT_MIN_PROJECTION_IOU,
                        help='Strict-quality minimum projected 3D box IoU')
    parser.add_argument('--render_align_strict_min_silhouette_iou', type=float, default=RENDER_ALIGN_STRICT_MIN_SILHOUETTE_IOU,
                        help='Strict-quality minimum rendered silhouette IoU')
    parser.add_argument('--render_align_strict_max_rel_depth_error', type=float, default=RENDER_ALIGN_STRICT_MAX_REL_DEPTH_ERROR,
                        help='Strict-quality maximum rendered-vs-real relative depth error')
    parser.add_argument('--render_align_require_strict', action='store_true',
                        help='Only accept render-back labels that pass the strict quality tier')
    parser.add_argument('--render_prior_min_ratio', type=float, default=RENDER_PRIOR_MIN_RATIO,
                        help='Minimum accepted dimension ratio relative to the class prior')
    parser.add_argument('--render_prior_max_ratio', type=float, default=RENDER_PRIOR_MAX_RATIO,
                        help='Maximum accepted dimension ratio relative to the class prior')
    parser.add_argument('--render_pnp_matcher', type=str, default=RENDER_PNP_MATCHER,
                        choices=['auto', 'boundary', 'orb', 'orb_edge', 'sift', 'akaze', 'opencv', 'mast3r', 'lightglue', 'loftr'],
                        help='2D-3D correspondence backend for render-back PnP')
    parser.add_argument('--mast3r_root', type=str, default=MAST3R_ROOT,
                        help='Path to the MASt3R repository root used for dense render-to-real matching')
    parser.add_argument('--mast3r_model_name', type=str, default=MAST3R_MODEL_NAME,
                        help='MASt3R checkpoint name or local checkpoint path')
    parser.add_argument('--mast3r_image_size', type=int, default=MAST3R_IMAGE_SIZE,
                        help='Square crop size passed to MASt3R')
    parser.add_argument('--mast3r_subsample', type=int, default=MAST3R_SUBSAMPLE,
                        help='MASt3R reciprocal-NN descriptor subsampling stride')
    parser.add_argument('--mast3r_crop_pad', type=float, default=MAST3R_CROP_PAD,
                        help='Padding factor around target/render masks before MASt3R matching')
    parser.add_argument('--render_yaw_offsets_deg', type=str, default=','.join(str(v) for v in RENDER_YAW_OFFSETS_DEG),
                        help='Comma-separated yaw offsets in degrees for multi-view render search')
    parser.add_argument('--render_scale_multipliers', type=str, default=','.join(str(v) for v in RENDER_SCALE_MULTIPLIERS),
                        help='Comma-separated scale multipliers for multi-scale render search')
    parser.add_argument('--no_aligned_mesh_obb', action='store_true',
                        help='Disable robust OBB extraction from the aligned mesh points')
    parser.add_argument('--no_silhouette_refine', action='store_true',
                        help='Disable render-back silhouette area/centroid refinement')
    parser.add_argument('--silhouette_area_scale_min', type=float, default=RENDER_SILHOUETTE_AREA_SCALE_MIN,
                        help='Minimum extra scale multiplier used when rendered silhouette is too small')
    parser.add_argument('--silhouette_area_scale_max', type=float, default=RENDER_SILHOUETTE_AREA_SCALE_MAX,
                        help='Maximum extra scale multiplier used when rendered silhouette is too large/small')
    parser.add_argument('--silhouette_min_render_pixels', type=int, default=RENDER_SILHOUETTE_MIN_RENDER_PIXELS,
                        help='Minimum rendered pixels before silhouette area refinement is considered')
    parser.add_argument('--silhouette_min_shift_pixels', type=float, default=RENDER_SILHOUETTE_MIN_SHIFT_PIXELS,
                        help='Minimum render-vs-mask centroid shift in pixels before translation refinement')
    parser.add_argument('--silhouette_max_shift_norm', type=float, default=RENDER_SILHOUETTE_MAX_SHIFT_NORM,
                        help='Maximum silhouette centroid translation as a fraction of object depth')
    parser.add_argument('--depth_edge_rel_thresh', type=float, default=DEPTH_EDGE_REL_THRESH,
                        help='Relative log-depth gradient threshold for removing mask depth-edge artifacts')
    parser.add_argument('--depth_edge_min_keep_ratio', type=float, default=DEPTH_EDGE_MIN_KEEP_RATIO,
                        help='Fallback to the original mask if depth-edge cleanup keeps less than this ratio')
    parser.add_argument('--fastsam3d_min_mask_pixels', type=int, default=FASTSAM3D_MIN_MASK_PIXELS,
                        help='Minimum mask pixels before running Fast-SAM3D')
    parser.add_argument('--fastsam3d_min_mask_area_ratio', type=float, default=FASTSAM3D_MIN_MASK_AREA_RATIO,
                        help='Minimum mask area ratio before running Fast-SAM3D')
    parser.add_argument('--no_clear_cuda_cache_per_object', action='store_true',
                        help='Disable per-object CUDA cache cleanup')

    args = parser.parse_args()

    PSEUDO3D_MIN_PROJECTION_IOU = args.pseudo3d_min_projection_iou
    PSEUDO3D_MIN_POINT_SUPPORT = args.pseudo3d_min_point_support
    RENDER_ALIGN_SCALE = args.render_align_scale
    RENDER_ALIGN_MAX_FACES = args.render_align_max_faces
    RENDER_ALIGN_MIN_SILHOUETTE_IOU = args.render_align_min_silhouette_iou
    RENDER_ALIGN_MIN_BBOX_IOU = args.render_align_min_bbox_iou
    RENDER_ALIGN_MAX_REL_DEPTH_ERROR = args.render_align_max_rel_depth_error
    RENDER_ALIGN_STRICT_MIN_PROJECTION_IOU = args.render_align_strict_min_projection_iou
    RENDER_ALIGN_STRICT_MIN_SILHOUETTE_IOU = args.render_align_strict_min_silhouette_iou
    RENDER_ALIGN_STRICT_MAX_REL_DEPTH_ERROR = args.render_align_strict_max_rel_depth_error
    RENDER_ALIGN_REQUIRE_STRICT = args.render_align_require_strict
    RENDER_PRIOR_MIN_RATIO = args.render_prior_min_ratio
    RENDER_PRIOR_MAX_RATIO = args.render_prior_max_ratio
    RENDER_PNP_MATCHER = args.render_pnp_matcher
    MAST3R_ROOT = args.mast3r_root
    MAST3R_MODEL_NAME = args.mast3r_model_name
    MAST3R_IMAGE_SIZE = args.mast3r_image_size
    MAST3R_SUBSAMPLE = args.mast3r_subsample
    MAST3R_CROP_PAD = args.mast3r_crop_pad
    RENDER_YAW_OFFSETS_DEG = parse_float_list(args.render_yaw_offsets_deg, RENDER_YAW_OFFSETS_DEG)
    RENDER_SCALE_MULTIPLIERS = parse_float_list(args.render_scale_multipliers, RENDER_SCALE_MULTIPLIERS)
    RENDER_USE_ALIGNED_MESH_OBB = not args.no_aligned_mesh_obb
    RENDER_SILHOUETTE_REFINE = not args.no_silhouette_refine
    RENDER_SILHOUETTE_AREA_SCALE_MIN = args.silhouette_area_scale_min
    RENDER_SILHOUETTE_AREA_SCALE_MAX = args.silhouette_area_scale_max
    RENDER_SILHOUETTE_MIN_RENDER_PIXELS = args.silhouette_min_render_pixels
    RENDER_SILHOUETTE_MIN_SHIFT_PIXELS = args.silhouette_min_shift_pixels
    RENDER_SILHOUETTE_MAX_SHIFT_NORM = args.silhouette_max_shift_norm
    DEPTH_EDGE_REL_THRESH = args.depth_edge_rel_thresh
    DEPTH_EDGE_MIN_KEEP_RATIO = args.depth_edge_min_keep_ratio
    
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
    
    # 实际输出目录 = base_dir / split (train 或 val)
    args.output_dir = os.path.join(args.output_dir, args.split)
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"📁 Output directory: {args.output_dir}")
    
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
                'fastsam3d_alignment_mode': args.fastsam3d_alignment_mode,
                'pseudo3d_min_projection_iou': args.pseudo3d_min_projection_iou,
                'pseudo3d_min_point_support': args.pseudo3d_min_point_support,
                'render_align_scale': args.render_align_scale,
                'render_align_max_faces': args.render_align_max_faces,
                'render_align_min_silhouette_iou': args.render_align_min_silhouette_iou,
                'render_align_min_bbox_iou': args.render_align_min_bbox_iou,
                'render_align_max_rel_depth_error': args.render_align_max_rel_depth_error,
                'render_align_strict_min_projection_iou': args.render_align_strict_min_projection_iou,
                'render_align_strict_min_silhouette_iou': args.render_align_strict_min_silhouette_iou,
                'render_align_strict_max_rel_depth_error': args.render_align_strict_max_rel_depth_error,
                'render_align_require_strict': args.render_align_require_strict,
                'render_prior_min_ratio': args.render_prior_min_ratio,
                'render_prior_max_ratio': args.render_prior_max_ratio,
                'render_pnp_matcher': args.render_pnp_matcher,
                'mast3r_root': args.mast3r_root,
                'mast3r_model_name': args.mast3r_model_name,
                'mast3r_image_size': args.mast3r_image_size,
                'mast3r_subsample': args.mast3r_subsample,
                'mast3r_crop_pad': args.mast3r_crop_pad,
                'render_yaw_offsets_deg': args.render_yaw_offsets_deg,
                'render_scale_multipliers': args.render_scale_multipliers,
                'no_aligned_mesh_obb': args.no_aligned_mesh_obb,
                'no_silhouette_refine': args.no_silhouette_refine,
                'silhouette_area_scale_min': args.silhouette_area_scale_min,
                'silhouette_area_scale_max': args.silhouette_area_scale_max,
                'silhouette_min_render_pixels': args.silhouette_min_render_pixels,
                'silhouette_min_shift_pixels': args.silhouette_min_shift_pixels,
                'silhouette_max_shift_norm': args.silhouette_max_shift_norm,
                'depth_edge_rel_thresh': args.depth_edge_rel_thresh,
                'depth_edge_min_keep_ratio': args.depth_edge_min_keep_ratio,
                'fastsam3d_min_mask_pixels': args.fastsam3d_min_mask_pixels,
                'fastsam3d_min_mask_area_ratio': args.fastsam3d_min_mask_area_ratio,
                'no_clear_cuda_cache_per_object': args.no_clear_cuda_cache_per_object,
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
            alignment_infos = result.get('alignment_info', [{} for _ in result['phrases']])
            json_results[image_id] = {
                'boxes': result['boxes'],
                'phrases': result['phrases'],
                'conf': result['conf'],
                'alignment_info': alignment_infos,
                'objects': []
            }
            for i in range(len(result['phrases'])):
                json_results[image_id]['objects'].append({
                    'label': result['phrases'][i],
                    'score': result['conf'][i],
                    'bbox2d': result['boxes'][i],
                    'center_cam': result['center_cam'][i],
                    'dimensions': result['dimensions'][i],
                    'vertices': result['boxes3d'][i].tolist() if hasattr(result['boxes3d'][i], 'tolist') else result['boxes3d'][i],
                    'alignment_info': alignment_infos[i] if i < len(alignment_infos) else {}
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
            alignment_infos = result.get('alignment_info', [{} for _ in result['phrases']])
            json_results[image_id] = {
                'boxes': result['boxes'],
                'phrases': result['phrases'],
                'conf': result['conf'],
                'alignment_info': alignment_infos,
                'objects': []
            }
            for i in range(len(result['phrases'])):
                json_results[image_id]['objects'].append({
                    'label': result['phrases'][i],
                    'score': result['conf'][i],
                    'bbox2d': result['boxes'][i],
                    'center_cam': result['center_cam'][i],
                    'dimensions': result['dimensions'][i],
                    'vertices': result['boxes3d'][i].tolist() if hasattr(result['boxes3d'][i], 'tolist') else result['boxes3d'][i],
                    'alignment_info': alignment_infos[i] if i < len(alignment_infos) else {}
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
    
    # 创建 COCO 格式的 images / annotations
    images = []
    annotations = []
    num = 1
    dataset_id = 1  # 可以根据数据集调整
    
    for image_id, result in all_results.items():
        metadata = result.get('_metadata', {})
        width = metadata.get('width')
        height = metadata.get('height')
        K = metadata.get('K')
        image_dataset_id = metadata.get('dataset_id', dataset_id)
        images.append({
            'id': image_id,
            'file_name': str(image_id),
            'width': width,
            'height': height,
            'K': K,
            'dataset_id': image_dataset_id,
        })
        phrases = result['phrases']
        scores = result['conf']
        boxes = result['boxes']
        boxes3d = result['boxes3d']
        center_cam = result['center_cam']
        dimensions = result['dimensions']
        R_cam = result['R_cam']
        alignment_infos = result.get('alignment_info', [{} for _ in phrases])
        
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
            bbox2d_tight = boxes[j].tolist() if hasattr(boxes[j], 'tolist') else boxes[j]
            if width is not None and height is not None:
                bbox2d_tight = convert_detection_box_to_xyxy_pixels(bbox2d_tight, width, height)

            dims_val = [float(x) for x in dimensions[j]]
            center_val = center_cam[j].tolist() if hasattr(center_cam[j], 'tolist') else center_cam[j]
            bbox3d_val = boxes3d[j].tolist() if hasattr(boxes3d[j], 'tolist') else boxes3d[j]
            R_val = R_cam[j].tolist() if hasattr(R_cam[j], 'tolist') else R_cam[j]
            alignment_info = alignment_infos[j] if j < len(alignment_infos) else {}
            alignment_reason = alignment_info.get('reason') if isinstance(alignment_info, dict) else None
            source_valid3d = alignment_reason in ('valid', 'valid_render_back')

            geometry_valid3d = (
                len(dims_val) == 3 and all(float(x) > 0 for x in dims_val) and
                len(center_val) == 3 and all(np.isfinite(float(x)) for x in center_val) and center_val[2] > 0
            )
            valid3d = bool(source_valid3d and geometry_valid3d)
            bbox2d_proj = bbox2d_tight
            bbox2d_trunc = bbox2d_tight
            step4_projection = {
                'checked': False,
                'projectable': False,
                'passes_gate': False,
            }
            if geometry_valid3d and K is not None and width is not None and height is not None:
                projection = project_vertices_to_box(bbox3d_val, K, width, height)
                if projection is None:
                    valid3d = False
                    step4_projection['checked'] = True
                else:
                    bbox2d_proj, bbox2d_trunc = projection
                    _, proj_iou, proj_center_norm, proj_area_ratio = projection_alignment_score(
                        bbox2d_trunc, bbox2d_tight
                    )
                    projection_gate_pass = passes_pseudo3d_projection_gate(
                        iou=proj_iou,
                        center_norm=proj_center_norm,
                        area_ratio=proj_area_ratio,
                    )
                    step4_projection = {
                        'checked': True,
                        'projectable': True,
                        'passes_gate': bool(projection_gate_pass),
                        'iou': float(proj_iou),
                        'center_norm': float(proj_center_norm),
                        'area_ratio': float(proj_area_ratio),
                    }
                    # If this annotation came from a legacy or bbox-only path
                    # without an alignment reason, use the projection gate as
                    # the final safety check. For the render-back/Sim3 path, the
                    # alignment gate already compared against the best available
                    # 2D evidence (often the mask box), while this Step 4 box may
                    # be a noisier GroundingDINO box.
                    if alignment_reason is None:
                        valid3d = bool(geometry_valid3d and projection_gate_pass)

            if not valid3d:
                center_val = [-1.0, -1.0, -1.0]
                dims_val = [-1.0, -1.0, -1.0]
                bbox3d_val = [[-1.0, -1.0, -1.0]] * 8
                R_val = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
            
            obj = {
                'id': dataset_id * 10000000 + num,
                'image_id': image_id,
                'dataset_id': dataset_id,
                'category_name': category_name,
                'category_id': thing_classes[category_name],
                'valid3D': bool(valid3d),
                'bbox2D_tight': bbox2d_tight,
                'bbox2D_trunc': bbox2d_trunc,
                'bbox2D_proj': bbox2d_proj,
                'bbox3D_cam': bbox3d_val,
                'center_cam': center_val,
                'dimensions': dims_val,
                'R_cam': R_val,
                'behind_camera': False,
                'visibility': 1.0,
                'truncation': 0.0,
                'segmentation_pts': -1,
                'lidar_pts': -1,
                'depth_error': -1,
                'score': score_val,
                'source_valid3D': bool(source_valid3d),
                'step4_projection': step4_projection,
                'fastsam3d_alignment': alignment_info
            }
            
            annotations.append(obj)
            num += 1
    
    # 保存 COCO 格式
    coco_output_dir = os.path.join(output_dir, 'coco_format')
    os.makedirs(coco_output_dir, exist_ok=True)
    
    coco_data = {
        'images': images,
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
