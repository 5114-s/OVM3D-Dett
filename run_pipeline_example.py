"""
快速使用示例脚本

需要先下载的模型:
1. Grounding DINO: https://huggingface.co/IDEA-Research/GroundingDINO-SwinT-OGC
2. SAM: https://github.com/facebookresearch/segment-anything
3. Fast-SAM3D: https://huggingface.co/facebook/sam-3d-objects (或 tuandao-zenai/sam-3d-objects)
"""

import os
import sys

# 确保路径正确
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

def download_checkpoints():
    """下载所需的模型权重"""
    print("=" * 60)
    print("📦 Download Required Checkpoints")
    print("=" * 60)
    
    checkpoints = {
        "Grounding DINO": {
            "url": "https://huggingface.co/IDEA-Research/GroundingDINO-SwinT-OGC/resolve/main/groundingdinov0.1義SwinT.pt",
            "path": "third_party/Grounded-Segment-Anything/GroundingDINO/weights/"
        },
        "SAM (ViT-H)": {
            "url": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
            "path": "third_party/Grounded-Segment-Anything/sam_weights/"
        },
        "Fast-SAM3D": {
            "url": "https://huggingface.co/tuandao-zenai/sam-3d-objects/tree/main/checkpoints",
            "path": "Fast-SAM3D/checkpoints/hf/"
        }
    }
    
    for name, info in checkpoints.items():
        print(f"\n📥 {name}:")
        print(f"   URL: {info['url']}")
        print(f"   Path: {info['path']}")


def run_with_existing_masks():
    """使用已存在的掩码运行 3D 重建（跳过 Step 1）"""
    print("=" * 60)
    print("🚀 Running with Existing Masks")
    print("=" * 60)
    
    # 示例用法
    from tools.ovm3d_pipeline import FastSAM3DReconstructor, MeshToBBoxExtractor
    import numpy as np
    import cv2
    from PIL import Image
    
    # 配置
    IMAGE_PATH = "path/to/your/image.png"
    MASK_DIR = "path/to/your/masks/"  # 包含 0.png, 1.png, ... 格式的掩码
    OUTPUT_DIR = "./output"
    
    # 加载 Fast-SAM3D
    reconstructor = FastSAM3DReconstructor()
    reconstructor.load_model(
        pipeline_config="Fast-SAM3D/checkpoints/hf/pipeline.yaml"
    )
    
    # 初始化边界框提取器
    bbox_extractor = MeshToBBoxExtractor()
    
    # 读取图像
    image = cv2.imread(IMAGE_PATH)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # 处理每个掩码
    mask_files = sorted([f for f in os.listdir(MASK_DIR) if f.endswith('.png')])
    
    for i, mask_file in enumerate(mask_files):
        mask = cv2.imread(os.path.join(MASK_DIR, mask_file), cv2.IMREAD_GRAYSCALE)
        mask = (mask > 127).astype(np.uint8)
        
        print(f"\n🔄 Processing {mask_file}...")
        
        # 3D 重建
        output = reconstructor.reconstruct(image, mask)
        
        # 保存结果
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        reconstructor.save_ply(output['gs'], f"{OUTPUT_DIR}/object_{i}.ply")
        
        # 提取边界框
        gs_xyz = output['gs']._xyz.detach().cpu().numpy()
        bbox = bbox_extractor.extract_from_pointcloud(gs_xyz)
        
        print(f"   Center: {bbox['center']}")
        print(f"   Dimensions: {bbox['dimensions']}")
        print(f"   Yaw: {bbox['yaw']:.2f} rad")


def run_full_pipeline():
    """运行完整的三步流程"""
    print("=" * 60)
    print("🚀 Running Full OVM3D-Det Pipeline")
    print("=" * 60)
    
    from tools.ovm3d_pipeline import OVM3DPipeline
    
    pipeline = OVM3DPipeline(gpu_id=0)
    
    # 注意: 需要先下载以下模型权重
    # - Grounding DINO checkpoint
    # - SAM checkpoint
    # - Fast-SAM3D checkpoints
    
    pipeline.setup(
        grounded_sam_checkpoint="path/to/sam_vit_h.pth",
        grounding_dino_checkpoint="path/to/groundingdinov0.1.pt",
        fastsam3d_config="Fast-SAM3D/checkpoints/hf/pipeline.yaml"
    )
    
    results = pipeline.run(
        image_path="path/to/image.png",
        text_prompt="chair table floor",
        output_dir="./output"
    )
    
    print("\n✅ Pipeline completed!")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, choices=['download', 'existing_masks', 'full'], 
                       default='download', help='Run mode')
    args = parser.parse_args()
    
    if args.mode == 'download':
        download_checkpoints()
    elif args.mode == 'existing_masks':
        run_with_existing_masks()
    elif args.mode == 'full':
        run_full_pipeline()
