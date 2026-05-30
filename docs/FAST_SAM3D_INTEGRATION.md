# Grounding-SAM 2 + Fast-SAM3D 伪标签生成流程

## 📊 完整流程

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    伪标签生成完整流程                                        │
└─────────────────────────────────────────────────────────────────────────────┘

Step 1: Grounding-SAM 2 (图像分割)
────────────────────────────────────────────────────────────
    输入: image.png + "bicycle. books. bottle. chair. cup. laptop. shoes. towel. blinds. window. lamp. shelves. mirror. sink. cabinet. bathtub. door. toilet. desk. box. bookcase. picture. table. counter. bed. night stand. pillow. sofa. television. floor mat. curtain. clothes. stationery. refrigerator. bin. stove. oven. machine."
                │
                ▼
    ┌─────────────────────────────────────────┐
    │  Grounding DINO (文本检测)              │
    │  输出: BBox (x1,y1,x2,y2) + 类别       │
    └─────────────────────────────────────────┘
                │
                ▼
    ┌─────────────────────────────────────────┐
    │  SAM (实例分割)                         │
    │  输入: BBox                             │
    │  输出: 精细掩码 mask                    │
    └─────────────────────────────────────────┘
                │
                ▼
    输出: mask_0_chair.png, mask_1_table.png, ...


Step 2: Fast-SAM3D (3D 重建)
────────────────────────────────────────────────────────────
    输入: image.png + mask_chair.png
                │
                ▼
    ┌─────────────────────────────────────────┐
    │  Sparse Structure Generator (SSG)        │
    │  - 生成稀疏结构点云                      │
    │  - 预测位姿 (R, t, s)                   │
    └─────────────────────────────────────────┘
                │
                ▼
    ┌─────────────────────────────────────────┐
    │  SLaT Generator                         │
    │  - 生成结构化潜在表示                    │
    │  - 3D 几何一致性                        │
    └─────────────────────────────────────────┘
                │
                ▼
    ┌─────────────────────────────────────────┐
    │  Mesh Decoder                           │
    │  - 输出 3D Mesh (高斯溅射)              │
    └─────────────────────────────────────────┘
                │
                ▼
    输出: chair.ply, chair.glb


Step 3: 从 Mesh 提取 3D 边界框
────────────────────────────────────────────────────────────
    输入: chair.ply (点云)
                │
                ▼
    ┌─────────────────────────────────────────┐
    │  PCA 分析                               │
    │  - 确定主方向 (yaw)                     │
    │  - 计算边界框尺寸                       │
    └─────────────────────────────────────────┘
                │
                ▼
    输出: center_cam, dimensions, R_cam, vertices


最终输出: info_3d.pth (与原有格式兼容)
────────────────────────────────────────────────────────────
```

## 🚀 使用方法

### 1. 下载所需模型

```bash
# 模型下载列表
1. Grounding DINO:
   https://huggingface.co/IDEA-Research/GroundingDINO-SwinT-OGC

2. SAM 2 Hiera-L:
   https://github.com/facebookresearch/segment-anything

3. Fast-SAM3D:
   https://huggingface.co/tuandao-zenai/sam-3d-objects
   (需要: ss_generator.ckpt ~6.7GB, slat_generator.ckpt ~4.9GB)
```

### 2. 运行完整流程

```bash
python tools/generate_pseudo_label_gsam.py \
    --image_dir datasets/SUNRGBD/images \
    --dataset SUNRGBD \
    --output_dir pseudo_label_gsam \
    --grounding_dino_checkpoint third_party/Grounded-SAM-2/gdino_checkpoints/groundingdino_swint_ogc.pth \
    --sam_checkpoint third_party/Grounded-SAM-2/checkpoints/sam2.1_hiera_large.pt \
    --fastsam3d_config Fast-SAM3D/checkpoints/hf/pipeline.yaml \
    --gpu 0
```

### 3. 输出格式

```
pseudo_label_gsam/
├── image_001/
│   ├── masks/
│   │   ├── chair_0.png
│   │   ├── table_1.png
│   │   └── floor_2.png
│   └── meshes/
│       ├── chair_0.ply
│       ├── chair_0.glb
│       ├── table_1.ply
│       └── table_1.glb
├── image_002/
│   └── ...
├── info_3d.pth        # 伪标签 (与原有格式兼容)
└── results.json       # JSON 格式结果
```

## 🔄 与原有流程对比

| 方面 | 原流程 (Depth) | 新流程 (Fast-SAM3D) |
|------|---------------|---------------------|
| Step 1 | 深度估计模型 | Grounding-SAM 2 |
| Step 2 | 点云生成 | 3D Mesh 生成 |
| Step 3 | PCA + 射线追踪 | PCA |
| 精度 | 中等 | 更高 |
| 速度 | 快 | 慢 (~230s/物体) |
| 模型大小 | ~500MB | ~15GB |

## ⚠️ 注意事项

1. **Fast-SAM3D 模型缺失**: 
   - `ss_generator.ckpt` (6.69 GB)
   - `slat_generator.ckpt` (4.91 GB)
   - 这两个文件必须下载才能运行 Step 2

2. **显存要求**: 
   - Fast-SAM3D 需要 ~24GB 显存
   - 建议使用 A100 或 3090+

3. **处理速度**:
   - 单物体 ~230 秒
   - 场景 (~10 物体) ~30 分钟

## 📝 代码位置

| 功能 | 文件 |
|------|------|
| 完整流程 | `tools/generate_pseudo_label_gsam.py` |
| Step 1 (分割) | `tools/generate_pseudo_label_gsam.py` (segment_with_grounding_sam2) |
| Step 2 (重建) | `tools/generate_pseudo_label_gsam.py` (reconstruct_3d) |
| Step 3 (边界框) | `tools/generate_pseudo_label_gsam.py` (extract_bbox_from_ply) |
| Mesh 提取器 | `cubercnn/generate_label/mesh_bbox_extractor.py` |
