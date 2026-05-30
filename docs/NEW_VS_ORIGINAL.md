# 新版 vs 原版流程对比

## 📊 原版 OVM3D-Det 流程

```
Step 1: UniDepth (深度估计)
    输入: RGB 图像
    输出: depth/*.npy

Step 2: Grounding-SAM 2 (分割)
    输入: RGB 图像 + SUNRGBD 38 类文本提示
    输出: mask/*.npy + ground_mask/*.npy

Step 3: generate_pseudo_bbox.py
    输入: mask + depth + K(内参) + category prior
    处理: 点云生成 → PCA → 射线追踪优化
    输出: info_3d.pth (boxes3d, center_cam, dimensions, R_cam)

Step 4: transform_to_coco.py
    输入: info_3d.pth
    输出: datasets/Omni3D_pl/{DATASET}_{train,val}.json
```

## 📊 新版 (Grounding-SAM 2 + Fast-SAM3D) 流程

```
Step 1: Grounding-SAM 2 (分割) ✅
    输入: RGB 图像 + SUNRGBD 38 类文本提示
    输出: mask/*.npy

Step 2: Fast-SAM3D (3D 重建) ✅
    输入: RGB 图像 + mask
    输出: 3D Mesh (.ply, .glb)

Step 3: 边界框提取 ✅ (与原版 Step 3 逻辑一致)
    输入: 3D Mesh
    处理: 
        - 地面平面检测
        - PCA 在 XZ 平面估计 yaw
        - 射线追踪 + inside_ratio 优化
        - 类别先验约束
    输出: boxes3d, center_cam, dimensions, R_cam

Step 4: 转换为 COCO 格式 ✅ (可选)
    输入: 伪标签
    输出: coco_format/*.json
```

## 🔑 Step 3 关键对比

| 功能 | 原版 | 新版 |
|------|------|------|
| 地面检测 | ✅ 使用 info_ground | ✅ 自动从 Mesh 检测 |
| 地面旋转约束 | ✅ rotation_matrix_from_vectors | ✅ 相同 |
| PCA Yaw | ✅ XZ 平面 | ✅ 相同 |
| 射线追踪 | ✅ calc_dis_ray_tracing | ✅ 相同 |
| Inside Ratio | ✅ calc_inside_ratio | ✅ 相同 |
| 先验约束 | ✅ | ✅ |
| 输出格式 | boxes3d, center_cam, dimensions, R_cam | ✅ 相同 |

### 关键差异

| 方面 | 原版 | 新版 |
|------|------|------|
| 输入点云来源 | mask × depth | Fast-SAM3D Mesh |
| 点云密度 | 稀疏 | 密集 |
| 点云精度 | 受深度图质量影响 | 更高 (3D Mesh) |

## 🚀 使用方法

### 新版完整流程

```bash
# Step 1-4 (包含 COCO 转换)
python tools/generate_pseudo_label_gsam.py \
    --image_dir datasets/SUNRGBD/images \
    --dataset SUNRGBD \
    --output_dir pseudo_label_gsam \
    --transform_to_coco \
    --gpu 0
```

### 仅使用 Step 1-3

```bash
python tools/generate_pseudo_label_gsam.py \
    --image_dir datasets/SUNRGBD/images \
    --dataset SUNRGBD \
    --output_dir pseudo_label_gsam \
    --gpu 0
```

### 单独使用边界框提取

```bash
python -m cubercnn.generate_label.mesh_bbox_extractor_improved \
    --mesh output/meshes/chair_0.ply \
    --category chair \
    --detect_ground
```

## 📁 输出文件

```
pseudo_label_gsam/
├── image_001/
│   ├── masks/
│   │   └── chair_0.png, table_1.png, ...
│   └── meshes/
│       ├── chair_0.ply, chair_0.glb
│       ├── table_1.ply, table_1.glb
│       └── ...
├── info_3d.pth          # 与原版格式兼容
├── results.json          # 详细结果
└── coco_format/
    └── SUNRGBD_pseudo_labels.json  # COCO 格式
```

## ⚠️ 依赖模型

| 模型 | 用途 | 大小 | 状态 |
|------|------|------|------|
| Grounding DINO | Step 1 检测 | ~2 GB | ❌ 已下载 |
| SAM 2 Hiera-L | Step 1 分割 | ~2.4 GB | ❌ 已下载 |
| Fast-SAM3D (ss_generator) | Step 2 重建 | 6.69 GB | ❌ 缺失 |
| Fast-SAM3D (slat_generator) | Step 2 重建 | 4.91 GB | ❌ 缺失 |

## 📝 代码文件

| 文件 | 功能 |
|------|------|
| `tools/generate_pseudo_label_gsam.py` | 主脚本 (Step 1-4) |
| `cubercnn/generate_label/mesh_bbox_extractor.py` | 基础边界框提取 |
| `cubercnn/generate_label/mesh_bbox_extractor_improved.py` | 改进版 (与原版一致) |
