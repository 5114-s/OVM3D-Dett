"""Retry OOM objects - Fixed"""
import os, sys, json, argparse
from pathlib import Path
from tqdm import tqdm
import numpy as np
import cv2
import torch

_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root))
sys.path.insert(0, str(_project_root / 'third_party' / 'Grounded-SAM-2'))
sys.path.insert(0, str(_project_root / 'Fast-SAM3D'))

SUNRGBD_38_CLASSES = "bicycle. books. bottle. chair. cup. laptop. shoes. towel. blinds. window. lamp. shelves. mirror. sink. cabinet. bathtub. door. toilet. desk. box. bookcase. picture. table. counter. bed. night stand. pillow. sofa. television. floor mat. curtain. clothes. stationery. refrigerator. bin. stove. oven. machine."
captions_38 = [c.strip() for c in SUNRGBD_38_CLASSES.strip(". ").split(". ") if c.strip()]

def is_oom(c):
    return np.all(np.array(c) == [-1, -1, -1])

def load_models():
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    from grounding_dino.groundingdino.util.inference import load_model as load_gdino
    from omegaconf import OmegaConf
    import importlib.util
    
    print("Loading Grounding DINO...")
    gd_cfg = str(_project_root / 'third_party' / 'Grounded-SAM-2' / 'grounding_dino' / 'groundingdino' / 'config' / 'GroundingDINO_SwinT_OGC.py')
    gd_ckpt = str(_project_root / 'third_party' / 'Grounded-SAM-2' / 'gdino_checkpoints' / 'groundingdino_swint_ogc.pth')
    gd_model = load_gdino(gd_cfg, gd_ckpt)
    gd_model.eval()
    
    print("Loading SAM 2...")
    original_cwd = os.getcwd()
    gsam2_dir = str(_project_root / 'third_party' / 'Grounded-SAM-2' / 'sam2')
    os.chdir(gsam2_dir)
    sam_cfg = 'configs/sam2.1/sam2.1_hiera_l.yaml'
    sam_ckpt = str(_project_root / 'third_party' / 'Grounded-SAM-2' / 'checkpoints' / 'sam2.1_hiera_large.pt')
    sam_pred = build_sam2(sam_cfg, sam_ckpt, device='cuda')
    sam_pred = SAM2ImagePredictor(sam_pred)
    os.chdir(original_cwd)
    
    print("Loading Fast-SAM3D...")
    cfg_path = str(_project_root / 'Fast-SAM3D' / 'checkpoints' / 'hf' / 'pipeline_unidepth.yaml')
    config = OmegaConf.load(cfg_path)
    config.workspace_dir = str(Path(cfg_path).parent)
    config['ss_generator_config_path'] = "ss_generator_faster.yaml"
    config['slat_generator_config_path'] = "slat_generator_faster.yaml"
    fastsam_root = str(_project_root / 'Fast-SAM3D')
    if fastsam_root not in sys.path:
        sys.path.insert(0, fastsam_root)
    spec = importlib.util.spec_from_file_location("inference", _project_root / 'Fast-SAM3D' / 'notebook' / 'inference.py')
    inf_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(inf_mod)
    Inference = inf_mod.Inference
    
    class Args:
        ss_cache_stride = 3; ss_warmup = 2; ss_order = 1; ss_momentum_beta = 0.5
        slat_thresh = 1.5; slat_warmup = 3; slat_carving_ratio = 0.1
        ss_inference_steps = 15; slat_inference_steps = 15
        mesh_spectral_threshold_low = 0.5; mesh_spectral_threshold_high = 0.7
        enable_acceleration = True; enable_mesh_aggregation = False
    
    inference = Inference(config, compile=False, args=Args())
    args = Args()
    inference._pipeline.ss_params = {
        'ss_cache_stride': args.ss_cache_stride,
        'inference_steps': args.ss_inference_steps,
        'ss_warmup': args.ss_warmup,
        'ss_order': args.ss_order,
        'ss_momentum_beta': args.ss_momentum_beta,
    }
    inference._pipeline.slat_params = {
        'slat_thresh': args.slat_thresh,
        'inference_steps': args.slat_inference_steps,
        'slat_warmup': args.slat_warmup,
        'slat_carving_ratio': args.slat_carving_ratio,
    }
    inference._pipeline.ss_inference_steps = args.ss_inference_steps
    inference._pipeline.slat_inference_steps = args.slat_inference_steps

    inference._pipeline.mesh_params = {
        'mesh_spectral_threshold_low': args.mesh_spectral_threshold_low,
        'mesh_spectral_threshold_high': args.mesh_spectral_threshold_high,
    }
    return gd_model, sam_pred, inference

def segment(image_path, grounding_dino_model, sam2_predictor):
    from grounding_dino.groundingdino.util.inference import predict, load_image
    from torchvision.ops import box_convert
    
    # 加载图像
    image_source, image_transformed = load_image(image_path)
    
    # Step 1: Grounding DINO 检测
    boxes, confidences, labels = predict(
        model=grounding_dino_model,
        image=image_transformed,
        caption=SUNRGBD_38_CLASSES,
        box_threshold=0.3,
        text_threshold=0.25,
        device='cuda'
    )
    
    if boxes is None or len(boxes) == 0:
        return None
    
    # Step 2: SAM 2 生成掩码
    h, w = image_source.shape[:2]
    
    # 转换 boxes 从 cxcywh 归一化格式到 xyxy 像素格式
    boxes_xyxy = boxes * torch.tensor([w, h, w, h], device=boxes.device)
    input_boxes = box_convert(boxes=boxes_xyxy, in_fmt="cxcywh", out_fmt="xyxy").numpy()
    
    # 设置图像 (使用原始图像)
    sam2_predictor.set_image(image_source)
    
    # 使用 bfloat16 加速
    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        masks, scores, logits = sam2_predictor.predict(
            point_coords=None,
            point_labels=None,
            box=input_boxes,
            multimask_output=False
        )
    
    # 处理返回的 masks 格式
    if masks.ndim == 4:
        masks = masks.squeeze(1)
    
    # 确保是 numpy 数组
    if isinstance(masks, torch.Tensor):
        masks = masks.cpu().numpy()
    if isinstance(scores, torch.Tensor):
        scores = scores.cpu().numpy()
    
    return {
        'boxes': input_boxes,
        'labels': list(labels),
        'masks': masks,
        'scores': scores
    }


def reconstruct(image, mask, inference):
    if mask.max() > 1.0: mask = mask.astype(np.float32) / 255.0
    else: mask = mask.astype(np.float32)
    return inference(image, mask, seed=42)

def save_ply(gs, path):
    import plyfile
    pos = gs._xyz.detach().cpu().numpy()
    col = gs._features_dc.detach().cpu().numpy()
    num = pos.shape[0]
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
             ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    v = np.zeros(num, dtype=dtype)
    v['x'], v['y'], v['z'] = pos[:, 0], pos[:, 1], pos[:, 2]
    v['nx'], v['ny'], v['nz'] = 0, 0, 1
    v['red'] = np.clip((col[:, 0, 0] * 0.5 + 0.5) * 255, 0, 255).astype(np.uint8)
    v['green'] = np.clip((col[:, 0, 1] * 0.5 + 0.5) * 255, 0, 255).astype(np.uint8)
    v['blue'] = np.clip((col[:, 0, 2] * 0.5 + 0.5) * 255, 0, 255).astype(np.uint8)
    plyfile.PlyData([plyfile.PlyElement.describe(v, 'vertex')]).write(path)

def extract_bbox(ply_path, prior=None):
    if prior is None: prior = [0.5, 0.5, 0.5]
    import open3d as o3d
    mesh = o3d.io.read_triangle_mesh(ply_path)
    verts = np.asarray(mesh.vertices)
    if len(verts) < 10: return None
    center = verts.mean(axis=0)
    pcd = o3d.geometry.PointCloud(points=o3d.utility.Vector3dVector(verts))
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    normals = np.asarray(pcd.normals)
    normals = normals / (np.linalg.norm(normals, axis=1, keepdims=True) + 1e-6)
    x_axis = normals.mean(axis=0) / (np.linalg.norm(normals.mean(axis=0)) + 1e-6)
    ref = np.array([1, 0, 0]) if abs(x_axis[1]) < 0.9 else np.array([0, 1, 0])
    y_axis = np.cross(ref, x_axis) / (np.linalg.norm(np.cross(ref, x_axis)) + 1e-6)
    R = np.column_stack([x_axis, y_axis, np.cross(x_axis, y_axis)])
    rot_verts = verts @ R.T
    dx, dy, dz = rot_verts.max(axis=0) - rot_verts.min(axis=0)
    dx, dy, dz = max(dx * prior[0], 0.05), max(dy * prior[1], 0.05), max(dz * prior[2], 0.05)
    def rot_y(a): c, s = np.cos(a), np.sin(a); return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    def box_verts(cx, cy, cz, dx, dy, dz, yaw):
        corners = np.array([[-dx/2,-dy/2,-dz/2],[dx/2,-dy/2,-dz/2],[dx/2,-dy/2,dz/2],[-dx/2,-dy/2,dz/2],
                           [-dx/2,dy/2,-dz/2],[dx/2,dy/2,-dz/2],[dx/2,dy/2,dz/2],[-dx/2,dy/2,dz/2]])
        r = np.zeros_like(corners)
        r[:, 0], r[:, 2] = np.cos(yaw)*corners[:, 0]-np.sin(yaw)*corners[:, 2], np.sin(yaw)*corners[:, 0]+np.cos(yaw)*corners[:, 2]
        r[:, 1] = corners[:, 1]
        return r + np.array([cx, cy, cz])
    best, min_loss = None, float('inf')
    for yaw in np.linspace(-np.pi/2, np.pi/2, 37):
        cx, cy, cz = center[0], center[1] - dy/2, center[2]
        rp = rot_verts @ rot_y(yaw).T
        inside = ((rp[:, 0] >= cx-dx/2) & (rp[:, 0] <= cx+dx/2) & (rp[:, 2] >= cz-dz/2) & (rp[:, 2] <= cz+dz/2)).sum()
        ratio = inside / len(rp) if len(rp) > 0 else 0
        if ratio >= 0.5:
            v = box_verts(cx, cy, cz, dx, dy, dz, 0) @ rot_y(-yaw).T
            if 1 - ratio < min_loss: min_loss, best = 1 - ratio, v
    if best is None: best = box_verts(cx, cy, cz, dx, dy, dz, 0)
    best = best @ R.T
    return {'center_cam': best.mean(axis=0).tolist(), 'dimensions': [dz, dy, dx], 'R_cam': R.tolist(), 'vertices': best.tolist()}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input_dir', required=True)
    ap.add_argument('--image_root', default='')
    ap.add_argument('--gpu', default='0')
    ap.add_argument('--checkpoint_interval', type=int, default=10)
    args = ap.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    
    pth_path = os.path.join(args.input_dir, 'info_3d.pth')
    print("Loading {}...".format(pth_path))
    results = torch.load(pth_path, map_location='cpu')
    print("Loaded {} images".format(len(results)))
    
    # 加载图片路径映射
    img_path_map = {}
    json_file = '/data/ZhaoX/OVM3D-Dett/datasets/Omni3D_pl/SUNRGBD_train.json'
    import json
    with open(json_file) as f:
        omni_data = json.load(f)
    for img in omni_data.get('images', []):
        img_path_map[img['id']] = '/data/ZhaoX/OVM3D-Dett/datasets/' + img['file_path']
    
    oom_list = []
    for iid, d in results.items():
        oom_idx = [j for j, c in enumerate(d['center_cam']) if is_oom(c)]
        if oom_idx:
            img_path = img_path_map.get(int(iid))
            oom_list.append((iid, img_path, oom_idx))
    
    print("Found {} images with OOM objects".format(len(oom_list)))
    if oom_list:
        print("Sample path: {}".format(oom_list[0][1]))
    if not oom_list: print("No OOM objects!"); return
    
    gd_model, sam_pred, inference = load_models()
    prior = {'chair': [0.5,1.2,0.5], 'table': [1.0,0.7,0.8], 'bed': [1.8,0.5,1.5], 'sofa': [1.8,0.8,0.8],
             'door': [0.8,2.0,0.1], 'window': [1.2,1.2,0.1], 'cabinet': [0.6,0.8,0.4], 'desk': [1.2,0.7,0.6],
             'box': [0.4,0.3,0.4], 'counter': [1.5,0.9,0.5], 'sink': [0.5,0.3,0.4], 'shelves': [0.8,1.5,0.3],
             'lamp': [0.3,0.8,0.3], 'bathtub': [1.7,0.5,0.6], 'toilet': [0.4,0.5,0.4], 'refrigerator': [0.7,1.8,0.7],
             'television': [1.0,0.6,0.1], 'laptop': [0.35,0.25,0.02], 'bottle': [0.08,0.2,0.08], 'cup': [0.08,0.1,0.08]}
    
    retry_cnt, success_cnt = 0, 0
    for iid, path, oom_idx in tqdm(oom_list, desc="Retrying"):
        if not path or not os.path.exists(path): continue
        print("\n{}: {} OOM objects".format(iid, len(oom_idx)))
        img = cv2.imread(path); img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        seg = segment(path, gd_model, sam_pred)
        if seg is None: continue
        out_dir = os.path.join(args.input_dir, str(iid))
        os.makedirs(os.path.join(out_dir, 'masks'), exist_ok=True)
        os.makedirs(os.path.join(out_dir, 'meshes'), exist_ok=True)
        for obj_idx in oom_idx:
            label = results[iid]['phrases'][obj_idx]
            mask = None
            for j, lbl in enumerate(seg['labels']):
                if lbl.lower() == label.lower(): mask = seg['masks'][j]; break
            if mask is None: print("  {}: cannot re-segment".format(label)); continue
            cv2.imwrite(os.path.join(out_dir, 'masks', "{}_{}.png".format(label.replace(' ', '_'), obj_idx)), mask.astype(np.uint8) * 255)
            try:
                print("  {}: Fast-SAM3D...".format(label))
                out = reconstruct(img, mask, inference)
                ply_path = os.path.join(out_dir, 'meshes', "{}_{}.ply".format(label.replace(' ', '_'), obj_idx))
                save_ply(out['gs'], ply_path)
                out['glb'].export(os.path.join(out_dir, 'meshes', "{}_{}.glb".format(label.replace(' ', '_'), obj_idx)))
                bbox = extract_bbox(ply_path, prior.get(label, [0.5, 0.5, 0.5]))
                if bbox:
                    results[iid]['center_cam'][obj_idx] = np.array(bbox['center_cam'])
                    results[iid]['dimensions'][obj_idx] = bbox['dimensions']
                    results[iid]['R_cam'][obj_idx] = np.array(bbox['R_cam'])
                    results[iid]['boxes3d'][obj_idx] = np.array(bbox['vertices'], dtype=np.float16)
                    print("  {}: SUCCESS".format(label)); success_cnt += 1
            except Exception as e:
                print("  {}: FAILED - {}".format(label, e))
            retry_cnt += 1
        if retry_cnt % args.checkpoint_interval == 0:
            torch.save(results, pth_path); print("Checkpoint: {}/{} OK".format(success_cnt, retry_cnt))
    torch.save(results, pth_path); print("\nDone! {}/{} succeeded".format(success_cnt, retry_cnt))

if __name__ == '__main__': main()
