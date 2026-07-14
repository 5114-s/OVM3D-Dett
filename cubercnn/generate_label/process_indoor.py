# Copyright (c) Meta Platforms, Inc. and affiliates
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import torch
from cubercnn import util
import math
from cubercnn.generate_label.util import *
from cubercnn.generate_label.raytrace import calc_dis_ray_tracing, calc_inside_ratio
from cubercnn.generate_label.projection_selection import (
    disabled_selection_metric,
    get_projection_selection_cfg,
    select_projected_candidate,
)
from tqdm import tqdm
from sklearn.cluster import DBSCAN

try:
    from scipy.spatial import cKDTree
except Exception:
    cKDTree = None


def create_uv_depth(depth, mask=None):
    """Generate UV-Depth point cloud."""
    if mask is not None:
        depth = depth * mask
    x, y = np.meshgrid(
        np.linspace(0, depth.shape[1] - 1, depth.shape[1]),
        np.linspace(0, depth.shape[0] - 1, depth.shape[0])
    )
    uv_depth = np.stack((x, y, depth), axis=-1)
    uv_depth = uv_depth.reshape(-1, 3)
    return uv_depth[uv_depth[:, 2] != 0]


def _env_bool(name, default=False):
    value = os.environ.get(name)
    if value is None:
        return bool(default)
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _env_float(name, default):
    value = os.environ.get(name)
    if value is None or str(value).strip() == "":
        return float(default)
    try:
        return float(value)
    except ValueError:
        return float(default)


def _env_int(name, default):
    value = os.environ.get(name)
    if value is None or str(value).strip() == "":
        return int(default)
    try:
        return int(value)
    except ValueError:
        return int(default)


def get_bidir_cluster_cfg():
    """Configuration for the optional OpenBox/DFU-style point cleaner.

    The defaults are conservative and the whole module is disabled unless
    OVM3D_USE_BIDIR_CLUSTER=1 is set by the caller.
    """
    return {
        "enabled": _env_bool("OVM3D_USE_BIDIR_CLUSTER", False),
        "eps": _env_float("OVM3D_CLUSTER_EPS", 0.18),
        "min_samples": _env_int("OVM3D_CLUSTER_MIN_SAMPLES", 6),
        "delta": _env_float("OVM3D_CLUSTER_DELTA", 0.14),
        "alpha": _env_float("OVM3D_CLUSTER_ALPHA", 0.20),
        "beta": _env_float("OVM3D_CLUSTER_BETA", 0.25),
        "max_points": _env_int("OVM3D_CLUSTER_MAX_POINTS", 1200),
        "min_keep_points": _env_int("OVM3D_CLUSTER_MIN_KEEP_POINTS", 12),
        "topk": _env_int("OVM3D_CLUSTER_TOPK", 2),
        "bbox_padding": _env_float("OVM3D_CLUSTER_BBOX_PADDING", 0.03),
    }


def _subsample_points(points, max_points):
    if points.shape[0] <= max_points:
        return points
    inds = np.random.choice(points.shape[0], int(max_points), replace=False)
    return points[inds]


def _nearest_ratio(src, dst, threshold):
    if src.shape[0] == 0 or dst.shape[0] == 0:
        return 0.0
    if cKDTree is not None:
        tree = cKDTree(dst)
        try:
            dists, _ = tree.query(src, k=1, workers=-1)
        except TypeError:
            dists, _ = tree.query(src, k=1)
        return float(np.mean(dists < threshold))
    # Fallback for environments without scipy: chunked pairwise distances.
    hit = 0
    chunk = 512
    thr2 = float(threshold) ** 2
    for start in range(0, src.shape[0], chunk):
        cur = src[start:start + chunk]
        dist2 = ((cur[:, None, :] - dst[None, :, :]) ** 2).sum(axis=2)
        hit += int(np.any(dist2 < thr2, axis=1).sum())
    return float(hit / max(src.shape[0], 1))


def _bbox_mask_from_xyxy(box, shape_hw, padding=0.0):
    h, w = shape_hw
    arr = np.asarray(box).reshape(-1).astype(float)
    if arr.shape[0] < 4 or np.any(~np.isfinite(arr[:4])):
        return None
    x1, y1, x2, y2 = arr[:4]
    if x2 <= x1 or y2 <= y1:
        return None
    pad_x = (x2 - x1) * float(padding)
    pad_y = (y2 - y1) * float(padding)
    x1 = int(max(0, math.floor(x1 - pad_x)))
    y1 = int(max(0, math.floor(y1 - pad_y)))
    x2 = int(min(w, math.ceil(x2 + pad_x)))
    y2 = int(min(h, math.ceil(y2 + pad_y)))
    if x2 <= x1 or y2 <= y1:
        return None
    mask = np.zeros((h, w), dtype=np.float32)
    mask[y1:y2, x1:x2] = 1.0
    return mask


def bidirectional_cluster_filter(core_points, context_points, cfg):
    """Keep context clusters that are mutually close to the instance core.

    This is a single-view adaptation of OpenBox/DFU-style cluster association:
    cluster a broader context point cloud, then retain only clusters that have
    bidirectional proximity support with the eroded instance point cloud.
    """
    metrics = {
        "enabled": bool(cfg.get("enabled", False)),
        "used": False,
        "reason": "disabled",
        "core_points": int(core_points.shape[0]) if core_points is not None else 0,
        "context_points": int(context_points.shape[0]) if context_points is not None else 0,
        "kept_points": 0,
        "clusters": 0,
        "kept_clusters": 0,
    }
    if not cfg.get("enabled", False):
        return core_points, metrics
    if core_points is None or core_points.shape[0] < int(cfg["min_keep_points"]):
        metrics["reason"] = "too_few_core_points"
        return core_points, metrics
    if context_points is None or context_points.shape[0] < int(cfg["min_keep_points"]):
        metrics["reason"] = "too_few_context_points"
        return core_points, metrics

    max_points = int(cfg["max_points"])
    core_ref = _subsample_points(np.asarray(core_points, dtype=np.float32), max_points)
    context_ref = _subsample_points(np.asarray(context_points, dtype=np.float32), max_points)
    if context_ref.shape[0] < int(cfg["min_keep_points"]):
        metrics["reason"] = "too_few_context_after_sample"
        return core_points, metrics

    labels = DBSCAN(
        eps=float(cfg["eps"]),
        min_samples=int(cfg["min_samples"]),
    ).fit_predict(context_ref)
    cluster_ids = [int(x) for x in sorted(set(labels.tolist())) if int(x) >= 0]
    metrics["clusters"] = len(cluster_ids)
    if not cluster_ids:
        metrics["reason"] = "no_clusters"
        return core_points, metrics

    candidates = []
    for cluster_id in cluster_ids:
        cluster = context_ref[labels == cluster_id]
        if cluster.shape[0] < int(cfg["min_keep_points"]):
            continue
        cluster_to_core = _nearest_ratio(cluster, core_ref, float(cfg["delta"]))
        core_to_cluster = _nearest_ratio(core_ref, cluster, float(cfg["delta"]))
        score = math.sqrt(max(cluster_to_core, 0.0) * max(core_to_cluster, 0.0))
        candidates.append((score, cluster_to_core, core_to_cluster, cluster_id, cluster))

    if not candidates:
        metrics["reason"] = "no_valid_clusters"
        return core_points, metrics

    candidates.sort(key=lambda x: x[0], reverse=True)
    passed = [
        item for item in candidates
        if item[1] >= float(cfg["alpha"]) and item[2] >= float(cfg["beta"])
    ]
    if not passed:
        metrics["reason"] = "no_bidirectional_match"
        metrics["best_score"] = float(candidates[0][0])
        metrics["best_cluster_to_core"] = float(candidates[0][1])
        metrics["best_core_to_cluster"] = float(candidates[0][2])
        return core_points, metrics

    keep_topk = max(1, int(cfg["topk"]))
    kept_clusters = passed[:keep_topk]
    cleaned = np.concatenate([item[4] for item in kept_clusters], axis=0)
    if cleaned.shape[0] < int(cfg["min_keep_points"]):
        metrics["reason"] = "too_few_cleaned_points"
        return core_points, metrics

    metrics.update({
        "used": True,
        "reason": "valid",
        "kept_points": int(cleaned.shape[0]),
        "kept_clusters": int(len(kept_clusters)),
        "best_score": float(candidates[0][0]),
        "best_cluster_to_core": float(candidates[0][1]),
        "best_core_to_cluster": float(candidates[0][2]),
    })
    return cleaned.astype(np.float32), metrics


def process_ground(info_ground, im_id, depth, input_folder, K):
    if im_id not in info_ground or not info_ground[im_id]:
        return False, None

    ground_mask = np.load(f'{input_folder}/ground_mask/{im_id}.npy')
    ground_mask = erode_mask(ground_mask.astype(float), 4, 4)

    ground_mask = ground_mask[np.argmax(info_ground[im_id]['conf'])]
    ground_depth = depth * ground_mask.squeeze()

    uv_depth = create_uv_depth(ground_depth)
    pseudo_lidar_ground = project_image_to_cam(uv_depth, np.array(K))

    # If the number of points is less than 10, the ground plane is not reliable
    if pseudo_lidar_ground.shape[0] > 10:
        ground_equ = extract_ground(pseudo_lidar_ground)
        return True, ground_equ
    return False, None


def process_instances(
    mask_instance,
    depth,
    K,
    info_i,
    cat_prior,
    has_ground,
    ground_equ,
    cluster_cfg=None,
    raw_mask_instance=None,
    projection_cfg=None,
):
    """Process each instance to generate 3D bounding boxes."""
    boxes3d = []
    center_cam_list = []
    dimension_list = []
    R_cam_list = []
    cluster_stats = []
    projection_stats = []

    for mask_ind, cur_mask in enumerate(mask_instance):
        # Remove instances with masks that are too small which are not reliable
        if cur_mask.sum() < 10:
            # Fill default values for invalid masks
            boxes3d.append(np.full((8, 3), -1))
            center_cam_list.append(-1 * np.ones(3))
            dimension_list.append([-1, -1, -1])
            R_cam_list.append(-1 * np.ones((3, 3)))
            cluster_stats.append({"enabled": bool(cluster_cfg and cluster_cfg.get("enabled", False)), "used": False, "reason": "small_mask"})
            projection_stats.append(
                disabled_selection_metric("small_mask")
                if not (projection_cfg and projection_cfg.get("enabled", False))
                else {
                    "enabled": True,
                    "eligible": False,
                    "switched": False,
                    "reason": "small_mask",
                }
            )
            continue

        # Generate pseudo lidar data
        cur_mask_2d = cur_mask.squeeze(0)
        cur_depth = depth * cur_mask_2d
        uv_depth = create_uv_depth(cur_depth)
        pseudo_lidar = project_image_to_cam(uv_depth, np.array(K))
        # This defensive fallback belongs to the separate point-cluster
        # experiment.  Keep it gated so disabling that experiment reproduces
        # the released point-to-box path exactly.
        if (
            cluster_cfg is not None
            and cluster_cfg.get("enabled", False)
            and pseudo_lidar.shape[0] < 3
        ):
            boxes3d.append(np.full((8, 3), -1))
            center_cam_list.append(-1 * np.ones(3))
            dimension_list.append([-1, -1, -1])
            R_cam_list.append(-1 * np.ones((3, 3)))
            cluster_stats.append({"enabled": bool(cluster_cfg and cluster_cfg.get("enabled", False)), "used": False, "reason": "too_few_depth_points"})
            projection_stats.append(
                {
                    "enabled": bool(projection_cfg and projection_cfg.get("enabled", False)),
                    "eligible": False,
                    "switched": False,
                    "reason": "too_few_depth_points",
                }
            )
            continue

        cluster_metric = {"enabled": bool(cluster_cfg and cluster_cfg.get("enabled", False)), "used": False, "reason": "disabled"}
        if cluster_cfg is not None and cluster_cfg.get("enabled", False):
            context_points = None
            boxes = info_i.get("boxes", [])
            if mask_ind < len(boxes):
                context_mask = _bbox_mask_from_xyxy(
                    boxes[mask_ind],
                    depth.shape[:2],
                    padding=float(cluster_cfg["bbox_padding"]),
                )
                if context_mask is not None:
                    context_uv_depth = create_uv_depth(depth, context_mask)
                    if context_uv_depth.shape[0] > 0:
                        context_points = project_image_to_cam(context_uv_depth, np.array(K))
            pseudo_lidar, cluster_metric = bidirectional_cluster_filter(
                pseudo_lidar,
                context_points,
                cluster_cfg,
            )

        # Estimate 3D bounding box with llm-generated prior
        category_name = info_i["phrases"][mask_ind]
        prior = np.array(cat_prior[category_name])
        raw_mask_2d = None
        if raw_mask_instance is not None and mask_ind < len(raw_mask_instance):
            raw_mask_2d = raw_mask_instance[mask_ind].squeeze()
        bbox_params = estimate_bbox(
            pseudo_lidar,
            prior,
            ground_equ if has_ground else None,
            raw_mask=raw_mask_2d,
            core_mask=cur_mask_2d,
            depth=depth,
            K=K,
            projection_cfg=projection_cfg,
            return_selection_metrics=True,
        )

        boxes3d.extend(bbox_params[0])
        center_cam_list.extend(bbox_params[1])
        dimension_list.extend(bbox_params[2])
        R_cam_list.extend(bbox_params[3])
        cluster_stats.append(cluster_metric)
        projection_stats.append(bbox_params[4])

    return (
        boxes3d,
        center_cam_list,
        dimension_list,
        R_cam_list,
        cluster_stats,
        projection_stats,
    )


def process_indoor(dataset, cat_prior, input_folder, output_folder):
    """Main function to process indoor data."""

    # vis_folder = os.path.join(output_folder, 'bbox_3d')
    # util.mkdir_if_missing(vis_folder)
    # vis_folder = os.path.join(output_folder, 'pseudo_lidar')
    # util.mkdir_if_missing(vis_folder)

    info = torch.load(os.path.join(input_folder, 'info.pth'))
    info_ground = torch.load(os.path.join(input_folder, 'info_ground.pth'))
    cluster_cfg = get_bidir_cluster_cfg()
    projection_cfg = get_projection_selection_cfg()
    cluster_totals = {
        "enabled": bool(cluster_cfg.get("enabled", False)),
        "instances": 0,
        "used": 0,
        "fallback": 0,
    }
    projection_totals = {
        "enabled": bool(projection_cfg.get("enabled", False)),
        "instances": 0,
        "eligible": 0,
        "switched": 0,
        "kept_original": 0,
        "invalid": 0,
    }

    for idx in tqdm(range(len(dataset._dataset))):
        im_id = dataset._dataset[idx]['image_id']
        if im_id not in info or not info[im_id]:
            continue

        depth = np.load(f'{input_folder}/depth/{im_id}.npy')
        raw_mask = np.load(f'{input_folder}/mask/{im_id}.npy')
        mask = adaptive_erode_mask(raw_mask.astype(float), 12, 2, 6, 2)
        K = dataset._dataset[idx]['K']

        # Process floor data and estimate ground plane
        has_ground, ground_equ = process_ground(info_ground, im_id, depth, input_folder, K)

        # # Generate whole pseudo lidar data
        # whole_mask = mask.squeeze(1).sum(0)
        # whole_mask[whole_mask > 1] = 1
        # pseudo_lidar = create_uv_depth(depth, whole_mask)
        # pseudo_lidar = project_image_to_cam(pseudo_lidar, np.array(K))
        # np.save(f'{output_folder}/pseudo_lidar/{im_id}.npy', pseudo_lidar[:, :3])

        # Process instances and generate 3D bounding boxes
        boxes3d, center_cam_list, dimension_list, R_cam_list, cluster_stats, projection_stats = process_instances(
            mask,
            depth,
            K,
            info[im_id],
            cat_prior,
            has_ground,
            ground_equ,
            cluster_cfg,
            raw_mask,
            projection_cfg,
        )
        if cluster_cfg.get("enabled", False):
            cluster_totals["instances"] += len(cluster_stats)
            cluster_totals["used"] += sum(1 for item in cluster_stats if item.get("used", False))
            cluster_totals["fallback"] += sum(1 for item in cluster_stats if not item.get("used", False))
        if projection_cfg.get("enabled", False):
            projection_totals["instances"] += len(projection_stats)
            projection_totals["eligible"] += sum(
                1 for item in projection_stats if item.get("eligible", False)
            )
            projection_totals["switched"] += sum(
                1 for item in projection_stats if item.get("switched", False)
            )
            projection_totals["kept_original"] += sum(
                1 for item in projection_stats if item.get("reason") == "kept_original"
            )
            projection_totals["invalid"] += sum(
                1
                for item in projection_stats
                if item.get("reason")
                in {
                    "small_mask",
                    "too_few_depth_points",
                    "no_original_candidate",
                    "missing_projection_inputs",
                    "invalid_core_mask",
                    "invalid_projection_score",
                }
            )

        # Update info dictionary
        info[im_id].update({
            'boxes3d': boxes3d,
            'center_cam': center_cam_list,
            'dimensions': dimension_list,
            'R_cam': R_cam_list,
            'bidir_cluster_stats': cluster_stats,
            'projection_selection_stats': projection_stats,
        })

        # # Save 3D bounding boxes
        # np.save(f'{output_folder}/bbox_3d/{im_id}.npy', np.array(boxes3d))

    # Save updated info
    if cluster_cfg.get("enabled", False):
        info["_bidir_cluster_summary"] = cluster_totals
    if projection_cfg.get("enabled", False):
        info["_projection_selection_summary"] = projection_totals
    torch.save(info, os.path.join(input_folder, 'info_3d.pth'))



def estimate_bbox(
    in_pc,
    prior,
    ground_equ=None,
    raw_mask=None,
    core_mask=None,
    depth=None,
    K=None,
    projection_cfg=None,
    return_selection_metrics=False,
):
    fix_candidate_consistency = _env_bool(
        "OVM3D_FIX_CANDIDATE_CONSISTENCY", False
    )
    if projection_cfg is None:
        projection_cfg = get_projection_selection_cfg()
    use_projection_selection = bool(projection_cfg.get("enabled", False))
    selection_metric = {
        "enabled": use_projection_selection,
        "eligible": False,
        "switched": False,
        "reason": "direct_fit" if use_projection_selection else "disabled",
    }

    def finish(values):
        if return_selection_metrics:
            return (*values, selection_metric)
        return values

    # Subsample input point cloud if needed
    if in_pc.shape[0] > 500:
        rand_ind = np.random.randint(0, in_pc.shape[0], 500)
        in_pc = in_pc[rand_ind]

    w, h, l = prior

    # rotate the point cloud to align with the ground plane
    if ground_equ is not None:
        dot_product = np.dot([0, -1, 0], ground_equ[:3])
        if dot_product <= 0:
            ground_equ = -ground_equ
        new_ground_equ = np.array([0, -1, 0, point_to_plane_distance(ground_equ, 0, 0, 0)])
        rotation_matrix = rotation_matrix_from_vectors([0, -1, 0], ground_equ[:3])
    else:
        rotation_matrix = np.eye(3)
    
    rotated_pc = np.dot(in_pc, rotation_matrix)

    # PCA to determine yaw
    pca = PCA(2)
    pca.fit(rotated_pc[:, [0, 2]])
    yaw_vec = pca.components_[0, :]
    yaw = np.arctan2(yaw_vec[1], yaw_vec[0])

    # Rotate the point cloud to align with the x-axis and z-axis
    rotated_pc_2 = rotate_y(yaw) @ rotated_pc.T
    x_min, x_max = rotated_pc_2[0, :].min(), rotated_pc_2[0, :].max()
    y_min, y_max = rotated_pc_2[1, :].min(), rotated_pc_2[1, :].max()
    z_min, z_max = rotated_pc_2[2, :].min(), rotated_pc_2[2, :].max()

    dx, dy, dz = x_max - x_min, y_max - y_min, z_max - z_min
    cx, cy, cz = (x_min + x_max) / 2, (y_min + y_max) / 2, (z_min + z_max) / 2

    if dy < h * 0.5:
        dy = h
        if ground_equ is not None:
            cdis = point_to_plane_distance(new_ground_equ, cx, cy, cz)
            # If an object is close to the ground but not directly touching it, place its bottom surface on the ground.
            if cdis - dy / 2 < 0.5:
                cy += cdis - dy / 2

    vertives_list, center_cam_list, dimension_list, R_cam_list = [], [], [], []

    # If the size of the object is in a reasonable range, we will directly use it to generate the 3D bounding box.
    # Otherwise, we will try to find the more reasonable size.
    if (l * 0.5 <= dx and w * 0.5 <= dz) or (l * 0.5 <= dz and w * 0.5 <= dx):
        vertives = convert_box_vertices(cx, cy, cz, dx, dy, dz, 0).astype(np.float16)
        vertives = np.dot(rotate_y(-yaw), vertives.T).T
        vertives = np.dot(vertives, rotation_matrix.T)
        vertives_list.append(vertives)
        center_cam = vertives.mean(0)
        dimension = [dz, dy, dx]
        R_cam = rotation_matrix @ rotate_y(-yaw)
        center_cam_list.append(center_cam)
        dimension_list.append(dimension)
        R_cam_list.append(R_cam)
    else:
        # generate all the proposal boxes.
        possible_bboxs = generate_possible_bboxs(cx, cz, dx, dz, w, l)
        min_loss, min_vertives = float('inf'), None
        min_dimension = None
        fallback_inside_ratio = float('-inf')
        fallback_vertives = None
        fallback_dimension = None
        original_index = None
        candidates = []
        candidate_R_cam = rotation_matrix @ rotate_y(-yaw)
        
        # find the best proposal box.
        for candidate_index, possible_bbox in enumerate(possible_bboxs):
            x_min, x_max, z_min, z_max = possible_bbox
            dx, dz = x_max - x_min, z_max - z_min
            cx, cz = (x_min + x_max) / 2, (z_min + z_max) / 2
            inside_ratio = calc_inside_ratio(rotated_pc_2, x_min, x_max, z_min, z_max)
            vertives = convert_box_vertices(cx, cy, cz, dx, dy, dz, 0).astype(np.float16)
            vertives = np.dot(rotate_y(-yaw), vertives.T).T
            new_cx, new_cz = vertives[:, 0].mean(), vertives[:, 2].mean()

            # calculate the ray tracing loss and inside ratio loss.
            pc_tensor = torch.from_numpy(rotated_pc).float()
            loss_ray_tracing = calc_dis_ray_tracing(torch.Tensor([dz, dx]), torch.Tensor([yaw]), pc_tensor[:, [0, 2]], torch.Tensor([new_cx, new_cz]))
            loss_inside_ratio = 1 - inside_ratio

            loss = loss_ray_tracing + 5 * loss_inside_ratio

            if use_projection_selection:
                vertices_cam = np.dot(vertives, rotation_matrix.T)
                candidates.append(
                    {
                        "index": int(candidate_index),
                        "vertices_aligned": vertives,
                        "vertices_cam": vertices_cam,
                        "center_cam": vertices_cam.mean(0),
                        "R_cam": candidate_R_cam,
                        "dimensions_xyz": np.array([dx, dy, dz], dtype=np.float64),
                        "dimension_whl": [dz, dy, dx],
                        "inside_ratio": float(inside_ratio),
                        "released_loss": float(loss),
                    }
                )

            if (
                fix_candidate_consistency
                and np.isfinite(inside_ratio)
                and np.all(np.isfinite(vertives))
                and inside_ratio > fallback_inside_ratio
            ):
                # Sparse masks can make every ray-tracing loss infinite. Keep
                # a deterministic fallback from the unchanged proposal set so
                # the selected vertices can never remain None.
                fallback_inside_ratio = float(inside_ratio)
                fallback_vertives = vertives
                fallback_dimension = [dz, dy, dx]

            if loss < min_loss:
                min_loss = loss
                min_vertives = vertives
                original_index = int(candidate_index)
                if fix_candidate_consistency:
                    # Keep every exported field tied to the candidate that
                    # actually minimized the unchanged original objective.
                    # The original code used dx/dz left over from the final
                    # loop iteration, which can swap length and width relative
                    # to min_vertives.
                    min_dimension = [dz, dy, dx]

        if use_projection_selection and original_index is None and candidates:
            # With the released sparse-ray behavior this branch is normally
            # unreachable.  It only prevents an invalid label when every
            # numerical score is non-finite.
            original_index = max(
                range(len(candidates)),
                key=lambda ind: candidates[ind]["inside_ratio"],
            )
            min_vertives = candidates[original_index]["vertices_aligned"]

        selected_index = original_index
        if use_projection_selection and original_index is not None:
            selected_index, selection_metric = select_projected_candidate(
                candidates,
                original_index,
                raw_mask,
                core_mask,
                depth,
                K,
                projection_cfg,
            )
            if selected_index is not None and selection_metric.get("switched", False):
                selected = candidates[int(selected_index)]
                min_vertives = selected["vertices_aligned"]
                # This is not the separate global consistency repair: only an
                # actually replaced candidate exports its own dimensions.
                min_dimension = selected["dimension_whl"]

        if fix_candidate_consistency and min_vertives is None:
            if fallback_vertives is None:
                return finish(
                    (
                        [np.full((8, 3), -1)],
                        [-1 * np.ones(3)],
                        [[-1, -1, -1]],
                        [-1 * np.ones((3, 3))],
                    )
                )
            min_vertives = fallback_vertives
            min_dimension = fallback_dimension
        
        min_vertives = np.dot(min_vertives, rotation_matrix.T)
        vertives_list.append(min_vertives)
        center_cam = min_vertives.mean(0)
        if use_projection_selection and selection_metric.get("switched", False):
            dimension = min_dimension
        else:
            dimension = min_dimension if fix_candidate_consistency else [dz, dy, dx]
        R_cam = rotation_matrix @ rotate_y(-yaw)
        center_cam_list.append(center_cam)
        dimension_list.append(dimension)
        R_cam_list.append(R_cam)

    return finish((vertives_list, center_cam_list, dimension_list, R_cam_list))
