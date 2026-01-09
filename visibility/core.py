#!/usr/bin/env python3
# -*-coding:utf-8 -*-
"""
可见性计算核心模块
==================

优化版本：
1. 只考虑侧面（前后左右）
2. 高效向量化射线检测
"""

import numpy as np
from typing import Dict, List, Tuple, Optional

from . import config


def _rotation_matrix_zyx(phi: float, theta: float, psi: float) -> np.ndarray:
    """计算ZYX顺序的旋转矩阵"""
    cz, sz = np.cos(phi), np.sin(phi)
    cy, sy = np.cos(theta), np.sin(theta)
    cx, sx = np.cos(psi), np.sin(psi)
    
    return np.array([
        [cz * cy, cz * sy * sx - sz * cx, cz * sy * cx + sz * sx],
        [sz * cy, sz * sy * sx + cz * cx, sz * sy * cx - cz * sx],
        [-sy, cy * sx, cy * cx]
    ], dtype=np.float64)


def _parse_bbox(bbox_dict: Dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """解析bbox字典"""
    cx = float(bbox_dict['position']['x'])
    cy = float(bbox_dict['position']['y'])
    cz = float(bbox_dict['position']['z'])
    center = np.array([cx, cy, cz], dtype=np.float64)
    
    w_idx = config.SIZE_ORDER['width']
    h_idx = config.SIZE_ORDER['height']
    l_idx = config.SIZE_ORDER['length']
    
    length = max(float(bbox_dict['size'][l_idx]), config.MIN_SIZE)
    width = max(float(bbox_dict['size'][w_idx]), config.MIN_SIZE)
    height = max(float(bbox_dict['size'][h_idx]), config.MIN_SIZE)
    
    half_dims = np.array([length/2, width/2, height/2], dtype=np.float64)
    
    phi = float(bbox_dict['orientation']['phi'])
    theta = float(bbox_dict['orientation'].get('theta', 0.0))
    psi = float(bbox_dict['orientation'].get('psi', 0.0))
    R = _rotation_matrix_zyx(phi, theta, psi)
    
    return center, half_dims, R


def _voxel_downsample(points: np.ndarray, voxel_size: float) -> np.ndarray:
    """
    体素降采样
    
    将空间划分为固定大小的体素格子，每个格子内只保留一个点（质心）
    
    Args:
        points: (N, 3) 点云
        voxel_size: 体素大小（米）
    
    Returns:
        降采样后的点云
    """
    if len(points) == 0 or voxel_size <= 0:
        return points
    
    # 计算每个点所属的体素索引
    voxel_indices = np.floor(points / voxel_size).astype(np.int32)
    
    # 将3D索引转换为唯一的1D键
    # 使用大质数避免碰撞
    keys = (voxel_indices[:, 0] * 73856093 ^ 
            voxel_indices[:, 1] * 19349663 ^ 
            voxel_indices[:, 2] * 83492791)
    
    # 找到每个体素的第一个点（或计算质心）
    unique_keys, inverse_indices = np.unique(keys, return_inverse=True)
    
    # 方法1：取每个体素的第一个点（快速）
    # first_indices = np.zeros(len(unique_keys), dtype=np.int32)
    # for i, key in enumerate(keys):
    #     idx = inverse_indices[i]
    #     if first_indices[idx] == 0 or i < first_indices[idx]:
    #         first_indices[idx] = i
    # return points[first_indices]
    
    # 方法2：计算每个体素的质心（更准确）
    n_voxels = len(unique_keys)
    voxel_sums = np.zeros((n_voxels, 3), dtype=np.float64)
    voxel_counts = np.zeros(n_voxels, dtype=np.int32)
    
    np.add.at(voxel_sums, inverse_indices, points)
    np.add.at(voxel_counts, inverse_indices, 1)
    
    centroids = voxel_sums / voxel_counts[:, np.newaxis]
    
    return centroids


def _filter_points_outside_bbox(
    points: np.ndarray,
    center: np.ndarray,
    half_dims: np.ndarray,
    R: np.ndarray,
    margin: float = 0.1
) -> np.ndarray:
    """
    排除bbox内部的点
    
    框内的点是目标本身的点云，不应该参与遮挡计算
    
    Args:
        points: (N, 3) 点云
        center: (3,) bbox中心
        half_dims: (3,) bbox半尺寸 [half_l, half_w, half_h]
        R: (3, 3) bbox旋转矩阵
        margin: 边界余量（米），稍微扩大排除范围
    
    Returns:
        框外的点云
    """
    if len(points) == 0:
        return points
    
    # 将点转换到bbox局部坐标系
    local_pts = (R.T @ (points - center).T).T
    
    # 检查是否在框内（加上margin）
    half_l, half_w, half_h = half_dims
    in_box = (
        (np.abs(local_pts[:, 0]) < half_l + margin) &
        (np.abs(local_pts[:, 1]) < half_w + margin) &
        (np.abs(local_pts[:, 2]) < half_h + margin)
    )
    
    # 返回框外的点
    return points[~in_box]


def _filter_points_fast(
    points: np.ndarray,
    sensor: np.ndarray,
    bbox_center: np.ndarray,
    bbox_dist: float,
    half_dims: np.ndarray
) -> np.ndarray:
    """
    快速空间过滤：只保留可能遮挡bbox的点
    
    使用简单的距离+方向过滤
    """
    if not config.SPATIAL_FILTER:
        return points
    
    if bbox_dist < 2.0:
        return points
    
    # 只保留在传感器和bbox之间的点（距离过滤）
    rel_pts = points - sensor
    pts_dist_sq = np.sum(rel_pts ** 2, axis=1)
    
    max_dist = bbox_dist + np.max(half_dims) * 2
    mask = pts_dist_sq < max_dist ** 2
    
    if mask.sum() < 100:
        return points
    
    # 方向过滤
    bbox_dir = (bbox_center - sensor) / bbox_dist
    pts_dist = np.sqrt(pts_dist_sq[mask])
    cos_angle = (rel_pts[mask] @ bbox_dir) / np.maximum(pts_dist, 0.1)
    
    angle_thresh = np.deg2rad(config.SPATIAL_FILTER_ANGLE)
    dir_mask = cos_angle > np.cos(angle_thresh)
    
    indices = np.where(mask)[0][dir_mask]
    
    if len(indices) < 100:
        return points
    
    return points[indices]


def _determine_visible_faces(sensor_local: np.ndarray) -> np.ndarray:
    """判断可见面（只考虑4个侧面）"""
    threshold = 0.01
    sx, sy, _ = sensor_local
    
    if config.SIDE_FACES_ONLY:
        visible = np.array([
            sx > threshold,    # +x (后)
            sx < -threshold,   # -x (前)
            sy > threshold,    # +y (右)
            sy < -threshold,   # -y (左)
            False,             # +z (顶) - 不考虑
            False,             # -z (底) - 不考虑
        ])
    else:
        sz = sensor_local[2]
        visible = np.array([
            sx > threshold, sx < -threshold,
            sy > threshold, sy < -threshold,
            sz > threshold, sz < -threshold,
        ])
    
    # 如果没有可见面，选择最可能的面
    if not any(visible[:4]):
        abs_xy = [abs(sx), abs(sy)]
        max_dim = np.argmax(abs_xy)
        if max_dim == 0:
            visible[0 if sx >= 0 else 1] = True
        else:
            visible[2 if sy >= 0 else 3] = True
    
    return visible


def _sample_surfaces(
    center: np.ndarray,
    half_dims: np.ndarray,
    R: np.ndarray,
    sensor: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """在可见侧面上采样"""
    half_l, half_w, half_h = half_dims
    
    sensor_local = R.T @ (sensor - center)
    visible = _determine_visible_faces(sensor_local)
    
    # 侧面面积
    areas = np.array([
        4 * half_w * half_h,  # +x
        4 * half_w * half_h,  # -x
        4 * half_l * half_h,  # +y
        4 * half_l * half_h,  # -y
        0, 0,  # 顶底
    ])
    
    visible_areas = areas * visible.astype(np.float64)
    total_area = visible_areas.sum()
    
    if total_area < 1e-6:
        max_idx = np.argmax(areas[:4])
        visible[max_idx] = True
        visible_areas = areas * visible.astype(np.float64)
        total_area = visible_areas.sum()
    
    samples_per_face = (config.TOTAL_SAMPLES * visible_areas / total_area + 0.5).astype(int)
    samples_per_face[~visible] = 0
    samples_per_face[samples_per_face > 0] = np.maximum(samples_per_face[samples_per_face > 0], 4)
    
    all_samples = []
    all_face_ids = []
    
    for face_id in range(4):
        n = samples_per_face[face_id]
        if n == 0:
            continue
        
        n_side = max(int(np.sqrt(n)), 2)
        u = np.linspace(-0.9, 0.9, n_side)
        v = np.linspace(-0.9, 0.9, n_side)
        uu, vv = np.meshgrid(u, v)
        uu, vv = uu.flatten(), vv.flatten()
        
        if face_id == 0:
            local = np.column_stack([np.full_like(uu, half_l), uu * half_w, vv * half_h])
        elif face_id == 1:
            local = np.column_stack([np.full_like(uu, -half_l), uu * half_w, vv * half_h])
        elif face_id == 2:
            local = np.column_stack([uu * half_l, np.full_like(uu, half_w), vv * half_h])
        else:
            local = np.column_stack([uu * half_l, np.full_like(uu, -half_w), vv * half_h])
        
        world = (R @ local.T).T + center
        all_samples.append(world)
        all_face_ids.append(np.full(len(world), face_id, dtype=np.int64))
    
    if not all_samples:
        return np.empty((0, 3)), np.empty(0, dtype=np.int64)
    
    return np.vstack(all_samples), np.concatenate(all_face_ids)


def _check_rays_blocked_vectorized(
    targets: np.ndarray,
    sensor: np.ndarray,
    points: np.ndarray
) -> np.ndarray:
    """
    高效向量化射线遮挡检测
    
    使用批处理减少循环开销
    """
    n_targets = len(targets)
    blocked = np.zeros(n_targets, dtype=bool)
    
    if len(points) == 0:
        return blocked
    
    tan_thresh = np.tan(np.deg2rad(config.ANGLE_THRESH_DEG))
    start_margin = config.RAY_START_MARGIN
    end_margin = config.RAY_END_MARGIN
    min_blockers = config.MIN_BLOCKERS
    
    # 预计算
    rel_pts = points - sensor
    
    # 批量处理targets
    batch_size = 16
    for batch_start in range(0, n_targets, batch_size):
        batch_end = min(batch_start + batch_size, n_targets)
        
        for i in range(batch_start, batch_end):
            target = targets[i]
            ray = target - sensor
            ray_len = np.linalg.norm(ray)
            
            if ray_len < 1e-6:
                continue
            
            ray_dir = ray / ray_len
            
            # 投影
            proj = rel_pts @ ray_dir
            
            # 有效范围内的点
            valid = (proj > start_margin) & (proj < ray_len - end_margin)
            
            if not np.any(valid):
                continue
            
            proj_v = proj[valid]
            
            # 垂直距离的平方
            closest = np.outer(proj_v, ray_dir)
            perp_sq = np.sum((rel_pts[valid] - closest) ** 2, axis=1)
            
            # 动态阈值
            thresh_sq = (proj_v * tan_thresh) ** 2
            
            # 统计
            if np.sum(perp_sq < thresh_sq) >= min_blockers:
                blocked[i] = True
    
    return blocked


def classify_visibility(score: float) -> str:
    """可见性分类"""
    if score >= config.VISIBILITY_THRESHOLDS['VISIBLE']:
        return 'VISIBLE'
    elif score >= config.VISIBILITY_THRESHOLDS['PARTIAL']:
        return 'PARTIAL'
    elif score >= config.VISIBILITY_THRESHOLDS['OCCLUDED']:
        return 'OCCLUDED'
    else:
        return 'BLOCKED'


def compute_visibility(
    bbox_dict: Dict,
    points: np.ndarray,
    sensor_origin: Optional[np.ndarray] = None
) -> Tuple[float, Dict]:
    """
    计算单个bbox的可见性
    
    Args:
        bbox_dict: 3D标注框字典
        points: (N, 3) 点云
        sensor_origin: (3,) 传感器位置，默认原点
    
    Returns:
        score: 可见性分数 [0, 1]
        details: 详细信息
    """
    if sensor_origin is None:
        sensor_origin = np.array([0.0, 0.0, 0.0])
    
    try:
        center, half_dims, R = _parse_bbox(bbox_dict)
    except Exception as e:
        return 0.0, {'score': 0.0, 'status': 'PARSE_ERROR', 'error': str(e)}
    
    points = np.ascontiguousarray(points[:, :3], dtype=np.float64)
    sensor = np.asarray(sensor_origin, dtype=np.float64)
    
    try:
        samples, face_ids = _sample_surfaces(center, half_dims, R, sensor)
    except Exception as e:
        return 0.0, {'score': 0.0, 'status': 'SAMPLE_ERROR', 'error': str(e)}
    
    n_samples = len(samples)
    if n_samples == 0:
        return 0.0, {'score': 0.0, 'status': 'NO_VISIBLE_SURFACE', 'n_samples': 0}
    
    # 计算bbox到传感器的距离
    bbox_dist = np.linalg.norm(center - sensor)
    
    # 空间过滤
    scene_pts = _filter_points_fast(points, sensor, center, bbox_dist, half_dims)
    
    # 排除框内的点（框内点是目标本身，不参与遮挡计算）
    scene_pts = _filter_points_outside_bbox(scene_pts, center, half_dims, R)
    
    # 体素降采样（均匀稳定）
    if len(scene_pts) > config.MAX_SCENE_POINTS:
        scene_pts = _voxel_downsample(scene_pts, config.VOXEL_SIZE)
    
    # 检测遮挡
    blocked = _check_rays_blocked_vectorized(samples, sensor, scene_pts)
    
    # 按面计算可见性，取最高值
    face_names = ['+x(后)', '-x(前)', '+y(右)', '-y(左)']
    face_visibility = {}
    max_visibility = 0.0
    best_face = None
    
    for fid in range(4):
        mask = face_ids == fid
        if not np.any(mask):
            continue
        
        face_samples = mask.sum()
        face_blocked = blocked[mask].sum()
        face_vis = (face_samples - face_blocked) / face_samples
        face_visibility[face_names[fid]] = float(face_vis)
        
        if face_vis > max_visibility:
            max_visibility = face_vis
            best_face = face_names[fid]
    
    # 使用最高可见面的可见性作为最终分数
    visibility = max_visibility
    
    n_blocked = blocked.sum()
    n_visible = n_samples - n_blocked
    
    status = classify_visibility(visibility)
    
    return float(visibility), {
        'score': float(visibility),
        'best_face': best_face,
        'face_visibility': face_visibility,
        'n_samples': int(n_samples),
        'n_blocked': int(n_blocked),
        'n_visible': int(n_visible),
        'n_scene_pts': len(scene_pts),
        'distance': float(bbox_dist),
        'status': status
    }


def compute_frame_visibility(
    label_3d_list: List[Dict],
    points: np.ndarray,
    sensor_origin: Optional[np.ndarray] = None
) -> Dict[str, Dict]:
    """
    计算一帧中所有bbox的可见性
    """
    results = {}
    
    for bbox_dict in label_3d_list:
        track_id = str(bbox_dict.get('track_id', 'unknown'))
        score, details = compute_visibility(bbox_dict, points, sensor_origin)
        results[track_id] = details
    
    return results
