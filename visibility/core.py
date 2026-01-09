#!/usr/bin/env python3
# -*-coding:utf-8 -*-
"""
可见性计算核心模块
==================

纯NumPy实现，稳定可靠
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
    """
    解析bbox字典
    
    Returns:
        center: (3,) 中心点
        half_dims: (3,) 半尺寸 [half_l, half_w, half_h]
        R: (3,3) 旋转矩阵
    """
    # 位置
    cx = float(bbox_dict['position']['x'])
    cy = float(bbox_dict['position']['y'])
    cz = float(bbox_dict['position']['z'])
    center = np.array([cx, cy, cz], dtype=np.float64)
    
    # 尺寸
    w_idx = config.SIZE_ORDER['width']
    h_idx = config.SIZE_ORDER['height']
    l_idx = config.SIZE_ORDER['length']
    
    length = max(float(bbox_dict['size'][l_idx]), config.MIN_SIZE)
    width = max(float(bbox_dict['size'][w_idx]), config.MIN_SIZE)
    height = max(float(bbox_dict['size'][h_idx]), config.MIN_SIZE)
    
    half_dims = np.array([length/2, width/2, height/2], dtype=np.float64)
    
    # 旋转
    phi = float(bbox_dict['orientation']['phi'])
    theta = float(bbox_dict['orientation']['theta'])
    psi = float(bbox_dict['orientation']['psi'])
    R = _rotation_matrix_zyx(phi, theta, psi)
    
    return center, half_dims, R


def _determine_visible_faces(sensor_local: np.ndarray) -> np.ndarray:
    """判断可见面"""
    threshold = 0.01
    sx, sy, sz = sensor_local
    
    visible = np.array([
        sx > threshold,    # +x
        sx < -threshold,   # -x
        sy > threshold,    # +y
        sy < -threshold,   # -y
        sz > threshold,    # +z
        False if config.EXCLUDE_BOTTOM else sz < -threshold,  # -z
    ])
    
    # 如果没有可见面，选择最可能的面
    if not any(visible[:5]):
        abs_local = np.abs(sensor_local)
        max_dim = np.argmax(abs_local)
        if sensor_local[max_dim] >= 0:
            visible[max_dim * 2] = True
        else:
            visible[max_dim * 2 + 1] = True
    
    return visible


def _sample_surfaces(
    center: np.ndarray,
    half_dims: np.ndarray,
    R: np.ndarray,
    sensor: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """在可见面上采样"""
    half_l, half_w, half_h = half_dims
    
    # 计算传感器在局部坐标系的位置
    sensor_local = R.T @ (sensor - center)
    
    # 判断可见面
    visible = _determine_visible_faces(sensor_local)
    
    # 计算面积
    areas = np.array([
        4 * half_w * half_h,
        4 * half_w * half_h,
        4 * half_l * half_h,
        4 * half_l * half_h,
        4 * half_l * half_w,
        4 * half_l * half_w,
    ])
    
    # 按面积分配采样
    visible_areas = areas * visible.astype(np.float64)
    total_area = visible_areas.sum()
    
    if total_area < 1e-6:
        max_idx = np.argmax(areas[:5])
        visible[max_idx] = True
        visible_areas = areas * visible.astype(np.float64)
        total_area = visible_areas.sum()
    
    samples_per_face = np.maximum(
        (config.TOTAL_SAMPLES * visible_areas / total_area + 0.5).astype(int), 0
    )
    samples_per_face[~visible] = 0
    samples_per_face[samples_per_face > 0] = np.maximum(
        samples_per_face[samples_per_face > 0], 4
    )
    
    # 采样
    all_samples = []
    all_face_ids = []
    
    for face_id in range(6):
        n = samples_per_face[face_id]
        if n == 0:
            continue
        
        n_side = max(int(np.sqrt(n)), 2)
        u = np.linspace(-1 + 1/n_side, 1 - 1/n_side, n_side)
        v = np.linspace(-1 + 1/n_side, 1 - 1/n_side, n_side)
        uu, vv = np.meshgrid(u, v)
        uu, vv = uu.flatten(), vv.flatten()
        
        if face_id == 0:
            local = np.column_stack([np.full_like(uu, half_l), uu * half_w, vv * half_h])
        elif face_id == 1:
            local = np.column_stack([np.full_like(uu, -half_l), uu * half_w, vv * half_h])
        elif face_id == 2:
            local = np.column_stack([uu * half_l, np.full_like(uu, half_w), vv * half_h])
        elif face_id == 3:
            local = np.column_stack([uu * half_l, np.full_like(uu, -half_w), vv * half_h])
        elif face_id == 4:
            local = np.column_stack([uu * half_l, vv * half_w, np.full_like(uu, half_h)])
        else:
            local = np.column_stack([uu * half_l, vv * half_w, np.full_like(uu, -half_h)])
        
        world = (R @ local.T).T + center
        all_samples.append(world)
        all_face_ids.append(np.full(len(world), face_id, dtype=np.int64))
    
    if not all_samples:
        return np.empty((0, 3)), np.empty(0, dtype=np.int64)
    
    return np.vstack(all_samples), np.concatenate(all_face_ids)


def _check_rays_blocked(
    targets: np.ndarray,
    sensor: np.ndarray,
    points: np.ndarray
) -> np.ndarray:
    """向量化射线遮挡检测"""
    n_targets = len(targets)
    blocked = np.zeros(n_targets, dtype=bool)
    
    angle_thresh = np.deg2rad(config.ANGLE_THRESH_DEG)
    tan_thresh = np.tan(angle_thresh)
    
    for i in range(n_targets):
        target = targets[i]
        ray = target - sensor
        ray_len = np.linalg.norm(ray)
        
        if ray_len < 1e-6:
            continue
        
        ray_dir = ray / ray_len
        
        # 计算投影
        rel_pts = points - sensor
        proj = rel_pts @ ray_dir
        
        # 只考虑在sensor和target之间的点
        valid_mask = (proj > config.RAY_START_MARGIN) & (proj < ray_len - config.RAY_END_MARGIN)
        
        if not np.any(valid_mask):
            continue
        
        proj_valid = proj[valid_mask]
        rel_pts_valid = rel_pts[valid_mask]
        
        # 计算垂直距离
        closest = np.outer(proj_valid, ray_dir)
        perp_dist_sq = np.sum((rel_pts_valid - closest) ** 2, axis=1)
        
        # 动态阈值
        thresh = proj_valid * tan_thresh
        
        # 统计遮挡点数
        n_blockers = np.sum(perp_dist_sq < thresh ** 2)
        
        if n_blockers >= config.MIN_BLOCKERS:
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
    
    # 解析bbox
    try:
        center, half_dims, R = _parse_bbox(bbox_dict)
    except Exception as e:
        return 0.0, {'score': 0.0, 'status': 'PARSE_ERROR', 'error': str(e)}
    
    # 准备点云
    points = np.ascontiguousarray(points[:, :3], dtype=np.float64)
    sensor = np.asarray(sensor_origin, dtype=np.float64)
    
    # 采样
    try:
        samples, face_ids = _sample_surfaces(center, half_dims, R, sensor)
    except Exception as e:
        return 0.0, {'score': 0.0, 'status': 'SAMPLE_ERROR', 'error': str(e)}
    
    n_samples = len(samples)
    if n_samples == 0:
        return 0.0, {'score': 0.0, 'status': 'NO_VISIBLE_SURFACE', 'n_samples': 0}
    
    # 降采样点云
    if len(points) > config.MAX_SCENE_POINTS:
        indices = np.random.choice(len(points), config.MAX_SCENE_POINTS, replace=False)
        scene_pts = points[indices]
    else:
        scene_pts = points
    
    # 检测遮挡
    blocked = _check_rays_blocked(samples, sensor, scene_pts)
    
    n_blocked = blocked.sum()
    n_visible = n_samples - n_blocked
    visibility = n_visible / n_samples
    
    status = classify_visibility(visibility)
    
    return float(visibility), {
        'score': float(visibility),
        'n_samples': int(n_samples),
        'n_blocked': int(n_blocked),
        'n_visible': int(n_visible),
        'status': status
    }


def compute_frame_visibility(
    label_3d_list: List[Dict],
    points: np.ndarray,
    sensor_origin: Optional[np.ndarray] = None
) -> Dict[str, Dict]:
    """
    计算一帧中所有bbox的可见性
    
    Args:
        label_3d_list: 3D标注框列表
        points: (N, 3) 点云
        sensor_origin: (3,) 传感器位置
    
    Returns:
        results: {track_id: details}
    """
    results = {}
    
    for bbox_dict in label_3d_list:
        track_id = str(bbox_dict.get('track_id', 'unknown'))
        score, details = compute_visibility(bbox_dict, points, sensor_origin)
        results[track_id] = details
    
    return results
