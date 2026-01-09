#!/usr/bin/env python3
# -*-coding:utf-8 -*-
"""
3D标注框可见性计算接口 (纯NumPy版本，无Numba依赖)
================================================

适用于Numba不可用或不稳定的环境。
使用向量化NumPy操作优化性能。
"""

import numpy as np
from typing import Dict, List, Tuple, Union, Optional


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


def _sample_visible_surfaces_numpy(
    center: np.ndarray,
    half_dims: np.ndarray,
    R: np.ndarray,
    sensor: np.ndarray,
    total_samples: int = 80,
    exclude_bottom: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """在可见面上采样（纯NumPy版本）"""
    half_l, half_w, half_h = half_dims
    
    # 计算传感器在局部坐标系的位置
    sensor_local = R.T @ (sensor - center)
    
    # 判断可见面
    visible = [
        sensor_local[0] > 0,  # +x
        sensor_local[0] < 0,  # -x
        sensor_local[1] > 0,  # +y
        sensor_local[1] < 0,  # -y
        sensor_local[2] > 0,  # +z
        False if exclude_bottom else sensor_local[2] < 0,  # -z
    ]
    
    # 计算面积
    areas = np.array([
        4 * half_w * half_h,  # +x, -x
        4 * half_w * half_h,
        4 * half_l * half_h,  # +y, -y
        4 * half_l * half_h,
        4 * half_l * half_w,  # +z, -z
        4 * half_l * half_w,
    ])
    
    # 按面积分配采样
    visible_areas = areas * np.array(visible, dtype=np.float64)
    total_area = visible_areas.sum()
    
    if total_area < 1e-6:
        return np.empty((0, 3)), np.empty(0, dtype=np.int64)
    
    samples_per_face = np.maximum((total_samples * visible_areas / total_area + 0.5).astype(int), 0)
    samples_per_face[~np.array(visible)] = 0
    samples_per_face[samples_per_face > 0] = np.maximum(samples_per_face[samples_per_face > 0], 4)
    
    # 采样
    all_samples = []
    all_face_ids = []
    
    for face_id in range(6):
        n = samples_per_face[face_id]
        if n == 0:
            continue
        
        # 网格采样
        n_side = max(int(np.sqrt(n)), 2)
        u = np.linspace(-1 + 1/n_side, 1 - 1/n_side, n_side)
        v = np.linspace(-1 + 1/n_side, 1 - 1/n_side, n_side)
        uu, vv = np.meshgrid(u, v)
        uu, vv = uu.flatten(), vv.flatten()
        
        if face_id == 0:  # +x
            local = np.column_stack([np.full_like(uu, half_l), uu * half_w, vv * half_h])
        elif face_id == 1:  # -x
            local = np.column_stack([np.full_like(uu, -half_l), uu * half_w, vv * half_h])
        elif face_id == 2:  # +y
            local = np.column_stack([uu * half_l, np.full_like(uu, half_w), vv * half_h])
        elif face_id == 3:  # -y
            local = np.column_stack([uu * half_l, np.full_like(uu, -half_w), vv * half_h])
        elif face_id == 4:  # +z
            local = np.column_stack([uu * half_l, vv * half_w, np.full_like(uu, half_h)])
        else:  # -z
            local = np.column_stack([uu * half_l, vv * half_w, np.full_like(uu, -half_h)])
        
        # 转到世界坐标
        world = (R @ local.T).T + center
        all_samples.append(world)
        all_face_ids.append(np.full(len(world), face_id, dtype=np.int64))
    
    if not all_samples:
        return np.empty((0, 3)), np.empty(0, dtype=np.int64)
    
    return np.vstack(all_samples), np.concatenate(all_face_ids)


def _check_rays_blocked_vectorized(
    targets: np.ndarray,
    sensor: np.ndarray,
    points: np.ndarray,
    angle_thresh: float = 0.00873,  # 0.5度
    min_blockers: int = 3
) -> np.ndarray:
    """
    向量化射线遮挡检测（纯NumPy）
    
    为了效率，使用批处理方式
    """
    n_targets = len(targets)
    blocked = np.zeros(n_targets, dtype=bool)
    
    # 预计算
    tan_thresh = np.tan(angle_thresh)
    
    for i in range(n_targets):
        target = targets[i]
        ray = target - sensor
        ray_len = np.linalg.norm(ray)
        
        if ray_len < 1e-6:
            continue
        
        ray_dir = ray / ray_len
        
        # 计算所有点到射线的投影
        rel_pts = points - sensor
        proj = rel_pts @ ray_dir  # 投影长度
        
        # 只考虑在sensor和target之间的点
        valid_mask = (proj > 0.5) & (proj < ray_len - 0.3)
        
        if not np.any(valid_mask):
            continue
        
        proj_valid = proj[valid_mask]
        rel_pts_valid = rel_pts[valid_mask]
        
        # 计算垂直距离
        closest = np.outer(proj_valid, ray_dir)
        perp_dist_sq = np.sum((rel_pts_valid - closest) ** 2, axis=1)
        
        # 动态阈值
        thresh = proj_valid * tan_thresh
        
        # 统计在阈值内的点数
        n_blockers = np.sum(perp_dist_sq < thresh ** 2)
        
        if n_blockers >= min_blockers:
            blocked[i] = True
    
    return blocked


def _parse_bbox_dict(bbox_dict: Dict) -> Tuple[np.ndarray, float, float, float, float, float, float]:
    """解析bbox字典"""
    cx = float(bbox_dict['position']['x'])
    cy = float(bbox_dict['position']['y'])
    cz = float(bbox_dict['position']['z'])
    center = np.array([cx, cy, cz], dtype=np.float64)
    
    length = float(bbox_dict['size'][2])
    width = float(bbox_dict['size'][0])
    height = float(bbox_dict['size'][1])
    
    length = max(length, 0.1)
    width = max(width, 0.1)
    height = max(height, 0.1)
    
    phi = float(bbox_dict['orientation']['phi'])
    theta = float(bbox_dict['orientation']['theta'])
    psi = float(bbox_dict['orientation']['psi'])
    
    return center, length, width, height, phi, theta, psi


def compute_bbox_visibility(
    bbox_dict: Dict,
    points: np.ndarray,
    sensor_origin: Optional[np.ndarray] = None,
    total_samples: int = 80,
    angle_thresh_deg: float = 0.5,
    min_blockers: int = 3,
    max_scene_points: int = 15000,
    exclude_bottom: bool = True
) -> Tuple[float, Dict]:
    """
    计算单个bbox的可见性（纯NumPy版本）
    """
    if sensor_origin is None:
        sensor_origin = np.array([0.0, 0.0, 0.0])
    
    # 解析bbox
    try:
        center, length, width, height, phi, theta, psi = _parse_bbox_dict(bbox_dict)
    except Exception as e:
        return 0.0, {'score': 0.0, 'status': 'PARSE_ERROR', 'error': str(e)}
    
    # 准备数据
    points = np.ascontiguousarray(points[:, :3], dtype=np.float64)
    sensor = np.asarray(sensor_origin, dtype=np.float64)
    angle_thresh = np.deg2rad(angle_thresh_deg)
    
    # 旋转矩阵
    R = _rotation_matrix_zyx(phi, theta, psi)
    half_dims = np.array([length/2, width/2, height/2], dtype=np.float64)
    
    # 采样
    try:
        samples, face_ids = _sample_visible_surfaces_numpy(
            center, half_dims, R, sensor, total_samples, exclude_bottom
        )
    except Exception as e:
        return 0.0, {'score': 0.0, 'status': 'SAMPLE_ERROR', 'error': str(e)}
    
    n_samples = len(samples)
    if n_samples == 0:
        return 0.0, {'score': 0.0, 'status': 'NO_VISIBLE_SURFACE', 'n_samples': 0}
    
    # 降采样场景点云
    if len(points) > max_scene_points:
        indices = np.random.choice(len(points), max_scene_points, replace=False)
        scene_pts = points[indices]
    else:
        scene_pts = points
    
    # 检测遮挡
    blocked = _check_rays_blocked_vectorized(
        samples, sensor, scene_pts, angle_thresh, min_blockers
    )
    
    n_blocked = blocked.sum()
    n_visible = n_samples - n_blocked
    visibility = n_visible / n_samples
    
    # 状态
    if visibility >= 0.7:
        status = 'VISIBLE'
    elif visibility >= 0.3:
        status = 'PARTIAL'
    elif visibility > 0.05:
        status = 'OCCLUDED'
    else:
        status = 'BLOCKED'
    
    return float(visibility), {
        'score': float(visibility),
        'n_samples': int(n_samples),
        'n_blocked': int(n_blocked),
        'n_visible': int(n_visible),
        'status': status
    }


def compute_frame_visibility(
    label_3d_list: List[Dict],
    pcd_pts: np.ndarray,
    sensor_origin: Optional[np.ndarray] = None,
    **kwargs
) -> Dict[str, Dict]:
    """
    计算一帧中所有bbox的可见性（纯NumPy版本）
    """
    results = {}
    
    for bbox_dict in label_3d_list:
        track_id = str(bbox_dict.get('track_id', 'unknown'))
        score, details = compute_bbox_visibility(bbox_dict, pcd_pts, sensor_origin, **kwargs)
        results[track_id] = details
    
    return results


def classify_visibility(score: float) -> str:
    if score >= 0.7: return "VISIBLE"
    if score >= 0.3: return "PARTIAL"
    if score > 0.05: return "OCCLUDED"
    return "BLOCKED"


# ============================================================
# 测试
# ============================================================
if __name__ == '__main__':
    import time
    
    print("纯NumPy版本可见性计算测试")
    print("=" * 50)
    
    np.random.seed(42)
    
    # 模拟数据
    label_3d_list = [
        {
            'position': {'x': 15.0, 'y': 3.0, 'z': 0.8},
            'size': [2.0, 1.5, 4.5],
            'orientation': {'phi': 0.1, 'theta': 0.0, 'psi': 0.0},
            'track_id': 'car_001'
        }
    ]
    
    # 点云
    pcd_pts = np.random.randn(100000, 3) * 20
    
    print(f"点云数量: {len(pcd_pts)}")
    
    # 预热
    compute_frame_visibility(label_3d_list, pcd_pts[:1000])
    
    # 测试
    t0 = time.time()
    results = compute_frame_visibility(label_3d_list, pcd_pts)
    t1 = time.time()
    
    print(f"耗时: {(t1-t0)*1000:.1f} ms")
    
    for tid, info in results.items():
        print(f"{tid}: {info['score']:.0%} ({info['status']})")
    
    # 批量测试
    print("\n批量测试 (10个bbox):")
    label_3d_list = label_3d_list * 10
    for i, bbox in enumerate(label_3d_list):
        bbox['track_id'] = f'car_{i:03d}'
    
    t0 = time.time()
    results = compute_frame_visibility(label_3d_list, pcd_pts)
    t1 = time.time()
    
    print(f"总耗时: {(t1-t0)*1000:.1f} ms")
    print(f"每框: {(t1-t0)/len(label_3d_list)*1000:.1f} ms")
