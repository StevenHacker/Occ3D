#!/usr/bin/env python3
# -*-coding:utf-8 -*-
"""
3D标注框可见性计算接口
=====================

【功能】
- 计算3D标注框在点云中的可见性（基于表面采样+射线遮挡检测）
- 可视化可见性结果到图像

【使用方法】
```python
from visibility_api import compute_frame_visibility, visualize_visibility

# 1. 计算可见性
results = compute_frame_visibility(label_3d_list, pcd_pts)

# 2. 可视化（可选）
img = visualize_visibility(img, label_3d_list, results, extrinsic, intrinsic)
cv2.imwrite('output.jpg', img)
```

【数据格式】
bbox格式:
{
    'position': {'x': float, 'y': float, 'z': float},
    'size': [w, h, l],  # size[0]=宽, size[1]=高, size[2]=长
    'orientation': {'phi': float, 'theta': float, 'psi': float},
    'track_id': str or int,
    ...
}

点云格式: numpy array (N, 3) 或 (N, 4+)
"""

import numpy as np
from typing import Dict, List, Tuple, Union, Optional

# 导入可视化工具
try:
    import cv2
    from visibility_visualizer import (
        VisibilityVisualizer,
        visualize_frame_visibility,
        project_bboxes_to_image,
        print_visibility_summary
    )
    HAS_VISUALIZER = True
except ImportError:
    HAS_VISUALIZER = False

# ============================================================
# Numba JIT 编译支持（可选，用于加速）
# ============================================================
try:
    from numba import njit
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    def njit(*args, **kwargs):
        def decorator(func): return func
        return decorator if not (args and callable(args[0])) else args[0]


# ============================================================
# 核心计算函数 (Numba加速)
# ============================================================

@njit(cache=True)
def _rotation_matrix_zyx(phi: float, theta: float, psi: float) -> np.ndarray:
    """
    计算ZYX顺序的旋转矩阵
    
    对应scipy的: R.from_euler('ZYX', [phi, theta, psi])
    
    Args:
        phi: 绕Z轴旋转角 (yaw)
        theta: 绕Y轴旋转角 (pitch)
        psi: 绕X轴旋转角 (roll)
    
    Returns:
        3x3旋转矩阵
    """
    # ZYX顺序: R = Rz(phi) @ Ry(theta) @ Rx(psi)
    cz, sz = np.cos(phi), np.sin(phi)
    cy, sy = np.cos(theta), np.sin(theta)
    cx, sx = np.cos(psi), np.sin(psi)
    
    R = np.empty((3, 3), dtype=np.float64)
    
    R[0, 0] = cz * cy
    R[0, 1] = cz * sy * sx - sz * cx
    R[0, 2] = cz * sy * cx + sz * sx
    
    R[1, 0] = sz * cy
    R[1, 1] = sz * sy * sx + cz * cx
    R[1, 2] = sz * sy * cx - cz * sx
    
    R[2, 0] = -sy
    R[2, 1] = cy * sx
    R[2, 2] = cy * cx
    
    return R


@njit(cache=True)
def _determine_visible_faces(sensor_local: np.ndarray, exclude_bottom: bool) -> np.ndarray:
    """判断哪些面朝向传感器"""
    sx, sy, sz = sensor_local[0], sensor_local[1], sensor_local[2]
    
    visible = np.array([
        sx > 0,   # +x面(前)
        sx < 0,   # -x面(后)
        sy > 0,   # +y面(左)
        sy < 0,   # -y面(右)
        sz > 0,   # +z面(顶)
        sz < 0,   # -z面(底)
    ])
    
    if exclude_bottom:
        visible[5] = False
    
    return visible


@njit(cache=True)
def _compute_face_areas(half_l: float, half_w: float, half_h: float) -> np.ndarray:
    """计算六个面的面积"""
    areas = np.empty(6, dtype=np.float64)
    
    area_x = (2 * half_w) * (2 * half_h)  # 前后面
    area_y = (2 * half_l) * (2 * half_h)  # 左右面
    area_z = (2 * half_l) * (2 * half_w)  # 顶底面
    
    areas[0] = area_x
    areas[1] = area_x
    areas[2] = area_y
    areas[3] = area_y
    areas[4] = area_z
    areas[5] = area_z
    
    return areas


@njit(cache=True)
def _sample_single_face_adaptive(
    face_id: int,
    half_l: float, half_w: float, half_h: float,
    n_samples: int
) -> np.ndarray:
    """在单个面上自适应采样"""
    # 根据面确定两个维度
    if face_id == 0 or face_id == 1:
        dim1, dim2 = half_w, half_h
    elif face_id == 2 or face_id == 3:
        dim1, dim2 = half_l, half_h
    else:
        dim1, dim2 = half_l, half_w
    
    # 按宽高比分配网格
    ratio = dim1 / dim2 if dim2 > 0.01 else 1.0
    n1 = int(np.sqrt(n_samples * ratio) + 0.5)
    n2 = int(np.sqrt(n_samples / ratio) + 0.5)
    if n1 < 1: n1 = 1
    if n2 < 1: n2 = 1
    
    actual_samples = n1 * n2
    samples = np.empty((actual_samples, 3), dtype=np.float64)
    
    idx = 0
    for i in range(n1):
        for j in range(n2):
            u = (i + 0.5) / n1 * 2 - 1
            v = (j + 0.5) / n2 * 2 - 1
            
            if face_id == 0:
                lx, ly, lz = half_l, u * half_w, v * half_h
            elif face_id == 1:
                lx, ly, lz = -half_l, u * half_w, v * half_h
            elif face_id == 2:
                lx, ly, lz = u * half_l, half_w, v * half_h
            elif face_id == 3:
                lx, ly, lz = u * half_l, -half_w, v * half_h
            elif face_id == 4:
                lx, ly, lz = u * half_l, v * half_w, half_h
            else:
                lx, ly, lz = u * half_l, v * half_w, -half_h
            
            samples[idx, 0] = lx
            samples[idx, 1] = ly
            samples[idx, 2] = lz
            idx += 1
    
    return samples


@njit(cache=True)
def _sample_visible_surfaces(
    center: np.ndarray,
    half_dims: np.ndarray,
    R: np.ndarray,
    sensor: np.ndarray,
    total_samples: int,
    exclude_bottom: bool
) -> Tuple[np.ndarray, np.ndarray]:
    """在所有可见面上采样（按面积分配）"""
    half_l, half_w, half_h = half_dims[0], half_dims[1], half_dims[2]
    
    # 计算传感器在局部坐标系的位置
    diff = np.empty(3, dtype=np.float64)
    diff[0] = sensor[0] - center[0]
    diff[1] = sensor[1] - center[1]
    diff[2] = sensor[2] - center[2]
    
    sensor_local = np.empty(3, dtype=np.float64)
    for i in range(3):
        sensor_local[i] = R[0, i] * diff[0] + R[1, i] * diff[1] + R[2, i] * diff[2]
    
    # 判断可见面
    visible_faces = _determine_visible_faces(sensor_local, exclude_bottom)
    
    # 计算各面面积，按比例分配采样数
    areas = _compute_face_areas(half_l, half_w, half_h)
    
    total_visible_area = 0.0
    for i in range(6):
        if visible_faces[i]:
            total_visible_area += areas[i]
    
    if total_visible_area < 1e-6:
        return np.empty((0, 3), dtype=np.float64), np.empty(0, dtype=np.int64)
    
    samples_per_face = np.zeros(6, dtype=np.int64)
    for i in range(6):
        if visible_faces[i]:
            n = int(total_samples * areas[i] / total_visible_area + 0.5)
            samples_per_face[i] = max(n, 4)
    
    # 采样
    actual_total = 0
    for i in range(6):
        actual_total += samples_per_face[i]
    
    samples_world = np.empty((actual_total, 3), dtype=np.float64)
    face_ids = np.empty(actual_total, dtype=np.int64)
    
    write_idx = 0
    for face_id in range(6):
        if samples_per_face[face_id] == 0:
            continue
        
        local_samples = _sample_single_face_adaptive(
            face_id, half_l, half_w, half_h, samples_per_face[face_id]
        )
        
        for i in range(local_samples.shape[0]):
            lx, ly, lz = local_samples[i, 0], local_samples[i, 1], local_samples[i, 2]
            
            wx = R[0, 0] * lx + R[0, 1] * ly + R[0, 2] * lz + center[0]
            wy = R[1, 0] * lx + R[1, 1] * ly + R[1, 2] * lz + center[1]
            wz = R[2, 0] * lx + R[2, 1] * ly + R[2, 2] * lz + center[2]
            
            samples_world[write_idx, 0] = wx
            samples_world[write_idx, 1] = wy
            samples_world[write_idx, 2] = wz
            face_ids[write_idx] = face_id
            write_idx += 1
    
    return samples_world[:write_idx], face_ids[:write_idx]


@njit(cache=True)
def _check_ray_blocked(
    target: np.ndarray,
    sensor: np.ndarray,
    points: np.ndarray,
    angle_thresh: float,
    min_blockers: int
) -> bool:
    """检查射线是否被遮挡"""
    dx = target[0] - sensor[0]
    dy = target[1] - sensor[1]
    dz = target[2] - sensor[2]
    ray_len = np.sqrt(dx*dx + dy*dy + dz*dz)
    
    if ray_len < 1e-6:
        return False
    
    dir_x = dx / ray_len
    dir_y = dy / ray_len
    dir_z = dz / ray_len
    tan_thresh = np.tan(angle_thresh)
    
    blocker_count = 0
    n = points.shape[0]
    
    for i in range(n):
        px = points[i, 0] - sensor[0]
        py = points[i, 1] - sensor[1]
        pz = points[i, 2] - sensor[2]
        
        proj = px * dir_x + py * dir_y + pz * dir_z
        
        if proj <= 0.5 or proj >= ray_len - 0.3:
            continue
        
        closest_x = proj * dir_x
        closest_y = proj * dir_y
        closest_z = proj * dir_z
        
        perp_dist_sq = (px - closest_x)**2 + (py - closest_y)**2 + (pz - closest_z)**2
        dynamic_thresh = proj * tan_thresh
        
        if perp_dist_sq < dynamic_thresh * dynamic_thresh:
            blocker_count += 1
            if blocker_count >= min_blockers:
                return True
    
    return False


@njit(cache=True)
def _compute_surface_visibility(
    sample_points: np.ndarray,
    face_ids: np.ndarray,
    sensor: np.ndarray,
    scene_points: np.ndarray,
    angle_thresh: float,
    min_blockers: int
) -> Tuple[int, int, np.ndarray]:
    """计算表面采样点的可见性"""
    n_samples = sample_points.shape[0]
    
    if n_samples == 0:
        return 0, 0, np.zeros((6, 2), dtype=np.int64)
    
    face_stats = np.zeros((6, 2), dtype=np.int64)
    n_blocked = 0
    
    for i in range(n_samples):
        target = sample_points[i]
        face_id = face_ids[i]
        face_stats[face_id, 0] += 1
        
        if _check_ray_blocked(target, sensor, scene_points, angle_thresh, min_blockers):
            n_blocked += 1
            face_stats[face_id, 1] += 1
    
    return n_samples, n_blocked, face_stats


# ============================================================
# 数据格式转换
# ============================================================

def _parse_bbox_dict(bbox_dict: Dict) -> Tuple[np.ndarray, float, float, float, float, float, float]:
    """
    解析您的bbox字典格式
    
    输入格式:
    {
        'position': {'x': ..., 'y': ..., 'z': ...},
        'size': [w, h, l],  # size[0]=宽, size[1]=高, size[2]=长 (可能是字符串)
        'orientation': {'phi': ..., 'theta': ..., 'psi': ...},  # 可能是字符串
        ...
    }
    
    返回: center, length, width, height, phi, theta, psi
    """
    # 位置 (强制转float，兼容字符串)
    cx = float(bbox_dict['position']['x'])
    cy = float(bbox_dict['position']['y'])
    cz = float(bbox_dict['position']['z'])
    center = np.array([cx, cy, cz], dtype=np.float64)
    
    # 尺寸 (注意您代码中的顺序，强制转float兼容字符串)
    # l, w, h = float(bbox['size'][2]), float(bbox['size'][0]), float(bbox['size'][1])
    length = float(bbox_dict['size'][2])  # 长
    width = float(bbox_dict['size'][0])   # 宽
    height = float(bbox_dict['size'][1])  # 高
    
    # 确保尺寸有效（至少0.1m）
    length = max(length, 0.1)
    width = max(width, 0.1)
    height = max(height, 0.1)
    
    # 旋转角 (强制转float，兼容字符串)
    phi = float(bbox_dict['orientation']['phi'])      # Z轴 (yaw)
    theta = float(bbox_dict['orientation']['theta'])  # Y轴 (pitch)
    psi = float(bbox_dict['orientation']['psi'])      # X轴 (roll)
    
    return center, length, width, height, phi, theta, psi


# ============================================================
# 主接口函数
# ============================================================

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
    计算单个3D标注框的可见性
    
    Args:
        bbox_dict: 3D标注框字典，格式:
            {
                'position': {'x': float, 'y': float, 'z': float},
                'size': [w, h, l],
                'orientation': {'phi': float, 'theta': float, 'psi': float},
                'track_id': ...,
                ...
            }
        points: (N, 3) 或 (N, 4+) 点云数组
        sensor_origin: (3,) 传感器位置，默认原点 [0,0,0]
        total_samples: 总采样点数，默认80
        angle_thresh_deg: 射线角度阈值（度），默认0.5°
        min_blockers: 最少遮挡点数，默认3
        max_scene_points: 场景点云最大采样数，默认15000
        exclude_bottom: 是否排除底面，默认True
    
    Returns:
        score: 可见性分数 [0, 1]
        details: 详细信息字典
    """
    # 默认传感器在原点
    if sensor_origin is None:
        sensor_origin = np.array([0.0, 0.0, 0.0])
    
    # 解析bbox
    center, length, width, height, phi, theta, psi = _parse_bbox_dict(bbox_dict)
    
    # 准备数据
    points = np.ascontiguousarray(points[:, :3], dtype=np.float64)
    sensor = np.asarray(sensor_origin, dtype=np.float64)
    angle_thresh = np.deg2rad(angle_thresh_deg)
    
    # 计算旋转矩阵
    R = _rotation_matrix_zyx(phi, theta, psi)
    half_dims = np.array([length/2, width/2, height/2], dtype=np.float64)
    
    # 表面采样
    try:
        samples, face_ids = _sample_visible_surfaces(
            center, half_dims, R, sensor, total_samples, exclude_bottom
        )
    except Exception as e:
        # 采样失败，返回默认值
        return 0.0, {
            'score': 0.0,
            'n_samples': 0,
            'n_blocked': 0,
            'n_visible': 0,
            'status': 'ERROR',
            'face_visibility': {},
            'n_visible_faces': 0,
            'error': str(e)
        }
    
    n_samples = samples.shape[0] if samples is not None else 0
    if n_samples == 0:
        return 0.0, {
            'score': 0.0,
            'n_samples': 0,
            'n_blocked': 0,
            'n_visible': 0,
            'status': 'NO_VISIBLE_SURFACE',
            'face_visibility': {},
            'n_visible_faces': 0
        }
    
    # 场景点云降采样
    if points.shape[0] > max_scene_points:
        indices = np.random.choice(points.shape[0], max_scene_points, replace=False)
        scene_pts = points[indices]
    else:
        scene_pts = points
    
    # 计算遮挡
    n_total, n_blocked, face_stats = _compute_surface_visibility(
        samples, face_ids, sensor, scene_pts, angle_thresh, min_blockers
    )
    
    n_visible = n_total - n_blocked
    visibility = n_visible / n_total if n_total > 0 else 0.0
    
    # 每个面的可见性
    face_names = ['+x(前)', '-x(后)', '+y(左)', '-y(右)', '+z(顶)', '-z(底)']
    face_visibility = {}
    n_visible_faces = 0
    
    for i in range(6):
        total_on_face = face_stats[i, 0]
        blocked_on_face = face_stats[i, 1]
        if total_on_face > 0:
            n_visible_faces += 1
            vis = (total_on_face - blocked_on_face) / total_on_face
            face_visibility[face_names[i]] = {
                'visibility': float(vis),
                'total': int(total_on_face),
                'blocked': int(blocked_on_face)
            }
    
    # 状态判断
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
        'n_samples': int(n_total),
        'n_blocked': int(n_blocked),
        'n_visible': int(n_visible),
        'status': status,
        'face_visibility': face_visibility,
        'n_visible_faces': n_visible_faces
    }


def compute_frame_visibility(
    label_3d_list: List[Dict],
    pcd_pts: np.ndarray,
    sensor_origin: Optional[np.ndarray] = None,
    **kwargs
) -> Dict[str, Dict]:
    """
    计算一帧中所有3D标注框的可见性
    
    【主接口函数】- 直接在您的代码中调用此函数
    
    Args:
        label_3d_list: 3D标注框列表，每个元素是bbox字典
        pcd_pts: (N, 3) 点云数组
        sensor_origin: (3,) 传感器位置，默认原点
        **kwargs: 其他参数传递给compute_bbox_visibility
    
    Returns:
        results: 字典，key是track_id，value是可见性信息
        {
            'track_id_1': {
                'score': 0.85,
                'status': 'VISIBLE',
                'n_samples': 75,
                'n_blocked': 11,
                'n_visible': 64,
                'face_visibility': {...},
                ...
            },
            'track_id_2': {...},
            ...
        }
    
    使用示例:
    ```python
    from visibility_api import compute_frame_visibility
    
    # 在process_frame函数中调用
    results = compute_frame_visibility(label_3d_lidar, filter_pcd_pts)
    
    for track_id, info in results.items():
        print(f"目标 {track_id}: 可见性={info['score']:.2f}, 状态={info['status']}")
    ```
    """
    # 预热numba (首次调用会编译)
    if len(label_3d_list) > 0:
        # 用第一个bbox做预热
        _ = compute_bbox_visibility(label_3d_list[0], pcd_pts[:1000], sensor_origin, **kwargs)
    
    results = {}
    
    for bbox_dict in label_3d_list:
        # 获取track_id
        track_id = str(bbox_dict.get('track_id', 'unknown'))
        
        # 计算可见性
        score, details = compute_bbox_visibility(bbox_dict, pcd_pts, sensor_origin, **kwargs)
        
        # 存储结果
        results[track_id] = details
    
    return results


def classify_visibility(score: float) -> str:
    """
    将可见性分数转换为状态标签
    
    Args:
        score: 可见性分数 [0, 1]
    
    Returns:
        状态标签: 'VISIBLE', 'PARTIAL', 'OCCLUDED', 'BLOCKED'
    """
    if score >= 0.7:
        return "VISIBLE"
    if score >= 0.3:
        return "PARTIAL"
    if score > 0.05:
        return "OCCLUDED"
    return "BLOCKED"


# ============================================================
# 可视化接口
# ============================================================

def visualize_visibility(
    img: np.ndarray,
    label_3d_list: List[Dict],
    visibility_results: Dict[str, Dict],
    extrinsic: np.ndarray,
    intrinsic: np.ndarray,
    distortion: Optional[np.ndarray] = None,
    width: Optional[int] = None,
    height: Optional[int] = None,
    style: str = 'full',
    draw_wireframe: bool = True,
    add_legend: bool = True,
    filter_camera: Optional[str] = None
) -> np.ndarray:
    """
    在图像上可视化可见性结果
    
    Args:
        img: 输入图像 (BGR格式)
        label_3d_list: 3D标注框列表
        visibility_results: 可见性计算结果
        extrinsic: (4, 4) 外参矩阵
        intrinsic: (3, 3) 内参矩阵
        distortion: 畸变系数
        width: 图像宽度
        height: 图像高度
        style: 标签样式 ('full', 'score', 'status', 'compact')
        draw_wireframe: 是否绘制3D框线框
        add_legend: 是否添加图例
        filter_camera: 过滤相机名称
    
    Returns:
        绑制后的图像
    """
    if not HAS_VISUALIZER:
        raise ImportError("可视化功能需要cv2和visibility_visualizer模块")
    
    visualizer = VisibilityVisualizer()
    
    img = visualizer.draw_visibility_on_image(
        img, label_3d_list, visibility_results,
        extrinsic, intrinsic, distortion,
        width, height, style, draw_wireframe, filter_camera
    )
    
    if add_legend:
        img = visualizer.create_visibility_legend(img, position='top-left')
    
    return img


def get_projections(
    label_3d_list: List[Dict],
    visibility_results: Dict[str, Dict],
    extrinsic: np.ndarray,
    intrinsic: np.ndarray,
    distortion: Optional[np.ndarray] = None,
    width: int = 1920,
    height: int = 1080,
    filter_camera: Optional[str] = None
) -> List[Dict]:
    """
    获取3D框投影到图像的坐标（不绑制）
    
    Returns:
        投影结果列表
    """
    if not HAS_VISUALIZER:
        raise ImportError("需要visibility_visualizer模块")
    
    return project_bboxes_to_image(
        label_3d_list, visibility_results,
        extrinsic, intrinsic, distortion,
        width, height, filter_camera
    )


# ============================================================
# 测试代码
# ============================================================

if __name__ == '__main__':
    import time
    
    print("=" * 60)
    print("可见性计算接口测试")
    print("=" * 60)
    
    np.random.seed(42)
    
    # 模拟您的数据格式
    label_3d_list = [
        {
            'position': {'x': 15.0, 'y': 3.0, 'z': 0.8},
            'size': [2.0, 1.5, 4.5],  # [w, h, l]
            'orientation': {'phi': 0.1, 'theta': 0.0, 'psi': 0.0},
            'track_id': 'car_001',
            'camera_ids': ['front_camera']
        },
        {
            'position': {'x': 25.0, 'y': -5.0, 'z': 0.8},
            'size': [2.0, 1.5, 4.5],
            'orientation': {'phi': -0.2, 'theta': 0.0, 'psi': 0.0},
            'track_id': 'car_002',
            'camera_ids': ['front_camera']
        },
        {
            'position': {'x': 80.0, 'y': 10.0, 'z': 0.8},
            'size': [2.0, 1.5, 4.5],
            'orientation': {'phi': 0.0, 'theta': 0.0, 'psi': 0.0},
            'track_id': 'car_003',
            'camera_ids': ['front_camera']
        }
    ]
    
    # 模拟点云
    bg = np.random.randn(20000, 3)
    bg[:, 0] *= 30
    bg[:, 1] *= 30
    bg[:, 2] = np.abs(bg[:, 2]) * 1.5 - 0.5
    
    # 遮挡物
    blocker = []
    for x in np.linspace(18, 20, 20):
        for y in np.linspace(-7, -3, 30):
            for z in np.linspace(0, 2, 15):
                blocker.append([x, y, z])
    blocker = np.array(blocker) + np.random.randn(len(blocker), 3) * 0.05
    
    pcd_pts = np.vstack([bg, blocker])
    print(f"点云数量: {len(pcd_pts)}")
    print(f"标注框数量: {len(label_3d_list)}")
    
    # 测试主接口
    print("\n" + "-" * 40)
    print("调用 compute_frame_visibility")
    print("-" * 40)
    
    t0 = time.time()
    results = compute_frame_visibility(label_3d_list, pcd_pts)
    t1 = time.time()
    
    print(f"\n总耗时: {(t1-t0)*1000:.1f} ms")
    print(f"每框耗时: {(t1-t0)/len(label_3d_list)*1000:.1f} ms")
    
    print("\n" + "-" * 40)
    print("结果:")
    print("-" * 40)
    for track_id, info in results.items():
        print(f"\n目标 {track_id}:")
        print(f"  可见性: {info['score']:.2f}")
        print(f"  状态: {info['status']}")
        print(f"  采样点: {info['n_samples']}, 被挡: {info['n_blocked']}")
        if info['face_visibility']:
            print(f"  各面可见性:")
            for face, finfo in info['face_visibility'].items():
                print(f"    {face}: {finfo['visibility']:.2f}")
    
    print("\n" + "=" * 60)
