"""
3D标注框可见性计算器 (表面采样 + 射线遮挡检测)
==============================================

【核心思想】
在bbox表面采样，检测从传感器到采样点的射线是否被其他点云遮挡。

【为什么不检测框内点？】
框内已有的点 = 激光雷达能打到 = 必然未被遮挡
所以检测框内点没有意义。

【方法】
1. 在bbox朝向传感器的表面均匀采样
2. 对每个采样点，检测射线是否被其他点云遮挡
3. 可见性 = 未被遮挡的采样点数 / 总采样点数

【使用】
from bbox_visibility import compute_visibility
score, details = compute_visibility(bbox, points, sensor_origin)
"""

import numpy as np
from typing import Tuple, Dict, List, Union, Optional

try:
    from numba import njit
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    def njit(*args, **kwargs):
        def decorator(func): return func
        return decorator if not (args and callable(args[0])) else args[0]


@njit(cache=True)
def _sample_visible_surfaces(
    cx: float, cy: float, cz: float,
    hl: float, hw: float, hh: float,
    cos_yaw: float, sin_yaw: float,
    sensor_x: float, sensor_y: float, sensor_z: float,
    n_samples_per_face: int
) -> np.ndarray:
    """在朝向传感器的表面采样点
    
    只采样朝向传感器的面（法向量与视线方向夹角<90°）
    """
    # 传感器相对于bbox中心的方向（在bbox局部坐标系）
    dx = sensor_x - cx
    dy = sensor_y - cy
    dz = sensor_z - cz
    
    # 转到bbox局部坐标系
    local_dx = cos_yaw * dx + sin_yaw * dy
    local_dy = -sin_yaw * dx + cos_yaw * dy
    local_dz = dz
    
    # 判断每个面是否朝向传感器
    # 面法向量: +x, -x, +y, -y, +z, -z
    face_visible = np.array([
        local_dx > 0,   # +x面: 传感器在+x侧才可见
        local_dx < 0,   # -x面
        local_dy > 0,   # +y面
        local_dy < 0,   # -y面
        local_dz > 0,   # +z面
        local_dz < 0,   # -z面
    ])
    
    # 计算可见面数量
    n_visible = 0
    for i in range(6):
        if face_visible[i]:
            n_visible += 1
    
    if n_visible == 0:
        return np.empty((0, 3), dtype=np.float64)
    
    # 每个面的采样数
    samples_per_visible = n_samples_per_face
    total_samples = n_visible * samples_per_visible
    
    samples = np.empty((total_samples, 3), dtype=np.float64)
    idx = 0
    
    # 在每个可见面采样
    n_side = int(np.sqrt(samples_per_visible))
    
    for face in range(6):
        if not face_visible[face]:
            continue
        
        for i in range(n_side):
            for j in range(n_side):
                # 在面上均匀采样（加随机扰动避免规则网格）
                u = (i + 0.5) / n_side
                v = (j + 0.5) / n_side
                
                # 局部坐标
                if face == 0:    # +x
                    lx, ly, lz = hl, (u-0.5)*2*hw, (v-0.5)*2*hh
                elif face == 1:  # -x
                    lx, ly, lz = -hl, (u-0.5)*2*hw, (v-0.5)*2*hh
                elif face == 2:  # +y
                    lx, ly, lz = (u-0.5)*2*hl, hw, (v-0.5)*2*hh
                elif face == 3:  # -y
                    lx, ly, lz = (u-0.5)*2*hl, -hw, (v-0.5)*2*hh
                elif face == 4:  # +z
                    lx, ly, lz = (u-0.5)*2*hl, (v-0.5)*2*hw, hh
                else:            # -z
                    lx, ly, lz = (u-0.5)*2*hl, (v-0.5)*2*hw, -hh
                
                # 转回世界坐标
                wx = cos_yaw * lx - sin_yaw * ly + cx
                wy = sin_yaw * lx + cos_yaw * ly + cy
                wz = lz + cz
                
                if idx < total_samples:
                    samples[idx, 0] = wx
                    samples[idx, 1] = wy
                    samples[idx, 2] = wz
                    idx += 1
    
    return samples[:idx]


@njit(cache=True)
def _check_ray_blocked(
    target: np.ndarray,     # 目标点 (3,)
    sensor: np.ndarray,     # 传感器 (3,)
    points: np.ndarray,     # 场景点云 (N, 3)
    angle_thresh: float,    # 角度阈值
    min_blockers: int       # 最少阻挡点数
) -> bool:
    """检查射线是否被阻挡"""
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
        px = points[i,0] - sensor[0]
        py = points[i,1] - sensor[1]
        pz = points[i,2] - sensor[2]
        
        proj = px*dir_x + py*dir_y + pz*dir_z
        
        # 只考虑在sensor和target之间的点
        if proj <= 0.5 or proj >= ray_len - 0.3:
            continue
        
        # 垂直距离
        cx = proj * dir_x
        cy = proj * dir_y
        cz = proj * dir_z
        perp_dist_sq = (px-cx)**2 + (py-cy)**2 + (pz-cz)**2
        
        # 动态阈值
        thresh = proj * tan_thresh
        
        if perp_dist_sq < thresh * thresh:
            blocker_count += 1
            if blocker_count >= min_blockers:
                return True
    
    return False


@njit(cache=True)
def _compute_surface_visibility(
    sample_points: np.ndarray,   # 表面采样点 (M, 3)
    sensor: np.ndarray,          # 传感器 (3,)
    scene_points: np.ndarray,    # 场景点云 (N, 3)
    angle_thresh: float,
    min_blockers: int
) -> Tuple[int, int]:
    """计算表面采样点的可见性"""
    n_samples = sample_points.shape[0]
    if n_samples == 0:
        return 0, 0
    
    n_blocked = 0
    
    for i in range(n_samples):
        target = sample_points[i]
        if _check_ray_blocked(target, sensor, scene_points, angle_thresh, min_blockers):
            n_blocked += 1
    
    return n_samples, n_blocked


def compute_visibility(
    bbox: Union[np.ndarray, List],
    points: np.ndarray,
    sensor_origin: Optional[np.ndarray] = None,
    n_samples_per_face: int = 25,
    angle_thresh_deg: float = 0.5,
    min_blockers: int = 3,
    max_scene_points: int = 10000
) -> Tuple[float, Dict]:
    """计算3D标注框可见性
    
    【方法】
    在bbox朝向传感器的表面采样，检测射线是否被其他点云遮挡。
    
    Args:
        bbox: [cx, cy, cz, length, width, height, yaw]
        points: (N, 3) 场景点云
        sensor_origin: (3,) 传感器位置
        n_samples_per_face: 每个可见面的采样数
        angle_thresh_deg: 射线角度阈值（度）
        min_blockers: 最少阻挡点数
        max_scene_points: 场景点云最大采样数
    
    Returns:
        score: 可见性 [0, 1]
        details: 详细信息
    """
    if sensor_origin is None:
        sensor_origin = np.array([0.0, 0.0, 0.0])
    
    bbox = np.asarray(bbox, dtype=np.float64)
    points = np.ascontiguousarray(points[:,:3], dtype=np.float64)
    sensor = np.asarray(sensor_origin, dtype=np.float64)
    
    cx, cy, cz = bbox[0], bbox[1], bbox[2]
    l, w, h, yaw = bbox[3], bbox[4], bbox[5], bbox[6]
    cos_yaw, sin_yaw = np.cos(-yaw), np.sin(-yaw)
    angle_thresh = np.deg2rad(angle_thresh_deg)
    
    # 1. 表面采样
    samples = _sample_visible_surfaces(
        cx, cy, cz, l/2, w/2, h/2,
        cos_yaw, sin_yaw,
        sensor[0], sensor[1], sensor[2],
        n_samples_per_face
    )
    
    n_samples = samples.shape[0]
    if n_samples == 0:
        return 0.0, {
            'score': 0.0,
            'n_samples': 0,
            'n_blocked': 0,
            'n_visible': 0,
            'status': 'NO_VISIBLE_SURFACE'
        }
    
    # 2. 场景点云降采样（随机采样，更公平）
    if points.shape[0] > max_scene_points:
        indices = np.random.choice(points.shape[0], max_scene_points, replace=False)
        scene_pts = points[indices]
    else:
        scene_pts = points
    
    # 3. 计算遮挡
    n_total, n_blocked = _compute_surface_visibility(
        samples, sensor, scene_pts, angle_thresh, min_blockers
    )
    
    n_visible = n_total - n_blocked
    visibility = n_visible / n_total if n_total > 0 else 0.0
    
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
        'n_samples': int(n_total),
        'n_blocked': int(n_blocked),
        'n_visible': int(n_visible),
        'status': status
    }


def compute_visibility_batch(
    bboxes: np.ndarray,
    points: np.ndarray,
    sensor_origin: Optional[np.ndarray] = None,
    **kwargs
) -> Tuple[np.ndarray, List[Dict]]:
    """批量计算"""
    bboxes = np.asarray(bboxes, dtype=np.float64)
    scores = np.zeros(bboxes.shape[0])
    details = []
    for i, bbox in enumerate(bboxes):
        s, d = compute_visibility(bbox, points, sensor_origin, **kwargs)
        scores[i] = s
        details.append(d)
    return scores, details


def classify(score: float) -> str:
    if score >= 0.7: return "VISIBLE"
    if score >= 0.3: return "PARTIAL"
    if score > 0.05: return "OCCLUDED"
    return "BLOCKED"


# ==================== 测试 ====================
if __name__ == '__main__':
    import time
    
    np.random.seed(42)
    
    print("=" * 60)
    print("表面采样 + 射线遮挡检测")
    print("=" * 60)
    
    sensor = np.array([0, 0, 1.8])
    
    # 背景点云
    bg = np.random.randn(20000, 3)
    bg[:,0] *= 30
    bg[:,1] *= 30
    bg[:,2] = np.abs(bg[:,2]) * 1.5 - 0.5
    
    # 目标1: 可见车辆
    car1 = [15, 3, 0.8, 4.5, 2, 1.5, 0.1]
    
    # 目标2: 被遮挡车辆
    car2 = [25, -5, 0.8, 4.5, 2, 1.5, -0.2]
    
    # 遮挡物（墙）
    blocker = []
    for x in np.linspace(18, 20, 20):
        for y in np.linspace(-7, -3, 30):
            for z in np.linspace(0, 2, 15):
                blocker.append([x, y, z])
    blocker = np.array(blocker) + np.random.randn(len(blocker), 3) * 0.05
    
    # 目标3: 远处车辆（无遮挡）
    car3 = [80, 10, 0.8, 4.5, 2, 1.5, 0]
    
    all_points = np.vstack([bg, blocker])
    print(f"场景点云: {len(all_points)}")
    
    # 预热
    compute_visibility(car1, all_points, sensor)
    
    # 测试
    print("\n" + "-" * 40)
    print("测试1: 可见车辆")
    print("-" * 40)
    t0 = time.time()
    s1, d1 = compute_visibility(car1, all_points, sensor)
    t1 = time.time()
    print(f"可见性: {s1:.2f} ({d1['status']})")
    print(f"采样点: {d1['n_samples']}, 被挡: {d1['n_blocked']}, 可见: {d1['n_visible']}")
    print(f"耗时: {(t1-t0)*1000:.1f} ms")
    
    print("\n" + "-" * 40)
    print("测试2: 被遮挡车辆 (前方有墙)")
    print("-" * 40)
    t0 = time.time()
    s2, d2 = compute_visibility(car2, all_points, sensor)
    t1 = time.time()
    print(f"可见性: {s2:.2f} ({d2['status']})")
    print(f"采样点: {d2['n_samples']}, 被挡: {d2['n_blocked']}, 可见: {d2['n_visible']}")
    print(f"耗时: {(t1-t0)*1000:.1f} ms")
    
    print("\n" + "-" * 40)
    print("测试3: 远处车辆 (无遮挡)")
    print("-" * 40)
    t0 = time.time()
    s3, d3 = compute_visibility(car3, all_points, sensor)
    t1 = time.time()
    print(f"可见性: {s3:.2f} ({d3['status']})")
    print(f"采样点: {d3['n_samples']}, 被挡: {d3['n_blocked']}, 可见: {d3['n_visible']}")
    print(f"耗时: {(t1-t0)*1000:.1f} ms")
    
    # 批量测试
    print("\n" + "-" * 40)
    print("批量性能测试")
    print("-" * 40)
    n_boxes = 50
    boxes = np.zeros((n_boxes, 7))
    boxes[:,:2] = np.random.uniform(-40, 40, (n_boxes, 2))
    boxes[:,2] = np.random.uniform(0, 2, n_boxes)
    boxes[:,3:6] = [4.5, 2, 1.5]
    boxes[:,6] = np.random.uniform(-np.pi, np.pi, n_boxes)
    
    t0 = time.time()
    scores, _ = compute_visibility_batch(boxes, all_points, sensor)
    t1 = time.time()
    print(f"框数: {n_boxes}")
    print(f"总耗时: {(t1-t0)*1000:.1f} ms")
    print(f"每框: {(t1-t0)/n_boxes*1000:.2f} ms")
    
    print("\n" + "=" * 60)
