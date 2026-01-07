"""
3D Bounding Box Visibility Calculator (Ray-based)
==================================================

基于OCC3D射线投射思想的3D标注框可见性量化工具。

核心原理（来自OCC3D）：
    - OCCUPIED: 体素反射了LiDAR点（有点云落入）
    - FREE: 体素被LiDAR射线穿透（射线经过但无点）
    - UNKNOWN: 既无点云也无射线经过（未观测区域）

可见性计算：
    - 观测率 = (OCCUPIED + FREE) / 总体素数
    - 占用率 = OCCUPIED / (OCCUPIED + FREE)

依赖：numpy, numba(可选，用于加速)

使用示例：
    >>> from bbox_visibility import compute_visibility
    >>> bbox = [10, 5, 1, 4.5, 2, 1.5, 0.5]  # [cx,cy,cz,l,w,h,yaw]
    >>> score, details = compute_visibility(bbox, points, sensor_origin)
"""

import numpy as np
from typing import Tuple, Dict, List, Optional, Union

# Numba加速（可选）
try:
    from numba import njit, prange
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    def njit(*args, **kwargs):
        def decorator(func):
            return func
        if len(args) == 1 and callable(args[0]):
            return args[0]
        return decorator
    prange = range


# ==================== 体素状态常量 ====================
VOXEL_UNKNOWN = 0    # 未观测
VOXEL_FREE = 1       # 射线穿透（空闲）
VOXEL_OCCUPIED = 2   # 点云占用


# ==================== 3D Bresenham射线追踪 ====================

@njit(cache=True)
def _bresenham_3d(x0: int, y0: int, z0: int, 
                  x1: int, y1: int, z1: int,
                  max_steps: int = 10000) -> np.ndarray:
    """3D Bresenham算法：获取射线经过的所有体素坐标
    
    从(x0,y0,z0)到(x1,y1,z1)的直线经过的所有整数格点。
    
    Returns:
        voxels: (N, 3) 经过的体素坐标数组
    """
    # 预分配结果数组
    result = np.empty((max_steps, 3), dtype=np.int32)
    count = 0
    
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    dz = abs(z1 - z0)
    
    sx = 1 if x1 > x0 else -1
    sy = 1 if y1 > y0 else -1
    sz = 1 if z1 > z0 else -1
    
    # 主导轴
    if dx >= dy and dx >= dz:
        # X为主导轴
        err_y = 2 * dy - dx
        err_z = 2 * dz - dx
        
        x, y, z = x0, y0, z0
        for _ in range(dx + 1):
            if count < max_steps:
                result[count, 0] = x
                result[count, 1] = y
                result[count, 2] = z
                count += 1
            
            if err_y > 0:
                y += sy
                err_y -= 2 * dx
            if err_z > 0:
                z += sz
                err_z -= 2 * dx
            
            err_y += 2 * dy
            err_z += 2 * dz
            x += sx
            
    elif dy >= dx and dy >= dz:
        # Y为主导轴
        err_x = 2 * dx - dy
        err_z = 2 * dz - dy
        
        x, y, z = x0, y0, z0
        for _ in range(dy + 1):
            if count < max_steps:
                result[count, 0] = x
                result[count, 1] = y
                result[count, 2] = z
                count += 1
            
            if err_x > 0:
                x += sx
                err_x -= 2 * dy
            if err_z > 0:
                z += sz
                err_z -= 2 * dy
            
            err_x += 2 * dx
            err_z += 2 * dz
            y += sy
            
    else:
        # Z为主导轴
        err_x = 2 * dx - dz
        err_y = 2 * dy - dz
        
        x, y, z = x0, y0, z0
        for _ in range(dz + 1):
            if count < max_steps:
                result[count, 0] = x
                result[count, 1] = y
                result[count, 2] = z
                count += 1
            
            if err_x > 0:
                x += sx
                err_x -= 2 * dz
            if err_y > 0:
                y += sy
                err_y -= 2 * dz
            
            err_x += 2 * dx
            err_y += 2 * dy
            z += sz
    
    return result[:count]


# ==================== 核心计算函数 ====================

@njit(cache=True)
def _rotate_point(x: float, y: float, cos_a: float, sin_a: float) -> Tuple[float, float]:
    """2D旋转"""
    return cos_a * x - sin_a * y, sin_a * x + cos_a * y


@njit(cache=True)
def _world_to_voxel(px: float, py: float, pz: float,
                    bbox_center: np.ndarray,
                    bbox_size: np.ndarray,
                    cos_yaw: float, sin_yaw: float,
                    voxel_size: float,
                    nx: int, ny: int, nz: int) -> Tuple[int, int, int, bool]:
    """世界坐标转体素索引
    
    Returns:
        vx, vy, vz: 体素索引
        valid: 是否在bbox范围内
    """
    # 平移到bbox中心
    dx = px - bbox_center[0]
    dy = py - bbox_center[1]
    dz = pz - bbox_center[2]
    
    # 逆旋转到局部坐标系
    local_x, local_y = _rotate_point(dx, dy, cos_yaw, -sin_yaw)
    local_z = dz
    
    # 偏移到[0, size]范围
    local_x += bbox_size[0] / 2
    local_y += bbox_size[1] / 2
    local_z += bbox_size[2] / 2
    
    # 转换为体素索引
    vx = int(local_x / voxel_size)
    vy = int(local_y / voxel_size)
    vz = int(local_z / voxel_size)
    
    # 边界检查
    valid = (0 <= vx < nx) and (0 <= vy < ny) and (0 <= vz < nz)
    
    return vx, vy, vz, valid


@njit(cache=True)
def _compute_ray_voxel_states(
    points: np.ndarray,
    sensor_origin: np.ndarray,
    bbox_center: np.ndarray,
    bbox_size: np.ndarray,
    yaw: float,
    voxel_size: float
) -> Tuple[np.ndarray, int, int, int]:
    """基于射线投射计算体素状态
    
    对每个点云点：
    1. 从传感器发射射线到该点
    2. 射线经过的体素标记为FREE
    3. 终点体素标记为OCCUPIED
    
    Returns:
        voxel_grid: (nx, ny, nz) 体素状态网格
        n_occupied: OCCUPIED体素数
        n_free: FREE体素数
        n_total: 总体素数
    """
    # 计算体素网格尺寸
    nx = max(1, int(np.ceil(bbox_size[0] / voxel_size)))
    ny = max(1, int(np.ceil(bbox_size[1] / voxel_size)))
    nz = max(1, int(np.ceil(bbox_size[2] / voxel_size)))
    n_total = nx * ny * nz
    
    # 创建体素网格（一维数组模拟3D）
    voxel_grid = np.zeros(n_total, dtype=np.uint8)
    
    cos_yaw = np.cos(-yaw)
    sin_yaw = np.sin(-yaw)
    
    # 传感器在体素坐标系中的位置
    sensor_vx, sensor_vy, sensor_vz, _ = _world_to_voxel(
        sensor_origin[0], sensor_origin[1], sensor_origin[2],
        bbox_center, bbox_size, cos_yaw, sin_yaw, voxel_size, nx, ny, nz
    )
    
    n_points = points.shape[0]
    
    for i in range(n_points):
        px, py, pz = points[i, 0], points[i, 1], points[i, 2]
        
        # 点的体素坐标
        point_vx, point_vy, point_vz, point_valid = _world_to_voxel(
            px, py, pz, bbox_center, bbox_size, 
            cos_yaw, sin_yaw, voxel_size, nx, ny, nz
        )
        
        # 射线追踪：从传感器到点
        ray_voxels = _bresenham_3d(
            sensor_vx, sensor_vy, sensor_vz,
            point_vx, point_vy, point_vz
        )
        
        # 标记射线经过的体素
        n_ray = ray_voxels.shape[0]
        for j in range(n_ray):
            rvx = ray_voxels[j, 0]
            rvy = ray_voxels[j, 1]
            rvz = ray_voxels[j, 2]
            
            # 检查是否在bbox范围内
            if 0 <= rvx < nx and 0 <= rvy < ny and 0 <= rvz < nz:
                idx = rvx * ny * nz + rvy * nz + rvz
                
                # 最后一个体素是终点（OCCUPIED），其他是FREE
                if j == n_ray - 1 and point_valid:
                    # OCCUPIED优先级高于FREE
                    voxel_grid[idx] = VOXEL_OCCUPIED
                else:
                    # 只有当前状态为UNKNOWN时才标记为FREE
                    if voxel_grid[idx] == VOXEL_UNKNOWN:
                        voxel_grid[idx] = VOXEL_FREE
    
    # 统计
    n_occupied = 0
    n_free = 0
    for i in range(n_total):
        if voxel_grid[i] == VOXEL_OCCUPIED:
            n_occupied += 1
        elif voxel_grid[i] == VOXEL_FREE:
            n_free += 1
    
    return voxel_grid, n_occupied, n_free, n_total


@njit(cache=True)
def _points_in_box(points: np.ndarray, 
                   center: np.ndarray, 
                   size: np.ndarray, 
                   yaw: float) -> np.ndarray:
    """判断点是否在旋转包围盒内"""
    n = points.shape[0]
    mask = np.zeros(n, dtype=np.bool_)
    
    cos_yaw = np.cos(-yaw)
    sin_yaw = np.sin(-yaw)
    half_l = size[0] / 2
    half_w = size[1] / 2
    half_h = size[2] / 2
    
    for i in range(n):
        dx = points[i, 0] - center[0]
        dy = points[i, 1] - center[1]
        dz = points[i, 2] - center[2]
        
        local_x = cos_yaw * dx - sin_yaw * dy
        local_y = sin_yaw * dx + cos_yaw * dy
        
        if (abs(local_x) <= half_l and 
            abs(local_y) <= half_w and 
            abs(dz) <= half_h):
            mask[i] = True
    
    return mask


@njit(cache=True)
def _compute_surface_observation(
    voxel_grid: np.ndarray,
    nx: int, ny: int, nz: int,
    surface_depth: int = 2
) -> np.ndarray:
    """计算各表面的观测情况
    
    Returns:
        surface_obs: (6, 3) 每个面的[observed, occupied, total]
        顺序: [+x, -x, +y, -y, +z, -z]
    """
    surface_obs = np.zeros((6, 3), dtype=np.int64)
    
    # +x面 (front)
    for y in range(ny):
        for z in range(nz):
            for x in range(max(0, nx - surface_depth), nx):
                idx = x * ny * nz + y * nz + z
                surface_obs[0, 2] += 1  # total
                if voxel_grid[idx] != VOXEL_UNKNOWN:
                    surface_obs[0, 0] += 1  # observed
                if voxel_grid[idx] == VOXEL_OCCUPIED:
                    surface_obs[0, 1] += 1  # occupied
    
    # -x面 (back)
    for y in range(ny):
        for z in range(nz):
            for x in range(min(nx, surface_depth)):
                idx = x * ny * nz + y * nz + z
                surface_obs[1, 2] += 1
                if voxel_grid[idx] != VOXEL_UNKNOWN:
                    surface_obs[1, 0] += 1
                if voxel_grid[idx] == VOXEL_OCCUPIED:
                    surface_obs[1, 1] += 1
    
    # +y面 (left)
    for x in range(nx):
        for z in range(nz):
            for y in range(max(0, ny - surface_depth), ny):
                idx = x * ny * nz + y * nz + z
                surface_obs[2, 2] += 1
                if voxel_grid[idx] != VOXEL_UNKNOWN:
                    surface_obs[2, 0] += 1
                if voxel_grid[idx] == VOXEL_OCCUPIED:
                    surface_obs[2, 1] += 1
    
    # -y面 (right)
    for x in range(nx):
        for z in range(nz):
            for y in range(min(ny, surface_depth)):
                idx = x * ny * nz + y * nz + z
                surface_obs[3, 2] += 1
                if voxel_grid[idx] != VOXEL_UNKNOWN:
                    surface_obs[3, 0] += 1
                if voxel_grid[idx] == VOXEL_OCCUPIED:
                    surface_obs[3, 1] += 1
    
    # +z面 (top)
    for x in range(nx):
        for y in range(ny):
            for z in range(max(0, nz - surface_depth), nz):
                idx = x * ny * nz + y * nz + z
                surface_obs[4, 2] += 1
                if voxel_grid[idx] != VOXEL_UNKNOWN:
                    surface_obs[4, 0] += 1
                if voxel_grid[idx] == VOXEL_OCCUPIED:
                    surface_obs[4, 1] += 1
    
    # -z面 (bottom)
    for x in range(nx):
        for y in range(ny):
            for z in range(min(nz, surface_depth)):
                idx = x * ny * nz + y * nz + z
                surface_obs[5, 2] += 1
                if voxel_grid[idx] != VOXEL_UNKNOWN:
                    surface_obs[5, 0] += 1
                if voxel_grid[idx] == VOXEL_OCCUPIED:
                    surface_obs[5, 1] += 1
    
    return surface_obs


# ==================== 主函数 ====================

def compute_visibility(
    bbox: Union[np.ndarray, List],
    points: np.ndarray,
    sensor_origin: Optional[np.ndarray] = None,
    voxel_size: float = 0.2,
    weights: Optional[Dict[str, float]] = None
) -> Tuple[float, Dict]:
    """计算3D标注框的可见性（基于OCC3D射线投射思想）
    
    核心原理：
        1. 对每个点云点，从传感器发射射线
        2. 射线经过的体素标记为FREE（空闲）
        3. 射线终点体素标记为OCCUPIED（占用）
        4. 未被任何射线触及的体素为UNKNOWN（未知）
    
    Args:
        bbox: (7,) 标注框 [cx, cy, cz, length, width, height, yaw]
        points: (N, 3) 点云
        sensor_origin: (3,) 传感器位置，默认[0,0,0]
        voxel_size: 体素大小（米），默认0.2m
        weights: 指标权重，默认均匀
    
    Returns:
        score: 综合可见性分数 [0, 1]
        details: 详细指标字典
            - observation_ratio: 观测率 = (OCCUPIED+FREE) / TOTAL
            - occupancy_ratio: 占用率 = OCCUPIED / (OCCUPIED+FREE)
            - n_occupied: OCCUPIED体素数
            - n_free: FREE体素数  
            - n_unknown: UNKNOWN体素数
            - n_total: 总体素数
            - surface_observation: 各表面观测情况
    """
    # 默认值
    if sensor_origin is None:
        sensor_origin = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    
    if weights is None:
        weights = {
            'observation_ratio': 0.4,   # 观测覆盖率
            'occupancy_ratio': 0.3,     # 占用密度
            'surface_visibility': 0.3,  # 表面可见性
        }
    
    # 类型转换
    bbox = np.asarray(bbox, dtype=np.float64)
    points = np.asarray(points[:, :3], dtype=np.float64).copy()
    sensor_origin = np.asarray(sensor_origin, dtype=np.float64)
    
    center = bbox[:3]
    size = bbox[3:6]
    yaw = bbox[6]
    
    # 计算体素网格尺寸
    nx = max(1, int(np.ceil(size[0] / voxel_size)))
    ny = max(1, int(np.ceil(size[1] / voxel_size)))
    nz = max(1, int(np.ceil(size[2] / voxel_size)))
    
    # 射线投射计算体素状态
    voxel_grid, n_occupied, n_free, n_total = _compute_ray_voxel_states(
        points, sensor_origin, center, size, yaw, voxel_size
    )
    
    n_unknown = n_total - n_occupied - n_free
    n_observed = n_occupied + n_free
    
    # 计算各项指标
    details = {}
    
    # 1. 观测率：被观测到的体素占比
    observation_ratio = n_observed / max(1, n_total)
    details['observation_ratio'] = float(observation_ratio)
    
    # 2. 占用率：在被观测的体素中，占用的比例
    if n_observed > 0:
        occupancy_ratio = n_occupied / n_observed
    else:
        occupancy_ratio = 0.0
    details['occupancy_ratio'] = float(occupancy_ratio)
    
    # 3. 表面观测情况
    surface_obs = _compute_surface_observation(voxel_grid, nx, ny, nz)
    surface_names = ['front', 'back', 'left', 'right', 'top', 'bottom']
    surface_details = {}
    surface_vis_sum = 0.0
    surface_count = 0
    
    for i, name in enumerate(surface_names):
        observed = surface_obs[i, 0]
        occupied = surface_obs[i, 1]
        total = surface_obs[i, 2]
        
        if total > 0:
            obs_ratio = observed / total
            occ_ratio = occupied / max(1, observed)
        else:
            obs_ratio = 0.0
            occ_ratio = 0.0
        
        surface_details[name] = {
            'observation_ratio': float(obs_ratio),
            'occupancy_ratio': float(occ_ratio),
            'observed': int(observed),
            'occupied': int(occupied),
            'total': int(total),
        }
        
        # 表面可见性：观测率 * 占用率的调和
        if obs_ratio > 0:
            surface_vis_sum += obs_ratio
            surface_count += 1
    
    surface_visibility = surface_vis_sum / max(1, surface_count)
    details['surface_visibility'] = float(surface_visibility)
    details['surface_details'] = surface_details
    
    # 体素统计
    details['n_occupied'] = int(n_occupied)
    details['n_free'] = int(n_free)
    details['n_unknown'] = int(n_unknown)
    details['n_total'] = int(n_total)
    details['voxel_grid_shape'] = (nx, ny, nz)
    
    # 框内点数
    in_box = _points_in_box(points, center, size, yaw)
    details['points_in_box'] = int(np.sum(in_box))
    
    # 综合分数
    total_weight = sum(weights.values())
    score = (
        weights.get('observation_ratio', 0.4) / total_weight * observation_ratio +
        weights.get('occupancy_ratio', 0.3) / total_weight * min(1.0, occupancy_ratio * 5) +
        weights.get('surface_visibility', 0.3) / total_weight * surface_visibility
    )
    
    details['visibility_score'] = float(score)
    
    return score, details


def compute_visibility_batch(
    bboxes: np.ndarray,
    points: np.ndarray,
    sensor_origin: Optional[np.ndarray] = None,
    voxel_size: float = 0.2,
    weights: Optional[Dict[str, float]] = None
) -> Tuple[np.ndarray, List[Dict]]:
    """批量计算多个框的可见性"""
    bboxes = np.asarray(bboxes, dtype=np.float64)
    n_boxes = bboxes.shape[0]
    
    scores = np.zeros(n_boxes)
    details_list = []
    
    for i in range(n_boxes):
        s, d = compute_visibility(bboxes[i], points, sensor_origin, voxel_size, weights)
        scores[i] = s
        details_list.append(d)
    
    return scores, details_list


def classify_visibility(score: float) -> str:
    """可见性等级分类"""
    if score >= 0.7:
        return "FULLY_VISIBLE"
    elif score >= 0.4:
        return "MOSTLY_VISIBLE"
    elif score >= 0.2:
        return "PARTIALLY_VISIBLE"
    elif score > 0.05:
        return "MOSTLY_OCCLUDED"
    else:
        return "FULLY_OCCLUDED"


def warmup():
    """预热JIT编译"""
    if not HAS_NUMBA:
        return
    dummy_pts = np.random.randn(100, 3).astype(np.float64) * 10
    dummy_bbox = np.array([5, 5, 1, 4, 2, 1.5, 0], dtype=np.float64)
    dummy_sensor = np.array([0, 0, 1.5], dtype=np.float64)
    compute_visibility(dummy_bbox, dummy_pts, dummy_sensor, voxel_size=0.5)


# ==================== 可视化辅助 ====================

def get_voxel_grid(
    bbox: np.ndarray,
    points: np.ndarray,
    sensor_origin: np.ndarray,
    voxel_size: float = 0.2
) -> Tuple[np.ndarray, Dict]:
    """获取体素网格用于可视化
    
    Returns:
        voxel_states: (nx, ny, nz) 体素状态
            0=UNKNOWN, 1=FREE, 2=OCCUPIED
        info: 包含网格尺寸、bbox信息等
    """
    bbox = np.asarray(bbox, dtype=np.float64)
    points = np.asarray(points[:, :3], dtype=np.float64)
    sensor_origin = np.asarray(sensor_origin, dtype=np.float64)
    
    center = bbox[:3]
    size = bbox[3:6]
    yaw = bbox[6]
    
    nx = max(1, int(np.ceil(size[0] / voxel_size)))
    ny = max(1, int(np.ceil(size[1] / voxel_size)))
    nz = max(1, int(np.ceil(size[2] / voxel_size)))
    
    voxel_flat, n_occ, n_free, n_total = _compute_ray_voxel_states(
        points, sensor_origin, center, size, yaw, voxel_size
    )
    
    voxel_grid = voxel_flat.reshape((nx, ny, nz))
    
    info = {
        'shape': (nx, ny, nz),
        'voxel_size': voxel_size,
        'bbox_center': center,
        'bbox_size': size,
        'bbox_yaw': yaw,
        'n_occupied': n_occ,
        'n_free': n_free,
        'n_unknown': n_total - n_occ - n_free,
    }
    
    return voxel_grid, info


# ==================== 测试 ====================

if __name__ == '__main__':
    import time
    
    print("=" * 65)
    print("3D BBox Visibility Calculator (Ray-based, OCC3D Style)")
    print("=" * 65)
    print(f"Numba: {HAS_NUMBA}")
    
    np.random.seed(42)
    
    # 场景设置
    sensor = np.array([0, 0, 1.8])
    
    # 背景点云
    bg = np.random.randn(3000, 3) * 25
    bg[:, 2] = np.abs(bg[:, 2]) * 0.3 - 0.5
    
    # 目标1：可见车辆（表面有密集点云）
    car1_center = np.array([10, 3, 0.8])
    car1_size = np.array([4.5, 2.0, 1.5])
    car1_yaw = 0.1
    
    # 在车辆表面生成点
    car1_pts = []
    for _ in range(300):
        face = np.random.randint(6)
        if face == 0:
            p = [car1_size[0]/2, np.random.uniform(-1,1)*car1_size[1]/2,
                 np.random.uniform(-1,1)*car1_size[2]/2]
        elif face == 1:
            p = [-car1_size[0]/2, np.random.uniform(-1,1)*car1_size[1]/2,
                 np.random.uniform(-1,1)*car1_size[2]/2]
        elif face == 2:
            p = [np.random.uniform(-1,1)*car1_size[0]/2, car1_size[1]/2,
                 np.random.uniform(-1,1)*car1_size[2]/2]
        elif face == 3:
            p = [np.random.uniform(-1,1)*car1_size[0]/2, -car1_size[1]/2,
                 np.random.uniform(-1,1)*car1_size[2]/2]
        elif face == 4:
            p = [np.random.uniform(-1,1)*car1_size[0]/2,
                 np.random.uniform(-1,1)*car1_size[1]/2, car1_size[2]/2]
        else:
            p = [np.random.uniform(-1,1)*car1_size[0]/2,
                 np.random.uniform(-1,1)*car1_size[1]/2, -car1_size[2]/2]
        car1_pts.append(p)
    car1_pts = np.array(car1_pts)
    c, s = np.cos(car1_yaw), np.sin(car1_yaw)
    car1_pts_rot = np.zeros_like(car1_pts)
    car1_pts_rot[:, 0] = c * car1_pts[:, 0] - s * car1_pts[:, 1]
    car1_pts_rot[:, 1] = s * car1_pts[:, 0] + c * car1_pts[:, 1]
    car1_pts_rot[:, 2] = car1_pts[:, 2]
    car1_pts_world = car1_pts_rot + car1_center
    
    # 目标2：遮挡车辆（很少点）
    car2_center = np.array([35, -8, 0.8])
    car2_size = np.array([4.5, 2.0, 1.5])
    car2_yaw = -0.2
    car2_pts = np.random.randn(20, 3) * 0.15 + car2_center
    
    # 合并
    all_points = np.vstack([bg, car1_pts_world, car2_pts])
    
    # 预热
    print("\nWarming up JIT...")
    warmup()
    print("Done.")
    
    # 测试1：可见目标
    print("\n" + "-" * 50)
    print("Test 1: Visible Object")
    print("-" * 50)
    bbox1 = np.array([*car1_center, *car1_size, car1_yaw])
    t0 = time.time()
    score1, details1 = compute_visibility(bbox1, all_points, sensor, voxel_size=0.2)
    t1 = time.time()
    
    print(f"Score: {score1:.3f} ({classify_visibility(score1)})")
    print(f"Observation Ratio: {details1['observation_ratio']:.3f}")
    print(f"  (OBSERVED={details1['n_occupied']+details1['n_free']}, "
          f"UNKNOWN={details1['n_unknown']}, TOTAL={details1['n_total']})")
    print(f"Occupancy Ratio: {details1['occupancy_ratio']:.3f}")
    print(f"  (OCCUPIED={details1['n_occupied']}, FREE={details1['n_free']})")
    print(f"Surface Visibility: {details1['surface_visibility']:.3f}")
    print(f"Points in Box: {details1['points_in_box']}")
    print(f"Time: {(t1-t0)*1000:.2f} ms")
    
    # 测试2：遮挡目标
    print("\n" + "-" * 50)
    print("Test 2: Occluded Object")
    print("-" * 50)
    bbox2 = np.array([*car2_center, *car2_size, car2_yaw])
    t0 = time.time()
    score2, details2 = compute_visibility(bbox2, all_points, sensor, voxel_size=0.2)
    t1 = time.time()
    
    print(f"Score: {score2:.3f} ({classify_visibility(score2)})")
    print(f"Observation Ratio: {details2['observation_ratio']:.3f}")
    print(f"  (OBSERVED={details2['n_occupied']+details2['n_free']}, "
          f"UNKNOWN={details2['n_unknown']}, TOTAL={details2['n_total']})")
    print(f"Occupancy Ratio: {details2['occupancy_ratio']:.3f}")
    print(f"  (OCCUPIED={details2['n_occupied']}, FREE={details2['n_free']})")
    print(f"Surface Visibility: {details2['surface_visibility']:.3f}")
    print(f"Points in Box: {details2['points_in_box']}")
    print(f"Time: {(t1-t0)*1000:.2f} ms")
    
    # 性能测试
    print("\n" + "-" * 50)
    print("Test 3: Batch Performance")
    print("-" * 50)
    n_boxes = 50
    test_boxes = np.random.randn(n_boxes, 7)
    test_boxes[:, :3] *= 20
    test_boxes[:, 3:6] = np.abs(test_boxes[:, 3:6]) * 2 + 1
    test_boxes[:, 6] *= np.pi
    
    t0 = time.time()
    scores, _ = compute_visibility_batch(test_boxes, all_points, sensor, voxel_size=0.3)
    t1 = time.time()
    
    print(f"Boxes: {n_boxes}")
    print(f"Total: {(t1-t0)*1000:.1f} ms")
    print(f"Per box: {(t1-t0)/n_boxes*1000:.2f} ms")
    
    print("\n" + "=" * 65)
    print("OCC3D-Style Visibility Calculation:")
    print("=" * 65)
    print("""
原理：
  1. 从传感器向每个点云点发射射线
  2. 射线穿过的体素 → FREE（空闲）
  3. 射线终点体素 → OCCUPIED（占用）
  4. 未被射线触及 → UNKNOWN（未知/遮挡）

可见性指标：
  - observation_ratio = (OCCUPIED + FREE) / TOTAL
  - occupancy_ratio = OCCUPIED / (OCCUPIED + FREE)
  
使用：
  from bbox_visibility import compute_visibility
  score, details = compute_visibility(bbox, points, sensor_origin)
""")
