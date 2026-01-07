"""
================================================================================
3D Bounding Box Visibility Calculator (Ray-based, OCC3D Style)
================================================================================

版本: 2.0
基于OCC3D射线投射思想，计算3D标注框的可见性量化指标。

================================================================================
方法描述
================================================================================

【核心思想】
基于OCC3D论文的体素可见性定义，通过射线投射判断体素状态：
  - OCCUPIED（占用）: 体素反射了LiDAR点（有点云落入）
  - FREE（空闲）: 体素被LiDAR射线穿透（射线经过但无点）
  - UNKNOWN（未知）: 既无点云也无射线经过（遮挡或未观测区域）

【可见性定义】
可见性主要取决于"主视面"（Primary Visible Surface）的观测情况：
  1. 根据传感器相对于bbox的方向，确定各表面的可见权重
  2. 与传感器视线最垂直的面是"主视面"，权重最高
  3. 综合可见性 = Σ(各面可见性 × 各面权重)

【表面权重计算】
假设传感器在bbox的左前上方：
                    
       传感器 ●
              ╲
               ╲  视线方向
                ╲
                 ╲
         ┌───────────┐
        /│          /│
       / │   TOP   / │
      /  │        /  │
     ┌───────────┐   │
     │   │ BACK  │   │
     │   └───────│───┘
     │  /  LEFT  │  /
     │ /         │ /     ← 主视面：LEFT（与视线最垂直）
     │/          │/
     └───────────┘
         FRONT

权重计算基于视线方向与各面法向量的夹角：
  weight = max(0, dot(view_direction, surface_normal))

【算法流程】
1. 射线投射：对每个点云点
   - 从传感器发射射线到该点
   - 射线经过的体素 → FREE
   - 射线终点体素 → OCCUPIED
   - 使用3D Bresenham算法追踪射线路径

2. 表面分析：
   - 统计各表面的体素状态（OCCUPIED/FREE/UNKNOWN）
   - 计算各表面的观测率 = (OCCUPIED + FREE) / TOTAL
   - 计算各表面的占用率 = OCCUPIED / (OCCUPIED + FREE)

3. 可见性评分：
   - 根据传感器位置计算各面的可见权重
   - 主视面权重最高，背面权重为0
   - 综合分数 = Σ(面观测率 × 面权重) / Σ(权重)

【输出指标】
  - visibility_score: 综合可见性分数 [0, 1]
  - observation_ratio: 总体观测率
  - primary_surface: 主视面名称及其可见性
  - surface_weights: 各面的可见权重
  - surface_details: 各面的详细统计

================================================================================
使用方法
================================================================================

from bbox_visibility import compute_visibility, classify_visibility

# 输入
bbox = [cx, cy, cz, length, width, height, yaw]  # 7个参数
points = np.array(...)  # (N, 3) 点云
sensor_origin = [0, 0, 1.8]  # 传感器位置

# 计算
score, details = compute_visibility(bbox, points, sensor_origin)

# 输出
print(f"Visibility: {score:.2f} ({classify_visibility(score)})")
print(f"Primary Surface: {details['primary_surface']['name']}")
print(f"Primary Visibility: {details['primary_surface']['visibility']:.2f}")

================================================================================
依赖: numpy, numba(可选，用于加速)
================================================================================
"""

import numpy as np
from typing import Tuple, Dict, List, Optional, Union

# Numba加速（可选）
try:
    from numba import njit
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    def njit(*args, **kwargs):
        def decorator(func):
            return func
        if len(args) == 1 and callable(args[0]):
            return args[0]
        return decorator


# ==================== 常量 ====================
VOXEL_UNKNOWN = 0
VOXEL_FREE = 1
VOXEL_OCCUPIED = 2

SURFACE_NAMES = ['front', 'back', 'left', 'right', 'top', 'bottom']
# 各面的法向量（在bbox局部坐标系中）
# front(+x), back(-x), left(+y), right(-y), top(+z), bottom(-z)
SURFACE_NORMALS = np.array([
    [1, 0, 0],   # front +x
    [-1, 0, 0],  # back -x
    [0, 1, 0],   # left +y
    [0, -1, 0],  # right -y
    [0, 0, 1],   # top +z
    [0, 0, -1],  # bottom -z
], dtype=np.float64)


# ==================== 3D Bresenham 射线追踪 ====================

@njit(cache=True)
def _bresenham_3d(x0: int, y0: int, z0: int,
                  x1: int, y1: int, z1: int,
                  max_steps: int = 5000) -> np.ndarray:
    """3D Bresenham算法：获取射线经过的所有体素"""
    result = np.empty((max_steps, 3), dtype=np.int32)
    count = 0
    
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    dz = abs(z1 - z0)
    
    sx = 1 if x1 > x0 else -1
    sy = 1 if y1 > y0 else -1
    sz = 1 if z1 > z0 else -1
    
    if dx >= dy and dx >= dz:
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
def _world_to_voxel(px: float, py: float, pz: float,
                    center_x: float, center_y: float, center_z: float,
                    half_lx: float, half_ly: float, half_lz: float,
                    cos_yaw: float, sin_yaw: float,
                    voxel_size: float,
                    nx: int, ny: int, nz: int) -> Tuple[int, int, int, bool]:
    """世界坐标转体素索引"""
    dx = px - center_x
    dy = py - center_y
    dz = pz - center_z
    
    local_x = cos_yaw * dx + sin_yaw * dy
    local_y = -sin_yaw * dx + cos_yaw * dy
    local_z = dz
    
    # 映射到[0, size]
    grid_x = local_x + half_lx
    grid_y = local_y + half_ly
    grid_z = local_z + half_lz
    
    vx = int(grid_x / voxel_size)
    vy = int(grid_y / voxel_size)
    vz = int(grid_z / voxel_size)
    
    valid = (0 <= vx < nx) and (0 <= vy < ny) and (0 <= vz < nz)
    return vx, vy, vz, valid


@njit(cache=True)
def _compute_voxel_states(
    points: np.ndarray,
    sensor_x: float, sensor_y: float, sensor_z: float,
    center_x: float, center_y: float, center_z: float,
    size_x: float, size_y: float, size_z: float,
    yaw: float,
    voxel_size: float,
    nx: int, ny: int, nz: int
) -> np.ndarray:
    """射线投射计算体素状态"""
    n_total = nx * ny * nz
    voxel_grid = np.zeros(n_total, dtype=np.uint8)
    
    cos_yaw = np.cos(-yaw)
    sin_yaw = np.sin(-yaw)
    half_lx = size_x / 2
    half_ly = size_y / 2
    half_lz = size_z / 2
    
    # 传感器体素坐标
    sensor_vx, sensor_vy, sensor_vz, _ = _world_to_voxel(
        sensor_x, sensor_y, sensor_z,
        center_x, center_y, center_z,
        half_lx, half_ly, half_lz,
        cos_yaw, sin_yaw, voxel_size, nx, ny, nz
    )
    
    n_points = points.shape[0]
    
    for i in range(n_points):
        px, py, pz = points[i, 0], points[i, 1], points[i, 2]
        
        point_vx, point_vy, point_vz, point_valid = _world_to_voxel(
            px, py, pz,
            center_x, center_y, center_z,
            half_lx, half_ly, half_lz,
            cos_yaw, sin_yaw, voxel_size, nx, ny, nz
        )
        
        ray_voxels = _bresenham_3d(
            sensor_vx, sensor_vy, sensor_vz,
            point_vx, point_vy, point_vz
        )
        
        n_ray = ray_voxels.shape[0]
        for j in range(n_ray):
            rvx = ray_voxels[j, 0]
            rvy = ray_voxels[j, 1]
            rvz = ray_voxels[j, 2]
            
            if 0 <= rvx < nx and 0 <= rvy < ny and 0 <= rvz < nz:
                idx = rvx * ny * nz + rvy * nz + rvz
                
                if j == n_ray - 1 and point_valid:
                    voxel_grid[idx] = VOXEL_OCCUPIED
                else:
                    if voxel_grid[idx] == VOXEL_UNKNOWN:
                        voxel_grid[idx] = VOXEL_FREE
    
    return voxel_grid


@njit(cache=True)
def _analyze_surfaces(
    voxel_grid: np.ndarray,
    nx: int, ny: int, nz: int,
    depth: int
) -> np.ndarray:
    """分析各表面的体素状态
    
    Returns:
        stats: (6, 3) [observed, occupied, total] for each surface
    """
    stats = np.zeros((6, 3), dtype=np.int64)
    
    # front (+x): x in [nx-depth, nx)
    for x in range(max(0, nx - depth), nx):
        for y in range(ny):
            for z in range(nz):
                idx = x * ny * nz + y * nz + z
                stats[0, 2] += 1
                if voxel_grid[idx] != VOXEL_UNKNOWN:
                    stats[0, 0] += 1
                if voxel_grid[idx] == VOXEL_OCCUPIED:
                    stats[0, 1] += 1
    
    # back (-x): x in [0, depth)
    for x in range(min(nx, depth)):
        for y in range(ny):
            for z in range(nz):
                idx = x * ny * nz + y * nz + z
                stats[1, 2] += 1
                if voxel_grid[idx] != VOXEL_UNKNOWN:
                    stats[1, 0] += 1
                if voxel_grid[idx] == VOXEL_OCCUPIED:
                    stats[1, 1] += 1
    
    # left (+y): y in [ny-depth, ny)
    for x in range(nx):
        for y in range(max(0, ny - depth), ny):
            for z in range(nz):
                idx = x * ny * nz + y * nz + z
                stats[2, 2] += 1
                if voxel_grid[idx] != VOXEL_UNKNOWN:
                    stats[2, 0] += 1
                if voxel_grid[idx] == VOXEL_OCCUPIED:
                    stats[2, 1] += 1
    
    # right (-y): y in [0, depth)
    for x in range(nx):
        for y in range(min(ny, depth)):
            for z in range(nz):
                idx = x * ny * nz + y * nz + z
                stats[3, 2] += 1
                if voxel_grid[idx] != VOXEL_UNKNOWN:
                    stats[3, 0] += 1
                if voxel_grid[idx] == VOXEL_OCCUPIED:
                    stats[3, 1] += 1
    
    # top (+z): z in [nz-depth, nz)
    for x in range(nx):
        for y in range(ny):
            for z in range(max(0, nz - depth), nz):
                idx = x * ny * nz + y * nz + z
                stats[4, 2] += 1
                if voxel_grid[idx] != VOXEL_UNKNOWN:
                    stats[4, 0] += 1
                if voxel_grid[idx] == VOXEL_OCCUPIED:
                    stats[4, 1] += 1
    
    # bottom (-z): z in [0, depth)
    for x in range(nx):
        for y in range(ny):
            for z in range(min(nz, depth)):
                idx = x * ny * nz + y * nz + z
                stats[5, 2] += 1
                if voxel_grid[idx] != VOXEL_UNKNOWN:
                    stats[5, 0] += 1
                if voxel_grid[idx] == VOXEL_OCCUPIED:
                    stats[5, 1] += 1
    
    return stats


def _compute_surface_weights(
    sensor_origin: np.ndarray,
    bbox_center: np.ndarray,
    bbox_yaw: float
) -> np.ndarray:
    """计算各表面的可见权重
    
    基于传感器视线方向与各面法向量的夹角。
    权重 = max(0, dot(view_dir, normal))
    
    主视面（与视线最垂直的面）权重最高。
    """
    # 从bbox中心指向传感器的方向（视线反方向）
    view_dir = sensor_origin - bbox_center
    view_dist = np.linalg.norm(view_dir)
    if view_dist < 1e-6:
        return np.ones(6) / 6
    view_dir = view_dir / view_dist
    
    # 将视线方向转换到bbox局部坐标系
    cos_yaw = np.cos(-bbox_yaw)
    sin_yaw = np.sin(-bbox_yaw)
    local_view_dir = np.array([
        cos_yaw * view_dir[0] + sin_yaw * view_dir[1],
        -sin_yaw * view_dir[0] + cos_yaw * view_dir[1],
        view_dir[2]
    ])
    
    # 计算与各面法向量的点积
    weights = np.zeros(6)
    for i in range(6):
        dot = np.dot(local_view_dir, SURFACE_NORMALS[i])
        weights[i] = max(0, dot)  # 只有面向传感器的面有权重
    
    # 归一化
    total = np.sum(weights)
    if total > 1e-6:
        weights = weights / total
    
    return weights


# ==================== 主函数 ====================

def compute_visibility(
    bbox: Union[np.ndarray, List],
    points: np.ndarray,
    sensor_origin: Optional[np.ndarray] = None,
    voxel_size: float = 0.2,
    surface_depth: int = 2
) -> Tuple[float, Dict]:
    """计算3D标注框的可见性
    
    【方法】
    1. 射线投射：从传感器向每个点发射射线
       - 穿过的体素 → FREE
       - 终点体素 → OCCUPIED
       - 未触及 → UNKNOWN
    
    2. 表面权重：根据传感器位置计算各面权重
       - 主视面（最正对传感器）权重最高
       - 背面权重为0
    
    3. 可见性 = Σ(各面观测率 × 权重)
    
    Args:
        bbox: (7,) [cx, cy, cz, length, width, height, yaw]
        points: (N, 3) 点云
        sensor_origin: (3,) 传感器位置，默认[0,0,0]
        voxel_size: 体素大小（米）
        surface_depth: 表面分析深度（体素层数）
    
    Returns:
        score: 综合可见性分数 [0, 1]
        details: 详细指标字典
    """
    if sensor_origin is None:
        sensor_origin = np.array([0.0, 0.0, 0.0])
    
    bbox = np.asarray(bbox, dtype=np.float64)
    points = np.asarray(points[:, :3], dtype=np.float64).copy()
    sensor_origin = np.asarray(sensor_origin, dtype=np.float64)
    
    center = bbox[:3]
    size = bbox[3:6]
    yaw = bbox[6]
    
    nx = max(1, int(np.ceil(size[0] / voxel_size)))
    ny = max(1, int(np.ceil(size[1] / voxel_size)))
    nz = max(1, int(np.ceil(size[2] / voxel_size)))
    n_total = nx * ny * nz
    
    # 射线投射
    voxel_grid = _compute_voxel_states(
        points,
        sensor_origin[0], sensor_origin[1], sensor_origin[2],
        center[0], center[1], center[2],
        size[0], size[1], size[2],
        yaw, voxel_size, nx, ny, nz
    )
    
    # 全局统计
    n_occupied = int(np.sum(voxel_grid == VOXEL_OCCUPIED))
    n_free = int(np.sum(voxel_grid == VOXEL_FREE))
    n_unknown = n_total - n_occupied - n_free
    n_observed = n_occupied + n_free
    
    # 表面分析
    surface_stats = _analyze_surfaces(voxel_grid, nx, ny, nz, surface_depth)
    
    # 表面权重（基于视角）
    surface_weights = _compute_surface_weights(sensor_origin, center, yaw)
    
    # 各表面可见性
    surface_visibility = np.zeros(6)
    surface_details = {}
    
    for i, name in enumerate(SURFACE_NAMES):
        observed = int(surface_stats[i, 0])
        occupied = int(surface_stats[i, 1])
        total = int(surface_stats[i, 2])
        
        if total > 0:
            obs_ratio = observed / total
        else:
            obs_ratio = 0.0
        
        surface_visibility[i] = obs_ratio
        surface_details[name] = {
            'observation_ratio': float(obs_ratio),
            'observed': observed,
            'occupied': occupied,
            'total': total,
            'weight': float(surface_weights[i]),
        }
    
    # 加权可见性分数
    weighted_visibility = np.sum(surface_visibility * surface_weights)
    
    # 主视面（权重最高的面）
    primary_idx = int(np.argmax(surface_weights))
    primary_name = SURFACE_NAMES[primary_idx]
    primary_visibility = surface_visibility[primary_idx]
    
    # 构建详情
    details = {
        'visibility_score': float(weighted_visibility),
        
        # 主视面信息
        'primary_surface': {
            'name': primary_name,
            'visibility': float(primary_visibility),
            'weight': float(surface_weights[primary_idx]),
        },
        
        # 全局统计
        'observation_ratio': float(n_observed / max(1, n_total)),
        'n_occupied': n_occupied,
        'n_free': n_free,
        'n_unknown': n_unknown,
        'n_total': n_total,
        'voxel_shape': (nx, ny, nz),
        
        # 各表面详情
        'surface_weights': {name: float(surface_weights[i]) 
                           for i, name in enumerate(SURFACE_NAMES)},
        'surface_details': surface_details,
    }
    
    return weighted_visibility, details


def compute_visibility_batch(
    bboxes: np.ndarray,
    points: np.ndarray,
    sensor_origin: Optional[np.ndarray] = None,
    voxel_size: float = 0.2
) -> Tuple[np.ndarray, List[Dict]]:
    """批量计算"""
    bboxes = np.asarray(bboxes, dtype=np.float64)
    n = bboxes.shape[0]
    scores = np.zeros(n)
    details_list = []
    for i in range(n):
        s, d = compute_visibility(bboxes[i], points, sensor_origin, voxel_size)
        scores[i] = s
        details_list.append(d)
    return scores, details_list


def classify_visibility(score: float) -> str:
    """可见性等级"""
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
    """预热JIT"""
    if not HAS_NUMBA:
        return
    pts = np.random.randn(50, 3).astype(np.float64) * 5
    bbox = np.array([2, 2, 1, 3, 2, 1.5, 0], dtype=np.float64)
    sensor = np.array([0, 0, 1.5], dtype=np.float64)
    compute_visibility(bbox, pts, sensor, voxel_size=0.5)


# ==================== 测试 ====================

if __name__ == '__main__':
    import time
    
    print("=" * 70)
    print("3D BBox Visibility Calculator (Ray-based, Primary Surface Weighted)")
    print("=" * 70)
    print(f"Numba: {HAS_NUMBA}")
    
    np.random.seed(42)
    
    # 场景
    sensor = np.array([0, 0, 1.8])
    
    # 背景
    bg = np.random.randn(3000, 3) * 25
    bg[:, 2] = np.abs(bg[:, 2]) * 0.3 - 0.5
    
    # 可见车辆（表面密集点云）
    car1_center = np.array([10, 3, 0.8])
    car1_size = np.array([4.5, 2.0, 1.5])
    car1_yaw = 0.1
    
    car1_pts = []
    for _ in range(300):
        face = np.random.randint(6)
        hs = car1_size / 2
        if face == 0:
            p = [hs[0], np.random.uniform(-1,1)*hs[1], np.random.uniform(-1,1)*hs[2]]
        elif face == 1:
            p = [-hs[0], np.random.uniform(-1,1)*hs[1], np.random.uniform(-1,1)*hs[2]]
        elif face == 2:
            p = [np.random.uniform(-1,1)*hs[0], hs[1], np.random.uniform(-1,1)*hs[2]]
        elif face == 3:
            p = [np.random.uniform(-1,1)*hs[0], -hs[1], np.random.uniform(-1,1)*hs[2]]
        elif face == 4:
            p = [np.random.uniform(-1,1)*hs[0], np.random.uniform(-1,1)*hs[1], hs[2]]
        else:
            p = [np.random.uniform(-1,1)*hs[0], np.random.uniform(-1,1)*hs[1], -hs[2]]
        car1_pts.append(p)
    car1_pts = np.array(car1_pts)
    c, s = np.cos(car1_yaw), np.sin(car1_yaw)
    car1_rot = np.zeros_like(car1_pts)
    car1_rot[:, 0] = c * car1_pts[:, 0] - s * car1_pts[:, 1]
    car1_rot[:, 1] = s * car1_pts[:, 0] + c * car1_pts[:, 1]
    car1_rot[:, 2] = car1_pts[:, 2]
    car1_world = car1_rot + car1_center
    
    # 遮挡车辆
    car2_center = np.array([35, -8, 0.8])
    car2_size = np.array([4.5, 2.0, 1.5])
    car2_yaw = -0.2
    car2_pts = np.random.randn(20, 3) * 0.15 + car2_center
    
    all_points = np.vstack([bg, car1_world, car2_pts])
    
    # 预热
    print("\nWarming up...")
    warmup()
    
    # 测试1
    print("\n" + "-" * 60)
    print("Test 1: Visible Object (sensor at left-front)")
    print("-" * 60)
    bbox1 = np.array([*car1_center, *car1_size, car1_yaw])
    t0 = time.time()
    score1, d1 = compute_visibility(bbox1, all_points, sensor, voxel_size=0.2)
    t1 = time.time()
    
    print(f"Visibility Score: {score1:.3f} ({classify_visibility(score1)})")
    print(f"Primary Surface: {d1['primary_surface']['name']} "
          f"(weight={d1['primary_surface']['weight']:.2f}, "
          f"visibility={d1['primary_surface']['visibility']:.2f})")
    print(f"Observation Ratio: {d1['observation_ratio']:.3f}")
    print(f"  OCCUPIED={d1['n_occupied']}, FREE={d1['n_free']}, UNKNOWN={d1['n_unknown']}")
    print(f"\nSurface Weights:")
    for name in SURFACE_NAMES:
        w = d1['surface_weights'][name]
        v = d1['surface_details'][name]['observation_ratio']
        print(f"  {name:8s}: weight={w:.3f}, visibility={v:.3f}")
    print(f"\nTime: {(t1-t0)*1000:.2f} ms")
    
    # 测试2
    print("\n" + "-" * 60)
    print("Test 2: Occluded Object")
    print("-" * 60)
    bbox2 = np.array([*car2_center, *car2_size, car2_yaw])
    t0 = time.time()
    score2, d2 = compute_visibility(bbox2, all_points, sensor, voxel_size=0.2)
    t1 = time.time()
    
    print(f"Visibility Score: {score2:.3f} ({classify_visibility(score2)})")
    print(f"Primary Surface: {d2['primary_surface']['name']} "
          f"(weight={d2['primary_surface']['weight']:.2f}, "
          f"visibility={d2['primary_surface']['visibility']:.2f})")
    print(f"Observation Ratio: {d2['observation_ratio']:.3f}")
    print(f"Time: {(t1-t0)*1000:.2f} ms")
    
    # 性能
    print("\n" + "-" * 60)
    print("Test 3: Batch Performance")
    print("-" * 60)
    n_boxes = 50
    test_boxes = np.random.randn(n_boxes, 7)
    test_boxes[:, :3] *= 20
    test_boxes[:, 3:6] = np.abs(test_boxes[:, 3:6]) * 2 + 1
    test_boxes[:, 6] *= np.pi
    
    t0 = time.time()
    scores, _ = compute_visibility_batch(test_boxes, all_points, sensor)
    t1 = time.time()
    print(f"Boxes: {n_boxes}, Total: {(t1-t0)*1000:.1f}ms, Per box: {(t1-t0)/n_boxes*1000:.2f}ms")
    
    print("\n" + "=" * 70)
