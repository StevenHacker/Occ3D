"""
================================================================================
3D Bounding Box Visibility Calculator (Ray-based, OCC3D Style)
================================================================================

版本: 3.0
基于OCC3D射线投射思想，计算3D标注框的可见性量化指标。

================================================================================
方法描述
================================================================================

【背景】
对于360度激光雷达（如Velodyne、Ouster），传感器全方位扫描，物体的
前/后/左/右各面理论上都可以被观测到。可见性降低的原因是"遮挡"，
而非"视角"。

【核心思想】
基于OCC3D论文的体素可见性定义：
  - OCCUPIED：体素反射了LiDAR点（有点云落入）
  - FREE：体素被LiDAR射线穿透（射线经过但无点停留）
  - UNKNOWN：既无点云也无射线经过（被遮挡或距离过远）

【可见性定义】
可见性主要取决于"主要面"（Primary Surface）的观测情况：
  - 主要面 = 面积最大的面（通常是车辆侧面）
  - 对于bbox尺寸为(L, W, H)的物体：
    * 侧面(left/right): L × H
    * 前后面(front/back): W × H  
    * 顶底面(top/bottom): L × W

【表面权重计算】
基于表面面积的权重，面积越大权重越高：

        ┌─────────────────────────┐
       /│                        /│
      / │      TOP (L×W)        / │
     /  │                      /  │
    ┌─────────────────────────┐   │
    │   │                     │   │
    │   │  BACK               │   │  SIDE
    │   │  (W×H)              │   │  (L×H)
    │   └─────────────────────│───┘  ← 最大面
    │  /                      │  /
    │ /      BOTTOM           │ /
    │/       (L×W)            │/
    └─────────────────────────┘
           FRONT (W×H)

对于典型车辆 (L=4.5, W=2.0, H=1.5)：
  - 侧面: 4.5 × 1.5 = 6.75 m²  ← 主要面
  - 前后: 2.0 × 1.5 = 3.00 m²
  - 顶底: 4.5 × 2.0 = 9.00 m²  (但底面通常不可见)

【算法流程】
1. 射线投射：对每个点云点
   - 从传感器发射射线到该点
   - 射线经过的体素 → FREE
   - 射线终点体素 → OCCUPIED
   - 使用3D Bresenham算法追踪射线路径

2. 表面分析：
   - 统计各表面区域的体素状态
   - 各面可见性 = 被观测体素数 / 总体素数

3. 面积加权：
   - 各面权重 = 面面积 / 总面积（可选排除底面）
   - 综合可见性 = Σ(面可见性 × 面权重)

4. 主要面评估：
   - 主要面 = 面积最大的面
   - 报告主要面的可见性作为关键指标

【输出指标】
  - visibility_score: 综合可见性分数 [0, 1]
  - primary_surface: 主要面（面积最大）的名称和可见性
  - observation_ratio: 总体观测率
  - surface_details: 各面详细统计

================================================================================
使用方法
================================================================================

from bbox_visibility import compute_visibility, classify_visibility

bbox = [cx, cy, cz, length, width, height, yaw]
points = np.array(...)  # (N, 3)
sensor_origin = [0, 0, 1.8]

score, details = compute_visibility(bbox, points, sensor_origin)

print(f"Score: {score:.2f}")
print(f"Primary Surface: {details['primary_surface']['name']}")
print(f"Primary Visibility: {details['primary_surface']['visibility']:.2f}")

================================================================================
依赖: numpy, numba(可选)
================================================================================
"""

import numpy as np
from typing import Tuple, Dict, List, Optional, Union

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

# 表面名称: front(+x), back(-x), left(+y), right(-y), top(+z), bottom(-z)
SURFACE_NAMES = ['front', 'back', 'left', 'right', 'top', 'bottom']


# ==================== 3D Bresenham ====================

@njit(cache=True)
def _bresenham_3d(x0: int, y0: int, z0: int,
                  x1: int, y1: int, z1: int,
                  max_steps: int = 5000) -> np.ndarray:
    """3D Bresenham射线追踪"""
    result = np.empty((max_steps, 3), dtype=np.int32)
    count = 0
    
    dx, dy, dz = abs(x1-x0), abs(y1-y0), abs(z1-z0)
    sx = 1 if x1 > x0 else -1
    sy = 1 if y1 > y0 else -1
    sz = 1 if z1 > z0 else -1
    
    if dx >= dy and dx >= dz:
        err_y, err_z = 2*dy - dx, 2*dz - dx
        x, y, z = x0, y0, z0
        for _ in range(dx + 1):
            if count < max_steps:
                result[count, 0], result[count, 1], result[count, 2] = x, y, z
                count += 1
            if err_y > 0: y += sy; err_y -= 2*dx
            if err_z > 0: z += sz; err_z -= 2*dx
            err_y += 2*dy; err_z += 2*dz; x += sx
    elif dy >= dx and dy >= dz:
        err_x, err_z = 2*dx - dy, 2*dz - dy
        x, y, z = x0, y0, z0
        for _ in range(dy + 1):
            if count < max_steps:
                result[count, 0], result[count, 1], result[count, 2] = x, y, z
                count += 1
            if err_x > 0: x += sx; err_x -= 2*dy
            if err_z > 0: z += sz; err_z -= 2*dy
            err_x += 2*dx; err_z += 2*dz; y += sy
    else:
        err_x, err_y = 2*dx - dz, 2*dy - dz
        x, y, z = x0, y0, z0
        for _ in range(dz + 1):
            if count < max_steps:
                result[count, 0], result[count, 1], result[count, 2] = x, y, z
                count += 1
            if err_x > 0: x += sx; err_x -= 2*dz
            if err_y > 0: y += sy; err_y -= 2*dz
            err_x += 2*dx; err_y += 2*dy; z += sz
    
    return result[:count]


# ==================== 核心计算 ====================

@njit(cache=True)
def _world_to_voxel(px: float, py: float, pz: float,
                    cx: float, cy: float, cz: float,
                    hx: float, hy: float, hz: float,
                    cos_yaw: float, sin_yaw: float,
                    voxel_size: float,
                    nx: int, ny: int, nz: int) -> Tuple[int, int, int, bool]:
    """世界坐标转体素索引"""
    dx, dy, dz = px - cx, py - cy, pz - cz
    local_x = cos_yaw * dx + sin_yaw * dy
    local_y = -sin_yaw * dx + cos_yaw * dy
    
    vx = int((local_x + hx) / voxel_size)
    vy = int((local_y + hy) / voxel_size)
    vz = int((dz + hz) / voxel_size)
    
    valid = (0 <= vx < nx) and (0 <= vy < ny) and (0 <= vz < nz)
    return vx, vy, vz, valid


@njit(cache=True)
def _compute_voxel_states(
    points: np.ndarray,
    sensor_x: float, sensor_y: float, sensor_z: float,
    cx: float, cy: float, cz: float,
    sx: float, sy: float, sz: float,
    yaw: float, voxel_size: float,
    nx: int, ny: int, nz: int
) -> np.ndarray:
    """射线投射计算体素状态"""
    voxel_grid = np.zeros(nx * ny * nz, dtype=np.uint8)
    
    cos_yaw, sin_yaw = np.cos(-yaw), np.sin(-yaw)
    hx, hy, hz = sx/2, sy/2, sz/2
    
    sensor_vx, sensor_vy, sensor_vz, _ = _world_to_voxel(
        sensor_x, sensor_y, sensor_z, cx, cy, cz,
        hx, hy, hz, cos_yaw, sin_yaw, voxel_size, nx, ny, nz
    )
    
    for i in range(points.shape[0]):
        px, py, pz = points[i, 0], points[i, 1], points[i, 2]
        point_vx, point_vy, point_vz, point_valid = _world_to_voxel(
            px, py, pz, cx, cy, cz,
            hx, hy, hz, cos_yaw, sin_yaw, voxel_size, nx, ny, nz
        )
        
        ray = _bresenham_3d(sensor_vx, sensor_vy, sensor_vz,
                           point_vx, point_vy, point_vz)
        
        for j in range(ray.shape[0]):
            rvx, rvy, rvz = ray[j, 0], ray[j, 1], ray[j, 2]
            if 0 <= rvx < nx and 0 <= rvy < ny and 0 <= rvz < nz:
                idx = rvx * ny * nz + rvy * nz + rvz
                if j == ray.shape[0] - 1 and point_valid:
                    voxel_grid[idx] = VOXEL_OCCUPIED
                elif voxel_grid[idx] == VOXEL_UNKNOWN:
                    voxel_grid[idx] = VOXEL_FREE
    
    return voxel_grid


@njit(cache=True)
def _analyze_surfaces(voxel_grid: np.ndarray,
                      nx: int, ny: int, nz: int,
                      depth: int) -> np.ndarray:
    """分析各表面体素状态 -> (6, 3) [observed, occupied, total]"""
    stats = np.zeros((6, 3), dtype=np.int64)
    
    # front (+x)
    for x in range(max(0, nx-depth), nx):
        for y in range(ny):
            for z in range(nz):
                idx = x*ny*nz + y*nz + z
                stats[0, 2] += 1
                if voxel_grid[idx] != VOXEL_UNKNOWN: stats[0, 0] += 1
                if voxel_grid[idx] == VOXEL_OCCUPIED: stats[0, 1] += 1
    
    # back (-x)
    for x in range(min(nx, depth)):
        for y in range(ny):
            for z in range(nz):
                idx = x*ny*nz + y*nz + z
                stats[1, 2] += 1
                if voxel_grid[idx] != VOXEL_UNKNOWN: stats[1, 0] += 1
                if voxel_grid[idx] == VOXEL_OCCUPIED: stats[1, 1] += 1
    
    # left (+y)
    for x in range(nx):
        for y in range(max(0, ny-depth), ny):
            for z in range(nz):
                idx = x*ny*nz + y*nz + z
                stats[2, 2] += 1
                if voxel_grid[idx] != VOXEL_UNKNOWN: stats[2, 0] += 1
                if voxel_grid[idx] == VOXEL_OCCUPIED: stats[2, 1] += 1
    
    # right (-y)
    for x in range(nx):
        for y in range(min(ny, depth)):
            for z in range(nz):
                idx = x*ny*nz + y*nz + z
                stats[3, 2] += 1
                if voxel_grid[idx] != VOXEL_UNKNOWN: stats[3, 0] += 1
                if voxel_grid[idx] == VOXEL_OCCUPIED: stats[3, 1] += 1
    
    # top (+z)
    for x in range(nx):
        for y in range(ny):
            for z in range(max(0, nz-depth), nz):
                idx = x*ny*nz + y*nz + z
                stats[4, 2] += 1
                if voxel_grid[idx] != VOXEL_UNKNOWN: stats[4, 0] += 1
                if voxel_grid[idx] == VOXEL_OCCUPIED: stats[4, 1] += 1
    
    # bottom (-z)
    for x in range(nx):
        for y in range(ny):
            for z in range(min(nz, depth)):
                idx = x*ny*nz + y*nz + z
                stats[5, 2] += 1
                if voxel_grid[idx] != VOXEL_UNKNOWN: stats[5, 0] += 1
                if voxel_grid[idx] == VOXEL_OCCUPIED: stats[5, 1] += 1
    
    return stats


def _compute_surface_areas(size: np.ndarray) -> np.ndarray:
    """计算各表面面积
    
    size = [length(x), width(y), height(z)]
    
    Returns:
        areas: (6,) [front, back, left, right, top, bottom]
    """
    L, W, H = size[0], size[1], size[2]
    return np.array([
        W * H,  # front (+x): width × height
        W * H,  # back (-x): width × height
        L * H,  # left (+y): length × height
        L * H,  # right (-y): length × height
        L * W,  # top (+z): length × width
        L * W,  # bottom (-z): length × width
    ])


def _compute_area_weights(size: np.ndarray, exclude_bottom: bool = True) -> np.ndarray:
    """基于面积计算各面权重
    
    Args:
        size: bbox尺寸 [L, W, H]
        exclude_bottom: 是否排除底面（通常被地面遮挡）
    
    Returns:
        weights: (6,) 归一化权重
    """
    areas = _compute_surface_areas(size)
    
    if exclude_bottom:
        areas[5] = 0  # bottom权重设为0
    
    total = np.sum(areas)
    if total > 1e-6:
        return areas / total
    else:
        return np.ones(6) / 6


# ==================== 主函数 ====================

def compute_visibility(
    bbox: Union[np.ndarray, List],
    points: np.ndarray,
    sensor_origin: Optional[np.ndarray] = None,
    voxel_size: float = 0.2,
    surface_depth: int = 2,
    exclude_bottom: bool = True
) -> Tuple[float, Dict]:
    """计算3D标注框的可见性（基于OCC3D射线投射 + 面积加权）
    
    【方法】
    1. 射线投射：360度雷达，所有方向都可能观测到
       - 穿过的体素 → FREE（空闲）
       - 终点体素 → OCCUPIED（占用）
       - 未触及 → UNKNOWN（被遮挡）
    
    2. 面积加权：根据各面面积分配权重
       - 面积大的面（如侧面）权重高
       - 底面通常被地面挡住，权重设为0
    
    3. 主要面：面积最大的面，其可见性是关键指标
    
    Args:
        bbox: (7,) [cx, cy, cz, length, width, height, yaw]
        points: (N, 3) 点云
        sensor_origin: (3,) 传感器位置
        voxel_size: 体素大小（米）
        surface_depth: 表面分析深度（体素层数）
        exclude_bottom: 是否排除底面
    
    Returns:
        score: 面积加权可见性分数 [0, 1]
        details: 详细指标
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
    
    # 表面分析
    surface_stats = _analyze_surfaces(voxel_grid, nx, ny, nz, surface_depth)
    
    # 面积权重
    surface_areas = _compute_surface_areas(size)
    surface_weights = _compute_area_weights(size, exclude_bottom)
    
    # 各面可见性
    surface_visibility = np.zeros(6)
    surface_details = {}
    
    for i, name in enumerate(SURFACE_NAMES):
        observed = int(surface_stats[i, 0])
        occupied = int(surface_stats[i, 1])
        total = int(surface_stats[i, 2])
        
        vis = observed / max(1, total)
        surface_visibility[i] = vis
        
        surface_details[name] = {
            'visibility': float(vis),
            'observed': observed,
            'occupied': occupied,
            'total': total,
            'area': float(surface_areas[i]),
            'weight': float(surface_weights[i]),
        }
    
    # 面积加权可见性
    weighted_visibility = float(np.sum(surface_visibility * surface_weights))
    
    # 主要面（面积最大，排除底面）
    effective_areas = surface_areas.copy()
    if exclude_bottom:
        effective_areas[5] = 0
    primary_idx = int(np.argmax(effective_areas))
    primary_name = SURFACE_NAMES[primary_idx]
    primary_visibility = surface_visibility[primary_idx]
    
    details = {
        'visibility_score': weighted_visibility,
        
        # 主要面（面积最大）
        'primary_surface': {
            'name': primary_name,
            'visibility': float(primary_visibility),
            'area': float(surface_areas[primary_idx]),
            'weight': float(surface_weights[primary_idx]),
        },
        
        # 全局统计
        'observation_ratio': float((n_occupied + n_free) / max(1, n_total)),
        'n_occupied': n_occupied,
        'n_free': n_free,
        'n_unknown': n_unknown,
        'n_total': n_total,
        'voxel_shape': (nx, ny, nz),
        
        # 各面详情
        'surface_areas': {name: float(surface_areas[i]) 
                         for i, name in enumerate(SURFACE_NAMES)},
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
    print("3D BBox Visibility Calculator (Ray-based, Area-Weighted)")
    print("=" * 70)
    print(f"Numba: {HAS_NUMBA}")
    print("\n【方法说明】")
    print("- 360度雷达全方位扫描，前后左右都可能被观测到")
    print("- 基于各面面积分配权重，面积大的面（侧面）权重高")
    print("- 主要面 = 面积最大的面，其可见性是关键指标")
    
    np.random.seed(42)
    sensor = np.array([0, 0, 1.8])
    
    # 背景
    bg = np.random.randn(3000, 3) * 25
    bg[:, 2] = np.abs(bg[:, 2]) * 0.3 - 0.5
    
    # 车辆尺寸
    car_size = np.array([4.5, 2.0, 1.5])  # L, W, H
    print(f"\n车辆尺寸: L={car_size[0]}, W={car_size[1]}, H={car_size[2]}")
    print("各面面积:")
    areas = _compute_surface_areas(car_size)
    weights = _compute_area_weights(car_size, exclude_bottom=True)
    for i, name in enumerate(SURFACE_NAMES):
        print(f"  {name:8s}: area={areas[i]:.2f}m², weight={weights[i]:.3f}")
    
    # 可见车辆
    car1_center = np.array([10, 3, 0.8])
    car1_yaw = 0.1
    car1_pts = []
    for _ in range(300):
        face = np.random.randint(6)
        hs = car_size / 2
        if face == 0: p = [hs[0], np.random.uniform(-1,1)*hs[1], np.random.uniform(-1,1)*hs[2]]
        elif face == 1: p = [-hs[0], np.random.uniform(-1,1)*hs[1], np.random.uniform(-1,1)*hs[2]]
        elif face == 2: p = [np.random.uniform(-1,1)*hs[0], hs[1], np.random.uniform(-1,1)*hs[2]]
        elif face == 3: p = [np.random.uniform(-1,1)*hs[0], -hs[1], np.random.uniform(-1,1)*hs[2]]
        elif face == 4: p = [np.random.uniform(-1,1)*hs[0], np.random.uniform(-1,1)*hs[1], hs[2]]
        else: p = [np.random.uniform(-1,1)*hs[0], np.random.uniform(-1,1)*hs[1], -hs[2]]
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
    car2_yaw = -0.2
    car2_pts = np.random.randn(20, 3) * 0.15 + car2_center
    
    all_points = np.vstack([bg, car1_world, car2_pts])
    
    print("\n" + "Warming up...")
    warmup()
    
    # 测试1
    print("\n" + "-" * 60)
    print("Test 1: Visible Object (360° LiDAR)")
    print("-" * 60)
    bbox1 = np.array([*car1_center, *car_size, car1_yaw])
    t0 = time.time()
    score1, d1 = compute_visibility(bbox1, all_points, sensor, voxel_size=0.2)
    t1 = time.time()
    
    print(f"Visibility Score: {score1:.3f} ({classify_visibility(score1)})")
    print(f"Primary Surface: {d1['primary_surface']['name']} "
          f"(area={d1['primary_surface']['area']:.2f}m², "
          f"visibility={d1['primary_surface']['visibility']:.2f})")
    print(f"\nAll Surfaces (area-weighted):")
    for name in SURFACE_NAMES:
        sd = d1['surface_details'][name]
        print(f"  {name:8s}: vis={sd['visibility']:.3f}, "
              f"area={sd['area']:.2f}m², weight={sd['weight']:.3f}")
    print(f"\nTime: {(t1-t0)*1000:.2f} ms")
    
    # 测试2
    print("\n" + "-" * 60)
    print("Test 2: Occluded Object")
    print("-" * 60)
    bbox2 = np.array([*car2_center, *car_size, car2_yaw])
    t0 = time.time()
    score2, d2 = compute_visibility(bbox2, all_points, sensor, voxel_size=0.2)
    t1 = time.time()
    
    print(f"Visibility Score: {score2:.3f} ({classify_visibility(score2)})")
    print(f"Primary Surface: {d2['primary_surface']['name']} "
          f"(visibility={d2['primary_surface']['visibility']:.2f})")
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
