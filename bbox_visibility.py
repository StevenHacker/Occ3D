"""
================================================================================
3D Bounding Box Visibility Calculator (Ray-based, OCC3D Style)
================================================================================

版本: 4.0 (Performance Optimized)
基于OCC3D射线投射思想，高效计算3D标注框的可见性。

================================================================================
效率优化设计
================================================================================

【问题分析】
对于一帧点云中有N个标注框：
  - 原始方法：每个框独立做射线投射 → O(N × M × ray_length)
  - 大量重复计算：同一个点的射线被追踪N次

【优化策略】

1. 全局体素预计算（关键优化）
   ┌─────────────────────────────────────────────────────────┐
   │  预计算阶段（每帧只做一次）：                              │
   │    - 对整个场景范围建立体素网格                            │
   │    - 对每个点云点追踪一次射线                              │
   │    - 标记所有穿过的体素为FREE，终点为OCCUPIED               │
   │                                                         │
   │  查询阶段（每个框O(1)查表）：                              │
   │    - 根据bbox范围，直接查询对应区域的体素状态               │
   │    - 无需重新计算射线                                     │
   └─────────────────────────────────────────────────────────┘

2. 时间复杂度对比
   ┌────────────────┬──────────────────────────────────────┐
   │ 方法           │ 复杂度                                │
   ├────────────────┼──────────────────────────────────────┤
   │ 原始方法       │ O(N × M × L)，N=框数，M=点数，L=射线长  │
   │ 优化方法       │ O(M × L) + O(N × V)，V=框内体素数       │
   └────────────────┴──────────────────────────────────────┘
   
   对于典型场景（N=50框，M=100000点）：
   - 原始：50 × 100000 × 100 = 5亿次操作
   - 优化：100000 × 100 + 50 × 5000 = 1025万次操作（快50倍）

3. 内存-时间权衡
   - 全局体素网格需要额外内存
   - 场景范围[-50,50] × [-50,50] × [-3,5]，体素0.2m
   - 网格大小：500 × 500 × 40 = 1000万体素 ≈ 10MB（可接受）

================================================================================
使用方法
================================================================================

【单次使用】（适合少量框）
    from bbox_visibility import compute_visibility
    score, details = compute_visibility(bbox, points, sensor)

【批量使用】（推荐，适合多框场景）
    from bbox_visibility import VisibilityCalculator
    
    # 创建计算器，预计算全局体素（只做一次）
    calc = VisibilityCalculator(points, sensor_origin, scene_range, voxel_size)
    
    # 快速查询各bbox可见性
    for bbox in bboxes:
        score, details = calc.query_visibility(bbox)
    
    # 或批量查询
    scores, details_list = calc.query_visibility_batch(bboxes)

================================================================================
"""

import numpy as np
from typing import Tuple, Dict, List, Optional, Union

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


# ==================== 常量 ====================
VOXEL_UNKNOWN = 0
VOXEL_FREE = 1
VOXEL_OCCUPIED = 2

SURFACE_NAMES = ['front', 'back', 'left', 'right', 'top', 'bottom']


# ==================== 3D Bresenham ====================

@njit(cache=True)
def _bresenham_3d_inplace(x0: int, y0: int, z0: int,
                          x1: int, y1: int, z1: int,
                          result: np.ndarray) -> int:
    """3D Bresenham，结果写入预分配数组，返回实际长度"""
    count = 0
    max_steps = result.shape[0]
    
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
    
    return count


# ==================== 全局体素预计算 ====================

@njit(cache=True)
def _precompute_global_voxels(
    points: np.ndarray,
    sensor_x: float, sensor_y: float, sensor_z: float,
    range_min_x: float, range_min_y: float, range_min_z: float,
    voxel_size: float,
    nx: int, ny: int, nz: int
) -> np.ndarray:
    """预计算全局体素状态（每帧只需调用一次）
    
    对每个点云点，追踪一次射线，标记：
    - 射线穿过的体素 → FREE
    - 射线终点体素 → OCCUPIED
    """
    n_total = nx * ny * nz
    voxel_grid = np.zeros(n_total, dtype=np.uint8)
    
    # 传感器体素坐标
    sensor_vx = int((sensor_x - range_min_x) / voxel_size)
    sensor_vy = int((sensor_y - range_min_y) / voxel_size)
    sensor_vz = int((sensor_z - range_min_z) / voxel_size)
    
    # 预分配射线缓冲区
    ray_buffer = np.empty((5000, 3), dtype=np.int32)
    
    n_points = points.shape[0]
    for i in range(n_points):
        px, py, pz = points[i, 0], points[i, 1], points[i, 2]
        
        # 点的体素坐标
        point_vx = int((px - range_min_x) / voxel_size)
        point_vy = int((py - range_min_y) / voxel_size)
        point_vz = int((pz - range_min_z) / voxel_size)
        
        # 检查点是否在范围内
        point_valid = (0 <= point_vx < nx and 
                       0 <= point_vy < ny and 
                       0 <= point_vz < nz)
        
        # 射线追踪
        ray_len = _bresenham_3d_inplace(
            sensor_vx, sensor_vy, sensor_vz,
            point_vx, point_vy, point_vz,
            ray_buffer
        )
        
        # 标记体素
        for j in range(ray_len):
            vx = ray_buffer[j, 0]
            vy = ray_buffer[j, 1]
            vz = ray_buffer[j, 2]
            
            if 0 <= vx < nx and 0 <= vy < ny and 0 <= vz < nz:
                idx = vx * ny * nz + vy * nz + vz
                
                if j == ray_len - 1 and point_valid:
                    # 终点：OCCUPIED（优先级最高）
                    voxel_grid[idx] = VOXEL_OCCUPIED
                elif voxel_grid[idx] == VOXEL_UNKNOWN:
                    # 路径：FREE（不覆盖OCCUPIED）
                    voxel_grid[idx] = VOXEL_FREE
    
    return voxel_grid


@njit(cache=True)
def _query_bbox_voxels(
    global_grid: np.ndarray,
    global_nx: int, global_ny: int, global_nz: int,
    range_min_x: float, range_min_y: float, range_min_z: float,
    voxel_size: float,
    bbox_center_x: float, bbox_center_y: float, bbox_center_z: float,
    bbox_size_x: float, bbox_size_y: float, bbox_size_z: float,
    bbox_yaw: float,
    surface_depth: int
) -> Tuple[np.ndarray, int, int, int, int, int, int]:
    """从全局体素网格中查询bbox区域的体素状态
    
    Returns:
        local_grid: bbox局部体素网格
        nx, ny, nz: 局部网格尺寸
        n_occupied, n_free, n_unknown: 各状态数量
    """
    # bbox局部网格尺寸
    local_nx = max(1, int(np.ceil(bbox_size_x / voxel_size)))
    local_ny = max(1, int(np.ceil(bbox_size_y / voxel_size)))
    local_nz = max(1, int(np.ceil(bbox_size_z / voxel_size)))
    local_total = local_nx * local_ny * local_nz
    
    local_grid = np.zeros(local_total, dtype=np.uint8)
    
    cos_yaw = np.cos(bbox_yaw)
    sin_yaw = np.sin(bbox_yaw)
    
    half_x = bbox_size_x / 2
    half_y = bbox_size_y / 2
    half_z = bbox_size_z / 2
    
    n_occupied = 0
    n_free = 0
    n_unknown = 0
    
    # 遍历局部体素
    for lx in range(local_nx):
        for ly in range(local_ny):
            for lz in range(local_nz):
                # 局部坐标（bbox坐标系）
                local_px = (lx + 0.5) * voxel_size - half_x
                local_py = (ly + 0.5) * voxel_size - half_y
                local_pz = (lz + 0.5) * voxel_size - half_z
                
                # 转换到全局坐标
                global_px = cos_yaw * local_px - sin_yaw * local_py + bbox_center_x
                global_py = sin_yaw * local_px + cos_yaw * local_py + bbox_center_y
                global_pz = local_pz + bbox_center_z
                
                # 全局体素索引
                gx = int((global_px - range_min_x) / voxel_size)
                gy = int((global_py - range_min_y) / voxel_size)
                gz = int((global_pz - range_min_z) / voxel_size)
                
                # 查询全局网格
                if 0 <= gx < global_nx and 0 <= gy < global_ny and 0 <= gz < global_nz:
                    global_idx = gx * global_ny * global_nz + gy * global_nz + gz
                    state = global_grid[global_idx]
                else:
                    state = VOXEL_UNKNOWN
                
                # 存储到局部网格
                local_idx = lx * local_ny * local_nz + ly * local_nz + lz
                local_grid[local_idx] = state
                
                if state == VOXEL_OCCUPIED:
                    n_occupied += 1
                elif state == VOXEL_FREE:
                    n_free += 1
                else:
                    n_unknown += 1
    
    return local_grid, local_nx, local_ny, local_nz, n_occupied, n_free, n_unknown


@njit(cache=True)
def _analyze_local_surfaces(
    local_grid: np.ndarray,
    nx: int, ny: int, nz: int,
    depth: int
) -> np.ndarray:
    """分析局部网格的各表面状态"""
    stats = np.zeros((6, 3), dtype=np.int64)  # [observed, occupied, total]
    
    # front (+x)
    for x in range(max(0, nx-depth), nx):
        for y in range(ny):
            for z in range(nz):
                idx = x*ny*nz + y*nz + z
                stats[0, 2] += 1
                if local_grid[idx] != VOXEL_UNKNOWN: stats[0, 0] += 1
                if local_grid[idx] == VOXEL_OCCUPIED: stats[0, 1] += 1
    
    # back (-x)
    for x in range(min(nx, depth)):
        for y in range(ny):
            for z in range(nz):
                idx = x*ny*nz + y*nz + z
                stats[1, 2] += 1
                if local_grid[idx] != VOXEL_UNKNOWN: stats[1, 0] += 1
                if local_grid[idx] == VOXEL_OCCUPIED: stats[1, 1] += 1
    
    # left (+y)
    for x in range(nx):
        for y in range(max(0, ny-depth), ny):
            for z in range(nz):
                idx = x*ny*nz + y*nz + z
                stats[2, 2] += 1
                if local_grid[idx] != VOXEL_UNKNOWN: stats[2, 0] += 1
                if local_grid[idx] == VOXEL_OCCUPIED: stats[2, 1] += 1
    
    # right (-y)
    for x in range(nx):
        for y in range(min(ny, depth)):
            for z in range(nz):
                idx = x*ny*nz + y*nz + z
                stats[3, 2] += 1
                if local_grid[idx] != VOXEL_UNKNOWN: stats[3, 0] += 1
                if local_grid[idx] == VOXEL_OCCUPIED: stats[3, 1] += 1
    
    # top (+z)
    for x in range(nx):
        for y in range(ny):
            for z in range(max(0, nz-depth), nz):
                idx = x*ny*nz + y*nz + z
                stats[4, 2] += 1
                if local_grid[idx] != VOXEL_UNKNOWN: stats[4, 0] += 1
                if local_grid[idx] == VOXEL_OCCUPIED: stats[4, 1] += 1
    
    # bottom (-z)
    for x in range(nx):
        for y in range(ny):
            for z in range(min(nz, depth)):
                idx = x*ny*nz + y*nz + z
                stats[5, 2] += 1
                if local_grid[idx] != VOXEL_UNKNOWN: stats[5, 0] += 1
                if local_grid[idx] == VOXEL_OCCUPIED: stats[5, 1] += 1
    
    return stats


# ==================== 可见性计算器类 ====================

class VisibilityCalculator:
    """高效的可见性计算器（推荐用于多框场景）
    
    【使用方法】
    calc = VisibilityCalculator(points, sensor_origin, scene_range, voxel_size)
    
    # 单个查询
    score, details = calc.query_visibility(bbox)
    
    # 批量查询
    scores, details_list = calc.query_visibility_batch(bboxes)
    
    【效率对比】
    - 原始方法：每个框重新计算射线 → O(N × M)
    - 本方法：预计算一次，快速查询 → O(M) + O(N × V)
    """
    
    def __init__(self,
                 points: np.ndarray,
                 sensor_origin: np.ndarray,
                 scene_range: Optional[List] = None,
                 voxel_size: float = 0.2):
        """初始化计算器，预计算全局体素
        
        Args:
            points: (N, 3) 点云
            sensor_origin: (3,) 传感器位置
            scene_range: [min_x, min_y, min_z, max_x, max_y, max_z]
                        默认根据点云自动推断
            voxel_size: 体素大小（米）
        """
        self.points = np.asarray(points[:, :3], dtype=np.float64)
        self.sensor_origin = np.asarray(sensor_origin, dtype=np.float64)
        self.voxel_size = voxel_size
        
        # 自动推断场景范围（限制最大范围避免内存爆炸）
        if scene_range is None:
            # 使用合理的默认范围（适合自动驾驶场景）
            # 典型范围：[-50, 50] x [-50, 50] x [-3, 5]
            max_range = 60.0  # 最大范围限制
            max_z_range = 10.0
            
            pts_min = np.min(self.points, axis=0)
            pts_max = np.max(self.points, axis=0)
            
            # 限制范围
            x_min = max(-max_range, min(pts_min[0], sensor_origin[0]) - 2)
            y_min = max(-max_range, min(pts_min[1], sensor_origin[1]) - 2)
            z_min = max(-max_z_range, min(pts_min[2], sensor_origin[2]) - 1)
            x_max = min(max_range, max(pts_max[0], sensor_origin[0]) + 2)
            y_max = min(max_range, max(pts_max[1], sensor_origin[1]) + 2)
            z_max = min(max_z_range, max(pts_max[2], sensor_origin[2]) + 1)
            
            scene_range = [x_min, y_min, z_min, x_max, y_max, z_max]
        
        self.range_min = np.array(scene_range[:3], dtype=np.float64)
        self.range_max = np.array(scene_range[3:], dtype=np.float64)
        
        # 计算网格尺寸
        self.nx = int(np.ceil((self.range_max[0] - self.range_min[0]) / voxel_size))
        self.ny = int(np.ceil((self.range_max[1] - self.range_min[1]) / voxel_size))
        self.nz = int(np.ceil((self.range_max[2] - self.range_min[2]) / voxel_size))
        
        # 预计算全局体素（关键步骤）
        self.global_grid = _precompute_global_voxels(
            self.points,
            self.sensor_origin[0], self.sensor_origin[1], self.sensor_origin[2],
            self.range_min[0], self.range_min[1], self.range_min[2],
            self.voxel_size,
            self.nx, self.ny, self.nz
        )
        
        # 统计
        self.n_total_voxels = self.nx * self.ny * self.nz
        self.n_occupied = int(np.sum(self.global_grid == VOXEL_OCCUPIED))
        self.n_free = int(np.sum(self.global_grid == VOXEL_FREE))
        self.n_unknown = self.n_total_voxels - self.n_occupied - self.n_free
    
    def query_visibility(self,
                        bbox: Union[np.ndarray, List],
                        surface_depth: int = 2,
                        exclude_bottom: bool = True) -> Tuple[float, Dict]:
        """查询单个bbox的可见性（快速，不重新计算射线）
        
        Args:
            bbox: (7,) [cx, cy, cz, length, width, height, yaw]
            surface_depth: 表面分析深度
            exclude_bottom: 是否排除底面
        
        Returns:
            score: 可见性分数
            details: 详细指标
        """
        bbox = np.asarray(bbox, dtype=np.float64)
        center = bbox[:3]
        size = bbox[3:6]
        yaw = bbox[6]
        
        # 从全局网格查询bbox区域
        local_grid, lnx, lny, lnz, n_occ, n_free, n_unk = _query_bbox_voxels(
            self.global_grid,
            self.nx, self.ny, self.nz,
            self.range_min[0], self.range_min[1], self.range_min[2],
            self.voxel_size,
            center[0], center[1], center[2],
            size[0], size[1], size[2],
            yaw, surface_depth
        )
        
        n_total = lnx * lny * lnz
        
        # 表面分析
        surface_stats = _analyze_local_surfaces(local_grid, lnx, lny, lnz, surface_depth)
        
        # 面积权重
        L, W, H = size[0], size[1], size[2]
        areas = np.array([W*H, W*H, L*H, L*H, L*W, L*W])
        weights = areas.copy()
        if exclude_bottom:
            weights[5] = 0
        weights = weights / np.sum(weights)
        
        # 各面可见性
        surface_vis = np.zeros(6)
        surface_details = {}
        for i, name in enumerate(SURFACE_NAMES):
            obs = int(surface_stats[i, 0])
            occ = int(surface_stats[i, 1])
            tot = int(surface_stats[i, 2])
            vis = obs / max(1, tot)
            surface_vis[i] = vis
            surface_details[name] = {
                'visibility': float(vis),
                'observed': obs,
                'occupied': occ,
                'total': tot,
                'area': float(areas[i]),
                'weight': float(weights[i]),
            }
        
        # 加权分数
        score = float(np.sum(surface_vis * weights))
        
        # 主要面
        effective_areas = areas.copy()
        if exclude_bottom:
            effective_areas[5] = 0
        primary_idx = int(np.argmax(effective_areas))
        
        details = {
            'visibility_score': score,
            'primary_surface': {
                'name': SURFACE_NAMES[primary_idx],
                'visibility': float(surface_vis[primary_idx]),
                'area': float(areas[primary_idx]),
            },
            'observation_ratio': float((n_occ + n_free) / max(1, n_total)),
            'n_occupied': n_occ,
            'n_free': n_free,
            'n_unknown': n_unk,
            'n_total': n_total,
            'surface_details': surface_details,
        }
        
        return score, details
    
    def query_visibility_batch(self,
                               bboxes: np.ndarray,
                               surface_depth: int = 2,
                               exclude_bottom: bool = True) -> Tuple[np.ndarray, List[Dict]]:
        """批量查询多个bbox"""
        bboxes = np.asarray(bboxes, dtype=np.float64)
        n = bboxes.shape[0]
        scores = np.zeros(n)
        details_list = []
        for i in range(n):
            s, d = self.query_visibility(bboxes[i], surface_depth, exclude_bottom)
            scores[i] = s
            details_list.append(d)
        return scores, details_list
    
    def get_stats(self) -> Dict:
        """获取全局体素统计"""
        return {
            'grid_shape': (self.nx, self.ny, self.nz),
            'n_total_voxels': self.n_total_voxels,
            'n_occupied': self.n_occupied,
            'n_free': self.n_free,
            'n_unknown': self.n_unknown,
            'memory_mb': self.global_grid.nbytes / 1024 / 1024,
        }


# ==================== 便捷函数（兼容旧接口） ====================

def compute_visibility(
    bbox: Union[np.ndarray, List],
    points: np.ndarray,
    sensor_origin: Optional[np.ndarray] = None,
    voxel_size: float = 0.2,
    surface_depth: int = 2,
    exclude_bottom: bool = True
) -> Tuple[float, Dict]:
    """计算单个bbox可见性（适合少量框，每次调用都会重新计算）
    
    注意：如果有多个框，建议使用 VisibilityCalculator 类以获得更好性能。
    """
    if sensor_origin is None:
        sensor_origin = np.array([0.0, 0.0, 0.0])
    
    calc = VisibilityCalculator(points, sensor_origin, voxel_size=voxel_size)
    return calc.query_visibility(bbox, surface_depth, exclude_bottom)


def compute_visibility_batch(
    bboxes: np.ndarray,
    points: np.ndarray,
    sensor_origin: Optional[np.ndarray] = None,
    voxel_size: float = 0.2
) -> Tuple[np.ndarray, List[Dict]]:
    """批量计算（内部使用优化版本）"""
    if sensor_origin is None:
        sensor_origin = np.array([0.0, 0.0, 0.0])
    
    calc = VisibilityCalculator(points, sensor_origin, voxel_size=voxel_size)
    return calc.query_visibility_batch(bboxes)


def classify_visibility(score: float) -> str:
    """可见性等级"""
    if score >= 0.7: return "FULLY_VISIBLE"
    elif score >= 0.4: return "MOSTLY_VISIBLE"
    elif score >= 0.2: return "PARTIALLY_VISIBLE"
    elif score > 0.05: return "MOSTLY_OCCLUDED"
    else: return "FULLY_OCCLUDED"


def warmup():
    """预热JIT"""
    if not HAS_NUMBA:
        return
    pts = np.random.randn(100, 3).astype(np.float64) * 10
    sensor = np.array([0, 0, 1.5], dtype=np.float64)
    calc = VisibilityCalculator(pts, sensor, voxel_size=0.5)
    bbox = np.array([5, 5, 1, 3, 2, 1.5, 0], dtype=np.float64)
    calc.query_visibility(bbox)


# ==================== 测试与性能评估 ====================

if __name__ == '__main__':
    import time
    
    print("=" * 70)
    print("3D BBox Visibility Calculator - Performance Optimized (v4.0)")
    print("=" * 70)
    print(f"Numba: {HAS_NUMBA}")
    
    np.random.seed(42)
    
    # 模拟真实场景
    print("\n【场景设置】")
    n_points = 100000  # 10万点云
    n_boxes = 50       # 50个框
    
    print(f"点云数量: {n_points:,}")
    print(f"标注框数量: {n_boxes}")
    
    # 生成点云（限制在合理范围内，模拟真实LiDAR）
    points = np.random.randn(n_points, 3)
    points[:, 0] = points[:, 0] * 25  # x: [-50, 50]
    points[:, 1] = points[:, 1] * 25  # y: [-50, 50]
    points[:, 2] = np.abs(points[:, 2]) * 1.5 - 0.5  # z: [-0.5, 4]
    
    # 生成框
    bboxes = np.zeros((n_boxes, 7))
    bboxes[:, 0] = np.random.uniform(-40, 40, n_boxes)  # x
    bboxes[:, 1] = np.random.uniform(-40, 40, n_boxes)  # y
    bboxes[:, 2] = np.random.uniform(0, 2, n_boxes)     # z
    bboxes[:, 3] = np.random.uniform(3, 5, n_boxes)     # length
    bboxes[:, 4] = np.random.uniform(1.5, 2.5, n_boxes) # width
    bboxes[:, 5] = np.random.uniform(1.2, 2, n_boxes)   # height
    bboxes[:, 6] = np.random.uniform(-np.pi, np.pi, n_boxes)  # yaw
    
    sensor = np.array([0, 0, 1.8])
    
    # 预热
    print("\n【预热JIT编译】")
    warmup()
    print("完成")
    
    # ========== 方法1：原始方法（每框独立计算）==========
    print("\n" + "-" * 60)
    print("方法1：原始方法（每框独立计算）")
    print("-" * 60)
    
    t0 = time.time()
    scores_v1 = []
    for bbox in bboxes[:10]:  # 只测10个，否则太慢
        calc = VisibilityCalculator(points, sensor, voxel_size=0.2)
        s, _ = calc.query_visibility(bbox)
        scores_v1.append(s)
    t1 = time.time()
    
    time_per_box_v1 = (t1 - t0) / 10 * 1000
    estimated_total_v1 = time_per_box_v1 * n_boxes
    print(f"每框耗时: {time_per_box_v1:.1f} ms")
    print(f"预估{n_boxes}框总耗时: {estimated_total_v1:.1f} ms")
    
    # ========== 方法2：优化方法（预计算+查询）==========
    print("\n" + "-" * 60)
    print("方法2：优化方法（预计算一次 + 快速查询）")
    print("-" * 60)
    
    # 预计算
    t0 = time.time()
    calc = VisibilityCalculator(points, sensor, voxel_size=0.2)
    t_precompute = time.time() - t0
    
    print(f"预计算耗时: {t_precompute*1000:.1f} ms")
    stats = calc.get_stats()
    print(f"全局体素网格: {stats['grid_shape']}")
    print(f"体素总数: {stats['n_total_voxels']:,}")
    print(f"内存占用: {stats['memory_mb']:.2f} MB")
    print(f"  OCCUPIED: {stats['n_occupied']:,}")
    print(f"  FREE: {stats['n_free']:,}")
    print(f"  UNKNOWN: {stats['n_unknown']:,}")
    
    # 批量查询
    t0 = time.time()
    scores_v2, details = calc.query_visibility_batch(bboxes)
    t_query = time.time() - t0
    
    print(f"\n查询{n_boxes}框耗时: {t_query*1000:.1f} ms")
    print(f"每框查询耗时: {t_query/n_boxes*1000:.3f} ms")
    
    total_v2 = (t_precompute + t_query) * 1000
    print(f"总耗时（预计算+查询）: {total_v2:.1f} ms")
    
    # ========== 性能对比 ==========
    print("\n" + "=" * 60)
    print("【性能对比】")
    print("=" * 60)
    print(f"原始方法预估: {estimated_total_v1:.1f} ms")
    print(f"优化方法实际: {total_v2:.1f} ms")
    print(f"加速比: {estimated_total_v1/total_v2:.1f}x")
    
    # 结果验证
    print("\n【结果样例】")
    for i in range(min(5, n_boxes)):
        print(f"Box {i}: score={scores_v2[i]:.3f} ({classify_visibility(scores_v2[i])})")
    
    print("\n" + "=" * 70)
    print("【使用建议】")
    print("=" * 70)
    print("""
对于单帧多框场景，推荐使用 VisibilityCalculator：

    from bbox_visibility import VisibilityCalculator
    
    # 创建计算器（预计算全局体素，只做一次）
    calc = VisibilityCalculator(points, sensor_origin, voxel_size=0.2)
    
    # 快速查询各bbox
    scores, details = calc.query_visibility_batch(bboxes)
    
    # 或逐个查询
    for bbox in bboxes:
        score, detail = calc.query_visibility(bbox)
""")
