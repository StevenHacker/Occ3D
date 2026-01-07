"""
3D Bounding Box Visibility Calculator
=====================================

一个完全独立的、即插即用的3D标注框可见性量化工具。

依赖：
    - numpy
    - numba (用于加速，如不需要可移除@njit装饰器)

输入：
    - bbox: 3D标注框 [center_x, center_y, center_z, length, width, height, yaw]
    - points: 点云 (N, 3)
    - sensor_origin: 传感器位置 (3,)，默认 [0, 0, 0]

输出：
    - visibility_score: 综合可见性分数 [0, 1]
    - details: 详细指标字典

使用示例：
    >>> from bbox_visibility import compute_visibility
    >>> bbox = [10, 5, 1, 4.5, 2, 1.5, 0.5]  # [cx, cy, cz, l, w, h, yaw]
    >>> points = np.random.randn(10000, 3)
    >>> score, details = compute_visibility(bbox, points)
    >>> print(f"Visibility: {score:.2f}")
"""

import numpy as np
from typing import Tuple, Dict, List, Optional, Union

# 尝试导入numba，如果没有则使用纯Python
try:
    from numba import njit
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    # 定义一个空装饰器
    def njit(*args, **kwargs):
        def decorator(func):
            return func
        if len(args) == 1 and callable(args[0]):
            return args[0]
        return decorator


# ==================== 核心计算函数 ====================

@njit(cache=True)
def _rotate_points_z(points: np.ndarray, angle: float) -> np.ndarray:
    """绕Z轴旋转点云"""
    cos_a = np.cos(angle)
    sin_a = np.sin(angle)
    n = points.shape[0]
    rotated = np.empty((n, 3), dtype=np.float64)
    for i in range(n):
        x, y, z = points[i, 0], points[i, 1], points[i, 2]
        rotated[i, 0] = cos_a * x - sin_a * y
        rotated[i, 1] = sin_a * x + cos_a * y
        rotated[i, 2] = z
    return rotated


@njit(cache=True)
def _points_in_box(points: np.ndarray, 
                   center: np.ndarray, 
                   size: np.ndarray, 
                   yaw: float) -> np.ndarray:
    """判断点是否在旋转包围盒内
    
    Args:
        points: (N, 3) 点云
        center: (3,) 中心点 [x, y, z]
        size: (3,) 尺寸 [length, width, height]
        yaw: 绕Z轴旋转角（弧度）
    
    Returns:
        mask: (N,) 布尔数组
    """
    n = points.shape[0]
    mask = np.zeros(n, dtype=np.bool_)
    
    cos_yaw = np.cos(-yaw)
    sin_yaw = np.sin(-yaw)
    half_l = size[0] / 2
    half_w = size[1] / 2
    half_h = size[2] / 2
    
    for i in range(n):
        # 平移到中心
        dx = points[i, 0] - center[0]
        dy = points[i, 1] - center[1]
        dz = points[i, 2] - center[2]
        
        # 逆旋转
        local_x = cos_yaw * dx - sin_yaw * dy
        local_y = sin_yaw * dx + cos_yaw * dy
        
        # 判断是否在框内
        if (abs(local_x) <= half_l and 
            abs(local_y) <= half_w and 
            abs(dz) <= half_h):
            mask[i] = True
    
    return mask


@njit(cache=True)
def _compute_voxel_occupancy(points_local: np.ndarray,
                              size: np.ndarray,
                              voxel_size: float) -> float:
    """计算体素占用率
    
    将bbox划分为体素网格，计算被点云占用的体素比例。
    """
    n_points = points_local.shape[0]
    if n_points == 0:
        return 0.0
    
    # 计算网格尺寸
    nx = max(1, int(np.ceil(size[0] / voxel_size)))
    ny = max(1, int(np.ceil(size[1] / voxel_size)))
    nz = max(1, int(np.ceil(size[2] / voxel_size)))
    total_voxels = nx * ny * nz
    
    # 使用一维数组模拟3D网格（numba兼容）
    voxel_occupied = np.zeros(total_voxels, dtype=np.bool_)
    
    half_l = size[0] / 2
    half_w = size[1] / 2
    half_h = size[2] / 2
    
    for i in range(n_points):
        # 映射到 [0, size] 范围
        x = points_local[i, 0] + half_l
        y = points_local[i, 1] + half_w
        z = points_local[i, 2] + half_h
        
        # 计算体素索引
        vx = min(nx - 1, max(0, int(x / voxel_size)))
        vy = min(ny - 1, max(0, int(y / voxel_size)))
        vz = min(nz - 1, max(0, int(z / voxel_size)))
        
        idx = vx * ny * nz + vy * nz + vz
        voxel_occupied[idx] = True
    
    # 统计占用数量
    occupied_count = 0
    for i in range(total_voxels):
        if voxel_occupied[i]:
            occupied_count += 1
    
    return occupied_count / total_voxels


@njit(cache=True)
def _compute_surface_points(points_local: np.ndarray,
                             size: np.ndarray,
                             threshold_ratio: float = 0.15) -> np.ndarray:
    """统计各表面附近的点数
    
    Returns:
        counts: (6,) [+x, -x, +y, -y, +z, -z] 各面点数
    """
    counts = np.zeros(6, dtype=np.int64)
    n = points_local.shape[0]
    
    if n == 0:
        return counts
    
    half_l = size[0] / 2
    half_w = size[1] / 2
    half_h = size[2] / 2
    
    thresh_l = size[0] * threshold_ratio
    thresh_w = size[1] * threshold_ratio
    thresh_h = size[2] * threshold_ratio
    
    for i in range(n):
        x, y, z = points_local[i, 0], points_local[i, 1], points_local[i, 2]
        
        if x > half_l - thresh_l:      # +x (front)
            counts[0] += 1
        if x < -half_l + thresh_l:     # -x (back)
            counts[1] += 1
        if y > half_w - thresh_w:      # +y (left)
            counts[2] += 1
        if y < -half_w + thresh_w:     # -y (right)
            counts[3] += 1
        if z > half_h - thresh_h:      # +z (top)
            counts[4] += 1
        if z < -half_h + thresh_h:     # -z (bottom)
            counts[5] += 1
    
    return counts


@njit(cache=True)
def _compute_surface_visibility(surface_counts: np.ndarray,
                                 size: np.ndarray,
                                 sensor_local: np.ndarray,
                                 total_points: int) -> Tuple[np.ndarray, float]:
    """计算各表面可见性
    
    Returns:
        surface_vis: (6,) 各表面可见性分数
        mean_vis: 平均可见性
    """
    surface_vis = np.zeros(6, dtype=np.float64)
    
    if total_points == 0:
        return surface_vis, 0.0
    
    # 表面面积
    areas = np.array([
        size[1] * size[2],  # +x/-x
        size[1] * size[2],
        size[0] * size[2],  # +y/-y
        size[0] * size[2],
        size[0] * size[1],  # +z/-z
        size[0] * size[1],
    ], dtype=np.float64)
    
    total_area = np.sum(areas)
    
    # 判断哪些面理论上可见（基于传感器位置）
    visible_mask = np.array([
        sensor_local[0] > 0,   # +x可见当传感器在+x侧
        sensor_local[0] < 0,   # -x
        sensor_local[1] > 0,   # +y
        sensor_local[1] < 0,   # -y
        sensor_local[2] > 0,   # +z
        sensor_local[2] < 0,   # -z
    ])
    
    visible_count = 0
    visible_sum = 0.0
    
    for i in range(6):
        if visible_mask[i]:
            # 期望点数（按面积比例）
            expected = max(1.0, total_points * areas[i] / total_area * 0.4)
            surface_vis[i] = min(1.0, surface_counts[i] / expected)
            visible_count += 1
            visible_sum += surface_vis[i]
        else:
            # 不可见面有点说明可能有反射或特殊情况
            if surface_counts[i] > 0:
                surface_vis[i] = 0.2
    
    mean_vis = visible_sum / max(1, visible_count)
    return surface_vis, mean_vis


@njit(cache=True)
def _compute_depth_distribution(points_local: np.ndarray,
                                 sensor_local: np.ndarray,
                                 n_bins: int = 8) -> float:
    """计算深度分布连续性
    
    评估点云在深度方向的覆盖连续性。
    """
    n = points_local.shape[0]
    if n < 3:
        return 0.0
    
    # 计算各点到传感器的距离
    depths = np.empty(n, dtype=np.float64)
    for i in range(n):
        dx = points_local[i, 0] - sensor_local[0]
        dy = points_local[i, 1] - sensor_local[1]
        dz = points_local[i, 2] - sensor_local[2]
        depths[i] = np.sqrt(dx*dx + dy*dy + dz*dz)
    
    min_d = depths[0]
    max_d = depths[0]
    for i in range(1, n):
        if depths[i] < min_d:
            min_d = depths[i]
        if depths[i] > max_d:
            max_d = depths[i]
    
    if max_d - min_d < 0.1:
        return 1.0
    
    bin_size = (max_d - min_d) / n_bins
    bin_counts = np.zeros(n_bins, dtype=np.int64)
    
    for i in range(n):
        idx = min(n_bins - 1, int((depths[i] - min_d) / bin_size))
        bin_counts[idx] += 1
    
    non_empty = 0
    for i in range(n_bins):
        if bin_counts[i] > 0:
            non_empty += 1
    
    return non_empty / n_bins


# ==================== 主函数 ====================

def compute_visibility(
    bbox: Union[np.ndarray, List],
    points: np.ndarray,
    sensor_origin: Optional[np.ndarray] = None,
    voxel_size: float = 0.1,
    weights: Optional[Dict[str, float]] = None
) -> Tuple[float, Dict]:
    """计算3D标注框的可见性
    
    Args:
        bbox: (7,) 标注框 [cx, cy, cz, length, width, height, yaw]
              - cx, cy, cz: 中心点坐标
              - length, width, height: 长宽高
              - yaw: 绕Z轴旋转角度（弧度）
        points: (N, 3) 或 (N, 3+) 点云数据
        sensor_origin: (3,) 传感器位置，默认 [0, 0, 0]
        voxel_size: 体素大小，用于体素占用率计算
        weights: 各指标权重字典，默认均匀权重
    
    Returns:
        score: 综合可见性分数 [0, 1]
        details: 详细指标字典，包含:
            - visibility_score: 综合分数
            - point_count: 框内点数
            - point_density: 点密度 (点/m³)
            - point_density_score: 密度分数
            - voxel_occupancy: 体素占用率
            - surface_visibility: 表面可见性均值
            - surface_details: 各表面可见性 {front, back, left, right, top, bottom}
            - depth_continuity: 深度连续性
    
    Example:
        >>> bbox = np.array([10, 5, 1, 4.5, 2, 1.5, 0.5])
        >>> points = np.random.randn(10000, 3) * 20
        >>> score, details = compute_visibility(bbox, points)
        >>> print(f"Score: {score:.2f}, Points: {details['point_count']}")
    """
    # 默认值处理
    if sensor_origin is None:
        sensor_origin = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    
    if weights is None:
        weights = {
            'point_density': 0.25,
            'voxel_occupancy': 0.25,
            'surface_visibility': 0.25,
            'depth_continuity': 0.25,
        }
    
    # 类型转换
    bbox = np.asarray(bbox, dtype=np.float64)
    points = np.asarray(points[:, :3], dtype=np.float64).copy()
    sensor_origin = np.asarray(sensor_origin, dtype=np.float64)
    
    # 解析bbox
    center = bbox[:3]
    size = bbox[3:6]
    yaw = bbox[6]
    
    # 1. 筛选框内点云
    in_box = _points_in_box(points, center, size, yaw)
    points_in_box = points[in_box]
    n_points = points_in_box.shape[0]
    
    # 2. 转换到局部坐标系
    if n_points > 0:
        points_centered = points_in_box - center
        points_local = _rotate_points_z(points_centered, -yaw)
        sensor_centered = sensor_origin - center
        sensor_local = _rotate_points_z(sensor_centered.reshape(1, 3), -yaw)[0]
    else:
        points_local = np.empty((0, 3), dtype=np.float64)
        sensor_local = np.zeros(3, dtype=np.float64)
    
    # 3. 计算各项指标
    details = {}
    
    # 3.1 点密度
    volume = size[0] * size[1] * size[2]
    point_density = n_points / max(volume, 1e-6)
    # 归一化（经验值：理想密度约50-100点/m³）
    density_score = min(1.0, point_density / 80.0)
    
    details['point_count'] = int(n_points)
    details['point_density'] = float(point_density)
    details['point_density_score'] = float(density_score)
    
    # 3.2 体素占用率
    if n_points >= 5:
        voxel_occ = _compute_voxel_occupancy(points_local, size, voxel_size)
    else:
        voxel_occ = 0.0
    details['voxel_occupancy'] = float(voxel_occ)
    
    # 3.3 表面可见性
    if n_points >= 5:
        surface_counts = _compute_surface_points(points_local, size)
        surface_vis, surface_mean = _compute_surface_visibility(
            surface_counts, size, sensor_local, n_points
        )
    else:
        surface_vis = np.zeros(6)
        surface_mean = 0.0
    
    details['surface_visibility'] = float(surface_mean)
    details['surface_details'] = {
        'front': float(surface_vis[0]),   # +x
        'back': float(surface_vis[1]),    # -x
        'left': float(surface_vis[2]),    # +y
        'right': float(surface_vis[3]),   # -y
        'top': float(surface_vis[4]),     # +z
        'bottom': float(surface_vis[5]),  # -z
    }
    
    # 3.4 深度连续性
    if n_points >= 5:
        depth_cont = _compute_depth_distribution(points_local, sensor_local)
    else:
        depth_cont = 0.0
    details['depth_continuity'] = float(depth_cont)
    
    # 4. 综合分数
    total_weight = sum(weights.values())
    score = (
        weights.get('point_density', 0.25) / total_weight * density_score +
        weights.get('voxel_occupancy', 0.25) / total_weight * voxel_occ +
        weights.get('surface_visibility', 0.25) / total_weight * surface_mean +
        weights.get('depth_continuity', 0.25) / total_weight * depth_cont
    )
    
    details['visibility_score'] = float(score)
    
    return score, details


def compute_visibility_batch(
    bboxes: np.ndarray,
    points: np.ndarray,
    sensor_origin: Optional[np.ndarray] = None,
    voxel_size: float = 0.1,
    weights: Optional[Dict[str, float]] = None
) -> Tuple[np.ndarray, List[Dict]]:
    """批量计算多个标注框的可见性
    
    Args:
        bboxes: (M, 7) 多个标注框
        points: (N, 3) 点云
        sensor_origin: (3,) 传感器位置
        voxel_size: 体素大小
        weights: 指标权重
    
    Returns:
        scores: (M,) 可见性分数数组
        details_list: 详细指标列表
    """
    bboxes = np.asarray(bboxes, dtype=np.float64)
    n_boxes = bboxes.shape[0]
    
    scores = np.zeros(n_boxes)
    details_list = []
    
    for i in range(n_boxes):
        s, d = compute_visibility(
            bboxes[i], points, sensor_origin, voxel_size, weights
        )
        scores[i] = s
        details_list.append(d)
    
    return scores, details_list


def classify_visibility(score: float) -> str:
    """将分数转换为可见性等级
    
    Args:
        score: 可见性分数 [0, 1]
    
    Returns:
        等级: FULLY_VISIBLE / MOSTLY_VISIBLE / PARTIALLY_VISIBLE / 
              MOSTLY_OCCLUDED / FULLY_OCCLUDED
    """
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
    """预热JIT编译（如果使用numba）
    
    在程序启动时调用一次，可避免首次计算的延迟。
    """
    if not HAS_NUMBA:
        return
    
    # 小规模数据触发编译
    dummy_pts = np.random.randn(50, 3).astype(np.float64)
    dummy_bbox = np.array([0, 0, 0, 2, 2, 2, 0], dtype=np.float64)
    compute_visibility(dummy_bbox, dummy_pts)


# ==================== 测试代码 ====================

if __name__ == '__main__':
    import time
    
    print("=" * 60)
    print("3D Bounding Box Visibility Calculator")
    print("=" * 60)
    print(f"Numba available: {HAS_NUMBA}")
    
    np.random.seed(42)
    
    # 创建测试场景
    # 背景点云
    bg_points = np.random.randn(5000, 3) * 30
    bg_points[:, 2] = np.abs(bg_points[:, 2]) * 0.3 - 1
    
    # 可见目标：在表面生成密集点云
    car_center = np.array([12, 4, 0.8])
    car_size = np.array([4.5, 2.0, 1.5])
    car_yaw = 0.15
    
    # 生成表面点
    n_surface = 400
    surface_pts = []
    for _ in range(n_surface):
        face = np.random.randint(6)
        if face == 0:
            p = [car_size[0]/2, np.random.uniform(-1, 1) * car_size[1]/2,
                 np.random.uniform(-1, 1) * car_size[2]/2]
        elif face == 1:
            p = [-car_size[0]/2, np.random.uniform(-1, 1) * car_size[1]/2,
                 np.random.uniform(-1, 1) * car_size[2]/2]
        elif face == 2:
            p = [np.random.uniform(-1, 1) * car_size[0]/2, car_size[1]/2,
                 np.random.uniform(-1, 1) * car_size[2]/2]
        elif face == 3:
            p = [np.random.uniform(-1, 1) * car_size[0]/2, -car_size[1]/2,
                 np.random.uniform(-1, 1) * car_size[2]/2]
        elif face == 4:
            p = [np.random.uniform(-1, 1) * car_size[0]/2,
                 np.random.uniform(-1, 1) * car_size[1]/2, car_size[2]/2]
        else:
            p = [np.random.uniform(-1, 1) * car_size[0]/2,
                 np.random.uniform(-1, 1) * car_size[1]/2, -car_size[2]/2]
        surface_pts.append(p)
    
    surface_pts = np.array(surface_pts)
    # 旋转
    c, s = np.cos(car_yaw), np.sin(car_yaw)
    rot_pts = np.zeros_like(surface_pts)
    rot_pts[:, 0] = c * surface_pts[:, 0] - s * surface_pts[:, 1]
    rot_pts[:, 1] = s * surface_pts[:, 0] + c * surface_pts[:, 1]
    rot_pts[:, 2] = surface_pts[:, 2]
    car_points = rot_pts + car_center
    
    # 遮挡目标：只有少量点
    occ_center = np.array([40, -8, 0.8])
    occ_size = np.array([4.5, 2.0, 1.5])
    occ_yaw = -0.3
    occ_points = np.random.randn(25, 3) * 0.2 + occ_center
    
    # 合并
    all_points = np.vstack([bg_points, car_points, occ_points])
    
    # 传感器位置
    sensor = np.array([0, 0, 1.8])
    
    # 预热
    print("\nWarming up...")
    warmup()
    
    # 测试可见目标
    print("\n--- Test 1: Visible Object ---")
    bbox1 = np.array([*car_center, *car_size, car_yaw])
    t0 = time.time()
    score1, details1 = compute_visibility(bbox1, all_points, sensor)
    t1 = time.time()
    
    print(f"Score: {score1:.3f} ({classify_visibility(score1)})")
    print(f"Points in box: {details1['point_count']}")
    print(f"Point density: {details1['point_density']:.1f} pts/m³")
    print(f"Voxel occupancy: {details1['voxel_occupancy']:.3f}")
    print(f"Surface visibility: {details1['surface_visibility']:.3f}")
    print(f"Depth continuity: {details1['depth_continuity']:.3f}")
    print(f"Time: {(t1-t0)*1000:.2f} ms")
    
    # 测试遮挡目标
    print("\n--- Test 2: Occluded Object ---")
    bbox2 = np.array([*occ_center, *occ_size, occ_yaw])
    t0 = time.time()
    score2, details2 = compute_visibility(bbox2, all_points, sensor)
    t1 = time.time()
    
    print(f"Score: {score2:.3f} ({classify_visibility(score2)})")
    print(f"Points in box: {details2['point_count']}")
    print(f"Point density: {details2['point_density']:.1f} pts/m³")
    print(f"Voxel occupancy: {details2['voxel_occupancy']:.3f}")
    print(f"Surface visibility: {details2['surface_visibility']:.3f}")
    print(f"Depth continuity: {details2['depth_continuity']:.3f}")
    print(f"Time: {(t1-t0)*1000:.2f} ms")
    
    # 批量性能测试
    print("\n--- Test 3: Batch Performance ---")
    n_boxes = 100
    test_boxes = np.random.randn(n_boxes, 7)
    test_boxes[:, :3] *= 25
    test_boxes[:, 3:6] = np.abs(test_boxes[:, 3:6]) * 2 + 1
    test_boxes[:, 6] *= np.pi
    
    t0 = time.time()
    scores, _ = compute_visibility_batch(test_boxes, all_points, sensor)
    t1 = time.time()
    
    print(f"Boxes: {n_boxes}")
    print(f"Total time: {(t1-t0)*1000:.2f} ms")
    print(f"Per box: {(t1-t0)/n_boxes*1000:.3f} ms")
    
    print("\n" + "=" * 60)
    print("USAGE:")
    print("=" * 60)
    print("""
from bbox_visibility import compute_visibility, classify_visibility, warmup

# 预热（可选，首次调用前执行）
warmup()

# 计算单个框
bbox = [cx, cy, cz, length, width, height, yaw]
score, details = compute_visibility(bbox, points, sensor_origin)

# 分类
level = classify_visibility(score)

# 批量计算
from bbox_visibility import compute_visibility_batch
scores, details_list = compute_visibility_batch(bboxes, points)
""")
