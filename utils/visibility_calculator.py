"""
3D Bounding Box Visibility Calculator

基于OCC3D体素遮挡思想，计算3D标注框的可见性量化指标。

输入：
    - bbox: 3D标注框 (center_x, center_y, center_z, length, width, height, yaw)
    - points: 点云 (N, 3) 或 (N, 4+)
    - sensor_origin: 传感器原点位置 (3,)，默认为(0,0,0)

输出：
    - visibility_score: 综合可见性分数 [0, 1]
    - visibility_details: 详细的可见性指标字典

Author: Auto-generated
"""

import numpy as np
import numba
from numba import njit, prange
from typing import Tuple, Dict, Optional, List, Union
import warnings


# ============== 常量定义 ==============
DEFAULT_VOXEL_SIZE = 0.1  # 默认体素大小
MIN_POINTS_FOR_VALID = 5  # 最小有效点数
SURFACE_NAMES = ['front', 'back', 'left', 'right', 'top', 'bottom']


# ============== 核心计算函数（numba加速） ==============

@njit(cache=True)
def rotate_points_z(points: np.ndarray, angle: float) -> np.ndarray:
    """绕Z轴旋转点云（numba加速）
    
    Args:
        points: (N, 3) 点云
        angle: 旋转角度（弧度）
    
    Returns:
        rotated_points: (N, 3) 旋转后的点云
    """
    cos_a = np.cos(angle)
    sin_a = np.sin(angle)
    
    rotated = np.empty_like(points)
    for i in range(points.shape[0]):
        x, y, z = points[i, 0], points[i, 1], points[i, 2]
        rotated[i, 0] = cos_a * x - sin_a * y
        rotated[i, 1] = sin_a * x + cos_a * y
        rotated[i, 2] = z
    
    return rotated


@njit(cache=True)
def points_in_bbox_fast(points: np.ndarray, 
                        center: np.ndarray, 
                        size: np.ndarray, 
                        yaw: float) -> np.ndarray:
    """快速判断点是否在旋转包围盒内（numba加速）
    
    Args:
        points: (N, 3) 点云
        center: (3,) 包围盒中心
        size: (3,) 包围盒尺寸 (length, width, height)
        yaw: 绕Z轴的旋转角度
    
    Returns:
        mask: (N,) 布尔数组，True表示点在框内
    """
    n_points = points.shape[0]
    mask = np.zeros(n_points, dtype=numba.boolean)
    
    # 预计算
    cos_yaw = np.cos(-yaw)  # 逆旋转
    sin_yaw = np.sin(-yaw)
    half_l, half_w, half_h = size[0] / 2, size[1] / 2, size[2] / 2
    
    for i in range(n_points):
        # 平移到以bbox中心为原点
        dx = points[i, 0] - center[0]
        dy = points[i, 1] - center[1]
        dz = points[i, 2] - center[2]
        
        # 逆旋转到bbox坐标系
        local_x = cos_yaw * dx - sin_yaw * dy
        local_y = sin_yaw * dx + cos_yaw * dy
        local_z = dz
        
        # 判断是否在框内
        if (abs(local_x) <= half_l and 
            abs(local_y) <= half_w and 
            abs(local_z) <= half_h):
            mask[i] = True
    
    return mask


@njit(cache=True)
def compute_voxel_occupancy(points_local: np.ndarray,
                            size: np.ndarray,
                            voxel_size: float) -> Tuple[float, np.ndarray]:
    """计算体素占用率（numba加速）
    
    Args:
        points_local: (N, 3) 局部坐标系下的点云（已在bbox内）
        size: (3,) 包围盒尺寸
        voxel_size: 体素大小
    
    Returns:
        occupancy_ratio: 体素占用率
        voxel_grid: 体素网格 (用于后续分析)
    """
    # 计算体素网格尺寸
    nx = max(1, int(np.ceil(size[0] / voxel_size)))
    ny = max(1, int(np.ceil(size[1] / voxel_size)))
    nz = max(1, int(np.ceil(size[2] / voxel_size)))
    
    # 创建体素网格
    voxel_grid = np.zeros((nx, ny, nz), dtype=numba.boolean)
    
    # 偏移量（将点从[-half, half]映射到[0, size]）
    offset_x = size[0] / 2
    offset_y = size[1] / 2
    offset_z = size[2] / 2
    
    # 填充体素
    for i in range(points_local.shape[0]):
        x = points_local[i, 0] + offset_x
        y = points_local[i, 1] + offset_y
        z = points_local[i, 2] + offset_z
        
        vx = min(nx - 1, max(0, int(x / voxel_size)))
        vy = min(ny - 1, max(0, int(y / voxel_size)))
        vz = min(nz - 1, max(0, int(z / voxel_size)))
        
        voxel_grid[vx, vy, vz] = True
    
    # 计算占用率
    total_voxels = nx * ny * nz
    occupied_voxels = 0
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                if voxel_grid[i, j, k]:
                    occupied_voxels += 1
    
    occupancy_ratio = occupied_voxels / total_voxels
    
    return occupancy_ratio, voxel_grid


@njit(cache=True)
def compute_surface_visibility(points_local: np.ndarray,
                               size: np.ndarray,
                               sensor_local: np.ndarray,
                               surface_threshold: float = 0.1) -> np.ndarray:
    """计算各表面的可见性（numba加速）
    
    基于点云分布和传感器位置，估算6个表面的可见性。
    
    Args:
        points_local: (N, 3) 局部坐标系下的点云
        size: (3,) 包围盒尺寸
        sensor_local: (3,) 传感器在局部坐标系下的位置
        surface_threshold: 表面判定阈值（相对于尺寸）
    
    Returns:
        surface_visibility: (6,) 各表面可见性 [front, back, left, right, top, bottom]
    """
    half_l, half_w, half_h = size[0] / 2, size[1] / 2, size[2] / 2
    surface_visibility = np.zeros(6, dtype=np.float64)
    surface_counts = np.zeros(6, dtype=np.int64)
    
    # 表面阈值
    thresh_l = size[0] * surface_threshold
    thresh_w = size[1] * surface_threshold
    thresh_h = size[2] * surface_threshold
    
    n_points = points_local.shape[0]
    if n_points == 0:
        return surface_visibility
    
    # 统计各表面附近的点
    for i in range(n_points):
        x, y, z = points_local[i, 0], points_local[i, 1], points_local[i, 2]
        
        # front (+x)
        if x > half_l - thresh_l:
            surface_counts[0] += 1
        # back (-x)
        if x < -half_l + thresh_l:
            surface_counts[1] += 1
        # left (+y)
        if y > half_w - thresh_w:
            surface_counts[2] += 1
        # right (-y)
        if y < -half_w + thresh_w:
            surface_counts[3] += 1
        # top (+z)
        if z > half_h - thresh_h:
            surface_counts[4] += 1
        # bottom (-z)
        if z < -half_h + thresh_h:
            surface_counts[5] += 1
    
    # 根据传感器位置判断哪些面理论上可见
    # 并计算表面可见性分数
    theoretical_visible = np.zeros(6, dtype=numba.boolean)
    
    # front (+x): 传感器在+x方向才可见
    theoretical_visible[0] = sensor_local[0] > 0
    # back (-x): 传感器在-x方向才可见
    theoretical_visible[1] = sensor_local[0] < 0
    # left (+y): 传感器在+y方向才可见
    theoretical_visible[2] = sensor_local[1] > 0
    # right (-y): 传感器在-y方向才可见
    theoretical_visible[3] = sensor_local[1] < 0
    # top (+z): 传感器在+z方向才可见
    theoretical_visible[4] = sensor_local[2] > 0
    # bottom (-z): 传感器在-z方向才可见
    theoretical_visible[5] = sensor_local[2] < 0
    
    # 计算各表面期望点数（基于表面面积）
    surface_areas = np.array([
        size[1] * size[2],  # front/back (y*z)
        size[1] * size[2],  # front/back
        size[0] * size[2],  # left/right (x*z)
        size[0] * size[2],  # left/right
        size[0] * size[1],  # top/bottom (x*y)
        size[0] * size[1],  # top/bottom
    ])
    
    # 归一化期望点数
    total_area = np.sum(surface_areas)
    expected_ratio = surface_areas / total_area
    
    # 计算可见性分数
    for i in range(6):
        if theoretical_visible[i]:
            expected_points = max(1, int(n_points * expected_ratio[i] * 0.5))
            surface_visibility[i] = min(1.0, surface_counts[i] / expected_points)
        else:
            # 理论上不可见的面，有点也算0.5分（说明有遮挡绕射）
            if surface_counts[i] > 0:
                surface_visibility[i] = 0.3
            else:
                surface_visibility[i] = 0.0
    
    return surface_visibility


@njit(cache=True)
def compute_ray_occlusion(points: np.ndarray,
                          bbox_center: np.ndarray,
                          bbox_size: np.ndarray,
                          bbox_yaw: float,
                          sensor_origin: np.ndarray,
                          n_rays: int = 100) -> float:
    """基于射线投射计算遮挡率（numba加速）
    
    从传感器向bbox发射多条射线，检测是否被其他点云遮挡。
    
    Args:
        points: (N, 3) 全部点云
        bbox_center: (3,) bbox中心
        bbox_size: (3,) bbox尺寸
        bbox_yaw: bbox旋转角
        sensor_origin: (3,) 传感器位置
        n_rays: 射线数量
    
    Returns:
        occlusion_ratio: 遮挡率 [0, 1]，0表示完全可见，1表示完全遮挡
    """
    # 生成bbox表面上的采样点
    half_l, half_w, half_h = bbox_size[0] / 2, bbox_size[1] / 2, bbox_size[2] / 2
    
    # 简化：在bbox表面均匀采样
    n_samples_per_dim = max(2, int(np.sqrt(n_rays / 6)))
    
    occluded_rays = 0
    total_rays = 0
    
    # 计算到bbox中心的距离
    dist_to_bbox = np.sqrt(
        (bbox_center[0] - sensor_origin[0])**2 +
        (bbox_center[1] - sensor_origin[1])**2 +
        (bbox_center[2] - sensor_origin[2])**2
    )
    
    # 对每个表面采样
    cos_yaw = np.cos(bbox_yaw)
    sin_yaw = np.sin(bbox_yaw)
    
    # 生成表面采样点（简化为前后左右4个面）
    for face in range(4):
        for i in range(n_samples_per_dim):
            for j in range(n_samples_per_dim):
                # 生成局部坐标
                if face == 0:  # front
                    local_x = half_l
                    local_y = -half_w + (i + 0.5) * (2 * half_w / n_samples_per_dim)
                    local_z = -half_h + (j + 0.5) * (2 * half_h / n_samples_per_dim)
                elif face == 1:  # back
                    local_x = -half_l
                    local_y = -half_w + (i + 0.5) * (2 * half_w / n_samples_per_dim)
                    local_z = -half_h + (j + 0.5) * (2 * half_h / n_samples_per_dim)
                elif face == 2:  # left
                    local_x = -half_l + (i + 0.5) * (2 * half_l / n_samples_per_dim)
                    local_y = half_w
                    local_z = -half_h + (j + 0.5) * (2 * half_h / n_samples_per_dim)
                else:  # right
                    local_x = -half_l + (i + 0.5) * (2 * half_l / n_samples_per_dim)
                    local_y = -half_w
                    local_z = -half_h + (j + 0.5) * (2 * half_h / n_samples_per_dim)
                
                # 旋转到全局坐标
                global_x = cos_yaw * local_x - sin_yaw * local_y + bbox_center[0]
                global_y = sin_yaw * local_x + cos_yaw * local_y + bbox_center[1]
                global_z = local_z + bbox_center[2]
                
                # 检查这条射线是否被遮挡
                ray_dir_x = global_x - sensor_origin[0]
                ray_dir_y = global_y - sensor_origin[1]
                ray_dir_z = global_z - sensor_origin[2]
                ray_len = np.sqrt(ray_dir_x**2 + ray_dir_y**2 + ray_dir_z**2)
                
                if ray_len < 1e-6:
                    continue
                
                # 归一化射线方向
                ray_dir_x /= ray_len
                ray_dir_y /= ray_len
                ray_dir_z /= ray_len
                
                total_rays += 1
                
                # 检查是否有点遮挡了这条射线
                for p_idx in range(points.shape[0]):
                    px = points[p_idx, 0] - sensor_origin[0]
                    py = points[p_idx, 1] - sensor_origin[1]
                    pz = points[p_idx, 2] - sensor_origin[2]
                    
                    # 点到射线的投影长度
                    proj_len = px * ray_dir_x + py * ray_dir_y + pz * ray_dir_z
                    
                    # 只考虑在传感器和目标之间的点
                    if proj_len <= 0 or proj_len >= ray_len * 0.95:
                        continue
                    
                    # 计算点到射线的距离
                    closest_x = proj_len * ray_dir_x
                    closest_y = proj_len * ray_dir_y
                    closest_z = proj_len * ray_dir_z
                    
                    dist_to_ray = np.sqrt(
                        (px - closest_x)**2 + 
                        (py - closest_y)**2 + 
                        (pz - closest_z)**2
                    )
                    
                    # 如果距离小于阈值，认为被遮挡
                    if dist_to_ray < 0.3:  # 遮挡判定阈值
                        occluded_rays += 1
                        break
    
    if total_rays == 0:
        return 0.0
    
    return occluded_rays / total_rays


@njit(cache=True)
def compute_depth_continuity(points_local: np.ndarray,
                             sensor_local: np.ndarray,
                             n_bins: int = 10) -> float:
    """计算深度连续性指标（numba加速）
    
    评估框内点云在深度方向上的分布连续性。
    连续性好说明遮挡少，连续性差说明可能有遮挡。
    
    Args:
        points_local: (N, 3) 局部坐标系下的点云
        sensor_local: (3,) 传感器在局部坐标系下的位置
        n_bins: 深度分箱数
    
    Returns:
        continuity_score: 连续性分数 [0, 1]
    """
    n_points = points_local.shape[0]
    if n_points < 3:
        return 0.0
    
    # 计算每个点到传感器的距离
    depths = np.empty(n_points, dtype=np.float64)
    for i in range(n_points):
        dx = points_local[i, 0] - sensor_local[0]
        dy = points_local[i, 1] - sensor_local[1]
        dz = points_local[i, 2] - sensor_local[2]
        depths[i] = np.sqrt(dx*dx + dy*dy + dz*dz)
    
    # 深度分箱
    min_depth = depths.min()
    max_depth = depths.max()
    
    if max_depth - min_depth < 0.1:
        return 1.0  # 深度范围很小，认为连续
    
    bin_size = (max_depth - min_depth) / n_bins
    bin_counts = np.zeros(n_bins, dtype=np.int64)
    
    for i in range(n_points):
        bin_idx = min(n_bins - 1, int((depths[i] - min_depth) / bin_size))
        bin_counts[bin_idx] += 1
    
    # 计算非空bin的比例
    non_empty_bins = 0
    for i in range(n_bins):
        if bin_counts[i] > 0:
            non_empty_bins += 1
    
    continuity_score = non_empty_bins / n_bins
    
    return continuity_score


# ============== 主计算类 ==============

class VisibilityCalculator:
    """3D标注框可见性计算器
    
    综合多种指标计算3D标注框的可见性分数。
    
    Attributes:
        voxel_size: 体素大小
        n_rays: 射线数量（用于遮挡检测）
        weights: 各指标权重
    """
    
    def __init__(self,
                 voxel_size: float = DEFAULT_VOXEL_SIZE,
                 n_rays: int = 64,
                 weights: Optional[Dict[str, float]] = None):
        """初始化计算器
        
        Args:
            voxel_size: 体素大小，默认0.1m
            n_rays: 射线投射数量
            weights: 各指标权重，默认为均匀权重
        """
        self.voxel_size = voxel_size
        self.n_rays = n_rays
        
        # 默认权重
        self.weights = weights or {
            'point_density': 0.25,
            'voxel_occupancy': 0.25,
            'surface_visibility': 0.25,
            'depth_continuity': 0.25,
        }
        
        # 归一化权重
        total_weight = sum(self.weights.values())
        self.weights = {k: v / total_weight for k, v in self.weights.items()}
    
    def compute_visibility(self,
                          bbox: np.ndarray,
                          points: np.ndarray,
                          sensor_origin: np.ndarray = None,
                          compute_occlusion: bool = False) -> Tuple[float, Dict]:
        """计算单个bbox的可见性
        
        Args:
            bbox: (7,) 标注框 [cx, cy, cz, l, w, h, yaw]
            points: (N, 3+) 点云
            sensor_origin: (3,) 传感器原点，默认(0,0,0)
            compute_occlusion: 是否计算射线遮挡（较慢）
        
        Returns:
            visibility_score: 综合可见性分数 [0, 1]
            details: 详细指标字典
        """
        # 默认传感器位置
        if sensor_origin is None:
            sensor_origin = np.array([0.0, 0.0, 0.0])
        
        # 确保数据类型
        bbox = np.asarray(bbox, dtype=np.float64)
        points = np.asarray(points[:, :3], dtype=np.float64)
        sensor_origin = np.asarray(sensor_origin, dtype=np.float64)
        
        # 解析bbox
        center = bbox[:3]
        size = bbox[3:6]  # length, width, height
        yaw = bbox[6]
        
        # 1. 获取框内点云
        in_box_mask = points_in_bbox_fast(points, center, size, yaw)
        points_in_box = points[in_box_mask]
        n_points_in_box = points_in_box.shape[0]
        
        # 将点转换到局部坐标系
        if n_points_in_box > 0:
            points_local = self._transform_to_local(points_in_box, center, yaw)
            sensor_local = self._transform_to_local(
                sensor_origin.reshape(1, 3), center, yaw
            )[0]
        else:
            points_local = np.empty((0, 3), dtype=np.float64)
            sensor_local = np.array([0.0, 0.0, 0.0])
        
        # 2. 计算各项指标
        details = {}
        
        # 2.1 点云密度指标
        volume = size[0] * size[1] * size[2]
        point_density = n_points_in_box / max(volume, 1e-6)
        # 归一化（假设理想密度为100点/m³）
        ideal_density = 100.0
        density_score = min(1.0, point_density / ideal_density)
        details['point_density'] = density_score
        details['raw_point_count'] = n_points_in_box
        details['raw_point_density'] = point_density
        
        # 2.2 体素占用率
        if n_points_in_box >= MIN_POINTS_FOR_VALID:
            voxel_occupancy, _ = compute_voxel_occupancy(
                points_local, size, self.voxel_size
            )
        else:
            voxel_occupancy = 0.0
        details['voxel_occupancy'] = voxel_occupancy
        
        # 2.3 表面可见性
        if n_points_in_box >= MIN_POINTS_FOR_VALID:
            surface_vis = compute_surface_visibility(
                points_local, size, sensor_local
            )
            # 计算可见面的平均可见性
            visible_surfaces = surface_vis[surface_vis > 0]
            if len(visible_surfaces) > 0:
                surface_visibility_score = np.mean(visible_surfaces)
            else:
                surface_visibility_score = 0.0
        else:
            surface_vis = np.zeros(6)
            surface_visibility_score = 0.0
        
        details['surface_visibility'] = surface_visibility_score
        details['surface_details'] = {
            name: float(surface_vis[i]) 
            for i, name in enumerate(SURFACE_NAMES)
        }
        
        # 2.4 深度连续性
        if n_points_in_box >= MIN_POINTS_FOR_VALID:
            depth_continuity = compute_depth_continuity(points_local, sensor_local)
        else:
            depth_continuity = 0.0
        details['depth_continuity'] = depth_continuity
        
        # 2.5 射线遮挡（可选，较慢）
        if compute_occlusion and n_points_in_box >= MIN_POINTS_FOR_VALID:
            occlusion_ratio = compute_ray_occlusion(
                points, center, size, yaw, sensor_origin, self.n_rays
            )
            details['occlusion_ratio'] = occlusion_ratio
            details['occlusion_free'] = 1.0 - occlusion_ratio
        
        # 3. 计算综合分数
        visibility_score = (
            self.weights['point_density'] * density_score +
            self.weights['voxel_occupancy'] * voxel_occupancy +
            self.weights['surface_visibility'] * surface_visibility_score +
            self.weights['depth_continuity'] * depth_continuity
        )
        
        details['visibility_score'] = visibility_score
        
        return visibility_score, details
    
    def compute_visibility_batch(self,
                                 bboxes: np.ndarray,
                                 points: np.ndarray,
                                 sensor_origin: np.ndarray = None,
                                 compute_occlusion: bool = False) -> Tuple[np.ndarray, List[Dict]]:
        """批量计算多个bbox的可见性
        
        Args:
            bboxes: (M, 7) 多个标注框
            points: (N, 3+) 点云
            sensor_origin: (3,) 传感器原点
            compute_occlusion: 是否计算射线遮挡
        
        Returns:
            visibility_scores: (M,) 可见性分数数组
            details_list: 详细指标列表
        """
        bboxes = np.asarray(bboxes, dtype=np.float64)
        n_boxes = bboxes.shape[0]
        
        visibility_scores = np.zeros(n_boxes)
        details_list = []
        
        for i in range(n_boxes):
            score, details = self.compute_visibility(
                bboxes[i], points, sensor_origin, compute_occlusion
            )
            visibility_scores[i] = score
            details_list.append(details)
        
        return visibility_scores, details_list
    
    def _transform_to_local(self, 
                           points: np.ndarray, 
                           center: np.ndarray, 
                           yaw: float) -> np.ndarray:
        """将点转换到bbox局部坐标系
        
        Args:
            points: (N, 3) 点云
            center: (3,) bbox中心
            yaw: 旋转角
        
        Returns:
            points_local: (N, 3) 局部坐标系下的点云
        """
        # 平移
        points_centered = points - center
        # 逆旋转
        points_local = rotate_points_z(points_centered, -yaw)
        return points_local


# ============== 预热函数（加速首次调用） ==============

def warmup():
    """预热JIT编译的函数，避免首次调用时的编译延迟
    
    建议在程序启动时调用一次。
    """
    # 创建小规模测试数据
    dummy_points = np.random.randn(100, 3).astype(np.float64)
    dummy_bbox = np.array([0, 0, 0, 2, 2, 2, 0], dtype=np.float64)
    dummy_center = dummy_bbox[:3]
    dummy_size = dummy_bbox[3:6]
    dummy_sensor = np.zeros(3, dtype=np.float64)
    
    # 预热所有numba函数
    _ = rotate_points_z(dummy_points, 0.1)
    _ = points_in_bbox_fast(dummy_points, dummy_center, dummy_size, 0.1)
    
    local_pts = np.random.randn(50, 3).astype(np.float64)
    _ = compute_voxel_occupancy(local_pts, dummy_size, 0.5)
    _ = compute_surface_visibility(local_pts, dummy_size, dummy_sensor)
    _ = compute_depth_continuity(local_pts, dummy_sensor)
    _ = compute_ray_occlusion(dummy_points, dummy_center, dummy_size, 0.0, dummy_sensor, 16)


# ============== 便捷函数 ==============

def compute_bbox_visibility(bbox: Union[np.ndarray, List],
                           points: np.ndarray,
                           sensor_origin: Optional[np.ndarray] = None,
                           voxel_size: float = 0.1,
                           compute_occlusion: bool = False) -> Tuple[float, Dict]:
    """便捷函数：计算单个bbox的可见性
    
    Args:
        bbox: (7,) 标注框 [cx, cy, cz, l, w, h, yaw]
        points: (N, 3+) 点云
        sensor_origin: (3,) 传感器原点
        voxel_size: 体素大小
        compute_occlusion: 是否计算射线遮挡
    
    Returns:
        visibility_score: 可见性分数
        details: 详细指标
    
    Example:
        >>> bbox = np.array([10, 5, 1, 4.5, 2, 1.5, 0.5])  # 一辆车
        >>> points = np.random.randn(10000, 3) * 50  # 模拟点云
        >>> score, details = compute_bbox_visibility(bbox, points)
        >>> print(f"Visibility: {score:.2f}")
    """
    calculator = VisibilityCalculator(voxel_size=voxel_size)
    return calculator.compute_visibility(
        np.asarray(bbox), points, sensor_origin, compute_occlusion
    )


def compute_bboxes_visibility(bboxes: np.ndarray,
                             points: np.ndarray,
                             sensor_origin: Optional[np.ndarray] = None,
                             voxel_size: float = 0.1,
                             compute_occlusion: bool = False) -> Tuple[np.ndarray, List[Dict]]:
    """便捷函数：批量计算多个bbox的可见性
    
    Args:
        bboxes: (M, 7) 多个标注框
        points: (N, 3+) 点云
        sensor_origin: (3,) 传感器原点
        voxel_size: 体素大小
        compute_occlusion: 是否计算射线遮挡
    
    Returns:
        visibility_scores: (M,) 可见性分数数组
        details_list: 详细指标列表
    """
    calculator = VisibilityCalculator(voxel_size=voxel_size)
    return calculator.compute_visibility_batch(
        bboxes, points, sensor_origin, compute_occlusion
    )


# ============== 可见性等级分类 ==============

def classify_visibility(score: float) -> str:
    """将可见性分数分类为等级
    
    Args:
        score: 可见性分数 [0, 1]
    
    Returns:
        visibility_level: 可见性等级字符串
    """
    if score >= 0.7:
        return "FULLY_VISIBLE"
    elif score >= 0.4:
        return "MOSTLY_VISIBLE"
    elif score >= 0.2:
        return "PARTIALLY_VISIBLE"
    elif score > 0:
        return "MOSTLY_OCCLUDED"
    else:
        return "FULLY_OCCLUDED"


def classify_visibility_batch(scores: np.ndarray) -> List[str]:
    """批量分类可见性等级
    
    Args:
        scores: (N,) 可见性分数数组
    
    Returns:
        levels: 可见性等级列表
    """
    return [classify_visibility(s) for s in scores]


# ============== 与OCC3D兼容的接口 ==============

def compute_visibility_from_occ3d_format(
    bboxes: np.ndarray,
    points: np.ndarray,
    ego2global: Optional[np.ndarray] = None,
    voxel_size: float = 0.1
) -> Tuple[np.ndarray, List[Dict], List[str]]:
    """从OCC3D格式的数据计算可见性
    
    Args:
        bboxes: (M, 7+) 标注框，格式为 [cx, cy, cz, l, w, h, yaw, ...]
        points: (N, 3+) 点云
        ego2global: (4, 4) 自车到全局的变换矩阵（可选）
        voxel_size: 体素大小
    
    Returns:
        scores: (M,) 可见性分数
        details: 详细指标列表
        levels: 可见性等级列表
    """
    # 假设传感器在自车坐标系原点上方
    sensor_origin = np.array([0.0, 0.0, 1.8])
    
    # 如果提供了ego2global，可以用于更精确的计算
    # 这里暂时不使用
    
    calculator = VisibilityCalculator(voxel_size=voxel_size)
    scores, details = calculator.compute_visibility_batch(
        bboxes[:, :7], points, sensor_origin
    )
    levels = classify_visibility_batch(scores)
    
    return scores, details, levels


# ============== 测试代码 ==============

if __name__ == '__main__':
    import time
    
    print("=" * 60)
    print("3D Bounding Box Visibility Calculator - Test")
    print("=" * 60)
    
    # 生成测试数据
    np.random.seed(42)
    
    # 模拟场景：一个可见的车和一个被遮挡的车
    # 1. 生成背景点云
    n_background = 5000
    background_points = np.random.randn(n_background, 3) * 30
    background_points[:, 2] = np.abs(background_points[:, 2]) * 0.5 - 1  # 地面附近
    
    # 2. 生成一个可见的车（前方）
    visible_car_center = np.array([15, 3, 0.8])
    visible_car_size = np.array([4.5, 2.0, 1.5])
    visible_car_yaw = 0.1
    
    # 在车的表面生成点云
    n_car_points = 500
    car_surface_points = []
    for _ in range(n_car_points):
        # 随机选择一个面
        face = np.random.randint(0, 6)
        if face == 0:  # front
            p = [visible_car_size[0]/2, 
                 np.random.uniform(-visible_car_size[1]/2, visible_car_size[1]/2),
                 np.random.uniform(-visible_car_size[2]/2, visible_car_size[2]/2)]
        elif face == 1:  # back
            p = [-visible_car_size[0]/2,
                 np.random.uniform(-visible_car_size[1]/2, visible_car_size[1]/2),
                 np.random.uniform(-visible_car_size[2]/2, visible_car_size[2]/2)]
        elif face == 2:  # left
            p = [np.random.uniform(-visible_car_size[0]/2, visible_car_size[0]/2),
                 visible_car_size[1]/2,
                 np.random.uniform(-visible_car_size[2]/2, visible_car_size[2]/2)]
        elif face == 3:  # right
            p = [np.random.uniform(-visible_car_size[0]/2, visible_car_size[0]/2),
                 -visible_car_size[1]/2,
                 np.random.uniform(-visible_car_size[2]/2, visible_car_size[2]/2)]
        elif face == 4:  # top
            p = [np.random.uniform(-visible_car_size[0]/2, visible_car_size[0]/2),
                 np.random.uniform(-visible_car_size[1]/2, visible_car_size[1]/2),
                 visible_car_size[2]/2]
        else:  # bottom
            p = [np.random.uniform(-visible_car_size[0]/2, visible_car_size[0]/2),
                 np.random.uniform(-visible_car_size[1]/2, visible_car_size[1]/2),
                 -visible_car_size[2]/2]
        car_surface_points.append(p)
    
    car_surface_points = np.array(car_surface_points)
    # 旋转并平移
    cos_yaw = np.cos(visible_car_yaw)
    sin_yaw = np.sin(visible_car_yaw)
    rotated_points = np.zeros_like(car_surface_points)
    rotated_points[:, 0] = cos_yaw * car_surface_points[:, 0] - sin_yaw * car_surface_points[:, 1]
    rotated_points[:, 1] = sin_yaw * car_surface_points[:, 0] + cos_yaw * car_surface_points[:, 1]
    rotated_points[:, 2] = car_surface_points[:, 2]
    visible_car_points = rotated_points + visible_car_center
    
    # 3. 生成一个被遮挡的车（少量点）
    occluded_car_center = np.array([35, -5, 0.8])
    occluded_car_size = np.array([4.5, 2.0, 1.5])
    occluded_car_yaw = -0.2
    # 只有少量点
    occluded_car_points = np.random.randn(30, 3) * 0.3 + occluded_car_center
    
    # 合并所有点云
    all_points = np.vstack([background_points, visible_car_points, occluded_car_points])
    
    # 定义bbox
    visible_car_bbox = np.array([
        visible_car_center[0], visible_car_center[1], visible_car_center[2],
        visible_car_size[0], visible_car_size[1], visible_car_size[2],
        visible_car_yaw
    ])
    
    occluded_car_bbox = np.array([
        occluded_car_center[0], occluded_car_center[1], occluded_car_center[2],
        occluded_car_size[0], occluded_car_size[1], occluded_car_size[2],
        occluded_car_yaw
    ])
    
    # 传感器位置（自车位置）
    sensor_origin = np.array([0.0, 0.0, 1.8])
    
    # 创建计算器
    calculator = VisibilityCalculator(voxel_size=0.1)
    
    # 测试单个bbox
    print("\n--- Test 1: Visible Car ---")
    start_time = time.time()
    score1, details1 = calculator.compute_visibility(
        visible_car_bbox, all_points, sensor_origin
    )
    time1 = time.time() - start_time
    
    print(f"Visibility Score: {score1:.3f}")
    print(f"Point Count: {details1['raw_point_count']}")
    print(f"Point Density Score: {details1['point_density']:.3f}")
    print(f"Voxel Occupancy: {details1['voxel_occupancy']:.3f}")
    print(f"Surface Visibility: {details1['surface_visibility']:.3f}")
    print(f"Depth Continuity: {details1['depth_continuity']:.3f}")
    print(f"Computation Time: {time1*1000:.2f} ms")
    print("Surface Details:", details1['surface_details'])
    
    print("\n--- Test 2: Occluded Car ---")
    start_time = time.time()
    score2, details2 = calculator.compute_visibility(
        occluded_car_bbox, all_points, sensor_origin
    )
    time2 = time.time() - start_time
    
    print(f"Visibility Score: {score2:.3f}")
    print(f"Point Count: {details2['raw_point_count']}")
    print(f"Point Density Score: {details2['point_density']:.3f}")
    print(f"Voxel Occupancy: {details2['voxel_occupancy']:.3f}")
    print(f"Surface Visibility: {details2['surface_visibility']:.3f}")
    print(f"Depth Continuity: {details2['depth_continuity']:.3f}")
    print(f"Computation Time: {time2*1000:.2f} ms")
    
    # 批量测试性能
    print("\n--- Test 3: Batch Performance ---")
    n_test_boxes = 100
    test_bboxes = np.random.randn(n_test_boxes, 7)
    test_bboxes[:, :3] = test_bboxes[:, :3] * 30  # 位置
    test_bboxes[:, 3:6] = np.abs(test_bboxes[:, 3:6]) * 3 + 1  # 尺寸
    test_bboxes[:, 6] = test_bboxes[:, 6] * np.pi  # 角度
    
    start_time = time.time()
    scores, _ = calculator.compute_visibility_batch(
        test_bboxes, all_points, sensor_origin
    )
    total_time = time.time() - start_time
    
    print(f"Processed {n_test_boxes} boxes in {total_time*1000:.2f} ms")
    print(f"Average time per box: {total_time/n_test_boxes*1000:.2f} ms")
    print(f"Score range: [{scores.min():.3f}, {scores.max():.3f}]")
    print(f"Mean score: {scores.mean():.3f}")
    
    # 测试可见性等级分类
    print("\n--- Test 4: Visibility Classification ---")
    print(f"Visible Car: {score1:.3f} -> {classify_visibility(score1)}")
    print(f"Occluded Car: {score2:.3f} -> {classify_visibility(score2)}")
    
    # 测试OCC3D格式接口
    print("\n--- Test 5: OCC3D Format Interface ---")
    all_bboxes = np.vstack([visible_car_bbox, occluded_car_bbox])
    scores, details, levels = compute_visibility_from_occ3d_format(
        all_bboxes, all_points, voxel_size=0.1
    )
    for i, (s, l) in enumerate(zip(scores, levels)):
        print(f"  Box {i}: score={s:.3f}, level={l}")
    
    print("\n" + "=" * 60)
    print("Test completed successfully!")
    print("=" * 60)
    
    # 打印使用说明
    print("\n" + "=" * 60)
    print("USAGE GUIDE")
    print("=" * 60)
    print("""
# 基本使用
from utils.visibility_calculator import compute_bbox_visibility, warmup

# 预热（可选，减少首次调用延迟）
warmup()

# 单个bbox
bbox = np.array([10, 5, 1, 4.5, 2, 1.5, 0.5])  # [cx, cy, cz, l, w, h, yaw]
score, details = compute_bbox_visibility(bbox, points)

# 批量处理
from utils.visibility_calculator import compute_bboxes_visibility
scores, details_list = compute_bboxes_visibility(bboxes, points)

# OCC3D格式
from utils.visibility_calculator import compute_visibility_from_occ3d_format
scores, details, levels = compute_visibility_from_occ3d_format(bboxes, points)
""")
