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

【bbox格式支持】
- 7参数: [cx, cy, cz, length, width, height, yaw]
- 8参数: [cx, cy, cz, length, width, height, yaw, pitch]
- 9参数: [cx, cy, cz, length, width, height, yaw, pitch, roll]

【坐标系约定】
- x: 前方 (车头方向)
- y: 左方
- z: 上方
- yaw (psi, ψ): 偏航角，绕z轴旋转，从x轴正向逆时针为正
- pitch (theta, θ): 俯仰角，绕y轴旋转
- roll (phi, φ): 横滚角，绕x轴旋转

【使用】
from bbox_visibility import compute_visibility
score, details = compute_visibility(bbox, points, sensor_origin)
"""

import numpy as np
from typing import Tuple, Dict, List, Union, Optional

# ============================================================
# Numba JIT 编译支持（可选，用于加速）
# ============================================================
try:
    from numba import njit
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    # 如果没有numba，定义一个空装饰器
    def njit(*args, **kwargs):
        def decorator(func): return func
        return decorator if not (args and callable(args[0])) else args[0]


# ============================================================
# 3D旋转矩阵计算
# ============================================================
@njit(cache=True)
def _rotation_matrix(yaw: float, pitch: float, roll: float) -> np.ndarray:
    """
    计算3D旋转矩阵 (ZYX顺序: 先roll, 再pitch, 最后yaw)
    
    旋转顺序说明:
    - 这是常用的"外旋"(extrinsic)顺序
    - 等价于: R = Rz(yaw) @ Ry(pitch) @ Rx(roll)
    
    Args:
        yaw: 偏航角(psi), 绕z轴, 弧度
        pitch: 俯仰角(theta), 绕y轴, 弧度
        roll: 横滚角(phi), 绕x轴, 弧度
    
    Returns:
        3x3旋转矩阵, 用于将局部坐标转换到世界坐标
    """
    # 预计算三角函数值
    cy, sy = np.cos(yaw), np.sin(yaw)      # c=cos, s=sin, y=yaw
    cp, sp = np.cos(pitch), np.sin(pitch)  # p=pitch
    cr, sr = np.cos(roll), np.sin(roll)    # r=roll
    
    # 构建旋转矩阵 R = Rz(yaw) @ Ry(pitch) @ Rx(roll)
    # 这里直接写出展开后的结果，避免矩阵乘法
    R = np.empty((3, 3), dtype=np.float64)
    
    R[0, 0] = cy * cp
    R[0, 1] = cy * sp * sr - sy * cr
    R[0, 2] = cy * sp * cr + sy * sr
    
    R[1, 0] = sy * cp
    R[1, 1] = sy * sp * sr + cy * cr
    R[1, 2] = sy * sp * cr - cy * sr
    
    R[2, 0] = -sp
    R[2, 1] = cp * sr
    R[2, 2] = cp * cr
    
    return R


@njit(cache=True)
def _rotation_matrix_transpose(yaw: float, pitch: float, roll: float) -> np.ndarray:
    """
    计算旋转矩阵的转置 (逆矩阵)
    
    用于将世界坐标转换到局部坐标: local = R^T @ (world - center)
    """
    # 对于旋转矩阵, 转置 = 逆
    cy, sy = np.cos(yaw), np.sin(yaw)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cr, sr = np.cos(roll), np.sin(roll)
    
    R_T = np.empty((3, 3), dtype=np.float64)
    
    # R^T 的元素就是R的行列互换
    R_T[0, 0] = cy * cp
    R_T[0, 1] = sy * cp
    R_T[0, 2] = -sp
    
    R_T[1, 0] = cy * sp * sr - sy * cr
    R_T[1, 1] = sy * sp * sr + cy * cr
    R_T[1, 2] = cp * sr
    
    R_T[2, 0] = cy * sp * cr + sy * sr
    R_T[2, 1] = sy * sp * cr - cy * sr
    R_T[2, 2] = cp * cr
    
    return R_T


# ============================================================
# 可见面判断与表面采样
# ============================================================
@njit(cache=True)
def _determine_visible_faces(
    sensor_local: np.ndarray,
    exclude_bottom: bool
) -> np.ndarray:
    """
    判断bbox的哪些面朝向传感器
    
    【原理】
    一个面"可见"的条件: 面的法向量与"面中心→传感器"向量的夹角 < 90°
    等价于: 法向量 · (传感器位置 - 面中心) > 0
    
    在bbox局部坐标系中:
    - +x面: 法向量(1,0,0), 面中心在+x侧
    - -x面: 法向量(-1,0,0), 面中心在-x侧
    - 以此类推...
    
    简化判断: 如果传感器在+x侧(sensor_local[0] > 0), 则+x面可见
    
    【车辆常见情况】
    - 传感器在车的右前方: +x面(前)和-y面(右)可见 → 2个面
    - 传感器在车的正前方: 只有+x面(前)可见 → 1个面
    - 传感器在车的上方时: 顶面也可见 → 可能3个面
    
    Args:
        sensor_local: 传感器在bbox局部坐标系中的位置 (3,)
        exclude_bottom: 是否排除底面(-z面), 车辆通常看不到底面
    
    Returns:
        长度为6的布尔数组, 表示6个面是否可见
        顺序: [+x, -x, +y, -y, +z, -z]
    """
    # 传感器在各轴的位置决定了哪些面可见
    sx, sy, sz = sensor_local[0], sensor_local[1], sensor_local[2]
    
    visible = np.array([
        sx > 0,   # +x面(前): 传感器在前方才可见
        sx < 0,   # -x面(后): 传感器在后方才可见
        sy > 0,   # +y面(左): 传感器在左侧才可见
        sy < 0,   # -y面(右): 传感器在右侧才可见
        sz > 0,   # +z面(顶): 传感器在上方才可见
        sz < 0,   # -z面(底): 传感器在下方才可见
    ])
    
    # 底面通常被地面遮挡，可以选择排除
    if exclude_bottom:
        visible[5] = False
    
    return visible


@njit(cache=True)
def _compute_face_areas(half_l: float, half_w: float, half_h: float) -> np.ndarray:
    """
    计算bbox六个面的面积
    
    【各面尺寸】
    - ±x面 (前后): 宽×高 = (2*half_w) × (2*half_h)
    - ±y面 (左右): 长×高 = (2*half_l) × (2*half_h)  
    - ±z面 (顶底): 长×宽 = (2*half_l) × (2*half_w)
    
    Returns:
        (6,) 数组，各面面积
    """
    areas = np.empty(6, dtype=np.float64)
    
    area_x = (2 * half_w) * (2 * half_h)  # 前后面 (宽×高)
    area_y = (2 * half_l) * (2 * half_h)  # 左右面 (长×高)
    area_z = (2 * half_l) * (2 * half_w)  # 顶底面 (长×宽)
    
    areas[0] = area_x  # +x (前)
    areas[1] = area_x  # -x (后)
    areas[2] = area_y  # +y (左)
    areas[3] = area_y  # -y (右)
    areas[4] = area_z  # +z (顶)
    areas[5] = area_z  # -z (底)
    
    return areas


@njit(cache=True)
def _sample_single_face_adaptive(
    face_id: int,
    half_l: float, half_w: float, half_h: float,
    n_samples: int
) -> np.ndarray:
    """
    在单个面上自适应采样（根据面的宽高比调整网格）
    
    【改进】
    不是简单的 sqrt(n) × sqrt(n) 网格，
    而是根据面的宽高比分配行列数，使采样点更均匀。
    
    例如: 侧面 4.5m × 1.5m，比例 3:1
         如果要采样 24 个点，用 6×4 而不是 5×5
    """
    # 根据面的编号确定该面的两个维度
    if face_id == 0 or face_id == 1:    # ±x面: y方向×z方向
        dim1 = half_w  # y
        dim2 = half_h  # z
    elif face_id == 2 or face_id == 3:  # ±y面: x方向×z方向
        dim1 = half_l  # x
        dim2 = half_h  # z
    else:                                # ±z面: x方向×y方向
        dim1 = half_l  # x
        dim2 = half_w  # y
    
    # 计算宽高比，决定网格的行列数
    # 目标: n1 * n2 ≈ n_samples, 且 n1/n2 ≈ dim1/dim2
    ratio = dim1 / dim2 if dim2 > 0.01 else 1.0
    
    # n1 * n2 = n_samples
    # n1 / n2 = ratio
    # => n1 = sqrt(n_samples * ratio), n2 = sqrt(n_samples / ratio)
    n1 = int(np.sqrt(n_samples * ratio) + 0.5)
    n2 = int(np.sqrt(n_samples / ratio) + 0.5)
    
    # 确保至少1个
    if n1 < 1: n1 = 1
    if n2 < 1: n2 = 1
    
    actual_samples = n1 * n2
    samples = np.empty((actual_samples, 3), dtype=np.float64)
    
    idx = 0
    for i in range(n1):
        for j in range(n2):
            # 在[-1, 1]区间均匀采样
            u = (i + 0.5) / n1 * 2 - 1
            v = (j + 0.5) / n2 * 2 - 1
            
            # 根据面的编号确定采样点坐标
            if face_id == 0:    # +x面 (前)
                lx, ly, lz = half_l, u * half_w, v * half_h
            elif face_id == 1:  # -x面 (后)
                lx, ly, lz = -half_l, u * half_w, v * half_h
            elif face_id == 2:  # +y面 (左)
                lx, ly, lz = u * half_l, half_w, v * half_h
            elif face_id == 3:  # -y面 (右)
                lx, ly, lz = u * half_l, -half_w, v * half_h
            elif face_id == 4:  # +z面 (顶)
                lx, ly, lz = u * half_l, v * half_w, half_h
            else:               # -z面 (底)
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
    """
    在所有可见面上采样（按面积分配采样数）
    
    【流程】
    1. 计算传感器在bbox局部坐标系中的位置
    2. 判断哪些面朝向传感器
    3. 按面积比例分配采样数（大面多采样，小面少采样）
    4. 在每个可见面上均匀采样
    5. 将采样点从局部坐标转换到世界坐标
    
    【面积自适应采样】
    例如车辆 4.5m × 2m × 1.5m:
    - 侧面 6.75m² → 分配更多采样点
    - 前面 3.0m²  → 分配较少采样点
    - 顶面 9.0m²  → 分配最多采样点
    
    Args:
        center: bbox中心 (3,)
        half_dims: [half_length, half_width, half_height]
        R: 旋转矩阵 (3,3), 用于局部→世界
        sensor: 传感器世界坐标 (3,)
        total_samples: 总采样数（会按面积分配到各面）
        exclude_bottom: 是否排除底面
    
    Returns:
        samples: 世界坐标下的采样点 (M, 3)
        face_ids: 每个采样点所属的面编号 (M,)
    """
    half_l, half_w, half_h = half_dims[0], half_dims[1], half_dims[2]
    
    # -------- 步骤1: 计算传感器在局部坐标系的位置 --------
    diff = np.empty(3, dtype=np.float64)
    diff[0] = sensor[0] - center[0]
    diff[1] = sensor[1] - center[1]
    diff[2] = sensor[2] - center[2]
    
    # sensor_local = R^T @ diff
    sensor_local = np.empty(3, dtype=np.float64)
    for i in range(3):
        sensor_local[i] = R[0, i] * diff[0] + R[1, i] * diff[1] + R[2, i] * diff[2]
    
    # -------- 步骤2: 判断可见面 --------
    visible_faces = _determine_visible_faces(sensor_local, exclude_bottom)
    
    # -------- 步骤3: 计算各面面积，按比例分配采样数 --------
    areas = _compute_face_areas(half_l, half_w, half_h)
    
    # 计算可见面的总面积
    total_visible_area = 0.0
    for i in range(6):
        if visible_faces[i]:
            total_visible_area += areas[i]
    
    if total_visible_area < 1e-6:
        return np.empty((0, 3), dtype=np.float64), np.empty(0, dtype=np.int64)
    
    # 按面积比例分配采样数
    samples_per_face = np.zeros(6, dtype=np.int64)
    for i in range(6):
        if visible_faces[i]:
            # 按面积比例分配，至少4个点
            n = int(total_samples * areas[i] / total_visible_area + 0.5)
            samples_per_face[i] = max(n, 4)
    
    # -------- 步骤4: 在每个可见面采样 --------
    # 计算总采样数
    actual_total = 0
    for i in range(6):
        actual_total += samples_per_face[i]
    
    samples_world = np.empty((actual_total, 3), dtype=np.float64)
    face_ids = np.empty(actual_total, dtype=np.int64)
    
    write_idx = 0
    for face_id in range(6):
        if samples_per_face[face_id] == 0:
            continue
        
        # 在该面上自适应采样（考虑面的宽高比）
        local_samples = _sample_single_face_adaptive(
            face_id, half_l, half_w, half_h, samples_per_face[face_id]
        )
        
        # 转换到世界坐标: world = R @ local + center
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


# ============================================================
# 射线遮挡检测
# ============================================================
@njit(cache=True)
def _check_ray_blocked(
    target: np.ndarray,
    sensor: np.ndarray,
    points: np.ndarray,
    angle_thresh: float,
    min_blockers: int
) -> bool:
    """
    检查从传感器到目标点的射线是否被其他点云遮挡
    
    【原理】
    1. 计算从sensor到target的射线方向
    2. 对于场景中的每个点，检查它是否"挡住"了射线
    3. 判断标准: 点到射线的垂直距离 < 动态阈值
    
    【动态阈值(角度阈值)的意义】
    LiDAR点云的特性是"近密远疏"，使用固定的距离阈值会导致:
    - 近处误判(阈值太大)
    - 远处漏判(阈值太小)
    
    使用角度阈值: threshold = distance * tan(angle_thresh)
    这样阈值会随距离自适应调整:
    - 近处(10m): thresh ≈ 0.09m
    - 远处(50m): thresh ≈ 0.44m
    
    【为什么需要min_blockers?】
    避免单个噪声点造成误判。要求至少min_blockers个点
    都在射线附近，才认为射线被遮挡。
    
    Args:
        target: 目标点（表面采样点）世界坐标 (3,)
        sensor: 传感器世界坐标 (3,)
        points: 场景点云 (N, 3)
        angle_thresh: 角度阈值（弧度），推荐0.5°≈0.0087rad
        min_blockers: 最少需要多少个点才认为被遮挡，推荐3
    
    Returns:
        True: 射线被遮挡
        False: 射线畅通
    """
    # -------- 计算射线方向 --------
    dx = target[0] - sensor[0]
    dy = target[1] - sensor[1]
    dz = target[2] - sensor[2]
    ray_len = np.sqrt(dx*dx + dy*dy + dz*dz)
    
    if ray_len < 1e-6:
        return False  # 目标点和传感器重合，不可能被遮挡
    
    # 单位方向向量
    dir_x = dx / ray_len
    dir_y = dy / ray_len
    dir_z = dz / ray_len
    
    # 预计算tan(angle_thresh)
    tan_thresh = np.tan(angle_thresh)
    
    # -------- 遍历场景点云，检查遮挡 --------
    blocker_count = 0
    n = points.shape[0]
    
    for i in range(n):
        # 计算点相对于传感器的向量
        px = points[i, 0] - sensor[0]
        py = points[i, 1] - sensor[1]
        pz = points[i, 2] - sensor[2]
        
        # 计算点在射线方向上的投影距离
        # proj = (p · dir) = 点到sensor的距离在射线方向上的分量
        proj = px * dir_x + py * dir_y + pz * dir_z
        
        # 只考虑在sensor和target之间的点
        # proj > 0.5: 排除传感器附近的点（可能是噪声）
        # proj < ray_len - 0.3: 排除目标点附近的点（可能是目标物体本身）
        if proj <= 0.5 or proj >= ray_len - 0.3:
            continue
        
        # 计算点到射线的垂直距离
        # 射线上最近点: sensor + proj * dir
        # 垂直距离 = |point - 最近点|
        closest_x = proj * dir_x
        closest_y = proj * dir_y
        closest_z = proj * dir_z
        
        perp_dist_sq = (px - closest_x)**2 + (py - closest_y)**2 + (pz - closest_z)**2
        
        # 动态阈值: 距离越远，允许的偏差越大
        # thresh = proj * tan(angle_thresh)
        dynamic_thresh = proj * tan_thresh
        
        # 判断是否在射线"锥体"内
        if perp_dist_sq < dynamic_thresh * dynamic_thresh:
            blocker_count += 1
            # 达到最小遮挡点数，确认被遮挡
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
    """
    计算所有采样点的可见性，并统计每个面的遮挡情况
    
    Args:
        sample_points: 表面采样点 (M, 3)
        face_ids: 每个采样点所属的面 (M,)
        sensor: 传感器位置 (3,)
        scene_points: 场景点云 (N, 3)
        angle_thresh: 角度阈值（弧度）
        min_blockers: 最少遮挡点数
    
    Returns:
        n_total: 总采样点数
        n_blocked: 被遮挡的采样点数
        face_stats: (6, 2) 每个面的[总数, 被挡数]
    """
    n_samples = sample_points.shape[0]
    
    if n_samples == 0:
        return 0, 0, np.zeros((6, 2), dtype=np.int64)
    
    # 统计每个面的情况
    face_stats = np.zeros((6, 2), dtype=np.int64)  # [total, blocked]
    n_blocked = 0
    
    for i in range(n_samples):
        target = sample_points[i]
        face_id = face_ids[i]
        
        # 更新该面的总采样数
        face_stats[face_id, 0] += 1
        
        # 检查是否被遮挡
        if _check_ray_blocked(target, sensor, scene_points, angle_thresh, min_blockers):
            n_blocked += 1
            face_stats[face_id, 1] += 1
    
    return n_samples, n_blocked, face_stats


# ============================================================
# 主接口函数
# ============================================================
def compute_visibility(
    bbox: Union[np.ndarray, List],
    points: np.ndarray,
    sensor_origin: Optional[np.ndarray] = None,
    total_samples: int = 80,
    angle_thresh_deg: float = 0.5,
    min_blockers: int = 3,
    max_scene_points: int = 15000,
    exclude_bottom: bool = True
) -> Tuple[float, Dict]:
    """
    计算3D标注框的可见性
    
    【算法流程】
    1. 解析bbox参数（支持7/8/9参数格式）
    2. 在bbox朝向传感器的表面均匀采样（按面积分配）
    3. 对每个采样点，检测射线是否被其他点云遮挡
    4. 可见性 = 未被遮挡的采样点数 / 总采样点数
    
    Args:
        bbox: 3D标注框，支持以下格式:
            - 7参数: [cx, cy, cz, length, width, height, yaw]
            - 8参数: [cx, cy, cz, length, width, height, yaw, pitch]
            - 9参数: [cx, cy, cz, length, width, height, yaw, pitch, roll]
        points: (N, 3) 场景点云
        sensor_origin: (3,) 传感器位置，默认原点
        total_samples: 总采样点数，会按面积比例分配到各可见面，默认80
        angle_thresh_deg: 射线角度阈值（度），默认0.5°
        min_blockers: 最少遮挡点数，默认3
        max_scene_points: 场景点云最大采样数，默认15000
        exclude_bottom: 是否排除底面，默认True（车辆底面通常不可见）
    
    Returns:
        score: 可见性分数 [0, 1]
        details: 详细信息字典，包含:
            - score: 可见性分数
            - n_samples: 总采样点数
            - n_blocked: 被遮挡点数
            - n_visible: 可见点数
            - status: 状态字符串
            - face_visibility: 每个面的可见性 {面名: 可见性}
            - n_visible_faces: 可见面数量
    """
    # -------- 参数预处理 --------
    if sensor_origin is None:
        sensor_origin = np.array([0.0, 0.0, 0.0])
    
    bbox = np.asarray(bbox, dtype=np.float64)
    points = np.ascontiguousarray(points[:, :3], dtype=np.float64)
    sensor = np.asarray(sensor_origin, dtype=np.float64)
    
    # -------- 解析bbox参数 --------
    cx, cy, cz = bbox[0], bbox[1], bbox[2]
    length, width, height = bbox[3], bbox[4], bbox[5]
    
    # 解析旋转角度（支持7/8/9参数）
    yaw = bbox[6] if len(bbox) > 6 else 0.0
    pitch = bbox[7] if len(bbox) > 7 else 0.0  # theta
    roll = bbox[8] if len(bbox) > 8 else 0.0   # phi
    
    # 转换为弧度（如果需要）
    angle_thresh = np.deg2rad(angle_thresh_deg)
    
    # 计算旋转矩阵
    R = _rotation_matrix(yaw, pitch, roll)
    
    center = np.array([cx, cy, cz], dtype=np.float64)
    half_dims = np.array([length/2, width/2, height/2], dtype=np.float64)
    
    # -------- 表面采样（按面积分配） --------
    samples, face_ids = _sample_visible_surfaces(
        center, half_dims, R, sensor,
        total_samples, exclude_bottom
    )
    
    n_samples = samples.shape[0]
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
    
    # -------- 场景点云降采样 --------
    # 使用随机采样，确保不同区域的点都有机会被选中
    if points.shape[0] > max_scene_points:
        indices = np.random.choice(points.shape[0], max_scene_points, replace=False)
        scene_pts = points[indices]
    else:
        scene_pts = points
    
    # -------- 计算遮挡 --------
    n_total, n_blocked, face_stats = _compute_surface_visibility(
        samples, face_ids, sensor, scene_pts, angle_thresh, min_blockers
    )
    
    n_visible = n_total - n_blocked
    visibility = n_visible / n_total if n_total > 0 else 0.0
    
    # -------- 计算每个面的可见性 --------
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
    
    # -------- 状态判断 --------
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


def compute_visibility_batch(
    bboxes: np.ndarray,
    points: np.ndarray,
    sensor_origin: Optional[np.ndarray] = None,
    **kwargs
) -> Tuple[np.ndarray, List[Dict]]:
    """
    批量计算多个bbox的可见性
    
    Args:
        bboxes: (N, 7/8/9) 多个bbox
        points: (M, 3) 场景点云
        sensor_origin: (3,) 传感器位置
        **kwargs: 传递给compute_visibility的其他参数
    
    Returns:
        scores: (N,) 每个bbox的可见性分数
        details: 每个bbox的详细信息列表
    """
    bboxes = np.asarray(bboxes, dtype=np.float64)
    scores = np.zeros(bboxes.shape[0])
    details = []
    
    for i, bbox in enumerate(bboxes):
        s, d = compute_visibility(bbox, points, sensor_origin, **kwargs)
        scores[i] = s
        details.append(d)
    
    return scores, details


def classify(score: float) -> str:
    """
    将可见性分数分类为状态标签
    
    分类标准:
    - VISIBLE (≥0.7): 大部分可见，可以正常标注
    - PARTIAL (0.3-0.7): 部分可见，需要谨慎
    - OCCLUDED (0.05-0.3): 大部分被遮挡
    - BLOCKED (<0.05): 几乎完全不可见
    """
    if score >= 0.7:
        return "VISIBLE"
    if score >= 0.3:
        return "PARTIAL"
    if score > 0.05:
        return "OCCLUDED"
    return "BLOCKED"


# ============================================================
# 测试代码
# ============================================================
if __name__ == '__main__':
    import time
    
    np.random.seed(42)
    
    print("=" * 70)
    print("3D标注框可见性计算器 - 表面采样 + 射线遮挡检测")
    print("=" * 70)
    
    # 传感器位置（假设在车顶）
    sensor = np.array([0, 0, 1.8])
    
    # 创建背景点云
    bg = np.random.randn(20000, 3)
    bg[:, 0] *= 30
    bg[:, 1] *= 30
    bg[:, 2] = np.abs(bg[:, 2]) * 1.5 - 0.5
    
    # 测试场景1: 可见车辆（无遮挡）
    # 车在右前方，yaw=0.1表示略微旋转
    car1 = [15, 3, 0.8, 4.5, 2, 1.5, 0.1]
    
    # 测试场景2: 被遮挡车辆
    # 车在前方偏左，前方有墙
    car2 = [25, -5, 0.8, 4.5, 2, 1.5, -0.2]
    
    # 创建遮挡物（墙）
    blocker = []
    for x in np.linspace(18, 20, 20):
        for y in np.linspace(-7, -3, 30):
            for z in np.linspace(0, 2, 15):
                blocker.append([x, y, z])
    blocker = np.array(blocker) + np.random.randn(len(blocker), 3) * 0.05
    
    # 测试场景3: 远处车辆（无遮挡，测试距离适应性）
    car3 = [80, 10, 0.8, 4.5, 2, 1.5, 0]
    
    # 测试场景4: 带俯仰角的车辆（测试pitch支持）
    # 8参数格式: [cx, cy, cz, l, w, h, yaw, pitch]
    car4 = [20, 0, 1.5, 4.5, 2, 1.5, 0.3, 0.1]
    
    all_points = np.vstack([bg, blocker])
    print(f"\n场景点云: {len(all_points)} 点")
    print(f"传感器位置: {sensor}")
    
    # 预热（JIT编译）
    compute_visibility(car1, all_points, sensor)
    
    # -------- 测试1: 可见车辆 --------
    print("\n" + "-" * 50)
    print("测试1: 可见车辆 (右前方, 无遮挡)")
    print("-" * 50)
    t0 = time.time()
    s1, d1 = compute_visibility(car1, all_points, sensor)
    t1 = time.time()
    print(f"可见性: {s1:.2f} ({d1['status']})")
    print(f"可见面数: {d1['n_visible_faces']}")
    print(f"采样点: {d1['n_samples']}, 被挡: {d1['n_blocked']}, 可见: {d1['n_visible']}")
    print(f"各面可见性:")
    for face, info in d1['face_visibility'].items():
        print(f"  {face}: {info['visibility']:.2f} ({info['total']-info['blocked']}/{info['total']})")
    print(f"耗时: {(t1-t0)*1000:.1f} ms")
    
    # -------- 测试2: 被遮挡车辆 --------
    print("\n" + "-" * 50)
    print("测试2: 被遮挡车辆 (前方有墙)")
    print("-" * 50)
    t0 = time.time()
    s2, d2 = compute_visibility(car2, all_points, sensor)
    t1 = time.time()
    print(f"可见性: {s2:.2f} ({d2['status']})")
    print(f"可见面数: {d2['n_visible_faces']}")
    print(f"采样点: {d2['n_samples']}, 被挡: {d2['n_blocked']}, 可见: {d2['n_visible']}")
    print(f"各面可见性:")
    for face, info in d2['face_visibility'].items():
        print(f"  {face}: {info['visibility']:.2f} ({info['total']-info['blocked']}/{info['total']})")
    print(f"耗时: {(t1-t0)*1000:.1f} ms")
    
    # -------- 测试3: 远处车辆 --------
    print("\n" + "-" * 50)
    print("测试3: 远处车辆 (80m, 测试角度阈值的距离适应性)")
    print("-" * 50)
    t0 = time.time()
    s3, d3 = compute_visibility(car3, all_points, sensor)
    t1 = time.time()
    print(f"可见性: {s3:.2f} ({d3['status']})")
    print(f"可见面数: {d3['n_visible_faces']}")
    print(f"采样点: {d3['n_samples']}, 被挡: {d3['n_blocked']}, 可见: {d3['n_visible']}")
    print(f"耗时: {(t1-t0)*1000:.1f} ms")
    
    # -------- 测试4: 带俯仰角的车辆 --------
    print("\n" + "-" * 50)
    print("测试4: 带俯仰角的车辆 (8参数格式, yaw=0.3, pitch=0.1)")
    print("-" * 50)
    t0 = time.time()
    s4, d4 = compute_visibility(car4, all_points, sensor)
    t1 = time.time()
    print(f"可见性: {s4:.2f} ({d4['status']})")
    print(f"可见面数: {d4['n_visible_faces']}")
    print(f"各面可见性:")
    for face, info in d4['face_visibility'].items():
        print(f"  {face}: {info['visibility']:.2f}")
    print(f"耗时: {(t1-t0)*1000:.1f} ms")
    
    # -------- 批量性能测试 --------
    print("\n" + "-" * 50)
    print("批量性能测试")
    print("-" * 50)
    n_boxes = 50
    boxes = np.zeros((n_boxes, 7))
    boxes[:, :2] = np.random.uniform(-40, 40, (n_boxes, 2))
    boxes[:, 2] = np.random.uniform(0, 2, n_boxes)
    boxes[:, 3:6] = [4.5, 2, 1.5]
    boxes[:, 6] = np.random.uniform(-np.pi, np.pi, n_boxes)
    
    t0 = time.time()
    scores, _ = compute_visibility_batch(boxes, all_points, sensor)
    t1 = time.time()
    
    print(f"bbox数量: {n_boxes}")
    print(f"点云数量: {len(all_points)}")
    print(f"总耗时: {(t1-t0)*1000:.1f} ms")
    print(f"每框耗时: {(t1-t0)/n_boxes*1000:.2f} ms")
    print(f"可见性分布: VISIBLE={sum(scores>=0.7)}, PARTIAL={sum((scores>=0.3)&(scores<0.7))}, OCCLUDED={sum((scores>0.05)&(scores<0.3))}, BLOCKED={sum(scores<=0.05)}")
    
    print("\n" + "=" * 70)
