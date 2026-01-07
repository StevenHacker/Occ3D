"""
3D标注框可见性计算器 (射线遮挡检测)
====================================

【核心思想】
对框内的每个点，检测从传感器到该点的射线是否被其他物体遮挡。

    传感器 ●─────────────────→ 框内点 ✓ (未遮挡)
    
    传感器 ●────── ✗ ──────→ 框内点   (被遮挡)
                   ↑
                其他点云

【可见性定义】
            未被遮挡的框内点数
可见性 = ─────────────────────
              框内总点数

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
def _points_in_rotated_box(points, cx, cy, cz, hl, hw, hh, cos_y, sin_y):
    """获取框内点的索引"""
    n = points.shape[0]
    r = max(hl, hw) * 1.5
    
    # 先计数
    count = 0
    for i in range(n):
        px, py, pz = points[i,0], points[i,1], points[i,2]
        if abs(px-cx) > r or abs(py-cy) > r or abs(pz-cz) > hh:
            continue
        dx, dy = px-cx, py-cy
        lx = cos_y*dx + sin_y*dy
        ly = -sin_y*dx + cos_y*dy
        if abs(lx) <= hl and abs(ly) <= hw and abs(pz-cz) <= hh:
            count += 1
    
    # 分配并填充
    indices = np.empty(count, dtype=np.int64)
    idx = 0
    for i in range(n):
        px, py, pz = points[i,0], points[i,1], points[i,2]
        if abs(px-cx) > r or abs(py-cy) > r or abs(pz-cz) > hh:
            continue
        dx, dy = px-cx, py-cy
        lx = cos_y*dx + sin_y*dy
        ly = -sin_y*dx + cos_y*dy
        if abs(lx) <= hl and abs(ly) <= hw and abs(pz-cz) <= hh:
            indices[idx] = i
            idx += 1
    
    return indices


@njit(cache=True)
def _check_ray_occlusion(
    target_pt,          # 目标点 (3,)
    sensor,             # 传感器位置 (3,)
    other_points,       # 其他点云 (M, 3)
    occlusion_thresh,   # 遮挡判定阈值
    min_occluders       # 最少遮挡点数（避免单点误判）
):
    """检查从sensor到target_pt的射线是否被other_points遮挡
    
    改进：需要多个点在射线附近才算遮挡，避免单点噪声误判
    
    Returns:
        True: 被遮挡
        False: 未遮挡
    """
    dx = target_pt[0] - sensor[0]
    dy = target_pt[1] - sensor[1]
    dz = target_pt[2] - sensor[2]
    ray_len = np.sqrt(dx*dx + dy*dy + dz*dz)
    
    if ray_len < 1e-6:
        return False
    
    dir_x = dx / ray_len
    dir_y = dy / ray_len
    dir_z = dz / ray_len
    
    thresh_sq = occlusion_thresh * occlusion_thresh
    occluder_count = 0
    
    n = other_points.shape[0]
    for i in range(n):
        px = other_points[i,0] - sensor[0]
        py = other_points[i,1] - sensor[1]
        pz = other_points[i,2] - sensor[2]
        
        # 投影长度
        proj = px*dir_x + py*dir_y + pz*dir_z
        
        # 只考虑在sensor和target之间的点（留margin避免边界问题）
        if proj <= 1.0 or proj >= ray_len - 0.5:
            continue
        
        # 垂直距离
        closest_x = proj * dir_x
        closest_y = proj * dir_y
        closest_z = proj * dir_z
        dist_sq = (px-closest_x)**2 + (py-closest_y)**2 + (pz-closest_z)**2
        
        if dist_sq < thresh_sq:
            occluder_count += 1
            if occluder_count >= min_occluders:
                return True
    
    return False


@njit(cache=True)
def _compute_occlusion_ratio(
    points,             # 所有点云 (N, 3)
    box_indices,        # 框内点索引
    sensor,             # 传感器位置
    occlusion_thresh,   # 遮挡阈值
    min_occluders,      # 最少遮挡点数
    max_check,          # 最大检测点数（降采样）
    max_other           # 最大其他点数（降采样）
):
    """计算框内点的遮挡比例"""
    n_in_box = box_indices.shape[0]
    
    if n_in_box == 0:
        return 0.0, 0, 0
    
    # 降采样框内点
    if n_in_box > max_check:
        step = n_in_box // max_check
        check_count = max_check
    else:
        step = 1
        check_count = n_in_box
    
    # 构建框外点集合（降采样）
    n_total = points.shape[0]
    box_set = set()
    for i in range(n_in_box):
        box_set.add(box_indices[i])
    
    # 收集框外点
    other_count = 0
    for i in range(n_total):
        if i not in box_set:
            other_count += 1
    
    if other_count > max_other:
        other_step = other_count // max_other
    else:
        other_step = 1
    
    other_points = np.empty((min(other_count, max_other), 3), dtype=np.float64)
    idx = 0
    cnt = 0
    for i in range(n_total):
        if i not in box_set:
            if cnt % other_step == 0 and idx < other_points.shape[0]:
                other_points[idx] = points[i]
                idx += 1
            cnt += 1
    other_points = other_points[:idx]
    
    # 检测遮挡
    occluded = 0
    checked = 0
    
    for i in range(0, n_in_box, step):
        if checked >= check_count:
            break
        
        pt_idx = box_indices[i]
        target = points[pt_idx]
        
        if _check_ray_occlusion(target, sensor, other_points, occlusion_thresh, min_occluders):
            occluded += 1
        checked += 1
    
    if checked == 0:
        return 0.0, 0, 0
    
    occlusion_ratio = occluded / checked
    return occlusion_ratio, checked, occluded


def compute_visibility(
    bbox: Union[np.ndarray, List],
    points: np.ndarray,
    sensor_origin: Optional[np.ndarray] = None,
    occlusion_thresh: float = 0.2,
    min_occluders: int = 3,
    max_check_points: int = 50,
    max_other_points: int = 5000
) -> Tuple[float, Dict]:
    """计算3D标注框可见性
    
    【方法】
    对框内的每个点，检测从传感器到该点的射线是否被其他点云遮挡。
    可见性 = 1 - 遮挡比例
    
    Args:
        bbox: [cx, cy, cz, length, width, height, yaw]
        points: (N, 3) 点云
        sensor_origin: (3,) 传感器位置，默认[0,0,0]
        occlusion_thresh: 遮挡判定距离阈值（米）
        min_occluders: 最少遮挡点数（避免单点误判）
        max_check_points: 最大检测点数（框内点降采样）
        max_other_points: 最大其他点数（场景点降采样）
    
    Returns:
        score: 可见性分数 [0, 1]，1=完全可见，0=完全遮挡
        details: 详细信息
    """
    if sensor_origin is None:
        sensor_origin = np.array([0.0, 0.0, 0.0])
    
    bbox = np.asarray(bbox, dtype=np.float64)
    points = np.ascontiguousarray(points[:,:3], dtype=np.float64)
    sensor = np.asarray(sensor_origin, dtype=np.float64)
    
    cx, cy, cz = bbox[0], bbox[1], bbox[2]
    l, w, h, yaw = bbox[3], bbox[4], bbox[5], bbox[6]
    cos_y, sin_y = np.cos(-yaw), np.sin(-yaw)
    
    # 获取框内点
    box_indices = _points_in_rotated_box(
        points, cx, cy, cz, l/2, w/2, h/2, cos_y, sin_y
    )
    n_in_box = box_indices.shape[0]
    
    if n_in_box == 0:
        return 0.0, {
            'score': 0.0,
            'n_points_in_box': 0,
            'n_checked': 0,
            'n_occluded': 0,
            'occlusion_ratio': 1.0,
            'visibility': 0.0,
            'status': 'NO_POINTS'
        }
    
    # 计算遮挡
    occ_ratio, n_checked, n_occluded = _compute_occlusion_ratio(
        points, box_indices, sensor,
        occlusion_thresh, min_occluders, max_check_points, max_other_points
    )
    
    visibility = 1.0 - occ_ratio
    
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
        'n_points_in_box': int(n_in_box),
        'n_checked': int(n_checked),
        'n_occluded': int(n_occluded),
        'occlusion_ratio': float(occ_ratio),
        'visibility': float(visibility),
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
    n = bboxes.shape[0]
    scores = np.zeros(n)
    details = []
    for i in range(n):
        s, d = compute_visibility(bboxes[i], points, sensor_origin, **kwargs)
        scores[i] = s
        details.append(d)
    return scores, details


def classify(score: float) -> str:
    """可见性等级"""
    if score >= 0.7: return "VISIBLE"
    if score >= 0.3: return "PARTIAL"
    if score > 0.05: return "OCCLUDED"
    return "BLOCKED"


# ==================== 测试 ====================
if __name__ == '__main__':
    import time
    
    np.random.seed(42)
    
    print("=" * 60)
    print("射线遮挡检测可见性计算器")
    print("=" * 60)
    
    # 场景：传感器在原点
    sensor = np.array([0, 0, 1.8])
    
    # 背景点云
    bg = np.random.randn(20000, 3)
    bg[:,0] = bg[:,0] * 30
    bg[:,1] = bg[:,1] * 30
    bg[:,2] = np.abs(bg[:,2]) * 1.5 - 0.5
    
    # 目标1：前方可见车辆（有表面点云）
    car1_center = [15, 3, 0.8]
    car1_size = [4.5, 2, 1.5]
    car1_yaw = 0.1
    
    # 在车辆表面生成点
    car1_pts = []
    for _ in range(200):
        face = np.random.randint(6)
        hs = np.array(car1_size) / 2
        if face == 0: p = [hs[0], np.random.uniform(-1,1)*hs[1], np.random.uniform(-1,1)*hs[2]]
        elif face == 1: p = [-hs[0], np.random.uniform(-1,1)*hs[1], np.random.uniform(-1,1)*hs[2]]
        elif face == 2: p = [np.random.uniform(-1,1)*hs[0], hs[1], np.random.uniform(-1,1)*hs[2]]
        elif face == 3: p = [np.random.uniform(-1,1)*hs[0], -hs[1], np.random.uniform(-1,1)*hs[2]]
        elif face == 4: p = [np.random.uniform(-1,1)*hs[0], np.random.uniform(-1,1)*hs[1], hs[2]]
        else: p = [np.random.uniform(-1,1)*hs[0], np.random.uniform(-1,1)*hs[1], -hs[2]]
        # 旋转平移
        c, s = np.cos(car1_yaw), np.sin(car1_yaw)
        rx = c*p[0] - s*p[1] + car1_center[0]
        ry = s*p[0] + c*p[1] + car1_center[1]
        rz = p[2] + car1_center[2]
        car1_pts.append([rx, ry, rz])
    car1_pts = np.array(car1_pts)
    
    # 目标2：被遮挡的车辆（前方有遮挡物）
    car2_center = [25, -5, 0.8]
    car2_size = [4.5, 2, 1.5]
    car2_yaw = -0.2
    
    # 车辆表面点
    car2_pts = []
    for _ in range(200):
        face = np.random.randint(6)
        hs = np.array(car2_size) / 2
        if face == 0: p = [hs[0], np.random.uniform(-1,1)*hs[1], np.random.uniform(-1,1)*hs[2]]
        elif face == 1: p = [-hs[0], np.random.uniform(-1,1)*hs[1], np.random.uniform(-1,1)*hs[2]]
        elif face == 2: p = [np.random.uniform(-1,1)*hs[0], hs[1], np.random.uniform(-1,1)*hs[2]]
        elif face == 3: p = [np.random.uniform(-1,1)*hs[0], -hs[1], np.random.uniform(-1,1)*hs[2]]
        elif face == 4: p = [np.random.uniform(-1,1)*hs[0], np.random.uniform(-1,1)*hs[1], hs[2]]
        else: p = [np.random.uniform(-1,1)*hs[0], np.random.uniform(-1,1)*hs[1], -hs[2]]
        c, s = np.cos(car2_yaw), np.sin(car2_yaw)
        rx = c*p[0] - s*p[1] + car2_center[0]
        ry = s*p[0] + c*p[1] + car2_center[1]
        rz = p[2] + car2_center[2]
        car2_pts.append([rx, ry, rz])
    car2_pts = np.array(car2_pts)
    
    # 遮挡物：在car2前方放置密集点云（墙状遮挡）
    blocker = []
    for x in np.linspace(18, 20, 20):
        for y in np.linspace(-7, -3, 30):
            for z in np.linspace(0, 2, 15):
                blocker.append([x, y, z])
    blocker = np.array(blocker) + np.random.randn(len(blocker), 3) * 0.05
    
    # 合并点云
    all_points = np.vstack([bg, car1_pts, car2_pts, blocker])
    print(f"总点云: {len(all_points)}")
    
    # 预热
    bbox1 = [*car1_center, *car1_size, car1_yaw]
    compute_visibility(bbox1, all_points, sensor)
    
    # 测试1：可见车辆
    print("\n" + "-" * 40)
    print("测试1: 可见车辆 (无遮挡)")
    print("-" * 40)
    
    t0 = time.time()
    score1, d1 = compute_visibility(bbox1, all_points, sensor)
    t1 = time.time()
    
    print(f"可见性: {score1:.2f} ({d1['status']})")
    print(f"框内点: {d1['n_points_in_box']}")
    print(f"检测点: {d1['n_checked']}")
    print(f"遮挡点: {d1['n_occluded']}")
    print(f"耗时: {(t1-t0)*1000:.1f} ms")
    
    # 测试2：被遮挡车辆
    print("\n" + "-" * 40)
    print("测试2: 被遮挡车辆 (前方有遮挡物)")
    print("-" * 40)
    
    bbox2 = [*car2_center, *car2_size, car2_yaw]
    t0 = time.time()
    score2, d2 = compute_visibility(bbox2, all_points, sensor)
    t1 = time.time()
    
    print(f"可见性: {score2:.2f} ({d2['status']})")
    print(f"框内点: {d2['n_points_in_box']}")
    print(f"检测点: {d2['n_checked']}")
    print(f"遮挡点: {d2['n_occluded']}")
    print(f"耗时: {(t1-t0)*1000:.1f} ms")
    
    # 测试3：远处车辆（点少但无遮挡）
    print("\n" + "-" * 40)
    print("测试3: 远处车辆 (点少但无遮挡)")
    print("-" * 40)
    
    # 远处车辆，只有少量点
    far_center = [80, 10, 0.8]
    far_pts = np.random.randn(30, 3) * 0.3
    far_pts[:,0] += far_center[0]
    far_pts[:,1] += far_center[1]
    far_pts[:,2] += far_center[2]
    
    all_points2 = np.vstack([all_points, far_pts])
    bbox3 = [*far_center, 4.5, 2, 1.5, 0]
    
    t0 = time.time()
    score3, d3 = compute_visibility(bbox3, all_points2, sensor)
    t1 = time.time()
    
    print(f"可见性: {score3:.2f} ({d3['status']})")
    print(f"框内点: {d3['n_points_in_box']}")
    print(f"检测点: {d3['n_checked']}")
    print(f"遮挡点: {d3['n_occluded']}")
    print(f"耗时: {(t1-t0)*1000:.1f} ms")
    
    # 批量性能测试
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
