"""
3D标注框可见性计算器
====================

【使用】
from bbox_visibility import compute_visibility, compute_visibility_batch

score, details = compute_visibility(bbox, points)
scores, details = compute_visibility_batch(bboxes, points)

bbox格式: [cx, cy, cz, length, width, height, yaw]
"""

import numpy as np
from typing import Tuple, Dict, List, Union

try:
    from numba import njit
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    def njit(*args, **kwargs):
        def decorator(func): return func
        return decorator if not (args and callable(args[0])) else args[0]


@njit(cache=True)
def _process_bbox(points, cx, cy, cz, hl, hw, hh, cos_y, sin_y):
    """处理单个bbox：筛选点、统计表面"""
    n = points.shape[0]
    r = max(hl, hw) * 1.5  # AABB半径
    
    count = 0
    surf = np.zeros(6, dtype=np.int64)
    thresh = 0.2
    tl, tw, th = hl*thresh, hw*thresh, hh*thresh
    
    for i in range(n):
        px, py, pz = points[i,0], points[i,1], points[i,2]
        
        # AABB快速过滤
        if abs(px-cx) > r or abs(py-cy) > r or abs(pz-cz) > hh:
            continue
        
        # 旋转到局部坐标
        dx, dy = px - cx, py - cy
        lx = cos_y*dx + sin_y*dy
        ly = -sin_y*dx + cos_y*dy
        lz = pz - cz
        
        # 精确判断
        if abs(lx) <= hl and abs(ly) <= hw and abs(lz) <= hh:
            count += 1
            # 统计表面
            if lx > hl - tl: surf[0] += 1
            if lx < -hl + tl: surf[1] += 1
            if ly > hw - tw: surf[2] += 1
            if ly < -hw + tw: surf[3] += 1
            if lz > hh - th: surf[4] += 1
            if lz < -hh + th: surf[5] += 1
    
    return count, surf


def compute_visibility(
    bbox: Union[np.ndarray, List],
    points: np.ndarray,
    density_ref: float = 50.0
) -> Tuple[float, Dict]:
    """计算单个标注框可见性
    
    Args:
        bbox: [cx, cy, cz, l, w, h, yaw]
        points: (N,3) 点云
        density_ref: 参考密度(点/m³)
    
    Returns:
        score: [0,1]
        details: 详细信息
    """
    bbox = np.asarray(bbox, dtype=np.float64)
    points = np.ascontiguousarray(points[:,:3], dtype=np.float64)
    
    cx, cy, cz = bbox[0], bbox[1], bbox[2]
    l, w, h, yaw = bbox[3], bbox[4], bbox[5], bbox[6]
    hl, hw, hh = l/2, w/2, h/2
    cos_y, sin_y = np.cos(-yaw), np.sin(-yaw)
    
    # 核心计算
    n_pts, surf = _process_bbox(points, cx, cy, cz, hl, hw, hh, cos_y, sin_y)
    
    # 点密度评分
    volume = l * w * h
    density = n_pts / max(volume, 0.01)
    density_score = min(1.0, density / density_ref)
    
    # 表面覆盖评分
    if n_pts > 0:
        areas = np.array([w*h, w*h, l*h, l*h, l*w, 0.0])  # 底面权重0
        weights = areas / max(areas.sum(), 0.01)
        expected = max(1, n_pts * 0.15)
        surf_scores = np.minimum(1.0, surf / expected)
        coverage = float(np.sum(surf_scores * weights))
    else:
        coverage = 0.0
    
    score = 0.5 * density_score + 0.5 * coverage
    
    return float(score), {
        'score': float(score),
        'n_points': int(n_pts),
        'density': float(density),
        'density_score': float(density_score),
        'coverage_score': float(coverage),
        'surface_counts': {
            'front': int(surf[0]), 'back': int(surf[1]),
            'left': int(surf[2]), 'right': int(surf[3]),
            'top': int(surf[4]), 'bottom': int(surf[5]),
        }
    }


def compute_visibility_batch(
    bboxes: np.ndarray,
    points: np.ndarray,
    density_ref: float = 50.0
) -> Tuple[np.ndarray, List[Dict]]:
    """批量计算"""
    bboxes = np.asarray(bboxes, dtype=np.float64)
    points = np.ascontiguousarray(points[:,:3], dtype=np.float64)
    
    n = bboxes.shape[0]
    scores = np.zeros(n)
    details = []
    
    for i in range(n):
        s, d = compute_visibility(bboxes[i], points, density_ref)
        scores[i] = s
        details.append(d)
    
    return scores, details


def classify(score: float) -> str:
    """可见性等级"""
    if score >= 0.6: return "VISIBLE"
    if score >= 0.3: return "PARTIAL" 
    if score > 0.05: return "OCCLUDED"
    return "INVISIBLE"


if __name__ == '__main__':
    import time
    
    np.random.seed(42)
    n_pts, n_box = 100000, 100
    
    pts = np.random.randn(n_pts, 3) * 30
    pts[:,2] = np.abs(pts[:,2]) * 0.5
    
    boxes = np.zeros((n_box, 7))
    boxes[:,:2] = np.random.uniform(-25, 25, (n_box, 2))
    boxes[:,2] = np.random.uniform(0, 2, n_box)
    boxes[:,3:6] = np.random.uniform(1, 5, (n_box, 3))
    boxes[:,6] = np.random.uniform(-np.pi, np.pi, n_box)
    
    # 预热
    compute_visibility(boxes[0], pts)
    
    # 测试
    t0 = time.time()
    scores, details = compute_visibility_batch(boxes, pts)
    t1 = time.time()
    
    print(f"点云: {n_pts:,}, 框: {n_box}")
    print(f"总耗时: {(t1-t0)*1000:.1f} ms")
    print(f"每框: {(t1-t0)/n_box*1000:.2f} ms")
    print(f"\n样例:")
    for i in range(5):
        print(f"  Box {i}: {scores[i]:.2f} ({classify(scores[i])}), pts={details[i]['n_points']}")
