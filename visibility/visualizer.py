#!/usr/bin/env python3
# -*-coding:utf-8 -*-
"""
可见性可视化模块
================
"""

import numpy as np
from typing import Dict, List, Tuple, Optional

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

from . import config


def _rotation_matrix_zyx(phi: float, theta: float, psi: float) -> np.ndarray:
    """计算旋转矩阵"""
    cz, sz = np.cos(phi), np.sin(phi)
    cy, sy = np.cos(theta), np.sin(theta)
    cx, sx = np.cos(psi), np.sin(psi)
    
    return np.array([
        [cz * cy, cz * sy * sx - sz * cx, cz * sy * cx + sz * sx],
        [sz * cy, sz * sy * sx + cz * cx, sz * sy * cx - cz * sx],
        [-sy, cy * sx, cy * cx]
    ], dtype=np.float64)


def project_point(
    point_3d: np.ndarray,
    extrinsic: np.ndarray,
    intrinsic: np.ndarray,
    width: int,
    height: int
) -> Tuple[Optional[Tuple[int, int]], float]:
    """将3D点投影到图像"""
    point_homo = np.append(point_3d, 1.0)
    point_cam = np.dot(np.linalg.inv(extrinsic), point_homo)[:3]
    
    depth = point_cam[2]
    if depth <= 0:
        return None, depth
    
    point_norm = point_cam[:2] / point_cam[2]
    u = intrinsic[0, 0] * point_norm[0] + intrinsic[0, 2]
    v = intrinsic[1, 1] * point_norm[1] + intrinsic[1, 2]
    
    if 0 <= u < width and 0 <= v < height:
        return (int(u), int(v)), depth
    return None, depth


def project_bbox_corners(
    bbox_dict: Dict,
    extrinsic: np.ndarray,
    intrinsic: np.ndarray,
    width: int,
    height: int
) -> List[Tuple[int, int]]:
    """投影bbox角点"""
    cx = float(bbox_dict['position']['x'])
    cy = float(bbox_dict['position']['y'])
    cz = float(bbox_dict['position']['z'])
    
    l = float(bbox_dict['size'][config.SIZE_ORDER['length']])
    w = float(bbox_dict['size'][config.SIZE_ORDER['width']])
    h = float(bbox_dict['size'][config.SIZE_ORDER['height']])
    
    phi = float(bbox_dict['orientation']['phi'])
    theta = float(bbox_dict['orientation']['theta'])
    psi = float(bbox_dict['orientation']['psi'])
    
    half_l, half_w, half_h = l/2, w/2, h/2
    corners_local = np.array([
        [-half_l, -half_w, -half_h],
        [-half_l, -half_w,  half_h],
        [-half_l,  half_w, -half_h],
        [-half_l,  half_w,  half_h],
        [ half_l, -half_w, -half_h],
        [ half_l, -half_w,  half_h],
        [ half_l,  half_w, -half_h],
        [ half_l,  half_w,  half_h],
    ])
    
    R = _rotation_matrix_zyx(phi, theta, psi)
    corners_world = (R @ corners_local.T).T + np.array([cx, cy, cz])
    
    corners_2d = []
    for corner in corners_world:
        pixel, _ = project_point(corner, extrinsic, intrinsic, width, height)
        if pixel is not None:
            corners_2d.append(pixel)
    
    return corners_2d


def draw_visibility(
    img: np.ndarray,
    label_3d_list: List[Dict],
    visibility_results: Dict[str, Dict],
    extrinsic: np.ndarray,
    intrinsic: np.ndarray,
    width: Optional[int] = None,
    height: Optional[int] = None,
    filter_camera: Optional[str] = None
) -> np.ndarray:
    """
    在图像上绘制可见性
    
    Args:
        img: 输入图像
        label_3d_list: 3D标注框列表
        visibility_results: 可见性结果
        extrinsic: 外参矩阵
        intrinsic: 内参矩阵
        width: 图像宽度
        height: 图像高度
        filter_camera: 相机过滤
    
    Returns:
        绑制后的图像
    """
    if not HAS_CV2:
        raise ImportError("需要cv2库")
    
    if width is None:
        width = img.shape[1]
    if height is None:
        height = img.shape[0]
    
    img = img.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    for bbox_dict in label_3d_list:
        track_id = str(bbox_dict.get('track_id', 'unknown'))
        
        # 过滤相机
        if filter_camera is not None:
            camera_ids = bbox_dict.get('camera_ids', [])
            if filter_camera not in camera_ids:
                continue
        
        # 获取可见性
        if track_id in visibility_results:
            info = visibility_results[track_id]
            score = info.get('score', 0)
            status = info.get('status', 'ERROR')
        else:
            score = 0
            status = 'ERROR'
        
        color = config.STATUS_COLORS.get(status, config.STATUS_COLORS['ERROR'])
        
        # 投影中心点
        center = np.array([
            float(bbox_dict['position']['x']),
            float(bbox_dict['position']['y']),
            float(bbox_dict['position']['z'])
        ])
        pixel, _ = project_point(center, extrinsic, intrinsic, width, height)
        
        if pixel is None:
            continue
        
        u, v = pixel
        
        # 绘制线框
        if config.VIS_DRAW_WIREFRAME:
            corners_2d = project_bbox_corners(bbox_dict, extrinsic, intrinsic, width, height)
            if len(corners_2d) >= 3:
                points = np.array(corners_2d, dtype=np.int32)
                hull = cv2.convexHull(points)
                cv2.polylines(img, [hull], True, color, config.VIS_THICKNESS)
        
        # 绘制标签
        if config.VIS_STYLE == 'full':
            text = f"ID:{track_id} {score:.0%} [{status}]"
        elif config.VIS_STYLE == 'score':
            text = f"{score:.0%}"
        elif config.VIS_STYLE == 'status':
            text = status
        else:
            text = f"{score:.0%}"
        
        # 文本背景
        (tw, th), _ = cv2.getTextSize(text, font, config.VIS_FONT_SCALE, config.VIS_THICKNESS)
        cv2.rectangle(img, (u-2, v-th-4), (u+tw+2, v+4), (0,0,0), -1)
        cv2.putText(img, text, (u, v), font, config.VIS_FONT_SCALE, color, config.VIS_THICKNESS)
        
        # 中心点标记
        cv2.circle(img, (u, v+8), 4, color, -1)
    
    # 图例
    if config.VIS_ADD_LEGEND:
        img = draw_legend(img)
    
    return img


def draw_legend(img: np.ndarray, position: str = 'top-left') -> np.ndarray:
    """绘制图例"""
    if not HAS_CV2:
        return img
    
    img = img.copy()
    
    items = [
        ('VISIBLE',  f'可见 (≥{config.VISIBILITY_THRESHOLDS["VISIBLE"]*100:.0f}%)'),
        ('PARTIAL',  f'部分 ({config.VISIBILITY_THRESHOLDS["PARTIAL"]*100:.0f}-{config.VISIBILITY_THRESHOLDS["VISIBLE"]*100:.0f}%)'),
        ('OCCLUDED', f'遮挡 ({config.VISIBILITY_THRESHOLDS["OCCLUDED"]*100:.0f}-{config.VISIBILITY_THRESHOLDS["PARTIAL"]*100:.0f}%)'),
        ('BLOCKED',  f'不可见 (<{config.VISIBILITY_THRESHOLDS["OCCLUDED"]*100:.0f}%)'),
    ]
    
    line_height = 25
    legend_h = len(items) * line_height + 20
    legend_w = 200
    margin = 10
    
    x, y = margin, margin
    
    # 背景
    overlay = img.copy()
    cv2.rectangle(overlay, (x, y), (x + legend_w, y + legend_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, img, 0.3, 0, img)
    cv2.rectangle(img, (x, y), (x + legend_w, y + legend_h), (255, 255, 255), 1)
    
    # 图例项
    for i, (status, label) in enumerate(items):
        color = config.STATUS_COLORS[status]
        item_y = y + 15 + i * line_height
        cv2.rectangle(img, (x + 10, item_y - 10), (x + 25, item_y + 5), color, -1)
        cv2.putText(img, label, (x + 35, item_y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    
    return img
